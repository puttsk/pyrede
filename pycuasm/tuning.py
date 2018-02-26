import math
import os
import contextlib

from pprint import pprint
from collections import namedtuple

from pycuasm.compiler.lexer import sass_lexer
from pycuasm.compiler.parser import sass_parser
from pycuasm.compiler.sass import Sass
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *

from pycuasm.compile import *

THREADS_PER_WARP = 32
WARPS_PER_SM = 64
THREAD_BLOCK_PER_SM = 32
REGISTERS_PER_SM = 65536
REGISTERS_PER_BLOCK = 32768
SHARED_MEM_PER_SM = 98304
SHARED_MEM_PER_BLOCK = 49152

WARP_ALLOC_UNIT = 2
REGISTER_ALLOC_UNIT = 256

GLOBAL_ACCESS_STALL = 200
SHARED_ACCESS_STALL = 24

DOUBLE_PRECISION_UNIT_RATIO_FACTOR = 32
SPECIAL_FUNCTION_UNIT_RATIO_FACTOR = 4

ProgramStatistic = namedtuple('ProgramStatistic', ['stall', 'inst_stat'])

class InstructionStatistic(object):
    def __init__(self):
        self.count = 0
        self.stall = 0
        self.visited = 0
    def __repr__(self):
        return str(self.__dict__) 

def compute_occupancy(register_size, shared_size, threadblock_size):
    
    # Compute thread block limit
    warps_per_block = int(math.ceil(threadblock_size / THREADS_PER_WARP))
    warps_per_block = int(math.ceil(warps_per_block / WARP_ALLOC_UNIT)) * WARP_ALLOC_UNIT 
    limit_block_alloc_per_sm = int(math.floor(math.floor(WARPS_PER_SM / warps_per_block) / WARP_ALLOC_UNIT) * WARP_ALLOC_UNIT) 
    
    # Compute register limit
    registers_per_warp = register_size * THREADS_PER_WARP
    warp_register_alloc = int(math.ceil(registers_per_warp / REGISTER_ALLOC_UNIT)) * REGISTER_ALLOC_UNIT # Size of register allocation is based on allocation granularity
    registers_per_block = warps_per_block * warp_register_alloc
    limit_reg_alloc_per_sm = int(math.floor(math.floor(REGISTERS_PER_SM / registers_per_block) / WARP_ALLOC_UNIT) * WARP_ALLOC_UNIT) 
    
    # Compute shared memory limit
    limit_shared_alloc_per_sm = int(math.floor(SHARED_MEM_PER_SM / shared_size)) if shared_size > 0 else THREAD_BLOCK_PER_SM
    
    max_thread_block_count = min([limit_block_alloc_per_sm, limit_reg_alloc_per_sm, limit_shared_alloc_per_sm]) 
    max_occupancy = max_thread_block_count * warps_per_block / WARPS_PER_SM

    #print("Block limited by threadblock: ", limit_block_alloc_per_sm)
    #print("Block limited by register: ", limit_reg_alloc_per_sm)
    #print("Block limited by shared memory: ", limit_shared_alloc_per_sm)

    limiters = []
    max_block = 0
    
    if limit_reg_alloc_per_sm <= min([limit_block_alloc_per_sm, limit_shared_alloc_per_sm]):
        limiters.append('register')
        max_block = limit_reg_alloc_per_sm
    if limit_block_alloc_per_sm <= min([limit_reg_alloc_per_sm, limit_shared_alloc_per_sm]):
        limiters.append('block')
        max_block = limit_block_alloc_per_sm
    if limit_shared_alloc_per_sm <= min([limit_reg_alloc_per_sm, limit_block_alloc_per_sm]):
        limiters.append('shared')
        max_block = limit_shared_alloc_per_sm
        
    return max_occupancy, max_block, limiters

def tune_occupancy(program, register_size, shared_size, threadblock_size):
    config = []
    
    print("=== Program Statistic ===")
    print("Static Instruction Count: ", len([x for x in program.ast if isinstance(x, Instruction)]))
    print("Register Usage: ", register_size)
    print("Shared Memory Usage: ", shared_size)
    print("Threadblock Size: ", threadblock_size)
    
    max_occupancy, max_block, limiters = compute_occupancy(register_size, shared_size, threadblock_size)
    
    print("Maximum occupancy: ", max_occupancy)
    print()
    # If register allocation is limiting factor
    if 'register' in limiters:
        tunable = True
        target_block_per_sm = max_block
        while tunable:
            # Find possible configuration
            warps_per_block = int(math.ceil(threadblock_size / THREADS_PER_WARP))
            warps_per_block = int(math.ceil(warps_per_block / WARP_ALLOC_UNIT)) * WARP_ALLOC_UNIT
            # Try to increase number of block per SM by 2
            target_block_per_sm = (target_block_per_sm + 2)
            # Per thread allocation
            target_reg_usage = int(math.floor(REGISTERS_PER_SM / target_block_per_sm))
            # Per warp allocation
            target_reg_usage = int(math.floor(target_reg_usage / warps_per_block / REGISTER_ALLOC_UNIT) * REGISTER_ALLOC_UNIT)
            # Per thread allocation
            target_reg_usage = int(target_reg_usage / THREADS_PER_WARP)
            register_to_demote = register_size - target_reg_usage + 2
            shared_required = register_to_demote * threadblock_size * 4
            
            max_occupancy, max_block, limiters = compute_occupancy(target_reg_usage, shared_required + shared_size, threadblock_size)
            
            if 'register' in limiters: 
                shared_avail = min([int(math.floor(SHARED_MEM_PER_SM / target_block_per_sm)) - shared_size, SHARED_MEM_PER_BLOCK])  
                if shared_avail < shared_required:
                    tunable = False
                else:
                    config.append(register_to_demote)
            else:
                tunable = False                
            
            if tunable:
                print("=== New Config ===")
                print("Target register usage: ", target_reg_usage)
                print("Number of registers for demotion: ", register_to_demote)
                print("Shared memory requirement: ", shared_required)
                print("Availabel shared memory: ", shared_avail)
                print("Maximum Occupancy: ", max_occupancy)
                print()
        
    return config

def __update_loop_statistic(cfg, loop_begin, loop_end, inst_stat, update_factor = 10):
    traverse_order = Cfg.generate_breadth_first_order(loop_begin, loop_end)
    
    if getattr(loop_begin, 'visited_source', False):
        if loop_end in loop_begin.visited_source:
            return
        else:
            loop_begin.visited_source.append(loop_end)
    else:
        setattr(loop_begin, 'visited_source', [loop_end])
    
    for block in traverse_order:    
        if getattr(block, 'inst_stat', False):
            for k in block.inst_stat:
                block.inst_stat[k].stall *= update_factor
                block.inst_stat[k].count *= update_factor
                block.inst_stat[k].visited += 1
            block.stall_count *= update_factor

def __get_function_statistic(cfg, function_block):
    if getattr(function_block, 'inst_stat', False):
        return copy.copy(function_block.inst_stat)
        
    traverse_order = Cfg.generate_breadth_first_order(function_block)
    traverse_id = Cfg.get_traverse_id()
    visit_tag = 'visited_level_' + str(traverse_id)
    
    inst_stat = {}
    stall_count = 0
    
    # Update block level. Set it level to the highest of predecessor level.     
    results = DFSResult()
    Cfg.update_block_level(function_block, results, visit_tag)
    
    # Update loop
    for block in traverse_order:
        if block.taken and block.is_backward_taken: #getattr(block.taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_statistic(cfg, block.taken, block, inst_stat)
    
        if block.not_taken and block.is_backward_not_taken: #getattr(block.not_taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_statistic(cfg, block.not_taken, block, inst_stat)
        
    for block in traverse_order:
        block_inst_stat = {}
        block_stall_count = 0
        
        if not isinstance(block, BasicBlock):
            continue
        elif isinstance(block, CallBlock):
            block_inst_stat = __get_function_statistic(cfg, cfg.function_blocks[block.target_function])
        else:            
            block_inst_stat = block.inst_stat
            
        for k in block_inst_stat:
            if k not in inst_stat:
                inst_stat[k] = InstructionStatistic()
                inst_stat[k].count = block_inst_stat[k].count
                inst_stat[k].stall = block_inst_stat[k].stall
                inst_stat[k].visited = block_inst_stat[k].visited
            else:
                inst_stat[k].count += block_inst_stat[k].count
                inst_stat[k].stall += block_inst_stat[k].stall
                inst_stat[k].visited += block_inst_stat[k].visited
        
        stall_count += block.stall_count
        
    setattr(function_block, 'inst_stat', inst_stat)
    setattr(function_block, 'stall_count', stall_count)
    return inst_stat

# Naive implementation
def collect_program_statistic_naive(program):
    cfg = Cfg(program)    
    
    inst_stat = {}
    stall_count = 0
    # Update instruction statistic of each cfg block
    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
        
        barrier = [None] * 7
        
        for inst in [x for x in block.instructions if isinstance(x, Instruction)]:
            stall_count = stall_count + inst.flags.stall

    return ProgramStatistic(stall_count, inst_stat)

def collect_program_statistic(program):
    cfg = Cfg(program)    
    # Update instruction statistic of each cfg block
    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
                
        inst_stat = {}
        stall_count = 0
        
        barrier = [None] * 7
        
        for inst in [x for x in block.instructions if isinstance(x, Instruction)]:
            inst_type = inst.opcode.type if inst.opcode.type != 'x32' else inst.opcode.name
            if not inst_type in inst_stat:
                inst_stat[inst_type] = InstructionStatistic()
            inst_stat[inst_type].count += 1
            
            if inst.opcode.type == 'x64':
                inst_stat[inst_type].stall += inst.flags.stall * DOUBLE_PRECISION_UNIT_RATIO_FACTOR * program.occupancy
                stall_count += inst.flags.stall * DOUBLE_PRECISION_UNIT_RATIO_FACTOR * program.occupancy
            elif inst.opcode.type == 'qtr':
                inst_stat[inst_type].stall += inst.flags.stall * SPECIAL_FUNCTION_UNIT_RATIO_FACTOR * program.occupancy
                stall_count += inst.flags.stall * SPECIAL_FUNCTION_UNIT_RATIO_FACTOR * program.occupancy                        
            else:
                inst_stat[inst_type].stall += inst.flags.stall
                stall_count += inst.flags.stall
                
            inst_stat[inst_type].visited += 1
                
            if inst.flags.read_barrier > 0:
                #barrier[inst.flags.read_barrier] = [inst, inst.flags.stall]
                barrier[inst.flags.read_barrier] = [inst, 0]
            
            if inst.flags.write_barrier > 0:
                #barrier[inst.flags.write_barrier] = [inst, inst.flags.stall]
                barrier[inst.flags.write_barrier] = [inst, 0] 
            
            for wait in inst.flags.wait_barrier_list:
                if barrier[wait]:
                    if barrier[wait][0].opcode.type == 'lmem' or barrier[wait][0].opcode.type == 'gmem':
                        if barrier[wait][1] < GLOBAL_ACCESS_STALL:
                            stall_count += GLOBAL_ACCESS_STALL - barrier[wait][1]
                            inst_stat[barrier[wait][0].opcode.type].stall += GLOBAL_ACCESS_STALL - barrier[wait][1]                      
                    elif barrier[wait][0].opcode.type == 'smem':
                        if barrier[wait][1] < SHARED_ACCESS_STALL:
                            stall_count += SHARED_ACCESS_STALL - barrier[wait][1]
                            inst_stat[barrier[wait][0].opcode.type].stall += SHARED_ACCESS_STALL - barrier[wait][1]

                    barrier[wait] = None
                    
            for i in range(7):
                if barrier[i]:
                    barrier[i][1] += inst.flags.stall
                    
        setattr(block, 'inst_stat', inst_stat)
        setattr(block, 'stall_count', stall_count)
        
    # Summarize instruction statistic in each function
    # Assume that there is no calling loop, e.g. A calls B and B calls A
    for function in cfg.function_blocks:
        __get_function_statistic(cfg, cfg.function_blocks[function])

    traverse_order = Cfg.generate_breadth_first_order(cfg.blocks[0])
    traverse_id = Cfg.get_traverse_id()
    visit_tag = 'visited_level_' + str(traverse_id)    

    for block in traverse_order:
        if isinstance(block, CallBlock):
            setattr(block, 'inst_stat', __get_function_statistic(cfg, cfg.function_blocks[block.target_function]))
    
    has_update = True
    # Update block level. Set it level to the highest of predecessor level.     
    results = DFSResult()
    Cfg.update_block_level(cfg.blocks[0], results, visit_tag)

    inst_stat = {}
    stall_count = 0
    cfg.create_dot_graph('cfg.dot')

    # Update the statistic of each loop 
    for block in traverse_order:    
        if block.taken and block.is_backward_taken: #getattr(block.taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_statistic(cfg, block.taken, block, inst_stat)
    
        if block.not_taken and block.is_backward_not_taken: #getattr(block.not_taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_statistic(cfg, block.not_taken, block, inst_stat)

    # Collect instruction statistic of each block 
    for block in traverse_order:
        if getattr(block, 'inst_stat', False):
            block_inst_stat = block.inst_stat
                
            for k in block_inst_stat:
                if k not in inst_stat.keys():
                    inst_stat[k] = InstructionStatistic()
                    inst_stat[k].stall = block_inst_stat[k].stall
                    inst_stat[k].count = block_inst_stat[k].count
                    inst_stat[k].visited = block_inst_stat[k].visited
                else:
                    inst_stat[k].stall += block_inst_stat[k].stall
                    inst_stat[k].count += block_inst_stat[k].count
                    inst_stat[k].visited += block_inst_stat[k].visited
            
            stall_count += block.stall_count
    return ProgramStatistic(stall_count, inst_stat)

def adjust(occupancy, a, b):
    return a * math.pow(occupancy, b)
    #return 0.20795 + 1.7717 * math.exp(-occupancy * 7.8366)

def tuning(args):
    sass = Sass(args.input_file)
    program = sass_parser.parse(sass.sass_raw, lexer=sass_lexer)
    program.set_constants(sass.constants)
    program.set_header(sass.header)
    program.update()
    
    config =  tune_occupancy(program, len(program.registers), program.shared_size, args.thread_block_size)
    
    occupancy, max_block, limiters = compute_occupancy(len(program.registers), program.shared_size, args.thread_block_size)
    program_stat = {}
    program_occupancy = {}
    program.occupancy = occupancy
    program_stat['orig'] = collect_program_statistic(program)
    program_occupancy['orig'] = occupancy
    
    for conf in config:
        print("Collect program statistic with %d register spill" % conf)
        args.spill_register = int(conf)
        for candidate in range(3):
            for flags in itertools.product(['0', '1'], repeat = 3):
                conf_program = copy.deepcopy(program)
                args.candidate_type = candidate
                args.avoid_conflict = int(flags[0])
                args.swap_spill_reg = int(flags[1])
                args.opt_access = int(flags[2])
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        compile_program(conf_program, args)     
                candidate_str = ''
                if candidate == 0:
                    candidate_str = 'cfg'
                elif candidate == 1:
                    candidate_str = 'access'
                else:
                    candidate_str = 'conflict'
                occupancy, max_block, limiters = compute_occupancy(len(conf_program.registers), program.shared_size, args.thread_block_size)
                conf_program.occupancy = occupancy
                program_stat['spill_' + str(conf) + '_' + candidate_str +'_'+ '_'.join(flags)] = collect_program_statistic(conf_program)
                program_occupancy['spill_' + str(conf) + '_' + candidate_str +'_'+ '_'.join(flags)] = occupancy
    if args.local_sass:
        sass_local = Sass(args.local_sass)
        program = sass_parser.parse(sass_local.sass_raw, lexer=sass_lexer)
        program.set_constants(sass_local.constants)
        program.set_header(sass_local.header)
        program.update()
        
        occupancy, max_block, limiters = compute_occupancy(len(program.registers), program.shared_size, args.thread_block_size)
        program.occupancy = occupancy
        program_stat['local'] = collect_program_statistic(program)
        program_occupancy['local'] = occupancy
    if args.local_sass_shared:
        sass_local = Sass(args.local_sass_shared)
        program = sass_parser.parse(sass_local.sass_raw, lexer=sass_lexer)
        program.set_constants(sass_local.constants)
        program.set_header(sass_local.header)
        program.update()
        
        occupancy, max_block, limiters = compute_occupancy(len(program.registers), program.shared_size, args.thread_block_size)
        program.occupancy = occupancy
        program_stat['local_shared'] = collect_program_statistic(program)
        program_occupancy['local_shared'] = occupancy
        
    print("configuration,", "    stall,", "occupancy,", "adjusted_stall,", "inst_count,", "dp_count")
    max_occupancy = max(program_occupancy.values())
    for conf in sorted(program_stat.keys()):
        inst_count = sum([program_stat[conf].inst_stat[x].count for x in program_stat[conf].inst_stat])
        dp_count = program_stat[conf].inst_stat['x64'].count if 'x64' in program_stat[conf].inst_stat else 0
        adjust_factor = adjust(program_occupancy[conf], 0.14088, -0.86281) / adjust(max_occupancy, 0.14088, -0.86281)
        #adjusted_stall = program_stat[conf].stall * 1.0/program_occupancy[conf] 
        adjusted_stall = program_stat[conf].stall * adjust_factor
        # For naive approace
        # adjusted_stall = program_stat[conf].stall 
        print ("%13s,%10d,%10.3f,%15.2f,%11d,%9d, %f" % (conf, program_stat[conf].stall, program_occupancy[conf], adjusted_stall, inst_count, dp_count, adjust_factor) )
        
    
