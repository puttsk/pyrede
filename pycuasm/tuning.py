import math

from pprint import pprint
from collections import namedtuple

from pycuasm.compiler.lexer import sass_lexer
from pycuasm.compiler.parser import sass_parser
from pycuasm.compiler.sass import Sass
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *

THREADS_PER_WARP = 32
WARPS_PER_SM = 64
WARP_ALLOC_UNIT = 2
THREAD_BLOCK_PER_SM = 32
REGISTERS_PER_SM = 65536
REGISTERS_PER_BLOCK = 32768
REGISTER_ALLOC_UNIT = 256
SHARED_MEM_PER_SM = 98304
SHARED_MEM_PER_BLOCK = 49152

class InstructionStatistic(object):
    def __init__(self):
        self.count = 0
        self.stall = 0
        self.visited = 0
    def __repr__(self):
        return str(self.__dict__) 

def __update_loop_statistic(cfg, loop_begin, loop_end, inst_stat, update_factor = 2):
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
                        
    setattr(function_block, 'inst_stat', inst_stat)
    return inst_stat

def tune_occupancy(program, register_size, shared_size, threadblock_size):
    config = []
    
    # Compute thread block limit
    warps_per_block = int(math.ceil(threadblock_size / THREADS_PER_WARP))
    warps_per_block = int(math.ceil(warps_per_block / WARP_ALLOC_UNIT)) * WARP_ALLOC_UNIT 
    limit_block_alloc_per_sm = int(math.floor(WARPS_PER_SM / warps_per_block)) 
    
    # Compute register limit
    registers_per_warp = register_size * THREADS_PER_WARP
    warp_register_alloc = int(math.ceil(registers_per_warp / REGISTER_ALLOC_UNIT)) * REGISTER_ALLOC_UNIT # Size of register allocation is based on allocation granularity
    registers_per_block = warps_per_block * warp_register_alloc
    limit_reg_alloc_per_sm = int(math.floor(REGISTERS_PER_SM / registers_per_block)) 
    
    # Compute shared memory limit
    limit_shared_alloc_per_sm = int(math.floor(SHARED_MEM_PER_SM / shared_size)) if shared_size > 0 else THREAD_BLOCK_PER_SM
    
    max_thread_block_count = min([limit_block_alloc_per_sm, limit_reg_alloc_per_sm, limit_shared_alloc_per_sm]) 
    max_occupancy = max_thread_block_count * warps_per_block / WARPS_PER_SM

    print("=== Program Statistic ===")
    print("Static Instruction Count: ", len([x for x in program.ast if isinstance(x, Instruction)]))
    print("Register Usage: ", register_size)
    print("Shared Memory Usage: ", shared_size)
    print("Threadblock Size: ", threadblock_size)
    
    print("Block limited by threadblock: ", limit_block_alloc_per_sm)
    print("Block limited by register: ", limit_reg_alloc_per_sm)
    print("Block limited by shared memory: ", limit_shared_alloc_per_sm)
    print("Maximum occupancy: ", max_occupancy)
    
    # If register allocation is limiting factor
    if limit_reg_alloc_per_sm < min([limit_block_alloc_per_sm, limit_shared_alloc_per_sm]):
        tunable = True
        target_block_per_sm = limit_reg_alloc_per_sm
        while tunable:
            # Find possible configuration
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
            shared_avail = min([int(math.floor(SHARED_MEM_PER_SM / target_block_per_sm)) - shared_size, SHARED_MEM_PER_BLOCK])  
            if shared_avail < shared_required:
                tunable = False
            else:
                config.append(register_to_demote)
            
            if tunable:
                print("=== New Config ===")
                print("Target register usage: ", target_reg_usage)
                print("Number of registers for demotion: ", register_to_demote)
                print("Shared memory requirement: ", shared_required)
                print("Availabel shared memory: ", shared_avail)
                print("Maximum Occupancy: ", target_block_per_sm * warps_per_block / WARPS_PER_SM)
        
    return config

def tuning(args):
    sass = Sass(args.input_file)
    program = sass_parser.parse(sass.sass_raw, lexer=sass_lexer)
    program.set_constants(sass.constants)
    program.set_header(sass.header)
    program.update()

    cfg = Cfg(program)
    
    config =  tune_occupancy(program, len(program.registers), program.shared_size, args.thread_block_size)
    
    # Update instruction statistic of each cfg block
    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
        
        inst_stat = {}
        stall_count = 0
        
        for inst in [x for x in block.instructions if isinstance(x, Instruction)]:
            inst_type = inst.opcode.type if inst.opcode.type != 'x32' else inst.opcode.name
            if not inst_type in inst_stat:
                inst_stat[inst_type] = InstructionStatistic()
            inst_stat[inst_type].count += 1
            inst_stat[inst_type].stall += inst.flags.stall
            inst_stat[inst_type].visited += 1
            
            stall_count += inst.flags.stall
        
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
                                
    pprint(inst_stat)
    
    #pprint(reg_count)    
    
