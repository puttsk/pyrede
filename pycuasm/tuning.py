from pprint import pprint
from collections import namedtuple

from pycuasm.compiler.lexer import sass_lexer
from pycuasm.compiler.parser import sass_parser
from pycuasm.compiler.sass import Sass
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *

class InstructionStatistic(object):
    def __init__(self):
        self.count = 0
        self.stall = 0
        self.visited = 0
    def __repr__(self):
        return str(self.__dict__) 

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
                if k not in inst_stat.keys():
                    inst_stat[k] = InstructionStatistic()
                    inst_stat[k].stall = block.inst_stat[k].stall
                    inst_stat[k].count = block.inst_stat[k].count
                    inst_stat[k].visited = block.inst_stat[k].visited
                else:
                    inst_stat[k].stall += block.inst_stat[k].stall * update_factor
                    inst_stat[k].count += block.inst_stat[k].count *update_factor
                    inst_stat[k].visited += 1

def __get_function_statistic(cfg, function_block):
    if getattr(function_block, 'inst_stat', False):
        return copy.copy(function_block.inst_stat)
        
    traverse_order = Cfg.generate_breadth_first_order(function_block)
    traverse_id = Cfg.get_traverse_id()
    visit_tag = 'visited_level_' + str(traverse_id)
    visited_tag = 'visited_' + str(traverse_id)
    
    inst_stat = {}
    stall_count = 0
    
    # Update block level. Set it level to the highest of predecessor level.     
    results = DFSResult()
    Cfg.update_block_level(function_block, results, visit_tag)
        
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
        
        if block.taken and getattr(block.taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_statistic(cfg, block.taken, block, inst_stat)
    
        if block.not_taken and getattr(block.not_taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_statistic(cfg, block.not_taken, block, inst_stat)
        
        # Self loop
        if block.taken and block.taken == block:
            __update_loop_statistic(cfg, block.taken, block, inst_stat)
    
        if block.not_taken and block.not_taken == block:
            __update_loop_statistic(cfg, block.taken, block, inst_stat)
            
    for block in traverse_order:
        delattr(block, visit_tag)
    
    setattr(function_block, 'inst_stat', inst_stat)
    return inst_stat


def tuning(args):
    sass = Sass(args.input_file)
    program = sass_parser.parse(sass.sass_raw, lexer=sass_lexer)
    program.set_constants(sass.constants)
    program.set_header(sass.header)
    program.update()

    cfg = Cfg(program)
    
    
    inst_count = {}
    stall_count = 0
    
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        inst_type = inst.opcode.type if inst.opcode.type != 'x32' and inst.opcode.type != 'smem'  and inst.opcode.type != 'lmem'  else inst.opcode.name    
        if not inst_type in inst_count:
            inst_count[inst_type] = InstructionStatistic()
        inst_count[inst_type].count += 1
        inst_count[inst_type].stall += inst.flags.stall
        
        stall_count += inst.flags.stall
    
    # Update instruction statistic of each cfg block
    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
        
        inst_stat = {}
        stall_count = 0
        
        for inst in [x for x in block.instructions if isinstance(x, Instruction)]:
            inst_type = inst.opcode.type if inst.opcode.type != 'x32' and inst.opcode.type != 'smem'  and inst.opcode.type != 'lmem' else inst.opcode.name
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
    visited_tag = 'visited_' + str(traverse_id)
    
    for block in traverse_order:
        if isinstance(block, CallBlock):
            setattr(block, 'inst_stat', __get_function_statistic(cfg, cfg.function_blocks[block.target_function]))
    
    has_update = True
    # Update block level. Set it level to the highest of predecessor level.     
    results = DFSResult()
    Cfg.update_block_level(cfg.blocks[0], results, visit_tag)

    inst_stat = {}
    cfg.create_dot_graph('cfg.dot')

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

    # Update the statistic if there is a loop
    for block in traverse_order:    
        if block.taken and getattr(block.taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_statistic(cfg, block.taken, block, inst_stat)
    
        if block.not_taken and getattr(block.not_taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_statistic(cfg, block.not_taken, block, inst_stat)
            
        # Self loop
        if block.taken and block.taken == block:
            print('self')
            __update_loop_statistic(cfg, block.taken, block, inst_stat)
    
        if block.not_taken and block.not_taken == block:
            print('self')
            __update_loop_statistic(cfg, block.taken, block, inst_stat)

    pprint(inst_stat)
    
    print("=== Program Statistic ===")
    print("Instruction Count: ", len([x for x in program.ast if isinstance(x, Instruction)]))
    
    #pprint(reg_count)    
    
