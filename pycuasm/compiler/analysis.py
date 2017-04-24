import itertools
import copy
import math

from pprint import pprint
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *

def analyse_register_interference(program, register_list = None):
    if not register_list:
        register_list = sorted([ x for x in program.registers], key=lambda x: int(x.replace('R','')))

    interference_dict = dict.fromkeys(register_list, [])

    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        interference_regs = [reg.name for reg in inst.operands if isinstance(reg, Register) and reg.name in register_list]
        interference_ptrs = [reg.register.name for reg in inst.operands if isinstance(reg, Pointer) and reg.register.name in register_list]
        interference_regs = interference_regs + interference_ptrs
        if len(interference_regs) > 1:
            for reg in interference_regs:   
                interference_dict[reg] = interference_dict[reg] + interference_regs

    interference_dict = {k: sorted(set(v)) for k, v in interference_dict.items()}
    for reg in interference_dict:
        if reg in interference_dict[reg]:
            interference_dict[reg].remove(reg)

    return interference_dict

def analyse_register_accesses(program, register_list = None):
    if not register_list:
        register_list = sorted([ x for x in program.registers], key=lambda x: int(x.replace('R','')))

    access_dict = dict.fromkeys(register_list)
    for reg in access_dict:
        access_dict[reg] = {'read':0, 'write':0}
    
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:        
        for op in inst.operands:
            if isinstance(op, Pointer):
                if op.register.name in access_dict:
                    # Read access
                    access_dict[op.register.name]['read'] = access_dict[op.register.name]['read'] + 1
            elif isinstance(op, Register):
                if op.name in access_dict:
                    # Read access
                    access_dict[op.name]['read'] = access_dict[op.name]['read'] + 1
        if isinstance(inst.dest, Pointer):
            if inst.dest.register.name in access_dict:
                # Read access
                access_dict[inst.dest.register.name]['read'] = access_dict[inst.dest.register.name]['read'] + 1
        elif isinstance(inst.dest, Register):
            if inst.dest.name in access_dict:
                # Write access
                access_dict[inst.dest.name]['write'] = access_dict[inst.dest.name]['write'] + 1 
    
    return access_dict
    
def generate_spill_candidates(program, exclude_registers=[], priority='access'):
    print("[ANA_SPILL] Generating spilled register candidates. Excluding registers: %s" % exclude_registers)
    reg_64 = collect_non_32bit_registers(program)
    #reg_64 = collect_64bit_registers(program)

    reg_remove = list(reg_64) + exclude_registers
    reg_candidates = sorted(list(set([ x for x in program.registers if x not in reg_remove])), key=lambda x: int(x.replace('R','')))
    
    interference_dict = analyse_register_interference(program, reg_candidates)
    access_dict = analyse_register_accesses(program, reg_candidates)

    if property == 'access':
        reg_candidates = sorted(reg_candidates, key=lambda x: access_dict[x]['read'] +  access_dict[x]['write'])
    else:
        reg_candidates = sorted(reg_candidates, key=lambda x: len(interference_dict[x]))

    return reg_candidates
    
def generate_64bit_spill_candidates(program, exclude_registers=[]):
    print("[ANA_SPILL] Generating spilled 64-bit register candidates. Excluding registers: %s" % exclude_registers)
    list_64bit_registers = collect_64bit_registers(program)
    reg_mem = []    
    
    exclude_list = []
    
    list_64bit_registers = list(itertools.chain(list_64bit_registers.union(reg_mem)))
    for reg64 in list_64bit_registers:
        if int(reg64[0].replace('R','')) != (int(reg64[1].replace('R','')) - 1):
            exclude_list.append(reg64)
    
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        if inst.opcode.op_bit > 32 and (inst.opcode.type == 'gmem' or inst.opcode.type == 'smem'):
            # Handle multi-word load instruction e.g. LDG.E.128 R4 [R0], which load 4 32-bit words to R4, R5, R6, and R7
            # TODO: Might need to implement this for 32-bit spilling as well
            dest_list = []
            if "LD" in inst.opcode.name:
                start_reg_id = int(inst.dest.name.replace('R',''))
                for i in range (0, int(inst.opcode.op_bit/32), 2):
                    dest_list.append(('R%d' % (start_reg_id + i), 'R%d' % (start_reg_id + i + 1)))
            elif "ST" in inst.opcode.name:
                if isinstance(inst.operands[1], Register):
                    start_reg_id = int(inst.operands[1].name.replace('R',''))
                    for i in range (0, int(inst.opcode.op_bit/32), 2):
                        dest_list.append(('R%d' % (start_reg_id + i), 'R%d' % (start_reg_id + i + 1)))
            exclude_list += dest_list
    
    list_64bit_registers = [x for x in list_64bit_registers if x not in exclude_list]
    reg_candidates = sorted(list_64bit_registers, key=lambda x: int(x[0].replace('R','')))

    list_first_registers = [x[0] for x in list_64bit_registers]

    interference_dict = analyse_register_interference(program, list_first_registers)
    access_dict = analyse_register_accesses(program, list_first_registers)
    
    reg_candidates = sorted(reg_candidates, key=lambda x: access_dict[x[0]]['read'] +  access_dict[x[0]]['write'])

    return reg_candidates


def collect_64bit_registers(program):
    reg64_dict = {}
    registers_dict = {}
    register_counter = 0
    
    reg64 = set()
    
    # Detecting 64-bit accesses
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        if (inst.opcode.type == 'x32' and 
            inst.opcode.integer_inst and 
            inst.opcode.reg_store and 
            inst.dest):
            if inst.dest.carry_bit:
                opcode = inst.opcode
                inst_pos = program.ast.index(inst)
                
                for next_inst in [x for x in program.ast[inst_pos:] if isinstance(x, Instruction)]:
                    if next_inst.opcode.use_carry_bit:
                        #if abs(int(inst.dest.name.replace('R','')) - int(next_inst.dest.name.replace('R',''))) == 1: 
                        reg64.add((inst.dest.name, next_inst.dest.name))
                        break
    
        if (inst.opcode.is_64 or inst.opcode.type == 'x64'):
            for reg in [x for x in inst.operands if isinstance(x, Register)]:
                reg_id = int(reg.name.replace('R',''))
                reg64.add(("R%d" % reg_id, "R%d" % (reg_id+1)))
            if inst.reg_store and inst.dest:
                reg_id = int(inst.dest.name.replace('R',''))
                reg64.add(("R%d" % reg_id, "R%d" % (reg_id+1)))
             
    return reg64
    
def collect_global_memory_access(program):
    mem_reg = set()
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        if inst.opcode.type == 'gmem' or inst.opcode.type == 'x64':
            main_reg = None
            if inst.opcode.name.find('LD') != -1:
                # Load instruction   
                main_reg = inst.operands[0].register
            elif inst.opcode.name.find('ST') != -1:
                # Store instruction
                main_reg = inst.operands[0].register
                
            if main_reg:
                main_reg_id = int(main_reg.name.replace('R',''))
                couple_reg_id = main_reg_id + 1
                couple_reg = "R%d" % couple_reg_id
                    
                mem_reg.add((main_reg.name, couple_reg))
    return mem_reg


def __update_loop_reg_access_old(cfg, loop_begin, loop_end, update_factor = 2):
    traverse_order = Cfg.generate_breadth_first_order(loop_begin, loop_end)
    
    if getattr(loop_begin, 'visited_source', False):
        if loop_end in loop_begin.visited_source:
            return
        else:
            loop_begin.visited_source.append(loop_end)
    else:
        setattr(loop_begin, 'visited_source', [loop_end])
    
    for block in traverse_order:    
        if getattr(block, 'register_access', False):
            for k in block.register_access:
                block.register_access[k] *= update_factor

                            
def __get_function_reg_access_old(cfg, function_block):

    if getattr(function_block, 'register_access', False):
        return copy.copy(function_block.register_access)
        
    traverse_order = Cfg.generate_breadth_first_order(function_block)
    traverse_id = Cfg.get_traverse_id()
    visit_tag = 'visited_level_' + str(traverse_id)
        
    reg_access = {}
        
    for block in traverse_order:
        pred_level = max([0] + [getattr(x, visit_tag, 0) for x in block.pred])
        cur_level = pred_level + 1
        setattr(block, visit_tag, cur_level)
        
        block_reg_access = {}
        
        if not isinstance(block, BasicBlock):
            continue
        elif isinstance(block, CallBlock):
            block_reg_access = __get_function_reg_access_old(cfg, cfg.function_blocks[block.target_function])
        else:            
            block_reg_access = block.register_access
            
        for k in block_reg_access:
            if k not in reg_access.keys():
                reg_access[k] = block_reg_access[k]
            else:
                reg_access[k] += block_reg_access[k]

        if getattr(block.taken, visit_tag, cur_level) < cur_level:
            __update_loop_reg_access_old(cfg, block.taken, block)
    
        if getattr(block.not_taken, visit_tag, cur_level) < cur_level:
            __update_loop_reg_access_old(cfg, block.not_taken, block)
    
    for block in traverse_order:
        delattr(block, visit_tag)
    
    setattr(function_block, 'register_access', reg_access)
    return reg_access
    
def generate_spill_candidates_cfg_old(program, cfg, exclude_registers=[]):
    # Update all read/write accesses of each register in each BasicBlock
    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
            
        reg_read_map = dict.fromkeys(block.registers)
        for k in reg_read_map:
            reg_read_map[k] = 0           
        
        reg_write_map = dict.fromkeys(block.registers)
        for k in reg_write_map:
            reg_write_map[k] = 0
        
        reg_access_map = dict.fromkeys(block.registers)
        
        for inst in block.instructions:
            for operand in inst.operands:
                op = operand
                if isinstance(op, Pointer):
                    op = op.register
                if not isinstance(op, Register) or op.is_special:
                    continue
                
                reg_read_map[op] += 1
                
            if inst.opcode.reg_store and isinstance(inst.dest, Register):
                reg_write_map[inst.dest] += 1
        
        for k in reg_access_map:
            reg_access_map[k] = reg_read_map[k] + reg_write_map[k]
        
        setattr(block, 'register_reads', reg_read_map)
        setattr(block, 'register_writes', reg_write_map)
        setattr(block, 'register_access', reg_access_map)
        
    # Summarize register accesses in each function
    # Assume that there is no calling loop, e.g. A calls B and B calls A
    
    for function in cfg.function_blocks:
        __get_function_reg_access_old(cfg, cfg.function_blocks[function])
        
    traverse_order = Cfg.generate_breadth_first_order(cfg.blocks[0])
    traverse_id = Cfg.get_traverse_id()
    visit_tag = 'visited_level_' + str(traverse_id)
    
    for block in traverse_order:
        if isinstance(block, CallBlock):
            setattr(block, 'register_access', __get_function_reg_access(cfg, cfg.function_blocks[block.target_function]))
    
    for block in traverse_order:
        pred_level = max([0] + [getattr(x, visit_tag, 0) for x in block.pred])
        cur_level = pred_level + 1        
        setattr(block, visit_tag, cur_level)
    
        if getattr(block.taken, visit_tag, cur_level) < cur_level:
            __update_loop_reg_access_old(cfg, block.taken, block)
    
        if getattr(block.not_taken, visit_tag, cur_level) < cur_level:
            __update_loop_reg_access_old(cfg, block.not_taken, block)
    
    for block in traverse_order:
        delattr(block, visit_tag)
        
    reg_access = {}
        
    for block in traverse_order:
        if getattr(block, 'register_access', False):
            block_reg_access = block.register_access
                
            for k in block_reg_access:
                if k not in reg_access.keys():
                    reg_access[k] = block_reg_access[k]
                else:
                    reg_access[k] += block_reg_access[k]
    
    non_32_registers = collect_non_32bit_registers(program)
    reg_remove = list(non_32_registers) + exclude_registers
    reg_candidates = sorted(list(set([ x for x in program.registers if x not in reg_remove])), key=lambda x: int(x.replace('R','')))

    reg_candidates = sorted(reg_candidates, key=lambda x: reg_access[Register(x)])

    return reg_candidates

def __update_loop_reg_access(cfg, loop_begin, loop_end, reg_access, update_factor = 2):
    traverse_order = Cfg.generate_breadth_first_order(loop_begin, loop_end)    
    
    if getattr(loop_begin, 'visited_source', False):
        if loop_end in loop_begin.visited_source:
            return
        else:
            loop_begin.visited_source.append(loop_end)
    else:
        setattr(loop_begin, 'visited_source', [loop_end])
    
    for block in traverse_order:    
        if getattr(block, 'register_access', False):
            for k in block.register_access:                
                block.register_access[k] *= update_factor


                            
def __get_function_reg_access(cfg, function_block):

    if getattr(function_block, 'register_access', False):
        return copy.copy(function_block.register_access)
        
    traverse_order = Cfg.generate_breadth_first_order(function_block)
    traverse_id = Cfg.get_traverse_id()
    visit_tag = 'visited_level_' + str(traverse_id) 
    
    reg_access = {}    
    
    # Update block level. Set it level to the highest of predecessor level.     
    results = DFSResult()
    Cfg.update_block_level(function_block, results, visit_tag)
    
    for block in traverse_order:
        if block.taken and block.is_backward_taken: #getattr(block.taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_reg_access(cfg, block.taken, block, reg_access)
    
        if block.not_taken and block.is_backward_not_taken: #getattr(block.not_taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_reg_access(cfg, block.not_taken, block, reg_access)
        
    for block in traverse_order:        
        block_reg_access = {}
        
        if not isinstance(block, BasicBlock):
            continue
        elif isinstance(block, CallBlock):
            block_reg_access = __get_function_reg_access(cfg, cfg.function_blocks[block.target_function])
        else:            
            block_reg_access = block.register_access
            
        for k in block_reg_access:
            if k not in reg_access.keys():
                reg_access[k] = block_reg_access[k]
            else:
                reg_access[k] += block_reg_access[k]
        
    for block in traverse_order:
        delattr(block, visit_tag)
    
    setattr(function_block, 'register_access', reg_access)
    return reg_access
    
def generate_spill_candidates_cfg(program, cfg, exclude_registers=[]):
    # Update all read/write accesses of each register in each BasicBlock
    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
            
        reg_read_map = dict.fromkeys(block.registers)
        for k in reg_read_map:
            reg_read_map[k] = 0           
        
        reg_write_map = dict.fromkeys(block.registers)
        for k in reg_write_map:
            reg_write_map[k] = 0
        
        reg_access_map = dict.fromkeys(block.registers)
        
        for inst in block.instructions:
            for operand in inst.operands:
                op = operand
                if isinstance(op, Pointer):
                    op = op.register
                if not isinstance(op, Register) or op.is_special:
                    continue
                
                reg_read_map[op] += 1
                
            if inst.opcode.reg_store and isinstance(inst.dest, Register):
                reg_write_map[inst.dest] += 1
        
        for k in reg_access_map:
            reg_access_map[k] = reg_read_map[k] + reg_write_map[k]
        
        setattr(block, 'register_reads', reg_read_map)
        setattr(block, 'register_writes', reg_write_map)
        setattr(block, 'register_access', reg_access_map)
        
    # Summarize register accesses in each function
    # Assume that there is no calling loop, e.g. A calls B and B calls A
    
    for function in cfg.function_blocks:
        __get_function_reg_access(cfg, cfg.function_blocks[function])
        
    traverse_order = Cfg.generate_breadth_first_order(cfg.blocks[0])
    traverse_id = Cfg.get_traverse_id()
    visit_tag = 'visited_level_' + str(traverse_id)
    
    for block in traverse_order:
        if isinstance(block, CallBlock):
            setattr(block, 'register_access', __get_function_reg_access(cfg, cfg.function_blocks[block.target_function]))
    
    # Update block level. Set it level to the highest of predecessor level.     
    results = DFSResult()
    Cfg.update_block_level(cfg.blocks[0], results, visit_tag)

    reg_access = {}

    # Update the count with loop
    for block in traverse_order:    
        if block.taken and block.is_backward_taken: #getattr(block.taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_reg_access(cfg, block.taken, block, reg_access)
    
        if block.not_taken and block.is_backward_not_taken: #getattr(block.not_taken, visit_tag) < getattr(block, visit_tag):
            __update_loop_reg_access(cfg, block.not_taken, block, reg_access)
    
    # Count register access of each block 
    for block in traverse_order:
        if getattr(block, 'register_access', False):
            block_reg_access = block.register_access
                
            for k in block_reg_access:
                if k not in reg_access.keys():
                    reg_access[k] = block_reg_access[k]
                else:
                    reg_access[k] += block_reg_access[k]
    
            
    non_32_registers = collect_non_32bit_registers(program)
    reg_remove = list(non_32_registers) + exclude_registers
    reg_candidates = sorted(list(set([ x for x in program.registers if x not in reg_remove])), key=lambda x: int(x.replace('R','')))

    reg_candidates = sorted(reg_candidates, key=lambda x: reg_access[Register(x)])

    return reg_candidates


def collect_non_32bit_registers(program):
    reg_dict = {}
    registers_dict = {}
    register_counter = 0
    
    reg_set = set()
    
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        # Detecting 64-bit accesses
        if (inst.opcode.type == 'x32' and 
            inst.opcode.integer_inst and 
            inst.opcode.reg_store and 
            inst.dest):
            if inst.dest.carry_bit:
                opcode = inst.opcode
                inst_pos = program.ast.index(inst)
                
                for next_inst in [x for x in program.ast[inst_pos:] if isinstance(x, Instruction)]:
                    if next_inst.opcode.use_carry_bit:
                        if next_inst.dest and abs(int(inst.dest.name.replace('R','')) - int(next_inst.dest.name.replace('R',''))) == 1: 
                            reg_set.add(inst.dest.name)
                            reg_set.add(next_inst.dest.name)
                            break
        
        # Handle LDG.E and STG.E which use 64-bit address
        if (inst.opcode.name == 'LDG' or inst.opcode.name == 'STG') and 'E' in inst.opcode.extension:
            addr_ptr = inst.operands[0]
            reg_id = addr_ptr.register.id
            reg_set.add("R%d" % reg_id)
            reg_set.add("R%d" % (reg_id +1))
             
        if (inst.opcode.op_bit > 32):
            reg_list = [x for x in inst.operands if isinstance(x, Register)]
            if inst.dest and isinstance(inst.dest, Register):
                reg_list.append(inst.dest)
            
            for reg in reg_list:
                reg_id = reg.id
                reg_set.add("R%d" % reg_id)
                
                if inst.opcode.op_bit > 32:
                    reg_need = int(inst.opcode.op_bit / 32)
                    if reg_id % reg_need != 0:
                        reg_id = reg_id - (reg_id % reg_need)
                    for r in range(0, int(inst.opcode.op_bit/32)):
                        reg_set.add("R%d" % (reg_id+r))
       
    return reg_set