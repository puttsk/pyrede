from pprint import pprint
from pycuasm.compiler.hir import *

def collect_64bit_registers(program):
    reg64_dict = {}
    registers_dict = {}
    register_counter = 0
    
    # Detecting 64-bit accesses
    for inst in program.ast:
        if not isinstance(inst, Instruction):
            continue
        
        if inst.opcode.type == 'gmem':
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
                 
                reg64_dict[main_reg.name] = couple_reg
                reg64_dict[couple_reg] = main_reg.name 
    
    return reg64_dict