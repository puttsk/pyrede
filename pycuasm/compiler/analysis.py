from pprint import pprint
from pycuasm.compiler.hir import *

def collect_64bit_registers(program):
    reg64_dict = {}
    registers_dict = {}
    register_counter = 0
    
    reg64 = set()
    
    # Detecting 64-bit accesses
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        if inst.opcode.type == 'x32' and inst.opcode.integer_inst and inst.opcode.reg_store:
            if inst.dest.name == 'R14':
                pprint(inst)
            if inst.dest.carry_bit:
                opcode = inst.opcode
                inst_pos = program.ast.index(inst)
                
                for next_inst in [x for x in program.ast[inst_pos:] if isinstance(x, Instruction)]:
                    if next_inst.opcode.use_carry_bit:
                        reg64.add((inst.dest.name, next_inst.dest.name))
                        break
                
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
