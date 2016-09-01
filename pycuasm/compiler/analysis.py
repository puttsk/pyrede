import itertools

from pprint import pprint
from pycuasm.compiler.hir import *

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
    
def generate_spill_candidates(program, exclude_registers=[]):
    print("[ANA_SPILL] Generating spilled register candidates. Excluding registers: %s" % exclude_registers)
    reg_64 = collect_64bit_registers(program)
    #reg_mem =  collect_global_memory_access(program)
    reg_mem = []
    #pprint(reg_64)
    
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        if inst.opcode.op_bit > 32 and (inst.opcode.type == 'gmem' or inst.opcode.type == 'smem'):
            # Handle multi-word load instruction e.g. LDG.E.128 R4 [R0], which load 4 32-bit words to R4, R5, R6, and R7
            # TODO: Might need to implement this for 32-bit spilling as well
            dest_list = []
            pprint(inst)
            if "LD" in inst.opcode.name:
                start_reg_id = int(inst.dest.name.replace('R',''))
                for i in range (0, int(inst.opcode.op_bit/32), 2):
                    dest_list.append(('R%d' % (start_reg_id + i), 'R%d' % (start_reg_id + i + 1)))
            elif "ST" in inst.opcode.name:
                start_reg_id = int(inst.operands[1].name.replace('R',''))
                for i in range (0, int(inst.opcode.op_bit/32), 2):
                    dest_list.append(('R%d' % (start_reg_id + i), 'R%d' % (start_reg_id + i + 1)))
            #pprint(dest_list)
            exclude_registers += dest_list
    
    reg_remove = list(itertools.chain(*reg_64.union(reg_mem))) + exclude_registers
    reg_candidates = sorted([ x for x in program.registers if x not in reg_remove], key=lambda x: int(x.replace('R','')))

    interference_dict = analyse_register_interference(program, reg_candidates)
    access_dict = analyse_register_accesses(program, reg_candidates)
    
    #pprint(interference_dict)
    #pprint(access_dict)

    reg_candidates = sorted(reg_candidates, key=lambda x: access_dict[x]['read'] +  access_dict[x]['write'])
    #reg_candidates = sorted(reg_candidates, key=lambda x: len(interference_dict[x]))

    return reg_candidates
    
def generate_64bit_spill_candidates(program, exclude_registers=[]):
    print("[ANA_SPILL] Generating spilled 64-bit register candidates. Excluding registers: %s" % exclude_registers)
    list_64bit_registers = collect_64bit_registers(program)
    #reg_mem =  collect_global_memory_access(program)
    reg_mem = []
    
    #pprint(list_64bit_registers)
    
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
            pprint(inst)
            if "LD" in inst.opcode.name:
                start_reg_id = int(inst.dest.name.replace('R',''))
                for i in range (0, int(inst.opcode.op_bit/32), 2):
                    dest_list.append(('R%d' % (start_reg_id + i), 'R%d' % (start_reg_id + i + 1)))
            elif "ST" in inst.opcode.name:
                start_reg_id = int(inst.operands[1].name.replace('R',''))
                for i in range (0, int(inst.opcode.op_bit/32), 2):
                    dest_list.append(('R%d' % (start_reg_id + i), 'R%d' % (start_reg_id + i + 1)))
            #pprint(dest_list)
            exclude_list += dest_list
    
    list_64bit_registers = [x for x in list_64bit_registers if x not in exclude_list]
    reg_candidates = sorted(list_64bit_registers, key=lambda x: int(x[0].replace('R','')))

    pprint(reg_candidates)

    list_first_registers = [x[0] for x in list_64bit_registers]

    interference_dict = analyse_register_interference(program, list_first_registers)
    access_dict = analyse_register_accesses(program, list_first_registers)
    
    #pprint(interference_dict)
    #pprint(access_dict)

    reg_candidates = sorted(reg_candidates, key=lambda x: access_dict[x[0]]['read'] +  access_dict[x[0]]['write'])
    #reg_candidates = sorted(reg_candidates, key=lambda x: len(interference_dict[x]))

    #pprint(reg_candidates)
    #reg_candidates = [('R20', 'R21'), ('R30', 'R31')]

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
            if inst.reg_store:
                reg_id = int(inst.dest.name.replace('R',''))
                reg64.add(("R%d" % reg_id, "R%d" % (reg_id+1)))
    
    return reg64
    
def collect_global_memory_access(program):
    mem_reg = set()
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        if inst.opcode.type == 'gmem' or inst.opcode.type == 'x64':
            main_reg = None
            pprint(inst)
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
