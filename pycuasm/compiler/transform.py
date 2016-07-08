import copy

from pprint import pprint
from pycuasm.compiler.hir import *
from pycuasm.compiler.analysis import collect_64bit_registers


# Register R0 and R1 are reserved for register spilling to shared memory
SPILL_REGISTER_ADDR = Register('R0')
SPILL_REGISTER = Register('R1')

#TODO: Bug
def relocate_registers(program):
    reg64_dict = collect_64bit_registers(program)
    registers_dict = {}
    register_counter = 0

    # Reassign registers        
    for inst in program.ast:
        if not isinstance(inst, Instruction):
            continue
        for op in inst.operands:
            reg = None
            if isinstance(op, Pointer):
                reg = op.register
            elif isinstance(op, Register) and not op.is_special:
                reg = op
            else:
                continue
                
            if reg.name not in registers_dict:
                # Check if reg is 64-bit reg
                if reg.name in reg64_dict:
                    # 64-bit reg
                    reg_id = int(reg.name.replace('R',''))
                    couple_reg_name = reg64_dict[reg.name]
                    couple_reg_id = int(couple_reg_name.replace('R',''))
                    
                    if couple_reg_id > reg_id:
                        pprint((reg.name, "R%d" % register_counter))
                        registers_dict[reg.name] = "R%d" % register_counter
                        register_counter = register_counter + 1
                        pprint((couple_reg_name, "R%d" % register_counter))
                        registers_dict[couple_reg_name] = "R%d" % register_counter
                        register_counter = register_counter + 1
                    else:
                        pprint((couple_reg_name, "R%d" % register_counter))
                        registers_dict[couple_reg_name] = "R%d" % register_counter
                        register_counter = register_counter + 1
                        pprint((reg.name, "R%d" % register_counter))
                        registers_dict[reg.name] = "R%d" % register_counter
                        register_counter = register_counter + 1     
                else:
                    pprint((reg.name, "R%d" % register_counter))
                    registers_dict[reg.name] = "R%d" % register_counter
                    register_counter = register_counter + 1
                
        if isinstance(inst.dest, Pointer):
            reg = inst.dest.register
        elif isinstance(inst.dest, Register):
            reg = inst.dest
        else:
            continue
            
        if reg.name not in registers_dict:
                # Check if reg is 64-bit reg
                if reg.name in reg64_dict:
                    # 64-bit reg
                    reg_id = int(reg.name.replace('R',''))
                    couple_reg_name = reg64_dict[reg.name]
                    couple_reg_id = int(couple_reg_name.replace('R',''))
                    
                    if couple_reg_id > reg_id:
                        pprint((reg.name, "R%d" % register_counter))
                        registers_dict[reg.name] = "R%d" % register_counter
                        register_counter = register_counter + 1
                        pprint((couple_reg_name, "R%d" % register_counter))
                        registers_dict[couple_reg_name] = "R%d" % register_counter
                        register_counter = register_counter + 1
                    else:
                        pprint((couple_reg_name, "R%d" % register_counter))
                        registers_dict[couple_reg_name] = "R%d" % register_counter
                        register_counter = register_counter + 1
                        pprint((reg.name, "R%d" % register_counter))
                        registers_dict[reg.name] = "R%d" % register_counter
                        register_counter = register_counter + 1     
                else:
                    pprint((reg.name, "R%d" % register_counter))
                    registers_dict[reg.name] = "R%d" % register_counter
                    register_counter = register_counter + 1
    
    rename_registers(program, registers_dict)
    
def rename_registers(program, registers_dict):
    """ Renaming registers in a program using rules in registers_dict 
        
        Args:
            program (Program): Target program for register renaming
            registers_dict (dict{str:str}): A dictionary containing a renaming rules. The key represent the old register name and a value contain new register name 
    """
    print("Renaming using dict:" )
    pprint(registers_dict)
    pprint(len(registers_dict))
    for inst in program.ast:
        if not isinstance(inst, Instruction):
            continue
        
        for op in inst.operands:
            if isinstance(op, Pointer):
                if op.register.name in registers_dict:
                    op.register.rename(Register(registers_dict[op.register.name]))
            elif isinstance(op, Register):
                if op.name in registers_dict:
                    inst.operands[inst.operands.index(op)].rename(Register(registers_dict[op.name]))
                
        if isinstance(inst.dest, Pointer):
            if inst.dest.register.name in registers_dict:
                inst.dest.register.rename(Register(registers_dict[inst.dest.register.name]))
        elif isinstance(inst.dest, Register):
            if inst.dest.name in registers_dict:
                inst.dest.rename(Register(registers_dict[inst.dest.name]))    
    program.update()

def rename_register(program, old_reg, new_reg):
    """ Replacing a register old_reg with a register new_reg in a program 
        
        Args:
            program (Program): target program for register renaming
            old_reg (Register): An original name of the register 
            new_reg (Register): A new name for the register 
    """
    print("Renaming %s to %s" % (old_reg, new_reg))
    rename_registers(program, {old_reg.name:new_reg.name})
    

def spill_register_to_shared(program, spilled_register, cfg, thread_block_size=256):
    """ Spill registers to shared memory 
        
        Args:
            program (Program): target program for register spilling
            spilled_register (Register): A register to be spilled  
            cfg (Cfg): A CFG of the input program
            thread_block_size (int): Size of thread block  
    """
    # TODO: Find free barrier for LDS Instruction
    print("Replacing %s with shared memory (TB:%d)" % (spilled_register, thread_block_size))
    
    tid_reg = None
    tid_inst = 0    
    # If this is the first time a register is spilled to shared memory,
    # add a new parameter to the program object for keeping track of register 
    # spilling and add instruction to compute the base address for spilled 
    # register
    # ASSUME: 1D thread block
    # TODO: Add support for 2d and 3d thread block
    if not getattr(program, 'shared_spill_count', False):
        # Add new parameter
        setattr(program, 'shared_spill_count', 0)
        
        # Find register containing thread id
        for inst in program.ast:
            if not isinstance(inst, Instruction):
                continue
            if inst.opcode.name == 'S2R':
                if inst.operands[0].name == 'SR_TID':
                    tid_reg = inst.dest
                    tid_inst = inst
                    break
        
        # Compute wait flag
        wait_flag = 1 << (tid_inst.flags.write_barrier-1)
        tid_inst.flags.stall = tid_inst.flags.stall + 1
        # Compute base address for spilled register
        base_addr_inst = Instruction(Flags(hex(wait_flag),'-','-','-','d'), 
            Opcode('SHL'), 
            operands=[SPILL_REGISTER_ADDR, tid_reg, 0x02])
        program.ast.insert(program.ast.index(tid_inst) + 1, base_addr_inst)
    
    # Assign the spilled register Identifier
    spill_reg_id = program.shared_spill_count
    spill_offset = spill_reg_id * thread_block_size * 4
    program.shared_spill_count += 1
        
    load_shr_inst = Instruction(Flags('--','1','2','-','d'), 
                        Opcode('LDS'),
                        operands=[SPILL_REGISTER, Pointer(SPILL_REGISTER_ADDR, spill_offset)])
    
    store_shr_inst = Instruction(Flags('--','-','-','-','d'), 
                        Opcode('STS'),
                        operands=[Pointer(SPILL_REGISTER_ADDR, spill_offset), SPILL_REGISTER])
    
    w_count = 0
    r_count = 0
    
    for inst in program.ast:
        if not isinstance(inst, Instruction):
            continue
        
        # If instruction contain spilled_register, rename the spilled_register to SPILL_REGISTER.
        # If the instruction read data from the spilled_register, load data in the shared memory 
        # to SPILL_REGISTER just before the instruction. 
        # If the instruction write data to the spilled_register, store data from SPILL_REGISTER 
        # to shared memory right after the instruction.
        for op in inst.operands:
            if isinstance(op, Pointer):
                if spilled_register == op.register:
                    r_count = r_count + 1
                    # Load data from shared memory
                    op.register.rename(SPILL_REGISTER)
                    # Set the instruction that read still register to wait for shared memory read 
                    inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3 
                    program.ast.insert(program.ast.index(inst), copy.deepcopy(load_shr_inst))
                    
            elif isinstance(op, Register):
                if spilled_register == op:
                    r_count = r_count + 1
                    # Read access
                    inst.operands[inst.operands.index(op)].rename(SPILL_REGISTER)
                    # Set the instruction that read spill register to wait for shared memory read
                    inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3
                    program.ast.insert(program.ast.index(inst), copy.deepcopy(load_shr_inst))
        
        if isinstance(inst.dest, Pointer):
            if spilled_register == inst.dest.register:
                # Read access
                w_count = w_count + 1
                inst.dest.register.rename(SPILL_REGISTER)
                # Set the instruction that read spill register to wait for shared memory read
                inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3
                program.ast.insert(program.ast.index(inst), copy.deepcopy(load_shr_inst))
        elif isinstance(inst.dest, Register):
            if spilled_register == inst.dest:
                # Write access
                w_count = w_count + 1
                inst.dest.rename(SPILL_REGISTER)
                st_inst = copy.deepcopy(store_shr_inst)
                # Set wait flag if the previous instruction sets write dependence flag. 
                # This will happen if the instruction store data in the spilled register.
                # Add 1 additional cycle to the store instruction. 
                if inst.flags.write_barrier != 0:
                    st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                    inst.flags.stall = inst.flags.stall + 1    
                else:
                    if(inst.opcode.grammar.type == 'gmem' or inst.opcode.grammar.type == 'smem'):
                        pprint(inst)
                        inst.flags.write_barrier = 6
                        st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                    inst.flags.stall = 13 
                program.ast.insert(program.ast.index(inst) + 1, st_inst)
                  
    print("Read accesses: %d Write accesses: %d" % (r_count, w_count))   
    program.update()
        