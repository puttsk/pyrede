import copy

from pprint import pprint
from pycuasm.compiler.hir import *

# Register R0 and R1 are reserved for register spilling to shared memory
SPILL_REGISTER_ADDR = Register('R0')
SPILL_REGISTER = Register('R1')

def rename_register(program, old_reg, new_reg):
    print("Renaming %s to %s" % (old_reg, new_reg))
    for inst in program.ast:
        if not isinstance(inst, Instruction):
            continue
        
        for op in inst.operands:
            if isinstance(op, Pointer):
                if old_reg == op.register:
                    op.register.rename(new_reg)
                    
            elif isinstance(op, Register):
                if old_reg == op:
                    inst.operands[inst.operands.index(op)].rename(new_reg)
        
        if isinstance(inst.dest, Pointer):
            if old_reg == inst.dest.register:
                inst.dest.register.rename(new_reg)
        elif isinstance(inst.dest, Register):
            if old_reg == inst.dest:
                inst.dest.rename(new_reg)    
    program.update()

def spill_register_to_shared(program, spilled_register, cfg, thread_block_size=256):
    # TODO: Find free barrier for LDS Instruction
    print("Replacing %s with shared memory" % spilled_register)
    
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
        # Compute base address for spilled register
        base_addr_inst = Instruction(Flags(hex(wait_flag),'-','-','-','d'), 
            Opcode('SHL'), 
            operands=[SPILL_REGISTER_ADDR, tid_reg, 0x02])
        program.ast.insert(program.ast.index(tid_inst) + 1, base_addr_inst)
    
    # Assign the spilled register Identifier
    spill_reg_id = program.shared_spill_count
    spill_offset = spill_reg_id * thread_block_size
    program.shared_spill_count += 1
        
    load_shr_inst = Instruction(Flags('--','1','2','-','d'), 
                        Opcode('LDS'),
                        operands=[SPILL_REGISTER, Pointer(SPILL_REGISTER_ADDR, spill_offset)])
    
    store_shr_inst = Instruction(Flags('--','-','-','-','d'), 
                        Opcode('STS'),
                        operands=[Pointer(SPILL_REGISTER_ADDR, spill_offset), SPILL_REGISTER])
    
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
                    # Load data from shared memory
                    op.register.rename(SPILL_REGISTER)
                    # Set the instruction that read still register to wait for shared memory read 
                    inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3 
                    program.ast.insert(program.ast.index(inst), load_shr_inst)
                    
            elif isinstance(op, Register):
                if spilled_register == op:
                    # Read access
                    inst.operands[inst.operands.index(op)].rename(SPILL_REGISTER)
                    # Set the instruction that read spill register to wait for shared memory read
                    inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3
                    program.ast.insert(program.ast.index(inst), load_shr_inst)
        
        if isinstance(inst.dest, Pointer):
            if spilled_register == inst.dest.register:
                # Read access
                inst.dest.register.rename(SPILL_REGISTER)
                # Set the instruction that read spill register to wait for shared memory read
                inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3
                program.ast.insert(program.ast.index(inst), load_shr_inst)
        elif isinstance(inst.dest, Register):
            if spilled_register == inst.dest:
                # Write access
                inst.dest.rename(SPILL_REGISTER)
                st_inst = copy.deepcopy(store_shr_inst)
                # Set wait flag if the previous instruction sets write dependence flag. 
                # This will happen if the instruction store data in the spilled register.
                # Add 1 additional cycle to the store instruction. 
                if inst.flags.write_barrier != 0:
                    st_inst.flags.wait_barrier = st_inst.flags.wait_barrier | inst.flags.write_barrier
                    inst.flags.stall = inst.flags.stall + 1
                program.ast.insert(program.ast.index(inst) + 1, st_inst)
            
    program.update()
        