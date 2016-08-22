import copy
import itertools

from pprint import pprint
from pycuasm.compiler.hir import *
from pycuasm.compiler.analysis import *

def relocate_registers(program):
    program_regs = sorted(program.registers, key=lambda x: int(x.replace('R','')))
    #pprint(program_regs)
    
    #reg_mem =  collect_global_memory_access(program)
    reg_mem = []
    reg_64 = collect_64bit_registers(program)    
    reg_64 = list(itertools.chain(*reg_64.union(reg_mem)))
        
    idx = 0;
    end = False
    reg_skip = []
    while not end:
        reg_cur = program_regs[idx]
        reg_cur_id = int(reg_cur.replace('R',''))
        reg_next = program_regs[idx+1]
        reg_next_id = int(reg_next.replace('R',''))
        
        if reg_cur_id + 1 == reg_next_id:
            idx = idx + 1
        elif reg_next not in reg_64:
            if len(reg_skip) != 0:
                reg_next_new = reg_skip.pop()
                reg_next_new_id = int(reg_next_new.replace('R',''))
                rename_register(program, Register(reg_next), Register(reg_next_new))
                program_regs[reg_next_new_id] = reg_next_new
                program_regs[idx+1] = reg_cur
                idx = idx + 1
            else:
                reg_next_new_id = reg_cur_id + 1
                reg_next_new = 'R%d' % reg_next_new_id        
                rename_register(program, Register(reg_next), Register(reg_next_new))
                program_regs[idx+1] = reg_next_new
                idx = idx + 1
        else:
            # reg_next is a 64-bit register
            reg_next_new_id = reg_cur_id + 1
            reg_next_new = 'R%d' % reg_next_new_id
            if reg_next_new_id % 2 == 0:
                # The available register is an even register. 
                # The 64-bit register can be move without any problem
                # Move both 64-bit registers together
                rename_register(program, Register(reg_next), Register(reg_next_new))
                rename_register(program, Register(program_regs[idx+2]), Register('R%d' % (reg_next_new_id+1)))
                program_regs[idx+1] = reg_next_new
                program_regs[idx+2] = 'R%d' % (reg_next_new_id+1)
                idx = idx + 2
            else:
                # The available register is an odd register.
                # The 64-bit register cannot be start with an odd register. 
                # Stores the odd register in reg_skip list and move to the next register
                reg_skip.append(reg_next_new)
                if reg_next_new_id != reg_next_id:
                    rename_register(program, Register(reg_next), Register('R%d' % (reg_next_new_id+1)))
                    rename_register(program, Register(program_regs[idx+2]), Register('R%d' % (reg_next_new_id+2)))
                    program_regs[idx+1] = 'R%d' % (reg_next_new_id+1)
                    program_regs[idx+2] = 'R%d' % (reg_next_new_id+2)
                    idx = idx + 2
                else:
                    idx = idx + 2

        if idx == len(program_regs)-1:
            end = True
    
def rename_registers(program, registers_dict):
    """ Renaming registers in a program using rules in registers_dict 
        
        Args:
            program (Program): Target program for register renaming
            registers_dict (dict{str:str}): A dictionary containing a renaming rules. The key represent the old register name and a value contain new register name 
    """
    #print("Renaming using dict:" )
    #pprint(registers_dict)
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
    
def spill_register_to_shared(
        program, 
        target_register, 
        cfg=None,
        spill_register=Register('R0'), 
        spill_register_addr=Register('R1'), 
        thread_block_size=256, ):
    """ Spill registers to shared memory 
        
        Args:
            program (Program): target program for register spilling
            spilled_register (Register): A register to be spilled  
            cfg (Cfg): A CFG of the input program
            thread_block_size (int): Size of thread block  
    """
    # TODO: Find free barrier for LDS Instruction
    print("Replacing %s with shared memory (TB:%d)" % (target_register, thread_block_size))
    
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
            operands=[spill_register_addr, tid_reg, 0x02])
        program.ast.insert(program.ast.index(tid_inst) + 1, base_addr_inst)
    
    # Assign the spilled register Identifier
    spill_reg_id = program.shared_spill_count
    spill_offset = spill_reg_id * thread_block_size * 4
    program.shared_spill_count += 1
        
    load_shr_inst = SpillLoadInstruction(Flags('--','1','2','-','d'), 
                        Opcode('LDS'),
                        operands=[spill_register, Pointer(spill_register_addr, spill_offset),
                            #Register(spilled_register.name + 'S')
                        ])
    
    # Add read barrier to make sure that the STS instruction is completed before next instreuction
    store_shr_inst = SpillStoreInstruction(Flags('--','4','-','-','d'), 
                        Opcode('STS'),
                        operands=[Pointer(spill_register_addr, spill_offset), spill_register, 
                            #Register(spilled_register.name + 'S')
                        ])
    
    w_count = 0
    r_count = 0
    
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        # If instruction contain spilled_register, rename the spilled_register to spill_register.
        # If the instruction read data from the spilled_register, load data in the shared memory 
        # to spill_register just before the instruction. 
        # If the instruction write data to the spilled_register, store data from spill_register 
        # to shared memory right after the instruction.
        for op in inst.operands:
            if isinstance(op, Pointer):
                if target_register == op.register:
                    r_count = r_count + 1
                    # Load data from shared memory
                    op.register.rename(spill_register)
                    # Set the instruction that read still register to wait for shared memory read 
                    inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3 
                    program.ast.insert(program.ast.index(inst), copy.deepcopy(load_shr_inst))
                    
            elif isinstance(op, Register):
                if target_register == op:
                    r_count = r_count + 1
                    # Read access
                    inst.operands[inst.operands.index(op)].rename(spill_register)
                    # Set the instruction that read spill register to wait for shared memory read
                    inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3
                    program.ast.insert(program.ast.index(inst), copy.deepcopy(load_shr_inst))
        
        if isinstance(inst.dest, Pointer):
            if target_register == inst.dest.register:
                # Read access
                r_count = r_count + 1
                inst.dest.register.rename(spill_register)
                # Set the instruction that read spill register to wait for shared memory read
                inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3
                program.ast.insert(program.ast.index(inst), copy.deepcopy(load_shr_inst))
        elif isinstance(inst.dest, Register):
            if target_register == inst.dest:
                # Write access
                w_count = w_count + 1
                st_inst = copy.deepcopy(store_shr_inst)
                inst.dest.rename(spill_register)
                
                # Set wait flag if the previous instruction sets write dependence flag. 
                # This will happen if the instruction store data in the spilled register.
                # Add 1 additional cycle to the store instruction. 
                if inst.flags.write_barrier != 0:
                    st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                    inst.flags.stall = inst.flags.stall + 1    
                else:
                    if(inst.opcode.grammar.type == 'gmem' or inst.opcode.grammar.type == 'smem'):
                        inst.flags.write_barrier = 6
                        st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                    inst.flags.stall = 13 
                program.ast.insert(program.ast.index(inst) + 1, st_inst)
                # Set wait flag of the next instruction to wait for store instruction to finish
                inst_next = program.ast[program.ast.index(st_inst) + 1] 
                if isinstance(inst_next, Instruction):
                    inst_next.flags.wait_barrier = inst_next.flags.wait_barrier | 8 
                  
    print("Read accesses: %d Write accesses: %d" % (r_count, w_count))   
    program.update()