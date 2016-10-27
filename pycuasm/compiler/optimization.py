from pprint import pprint
from pycuasm.compiler.hir import *
from pycuasm.compiler.analysis import *

REL_OFFSETS = ['BRA', 'SYNC', 'CAL', 'PBK', 'PCNT']
ABS_OFFSETS = ['JCAL']
JUMP_OPS = REL_OFFSETS + ABS_OFFSETS

SHARED_MEMORY_STALL = 40

def rearrange_spill_instruction(program, spill_register, spill_addr_register):
    # Rearrange spilling Instruction
    for i in range(len(program.ast)):
        inst = program.ast[i]
        if isinstance(inst, SpillStoreInstruction):
            # Remove wait flags if there is no immediate reuse.
            inst_idx = i
            stall_count = 0
            # Shared memory write takes about SHARED_MEMORY_STALL cycles
            while stall_count < SHARED_MEMORY_STALL and inst_idx < len(program.ast):
                next_inst = program.ast[inst_idx + 1]
                if isinstance(next_inst, Label):
                    # End of the current basic block. Wait for the memory store opreation to finish at this point. 
                    last_inst = program.ast[inst_idx]
                    if not isinstance(last_inst, SpillStoreInstruction):
                        if stall_count > 6:
                            inst.flags.stall = 1
                        wait_flag = 1 << (inst.flags.read_barrier-1) 
                        program.ast[i+1].flags.wait_barrier = program.ast[i+1].flags.wait_barrier & ~wait_flag
                        last_inst.flags.wait_barrier = last_inst.flags.wait_barrier | wait_flag
                        print("[SPILL_OPT] Move store barrier from %s to label %s" % (inst, next_inst))    
                    break
                if next_inst.opcode.name in JUMP_OPS:
                    # End of the current basic block. Wait for the memory store opreation to finish at this point. 
                    if stall_count > 6:
                        inst.flags.stall = 1
                    wait_flag = 1 << (inst.flags.read_barrier-1) 
                    program.ast[i+1].flags.wait_barrier = program.ast[i+1].flags.wait_barrier & ~wait_flag
                    next_inst.flags.wait_barrier = next_inst.flags.wait_barrier | wait_flag
                    print("[SPILL_OPT] Move store barrier from %s to %s" % (inst, next_inst))
                    break
                    
                if next_inst.dest == spill_register:
                    # The next_inst instruction will rewrite the value of spill register. 
                    # Wait for the memory store opreation to finish before updating the register.                     
                    if stall_count > 6:
                        inst.flags.stall = 1
                    wait_flag = 1 << (inst.flags.read_barrier-1) 
                    program.ast[i+1].flags.wait_barrier = program.ast[i+1].flags.wait_barrier & ~wait_flag
                    next_inst.flags.wait_barrier = next_inst.flags.wait_barrier | wait_flag
                    print("[SPILL_OPT] Move store barrier from %s to %s" % (inst, next_inst))
                    break
                    
                stall_count += next_inst.flags.stall
                inst_idx += 1
            
            if stall_count >= SHARED_MEMORY_STALL:
                # There is no load/store to shared memory after SHARED_MEMORY_STALL stall cycles
                inst.flags.stall = 1             
                wait_flag = 1 << (inst.flags.read_barrier-1)
                program.ast[i+1].flags.wait_barrier = program.ast[i+1].flags.wait_barrier & ~wait_flag
                print("[SPILL_OPT] Remove stall from instruction: %s" % repr(inst))            
                
        elif isinstance(inst, SpillLoadInstruction):
            inst_idx = i            
            stall_count = 0
            
            while stall_count < SHARED_MEMORY_STALL and inst_idx >= 0:
                next_inst = program.ast[inst_idx - 1]
                if isinstance(next_inst, Label):
                    # To guarantee correctness, wait for the memory store opreation to finish at this point. 
                    inst.flags.stall = 1
                    if stall_count > 2:
                        inst.flags.yield_hint = True
                    program.ast.remove(inst)
                    program.ast.insert(inst_idx, inst) 
                    print("[SPILL_OPT] Move load instruction %s after label: %s" % (inst, next_inst))    
                    break
                
                if next_inst.opcode.name in JUMP_OPS:
                    # To guarantee correctness, wait for the memory store opreation to finish at this point. 
                    inst.flags.stall = 1
                    program.ast.remove(inst)
                    program.ast.insert(inst_idx, inst) 
                    print("[SPILL_OPT] Move load instruction %s to jump instruction: %s" % (inst, next_inst))    
                    break
                    
                if spill_register in next_inst.operands:
                    # Prevent loading new value to spill register while it is read.
                    if inst_idx != program.ast.index(inst): 
                        wait_flag = 0;
                        if next_inst.flags.read_barrier > 0: 
                            wait_flag = 1 << (next_inst.flags.read_barrier-1)
                        inst.flags.stall = 1
                        inst.flags.wait_barrier = inst.flags.wait_barrier | wait_flag 
                        program.ast.remove(inst)
                        program.ast.insert(inst_idx, inst)
                        
                        print("[SPILL_OPT] Move load instrution %s to instruction: %s" % (inst, next_inst))    
                    break
                
                stall_count += next_inst.flags.stall
                inst_idx -= 1
                
            if stall_count >= SHARED_MEMORY_STALL:
                # There is no load/store to shared memory after SHARED_MEMORY_STALL stall cycles
                inst.flags.stall = 1
                program.ast.remove(inst)
                program.ast.insert(inst_idx+1, inst)             
                print("[SPILL_OPT] Move %s up %d cycles" % (repr(inst), SHARED_MEMORY_STALL))
        
    program.update()
        
def remove_redundant_spill_instruction(program, spill_addr_register):
    # Removing redundant spill load/store. The current implementation only work with 32-bit spill
    to_remove_list = []
    prev_spill_inst = None        
    for inst in program.ast:
        # At the end of basic block
        if isinstance(inst, Label):
            prev_spill_inst = None
            continue
            
        # At the end of basic block        
        if isinstance(inst, Instruction) and inst.opcode.name in JUMP_OPS:
            prev_spill_inst = None
            continue
                
        if isinstance(inst, SpillStoreInstruction) or isinstance(inst, SpillLoadInstruction):
            if prev_spill_inst != None:
                if isinstance(inst, SpillStoreInstruction) and isinstance(prev_spill_inst, SpillStoreInstruction):
                    cur_dest = inst.operands[0]
                    prev_dest = prev_spill_inst.operands[0]
                    # Both instruction update the value of the same spilled Register
                    # This should be rare
                    if cur_dest.offset == prev_dest.offset:
                        to_remove_list.append(prev_spill_inst)
                        inst_idx = program.ast.index(prev_spill_inst)
                        next_inst = program.ast[inst_idx-1]
                        if isinstance(next_inst, Instruction):
                            wait_flag = 1 << (prev_spill_inst.flags.read_barrier-1)
                            next_inst.flags.wait_barrier = next_inst.flags.wait_barrier & ~wait_flag
                elif isinstance(inst, SpillLoadInstruction) and isinstance(prev_spill_inst, SpillStoreInstruction):
                    cur_dest = inst.operands[0]
                    prev_dest = prev_spill_inst.operands[0]
                    # The current instruction loads the most recent value. No need to load
                    if cur_dest.offset == prev_dest.offset:
                        to_remove_list.append(inst)
                        inst_idx = program.ast.index(inst)
                        next_inst = program.ast[inst_idx+1]
                        if isinstance(next_inst, Instruction):
                            wait_flag = 1 << (inst.flags.read_barrier-1) | 1 << (inst.flags.write_barrier-1)
                            next_inst.flags.wait_barrier = next_inst.flags.wait_barrier & ~wait_flag
                elif isinstance(inst, SpillLoadInstruction) and isinstance(prev_spill_inst, SpillLoadInstruction):
                    cur_dest = inst.operands[0]
                    prev_dest = prev_spill_inst.operands[0]
                    # The current instruction loads the most recent value. No need to load
                    if cur_dest.offset == prev_dest.offset:
                        to_remove_list.append(inst)
                        inst_idx = program.ast.index(inst)
                        next_inst = program.ast[inst_idx+1]
                        if isinstance(next_inst, Instruction):
                            wait_flag = 1 << (inst.flags.read_barrier-1) | 1 << (inst.flags.write_barrier-1)
                            next_inst.flags.wait_barrier = next_inst.flags.wait_barrier & ~wait_flag                   

            prev_spill_inst = inst
  
    for inst in to_remove_list:
        print("[SPILL_OPT] Remove redundant load/store: %s" % repr(inst))        
        program.ast.remove(inst)        
            
# Skip instructions to avoid spill register calculation
#passed_addr_comp = False
#if not passed_addr_comp:
#    if isinstance(inst, Instruction):
#        # Check if the instruction is spill register address computation
#        if inst.opcode.name == 'SHL' and inst.dest == spill_addr_register:
#            pprint(inst)
#           passed_addr_comp = True
#    continue

            
            
        