from pprint import pprint
from pycuasm.compiler.hir import *
from pycuasm.compiler.analysis import *

REL_OFFSETS = ['BRA', 'SSY', 'CAL', 'PBK', 'PCNT']
ABS_OFFSETS = ['JCAL']
JUMP_OPS = REL_OFFSETS + ABS_OFFSETS

def remove_redundant_spill_instruction(program, spill_addr_register):
    # Removing redundant spill load/store. The current implementation only work with 32-bit spill
    to_remove_list = []
    prev_spill_inst = None        
    for inst in program.ast:
        # Need to be conservative
        if isinstance(inst, Label):
            prev_spill_inst = None
            continue
            
        # Need to be conservative        
        if isinstance(inst, Instruction) and inst.opcode.name in JUMP_OPS:
            prev_spill_inst = None
            continue
                
        if isinstance(inst, SpillStoreInstruction) or isinstance(inst, SpillLoadInstruction):
            if prev_spill_inst != None:
                if isinstance(inst, SpillStoreInstruction) and isinstance(prev_spill_inst, SpillStoreInstruction):
                    cur_dest = inst.operands[0]
                    prev_dest = prev_spill_inst.operands[0]
                    # Both instruction update the value of the same spilled Register
                    # This should be really rare to happen
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
                    # The load instruction load the most recent value. No need to load
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
                    # The load instruction load the most recent value. No need to load
                    if cur_dest.offset == prev_dest.offset:
                        to_remove_list.append(inst)
                        inst_idx = program.ast.index(inst)
                        next_inst = program.ast[inst_idx+1]
                        if isinstance(next_inst, Instruction):
                            wait_flag = 1 << (inst.flags.read_barrier-1) | 1 << (inst.flags.write_barrier-1)
                            next_inst.flags.wait_barrier = next_inst.flags.wait_barrier & ~wait_flag                   
                #elif isinstance(inst, SpillStoreInstruction) and isinstance(prev_spill_inst, SpillLoadInstruction):

            prev_spill_inst = inst
  
    for inst in to_remove_list:
        print("Remove: %s" % repr(inst))        
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

            
            
        