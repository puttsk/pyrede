from pprint import pprint
from pycuasm.compiler.hir import *
from pycuasm.compiler.analysis import *
from pycuasm.compiler.utils import *
from pycuasm.compiler.transform import *

REL_OFFSETS = ['BRA', 'SYNC', 'CAL', 'PBK', 'PCNT']
ABS_OFFSETS = ['JCAL']
JUMP_OPS = REL_OFFSETS + ABS_OFFSETS

SHARED_MEMORY_STALL = 40

def rearrange_spill_instruction(program, spill_register, spill_addr_register):
    print("[REG_OPT]: rearrange_spill_instruction ")
    
    cfg = Cfg(program)
    cfg.analyze_liveness()
    
    regs_non_32 = set(collect_non_32bit_registers(program))
    regs_32 = set(program.registers) - regs_non_32

    # Finding free registers
    for function in cfg.function_blocks:
        function_block = cfg.function_blocks[function]
        traverse_order = Cfg.generate_breadth_first_order(function_block)
        
        for block in traverse_order:
            live_reg = block.live_in | block.live_out | regs_non_32 | block.var_def
            free_reg = regs_32 - live_reg
            setattr(block, "free_reg", free_reg)    
        
    traverse_order = Cfg.generate_breadth_first_order(cfg.blocks[0])

    for block in traverse_order:
        live_reg = block.live_in | block.live_out | regs_non_32 | block.var_def
        free_reg = regs_32 - live_reg
        setattr(block, "free_reg", free_reg)
    
    block_list = []
    
    for block in traverse_order:
        if not isinstance(block, BasicBlock):
            continue
            
        if any(isinstance(x, SpillInstruction) for x in block.instructions):
            block_list.append(block)
            
    for block in block_list:
    #for block in range(2):
        #block = block_list[block]
        pprint(block.line)
        pprint(block.free_reg)
        pprint(block.instructions)
        spill_insts = [x for x in block.instructions if isinstance(x, SpillInstruction)]
        
        available_reg = copy.copy(block.free_reg)
        available_reg = sorted(available_reg, key=lambda x: int(x.replace('R','')))
        if 'R1' in available_reg:
            available_reg.remove('R1')
        
        pprint(available_reg)

        barrier_tracker = BarrierTracker()
        barrier_tracker.reset()
        
        for inst in block.instructions:
            if isinstance(inst, SpillLoadInstruction):
                inst_idx = block.instructions.index(inst)   
                paired_inst = block.instructions[block.instructions.index(inst)+1]
                start_idx = inst_idx
                spill_reg = inst.spill_reg
                if len(available_reg) > 0:
                    new_reg = available_reg.pop()
                    
                    next_spill_idx = spill_insts.index(inst)+1
                    if next_spill_idx == len(spill_insts):
                        next_spill_idx = -1
                    
                    if next_spill_idx > -1:
                        next_spill_idx = block.instructions.index(spill_insts[next_spill_idx])
                        
                    rename_registers_inst(block.instructions[start_idx:next_spill_idx], {spill_reg.name:new_reg}, update_dest = False)
                    inst.dest = Register(new_reg)
                    rename_registers_inst([paired_inst], {spill_reg.name:new_reg}, update_dest = False)

            elif isinstance(inst, SpillStoreInstruction):
                inst_idx = block.instructions.index(inst)
                paired_inst = block.instructions[inst_idx-1]
                start_idx = inst_idx - 1
                spill_reg = inst.spill_reg
                
                if len(available_reg) > 0:
                    new_reg = available_reg.pop()
                    
                    next_spill_idx = spill_insts.index(inst)+1
                    if next_spill_idx == len(spill_insts):
                        next_spill_idx = -1
                    
                    if next_spill_idx > -1:
                        next_spill_idx = block.instructions.index(spill_insts[next_spill_idx])
                        
                    rename_registers_inst(block.instructions[start_idx:next_spill_idx], {spill_reg.name:new_reg}, update_dest = False)
                    paired_inst.dest = Register(new_reg)

            barrier_tracker.update_flags(inst.flags)
        pprint(block.instructions)
    cfg.create_dot_graph("cfg.dot")
    
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

            
            
        