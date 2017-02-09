from pprint import pprint
from pycuasm.compiler.hir import *
from pycuasm.compiler.analysis import *
from pycuasm.compiler.utils import *
from pycuasm.compiler.transform import *

REL_OFFSETS = ['BRA', 'SYNC', 'CAL', 'PBK', 'PCNT']
ABS_OFFSETS = ['JCAL']
JUMP_OPS = REL_OFFSETS + ABS_OFFSETS

SHARED_MEMORY_STALL = 40

def analyze_block_liveness(block):
    # Perform liveness analysis at block level
    converge = False
    next_inst = None
    
    for inst in reversed(block.instructions):
        setattr(inst, 'next_inst', next_inst)
        setattr(inst, 'live_in', set())
        setattr(inst, 'live_out', set())
        setattr(inst, 'var_def', set())
        setattr(inst, 'var_use', set())
        if isinstance(inst.dest, Register):
            inst.var_def |=  set([inst.dest.name])
        elif isinstance(inst.dest, Pointer):
            inst.var_use |= set([inst.dest.register.name])
        
        for op in inst.operands:
            if isinstance(op, Register):
                inst.var_use |=  set([op.name])
            elif isinstance(op, Pointer):
                inst.var_use |= set([op.register.name])
        next_inst = inst
        
    while not converge:
        for inst in reversed(block.instructions):
            setattr(inst, 'old_live_in', inst.live_in.copy())
            setattr(inst, 'old_live_out', inst.live_out.copy()) 
            if inst.next_inst:
                inst.live_out = inst.next_inst.live_in  
            inst.live_in = inst.var_use | (inst.live_out - inst.var_def)
            
        converge = True
        for inst in reversed(block.instructions):
            if inst.old_live_in != inst.live_in:
                converge = False


def find_new_spill_reg(block, available_reg, start_idx, next_spill_idx, avoid_conflict):
    new_reg = None
    if avoid_conflict:  
        reg_bank_list = [0,0,0,0]
        for local_inst in block.instructions[start_idx:next_spill_idx]:
            if local_inst.opcode.type == 'x32':
                for op in local_inst.operands:
                    if isinstance(op, Register):
                        reg_bank_list[op.id % 4] += 1
                        
        reg_bank = sorted(range(len(reg_bank_list)), key=lambda k: reg_bank_list[k])

        for bank in reg_bank:
            for reg in available_reg:
                if int(reg.replace('R','')) % 4 == bank:
                    new_reg = reg
                    break
            if new_reg:
                break
    else:
        new_reg = available_reg[-1]
    
    return new_reg
        
def rearrange_spill_instruction(program, spill_register, spill_addr_register, avoid_conflict=True):
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
    #for block in block_list[:1]:
        pprint(block.line)
        pprint(block.instructions)
         
        next_inst = None
    
        # Add attribute storing next instruction
        for inst in reversed(block.instructions):
            setattr(inst, 'next_inst', next_inst)
            next_inst = inst
                 
        available_reg = copy.copy(block.free_reg)
        available_reg = sorted(available_reg, key=lambda x: int(x.replace('R','')))
        if 'R1' in available_reg:
            available_reg.remove('R1')
        
        #pprint(available_reg)

        barrier_tracker = BarrierTracker()
        barrier_tracker.reset()
                
        spill_location_dict = {}    
        spill_store_dict = {}
        
        # Perform liveness analysis at block level
        analyze_block_liveness(block)
                
        # Collect spill instructions in the basic block                    
        spill_insts = [x for x in block.instructions if isinstance(x, SpillInstruction)]
        
        to_remove_list = []
        
        for inst in block.instructions:
            if isinstance(inst, SpillLoadInstruction):
                inst_idx = block.instructions.index(inst)   
                start_idx = inst_idx
                spill_reg = inst.spill_reg
                
                new_reg = None
                
                # Getting instruction that is the target of this spill Instruction 
                paired_inst = block.instructions[block.instructions.index(inst)+1]
                
                next_spill_idx = spill_insts.index(inst)+1
                if next_spill_idx == len(spill_insts):
                    next_spill_idx = -1
                else:
                    next_spill_idx = block.instructions.index(spill_insts[next_spill_idx])
                
                # There is no need to load
                if inst.shared_offset in spill_location_dict:
                    new_reg = spill_location_dict[inst.shared_offset]
                    to_remove_list.append(inst)
                    flags = (1 << (inst.flags.read_barrier-1) | 1 << (inst.flags.write_barrier-1))
                    paired_inst.flags.wait_barrier = paired_inst.flags.wait_barrier & ~flags
                    paired_inst.flags.wait_barrier |= inst.flags.wait_barrier  

                elif len(available_reg) > 0:
                    new_reg = find_new_spill_reg(block, available_reg, start_idx, next_spill_idx, avoid_conflict)
                    if new_reg:
                        print(new_reg, spill_reg)
                        pprint(block.instructions[start_idx:next_spill_idx])
                        available_reg.remove(new_reg)
                        spill_location_dict[inst.shared_offset] = new_reg
                        spill_store_dict[inst.shared_offset] = []
                        pprint(spill_location_dict)
                    
                if new_reg:
                    rename_registers_inst(block.instructions[start_idx:next_spill_idx], {spill_reg.name:new_reg}, update_dest = False)
                    inst.dest = Register(new_reg)
                    rename_registers_inst([paired_inst], {spill_reg.name:new_reg}, update_dest = False)
                                        
            elif isinstance(inst, SpillStoreInstruction):
                inst_idx = block.instructions.index(inst)
                start_idx = inst_idx - 1
                spill_reg = inst.spill_reg
                
                new_reg = None
                
                # Getting instruction that produce value for this store
                paired_inst = block.instructions[inst_idx-1]

                next_spill_idx = spill_insts.index(inst)+1
                if next_spill_idx == len(spill_insts):
                    next_spill_idx = -1
                else:
                    next_spill_idx = block.instructions.index(spill_insts[next_spill_idx])
                
                if inst.shared_offset in spill_location_dict:
                    new_reg = spill_location_dict[inst.shared_offset]
                    # There is no need for this value
                    if not new_reg in inst.live_out:
                        #pprint(new_reg)
                        spill_location_dict.pop(inst.shared_offset)
                        available_reg.append(new_reg)
                    
                elif len(available_reg) > 0:
                    new_reg = find_new_spill_reg(block, available_reg, start_idx, next_spill_idx, avoid_conflict)
                    if new_reg:
                        print(new_reg, spill_reg)
                        pprint(block.instructions[start_idx:next_spill_idx])
                        available_reg.remove(new_reg)
                        spill_location_dict[inst.shared_offset] = new_reg
                        spill_store_dict[inst.shared_offset] = []  
                        pprint(spill_location_dict)
                                              
                if new_reg:
                    rename_registers_inst(block.instructions[start_idx:next_spill_idx], {spill_reg.name:new_reg}, update_dest = False)
                    paired_inst.dest = Register(new_reg)
                    spill_store_dict[inst.shared_offset].append(inst)
                    
            barrier_tracker.update_flags(inst.flags)
            
        for inst in to_remove_list:
            print("[REG_OPT] Remove: %s" % inst)
            program.ast.remove(inst)
            block.instructions.remove(inst)
        
        analyze_block_liveness(block)
        #pprint(block.instructions)
        #pprint(spill_store_dict)
        
        for offset in spill_store_dict:
            if not spill_store_dict[offset]:
                continue
                
            for inst in spill_store_dict[offset][:-1]:
                print("[REG_OPT] Remove: %s" % inst)
                flags = (1 << (inst.flags.read_barrier-1))
                inst.next_inst.flags.wait_barrier = inst.next_inst.flags.wait_barrier & ~flags
                inst.next_inst.flags.wait_barrier |= inst.flags.wait_barrier  
                program.ast.remove(inst)
                block.instructions.remove(inst)
            
            inst = spill_store_dict[offset][-1]
            inst.flags.yield_hint = True
            flags = (1 << (inst.flags.read_barrier-1))
            inst.next_inst.flags.wait_barrier = inst.next_inst.flags.wait_barrier & ~flags
            inst.next_inst.flags.wait_barrier |= inst.flags.wait_barrier
        
        analyze_block_liveness(block)

        #for inst in block.instructions:
        #    pprint(inst.live_in, width=200)
        #    pprint(inst)
        #    pprint(inst.live_out, width=200)
        
        #pprint(spill_store_dict)   
        #pprint(block.instructions)
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
            

            
            
        