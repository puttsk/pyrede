from pprint import pprint
from pycuasm.compiler.hir import *
from pycuasm.compiler.analysis import *
from pycuasm.compiler.utils import *
from pycuasm.compiler.transform import *

REL_OFFSETS = ['BRA', 'SYNC', 'CAL', 'PBK', 'PCNT']
ABS_OFFSETS = ['JCAL']
JUMP_OPS = REL_OFFSETS + ABS_OFFSETS

SHARED_MEMORY_STALL = 40

def __analyze_block_liveness(block):
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


def __find_new_spill_reg(block, available_reg, start_idx, next_spill_idx, avoid_conflict):
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
        
def opt_swap_spill_register(program, avoid_conflict=True):
    """
    Optimize each basic block by renaming spill register to free register. 
    Therefore, multiple spill values can coexist at the same time.
    In addition, some spilling instructions and barriers can be removed.       
    """
    
    print("[SPREG_OPT]: optimize_spill_register ")
    
    cfg = Cfg(program)
    cfg.analyze_liveness()
    
    regs_non_32 = set(collect_non_32bit_registers(program))
    regs_32 = set(program.registers) - regs_non_32
    #pprint(program.registers)

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
    
    # Pruning block without any spill instruction
    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
            
        if any(isinstance(x, SpillInstruction) for x in block.instructions):
            block_list.append(block)
                
    for block in block_list:
    #for block in block_list[23:24]:
    #for block in block_list[:1]:
        #pprint(block.__class__)
        #pprint(block.line)
        #pprint(block.instructions)
         
        next_inst = None
    
        # Add attribute storing next instruction
        for inst in reversed(block.instructions):
            setattr(inst, 'next_inst', next_inst)
            next_inst = inst
                 
        available_reg = copy.copy(block.free_reg)
        available_reg = sorted(available_reg, key=lambda x: int(x.replace('R','')))
        if 'R1' in available_reg:
            available_reg.remove('R1')
        
        available_reg = available_reg[:6]
        
        #pprint(available_reg)
                
        spill_location_dict = {}    
        spill_store_dict = {}
        
        # Perform liveness analysis at block level
        __analyze_block_liveness(block)
                
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
                    new_reg = __find_new_spill_reg(block, available_reg, start_idx, next_spill_idx, avoid_conflict)
                    if new_reg:
                        available_reg.remove(new_reg)
                        spill_location_dict[inst.shared_offset] = new_reg
                        spill_store_dict[inst.shared_offset] = []
                    
                if new_reg:
                    if next_spill_idx != -1:
                        rename_registers_inst(block.instructions[start_idx:next_spill_idx], {spill_reg.name:new_reg}, update_dest = False)
                    else:
                        rename_registers_inst(block.instructions[start_idx:], {spill_reg.name:new_reg}, update_dest = False)

                    inst.dest = Register(new_reg)
                    rename_registers_inst([paired_inst], {spill_reg.name:new_reg}, update_dest = False)
                    print("[SPREG_OPT] Rename load spill regiter %s to %s" % (spill_reg.name, new_reg))
                
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
                        spill_location_dict.pop(inst.shared_offset)
                        available_reg.append(new_reg)
                    
                elif len(available_reg) > 0:
                    new_reg = __find_new_spill_reg(block, available_reg, start_idx, next_spill_idx, avoid_conflict)
                    if new_reg:
                        available_reg.remove(new_reg)
                        spill_location_dict[inst.shared_offset] = new_reg
                        spill_store_dict[inst.shared_offset] = []  
                                              
                if new_reg:
                    print("[SPREG_OPT] Rename store spill regiter %s to %s" % (spill_reg.name, new_reg))
                    if next_spill_idx != -1:
                        rename_registers_inst(block.instructions[start_idx+1:next_spill_idx], {spill_reg.name:new_reg}, update_dest = False)
                    else:
                        rename_registers_inst(block.instructions[start_idx+1:], {spill_reg.name:new_reg}, update_dest = False)                                        
                    paired_inst.dest = Register(new_reg)
                    spill_store_dict[inst.shared_offset].append(inst)
                
            #pprint(spill_location_dict)
        
        # Remove unnecessary spill load/store instruction
        for inst in to_remove_list:
            print("[SPREG_OPT] Remove: %x: %s" % (inst.addr, inst))
            program.ast.remove(inst)
            block.instructions.remove(inst)
        
        __analyze_block_liveness(block)
        
        for offset in spill_store_dict:
            if not spill_store_dict[offset]:
                continue
                
            for inst in spill_store_dict[offset][:-1]:
                print("[SPREG_OPT] Remove: %x: %s" % (inst.addr, inst))
                flags = (1 << (inst.flags.read_barrier-1))
                inst.next_inst.flags.wait_barrier = inst.next_inst.flags.wait_barrier & ~flags
                inst.next_inst.flags.wait_barrier |= inst.flags.wait_barrier  
                program.ast.remove(inst)
                block.instructions.remove(inst)
            
            inst = spill_store_dict[offset][-1]
            inst.flags.yield_hint = True
            flags = (1 << (inst.flags.read_barrier-1))
            if inst.next_inst:
                inst.next_inst.flags.wait_barrier = inst.next_inst.flags.wait_barrier & ~flags
                inst.next_inst.flags.wait_barrier |= inst.flags.wait_barrier
        
        #pprint(block.instructions)
        __analyze_block_liveness(block)

    cfg.create_dot_graph("cfg.dot")
    
    program.update()
        
def opt_remove_redundant_spill_inst(program, spill_addr_register):
    """
    Removing redundant spill load/store. The current implementation only work with 32-bit spill
    This optimization track the current value of the spill register and 
    remove unneccessary spill load/store.
    """
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
            
def opt_hoist_spill_instruction(program):
    print("[SPINT_OPT]: hoist_spill_instruction ")
    
    cfg = Cfg(program)
    cfg.analyze_liveness()
    
    block_list = []
        
    # Pruning block without any spill instruction
    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
            
        if any(isinstance(x, SpillInstruction) for x in block.instructions):
            block_list.append(block)
                
    for block in block_list:
    #for block in block_list[23:24]:
        #pprint(block.line)
        #pprint(block.instructions)

        # Hoisting spill load instruction to perform as early as possible
        for inst in block.instructions:
            if isinstance(inst, SpillLoadInstruction):
                block_idx =  block.instructions.index(inst)
                ast_idx =  program.ast.index(inst)
                
                for rev_inst in reversed(block.instructions[:block_idx]):
                    target_block_idx = 0
                    # Found an instruction that uses the spill register or other spill instruction
                    # This location is the earliest the load can happen
                    if rev_inst.has_register(inst.spill_reg) or isinstance(rev_inst, SpillLoadInstruction):
                        target_block_idx =  block.instructions.index(rev_inst) + 1
                        break
                    elif isinstance(rev_inst, SpillStoreInstruction):
                        if rev_inst.spill_reg == inst.spill_reg:
                            target_block_idx =  block.instructions.index(rev_inst) + 1
                            break
                        else:
                            #print(inst, rev_inst)
                            #print(inst.spill_reg, rev_inst.spill_reg)
                            continue
                    
                target_ast_idx = program.ast.index(block.instructions[target_block_idx])
                
                if target_ast_idx < ast_idx:    
                    block.instructions.insert(target_block_idx, block.instructions.pop(block_idx))
                    program.ast.insert(target_ast_idx, program.ast.pop(ast_idx))
                
                    print("[SPINT_OPT]: Move %s from %d to %d" % (inst, ast_idx, target_ast_idx))
        
        # Finding instruction that is waited for spill instruction barrier
        pair_spill_inst_dict = {}
        for inst in block.instructions:
            if isinstance(inst, SpillInstruction):
                # Find pair inst 
                block_idx =  block.instructions.index(inst)
                p_block_idx = -1
                for p_inst in block.instructions[block_idx+1:]:
                    if p_inst.has_register(inst.spill_reg):
                        # Check if barrier match
                        inst_flag = 0
                        if isinstance(inst, SpillLoadInstruction): 
                            inst_flags = 1 << (inst.flags.read_barrier-1) | 1 << (inst.flags.write_barrier-1)
                        else:
                            inst_flags = 1 << (inst.flags.read_barrier-1) 
                            
                        if p_inst.flags.wait_barrier & inst_flags != 0:
                            p_block_idx = block.instructions.index(p_inst)                       
                            break
                            
                pair_spill_inst_dict[block_idx] = p_block_idx
            
            #print(block.instructions.index(inst),inst)
        
        # Update the barrier to avoid unneccessary wait
        b_tracker = BarrierTracker()
        for inst in block.instructions:
            inst_idx = block.instructions.index(inst)
            if isinstance(inst, SpillLoadInstruction):
                # Read barrier is used by early instructions
                if not b_tracker.is_available(inst.flags.read_barrier):
                    # Find next wait on the BarrierTracker
                    # If the wait is farther than SHARED_MEMORY_STALL cycle,
                    # there is no need to change.  
                    stall_count = 0
                    change_barrier = False
                    for n_inst in block.instructions[inst_idx+1:pair_spill_inst_dict[inst_idx]]:                    
                        if stall_count > SHARED_MEMORY_STALL:
                            break
                        
                        if n_inst.flags.wait_barrier & (1 << (inst.flags.read_barrier-1)) != 0:
                            # Should change barrier id
                            change_barrier = True
                            break
                        
                        stall_count += n_inst.flags.stall
                        
                    if change_barrier:
                        b_tracker_list = copy.copy(b_tracker.barriers)
                        for n_inst in block.instructions[inst_idx:pair_spill_inst_dict[inst_idx]]:
                            if n_inst.flags.read_barrier != 0:
                                b_tracker_list[n_inst.flags.read_barrier-1] = 'r'
                            if n_inst.flags.write_barrier != 0:
                                b_tracker_list[n_inst.flags.read_barrier-1] = 'w'
                        
                        # Check if ther is available barrier
                        if '-' in b_tracker_list:
                            new_barrier = b_tracker_list.index('-') + 1
                            print("[SPINT_OPT]: Change read barrier of %s to %d" % (inst, new_barrier))
                            
                            p_inst = block.instructions[pair_spill_inst_dict[inst_idx]]
                            p_inst.flags.wait_barrier = p_inst.flags.wait_barrier & ~(1 << (inst.flags.read_barrier-1))
                            inst.flags.read_barrier = new_barrier
                            p_inst.flags.wait_barrier = p_inst.flags.wait_barrier | (1 << (inst.flags.read_barrier-1))

                # Write barrier is used by early instructions
                if not b_tracker.is_available(inst.flags.write_barrier):
                    # Find next wait on the BarrierTracker
                    # If the wait is farther than SHARED_MEMORY_STALL cycle,
                    # there is no need to change.  
                    stall_count = 0
                    change_barrier = False
                    for n_inst in block.instructions[inst_idx+1:pair_spill_inst_dict[inst_idx]]:                    
                        if stall_count > SHARED_MEMORY_STALL:
                            break
                        
                        if n_inst.flags.wait_barrier & (1 << (inst.flags.write_barrier-1)) != 0:
                            # Should change barrier id
                            change_barrier = True
                            break
                        
                        stall_count += n_inst.flags.stall
                        
                    if change_barrier:
                        b_tracker_list = copy.copy(b_tracker.barriers)
                        for n_inst in block.instructions[inst_idx:pair_spill_inst_dict[inst_idx]]:
                            if n_inst.flags.read_barrier != 0:
                                b_tracker_list[n_inst.flags.read_barrier-1] = 'r'
                            if n_inst.flags.write_barrier != 0:
                                b_tracker_list[n_inst.flags.write_barrier-1] = 'w'
                        
                        # Check if ther is available barrier
                        if '-' in b_tracker_list:
                            new_barrier = b_tracker_list.index('-') + 1
                            print("[SPINT_OPT]: Change write barrier of %s to %d" % (inst, new_barrier))
                            
                            p_inst = block.instructions[pair_spill_inst_dict[inst_idx]]
                            p_inst.flags.wait_barrier = p_inst.flags.wait_barrier & ~(1 << (inst.flags.write_barrier-1))
                            inst.flags.write_barrier = new_barrier
                            p_inst.flags.wait_barrier = p_inst.flags.wait_barrier | (1 << (inst.flags.write_barrier-1))
                        
                        
            b_tracker.update_flags(inst.flags)
            
            
        pprint(pair_spill_inst_dict)
        pprint(block.instructions)
    
    program.update()
                                        
                    
                        
    