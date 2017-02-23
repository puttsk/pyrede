import copy
import itertools
import math

from pprint import pprint
from pycuasm.compiler.hir import *
from pycuasm.compiler.analysis import *
from pycuasm.compiler.utils import *

class RelocatableRegister(object):
    def __init__(self, lead_register, bits=32 ):
        self.lead_register = lead_register
        self.registers = []     
        self.relocated_register = []    
        self.bits = bits
        self.new_lead_register = None
        self.new_registers = []

        reg_id = lead_register.id
        if self.bits > 32:
            reg_count = int(self.bits / 32)
            #if reg_id % reg_count != 0:
            #    reg_id = reg_id - (reg_id % reg_count)
            for r in range(0, reg_count):
                self.registers.append(Register("R%d" % (reg_id+r)))
        else:
            self.registers.append(self.lead_register)
    
        self.lead_register = self.registers[0]
    
    def __repr__(self):
        repr_str =  "%3d-bit Register %s: %s" % (self.bits, self.lead_register, self.registers)
        if self.new_lead_register:
            repr_str +=  " -> Register %s: %s" % (self.new_lead_register, self.new_registers)
        return repr_str
    
    def __hash__(self):
        return hash(frozenset(self.registers))
    
    def __eq__(self, other):
        if not isinstance(other, RelocatableRegister):
            return False
        
        if len(self.registers) != len(other.registers):
            return False
        
        for i in range(len(self.registers)):
            if self.registers[i] != other.registers[i]:
                return False
        return True
    
    def move(self, new_register):
        self.new_lead_register = new_register
        if self.bits > 32:
            reg_count = int(self.bits / 32)
            #if new_register.id % reg_count != 0:
            #    new_register.id = new_register.id - (new_register.id % reg_count)
            for r in range(0, reg_count):
                self.new_registers.append(Register("R%d" % (new_register.id+r)))
        else:
            self.new_registers.append(self.new_lead_register)

def relocate_registers_conflict(program):
    print("[REG_RELOC] Relocating registers.")
    relocation_space_size = int(program.registers[-1].replace('R','')) + 1
    relocation_space = [None] * (relocation_space_size + 1)
    
    # Creating a list of non 32bit register
    # 64-bit integer and 64-bit floating point use different rules.
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        # Detecting 64-bit accesses
        if (inst.opcode.type == 'x32' and 
            inst.opcode.integer_inst and 
            inst.opcode.reg_store and 
            inst.dest):
            if inst.dest.carry_bit:
                opcode = inst.opcode
                inst_pos = program.ast.index(inst)
                
                for next_inst in [x for x in program.ast[inst_pos:] if isinstance(x, Instruction)]:
                    if next_inst.opcode.use_carry_bit:
                        relocate_reg = RelocatableRegister(inst.dest, bits=64) 
                        for reg in relocate_reg.registers:
                            if not relocation_space[reg.id]: 
                                relocation_space[reg.id] = relocate_reg
                            elif relocation_space[reg.id].bits < relocate_reg.bits:
                                relocation_space[reg.id] = relocate_reg
                        #reg_set.add(next_inst.dest.name)
                        break
        
        #if (inst.opcode.op_bit != 32):
        reg_list = [x for x in inst.operands if isinstance(x, Register)]
        if inst.dest and isinstance(inst.dest, Register):
            reg_list.append(inst.dest)
        
        for reg in reg_list:
            if reg.is_special:
                continue
                
            relocate_reg = RelocatableRegister(reg, bits=inst.opcode.op_bit)
            for reg in relocate_reg.registers:
                if not relocation_space[reg.id]: 
                    relocation_space[reg.id] = relocate_reg
                elif relocation_space[reg.id].bits < relocate_reg.bits:
                    relocation_space[reg.id] = relocate_reg
    
    for reg in program.registers:
        reg_id = int(reg.replace('R',''))
        if relocation_space[reg_id] == None:    
            relocation_space[reg_id] = RelocatableRegister(Register(reg))

    empty_location_list = []
    for loc in range(relocation_space_size):
        if not relocation_space[loc]:
            empty_location_list.append(loc)
    
    pprint(relocation_space)

    empty_idx = 0
    print("[REG_RELOC] Empty localtion %s" % (empty_location_list))
    while len(empty_location_list) > 0:
        empty_loc = empty_location_list.pop(0)
        empty_loc_bank = empty_loc % 4 # Computing register bank
        cont_loc_count = 1
        
        # Finding continuous empty location
        while empty_location_list and empty_location_list[0] - empty_loc == cont_loc_count:
            empty_location_list.pop(0)
            cont_loc_count += 1
        
        while cont_loc_count > 0:
            # Get next register 
            next_reg = None
            next_reg_idx = empty_loc 
            while not next_reg and next_reg_idx < len(relocation_space)-1:
                next_reg_idx = next_reg_idx + 1
                next_reg = relocation_space[next_reg_idx]
            if not next_reg:
                # End of list
                break
             
            if empty_loc % 2 == 1 and next_reg.bits > 32:
                # The available register is odd register
                # Only 32-bit register is allowed here
                # Find 32-bit register from the back of reglocation_space
                found_reg = False                
                for i in reversed(range(len(relocation_space))):
                    reg_reloc = relocation_space[i]
                    if reg_reloc and reg_reloc.bits == 32 and i > empty_loc:
                        if reg_reloc.lead_register.id % 4 != empty_loc_bank:
                            continue
                        
                        relocation_space[empty_loc] = relocation_space[i]
                        relocation_space[i] = None
                        print("[REG_RELOC] Move R%d to R%d"  % (i, empty_loc))
                        relocation_space[empty_loc].move(Register('R%d' % empty_loc))
                        found_reg = True
                        break
                    
                cont_loc_count = cont_loc_count - 1
                # Move to next location
                empty_loc = empty_loc + 1
            else:
                # The available register is even register
                # Check if the next register is movable
                non_conflict_loc = None
                for loc in relocation_space[empty_loc + cont_loc_count:]:
                    if loc and loc.lead_register.id % 4 == empty_loc %4:
                        required_reg_count = max(int(loc.bits / 32), 1)
                        if required_reg_count <= cont_loc_count:
                            non_conflict_loc = loc
                            break

                if non_conflict_loc:                    
                    non_conflict_idx = relocation_space.index(non_conflict_loc)
                    #relocation_space[empty_loc] = copy.copy(relocation_space[non_conflict_idx])
                    #relocation_space[empty_loc].move(Register('R%d' % (empty_loc)))
                    print("[REG_RELOC] Move %d-bit R%s to R%d - avoid conflict"  % (non_conflict_loc.bits, non_conflict_loc.lead_register, empty_loc))
                    for i in range(required_reg_count):
                        relocation_space[empty_loc + i] = relocation_space[non_conflict_idx + i]
                        relocation_space[non_conflict_idx + i] = None
                        empty_location_list.append(non_conflict_idx + i)
                    relocation_space[empty_loc].move(Register('R%d' % (empty_loc)))
                    required_reg_count = max(int(non_conflict_loc.bits / 32), 1)
                    empty_loc = empty_loc + required_reg_count
                    cont_loc_count -= required_reg_count
                    empty_location_list = list(sorted(empty_location_list))
                    #pprint(empty_location_list)
                else:
                    required_reg_count = max(int(next_reg.bits / 32), 1)
                    if empty_loc % required_reg_count == 0:
                        # The register is movable. Move register to empty location
                        for i in range(required_reg_count):
                            relocation_space[empty_loc + i] = relocation_space[next_reg_idx + i]
                            relocation_space[next_reg_idx + i] = None
                        print("[REG_RELOC] Move %d-bit R%d to R%d - move 1"  % (next_reg.bits, next_reg_idx, empty_loc))
                        relocation_space[empty_loc].move(Register('R%d' % empty_loc))  
                        # Move to next new empty space without reducing the size
                        empty_loc = empty_loc + required_reg_count
                    else: 
                        # TODO: The register is not movable
                        possible_empty_loc = (int(math.floor(empty_loc / required_reg_count)) + 1) * required_reg_count
                        # The register is movable. Move register to empty location
                        for i in range(required_reg_count):
                            relocation_space[possible_empty_loc + i] = relocation_space[next_reg_idx + i]
                            relocation_space[next_reg_idx + i] = None
                        print("[REG_RELOC] Move %d-bit R%d to R%d - move 2"  % (next_reg.bits, next_reg_idx, possible_empty_loc))
                        relocation_space[possible_empty_loc].move(Register('R%d' % possible_empty_loc))
                        #pprint(relocation_space)
                        empty_loc = possible_empty_loc + required_reg_count
                        #raise RuntimeError('[REG_RELOC] Register %s to %d is %d-bit and not movable' % (next_reg, empty_loc, required_reg_count * 32)) 
                
                if empty_location_list and empty_loc + cont_loc_count >= empty_location_list[0]:
                    # The current empty space merge with the next one.
                    # Update the size of empty space 
                    while empty_location_list and empty_location_list[0] - empty_loc == cont_loc_count:
                        empty_location_list.pop(0)
                        cont_loc_count += 1

    pprint(relocation_space)
    has_empty_space = False
    for reg_reloc in relocation_space:
        if not reg_reloc:
            has_empty_space = True
        if reg_reloc and reg_reloc.new_lead_register:
            for i in range(len(reg_reloc.new_registers)):
                rename_register(program, reg_reloc.registers[i], reg_reloc.new_registers[i])

    program.update()
    if has_empty_space:
        relocate_registers(program)

def relocate_registers(program):
    print("[REG_RELOC] Relocating registers.")
    
    relocation_space_size = int(program.registers[-1].replace('R','')) + 1
    relocation_space = [None] * (relocation_space_size + 1)
    
    # Creating a list of non 32bit register
    # 64-bit integer and 64-bit floating point use different rules.
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        # Detecting 64-bit accesses
        if (inst.opcode.type == 'x32' and 
            inst.opcode.integer_inst and 
            inst.opcode.reg_store and 
            inst.dest):
            if inst.dest.carry_bit:
                opcode = inst.opcode
                inst_pos = program.ast.index(inst)
                
                for next_inst in [x for x in program.ast[inst_pos:] if isinstance(x, Instruction)]:
                    if next_inst.opcode.use_carry_bit:
                        relocate_reg = RelocatableRegister(inst.dest, bits=64) 
                        for reg in relocate_reg.registers:
                            if not relocation_space[reg.id]: 
                                relocation_space[reg.id] = relocate_reg
                            elif relocation_space[reg.id].bits < relocate_reg.bits:
                                relocation_space[reg.id] = relocate_reg
                        #reg_set.add(next_inst.dest.name)
                        break
        
        #if (inst.opcode.op_bit != 32):
        reg_list = [x for x in inst.operands if isinstance(x, Register)]
        if inst.dest and isinstance(inst.dest, Register):
            reg_list.append(inst.dest)
        
        for reg in reg_list:
            if reg.is_special:
                continue
                
            relocate_reg = RelocatableRegister(reg, bits=inst.opcode.op_bit)
            for reg in relocate_reg.registers:
                if not relocation_space[reg.id]: 
                    relocation_space[reg.id] = relocate_reg
                elif relocation_space[reg.id].bits < relocate_reg.bits:
                    relocation_space[reg.id] = relocate_reg
    
    for reg in program.registers:
        reg_id = int(reg.replace('R',''))
        if relocation_space[reg_id] == None:    
            relocation_space[reg_id] = RelocatableRegister(Register(reg))

    empty_location_list = []
    for loc in range(relocation_space_size):
        if not relocation_space[loc]:
            empty_location_list.append(loc)
    
    #pprint(relocation_space)

    empty_idx = 0
    print("[REG_RELOC] Empty localtion %s" % (empty_location_list))
    while len(empty_location_list) > 0:
        empty_loc = empty_location_list.pop(0)
        cont_loc_count = 1
        while empty_location_list and empty_location_list[0] - empty_loc == cont_loc_count:
            empty_location_list.pop(0)
            cont_loc_count += 1
        
        while cont_loc_count > 0:
            # Get next register 
            next_reg = None
            next_reg_loc = empty_loc 
            while not next_reg and next_reg_loc < len(relocation_space)-1:
                next_reg_loc = next_reg_loc + 1
                next_reg = relocation_space[next_reg_loc]
            if not next_reg:
                # End of list
                break

            if empty_loc % 2 == 1 and next_reg.bits > 32:
                # The available register is odd register
                # Only 32-bit register is allowed here
                # Find 32-bit register from the back of reglocation_space
                for i in reversed(range(len(relocation_space))):
                    reg_reloc = relocation_space[i]
                    if reg_reloc and reg_reloc.bits == 32 and i > empty_loc:
                        relocation_space[empty_loc] = relocation_space[i]
                        relocation_space[i] = None
                        print("[REG_RELOC] Move R%d to R%d"  % (i, empty_loc))
                        relocation_space[empty_loc].move(Register('R%d' % empty_loc))
                        break
                cont_loc_count = cont_loc_count - 1
                # Move to next location
                empty_loc = empty_loc + 1
            else:
                # The available register is even register
                # Check if the next register is movable
                required_reg_count = max(int(next_reg.bits / 32), 1)
                if empty_loc % required_reg_count == 0:
                    # The register is movable. Move register to empty location
                    for i in range(required_reg_count):
                        relocation_space[empty_loc + i] = relocation_space[next_reg_loc + i]
                        relocation_space[next_reg_loc + i] = None
                    print("[REG_RELOC] Move %d-bit R%d to R%d"  % (next_reg.bits, next_reg_loc, empty_loc))
                    relocation_space[empty_loc].move(Register('R%d' % empty_loc))  
                    # Move to next new empty space without reducing the size
                    empty_loc = empty_loc + required_reg_count
                else: 
                    # TODO: The register is not movable
                    possible_empty_loc = (int(math.floor(empty_loc / required_reg_count)) + 1) * required_reg_count
                    # The register is movable. Move register to empty location
                    for i in range(required_reg_count):
                        relocation_space[possible_empty_loc + i] = relocation_space[next_reg_loc + i]
                        relocation_space[next_reg_loc + i] = None
                    print("[REG_RELOC] Move %d-bit R%d to R%d"  % (next_reg.bits, next_reg_loc, possible_empty_loc))
                    relocation_space[possible_empty_loc].move(Register('R%d' % possible_empty_loc))
                    pprint(relocation_space)
                    empty_loc = possible_empty_loc + required_reg_count
                    #raise RuntimeError('[REG_RELOC] Register %s to %d is %d-bit and not movable' % (next_reg, empty_loc, required_reg_count * 32)) 
            
            if empty_location_list and empty_loc + cont_loc_count >= empty_location_list[0]:
                # The current empty space merge with the next one.
                # Update the size of empty space 
                while empty_location_list and empty_location_list[0] - empty_loc == cont_loc_count:
                    empty_location_list.pop(0)
                    cont_loc_count += 1
                #pprint((empty_loc, cont_loc_count))
            #pprint(relocation_space)
            #input("Press Enter to continue...")
    pprint(relocation_space)
    for reg_reloc in relocation_space:
        if reg_reloc and reg_reloc.new_lead_register:
            for i in range(len(reg_reloc.new_registers)):
                rename_register(program, reg_reloc.registers[i], reg_reloc.new_registers[i])

def relocate_registers_old(program):
    print("[REG_RELOC] Relocating registers.")
    program_regs = sorted(program.registers, key=lambda x: int(x.replace('R','')))
    #pprint(program_regs)
    
    #reg_mem =  collect_global_memory_access(program)
    reg_mem = []
    reg_64 = collect_64bit_registers(program)
    exclude_64 = []
    for reg in reg_64:
        if abs(int(reg[0].replace('R','')) - int(reg[1].replace('R',''))) != 1:
            exclude_64.append(reg)
    reg_64 = set([x for x in reg_64 if x not in exclude_64])            
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
                #if reg_next_new not in reg_64:
                reg_skip.append(reg_next_new)
                if reg_next_new_id != reg_next_id:
                    rename_register(program, Register(reg_next), Register('R%d' % (reg_next_new_id+1)))
                    rename_register(program, Register('R%d' % (reg_next_id+1)), Register('R%d' % (reg_next_new_id+2)))
                    program_regs[idx+1] = 'R%d' % (reg_next_new_id+1)
                    program_regs[idx+2] = 'R%d' % (reg_next_new_id+2)
                    idx = idx + 2
                else:
                    idx = idx + 2

        if idx == len(program_regs)-1:
            end = True
    

def rename_registers_inst(instructions, registers_dict, update_dest = True):
    for inst in instructions:
        if not isinstance(inst, Instruction):
            continue
        
        for op in inst.operands:
            if isinstance(op, Pointer):
                if op.register.name in registers_dict:
                    op.register.rename(Register(registers_dict[op.register.name]))
            elif isinstance(op, Constant) and op.pointer:
                if op.pointer.register.name in registers_dict:
                    op.pointer.register.rename(Register(registers_dict[op.pointer.register.name]))
            elif isinstance(op, Register):
                if op.name in registers_dict:
                    inst.operands[inst.operands.index(op)].rename(Register(registers_dict[op.name]))
                
        if update_dest:
            if isinstance(inst.dest, Pointer):
                if inst.dest.register.name in registers_dict:
                    inst.dest.register.rename(Register(registers_dict[inst.dest.register.name]))
            elif isinstance(inst.dest, Register):
                if inst.dest.name in registers_dict:
                    inst.dest.rename(Register(registers_dict[inst.dest.name]))    

def rename_registers(program, registers_dict):
    """ Renaming registers in a program using rules in registers_dict 
        
        Args:
            program (Program): Target program for register renaming
            registers_dict (dict{str:str}): A dictionary containing a renaming rules. The key represent the old register name and a value contain new register name 
    """
    #print("Renaming using dict:" )
    #pprint(registers_dict)
    rename_registers_inst(program.ast, registers_dict)    
    program.update()

def rename_register(program, old_reg, new_reg):
    """ Replacing a register old_reg with a register new_reg in a program 
        
        Args:
            program (Program): target program for register renaming
            old_reg (Register): An original name of the register 
            new_reg (Register): A new name for the register 
    """
    print("[REG_RNAME] Renaming %s to %s" % (old_reg, new_reg))
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
    
    is_2d_thread_block = False
    
    tid_reg = None
    tid_inst = 0    
    # If this is the first time a register is spilled to shared memory,
    # add a new parameter to the program object for keeping track of register 
    # spilling and add instruction to compute the base address for spilled 
    # register
    # ASSUME: 1D thread block
    # TODO: Add support for 2d and 3d thread block
    if not getattr(program, 'shared_spill_count', False):
        tid_inst_copy = None
        
        # Add new parameter
        setattr(program, 'shared_spill_count', 0)
                
        # Check if the kernel uses 2D thread block
        for inst in program.ast:
            if not isinstance(inst, Instruction):
                continue
            if inst.opcode.name == 'S2R':
                # Look for S2R R19, SR_TID.X instruction.
                if inst.operands[0].name == 'SR_TID' and 'Y' in inst.operands[0].extension:
                    is_2d_thread_block = True
                    break
        
        if is_2d_thread_block:
            tid_x_inst = Instruction(Flags('--','-','1','-','6'), 
                Opcode('S2R'), 
                operands=[Register('R2'), Register('SR_TID.X', is_special=True)])
            
            tid_y_inst = Instruction(Flags('--','-','2','-','6'), 
                Opcode('S2R'), 
                operands=[Register('R3'), Register('SR_TID.Y', is_special=True)])
            
            block_dim_x_inst = Instruction(Flags('--','-','-','-','6'), 
                Opcode('MOV'), 
                operands=[Register('R4'), 'blockDimX'])
            
            tid_mad_inst = Instruction(Flags('03','-','-','-','6'), 
                Opcode('XMAD'), 
                operands=[Register('R2'), Register('R3'), Register('R4'), Register('R2')])
            
            base_addr_inst = Instruction(Flags('--','-','-','-','6'), 
                Opcode('SHL'), 
                operands=[spill_register_addr, Register('R2'), 0x02])

            program.ast.insert(1, tid_x_inst)
            program.ast.insert(2, tid_y_inst)
            program.ast.insert(3, block_dim_x_inst)
            program.ast.insert(4, tid_mad_inst)
            program.ast.insert(5, base_addr_inst)
        else:
            tid_x_inst = Instruction(Flags('--','-','1','-','6'), 
                Opcode('S2R'), 
                operands=[Register('R2'), Register('SR_TID.X', is_special=True)])
            
            base_addr_inst = Instruction(Flags('01','-','-','-','6'), 
                Opcode('SHL'), 
                operands=[spill_register_addr, Register('R2'), 0x02])

            program.ast.insert(1, tid_x_inst)
            program.ast.insert(2, base_addr_inst)
    
    # Assign the spilled register Identifier
    spill_reg_id = program.shared_spill_count
    spill_offset = spill_reg_id * thread_block_size * 4 + program.shared_size
    program.shared_spill_count += 1
    
    print("[REG_SPILL] Replacing %s with shared memory %s[%s+%d] (TB:%d Offset:%d)" % (target_register,spill_register, spill_register_addr, spill_offset, thread_block_size,program.shared_size ))
    
    load_shr_inst = SpillLoadInstruction(Flags('--','1','2','-','6'), 
                        Opcode('LDS'),
                        operands=[spill_register, Pointer(spill_register_addr, spill_offset),
                        ])
    
    # Add read barrier to make sure that the STS instruction is completed before next instreuction
    store_shr_inst = SpillStoreInstruction(Flags('--','4','-','-','6'), 
                        Opcode('STS'),
                        operands=[Pointer(spill_register_addr, spill_offset), spill_register, 
                        ])
    
    w_count = 0
    r_count = 0
    
    barrier_tracker = BarrierTracker()
    
    for inst in program.ast:
        # If instruction contain spilled_register, rename the spilled_register to spill_register.
        # If the instruction read data from the spilled_register, load data in the shared memory 
        # to spill_register just before the instruction. 
        # If the instruction write data to the spilled_register, store data from spill_register 
        # to shared memory right after the instruction.
        if isinstance(inst, Label):
            barrier_tracker.reset()
            continue
        
        inst_idx = program.ast.index(inst)
        prev_idx = inst_idx-1
        prev_inst = None
        
        if prev_idx > 0:
            prev_inst = program.ast[prev_idx]
            while not isinstance(prev_inst, Instruction):
                prev_idx = prev_idx-1
                prev_inst = program.ast[prev_idx]
            
        for op in inst.operands:
            if isinstance(op, Pointer):
                if target_register == op.register:
                    r_count = r_count + 1
                    # Load data from shared memory
                    op.register.rename(spill_register)
                    # Set the instruction that read still register to wait for shared memory read 
                    shared_load_inst = copy.deepcopy(load_shr_inst)
                    shared_load_inst.flags.read_barrier = barrier_tracker.get_available_flags('r')
                    shared_load_inst.flags.write_barrier = barrier_tracker.get_available_flags('w')
                    inst.flags.wait_barrier = inst.flags.wait_barrier | (1 << (shared_load_inst.flags.read_barrier-1) | 1 << (shared_load_inst.flags.write_barrier-1))
                    if isinstance(prev_inst, SpillStoreInstruction):
                         shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8
                    shared_load_inst.condition = inst.condition              
                    program.ast.insert(program.ast.index(inst), shared_load_inst)
                    
            elif isinstance(op, Register):
                if target_register == op:
                    r_count = r_count + 1
                    # Read access
                    inst.operands[inst.operands.index(op)].rename(spill_register)
                    # Set the instruction that read spill register to wait for shared memory read
                    shared_load_inst = copy.deepcopy(load_shr_inst)
                    shared_load_inst.flags.read_barrier = barrier_tracker.get_available_flags('r')
                    shared_load_inst.flags.write_barrier = barrier_tracker.get_available_flags('w')
                    inst.flags.wait_barrier = inst.flags.wait_barrier | (1 << (shared_load_inst.flags.read_barrier-1) | 1 << (shared_load_inst.flags.write_barrier-1))
                    if isinstance(prev_inst, SpillStoreInstruction):
                         shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8
                    shared_load_inst.condition = inst.condition 
                    program.ast.insert(program.ast.index(inst), shared_load_inst)
            elif isinstance(op, Constant) and op.pointer:
                if target_register == op.pointer.register:
                    r_count = r_count + 1
                    # Load data from shared memory
                    op.pointer.register.rename(spill_register)
                    # Set the instruction that read still register to wait for shared memory read 
                    shared_load_inst = copy.deepcopy(load_shr_inst)
                    shared_load_inst.flags.read_barrier = barrier_tracker.get_available_flags('r')
                    shared_load_inst.flags.write_barrier = barrier_tracker.get_available_flags('w')
                    inst.flags.wait_barrier = inst.flags.wait_barrier | (1 << (shared_load_inst.flags.read_barrier-1) | 1 << (shared_load_inst.flags.write_barrier-1))
                    if isinstance(prev_inst, SpillStoreInstruction):
                         shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8 
                    shared_load_inst.condition = inst.condition
                    program.ast.insert(program.ast.index(inst), shared_load_inst)
        
        if isinstance(inst.dest, Pointer):
            if target_register == inst.dest.register:
                # Read access
                r_count = r_count + 1
                inst.dest.register.rename(spill_register)
                # Set the instruction that read spill register to wait for shared memory read
                shared_load_inst = copy.deepcopy(load_shr_inst)
                shared_load_inst.flags.read_barrier = barrier_tracker.get_available_flags('r')
                shared_load_inst.flags.write_barrier = barrier_tracker.get_available_flags('w')
                inst.flags.wait_barrier = inst.flags.wait_barrier | (1 << (shared_load_inst.flags.read_barrier-1) | 1 << (shared_load_inst.flags.write_barrier-1))
                if isinstance(prev_inst, SpillStoreInstruction):
                        shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8
                shared_load_inst.condition = inst.condition                    
                program.ast.insert(program.ast.index(inst), shared_load_inst)
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
                    if inst.flags.read_barrier != 0:
                        st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.read_barrier-1))
                    inst.flags.stall = inst.flags.stall + 1
                    inst.flags.yield_hint = False
                else:
                    if(inst.opcode.grammar.type == 'gmem' or inst.opcode.grammar.type == 'smem'):
                        inst.flags.write_barrier = barrier_tracker.get_available_flags('w')
                        st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                    elif (inst.opcode.name in ['RRO', 'MUFU']):
                        if inst.flags.write_barrier == 0:
                            inst.flags.write_barrier = barrier_tracker.get_available_flags('w')
                        st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                    if inst.flags.stall < 15:
                        inst.flags.stall += 1
                        inst.flags.stall = 13 
                    inst.flags.yield_hint = False
                st_inst.flags.read_barrier = barrier_tracker.get_available_flags('r')
                st_inst.condition = inst.condition
                program.ast.insert(program.ast.index(inst) + 1, st_inst)
                # Set wait flag of the next instruction to wait for store instruction to finish
                inst_next = program.ast[program.ast.index(st_inst) + 1] 
                if isinstance(inst_next, Instruction):
                    inst_next.flags.wait_barrier = inst_next.flags.wait_barrier | (1 << (st_inst.flags.read_barrier -1))
                    if inst_next.flags.stall == 0:
                        inst_next.flags.stall = 6
        
        barrier_tracker.update_flags(inst.flags)
                                   
    print("[REG_SPILL] Read accesses: %d Write accesses: %d" % (r_count, w_count))   
    program.update()
    


def spill_64bit_register_to_shared(
        program, 
        target_register64, 
        cfg=None,
        spill_register=Register('R0'),
        spill_register2=Register('R1'), 
        spill_register_addr=Register('R2'), 
        thread_block_size=256, ):
    """ Spill registers to shared memory 
        
        Args:
            program (Program): target program for register spilling
            spilled_register (Register): A register to be spilled  
            cfg (Cfg): A CFG of the input program
            thread_block_size (int): Size of thread block  
    """
    # TODO: Find free barrier for LDS Instruction
    print("[REG_SPILL] Replacing %s with shared memory %s[%s] and %s[%s] (TB:%d)" % (target_register64,spill_register, spill_register_addr,spill_register2, spill_register_addr, thread_block_size))
    
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
        base_addr_inst = Instruction(Flags(hex(wait_flag),'-','-','-','6'), 
            Opcode('SHL'), 
            operands=[spill_register_addr, tid_reg, 0x02])
        program.ast.insert(program.ast.index(tid_inst) + 1, base_addr_inst)
    
    # Assign the spilled register Identifier
    target_register = Register(target_register64[0])
    target_register2 = Register(target_register64[1])
    spill_reg_id = program.shared_spill_count
    spill_reg2_id = program.shared_spill_count + 1
    spill_offset = spill_reg_id * thread_block_size * 4 + program.shared_size
    spill_offset2 = spill_reg2_id * thread_block_size * 4 + program.shared_size
    
    program.shared_spill_count += 2
        
    load_shr_inst = SpillLoadInstruction(Flags('--','1','2','-','6'), 
                        Opcode('LDS'),
                        operands=[spill_register, Pointer(spill_register_addr, spill_offset),
                        ])
    load_shr_inst2 = SpillLoadInstruction(Flags('--','1','2','-','6'), 
                        Opcode('LDS'),
                        operands=[spill_register2, Pointer(spill_register_addr, spill_offset2),
                        ])
    
    # Add read barrier to make sure that the STS instruction is completed before next instreuction
    store_shr_inst = SpillStoreInstruction(Flags('--','4','-','-','6'), 
                        Opcode('STS'),
                        operands=[Pointer(spill_register_addr, spill_offset), spill_register, 
                        ])
    store_shr_inst2 = SpillStoreInstruction(Flags('--','4','-','-','6'), 
                        Opcode('STS'),
                        operands=[Pointer(spill_register_addr, spill_offset2), spill_register2, 
                        ])
    
    w_count = 0
    r_count = 0
    
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        # If instruction contain spilled_register, rename the spilled_register to spill_register.
        # If the instruction read data from the spilled_register, load data in the shared memory 
        # to spill_register just before the instruction. 
        # If the instruction write data to the spilled_register, store data from spill_register 
        # to shared memory right after the instruction.
        inst_idx = program.ast.index(inst)
        prev_idx = inst_idx-1
        prev_inst = None
        
        if prev_idx > 0:
            prev_inst = program.ast[prev_idx]
            while not isinstance(prev_inst, Instruction):
                prev_idx = prev_idx-1
                prev_inst = program.ast[prev_idx]
        
        for op in inst.operands:
            if isinstance(op, Pointer):
                if target_register == op.register or target_register2 == op.register:
                    r_count = r_count + 1
                    # Load data from shared memory
                    if target_register == op.register:
                        op.register.rename(spill_register)
                    else:
                        op.register.rename(spill_register2)
                    # Set the instruction that read still register to wait for shared memory read 
                    inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3
                    
                    shared_load_inst = copy.deepcopy(load_shr_inst)
                    if isinstance(prev_inst, SpillStoreInstruction):
                         shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8 
                    program.ast.insert(program.ast.index(inst), shared_load_inst)
                    shared_load_inst = copy.deepcopy(load_shr_inst2)
                    if isinstance(prev_inst, SpillStoreInstruction):
                         shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8 
                    program.ast.insert(program.ast.index(inst), shared_load_inst)
                    
            elif isinstance(op, Register):
                if target_register == op or target_register2 == op:
                    r_count = r_count + 1
                    # Read access
                    if target_register == op:
                        inst.operands[inst.operands.index(op)].rename(spill_register)
                    else:
                        inst.operands[inst.operands.index(op)].rename(spill_register2)
                    
                    # Set the instruction that read spill register to wait for shared memory read
                    inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3
                    shared_load_inst = copy.deepcopy(load_shr_inst)
                    if isinstance(prev_inst, SpillStoreInstruction):
                         shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8 
                    program.ast.insert(program.ast.index(inst), shared_load_inst)
                    shared_load_inst = copy.deepcopy(load_shr_inst2)
                    if isinstance(prev_inst, SpillStoreInstruction):
                         shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8 
                    program.ast.insert(program.ast.index(inst), shared_load_inst)
        
        if isinstance(inst.dest, Pointer):
            if target_register == inst.dest.register or target_register2 == inst.dest.register:
                # Read access
                r_count = r_count + 1
                if target_register == inst.dest.register:
                    inst.dest.register.rename(spill_register)
                else:
                    inst.dest.register.rename(spill_register2)
                
                # Set the instruction that read spill register to wait for shared memory read
                inst.flags.wait_barrier = inst.flags.wait_barrier | 0x3
                shared_load_inst = copy.deepcopy(load_shr_inst)
                if isinstance(prev_inst, SpillStoreInstruction):
                    shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8 
                program.ast.insert(program.ast.index(inst), shared_load_inst)
                shared_load_inst = copy.deepcopy(load_shr_inst2)
                if isinstance(prev_inst, SpillStoreInstruction):
                    shared_load_inst.flags.wait_barrier = shared_load_inst.flags.wait_barrier | 0x8 
                program.ast.insert(program.ast.index(inst), shared_load_inst)
        elif isinstance(inst.dest, Register):
            if target_register == inst.dest or target_register2 == inst.dest:
                # Write access
                w_count = w_count + 1
                if target_register == inst.dest:
                    st_inst = copy.deepcopy(store_shr_inst)
                    st_inst2 = copy.deepcopy(store_shr_inst2)
                    inst.dest.rename(spill_register)
                    
                    # Set wait flag if the previous instruction sets write dependence flag. 
                    # This will happen if the instruction store data in the spilled register.
                    # Add 1 additional cycle to the store instruction.
                    if inst.flags.read_barrier != 0:
                        st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.read_barrier-1)) 
                    if inst.flags.write_barrier != 0:
                        st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                        if inst.flags.stall < 15: 
                            inst.flags.stall = inst.flags.stall + 1
                    else:
                        if(inst.opcode.grammar.type == 'gmem' or inst.opcode.grammar.type == 'smem'):
                            inst.flags.write_barrier = 6
                            st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                        elif (inst.opcode.name in ['RRO', 'MUFU']):
                            if inst.flags.write_barrier == 0:
                                inst.flags.write_barrier = 6
                            st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                        inst.flags.stall = 13 
                        inst.flags.yield_hint = False
                        
                    st_inst2.flags = copy.deepcopy(st_inst.flags)
                    program.ast.insert(program.ast.index(inst) + 1, st_inst)
                    if inst.opcode.op_bit > 32:
                        program.ast.insert(program.ast.index(inst) + 1, st_inst2)
                    # Set wait flag of the next instruction to wait for store instruction to finish
                    inst_next = program.ast[program.ast.index(st_inst) + 1] 
                    if isinstance(inst_next, Instruction):
                        inst_next.flags.wait_barrier = inst_next.flags.wait_barrier | 8
                else:
                    st_inst = copy.deepcopy(store_shr_inst2)
                    inst.dest.rename(spill_register2)
                    
                    # Set wait flag if the previous instruction sets write dependence flag. 
                    # This will happen if the instruction store data in the spilled register.
                    # Add 1 additional cycle to the store instruction. 
                    if inst.flags.write_barrier != 0:
                        st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                        if inst.flags.stall < 15: 
                            inst.flags.stall = inst.flags.stall + 1
                    else:
                        if(inst.opcode.grammar.type == 'gmem' or inst.opcode.grammar.type == 'smem'):
                            inst.flags.write_barrier = 6
                            st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                        elif (inst.opcode.name in ['RRO', 'MUFU']):
                            if inst.flags.write_barrier == 0:
                                inst.flags.write_barrier = 6
                            st_inst.flags.wait_barrier = st_inst.flags.wait_barrier |( 1 << (inst.flags.write_barrier-1))
                        inst.flags.stall = 13 
                        inst.flags.yield_hint = False
                    program.ast.insert(program.ast.index(inst) + 1, st_inst)
                    # Set wait flag of the next instruction to wait for store instruction to finish
                    inst_next = program.ast[program.ast.index(st_inst) + 1] 
                    if isinstance(inst_next, Instruction):
                        inst_next.flags.wait_barrier = inst_next.flags.wait_barrier | 8
            
                  
    print("[REG_SPILL] Read accesses: %d Write accesses: %d" % (r_count, w_count))   
    program.update()

def spill_local_memory(program, thread_block_size):
    print("[LOCAL_SPILL]")
    local_base_reg = None
    local_offset_list = []
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        # Collect register with local memory base address
        if inst.opcode.name == "MOV" and inst.operands[0].name == "c[0x0][0x20]":
            local_base_reg = inst.dest

        # Collect all local load/store instructions
        if inst.opcode.name == "LDL":
            local_mem_ptr = inst.operands[0]
            if local_mem_ptr.offset not in local_offset_list:
                local_offset_list.append(local_mem_ptr.offset)

        if inst.opcode.name == "STL":
            local_mem_ptr = inst.operands[0]
            if local_mem_ptr.offset not in local_offset_list:
                local_offset_list.append(local_mem_ptr.offset)
                            
    print("[LOCAL_SPILL] Local Memory Base Address Register: %s" % local_base_reg)
    print("[LOCAL_SPILL] Local Memory Base Address Offsets: %s" % local_offset_list)
    print("[LOCAL_SPILL] Shared Memory Location Needed: %d" % len(local_offset_list))
    print("[LOCAL_SPILL] Threadblock Size: %d" % thread_block_size)

    if len(local_offset_list) > 0:

        is_2d_thread_block = False
        spill_register_addr = Register('R55')

        # Check if the kernel uses 2D thread block
        for inst in program.ast:
            if not isinstance(inst, Instruction):
                continue
            if inst.opcode.name == 'S2R':
                # Look for S2R R19, SR_TID.X instruction.
                if inst.operands[0].name == 'SR_TID' and 'Y' in inst.operands[0].extension:
                    is_2d_thread_block = True
                    break
        
        # Compute shared memory base address
        if is_2d_thread_block:
            tid_x_inst = Instruction(Flags('--','-','1','-','6'), 
                Opcode('S2R'), 
                operands=[Register('R2'), Register('SR_TID.X', is_special=True)])
            
            tid_y_inst = Instruction(Flags('--','-','2','-','6'), 
                Opcode('S2R'), 
                operands=[Register('R3'), Register('SR_TID.Y', is_special=True)])
            
            block_dim_x_inst = Instruction(Flags('--','-','-','-','6'), 
                Opcode('MOV'), 
                operands=[Register('R4'), 'blockDimX'])
            
            tid_mad_inst = Instruction(Flags('03','-','-','-','6'), 
                Opcode('XMAD'), 
                operands=[Register('R2'), Register('R3'), Register('R4'), Register('R2')])
            
            base_addr_inst = Instruction(Flags('--','-','-','-','6'), 
                Opcode('SHL'), 
                operands=[spill_register_addr, Register('R2'), 0x02])

            program.ast.insert(1, tid_x_inst)
            program.ast.insert(2, tid_y_inst)
            program.ast.insert(3, block_dim_x_inst)
            program.ast.insert(4, tid_mad_inst)
            program.ast.insert(5, base_addr_inst)
        else:
            tid_x_inst = Instruction(Flags('--','-','1','-','6'), 
                Opcode('S2R'), 
                operands=[Register('R2'), Register('SR_TID.X', is_special=True)])
            
            base_addr_inst = Instruction(Flags('01','-','-','-','6'), 
                Opcode('SHL'), 
                operands=[spill_register_addr, Register('R2'), 0x02])

            program.ast.insert(1, tid_x_inst)
            program.ast.insert(2, base_addr_inst)

        
        local_shared_dict = dict.fromkeys(local_offset_list)
        spill_offset = program.shared_size
        spill_count = 0

        pprint(sorted(local_offset_list, key= lambda x: int(str(x), 16)))
        for local in sorted(local_offset_list, key= lambda x: int(str(x), 16)):
            local_shared_dict[local] = spill_count * thread_block_size * 4 + program.shared_size
            spill_count += 1

        pprint(local_shared_dict)

        # Convert local load/store to shared load/store
        for inst in [x for x in program.ast if isinstance(x, Instruction)]:
            # Convert local load to shared load
            if inst.opcode.name == "LDL":
                if local_shared_dict[inst.operands[0].offset] != None:
                    inst.opcode.name = "LDS"
                    inst.opcode.extension = []
                    inst.flags.yield_hint = False
                    inst.flags.stall = 6
                    if inst.flags.read_barrier == 0:
                        inst.flags.read_barrier = 1
                    if inst.flags.write_barrier == 0:
                        inst.flags.write_barrier = 6
                    inst_next = program.ast[program.ast.index(inst)+1]
                    inst_next.flags.wait_barrier = inst_next.flags.wait_barrier | (1 << (inst.flags.read_barrier -1))
                    inst_next.flags.wait_barrier = inst_next.flags.wait_barrier | (1 << (inst.flags.write_barrier -1))
                    inst.operands[0].register = spill_register_addr
                    inst.operands[0].offset = local_shared_dict[inst.operands[0].offset]

            if inst.opcode.name == "STL":
                if local_shared_dict[inst.operands[0].offset] != None:
                    inst.opcode.name = "STS"
                    inst.opcode.extension = []
                    inst.flags.yield_hint = False
                    inst.flags.stall = 6
                    if inst.flags.read_barrier == 0:
                        inst.flags.read_barrier = 1
                    inst_next = program.ast[program.ast.index(inst)+1]
                    inst_next.flags.wait_barrier = inst_next.flags.wait_barrier | (1 << (inst.flags.read_barrier -1))
                    inst.operands[0].register = spill_register_addr
                    inst.operands[0].offset = local_shared_dict[inst.operands[0].offset]
        
        program.update()