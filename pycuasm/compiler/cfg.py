from pprint import pformat, pprint
from pycuasm.compiler.hir import *

import json

JUMP_OPS = ['BRA', 'PBK', 'PCNT', 'SYNC']
CALL_OPS = ['CAL', 'JCAL'] 

class RegisterAccess(object):
    def __init__(self, first_read=-1, first_write=-1, next_write=-1, last_read=-1, read_count=0):
        self.first_read = first_read
        self.first_write = first_write
        self.next_write = next_write
        self.last_read = last_read
        self.read_count = read_count
        
    def __repr__(self):
        return "(FW:%s, FR:%s, NW:%s, LR:%s, CT:%s)" % (self.first_write, 
            self.first_read, self.next_write, self.last_read, self.read_count) 

class Block(object):
    def __init__(self, taken=None, not_taken=None):
        self.__taken = taken
        self.__not_taken = not_taken
        self.__pred = []
        self.condition = None
        
        self.var_use = set()
        self.var_def = set()
        self.live_in = set()
        self.live_out = set()
        
    @property    
    def taken(self):
        return self.__taken
    
    @property
    def not_taken(self):
        return self.__not_taken
        
    @property
    def pred(self):
        return self.__pred
        
    def get_dot_node(self):
        return self.__repr__()
        
    def connect_taken(self, block):
        # Branch taken path
        self.__taken = block
        block.__pred.append(self)
    
    def connect_not_taken(self, block):
        # Branch not taken path
        self.__not_taken = block
        block.__pred.append(self)
        
class StartBlock(Block):
    def __repr__(self):
        return "<Start>"

class EndBlock(Block):
    def __repr__(self):
        return "<End>"

class FunctionBlock(Block):
    def __repr__(self):
        return "<Function Call>"
    
    def update(self):
        pass


        
class ReturnBlock(Block):
    def __repr__(self):
        return "<Return>"

class BasicBlock(Block):
    def __init__(self, instructions, label=None, taken=None, not_taken=None):
        super().__init__(taken, not_taken)
    
        if isinstance(instructions, list):
            self.instructions = instructions
        elif isinstance(instructions, Instruction):
            self.instructions = [instructions]
        else:
            raise ValueError("Invalid parameter")
        
        self.label = label
        self.condition = self.instructions[-1].condition
        self.sync_point = None
        
        registers = []
        pointers = []
        
        for inst in self.instructions:
            if isinstance(inst, Instruction):
                regs = [x for x in inst.operands + [inst.dest] if isinstance(x, Register) and not x.is_special]
                regs += [x.register for x in inst.operands + [inst.dest] if isinstance(x, Pointer)]
                registers += [x for x in regs if x not in registers]
                
                ptrs = [x for x in inst.operands + [inst.dest] if isinstance(x, Pointer)]
                pointers += [x for x in ptrs if x not in pointers]
                
                if inst.opcode.name == 'SSY':
                    self.sync_point = inst.operands[0].name
            else:
                raise ValueError("Invalid IR Type: %s %s" % (inst.__class__, inst))

        self.registers = registers
        self.pointer_accesses = pointers
        self.__analyze_use_def()
        self.__analyze_reg_access()
        
    def __repr__(self):
        return "%s:%s" % (  
            self.instructions[0].opcode.name,
            self.instructions[-1].opcode.name,
        )
    
    def __analyze_reg_access(self):
        reg_access_map = dict.fromkeys(self.registers)
        
        for reg in reg_access_map:
            reg_access_map[reg] = [RegisterAccess()]
                     
        for inst in self.instructions:
            for operand in inst.operands:
                op = operand
                if isinstance(op, Pointer):
                    op = op.register
                if not isinstance(op, Register) or op.is_special:
                    continue
                
                if reg_access_map[op][-1].first_read < 0:
                    reg_access_map[op][-1].first_read = inst.addr 
                reg_access_map[op][-1].last_read = inst.addr
                reg_access_map[op][-1].read_count += 1
                
            if inst.opcode.reg_store and isinstance(inst.dest, Register):
                if len(reg_access_map[inst.dest]) > 0:
                    reg_access_map[inst.dest][-1].next_write = inst.addr
                reg_access_map[inst.dest].append(RegisterAccess(first_write = inst.addr))
                     
        setattr(self, 'reg_usage', reg_access_map)
    
    def __analyze_use_def(self):
        reg_read_map = dict.fromkeys(self.registers)
        for k in reg_read_map:
            reg_read_map[k] = []
        
        reg_write_map = dict.fromkeys(self.registers)
        for k in reg_write_map:
            reg_write_map[k] = []
        
        for inst in self.instructions:
            for operand in inst.operands:
                op = operand
                if isinstance(op, Pointer):
                    op = op.register
                if not isinstance(op, Register) or op.is_special:
                    continue
                
                reg_read_map[op].append(inst.addr)
                
            if inst.opcode.reg_store and isinstance(inst.dest, Register):
                reg_write_map[inst.dest].append(inst.addr)
        
        for reg in reg_read_map.keys():
            # First read is before first write
            if reg_read_map[reg] and reg_write_map[reg] and reg_read_map[reg][0] <= reg_write_map[reg][0]:
                self.var_use.add(reg)
                
            if reg_read_map[reg] and len(reg_write_map[reg]) == 0:
                self.var_use.add(reg)
            
            if reg_write_map[reg] and len(reg_write_map[reg]) > 0:
                self.var_def.add(reg)
    
    def get_dot_node(self):
        repr = ''
        for inst in self.instructions:
            inst_str = str(inst)
            inst_str = inst_str.replace("'{'", ' ').replace("'}'", ' ')
            repr += inst_str + "\l"
        return repr
        
    def add_instruction(self, inst):
        self.instructions.append(inst)
        
    def attach_label(self, label):
        self.label = label
    
class CallBlock(BasicBlock):
    def update(self):
        pass

class Cfg(object):
    """ Control flow graph
    """
    def __init__(self, program=None):
        """
            Args:
                program: A program object generated from parser
        """
        self.__blocks = []
        self.__return_block = None
        self.__functions = {}
        if program:
            self.update(program)
    
    def __repr__(self):
        repr = ""
        for block in self.__blocks:
            repr += "%s\n\t%s\n\t%s\n" % (block, block.taken, block.not_taken)
            repr += "\tPred:%s\n" % block.pred
        return repr    
    
    @property
    def blocks(self):
        return self.__blocks
    
    def add_basic_block(self, block):
        self.__blocks.append(block)
    
    def add_function(self, name, block):
        self.__functions[name] = block
    
    def create_dot_graph(self, outfile):
        """ Generating dot file representing the CFG
            
            Args:
                outfile: Output file name
        """
        nodes = ""
        for block in self.__blocks:
            node = "block%d " % self.__blocks.index(block)
            if isinstance(block, BasicBlock):
                param = '<label> %s|' % (block.label if block.label else hex(block.instructions[0].addr))
                param += 'USE: %s|' % (list(block.var_use) if block.var_use else "[]")
                if block.sync_point:
                    param += 'SYNC Point: %s|' % block.sync_point
                param += "{%s}" % block.get_dot_node()
                param += '| DEF: %s' % (list(block.var_def) if block.var_def else "[]")
                if block.condition:
                    param += "|<branch> %s" % block.instructions[-1].opcode
                
                node += '[shape=record, label="{%s}", %s]' % (param, "" if not isinstance(block, CallBlock) else "color=red");
            else:
                node += '[labeljust=l, shape=rectangle, label="%s"]' % str(block)
            nodes += node + ";\n"
        
        for block in self.__blocks:
            if block.taken:
                if block.condition:
                    nodes += 'block%s:branch -> block%s%s [label="taken", headport="ne", tailport="se"];\n' % (
                        self.__blocks.index(block),
                        self.__blocks.index(block.taken),
                        ":label" if block.taken.label else "")
                else:
                    nodes += 'block%s -> block%s;\n' % (self.__blocks.index(block), self.__blocks.index(block.taken))
            if block.not_taken:
                if block.condition:
                    nodes += 'block%s:branch -> block%s%s [label="not taken"];\n' % (
                        self.__blocks.index(block),
                        self.__blocks.index(block.not_taken), 
                        ":label" if block.not_taken.label else "")
                        
                else:
                    nodes += 'block%s -> block%s;\n' % (self.__blocks.index(block), self.__blocks.index(block.not_taken))
                
        dot = "digraph cfg{labeljust=l; %s }" % nodes
        
        print("Writing CFG to %s" % outfile)
        f = open(outfile, 'w')
        f.write(dot)
        f.close()
        
    def update(self, program):
        """ Building CFG for a program
            
            Args:
                program: An input Program object
        """
        if not isinstance(program, Program):
            raise TypeError("Expect Program but got %s instead" % (program.__class__))
                
        print("Creating CFG")
        
        # Find the beginning of basic blocks. A basic block begin at the start
        # of a program, after a label, or a new predicate is found. 
        call_targets = []
        leader = []
        read_leader = True
        
        for inst in program.ast:
            if isinstance(inst, Instruction) and read_leader:
                # Mark the instruction as the beginning of a new basic block 
                leader.append(inst)
                prev_condition = inst.condition
                if inst.opcode.name in JUMP_OPS:
                    read_leader = True
                elif inst.opcode.name in CALL_OPS:
                    read_leader = True
                    call_targets.append(inst.operands[0].name)
                else:
                    read_leader = False               
            elif isinstance(inst, Instruction) and not read_leader:             
                if inst.opcode.name in JUMP_OPS:
                    read_leader = True
                if inst.opcode.name in CALL_OPS:
                    leader.append(inst)
                    read_leader = True
                    call_targets.append(inst.operands[0].name)
                    
            elif isinstance(inst, Label):
                read_leader = True
        
        call_targets = list(set(call_targets))
        
        # Construct CFG basic blocks
        label_table = {} 
        sync_point = None
        
        self.add_basic_block(StartBlock())
        for lead_inst in leader:
            next_leader = leader.index(lead_inst)+1 
            
            ast_idx = program.ast.index(lead_inst)
            
            if next_leader < len(leader):
                ast_idx_next = program.ast.index(leader[next_leader])
            else:
                ast_idx_next = len(program.ast)

            if isinstance(program.ast[ast_idx_next -1], Label):
                ast_idx_next -= 1
            
            # Create a basic block containing instructions between the current 
            # leader and the next leader
            block = None
            if(len(program.ast[ast_idx:ast_idx_next]) == 1) and program.ast[ast_idx].opcode.name in CALL_OPS:
                block = CallBlock(program.ast[ast_idx:ast_idx_next],)
            else:
                block = BasicBlock(program.ast[ast_idx:ast_idx_next],)        
            self.add_basic_block(block)
                        
            if ast_idx > 0 and isinstance(program.ast[ast_idx-1], Label):
                label = program.ast[ast_idx-1]
                block.attach_label(label)
                label_table[label.name] = block
                if label.name == sync_point:
                    sync_point = None
            
            if block.sync_point:
                sync_point = block.sync_point
            else:
                block.sync_point = sync_point
            
        self.__return_block = []
                
        self.__end_block = EndBlock()
        self.add_basic_block(self.__end_block)
                    
        self.__blocks[0].connect_taken(self.__blocks[1]) 
        # Connect blocks in CFG
        for block in self.__blocks[1:-1]:
            if isinstance(block, EndBlock) or isinstance(block, ReturnBlock):
                continue
            idx = self.__blocks.index(block)
            last_inst = block.instructions[-1]
            
            if block.label and block.label.name in call_targets:
                call_block = FunctionBlock()
                call_block.connect_taken(block)
                self.add_basic_block(call_block)
                self.add_function(block.label.name, call_block)
            
            if last_inst.opcode.name not in JUMP_OPS and idx < len(self.__blocks)-1:
                if last_inst.opcode.name == 'EXIT':
                    block.connect_taken(self.__end_block)
                elif last_inst.opcode.name == 'RET':
                    ret_block = ReturnBlock()
                    self.__return_block.append(ret_block)
                    block.connect_taken(ret_block)
                else:
                    block.connect_taken(self.__blocks[idx+1])
            elif last_inst.opcode.name in JUMP_OPS:
                if block.condition:
                    if block.condition.condition:
                        if last_inst.opcode.name == 'SYNC':
                            block.connect_taken(label_table[block.sync_point])
                        else:
                            block.connect_taken(label_table[last_inst.operands[0].name]) 
                        block.connect_not_taken(self.__blocks[idx+1] if idx < len(self.__blocks)-1 else None)
                    else:
                        if last_inst.opcode.name == 'SYNC':
                            block.connect_not_taken(label_table[block.sync_point])
                        else:
                            block.connect_not_taken(label_table[last_inst.operands[0].name]) 
                        block.connect_taken(self.__blocks[idx+1] if idx < len(self.__blocks)-1 else None)
                else:
                    if last_inst.opcode.name == 'SYNC':
                        block.connect_taken(label_table[block.sync_point])
                    else:
                        block.connect_taken(label_table[last_inst.operands[0].name])
            for block in self.__return_block:
                self.add_basic_block(block)
                
    def analyze_liveness(self):
        """ Perform liveness analysis
        """
        
        end_blocks = self.__return_block + [self.__end_block]
        
        for block in end_blocks:
            # Generate a list of BasicBlock in reverse order
            traversed_block = [block]
            sorted_blocks = [block] 
            while len(traversed_block) > 0:
                curBlock = traversed_block.pop()
                # Tag node as visited
                setattr(curBlock, 'visited',True)
                for pred in curBlock.pred:
                    if not getattr(pred, 'visited', False):
                        traversed_block.append(pred)
                        sorted_blocks.append(pred)
            
            # Clean up visited tag
            for node in self.__blocks:
                if not isinstance(node, StartBlock) and getattr(node, 'visited', None):
                    delattr(node, 'visited')        
            
            # Compute live in and out
            converge = False
            for node in sorted_blocks:
                node.live_in = set()
                node.live_out = set()
            
            if isinstance(block, ReturnBlock):
                for node in sorted_blocks:
                    block.var_use |= node.var_def           

            while not converge:
                for node in sorted_blocks:
                    setattr(node, 'old_live_in', node.live_in.copy())
                    setattr(node, 'old_live_out', node.live_out.copy())
                    node.live_out = (node.taken.live_in if node.taken else set()) | \
                                    (node.not_taken.live_in if node.not_taken else set()) 
                    node.live_in = node.var_use | (node.live_out - node.var_def)
                converge = True
                for node in sorted_blocks:
                    if node.old_live_in != node.live_in:
                        converge = False    