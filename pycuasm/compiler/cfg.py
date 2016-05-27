from pprint import pformat, pprint
from pycuasm.compiler.hir import *

REL_OFFSETS = ['BRA', 'SSY', 'CAL', 'PBK', 'PCNT']
ABS_OFFSETS = ['JCAL']
JUMP_OPS = REL_OFFSETS + ABS_OFFSETS

class Block(object):
    def __init__(self, taken=None, not_taken=None):
        self.__taken = taken
        self.__not_taken = not_taken
        self.__pred = []
        self.condition = None
        
    @property    
    def taken(self):
        return self.__taken
    
    @property
    def not_taken(self):
        return self.__not_taken
        
    @property
    def pred(self):
        return self.__pred
        
    def get_node(self):
        return self.__repr__()
        
    def connect_taken(self, block):
        # Branch taken path
        self.__taken = block
        block.__pred.append(self)
    
    def connect_not_taken(self, block):
        # Branch not taken path
        self.__not_taken = block
        block.__pred.append(self)

class Cfg(object):
    def __init__(self, program=None):
        self.blocks = []
        if program:
            self.create_cfg(program)    
    
    def __repr__(self):
        repr = ""
        for block in self.blocks:
            repr += "%s\n\t%s\n\t%s\n" % (block, block.taken, block.not_taken)
            repr += "\tPred:%s\n" % block.pred
        return repr
    
    def create_dot_graph(self, outfile="path.dot"):
        nodes = ""
        for block in self.blocks:
            node = "block%d " % self.blocks.index(block)
            if isinstance(block, BasicBlock):
                if not block.condition and not block.label:
                    node += '[shape=rectangle, labeljust=l, label="%s"]' % block.get_node()
                else:
                    param = '<label> %s|' % (block.label if block.label else "")    
                    param += "{%s}" % block.get_node()
                    if block.condition:
                        param += "|<branch> %s" % block.instructions[-1]
                    
                    node += '[shape=record, label="{%s}"]' % param;
            else:
                node += '[labeljust=l, shape=rectangle, label="%s"]' % str(block)
            nodes += node + ";\n"
        
        for block in self.blocks:
            if block.taken:
                if block.condition:
                    nodes += 'block%s:branch -> block%s:label [label="taken", headport="ne", tailport="se"];\n' % (self.blocks.index(block), self.blocks.index(block.taken))
                else:
                    nodes += 'block%s -> block%s;\n' % (self.blocks.index(block), self.blocks.index(block.taken))
            if block.not_taken:
                if block.condition:
                    nodes += 'block%s:branch -> block%s:label [label="not taken"];\n' % (self.blocks.index(block), self.blocks.index(block.not_taken))
                else:
                    nodes += 'block%s -> block%s;\n' % (self.blocks.index(block), self.blocks.index(block.not_taken))
                
        dot = "digraph cfg{ %s }" % nodes
        f = open(outfile, 'w')
        f.write(dot)
        f.close()
    
       
    def add_basic_block(self, block):
        self.blocks.append(block)
        
    def create_cfg(self, program):
        # Build CFG
        print("Creating CFG")
        
        # Find the beginning of basic blocks. A basic block begin at the start
        # of a program, after a label, or a new predicate is found. 
        leader = []
        read_leader = True
        for inst in program.ast:
            if isinstance(inst, Instruction) and read_leader:
                # Mark the instruction as the beginning of a new basic block 
                leader.append(inst)
                prev_condition = inst.condition
                read_leader = False
                
            if isinstance(inst, Instruction) and not read_leader:             
                if inst.opcode.name in JUMP_OPS:
                    read_leader = True
                    
            if isinstance(inst, Label):
                read_leader = True
        
        # Construct CFG basic blocks
        label_table = {} 

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
            block = BasicBlock(program.ast[ast_idx:ast_idx_next],)        
            self.add_basic_block(block)
            
            if ast_idx > 0 and isinstance(program.ast[ast_idx-1], Label):
                label = program.ast[ast_idx-1]
                block.attach_label(label)
                label_table[label.name] = block         
        self.add_basic_block(EndBlock())
            
        pprint(self.blocks)
            
        self.blocks[0].connect_taken(self.blocks[1]) 
        # Connect blocks in CFG
        for block in self.blocks[1:-1]:
            idx = self.blocks.index(block)
            last_inst = block.instructions[-1]
            
            if last_inst.opcode.name not in JUMP_OPS and idx < len(self.blocks)-1:
                block.connect_taken(self.blocks[idx+1])
            elif last_inst.opcode.name in JUMP_OPS:
                if block.condition:
                    if block.condition.condition:
                        block.connect_taken(label_table[last_inst.operands[0]]) 
                        block.connect_not_taken(self.blocks[idx+1] if idx < len(self.blocks)-1 else None)
                    else:
                        block.connect_not_taken(label_table[last_inst.operands[0]]) 
                        block.connect_taken(self.blocks[idx+1] if idx < len(self.blocks)-1 else None)
                else:
                    block.connect_taken(label_table[last_inst.operands[0]])
        self.create_dot_graph()
        
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
        
        registers = []
        
        for inst in self.instructions:
            if isinstance(inst, Instruction):
                regs = [x for x in inst.operands + [inst.dest] if isinstance(x, Register) and not x.is_special]
                registers += [x for x in regs if x not in registers]
            else:
                raise ValueError("Invalid IR Type: %s %s" % (inst.__class__, inst))

        self.registers = registers
        self.read_access = None
        self.write_access = None
        
        self.generate_access_map()
        
    def get_node(self):
        repr = ''
        for inst in self.instructions:
            repr += str(inst) + "\l"
        return repr
        
    def __repr__(self):
        return "%s %s:%s %s" % ( 
            "<" + self.label.name + "> " if self.label else "", 
            self.instructions[0].opcode.name,
            self.condition if self.condition else "",
            self.instructions[-1].opcode.name,
            #pformat(self.read_access), 
            #pformat(self.write_access),
            #self.pred
        )
    
    
    
    def add_instruction(self, inst):
        self.instructions.append(inst)
        
    def attach_label(self, label):
        self.label = label
    
    def generate_access_map(self):
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
        
        self.read_access = reg_read_map
        self.write_access = reg_write_map

class StartBlock(Block):
    def __repr__(self):
        return "<Start>"

class EndBlock(Block):
    def __repr__(self):
        return "<End>"