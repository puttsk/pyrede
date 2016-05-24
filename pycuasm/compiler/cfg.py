from pprint import pformat

from pycuasm.compiler.hir import *

class Cfg():
    def __init__(self):
        self.blocks = []
    
    def __repr__(self):
        return pformat(self.blocks)
    
    def add_basic_block(self, block):
        self.blocks.append(block)    
    
class BasicBlock():
    def __init__(self, instructions, label=None, taken=None, not_taken=None):
        if isinstance(instructions, list):
            self.instructions = instructions
        elif isinstance(instructions, Instruction):
            self.instructions = [instructions]
        else:
            raise ValueError("Invalid parameter")
    
        self.label = label
        self.taken = taken
        self.not_taken = not_taken
        self.condition = self.instructions[-1].condition
        
    def __repr__(self):
        return ("<" + self.label.name + "> " if self.label else "") + \
            self.instructions[0].opcode.name + ":" + \
            (str(self.condition) + " " if self.condition else "") + \
            self.instructions[-1].opcode.name + " "
    
    def add_instruction(self, inst):
        self.instructions.append(inst)
    
    def connect_taken(self, block):
        # Branch taken path
        self.taken = block
    
    def connect_not_taken(self, block):
        # Branch not taken path
        self.not_taken = block
        
    def attach_label(self, label):
        self.label = label