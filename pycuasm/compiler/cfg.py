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
    def __init__(self, instructions, label=None, left=None, right=None, predicate=None):
        if isinstance(instructions, list):
            self.instructions = instructions
        elif isinstance(instructions, Instruction):
            self.instructions = [instructions]
        else:
            raise ValueError("Invalid parameter")
    
        self.label = label
        self.left = left
        self.right = right
        self.predicate = predicate
        
    def __repr__(self):
        return ("<" + self.label.name + "> " if self.label else "") + \
            self.instructions[0].opcode.name
    
    def add_instruction(self, inst):
        self.instructions.append(inst)
    
    def connect_left(self, block):
        self.left = block
    
    def connect_right(self, block):
        self.right = block
        
    def attach_label(self, label):
        self.label = label