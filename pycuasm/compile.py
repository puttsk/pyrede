import json

from pprint import pprint
from pycuasm.compiler.lexer import sass_lexer
from pycuasm.compiler.parser import sass_parser 
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *

class Sass():
    def __init__(self, sass_file):
        self.header = []
        with open(sass_file) as sass_file:
            for line in sass_file:
                if line[0] == '#':
                    self.header.append(line)
                else:
                    break    
            line = sass_file.readline()
            
            if line != "<CONSTANT_MAPPING>\n":
                raise ValueError("Invalid SASS File")
            
            json_str = ""
            for line in sass_file:
                if line != "</CONSTANT_MAPPING>\n":
                    json_str += line
                else:
                    break
            
            self.constants = json.loads(json_str)
            self.sass_raw = "".join(sass_file.readlines()).strip()    
            
def compile(args):
    sass = Sass(args.input_file)
    program = sass_parser.parse(sass.sass_raw, lexer=sass_lexer)
   
    print("Register usage: %s" % program.registers)
    pprint(program.ast)
    
    reg_live_map = dict.fromkeys(program.registers)
    for k in reg_live_map:
        reg_live_map[k] = []
        
    reg_scratch_map = dict.fromkeys(program.registers)    
            
    cfg = Cfg(program)
    #print(cfg)
    
    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
            
        for inst in block.instructions:
            for operand in inst.operands:
                op = operand
                if isinstance(op, Pointer):
                    op = op.register
                if not isinstance(op, Register) or op.is_special:
                    continue
                
                reg_scratch_map[op][1] = inst.addr
                
            if inst.opcode.reg_store and isinstance(inst.dest, Register):
                if reg_scratch_map[inst.dest]:
                    reg_live_map[inst.dest].append(tuple(reg_scratch_map[inst.dest]))
                reg_scratch_map[inst.dest] = [inst.addr, -1]
                 
    for k in reg_scratch_map:
        if reg_scratch_map[k] not in reg_live_map[k]:
            reg_live_map[k].append(tuple(reg_scratch_map[k]))
                
    pprint(reg_live_map)
    
            
def test_lexer(sass):
    sass_lexer.input(sass.sass_raw)
    while True:
        tok = sass_lexer.token()
        if not tok: 
            break      # No more input
        print(tok.type + " ", end="")
        if tok.type == ';' or tok.type == 'LABEL':
            print()
                
        
        
    
    