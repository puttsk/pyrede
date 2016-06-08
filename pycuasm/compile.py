import json
from operator import itemgetter

from pprint import pprint
from pycuasm.compiler.lexer import sass_lexer
from pycuasm.compiler.parser import sass_parser 
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *
from pycuasm.compiler.transform import spill_register_to_shared, rename_register

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
    program.constants = sass.constants
    program.header = sass.header
    
    print("Register usage: %s" % program.registers)
    pprint(program.ast)
                
    rename_register(program, Register('R0'), Register('R6'))
    rename_register(program, Register('R1'), Register('R7'))
                
    cfg = Cfg(program)

    reg_usage_map = dict.fromkeys(program.registers)
    for reg in reg_usage_map:
        reg_usage_map[reg] = []

    for block in cfg.blocks:
        if not isinstance(block, BasicBlock):
            continue
        for reg in block.reg_usage:
            reg_usage_map[reg] += block.reg_usage[reg]
    
    shared_candidates = list(sorted(reg_usage_map, key=lambda k: len(reg_usage_map[k])))
    
    pprint(reg_usage_map)
    
    spill_register_to_shared(program, Register('R4'), cfg, 256)
    spill_register_to_shared(program, Register('R6'), cfg, 256)
    
    program.save('new.sass')
                                                  
    cfg.create_dot_graph("cfg.dot")
                
def test_lexer(sass):
    sass_lexer.input(sass.sass_raw)
    while True:
        tok = sass_lexer.token()
        if not tok: 
            break      # No more input
        print(tok.type + " ", end="")
        if tok.type == ';' or tok.type == 'LABEL':
            print()
                
        
        
    
    