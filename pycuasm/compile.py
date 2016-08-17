import json
import itertools
import os
import subprocess
import re
import math

from operator import itemgetter

from pprint import pprint
from pycuasm.compiler.lexer import sass_lexer
from pycuasm.compiler.parser import sass_parser 
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *
from pycuasm.compiler.analysis import *
from pycuasm.compiler.transform import * 

from pycuasm.tool import *

class Sass():
    def __init__(self, sass_file):
        self.header = []
        print('Opening ' +  sass_file)
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
    
    program.save('out.sass')
    return

    print("Register usage: %s" % program.registers)
    cfg = Cfg(program)
    cfg.create_dot_graph("cfg.dot")
    
    #cfd_register_sweep(program, size=2)
    
    reg_candidates = generate_spill_candidates(program, exclude_registers=['R0','R1'])
    interference_dict = analyse_register_interference(program, reg_candidates)
    access_dict = analyse_register_accesses(program, reg_candidates)
    pprint(reg_candidates)

    spilled_count = 0
    spilled_target = 14

    while spilled_count < spilled_target:
        spilled_reg = reg_candidates.pop(0)
        spill_register_to_shared(
            program, 
            Register(spilled_reg), 
            spill_register = Register('R68'),
            spill_register_addr = Register('R69'),
            thread_block_size=192)
        
        for interference_reg in interference_dict[spilled_reg]:
            if interference_reg in reg_candidates:
                print("Remove: ", interference_reg)
                reg_candidates.remove(interference_reg)
        
        reg_candidates = sorted(reg_candidates, key=lambda x: access_dict[x]['read'] +  access_dict[x]['write'])
        spilled_count = spilled_count + 1

    relocate_registers(program)
    
def test_lexer(sass):
    sass_lexer.input(sass.sass_raw)
    while True:
        tok = sass_lexer.token()
        if not tok: 
            break      # No more input
        print(tok.type + " ", end="")
        if tok.type == ';' or tok.type == 'LABEL':
            print()
                
        
        
    
    
    
