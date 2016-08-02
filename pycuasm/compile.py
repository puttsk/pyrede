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
    cfg = Cfg(program)
    cfg.create_dot_graph("cfg.dot")
    
    cfd_register_sweep(program, size=2)
    
    '''
    spill_register_to_shared(
        program, 
        Register('R33'), 
        spill_register = Register('R68'),
        spill_register_addr = Register('R69'),
        thread_block_size=192)
        
    spill_register_to_shared(
        program, 
        Register('R34'), 
        spill_register = Register('R68'),
        spill_register_addr = Register('R69'),
        thread_block_size=192)
    '''
    
    program.save('out.sass')

def test_lexer(sass):
    sass_lexer.input(sass.sass_raw)
    while True:
        tok = sass_lexer.token()
        if not tok: 
            break      # No more input
        print(tok.type + " ", end="")
        if tok.type == ';' or tok.type == 'LABEL':
            print()
                
        
        
    
    
    
