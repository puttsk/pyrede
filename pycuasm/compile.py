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
    program.set_constants(sass.constants)
    program.set_header(sass.header)
    
    print("Register usage: %s" % sorted(program.registers, key=lambda x: int(x.replace('R',''))))
    #cfg = Cfg(program)
    #cfg.create_dot_graph("cfg.dot")
    
    #cfd_register_sweep(program, size=2)
    
    if args.spill_register:
        print("[RES_SPILL] Spilling %d registers to shared memory. Threadblock Size: %d" % (args.spill_register, args.thread_block_size))
        reg_candidates = generate_spill_candidates(program, exclude_registers=['R0','R1'])
        interference_dict = analyse_register_interference(program, reg_candidates)
        access_dict = analyse_register_accesses(program, reg_candidates)
        pprint(reg_candidates)

        last_reg = sorted(program.registers, key=lambda x: int(x.replace('R','')), reverse=True)[0]
        last_reg_id = int(last_reg.replace('R',''))

        spilled_count = 0
        spilled_target = args.spill_register

        while len(reg_candidates) > 0 and spilled_count < spilled_target:
            spilled_reg = reg_candidates.pop(0)
            spill_register_to_shared(
                program, 
                Register(spilled_reg), 
                spill_register = Register('R%d' % (last_reg_id+1)),
                spill_register_addr = Register('R%d' % (last_reg_id+2)),
                thread_block_size=args.thread_block_size)
            
            for interference_reg in interference_dict[spilled_reg]:
                if interference_reg in reg_candidates:
                    print("Remove: ", interference_reg)
                    reg_candidates.remove(interference_reg)
            
            reg_candidates = sorted(reg_candidates, key=lambda x: access_dict[x]['read'] +  access_dict[x]['write'])
            spilled_count = spilled_count + 1

        relocate_registers(program)
    '''
    last_reg = sorted(program.registers, key=lambda x: int(x.replace('R','')), reverse=True)[0]
    last_reg_id = int(last_reg.replace('R',''))
    
    spill_register_to_shared(
            program, 
            Register('R11'), 
            spill_register = Register('R%d' % (last_reg_id+1)),
            spill_register_addr = Register('R%d' % (last_reg_id+2)),
            thread_block_size=args.thread_block_size)
    '''
    #pprint(generate_spill_candidates(program, exclude_registers=['R0','R1']))
    program.save('out.sass')
    
    #myocyte_register_sweep(program, size=1)
    return
    
def test_lexer(sass):
    sass_lexer.input(sass.sass_raw)
    while True:
        tok = sass_lexer.token()
        if not tok: 
            break      # No more input
        print(tok.type + " ", end="")
        if tok.type == ';' or tok.type == 'LABEL':
            print()
                
        
        
    
    
    
