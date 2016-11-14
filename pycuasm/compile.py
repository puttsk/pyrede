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
from pycuasm.compiler.optimization import *  

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
    
    cfg = Cfg(program)
    
    if args.spill_register:
        print("[REG_SPILL] Spilling %d registers to shared memory. Threadblock Size: %d" % (args.spill_register, args.thread_block_size))
        exclude_registers = ['R0', 'R1']
        
        if args.exclude_registers:
            exclude_registers.append(args.exclude_registers)
        
        reg_candidates = generate_spill_candidates_cfg(program, cfg, exclude_registers=exclude_registers)
        pprint(reg_candidates)
        #reg_candidates = generate_spill_candidates(program, exclude_registers=exclude_registers)
        #pprint(reg_candidates)
        skipped_candidates = []
        interference_dict = analyse_register_interference(program, reg_candidates)
        access_dict = analyse_register_accesses(program, reg_candidates)        
        
        cfg.create_dot_graph("cfg.dot")
        
        last_reg = sorted(program.registers, key=lambda x: int(x.replace('R','')), reverse=True)[0]
        last_reg_id = int(last_reg.replace('R',''))

        spilled_count = 0
        spilled_target = args.spill_register
        
        if (last_reg_id + 1) % 2 != 1:
            last_reg_id = last_reg_id + 1

        spill_register_id = last_reg_id + 2
        spill_register_64_id = last_reg_id + 3
        spill_register_addr_id = last_reg_id + 1

        while len(reg_candidates) > 0 and spilled_count < spilled_target:
            spilled_reg = reg_candidates.pop(0)
            spill_register_to_shared(
                program, 
                Register(spilled_reg), 
                spill_register = Register('R%d' % (spill_register_id)),
                spill_register_addr = Register('R%d' % (spill_register_addr_id)),
                thread_block_size=args.thread_block_size)
            
            for interference_reg in interference_dict[spilled_reg]:
                if interference_reg in reg_candidates:
                    print("[REG_SPILL] Remove candidate: ", interference_reg)
                    skipped_candidates.append(interference_reg)
                    reg_candidates.remove(interference_reg)
            
            reg_candidates = sorted(reg_candidates, key=lambda x: access_dict[x]['read'] +  access_dict[x]['write'])
            spilled_count = spilled_count + 1        
        
        if spilled_target - spilled_count > 0:
            print("[REG_SPILL] Spilling 64-bit registers to shared memory.")
             
            reg_candidates = generate_64bit_spill_candidates(program)
            reg_candidates_first_reg = [x[0] for x in reg_candidates]
            
            interference_dict = analyse_register_interference(program, reg_candidates_first_reg)
            access_dict = analyse_register_accesses(program, reg_candidates_first_reg)
            
            spilled_target = spilled_target + 1
            
            while len(reg_candidates) > 0 and spilled_count < spilled_target:
                spilled_64bit_reg = reg_candidates.pop(0)
                spill_64bit_register_to_shared(
                    program, 
                    spilled_64bit_reg, 
                    spill_register = Register('R%d' % (spill_register_id)),
                    spill_register2 = Register('R%d' % (spill_register_64_id)),
                    spill_register_addr = Register('R%d' % (spill_register_addr_id)),
                    thread_block_size=args.thread_block_size)
                                
                for interference_reg in interference_dict[spilled_64bit_reg[0]]:
                    interference_reg_2 = 'R%d' % (int(interference_reg.replace('R','')) + 1)
                    interference_reg_64 = (interference_reg, interference_reg_2)
                    if interference_reg_64 in reg_candidates:
                        print("[REG_SPILL] Remove candidate: ", interference_reg_64)
                        skipped_candidates.append(interference_reg_64)
                        reg_candidates.remove(interference_reg_64)
                
                reg_candidates = sorted(reg_candidates, key=lambda x: access_dict[x[0]]['read'] +  access_dict[x[0]]['write'])
                spilled_count = spilled_count + 2 
        
        print("[REG_SPILL] Spilled %d registers to shared memory." % (spilled_count))
        
        remove_redundant_spill_instruction(program, Register("R%d" % spill_register_addr_id))
        rearrange_spill_instruction(program, Register("R%d" % spill_register_id) ,Register("R%d" % spill_register_addr_id)) 
        
    if not args.no_register_relocation:
        #relocate_registers(program)
        relocate_registers_new(program)
        
    program.save(args.output)
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
                
        
        
    
    
    
