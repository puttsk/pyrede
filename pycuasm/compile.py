import json
import itertools
import os
import subprocess

from operator import itemgetter

from pprint import pprint
from pycuasm.compiler.lexer import sass_lexer
from pycuasm.compiler.parser import sass_parser 
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *
from pycuasm.compiler.analysis import *
from pycuasm.compiler.transform import * 

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
    #pprint(program.ast)
                
    rename_register(program, Register('R0'), Register('R68'))
    rename_register(program, Register('R1'), Register('R69'))
     
    cfg = Cfg(program)
    cfg.create_dot_graph("cfg.dot")
    
    reg_64 = collect_64bit_registers(program)
    reg_mem =  collect_global_memory_access(program)
    
    
    reg_remove = list(itertools.chain(*reg_64.intersection(reg_mem)))
    reg_candidates = sorted([ x for x in program.registers if x not in reg_remove])
    
    pprint(reg_candidates)
    '''
    count = 1
    size = 1
    reg_combination = itertools.combinations(reg_candidates, size)
    
    for reg_list in reg_combination:
        program_tmp = copy.deepcopy(program)
        program_tmp.update()
        for reg in reg_list:
            spill_register_to_shared(program_tmp, Register(reg), cfg, 192)
        
        reg_remain = sorted(program_tmp.registers, key=lambda x:int(x.replace('R', '')), reverse=True)
        reg_list = list(sorted(reg_list, key=lambda x:int(x.replace('R', ''))))
        
        reg_idx = 0
        for reg in reg_remain:
            if reg_idx < size and int(reg.replace('R','')) > int(reg_list[reg_idx].replace('R','')):
                    rename_register(program_tmp, Register(reg), Register(reg_list[reg_idx]))
                    reg_idx = reg_idx + 1

        #pprint(sorted(program_tmp.registers, key=lambda x:int(x.replace('R', ''))))
        
        directory = 'output/' + str(count) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        program_tmp.save(directory + 'out.sass')
        f = open(directory + 'conf', 'w')
        f.write(pformat(reg_list))
        f.close()
        
        os.system("cp Makefile_sub " + directory + "Makefile")
        os.system("cp euler3d.cu " + directory)
        os.system("cp make_sub.sh " + directory)
        
        base_dir = os.getcwd()
        os.chdir(directory)
        os.system("sh make_sub.sh > make.log 2>&1")
        os.chdir(base_dir)
        
        count = count+1
    '''          
    '''
    spill_register_to_shared(program, Register('R15'), cfg, 192)
    spill_register_to_shared(program, Register('R22'), cfg, 192)
    spill_register_to_shared(program, Register('R23'), cfg, 192)
    spill_register_to_shared(program, Register('R35'), cfg, 192)
    spill_register_to_shared(program, Register('R37'), cfg, 192)
    spill_register_to_shared(program, Register('R38'), cfg, 192)
    spill_register_to_shared(program, Register('R39'), cfg, 192)
    spill_register_to_shared(program, Register('R66'), cfg, 192)
     
    spill_register_to_shared(program, Register('R67'), cfg, 192)
    spill_register_to_shared(program, Register('R62'), cfg, 192)
    
    #TODO: Incorrect result
    #spill_register_to_shared(program, Register('R36'), cfg, 192)
    #spill_register_to_shared(program, Register('R64'), cfg, 192)
    #TODO: Incorrect result
    #spill_register_to_shared(program, Register('R32'), cfg, 192)
    #TODO: Incorrect result
    #spill_register_to_shared(program, Register('R34'), cfg, 192)
    #spill_register_to_shared(program, Register('R65'), cfg, 192)
    #TODO: Incorrect result
    #spill_register_to_shared(program, Register('R33'), cfg, 192)
    
    
    #pprint(sorted([int(x.name.replace('R','')) for x in program.registers]))
    
    rename_register(program, Register('R68'), Register('R15'))
    rename_register(program, Register('R69'), Register('R22'))
    #rename_register(program, Register('R67'), Register('R37'))
    #rename_register(program, Register('R66'), Register('R38'))
    rename_register(program, Register('R65'), Register('R23'))
    rename_register(program, Register('R64'), Register('R35'))
    rename_register(program, Register('R63'), Register('R37'))
    #rename_register(program, Register('R62'), Register('R38'))
    rename_register(program, Register('R61'), Register('R38'))
    rename_register(program, Register('R60'), Register('R39'))
    '''
    #relocate_registers(program)
    
    #pprint(sorted([int(x.name.replace('R','')) for x in program.registers]))
                   
                
def test_lexer(sass):
    sass_lexer.input(sass.sass_raw)
    while True:
        tok = sass_lexer.token()
        if not tok: 
            break      # No more input
        print(tok.type + " ", end="")
        if tok.type == ';' or tok.type == 'LABEL':
            print()
                
        
        
    
    
    
