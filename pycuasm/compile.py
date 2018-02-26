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
from pycuasm.compiler.sass import Sass 
from pycuasm.compiler.hir import *
from pycuasm.compiler.cfg import *
from pycuasm.compiler.analysis import *
from pycuasm.compiler.transform import *
from pycuasm.compiler.optimization import *  

from pycuasm.tool import *

def compile_program(program, args):
    # Check argument
    opt_level = args.opt_level
    
    # Default level = 1
    opt_conflict_avoidance = True
    opt_remove_redundant_inst = True
    opt_swap_spill_reg = True
    opt_hoist_spill_inst = False
        
    if opt_level == 0:
        opt_conflict_avoidance = False
        opt_remove_redundant_inst = False
        opt_swap_spill_reg = False
        opt_hoist_spill_inst = False
        
    if opt_level > 0:
        opt_conflict_avoidance = True
        opt_remove_redundant_inst = True
        opt_swap_spill_reg = True
        if opt_level > 1:
            opt_hoist_spill_inst = True
    
    # Overriding optimization level
    if args.avoid_conflict == 0: 
        opt_conflict_avoidance = False 
    
    if args.avoid_conflict == 1:
        opt_conflict_avoidance = True
    
    if args.swap_spill_reg == 0: 
        opt_swap_spill_reg = False 
    
    if args.swap_spill_reg == 1:
        opt_swap_spill_reg = True
        
    if args.opt_access == 0:
        opt_hoist_spill_inst = False
        opt_remove_redundant_inst = False
    
    if args.opt_access == 1:
        opt_hoist_spill_inst = True
        opt_remove_redundant_inst = True
    
    register_relocation = True
    if args.no_register_relocation:
        register_relocation = False
    
    print("[COMPILE]: Candidate Type: ", args.candidate_type)
    print("[COMPILE]: Optimization level: ", opt_level)
    print("[COMPILE]: Register conflict avoidance: ", opt_conflict_avoidance)
    print("[COMPILE]: Remove redundant spill instruction: ", opt_remove_redundant_inst)
    print("[COMPILE]: Swap spill register: ", opt_swap_spill_reg)
    print("[COMPILE]: Hoist spill instruction: ", opt_hoist_spill_inst)
    print("[COMPILE]: Register relocation: ", register_relocation)
    print("[COMPILE]: Register usage: %s" % sorted(program.registers, key=lambda x: int(x.replace('R',''))))
    
    cfg = Cfg(program)
    
    if args.spill_register:
        print("[REG_SPILL] Spilling %d registers to shared memory. Threadblock Size: %d" % (args.spill_register, args.thread_block_size))
        exclude_registers = ['R0', 'R1']
        #exclude_registers = []
        
        if args.exclude_registers:
            exclude_registers.append(args.exclude_registers)
        
        reg_candidates = []
        if args.candidate_type == 0:
            reg_candidates = generate_spill_candidates_cfg(program, cfg, exclude_registers=exclude_registers)
        elif args.candidate_type == 1:
            reg_candidates = generate_spill_candidates(program, exclude_registers=exclude_registers, priority='access')
        elif args.candidate_type == 2:
            reg_candidates = generate_spill_candidates(program, exclude_registers=exclude_registers, priority='conflict')
        pprint(reg_candidates)
        skipped_candidates = []
        interference_dict = analyse_register_interference(program, reg_candidates)
        access_dict = analyse_register_accesses(program, reg_candidates)        
        
        last_reg = sorted(program.registers, key=lambda x: int(x.replace('R','')), reverse=True)[0]
        last_reg_id = int(last_reg.replace('R',''))

        spilled_count = 0
        spilled_target = args.spill_register
        
        if (last_reg_id + 1) % 2 != 1:
            last_reg_id = last_reg_id + 1

        spill_register_id = last_reg_id + 2
        spill_register_64_id = last_reg_id + 3
        spill_register_addr_id = last_reg_id + 1

        spill_bank_list = [[],[],[],[]]

        while len(reg_candidates) > 0 and spilled_count < spilled_target:
            spilled_reg = Register(reg_candidates.pop(0))
            spill_register_to_shared(
                program, 
                spilled_reg, 
                spill_register = Register('R%d' % (spill_register_id)),
                spill_register_addr = Register('R%d' % (spill_register_addr_id)),
                thread_block_size=args.thread_block_size)
            
            spill_bank_list[spilled_reg.id % 4].append(spilled_reg)
            
            for interference_reg in interference_dict[spilled_reg]:
                if interference_reg in reg_candidates:
                    print("[REG_SPILL] Remove candidate: ", interference_reg)
                    skipped_candidates.append(interference_reg)
                    reg_candidates.remove(interference_reg)
            
            reg_candidates = sorted(reg_candidates, key=lambda x: access_dict[x]['read'] +  access_dict[x]['write'])
            spilled_count = spilled_count + 1        
        pprint(skipped_candidates)
        if opt_conflict_avoidance:
            max_spill_bank = 0
            max_spill_count = 0
            for i in range(4):
                if len(spill_bank_list[i]) > max_spill_count:
                    max_spill_bank = i
                    max_spill_count = len(spill_bank_list[i])
                
            if spill_register_id % 4 != max_spill_bank:
                new_reg_id = int(math.ceil(spill_register_id / 4))*4 + max_spill_bank
                rename_register(program, Register('R%d' % (spill_register_id)), Register('R%d' % (new_reg_id)))
                print("[REG_SPILL] Change spill regiter R%d to R%d" % (spill_register_id, new_reg_id))
            
        
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
            program.update()
            
        print("[REG_SPILL] Spilled %d registers to shared memory." % (spilled_count))
        
        if opt_remove_redundant_inst:
            opt_remove_redundant_spill_inst(program, Register("R%d" % spill_register_addr_id))
        
        if opt_swap_spill_reg:
            opt_swap_spill_register(program, avoid_conflict=opt_conflict_avoidance)
            
        if opt_hoist_spill_inst:
            opt_hoist_spill_instruction(program)
        
    if args.use_local_spill:
        spill_local_memory(program, args.thread_block_size)
    
    if register_relocation:
        if opt_conflict_avoidance:
            relocate_registers_conflict(program)
        else:
            relocate_registers(program)
            
def compile(args):
    sass = Sass(args.input_file)
    program = sass_parser.parse(sass.sass_raw, lexer=sass_lexer)
    program.set_constants(sass.constants)
    program.set_header(sass.header)
    
    compile_program(program, args)
            
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
                
        
        
    
    
    
