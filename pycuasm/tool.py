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

def cfd_register_sweep(program, size=1, output_file='result.csv'):
    print("Sweeping registers for spilling")

    last_reg = sorted(program.registers, key=lambda x: int(x.replace('R','')), reverse=True)[0]
    last_reg_id = int(last_reg.replace('R',''))
        
    #rename_register(program, Register('R0'), Register('R%d' % (last_reg_id+1)))
    #rename_register(program, Register('R1'), Register('R%d' % (last_reg_id+2)))

    reg_64 = collect_64bit_registers(program)
    reg_mem =  collect_global_memory_access(program)
    
    reg_remove = list(itertools.chain(*reg_64.intersection(reg_mem)))
    reg_candidates = sorted([ x for x in program.registers if x not in reg_remove], key=lambda x: int(x.replace('R','')))

    '''
    for inst in [x for x in program.ast if isinstance(x, Instruction)]:
        if inst.flags.yield_hint:
            pprint(inst)
            for reg in [x for x in inst.operands if isinstance(x, Register)]:
                if reg.name in reg_candidates:
                    reg_candidates.remove(reg.name)
            if inst.dest and inst.dest.name in reg_candidates:
                reg_candidates.remove(inst.dest.name)
    '''
    
    print("Spilled Register Candidate: ")
    
    reg_candidates.remove('R0')
    reg_candidates.remove('R1')
        
    pprint(reg_candidates)
    
    #return
    
    count = 1
    reg_combination = itertools.combinations(reg_candidates, size)
    resule_file = open(output_file, 'w')    
    for r in reg_candidates:
        resule_file.write(pformat(r) + ',')
    resule_file.write('time, error code, error message\n')
    
    for reg_list in reg_combination:
        rename_dict = {}
        program_tmp = copy.deepcopy(program)
        program_tmp.update()
        for reg in reg_list:
            spill_register_to_shared(
                program_tmp, 
                Register(reg), 
                spill_register = Register('R%d' % (last_reg_id+1)),
                spill_register_addr = Register('R%d' % (last_reg_id+2)),
                thread_block_size=192)
        
        reg_remain = sorted(program_tmp.registers, key=lambda x:int(x.replace('R', '')), reverse=True)
        reg_list = list(sorted(reg_list, key=lambda x:int(x.replace('R', ''))))
        
        reg_idx = 0
        for reg in reg_remain:
            if reg_idx < size and int(reg.replace('R','')) > int(reg_list[reg_idx].replace('R','')):
                    rename_dict[reg] = reg_list[reg_idx]
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
        
        for r in reg_candidates:
            if r in reg_list:
                resule_file.write('Y,')
            else:
                resule_file.write('%s,' % rename_dict.get(r, 'N'))
        try:
            result = subprocess.check_output(['nvprof', './euler3d', '../../fvcorr.domn.097K'], stderr=subprocess.STDOUT, universal_newlines=True)
            match = re.search(r"(?P<time>\d+\.\d+)ms.*cuda_compute_flux", result)
            
            resule_file.write(match.group('time'))
            
            run_output = open('density', 'r')
            run_output.readline()
            
            try:
                val = float(run_output.readline())
                if not math.isnan(val):    
                    resule_file.write(',0, %f\n' % val)
                else:
                    resule_file.write(',-1, %f\n' % val)
            except ValueError:
                resule_file.write(',-2,\n')
            
            run_output.close()
            
        except subprocess.CalledProcessError as err:
            match = re.search(r"getLastCudaError.*\((?P<err_code>\d+)\)(?P<msg>.+)", err.output)
            if not match:
                match = re.search(r"CUDA error at.*code=(?P<err_code>\d+)\((?P<msg>.+)\)", err.output)
            
            resule_file.write(',0,')
            resule_file.write(match.group('err_code')+',')
            resule_file.write(match.group('msg')+'\n')
        except:
            resule_file.write(',-2\n')
        
        os.chdir(base_dir)
        
        count = count+1
    
    resule_file.close()