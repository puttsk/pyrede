#!/usr/bin/python3

import argparse
import pprint
import sys

from pycuasm import *
from pycuasm.cubin import Cubin

def main():
    parser = argparse.ArgumentParser(description='Python CUDA SASS Assembler')
    # Compiler operation
    parser.add_argument('-l','--list', action='store_true', default=False ,help="List kernels and symbols in the cubin file")
    parser.add_argument('-e','--extract', action='store_true', default=False, help="Extract a single kernel into an asm file from a cubin. Works much like cuobjdump but outputs in a format that can be re-assembled back into the cubin.")
    parser.add_argument('-c','--compiler', action='store_true', default=True, help="Compiler and optimize input SASS file. (default)")
    parser.add_argument('--tuning', action='store_true', default=False, help="Analyse the benefit of register demotion")
    parser.add_argument('-k','--kernel', help="Specify kernel name for extract operation.")
    parser.add_argument('-o','--output', help="Specify output assembly file name.", default="out.sass")
    
    # Register spilling
    parser.add_argument('-r','--spill-register', help="Spill a specific number of registers to shared memory", type=int)
    parser.add_argument('--exclude-registers', help="Exclude specific registers from spilling candidate", default=None)
    parser.add_argument('-t','--thread-block-size', help="Number of threads in thread block", type=int, default=256)
    
    # Compiler optimization
    parser.add_argument('-O','--opt-level',type=int, help="Specify optimization level", default=1)
    parser.add_argument('--use-local-spill', action='store_true', help="Convert local spill to shared spill", default=False)
    parser.add_argument('--no-register-relocation', action='store_true', default=False, help="Disable register relocation after spilling")
    parser.add_argument('--avoid-conflict', type=int, default=2, help="0: Disable / 1:Enable register conflict avoidance")
    parser.add_argument('--swap-spill-reg', type=int, default=2, help="0: Disable / 1:Enable spill register swapping")
    parser.add_argument('--opt-access', type=int, default=2, help="0: Disable / 1:Enable spill register swapping")
    parser.add_argument('--candidate_type', type=int, default=0, help="0: CFG / 1:Static Access / 2: Static Conflict")
    
    # Debugging
    parser.add_argument('--cuobjdump', help="Specify an input cuobjdump file. For debugging purpose only when cuobjdume does not exist in the system.")
    
    # tuning
    parser.add_argument('--local-sass', type=str, help="SASS code with local spilling")
    parser.add_argument('--local-sass-shared', type=str, help="SASS code with local spilling to shared")
    
    # Default argument
    parser.add_argument('input_file', type=str)
    
    args = parser.parse_args()

    #List kernels and symbol inside the cubin file
    if args.list:
        Cubin(args.input_file).print_info()
    elif args.extract:
        print("Extracting " + args.input_file)
        extract(args)
    elif args.tuning:
        tuning(args)
    else:
        compile(args)

if __name__ == "__main__":
    # execute only if run as a script
    main()

