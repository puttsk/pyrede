import argparse
import subprocess
import sys

from pycuasm import *
from pycuasm.cubin import Cubin

def main():
    parser = argparse.ArgumentParser(description='Python CUDA SASS Assembler')
    parser.add_argument('-l','--list', action='store_true' ,help="List kernels and symbols in the cubin file")
    parser.add_argument('-e','--extract', action='store_true', help="Extract a single kernel into an asm file from a cubin. Works much like cuobjdump but outputs in a format that can be re-assembled back into the cubin.")
    parser.add_argument('-k','--kernel', help="Specify kernel name for extract operation.")
    parser.add_argument('-o','--output', help="Specify output assembly file name.")
    parser.add_argument('--cuobjdump', help="Specify an input cuobjdump file. For debugging purpose only when cuobjdume does not exist in the system.")
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()

    #List kernels and symbol inside the cubin file
    if args.list:
        Cubin(args.input_file).print_info()
    elif args.extract:
        extract(args)        
    else:
        compile(args)

if __name__ == "__main__":
    # execute only if run as a script
    main()
