import argparse
import subprocess
import sys

from pyCuAsm.pycuasm import *
from pyCuAsm.cubin import Cubin

def printCubinInfo(cubin_file, cubin):
    print("%s: \n\tarch: sm_%d \n\tmachine: %d bit \n\taddress_size: %d bit\n" % (cubin_file, cubin.arch, cubin.addressSize, cubin.addressSize))
    for kernel in cubin.kernels:
        printKernelInfo(cubin.kernels[kernel])
    for symbol in cubin.symbols:
        print("Symbol: " + symbol + "\n")

def printKernelInfo(kernel):
    print("Kernel: " + kernel['Name'])
    print("\tLinkage: %s \n\tParams: %d \n\tSize: %d \n\tRegisters: %d \n\tSharedMem: %d \n\tBarriers: %d" % 
    (   kernel['Linkage'], 
        kernel['ParameterCount'], 
        kernel['size'],
        kernel['RegisterCount'],
        kernel['SharedSize'],
        kernel['BarrierCount']))

def main():
    parser = argparse.ArgumentParser(description='Python CUDA SASS Assembler')
    parser.add_argument('-l','--list',action='store_true', default=False, help="List kernels and symbols in the cubin file")
    parser.add_argument('-e','--extract', action='store_true', default=False, help="Extract a single kernel into an asm file from a cubin. Works much like cuobjdump but outputs in a format that can be re-assembled back into the cubin.")
    parser.add_argument('-k','--kernel', help="Specify kernel name for extract operation.")
    parser.add_argument('-o','--output', help="Specify output assembly file name.")
    parser.add_argument('--cuobjdump', help="Specify an input cuobjdump file. For debugging purpose only when cuobjdume does not exist in the system.")
    parser.add_argument('cubin_file', type=str)
    args = parser.parse_args()

    #List kernels and symbol inside the cubin file
    if args.list:
        cubin = Cubin(args.cubin_file)
        printCubinInfo(args.cubin_file, cubin)
    elif args.extract:
        kernelName = args.kernel
        outputName = args.output
        outputFile = None
        kernel = None
        
        cubin = Cubin(args.cubin_file)

        if kernelName == None:
            kernelName = list(cubin.kernels.keys())[0]
            kernel = cubin.kernels[kernelName]
             
        cuobjdumpSass = ""
        
        if args.cuobjdump:
            sassFile = open(args.cuobjdump, 'r') 
            cuobjdumpSass = sassFile.readlines()
            sassFile.close()
        else:
            try:
                cuobjdumpSass = subprocess.check_output(['cuobjdump','-arch', "sm_"+str(cubin.arch),'-sass','-fun', kernelName, args.cubin_file], universal_newlines=True)
                cuobjdumpSass = cuobjdumpSass.split('\n')
            except FileNotFoundError:
                print("cuobjdump does not exist in this system. Please install CUDA toolkits before using this tool.")
                exit()
            except subprocess.CalledProcessError as err:
                print(err.cmd)
                exit()
        
        if outputName == None:
            outputFile = sys.stdout
        else:
            outputFile = open(outputName, 'w')
        
        outputFile.write("# Kernel: "+ kernelName +"\n" )
        outputFile.write("# Arch: sm_"+ str(cubin.arch) +"\n" )
        outputFile.write("# InsCnt: " +"\n" )
        outputFile.write("# RegCnt: "+ str(kernel['RegisterCount']) +"\n" )
        outputFile.write("# SharedSize: "+ str(kernel['SharedSize']) +"\n" )
        outputFile.write("# BarCnt: "+ str(kernel['BarrierCount']) +"\n" )
        outputFile.write("# Params(" + str(kernel['ParameterCount']) +"):\n#\tord:addr:size:align\n")
        for param in kernel['Parameters']:
            outputFile.write("#\t" + param + "\n")
        outputFile.write("# Instructions:\n\n")
        
        extract(cuobjdumpSass, outputFile, kernel['Parameters']) 
        
    else:
        parser.print_help()

if __name__ == "__main__":
    # execute only if run as a script
    main()
