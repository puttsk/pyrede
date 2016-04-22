import argparse

from pyCuAsm.cubin import Cubin

def main():
    parser = argparse.ArgumentParser(description='Python CUDA SASS Assembler')
    parser.add_argument('-l','--list',action='store_true', default=False, help="List kernels and symbols in the cubin file")
    parser.add_argument('-e','--extract', action='store_true', default=False, help="Extract a single kernel into an asm file from a cubin. Works much like cuobjdump but outputs in a format that can be re-assembled back into the cubin.")
    parser.add_argument('-k','--kernel', help="Specify kernel name for extract operation.")
    parser.add_argument('cubin_file', type=str)
    args = parser.parse_args()

    #List kernels and symbol inside the cubin file
    if args.list:
        cubin = Cubin(args.cubin_file)
        print("%s: \n\tarch: sm_%d \n\tmachine: %d bit \n\taddress_size: %d bit\n" % (args.cubin_file, cubin.arch, cubin.addressSize, cubin.addressSize))
        for kernel in cubin.kernels:
            print("Kernel: " + kernel)
            print("\tLinkage: %s \n\tParams: %d \n\tSize: %d \n\tRegisters: %d \n\tSharedMem: %d \n\tBarriers: %d" % 
                (   cubin.kernels[kernel]['Linkage'], 
                    cubin.kernels[kernel]['ParameterCount'], 
                    cubin.kernels[kernel]['size'],
                    cubin.kernels[kernel]['RegisterCount'],
                    cubin.kernels[kernel]['SharedSize'],
                    cubin.kernels[kernel]['BarrierCount']))
        for symbol in cubin.symbols:
            print("Symbol: " + symbol + "\n")
    elif args.extract:
        kernelName = args.kernel
        cubin = Cubin(args.cubin_file)
        
        
    else:
        parser.print_help()

if __name__ == "__main__":
    # execute only if run as a script
    main()
