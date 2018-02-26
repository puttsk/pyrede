# pyRede

## Requirement
+ Python 3.4 

## Running the Translator
Uses command ./pyCuAsm.py to run the translator

## Parameters

```
usage: pyCuAsm.py [-h] [-l] [-e] [-c] [--tuning] [-k KERNEL] [-o OUTPUT]
                  [-r SPILL_REGISTER] [--exclude-registers EXCLUDE_REGISTERS]
                  [-t THREAD_BLOCK_SIZE] [-O OPT_LEVEL] [--use-local-spill]
                  [--no-register-relocation] [--avoid-conflict AVOID_CONFLICT]
                  [--swap-spill-reg SWAP_SPILL_REG] [--opt-access OPT_ACCESS]
                  [--candidate_type CANDIDATE_TYPE] [--cuobjdump CUOBJDUMP]
                  [--local-sass LOCAL_SASS]
                  [--local-sass-shared LOCAL_SASS_SHARED]
                  input_file

Python CUDA SASS Assembler

positional arguments:
  input_file

optional arguments:
  -h, --help            show this help message and exit
  -l, --list            List kernels and symbols in the cubin file
  -e, --extract         Extract a single kernel into an asm file from a cubin.
                        Works much like cuobjdump but outputs in a format that
                        can be re-assembled back into the cubin.
  -c, --compiler        Compiler and optimize input SASS file. (default)
  --tuning              Analyse the benefit of register demotion
  -k KERNEL, --kernel KERNEL
                        Specify kernel name for extract operation.
  -o OUTPUT, --output OUTPUT
                        Specify output assembly file name.
  -r SPILL_REGISTER, --spill-register SPILL_REGISTER
                        Spill a specific number of registers to shared memory
  --exclude-registers EXCLUDE_REGISTERS
                        Exclude specific registers from spilling candidate
  -t THREAD_BLOCK_SIZE, --thread-block-size THREAD_BLOCK_SIZE
                        Number of threads in thread block
  -O OPT_LEVEL, --opt-level OPT_LEVEL
                        Specify optimization level
  --use-local-spill     Convert local spill to shared spill
  --no-register-relocation
                        Disable register relocation after spilling
  --avoid-conflict AVOID_CONFLICT
                        0: Disable / 1:Enable register conflict avoidance
  --swap-spill-reg SWAP_SPILL_REG
                        0: Disable / 1:Enable spill register swapping
  --opt-access OPT_ACCESS
                        0: Disable / 1:Enable spill register swapping
  --candidate_type CANDIDATE_TYPE
                        0: CFG / 1:Static Access / 2: Static Conflict
  --cuobjdump CUOBJDUMP
                        Specify an input cuobjdump file. For debugging purpose
                        only when cuobjdume does not exist in the system.
  --local-sass LOCAL_SASS
                        SASS code with local spilling
  --local-sass-shared LOCAL_SASS_SHARED
                        SASS code with local spilling to shared
```

# NOTES #
* MaxAs does not work with CUDA 7.0 and newer version
