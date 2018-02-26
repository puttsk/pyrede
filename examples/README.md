# README #

## Compiling Benchmark

+ To generate the register demotion variant, run the make_clean.sh script.
+ To generate the nvcc-generated variant, use the Makefile
+ To generate the local spilling variant, use make command with reg option.  
+ To generate the local-shared variant, run the make_clean_local.sh script.

## Run Tuning Script

Run `make_clean_tuning.sh` to run automatic tuning of the benchmark. The final report show estimated stalls caused by each variants of the program.


## NOTES

+ Compiling the benchmark requires MaxAs assembler [https://github.com/NervanaSystems/maxas] installed in the system.
+ All results in the paper was run on Ubuntu 14.04 with CUDA 6.5
+ Compiler flag `-D_FORCE_INLINES` was added to all benchmarks Makefile to make the benchmarks compilable on Ubuntu 16.04. 
  See. https://github.com/BVLC/caffe/issues/4046 for original discussion. 
  Following is the error message without `-D_FORCE_INLINES`
  
```
/usr/include/string.h: In function ‘void* __mempcpy_inline(void*, const void*, size_t)’:
/usr/include/string.h:652:42: error: ‘memcpy’ was not declared in this scope
```
