# README #

## Compiling Benchmark

+ To generate the register demotion variant, run the make_clean.sh script.
+ To generate the nvcc-generated variant, use the Makefile
+ To generate the local spilling variant, use make command with reg option.  
+ To generate the local-shared variant, run the make_clean_local.sh script.

## Run Tuning Script

Run `make_clean_tuning.sh` to run automatic tuning of the benchmark. The final report show estimated stalls caused by each variants of the program.


## NOTES

Compiling the benchmark requires MaxAs assembler [https://github.com/NervanaSystems/maxas] installed in the system.