make clean
make reg

~/pyCuAsm/pyCuAsm.py -e -k _Z16compute_lj_forceId7double37double4EvPT0_PKT1_iPKiT_S9_S9_i MD.sm_52.cubin -o md.sass
~/pyCuAsm/pyCuAsm.py -e -k _Z16compute_lj_forceId7double37double4EvPT0_PKT1_iPKiT_S9_S9_i MD.sm_52.cubin -o md.sass.local
~/pyCuAsm/pyCuAsm.py md.sass -t 256 --use-local-spill
#~/pyCuAsm/pyCuAsm.py md.sass -r 3 -t 512 $@

if [ -f out.sass ]
then
	maxas.pl -i -n out.sass MD.sm_52.cubin
fi

fatbinary --create="MD.fatbin" -64 --key="a494882937c0668f" --ident="MD.cu" --cmdline="-v  " "--image=profile=sm_52,file=MD.sm_52.cubin" "--image=profile=compute_52,file=MD.ptx" --embedded-fatbin="MD.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=520 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS   -O3 -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "MD.cu.cpp.ii" "MD.cudafe1.cpp" 
gcc -c -x c++ -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "MD.o" "MD.cu.cpp.ii" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "CTimer.o" "common/CTimer.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "InvalidArgValue.o" "common/InvalidArgValue.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "Option.o" "common/Option.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "OptionParser.o" "common/OptionParser.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "ProgressBar.o" "common/ProgressBar.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "ResultDatabase.o" "common/ResultDatabase.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "Timer.o" "common/Timer.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "main.o" "common/main.cpp" 
nvlink --arch=sm_52 --register-link-binaries="md_dlink.reg.c" -m64 -L"/usr/local/cuda-6.5/lib64"   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" "MD.o" "CTimer.o" "InvalidArgValue.o" "Option.o" "OptionParser.o" "ProgressBar.o" "ResultDatabase.o" "Timer.o" "main.o"  -lcudadevrt  -o "md_dlink.sm_52.cubin"
fatbinary --create="md_dlink.fatbin" -64 --key="md_dlink" --ident="MD.cu common/CTimer.cpp common/InvalidArgValue.cpp common/Option.cpp common/OptionParser.cpp common/ProgressBar.cpp common/ResultDatabase.cpp common/Timer.cpp common/main.cpp " --cmdline="-v  " -link "--image=profile=sm_52,file=md_dlink.sm_52.cubin" --embedded-fatbin="md_dlink.fatbin.c" 
gcc -c -x c++ -DFATBINFILE="\"md_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"md_dlink.reg.c\"" -I. -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "md_dlink.o" "/usr/local/cuda-6.5/bin/crt/link.stub" 
g++ -O3 -m64 -o "md" -Wl,--start-group "md_dlink.o" "MD.o" "CTimer.o" "InvalidArgValue.o" "Option.o" "OptionParser.o" "ProgressBar.o" "ResultDatabase.o" "Timer.o" "main.o" -L"/usr/local/cuda-6.5/lib64"   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 

