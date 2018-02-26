make clean
make reg_shared

~/pyCuAsm/pyCuAsm.py -e -k _Z10QTC_devicePfPcS0_PiS1_S1_S_S1_iiifiiiib QTC.sm_52.cubin -o qtc.sass
~/pyCuAsm/pyCuAsm.py qtc.sass --use-local-spill -t 64 

if [ -f out.sass ]
then

	maxas.pl -i -n out.sass QTC.sm_52.cubin

	fatbinary --create="QTC.fatbin" -64 --key="996e26911bfed868" --ident="QTC.cu" --cmdline="-v  " "--image=profile=sm_52,file=QTC.sm_52.cubin" "--image=profile=compute_52,file=QTC.ptx" --embedded-fatbin="QTC.fatbin.c" --cuda
	gcc -D__CUDA_ARCH__=520 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS   -O3 -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "QTC.cu.cpp.ii" "QTC.cudafe1.cpp" 
	gcc -c -x c++ -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "QTC.o" "QTC.cu.cpp.ii" 
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "comm.o" "comm.cpp" 
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "libdata.o" "libdata.cpp" 	
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "CTimer.o" "common/CTimer.cpp" 
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "InvalidArgValue.o" "common/InvalidArgValue.cpp" 
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "Option.o" "common/Option.cpp" 
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "OptionParser.o" "common/OptionParser.cpp" 
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "ProgressBar.o" "common/ProgressBar.cpp" 
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "ResultDatabase.o" "common/ResultDatabase.cpp" 
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "Timer.o" "common/Timer.cpp" 
	gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "main.o" "common/main.cpp" 
	nvlink --arch=sm_52 --register-link-binaries="qtc_dlink.reg.c" -m64 -L"/usr/local/cuda-6.5/lib64"   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" "QTC.o" "comm.o" "libdata.o" "CTimer.o" "InvalidArgValue.o" "Option.o" "OptionParser.o" "ProgressBar.o" "ResultDatabase.o" "Timer.o" "main.o"  -lcudadevrt  -o "qtc_dlink.sm_52.cubin"
	fatbinary --create="qtc_dlink.fatbin" -64 --key="qtc_dlink" --ident="QTC.cu comm.cpp libdata.cpp common/CTimer.cpp common/InvalidArgValue.cpp common/Option.cpp common/OptionParser.cpp common/ProgressBar.cpp common/ResultDatabase.cpp common/Timer.cpp common/main.cpp " --cmdline="-v  " -link "--image=profile=sm_52,file=qtc_dlink.sm_52.cubin" --embedded-fatbin="qtc_dlink.fatbin.c" 
	gcc -c -x c++ -DFATBINFILE="\"qtc_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"qtc_dlink.reg.c\"" -I. -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "qtc_dlink.o" "/usr/local/cuda-6.5/bin/crt/link.stub" 
	g++ -O3 -m64 -o "qtc" -Wl,--start-group "qtc_dlink.o" "QTC.o" "comm.o" "libdata.o" "CTimer.o" "InvalidArgValue.o" "Option.o" "OptionParser.o" "ProgressBar.o" "ResultDatabase.o" "Timer.o" "main.o" -L"/usr/local/cuda-6.5/lib64"   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 

fi
