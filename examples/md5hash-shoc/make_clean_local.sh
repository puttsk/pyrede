make clean
make reg_shared

~/pyCuAsm/pyCuAsm.py -e -k _Z24FindKeyWithDigest_KerneljjjjiiiPiPhPj MD5Hash.sm_52.cubin -o md5hash.sass
~/pyCuAsm/pyCuAsm.py -e -k _Z24FindKeyWithDigest_KerneljjjjiiiPiPhPj MD5Hash.sm_52.cubin -o md5hash.sass.local
~/pyCuAsm/pyCuAsm.py md5hash.sass --use-local-spill -t 256 

if [ -f out.sass ]
then
	sed -i -e 's/0xf5bb1/-0xa44f/g' out.sass
	sed -i -e 's/0xa3942/-0x5c6be/g' out.sass

	maxas.pl -i -n out.sass MD5Hash.sm_52.cubin
fi

fatbinary --create="MD5Hash.fatbin" -64 --key="e832115f5d7b559d" --ident="MD5Hash.cu" --cmdline="-v  " "--image=profile=sm_52,file=MD5Hash.sm_52.cubin" "--image=profile=compute_52,file=MD5Hash.ptx" --embedded-fatbin="MD5Hash.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=520 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS   -O3 -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "MD5Hash.cu.cpp.ii" "MD5Hash.cudafe1.cpp" 
gcc -c -x c++ -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "MD5Hash.o" "MD5Hash.cu.cpp.ii" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "CTimer.o" "common/CTimer.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "InvalidArgValue.o" "common/InvalidArgValue.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "Option.o" "common/Option.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "OptionParser.o" "common/OptionParser.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "ProgressBar.o" "common/ProgressBar.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "ResultDatabase.o" "common/ResultDatabase.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "Timer.o" "common/Timer.cpp" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "main.o" "common/main.cpp" 
nvlink --arch=sm_52 --register-link-binaries="md5hash_dlink.reg.c" -m64 -L"/usr/local/cuda-6.5/lib64"   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" "MD5Hash.o" "CTimer.o" "InvalidArgValue.o" "Option.o" "OptionParser.o" "ProgressBar.o" "ResultDatabase.o" "Timer.o" "main.o"  -lcudadevrt  -o "md5hash_dlink.sm_52.cubin"
fatbinary --create="md5hash_dlink.fatbin" -64 --key="md5hash_dlink" --ident="MD5Hash.cu common/CTimer.cpp common/InvalidArgValue.cpp common/Option.cpp common/OptionParser.cpp common/ProgressBar.cpp common/ResultDatabase.cpp common/Timer.cpp common/main.cpp " --cmdline="-v  " -link "--image=profile=sm_52,file=md5hash_dlink.sm_52.cubin" --embedded-fatbin="md5hash_dlink.fatbin.c" 
gcc -c -x c++ -DFATBINFILE="\"md5hash_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"md5hash_dlink.reg.c\"" -I. -O3 -I"/usr/local/cuda-6.5/include/" -I"common/" -I"." "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "md5hash_dlink.o" "/usr/local/cuda-6.5/bin/crt/link.stub" 
g++ -O3 -m64 -o "md5hash" -Wl,--start-group "md5hash_dlink.o" "MD5Hash.o" "CTimer.o" "InvalidArgValue.o" "Option.o" "OptionParser.o" "ProgressBar.o" "ResultDatabase.o" "Timer.o" "main.o" -L"/usr/local/cuda-6.5/lib64"   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 

