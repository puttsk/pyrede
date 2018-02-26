make clean
make reg_shared

~/pyCuAsm/pyCuAsm.py -e -k _Z24d_recursiveGaussian_rgbaPjS_iiffffffff recursiveGaussian_cuda.sm_52.cubin -o recursiveGaussian.sass
~/pyCuAsm/pyCuAsm.py -e -k _Z24d_recursiveGaussian_rgbaPjS_iiffffffff recursiveGaussian_cuda.sm_52.cubin -o recursiveGaussian.sass.local
~/pyCuAsm/pyCuAsm.py recursiveGaussian.sass --use-local-spill -t 64
#sed -i -e 's/R32/R31/g' out.sass

if [ -f out.sass ]
then
	maxas.pl -i -n out.sass recursiveGaussian_cuda.sm_52.cubin
fi

fatbinary --create="recursiveGaussian_cuda.fatbin" -64 --key="e5ab58a3baa769ae" --ident="recursiveGaussian_cuda.cu" --cmdline="-v  " "--image=profile=sm_52,file=recursiveGaussian_cuda.sm_52.cubin" "--image=profile=compute_52,file=recursiveGaussian_cuda.ptx" --embedded-fatbin="recursiveGaussian_cuda.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=520 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS   -O3 -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT -I"/usr/local/cuda-6.5/include/" -I"." -I"./inc" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "recursiveGaussian_cuda.cu.cpp.ii" "recursiveGaussian_cuda.cudafe1.cpp" 
gcc -c -x c++ -O3 -I"/usr/local/cuda-6.5/include/" -I"." -I"./inc" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "recursiveGaussian_cuda.o" "recursiveGaussian_cuda.cu.cpp.ii" 
gcc -c -x c++ -D__NVCC__  -O3 -I"/usr/local/cuda-6.5/include/" -I"." -I"./inc" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "recursiveGaussian.o" "recursiveGaussian.cpp" 
nvlink --arch=sm_52 --register-link-binaries="recgaussian_dlink.reg.c" -m64 -L"/usr/local/cuda-6.5/lib64" -lGL -lGLU -lglut -lGLEW   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" "recursiveGaussian_cuda.o" "recursiveGaussian.o"  -lcudadevrt  -o "recgaussian_dlink.sm_52.cubin"
fatbinary --create="recgaussian_dlink.fatbin" -64 --key="recgaussian_dlink" --ident="recursiveGaussian_cuda.cu recursiveGaussian.cpp " --cmdline="-v  " -link "--image=profile=sm_52,file=recgaussian_dlink.sm_52.cubin" --embedded-fatbin="recgaussian_dlink.fatbin.c" 
gcc -c -x c++ -DFATBINFILE="\"recgaussian_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"recgaussian_dlink.reg.c\"" -I. -O3 -I"/usr/local/cuda-6.5/include/" -I"." -I"./inc" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "recgaussian_dlink.o" "/usr/local/cuda-6.5/bin/crt/link.stub" 
g++ -O3 -m64 -o "recgaussian" -Wl,--start-group "recgaussian_dlink.o" "recursiveGaussian_cuda.o" "recursiveGaussian.o" -L"/usr/local/cuda-6.5/lib64" -lGL -lGLU -lglut -lGLEW   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 

