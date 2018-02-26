make clean
make reg_shared

#~/pyCuAsm/pyCuAsm.py -e -k _Z21convolutionRowsKernelPfS_iii convolutionSeparable.sm_52.cubin -o conv.sass
#~/pyCuAsm/pyCuAsm.py conv.sass -r 4 -t 64
~/pyCuAsm/pyCuAsm.py -e -k _Z24convolutionColumnsKernelPfS_iii convolutionSeparable.sm_52.cubin -o conv.sass
~/pyCuAsm/pyCuAsm.py -e -k _Z24convolutionColumnsKernelPfS_iii convolutionSeparable.sm_52.cubin -o conv.sass.local
~/pyCuAsm/pyCuAsm.py conv.sass -t 128 --use-local-spill

sed -i -e 's/0xffff8/-0x00008/g' out.sass

if [ -f out.sass ]
then
	maxas.pl -i -n out.sass convolutionSeparable.sm_52.cubin
fi

fatbinary --create="convolutionSeparable.fatbin" -64 --key="e6a7040b0c8adcd2" --ident="convolutionSeparable.cu" --cmdline="-v  " "--image=profile=sm_52,file=convolutionSeparable.sm_52.cubin" "--image=profile=compute_52,file=convolutionSeparable.ptx" --embedded-fatbin="convolutionSeparable.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=520 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS   -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT -I"/usr/local/cuda-6.5/include/" -I"." -I"/usr/local/cuda-6.5/samples/common/inc/" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "convolutionSeparable.cu.cpp.ii" "convolutionSeparable.cudafe1.cpp"
gcc -c -x c++ -I"/usr/local/cuda-6.5/include/" -I"." -I"/usr/local/cuda-6.5/samples/common/inc/" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "convolutionSeparable.o" "convolutionSeparable.cu.cpp.ii"
gcc -c -x c++ -D__NVCC__  -I"/usr/local/cuda-6.5/include/" -I"." -I"/usr/local/cuda-6.5/samples/common/inc/" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "convolutionSeparable_gold.o" "convolutionSeparable_gold.cpp"
gcc -c -x c++ -D__NVCC__  -I"/usr/local/cuda-6.5/include/" -I"." -I"/usr/local/cuda-6.5/samples/common/inc/" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "main.o" "main.cpp"
nvlink --arch=sm_52 --register-link-binaries="conv_dlink.reg.c" -m64 -L"/usr/local/cuda-6.5/lib64" -lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW -lcurand   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" "convolutionSeparable.o" "convolutionSeparable_gold.o" "main.o"  -lcudadevrt  -o "conv_dlink.sm_52.cubin"
fatbinary --create="conv_dlink.fatbin" -64 --key="conv_dlink" --ident="convolutionSeparable.cu convolutionSeparable_gold.cpp main.cpp " --cmdline="-v  " -link "--image=profile=sm_52,file=conv_dlink.sm_52.cubin" --embedded-fatbin="conv_dlink.fatbin.c"
gcc -c -x c++ -DFATBINFILE="\"conv_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"conv_dlink.reg.c\"" -I. -I"/usr/local/cuda-6.5/include/" -I"." -I"/usr/local/cuda-6.5/samples/common/inc/" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "conv_dlink.o" "/usr/local/cuda-6.5/bin/crt/link.stub"
g++ -m64 -o "conv" -Wl,--start-group "conv_dlink.o" "convolutionSeparable.o" "convolutionSeparable_gold.o" "main.o" -L"/usr/local/cuda-6.5/lib64" -lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW -lcurand   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group

