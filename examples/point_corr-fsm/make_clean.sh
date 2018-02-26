make clean
make DIM=7 RADIUS=0.32f

~/pyCuAsm/pyCuAsm.py -e -k _Z19compute_correlation17_pc_kernel_params pc_kernel.sm_52.cubin -o pc.sass
sed -i -e 's/0.5.NEG/-0.5/g' pc.sass
~/pyCuAsm/pyCuAsm.py pc.sass -r 6 -t 256 $@
sed -i -e 's/-0.5/0.5.NEG/g' out.sass

if [ -f out.sass ]
then
	maxas.pl -i -n out.sass pc_kernel.sm_52.cubin
fi

fatbinary --create="pc_kernel.fatbin" -64 --key="2cde607eb8939ae6" --ident="pc_kernel.cu" --cmdline="-v  " "--image=profile=sm_52,file=pc_kernel.sm_52.cubin" "--image=profile=compute_52,file=pc_kernel.ptx" --embedded-fatbin="pc_kernel.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=520 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS   -O2 -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "pc_kernel.cu.cpp.ii" "pc_kernel.cudafe1.cpp" 
gcc -c -x c++ -O2 "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "pc_kernel.o" "pc_kernel.cu.cpp.ii" 
/usr/local/cuda-6.5/bin/nvcc -o pc pc.o pc_block.o pc_kernel_mem.o hashtab.o pc_kernel.o

