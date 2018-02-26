make clean 
make -f Makefile-reg-shared DIM=7

~/pyCuAsm/pyCuAsm.py -e -k _Z13search_kernel10__GPU_treeP5PointP11__GPU_pointS1_ kernel.sm_52.cubin -o vptree.sass
sed -i -e 's/0.5.NEG/-0.5/g' vptree.sass
cp vptree.sass vptree.sass.local
~/pyCuAsm/pyCuAsm.py vptree.sass -t 256 --use-local-spill
sed -i -e 's/-0.5/0.5.NEG/g' out.sass

if [ -f out.sass ]
then
	maxas.pl -i -n out.sass kernel.sm_52.cubin
fi

fatbinary --create="kernel.fatbin" -64 --key="ebabf95d1291f03c" --ident="kernel.cu" --cmdline="-v  " "--image=profile=sm_52,file=kernel.sm_52.cubin" "--image=profile=compute_52,file=kernel.ptx" --embedded-fatbin="kernel.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=520 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS   -O2 -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "kernel.cu.cpp.ii" "kernel.cudafe1.cpp" 
gcc -c -x c++ -O2 "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "kernel.o" "kernel.cu.cpp.ii" 
/usr/local/cuda-6.5/bin/nvcc -lm -lpthread -o vptree vptree.o ptrtab.o gpu_tree.o kernel.o

