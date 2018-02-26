make clean
make reg_shared DIM=7

~/pyCuAsm/pyCuAsm.py -e -k _Z23nearest_neighbor_search9_gpu_treeP10gpu_point_iS1_i nn.sm_52.cubin -o nn.sass
sed -i -e 's/0.5.NEG/-0.5/g' nn.sass
~/pyCuAsm/pyCuAsm.py nn.sass --use-local-spill -t 192
sed -i -e 's/-0.5/0.5.NEG/g' out.sass

if [ -f out.sass ]
then
	maxas.pl -i -n out.sass nn.sm_52.cubin
fi

fatbinary --create="nn.fatbin" -64 --key="42d05ea61261e164" --ident="nn.cu" --cmdline="-v  " "--image=profile=sm_52,file=nn.sm_52.cubin" "--image=profile=compute_52,file=nn.ptx" --embedded-fatbin="nn.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=520 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS   -O2 -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "nn.cu.cpp.ii" "nn.cudafe1.cpp" 
gcc -c -x c++ -O2 "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "nn.o" "nn.cu.cpp.ii"
nvlink --arch=sm_52 --register-link-binaries="nn_dlink.reg.c" -m64 -lm -lpthread   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" "nn.o" "kdtree.o" "gpu_tree.o" "gpu_points.o"  -lcudadevrt  -o "nn_dlink.sm_52.cubin"
fatbinary --create="nn_dlink.fatbin" -64 --key="nn_dlink" --ident="nn.cu kdtree.cu gpu_tree.cu gpu_points.cu " --cmdline="-v  " -link "--image=profile=sm_52,file=nn_dlink.sm_52.cubin" --embedded-fatbin="nn_dlink.fatbin.c" 
gcc -c -x c++ -DFATBINFILE="\"nn_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"nn_dlink.reg.c\"" -I. -O2 "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "nn_dlink.o" "/usr/local/cuda-6.5/bin/crt/link.stub" 
g++ -O2 -m64 -o "nn" -Wl,--start-group "nn_dlink.o" "nn.o" "kdtree.o" "gpu_tree.o" "gpu_points.o" -lm -lpthread   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 
 
