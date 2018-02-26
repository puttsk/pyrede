make clean
make

~/pyCuAsm/pyCuAsm.py -e -k _Z17cuda_compute_fluxiPiPfS0_S0_S0_6float3S1_S1_S1_ euler3d.sm_52.cubin -o euler3d.sass
~/pyCuAsm/pyCuAsm.py euler3d.sass -r 14 -t 192 $@

if [ -f out.sass ]
then

	maxas.pl -i out.sass euler3d.sm_52.cubin
	~/pyCuAsm/pyCuAsm.py -e -k _Z17cuda_compute_fluxiPiPfS0_S0_S0_6float3S1_S1_S1_ euler3d.sm_52.cubin -o euler3d.out.sass

	fatbinary --create="euler3d.fatbin" -64 --key="f5b3ba595c41e47a" --ident="euler3d.cu" --cmdline="-v  " "--image=profile=sm_52,file=euler3d.sm_52.cubin" "--image=profile=compute_52,file=euler3d.ptx" --embedded-fatbin="euler3d.fatbin.c" --cuda
	gcc -D__CUDA_ARCH__=520 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS   -O3 -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT -I"/usr/local/cuda-6.5/samples/common/inc/" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "euler3d.cu.cpp.ii" "euler3d.cudafe1.cpp" 
	gcc -c -x c++ -O3 -I"/usr/local/cuda-6.5/samples/common/inc/" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -fpreprocessed -m64 -o "euler3d.o" "euler3d.cu.cpp.ii" 
	nvlink --arch=sm_52 --register-link-binaries="euler3d_dlink.reg.c" -m64 -L"/usr/local/cuda-6.5/lib64"   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" "euler3d.o"  -lcudadevrt  -o "euler3d_dlink.sm_52.cubin"
	fatbinary --create="euler3d_dlink.fatbin" -64 --key="euler3d_dlink" --ident="euler3d.cu " --cmdline="-v  " -link "--image=profile=sm_52,file=euler3d_dlink.sm_52.cubin" --embedded-fatbin="euler3d_dlink.fatbin.c" 
	gcc -c -x c++ -DFATBINFILE="\"euler3d_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"euler3d_dlink.reg.c\"" -I. -O3 -I"/usr/local/cuda-6.5/samples/common/inc/" "-I/usr/local/cuda-6.5/bin/../targets/x86_64-linux/include"   -m64 -o "euler3d_dlink.o" "/usr/local/cuda-6.5/bin/crt/link.stub" 
	g++ -O3 -m64 -o "euler3d" -Wl,--start-group "euler3d_dlink.o" "euler3d.o" -L"/usr/local/cuda-6.5/lib64"   "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-6.5/bin/../targets/x86_64-linux/lib" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 

fi
