
all: euler3d  

euler3d: euler3d.cu
	/usr/local/cuda-6.5/bin/nvcc $(KERNEL_DIM) -D_FORCE_INLINES --maxrregcount=68  -v -O2 --keep -arch=sm_52 -Xptxas -v euler3d.cu -o euler3d -I /usr/local/cuda-6.5/samples/common/inc/ -L /usr/local/cuda-6.5/lib64

reg: euler3d.cu
	/usr/local/cuda-6.5/bin/nvcc $(KERNEL_DIM) -D_FORCE_INLINES --maxrregcount=56 -v -O2 --keep -arch=sm_52 -Xptxas -v euler3d.cu -o euler3d -I /usr/local/cuda-6.5/samples/common/inc/ -L /usr/local/cuda-6.5/lib64

reg_shared: euler3d.cu
	/usr/local/cuda-6.5/bin/nvcc $(KERNEL_DIM) -D_FORCE_INLINES --maxrregcount=55 -v -O2 --keep -arch=sm_52 -Xptxas -v euler3d.cu -o euler3d -I /usr/local/cuda-6.5/samples/common/inc/ -L /usr/local/cuda-6.5/lib64

clean:
	rm -f euler3d density momentum *.sass *.linkinfo *.cudafe* *_dlink* *.fatbin* *.o *.ptx *.cpp* *.cubin *.hash *.module_id *.log momentum density density_energy *.dot
