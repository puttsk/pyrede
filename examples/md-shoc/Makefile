all:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES  -o md --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu common/*.cpp  -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I common/ -I .

reg:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES --maxrregcount=32 -o md --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu common/*.cpp  -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I common/ -I .

reg_shared:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES --maxrregcount=31 -o md --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu common/*.cpp  -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I common/ -I .

clean:
	rm md *.sass *.linkinfo *.cudafe* *_dlink* *.fatbin* *.o *.ptx *.cpp.* *.cpp1.* *.cpp2.* *.cpp3.* *.cpp4.* *.cubin *.hash *.module_id *.log
