all:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES -o md5hash --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu common/*.cpp  -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I common/ -I .

reg:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES -o md5hash --maxrregcount=32 --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu common/*.cpp  -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I common/ -I .

reg_shared:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES -o md5hash --maxrregcount=31 --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu common/*.cpp  -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I common/ -I .

clean:
	rm md5hash *.sass *.linkinfo *.cudafe* *_dlink* *.fatbin* *.o *.ptx *.cpp.* *.cpp1.* *.cpp2.* *.cpp3.* *.cpp4.* *.cubin *.hash *.module_id *.log
