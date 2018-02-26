all:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu *.cpp common/*.cpp -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I common/ -I . -o qtc

reg:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES --maxrregcount=48 --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu *.cpp common/*.cpp -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I common/ -I . -o qtc

reg_shared:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES --maxrregcount=47 --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu *.cpp common/*.cpp -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I common/ -I . -o qtc
	
clean:
	rm qtc *.sass *.linkinfo *.cudafe* *_dlink* *.fatbin* *.o *.ptx *.cpp.* *.cpp1.* *.cpp2.* *.cpp3.* *.cpp4.* *.cubin *.hash *.module_id *.log
