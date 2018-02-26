all:
	/usr/local/cuda-6.5/bin/nvcc -o conv --keep -v -Xptxas -v -arch=sm_52 *.cu *.cpp -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I . -I /usr/local/cuda-6.5/samples/common/inc/ -lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW -lcurand

reg:
	/usr/local/cuda-6.5/bin/nvcc -o conv --maxrregcount=32 --keep -v -Xptxas -v -arch=sm_52 *.cu *.cpp -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I . -I /usr/local/cuda-6.5/samples/common/inc/ -lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW -lcurand

reg_shared:
	/usr/local/cuda-6.5/bin/nvcc -o conv --maxrregcount=31 --keep -v -Xptxas -v -arch=sm_52 *.cu *.cpp -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I . -I /usr/local/cuda-6.5/samples/common/inc/ -lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW -lcurand

clean:
	rm conv *.sass *.linkinfo *.cudafe* *_dlink* *.fatbin* *.o *.ptx *.cpp.* *.cpp1.* *.cpp2.* *.cpp3.* *.cpp4.* *.cubin *.hash *.module_id *.log


