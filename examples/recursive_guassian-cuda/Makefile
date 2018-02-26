all:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES  -o recgaussian --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu *.cpp -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I . -I ./inc -lGL -lGLU -lglut -lGLEW
 
reg:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES --maxrregcount=40 -o recgaussian --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu *.cpp -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I . -I ./inc -lGL -lGLU -lglut -lGLEW 

reg_shared:
	/usr/local/cuda-6.5/bin/nvcc -D_FORCE_INLINES --maxrregcount=39 -o recgaussian --keep -v -O3 -Xptxas -v -arch=sm_52 *.cu *.cpp -I /usr/local/cuda-6.5/include/ -L /usr/local/cuda-6.5/lib64 -I . -I ./inc -lGL -lGLU -lglut -lGLEW

clean:
	rm recgaussian *.sass *.linkinfo *.cudafe* *_dlink* *.fatbin* *.o *.ptx *.cpp.* *.cpp1.* *.cpp2.* *.cpp3.* *.cpp4.* *.cubin *.hash *.module_id *.log

