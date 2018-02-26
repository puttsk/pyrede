include ./cuda.mk

all: nn

nn:  
	$(NVCC) $(NVCC_OPTIONS) $(NVCC_LINK_OPTIONS) -o nn nn.cu kdtree.cu gpu_tree.cu gpu_points.cu 

reg:  
	$(NVCC) $(NVCC_OPTIONS) --maxrregcount=32 $(NVCC_LINK_OPTIONS) -o nn nn.cu kdtree.cu gpu_tree.cu gpu_points.cu 

reg_shared:  
	$(NVCC) $(NVCC_OPTIONS) --maxrregcount=31 $(NVCC_LINK_OPTIONS) -o nn nn.cu kdtree.cu gpu_tree.cu gpu_points.cu 

nn.o: nn.cu nn.h
	$(NVCC) $(NVCC_OPTIONS) -c nn.cu

kdtree.o: kdtree.cu nn.h
	$(NVCC) $(NVCC_OPTIONS) -c kdtree.cu

gpu_tree.o: gpu_tree.cu gpu_tree.h nn.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_tree.cu

gpu_points.o: gpu_points.cu nn.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_points.cu

kernel.o: kernel.cu nn.h nn_gpu.h nn_kernel_macros.inc
	$(NVCC) $(NVCC_OPTIONS) -c kernel.cu

clean:
	rm -f *.o nn *~ *.sass *.linkinfo *.cudafe* *_dlink* *.fatbin* *.o *.ptx *.cpp.* *.cpp1.* *.cpp2.* *.cpp3.* *.cpp4.* *.cubin *.hash *.module_id *.log


