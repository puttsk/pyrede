include ./cuda.mk

all: pc

pc.o: pc.cu pc.h 
	$(NVCC) $(NVCC_OPTIONS) -c pc.cu

pc_block.o: pc_block.cu pc_block.h
	$(NVCC) $(NVCC_OPTIONS) -c pc_block.cu

pc_kernel_mem.o: pc_kernel_mem.cu pc_kernel_mem.cu
	$(NVCC) $(NVCC_OPTIONS) -c pc_kernel_mem.cu

pc_kernel.o: pc_kernel.cu pc_kernel.h
	$(NVCC) $(NVCC_OPTIONS) -c pc_kernel.cu

hashtab.o: hashtab.c hashtab.h
	$(NVCC) $(NVCC_OPTIONS) -c hashtab.c

pc: pc.o pc_block.o pc_kernel_mem.o hashtab.o pc_kernel.o
	$(NVCC) -o pc pc.o pc_block.o pc_kernel_mem.o hashtab.o pc_kernel.o

clean:
	rm -f *.o *~ pc *~ *.sass *.linkinfo *.cudafe* *_dlink* *.fatbin* *.o *.ptx *.cpp.* *.cpp1.* *.cpp2.* *.cpp3.* *.cpp4.* *.cubin *.hash *.module_id *.log


