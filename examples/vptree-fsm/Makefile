include ./cuda.mk

all: vptree

vptree.o: vptree.cu vptree.h
	$(NVCC) $(NVCC_OPTIONS) -c vptree.cu

gpu_tree.o: gpu_tree.cu vptree.h ptrtab.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_tree.cu

kernel.o: kernel.cu vptree.h
	$(NVCC) $(NVCC_OPTIONS) -c kernel.cu

ptrtab.o: ptrtab.cu ptrtab.h
	$(NVCC) $(NVCC_OPTIONS) -c ptrtab.cu

vptree: vptree.o ptrtab.o gpu_tree.o kernel.o
	$(NVCC) $(NVCC_LINK_OPTIONS) -o vptree vptree.o ptrtab.o gpu_tree.o kernel.o

clean:
	rm -f *.o *~ vptree *.sass *.linkinfo *.cudafe* *_dlink* *.fatbin* *.o *.ptx *.cpp.* *.cpp1.* *.cpp2.* *.cpp3.* *.cpp4.* *.cubin *.hash *.module_id *.log


