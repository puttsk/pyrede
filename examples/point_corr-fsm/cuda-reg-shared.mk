#CUDA_20 = 1
#DEBUG = 1

ifeq ($(TRACK_TRAVERSALS),1)
	COMMON_COMPILE_FLAGS += -DTRACK_TRAVERSALS=$(TRACK_TRAVERSALS)
endif

ifdef RADIUS
ifneq ($(RADIUS),0)
        COMMON_COMPILE_FLAGS += -DRADIUS=$(RADIUS)
endif
endif

ifneq ($(DEBUG),1)
	COMMON_COMPILE_FLAGS += -O2
else
	COMMON_COMPILE_FLAGS += -g
endif

ifdef SPLICE_DEPTH
ifneq ($(SPLICE_DEPTH),100000)
	COMMON_COMPILE_FLAGS += -DSPLICE_DEPTH=$(SPLICE_DEPTH)
endif
endif

ifdef DIM
	COMMON_COMPILE_FLAGS += -DDIM=$(DIM)
endif

COMMON_LINK_FLAGS += -lm -lpthread

CUDA_PATH = /usr/local
NVCC = /usr/local/cuda-6.5/bin/nvcc

NVCC_OPTIONS = --keep -v -arch sm_52 -Xptxas -v $(COMMON_COMPILE_FLAGS) --maxrregcount=31

NVCC_LINK_OPTIONS = $(COMMON_LINK_FLAGS)

