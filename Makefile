#NVCC compiler and flags
CUDA_DIR ?= /usr/local/cuda
NVCC = nvcc
NVCCFLAGS   = -O3 --compiler-options '-fPIC' --compiler-bindir=/usr/bin/gcc --shared -Xcompiler -Wall -arch=sm_61 -lrt

# linker options
CUDA_LDFLAGS  = -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib64 -lcudart -lcufft
LFLAGS_PGPLOT = -L/usr/lib64/pgplot -lpgplot -lcpgplot -lX11 -lgfortran
HP_LDFLAGS = -L/usr/local/lib -lhashpipe -lhashpipestatus -lrt -lm -lpthread

NVCC_FLAGS = $(NVCCFLAGS) $(CUDA_LDFLAGS) $(LFLAGS_PGPLOT) $(HP_LDFLAGS)

# HASHPIPE
HP_LIB_TARGET   = demo4_hashpipe.o
HP_LIB_SOURCES  = demo4_net_thread.c \
			  demo4_net_thread2.c \
		      demo4_output_thread.c \
		      demo4_databuf.c
HP_LIB_OBJECTS = $(patsubst %.c,%.o,$(HP_LIB_SOURCES))
HP_LIB_INCLUDES = demo4_databuf.h demo4_gpu_thread.h
HP_TARGET = demo4_hashpipe.so
# GPU
GPU_LIB_TARGET = demo4_gpu_kernels.o
GPU_LIB_SOURCES = demo4_gpu_kernels.cu demo4_gpu_thread.cu
GPU_LIB_INCLUDES =  demo4_gpu_thread.h

#PLOT
GPU_PLOT_TARGET = demo4_plot.o
# Filterbank
FILTERBANK_OBJECT   = filterbank.o

all: $(GPU_LIB_TARGET) $(FILTERBANK_OBJECT) $(GPU_PLOT_TARGET) $(HP_LIB_TARGET) $(HP_TARGET)

$(GPU_LIB_TARGET): $(GPU_LIB_SOURCES)
	$(NVCC) -c $^ $(NVCC_FLAGS)
	
$(FILTERBANK_OBJECT): filterbank.cpp filterbank.h
	$(NVCC) -c $< $(NVCC_FLAGS)

$(GPU_PLOT_TARGET): demo4_plot.cu demo4_plot.h
	$(NVCC) -c $< $(NVCC_FLAGS)

$(HP_LIB_TARGET): $(HP_LIB_SOURCES)
	$(NVCC) -c $^ $(NVCC_FLAGS)

# Link HP_OBJECTS together to make plug-in .so file
$(HP_TARGET): $(GPU_LIB_TARGET) $(FILTERBANK_OBJECT)
	$(NVCC) *.o -o $@ $(NVCC_FLAGS)
tags:
	ctags -R .
clean:
	rm -f $(HP_LIB_TARGET) $(GPU_LIB_TARGET) $(FILTERBANK_OBJECT) $(GPU_PLOT_TARGET) $(HP_TARGET) *.o tags 

prefix=/home/peix/local
LIBDIR=$(prefix)/lib
BINDIR=$(prefix)/bin
install-lib: $(HP_TARGET)
	mkdir -p "$(LIBDIR)"
	install -p $^ "$(LIBDIR)"
install: install-lib

.PHONY: all tags clean install install-lib
# vi: set ts=8 noet :
