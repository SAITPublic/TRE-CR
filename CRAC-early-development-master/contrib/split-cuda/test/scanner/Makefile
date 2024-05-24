CUDA_INSTALL_PATH = /usr/local/cuda

GPLUSPLUS = $(GCC_INSTALL_DIR)/bin/g++
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc

LDFLAGS = -L$(CUDA_INSTALL_PATH)/lib64
LIB = -lcudart -lcurand -lcuda

WARNING_FLAGS=-Wall -Wno-deprecated-declarations -Werror
NVCC_FLAGS=-ccbin=$(GPLUSPLUS) -Xcompiler '-g3 -O0 $(WARNING_FLAGS)'

CPPFILES = scanner.cpp
CUFILES = scan_on_gpu.cu
OBJECTS = scanner.o scan_on_gpu.o
EXECNAME = scanner
 
all:
	$(GPLUSPLUS) -c $(CPPFILES)
	$(NVCC) -c $(CUFILES)
	$(GPLUSPLUS) -o $(EXECNAME) $(LDFLAGS) $(LIB) $(OBJECTS)
 
clean:
	rm -f *.o $(EXECNAME)
