CUDA_INSTALL_PATH = /usr/local/cuda-11.3
MPI_INSTALL_PATH = /home/xingbiao/DMTCP_Tools/mpich-3.2.1/mpich-install
#MPI_INSTALL_PATH = /home/xingbiao/lammps/openmpi_install

NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
MPICC = $(MPI_INSTALL_PATH)/bin/mpicc
 
LDFLAGS = -L$(CUDA_INSTALL_PATH)/lib64
LIB = -lcudart -lcurand -lcuda

WARNING_FLAGS=-Wall -Wno-deprecated-declarations -Werror
NVCC_FLAGS=-ccbin=/usr/bin/g++ -Xcompiler '-g3 -O0 $(WARNING_FLAGS)'

CFILES = main.c
CUFILES = compute.cu
OBJECTS = main.o compute.o
EXECNAME = test
 
all:
	$(MPICC) -c $(CFILES)
	$(NVCC) -c $(CUFILES)
	$(MPICC) -o $(EXECNAME) $(LDFLAGS) $(LIB) $(OBJECTS)
 
clean:
	rm -f *.o $(EXECNAME)
