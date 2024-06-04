nvcc -ptx matSumKernel.cu -o matSumKernel.ptx
nvcc drivertest.cpp -o drivertest.exe -lcuda --cudart=shared
