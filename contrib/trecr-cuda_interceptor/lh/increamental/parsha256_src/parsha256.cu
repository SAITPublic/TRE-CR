
#include "parsha256_on_gpu.h"
#include <iostream>
#include <string>
void parsha256_on_gpu_test();

__global__ void print(int * A_dev,int nBytes)
{
    printf("in kernel\n");
    printf("%d\n", A_dev[0]);
}

std::string compute_hash(int* A_dev, long long nBytes){
    // CUcontext prev_ctx;
    // cuCtxGetCurrent(&prev_ctx);
    // print<<<1,1>>>((int*)A_dev, nBytes/4);
    // gpuErrchk(cudaPeekAtLastError());
    char result[100];
    parsha256_gpu_mem((int*)A_dev, nBytes/2, result);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return std::string(result);
}


void parsha256_on_gpu_test() {
    // long int n = 1000000000;
    // long int nBytes = n*sizeof(int);
    // int* A_host = ( int*)malloc(nBytes);
    // int* A_dev = NULL;
    // for (long int i = 0; i < n; i++){
    //     A_host[i] = 1;
    // }
    // gpuErrchk(cudaMalloc((void**)&A_dev, nBytes));
    // A_host[n-1] = 0;
    // gpuErrchk(cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice));
    // std::string result = parsha256_gpu_mem(A_dev, nBytes);
    // std::cout << result << std::endl;
}
