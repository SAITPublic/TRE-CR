#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>

extern "C" void cudaFun(int devid, int is, int nx ,int nz)
{
    float *vp_device , *vp_host;
    cudaError_t cudaStatus;
    int dev_cnt;
    cudaDeviceProp prop;
    int dev;
    float mstimer;

    printf("[XB2] cuda start. \n");
    cudaStatus = cudaGetDeviceCount(&dev_cnt);
    if(cudaStatus != cudaSuccess)
    {
        printf("cudaGetDeviceCount failed! \n");
        return ;
    }
    printf("[XB2] The number of gpu is %d. \n", dev_cnt);
    sleep(3);
	printf("Dev: %d, GPU: %d\n", devid, devid % dev_cnt);
	cudaSetDevice(devid % dev_cnt);
	cudaGetDevice(&dev);
	cudaGetDeviceProperties(&prop, dev);
	printf("Name:                     %s\n", prop.name);
	cudaMalloc(&vp_device, nx*nz*sizeof(float));  
	cudaMemset(vp_device, 0, nx*nz*sizeof(float));

	vp_host=(float*)malloc(nx*nz*sizeof(float));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  //  sleep(3);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mstimer, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(vp_device);
	free(vp_host);

    printf("[XB2] Finished... \n");
}
