#include <stdio.h>
#include <stdlib.h>
#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<unistd.h>
#include<sys/time.h>
#include <cuda.h>
#include <builtin_types.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include "matSumKernel.h"

// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

// --- global variables ----------------------------------------------------
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;
//CUstream stream;
size_t     totalGlobalMem;

char       *module_file = (char*) "matSumKernel.ptx";
char       *kernel_name = (char*) "matSum";

CUdeviceptr d_a, d_b, d_c;

// --- functions -----------------------------------------------------------
void initCUDA(int argc, char **argv)
{
    int deviceCount = 0;
    CUresult err = cuInit(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    printf("Get device count : %d\n", deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA device
    checkCudaErrors(cuDeviceGet(&device, 0));
    CUuuid uuid;
    checkCudaErrors(cuDeviceGetUuid(&uuid, device));
    CUdevprop prop;
    checkCudaErrors(cuDeviceGetProperties(&prop, device));
    char name[100];
    cuDeviceGetName(name, 100, device);

    // get compute capabilities and the devicename
    checkCudaErrors( cuDeviceComputeCapability(&major, &minor, device) );
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    checkCudaErrors( cuDeviceTotalMem(&totalGlobalMem, device) );
    printf("  Total amount of global memory:   %llu bytes\n",
           (unsigned long long)totalGlobalMem);
    printf("  64-bit Memory Address:           %s\n",
           (totalGlobalMem > (unsigned long long)4*1024*1024*1024L)?
           "YES" : "NO");

    unsigned int flags;
    int active;
    cuDevicePrimaryCtxGetState(device, &flags, &active);
    printf("cuDevicePrimaryCtxGetState flags = %d, active = %d\n", flags, active);

    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context. %d\n", err);
        cuCtxDestroy(context);
        exit(-1);
    }
	unsigned int version;
	cuCtxGetApiVersion(context, &version);
	printf("The API version = %d\n", version);
	//CUdevice tmp_device;
	cuCtxGetDevice(&device);
	CUsharedconfig config;
	cuCtxGetSharedMemConfig(&config);
	printf("cuCtxGetSharedMemConfig = %d\n", config);

	CUfunc_cache cache_config;
	cuCtxGetCacheConfig(&cache_config);
	printf("cuCtxGetCacheConfig = %d\n", cache_config);

	size_t value = 0;
	cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 10240);
	cuCtxGetLimit(&value, CU_LIMIT_STACK_SIZE);
	printf("CU_LIMIT_STACK_SIZE value = %d\n", value);

	CUcontext tmp_context;
	cuCtxGetCurrent(&tmp_context);
    cuCtxSynchronize();
   //std::cout << "Loading PTX kernel with driver\n" << ptx_kernel; 

    const unsigned int num_opts=2;
    CUjit_option options[num_opts];
    void *values[num_opts];
    options[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    values[0] = (void *)(int)10240;
    // set up pointer to the compilation log buffer
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    char clog[10240];
    values[1] = clog;


    int fd = open("moduledata.dat", O_CREAT | O_WRONLY | O_APPEND, 0600);

    std::ifstream ptxfile( argv[1] );
    std::stringstream buffer;
    buffer << ptxfile.rdbuf();

    std::string ptx_kernel = buffer.str();
    const char* p = ptx_kernel.c_str();
    write(fd, &p, sizeof(char*));
    long len = ptx_kernel.length();
    write(fd, &len, sizeof(long));
    write(fd, ptx_kernel.c_str(), len);

    ptxfile.close();

    close(fd);
    //err = cuModuleLoad(&module, module_file);
    err = cuModuleLoadData(&module, p);
    //err=cuModuleLoadDataEx(&module, p, num_opts,
    //                                options,(void **)values);   
    //sleep(5);
//    sleep(10);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error loading the module %s\n", module_file);
        cuCtxDestroy(context);
        exit(-1);
    }

//    err=cuModuleLoadDataEx(&module,ptx_kernel.c_str(),num_opts,
//                                    options,(void **)values);   


    printf("module addr %p, kernal_name %s, context %p\n", &module, kernel_name, &context);
    
    err = cuModuleGetFunction(&function, module, kernel_name);
    //sleep(10);
    printf("after sleep111 ....\n");
 

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        cuCtxDestroy(context);
        exit(-1);
    }

    //CUstream *pStream = new CUstream();
    //cuStreamCreate(pStream, 0);
    //stream = *pStream;
}

void finalizeCUDA()
{
    cuCtxDestroy(context);
}

void setupDeviceMemory(CUdeviceptr *d_a, CUdeviceptr *d_b, CUdeviceptr *d_c)
{
    checkCudaErrors( cuMemAlloc(d_a, sizeof(int) * N) );
    checkCudaErrors( cuMemAlloc(d_b, sizeof(int) * N) );
    checkCudaErrors( cuMemAlloc(d_c, sizeof(int) * N) );
    //checkCudaErrors( cuMemAllocManaged(d_a, sizeof(int) * N, CU_MEM_ATTACH_HOST));
	//checkCudaErrors( cuMemAllocManaged(d_b, sizeof(int) * N, CU_MEM_ATTACH_HOST));
	//checkCudaErrors( cuMemAllocManaged(d_c, sizeof(int) * N, CU_MEM_ATTACH_HOST));
	size_t len = sizeof(int) * N;
	CUdeviceptr base;
	cuMemGetAddressRange(&base, &len, *d_a);
	printf("memory start: %p.\n", &base);
}

void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
{
    printf("check the cuMemFree d_a result\n");
    checkCudaErrors( cuMemFree(d_a) );
	printf("check the cuMemFree d_b result\n");
    checkCudaErrors( cuMemFree(d_b) );
	printf("check the cuMemFree d_c result\n");
    checkCudaErrors( cuMemFree(d_c) );
}

void runKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
{
    printf("in runKernel ...skdjfsljdfkljsdlkf.\n");
    void *args[3] = { &d_a, &d_b, &d_c };

    CUcontext  tempcontext{0};

    //cuDevicePrimaryCtxRetain(&context, device);
//    checkCudaErrors( cuCtxGetDevice(context, &device) );
    //cuCtxCreate(&tempcontext, 0, device); 
    
    //cuCtxAttach(&tempcontext, 0);
    //cuModuleGetFunction(&function, module, kernel_name);
    // grid for kernel: <<<N, 1>>>
    checkCudaErrors( cuLaunchKernel(function, N, 1, 1,  // Nx1x1 blocks
                                    1, 1, 1,            // 1x1x1 threads
                                    0, 0, args, 0) );
}


void callback(CUstream stream, cudaError_enum status, void * data)
{ 
   int* p = (int *)data;
   printf("###################stream sync finished k = %d\n", *p);
}


void copyDataToDev(CUstream stream)
{
    // allocate memory
    int a[N], b[N], c[N];
    // initialize host arrays
    for (int i = 0; i < N; ++i) {
        a[i] = N - i;
        b[i] = i * i;
    }
    //sleep(10); 
    setupDeviceMemory(&d_a, &d_b, &d_c);
    // copy arrays to device
    //checkCudaErrors( cuMemcpyHtoD(d_a, a, sizeof(int) * N) ); 
    //checkCudaErrors( cuMemcpyHtoD(d_b, b, sizeof(int) * N) );
    checkCudaErrors( cuMemcpyHtoDAsync(d_a, a, sizeof(int) * N, stream) );
    checkCudaErrors( cuMemcpyHtoDAsync(d_b, b, sizeof(int) * N, stream) );
    printf("# Running the kernel...\n");
    runKernel(d_a, d_b, d_c);
    printf("# Kernel complete.\n");
    // copy results to host and report
    //    sleep(10);
    checkCudaErrors( cuMemcpyDtoHAsync(c, d_c, sizeof(int) * N, stream));
}

int main(int argc, char **argv)
{
    // initialize
    printf("- Initializing...\n");
    initCUDA(argc, argv);
    CUstream stream, p_stream;
	int priority = 1;
	unsigned int flags = 0;
    int minValue, maxValue;
    cuStreamCreate(&stream, 0);

    checkCudaErrors(cuStreamCreateWithPriority(&p_stream, 0, -2));
	CUresult ret = cuStreamQuery(stream);
	printf("The status of stream is: %d\n", ret);
    cuCtxGetStreamPriorityRange(&minValue, &maxValue);
    printf("minValue = %d, maxValue = %d\n", minValue, maxValue);
	checkCudaErrors(cuStreamGetFlags(stream, &flags));
	printf("The flags of stream is: %d\n", flags);
    checkCudaErrors((cuStreamGetPriority(stream, &priority)));
	printf("The priority of stream is: %d\n", priority);
	checkCudaErrors((cuStreamGetPriority(p_stream, &priority)));
	printf("The priority of p_stream is: %d\n", priority);
    //cuEventRecord(startEvent, stream);
    //cuEventSynchronize(startEvent); 

    CUevent startEvent, stopEvent;
    cuEventCreate(&startEvent,0);
    cuEventCreate(&stopEvent,0);
    cuEventRecord(startEvent, stream);
	cuStreamGetCtx(stream, &context);

    //CUevent stopEvent;
    //CUevent startEvent;
    //cuEventCreate(&startEvent,0); 
    copyDataToDev(stream);
    //cuEventCreate(&stopEvent,0);
    // for (int i = 0; i < N; ++i) {
    //     if (c[i] != a[i] + b[i])
    //         printf("* Error at array position %d: Expected %d, Got %d\n",
    //                i, a[i]+b[i], c[i]);
    // }
    cuEventRecord(stopEvent, stream);
	cuStreamWaitEvent(stream, stopEvent, 0);
	CUstreamCaptureStatus captureStatus;
	cuStreamIsCapturing(stream, &captureStatus);
	printf("The captureStatus of stream is: %d\n", captureStatus);

    cuEventDestroy(startEvent);
    cuEventDestroy(stopEvent);
    cuStreamSynchronize(stream);
    sleep(10);
    //cuStreamAddCallback(stream, callback, &priority, 0);
    printf("*** All checks complete\n");

    // finish
    printf("- Finalizing...\n");
    cuStreamDestroy(stream);
    releaseDeviceMemory(d_a, d_b, d_c);
    finalizeCUDA();
    printf("- Finalizing...OK\n");
    return 0;
}
