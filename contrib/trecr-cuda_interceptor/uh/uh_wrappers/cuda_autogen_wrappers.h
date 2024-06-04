#ifndef CUDA_AUTOGEN_WRAPPERS_H
#define CUDA_AUTOGEN_WRAPPERS_H


#include <cuda_runtime_api.h>
// #include <cublas.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cufft.h>
#include <nccl.h>

extern "C" cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc) __attribute__((weak));
extern "C" cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) __attribute__((weak));
#define cudaDestroyTextureObject(texObject) (cudaDestroyTextureObject ? cudaDestroyTextureObject(texObject) : 0)

extern "C" cudaError_t cudaBindTexture ( size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size) __attribute__((weak));
#define cudaBindTexture(offset, texref, devPtr, desc, size) (cudaBindTexture ? cudaBindTexture(offset, texref, devPtr, desc, size) : 0)

extern "C" cudaError_t cudaBindTexture2D ( size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t width, size_t height, size_t pitch ) __attribute__((weak));
#define cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch) (cudaBindTexture2D ? cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch) : 0)

extern "C" cudaError_t cudaBindTextureToArray ( const textureReference* texref, cudaArray_const_t array, const cudaChannelFormatDesc* desc ) __attribute__((weak));
#define cudaBindTextureToArray(texref, array, desc) (cudaBindTextureToArray ? cudaBindTextureToArray(texref, array, desc) : 0)

extern "C" cudaError_t cudaUnbindTexture(const struct textureReference * texref) __attribute__((weak));
#define cudaUnbindTexture(texref) (cudaUnbindTexture ? cudaUnbindTexture(texref) : 0)

extern "C" cudaChannelFormatDesc cudaCreateChannelDesc ( int  x, int  y, int  z, int  w, cudaChannelFormatKind f ) __attribute__((weak));
#define cudaCreateChannelDesc(x, y, z, w, f) (cudaCreateChannelDesc ? cudaCreateChannelDesc(x, y, z, w, f) : 0)

extern "C" cudaError_t cudaEventCreate(cudaEvent_t * event) __attribute__((weak));
#define cudaEventCreate(event) (cudaEventCreate ? cudaEventCreate(event) : 0)

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags) __attribute__((weak));
#define cudaEventCreateWithFlags(event, flags) (cudaEventCreateWithFlags ? cudaEventCreateWithFlags(event, flags) : 0)

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) __attribute__((weak));
#define cudaEventDestroy(event) (cudaEventDestroy ? cudaEventDestroy(event) : 0)

extern "C" cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end) __attribute__((weak));
#define cudaEventElapsedTime(ms, start, end) (cudaEventElapsedTime ? cudaEventElapsedTime(ms, start, end) : 0)

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) __attribute__((weak));
#define cudaEventQuery(event) (cudaEventQuery ? cudaEventQuery(event) : 0)

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) __attribute__((weak));
#define cudaEventRecord(event, stream) (cudaEventRecord ? cudaEventRecord(event, stream) : 0)

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) __attribute__((weak));
#define cudaEventSynchronize(event) (cudaEventSynchronize ? cudaEventSynchronize(event) : 0)

extern "C" cudaError_t cudaMalloc(void ** pointer, size_t size) __attribute__((weak));
#define cudaMalloc(pointer, size) (cudaMalloc ? cudaMalloc(pointer, size) : 0)

extern "C" cudaError_t cudaFree ( void * pointer ) __attribute__((weak));
#define cudaFree(pointer) (cudaFree ? cudaFree(pointer) : 0)

extern "C" cudaError_t cudaMallocArray(struct cudaArray ** array, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags) __attribute__((weak));
#define cudaMallocArray(array, desc, width, height, flags) (cudaMallocArray ? cudaMallocArray(array, desc, width, height, flags) : 0)

extern "C" cudaError_t cudaFreeArray(struct cudaArray * array) __attribute__((weak));
#define cudaFreeArray(array) (cudaFreeArray ? cudaFreeArray(array) : 0)

extern "C" cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int  flags ) __attribute__((weak));
#define cudaHostRegister(ptr, size, flags) (cudaHostRegister ? cudaHostRegister(ptr, size, flags) : 0)

extern "C" cudaError_t cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device ) __attribute__((weak));
#define cudaDeviceGetAttribute(value, attr, device) (cudaDeviceGetAttribute ? cudaDeviceGetAttribute(value, attr, device) : 0)

extern "C" cudaError_t cudaMallocHost ( void ** ptr , size_t size ) __attribute__((weak));
#define cudaMallocHost(ptr, size) (cudaMallocHost ? cudaMallocHost(ptr, size) : 0)

extern "C" cudaError_t cudaFreeHost ( void* ptr ) __attribute__((weak));
#define cudaFreeHost(ptr) (cudaFreeHost ? cudaFreeHost(ptr) : 0)

extern "C" cudaError_t cudaHostAlloc ( void ** ptr , size_t size , unsigned int flags ) __attribute__((weak));
#define cudaHostAlloc(ptr, size, flags) (cudaHostAlloc ? cudaHostAlloc(ptr, size, flags) : 0)

extern "C" cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height) __attribute__((weak));
#define cudaMallocPitch(devPtr, pitch, width, height) (cudaMallocPitch ? cudaMallocPitch(devPtr, pitch, width, height) : 0)

extern "C" cudaError_t cudaGetDevice(int * device) __attribute__((weak));
#define cudaGetDevice(device) (cudaGetDevice ? cudaGetDevice(device) : 0)

extern "C" cudaError_t cudaSetDevice(int device) __attribute__((weak));
#define cudaSetDevice(device) (cudaSetDevice ? cudaSetDevice(device) : 0)

extern "C" cudaError_t cudaDeviceGetLimit ( size_t* pValue, cudaLimit limit ) __attribute__((weak));
#define cudaDeviceGetLimit(pValue, limit) (cudaDeviceGetLimit ? cudaDeviceGetLimit(pValue, limit) : 0)

extern "C" cudaError_t cudaDeviceSetLimit ( cudaLimit limit, size_t value ) __attribute__((weak));
#define cudaDeviceSetLimit(limit, value) (cudaDeviceSetLimit ? cudaDeviceSetLimit(limit, value) : 0)

extern "C" cudaError_t cudaGetDeviceCount(int * count) __attribute__((weak));
#define cudaGetDeviceCount(count) (cudaGetDeviceCount ? cudaGetDeviceCount(count) : 0)

extern "C" cudaError_t cudaDeviceSetCacheConfig ( cudaFuncCache cacheConfig ) __attribute__((weak));
#define cudaDeviceSetCacheConfig(cacheConfig) (cudaDeviceSetCacheConfig ? cudaDeviceSetCacheConfig(cacheConfig) : 0)

extern "C" cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device ) __attribute__((weak));
#define cudaGetDeviceProperties(prop, device) (cudaGetDeviceProperties ? cudaGetDeviceProperties(prop, device) : 0)

extern "C" cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice) __attribute__((weak));
#define cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice) (cudaDeviceCanAccessPeer ? cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice) : 0)

extern "C" cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device) __attribute__((weak));
#define cudaDeviceGetPCIBusId(pciBusId, len, device) (cudaDeviceGetPCIBusId ? cudaDeviceGetPCIBusId(pciBusId, len, device) : 0)

extern "C" cudaError_t cudaDeviceReset() __attribute__((weak));
#define cudaDeviceReset() (cudaDeviceReset ? cudaDeviceReset() : 0)

extern "C" cudaError_t cudaDeviceSynchronize() __attribute__((weak));
#define cudaDeviceSynchronize() (cudaDeviceSynchronize ? cudaDeviceSynchronize() : 0)

extern "C" cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) __attribute__((weak));
#define cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream) (cudaLaunchKernel ? cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream) : 0)

extern "C" cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags) __attribute__((weak));
#define cudaMallocManaged(devPtr, size, flags) (cudaMallocManaged ? cudaMallocManaged(devPtr, size, flags) : 0)

extern "C" cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) __attribute__((weak));
#define cudaMemcpy(dst, src, count, kind) (cudaMemcpy ? cudaMemcpy(dst, src, count, kind) : 0)

extern "C" cudaError_t cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind ) __attribute__((weak));
#define cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind) (cudaMemcpy2D ? cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind) : 0)

extern "C" cudaError_t cudaMemcpyToArray (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind ) __attribute__((weak));
#define cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind) (cudaMemcpyToArray ? cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind) : 0)

extern "C" cudaError_t cudaMemcpyToSymbol ( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind) __attribute__((weak));
#define cudaMemcpyToSymbol(symbol, src, count, offset, kind) (cudaMemcpyToSymbol ? cudaMemcpyToSymbol(symbol, src, count, offset, kind) : 0)

extern "C" cudaError_t cudaMemcpyToSymbolAsync ( const void* symbol, const void* src, size_t count, size_t offset,  								 cudaMemcpyKind kind, cudaStream_t stream) __attribute__((weak));
#define cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream) (cudaMemcpyToSymbolAsync ? cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream) : 0)

extern "C" cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) __attribute__((weak));
#define cudaMemcpyAsync(dst, src, count, kind, stream) (cudaMemcpyAsync ? cudaMemcpyAsync(dst, src, count, kind, stream) : 0)

extern "C" cudaError_t cudaMemset(void * devPtr, int value, size_t count) __attribute__((weak));
#define cudaMemset(devPtr, value, count) (cudaMemset ? cudaMemset(devPtr, value, count) : 0)

extern "C" cudaError_t cudaMemset2D ( void* devPtr, size_t pitch, int value, size_t width, size_t height ) __attribute__((weak));
#define cudaMemset2D(devPtr, pitch, value, width, height) (cudaMemset2D ? cudaMemset2D(devPtr, pitch, value, width, height) : 0)

extern "C" cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream) __attribute__((weak));
#define cudaMemsetAsync(devPtr, value, count, stream) (cudaMemsetAsync ? cudaMemsetAsync(devPtr, value, count, stream) : 0)

extern "C" cudaError_t cudaMemGetInfo(size_t * free, size_t * total) __attribute__((weak));
#define cudaMemGetInfo(free, total) (cudaMemGetInfo ? cudaMemGetInfo(free, total) : 0)

extern "C" cudaError_t cudaMemAdvise(const void * devPtr, size_t count, enum cudaMemoryAdvise advice, int device) __attribute__((weak));
#define cudaMemAdvise(devPtr, count, advice, device) (cudaMemAdvise ? cudaMemAdvise(devPtr, count, advice, device) : 0)

extern "C" cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream) __attribute__((weak));
#define cudaMemPrefetchAsync(devPtr, count, dstDevice, stream) (cudaMemPrefetchAsync ? cudaMemPrefetchAsync(devPtr, count, dstDevice, stream) : 0)

extern "C" cudaError_t cudaStreamCreate(cudaStream_t * pStream) __attribute__((weak));
#define cudaStreamCreate(pStream) (cudaStreamCreate ? cudaStreamCreate(pStream) : 0)

extern "C" cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority ) __attribute__((weak));
#define cudaStreamCreateWithPriority(pStream, flags, priority) (cudaStreamCreateWithPriority ? cudaStreamCreateWithPriority(pStream, flags, priority) : 0)

extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int flags) __attribute__((weak));
#define cudaStreamCreateWithFlags(pStream, flags) (cudaStreamCreateWithFlags ? cudaStreamCreateWithFlags(pStream, flags) : 0)

extern "C" cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) __attribute__((weak));
#define cudaStreamIsCapturing(stream, pCaptureStatus) (cudaStreamIsCapturing ? cudaStreamIsCapturing(stream, pCaptureStatus) : 0)
extern "C" cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId) __attribute__((weak));
#define cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId) (cudaStreamGetCaptureInfo ? cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId) : 0)
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) __attribute__((weak));
#define cudaStreamDestroy(stream) (cudaStreamDestroy ? cudaStreamDestroy(stream) : 0)

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) __attribute__((weak));
#define cudaStreamSynchronize(stream) (cudaStreamSynchronize ? cudaStreamSynchronize(stream) : 0)

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) __attribute__((weak));
#define cudaStreamWaitEvent(stream, event, flags) (cudaStreamWaitEvent ? cudaStreamWaitEvent(stream, event, flags) : 0)

extern "C" cudaError_t cudaThreadSynchronize () __attribute__((weak));
#define cudaThreadSynchronize() (cudaThreadSynchronize ? cudaThreadSynchronize() : 0)

extern "C" cudaError_t cudaThreadExit () __attribute__((weak));
#define cudaThreadExit() (cudaThreadExit ? cudaThreadExit() : 0)

extern "C" cudaError_t cudaPointerGetAttributes ( cudaPointerAttributes* attributes, const void* ptr ) __attribute__((weak));
#define cudaPointerGetAttributes(attributes, ptr) (cudaPointerGetAttributes ? cudaPointerGetAttributes(attributes, ptr) : 0)

extern "C" const char* cudaGetErrorString ( cudaError_t error ) __attribute__((weak));
#define cudaGetErrorString(error) (cudaGetErrorString ? cudaGetErrorString(error) : 0)

extern "C" const char* cudaGetErrorName ( cudaError_t error ) __attribute__((weak));
#define cudaGetErrorName(error) (cudaGetErrorName ? cudaGetErrorName(error) : 0)

extern "C" cudaError_t cudaGetLastError() __attribute__((weak));
#define cudaGetLastError() (cudaGetLastError ? cudaGetLastError() : 0)

extern "C" cudaError_t cudaPeekAtLastError() __attribute__((weak));
#define cudaPeekAtLastError() (cudaPeekAtLastError ? cudaPeekAtLastError() : 0)

extern "C" cudaError_t cudaFuncSetCacheConfig ( const void* func, cudaFuncCache cacheConfig ) __attribute__((weak));
#define cudaFuncSetCacheConfig(func, cacheConfig) (cudaFuncSetCacheConfig ? cudaFuncSetCacheConfig(func, cacheConfig) : 0)

extern "C" char __cudaInitModule(void **fatCubinHandle) __attribute__((weak));
#define __cudaInitModule(fatCubinHandle) (__cudaInitModule ? __cudaInitModule(fatCubinHandle) : 0)

extern "C" cudaError_t __cudaPopCallConfiguration( dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream ) __attribute__((weak));
#define __cudaPopCallConfiguration(gridDim, blockDim, sharedMem, stream) (__cudaPopCallConfiguration ? __cudaPopCallConfiguration(gridDim, blockDim, sharedMem, stream) : 0)

extern "C" unsigned int __cudaPushCallConfiguration( dim3 gridDim, dim3 blockDim, size_t sharedMem, void * stream ) __attribute__((weak));
#define __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream) (__cudaPushCallConfiguration ? __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream) : 0)

extern "C" void** __cudaRegisterFatBinary(void *fatCubin) __attribute__((weak));
#define __cudaRegisterFatBinary(fatCubin) (__cudaRegisterFatBinary ? __cudaRegisterFatBinary(fatCubin) : 0)

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) __attribute__((weak));
#define __cudaUnregisterFatBinary(fatCubinHandle) (__cudaUnregisterFatBinary ? __cudaUnregisterFatBinary(fatCubinHandle) : 0)

extern "C" void __cudaRegisterFunction( void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize ) __attribute__((weak));
#define __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize) (__cudaRegisterFunction ? __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize) : 0)

extern "C" void __cudaRegisterManagedVar( void **fatCubinHandle, void **hostVarPtrAddress, char  *deviceAddress, const char  *deviceName, int    ext, size_t size, int    constant, int    global ) __attribute__((weak));
#define __cudaRegisterManagedVar(fatCubinHandle, hostVarPtrAddress, deviceAddress, deviceName, ext, size, constant, global) (__cudaRegisterManagedVar ? __cudaRegisterManagedVar(fatCubinHandle, hostVarPtrAddress, deviceAddress, deviceName, ext, size, constant, global) : 0)

extern "C" void __cudaRegisterTexture( void  **fatCubinHandle, const struct textureReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext ) __attribute__((weak));
#define __cudaRegisterTexture(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext) (__cudaRegisterTexture ? __cudaRegisterTexture(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext) : 0)

extern "C" void __cudaRegisterSurface( void **fatCubinHandle, const struct surfaceReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext ) __attribute__((weak));
#define __cudaRegisterSurface(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext) (__cudaRegisterSurface ? __cudaRegisterSurface(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext) : 0)

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char  *deviceAddress, const char  *deviceName, int ext, size_t size, int constant, int global) __attribute__((weak));
#define __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global) (__cudaRegisterVar ? __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global) : 0)

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags ) __attribute__((weak));
#define cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags) (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ? cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags) : 0)

extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) __attribute__((weak));
#define cudaFuncGetAttributes(attr, func) (cudaFuncGetAttributes ? cudaFuncGetAttributes(attr, func) : 0)

extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t * handle) __attribute__((weak));
#define cublasCreate_v2(handle) (cublasCreate_v2 ? cublasCreate_v2(handle) : 0)

extern "C" cublasStatus_t cublasSgemm_v2 (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, /* host or device pointer */ const float *A, int lda, const float *B, int ldb, const float *beta, /* host or device pointer */ float *C, int ldc) __attribute__((weak));
#define cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasSgemm_v2 ? cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)
extern "C" cublasStatus_t cublasSgemmStridedBatched (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,  /* host or device pointer */ const float *A, int lda, long long int strideA,   /* purposely signed */ const float *B, int ldb, long long int strideB, const float *beta,   /* host or device pointer */ float *C, int ldc, long long int strideC, int batchCount) __attribute__((weak));
#define cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount) (cublasSgemmStridedBatched ? cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount) : 0)
extern "C" cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle) __attribute__((weak));
#define cublasLtCreate(lightHandle) (cublasLtCreate ? cublasLtCreate(lightHandle) : 0)
extern "C" cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) __attribute__((weak));
#define cublasLtDestroy(lightHandle) (cublasLtDestroy ? cublasLtDestroy(lightHandle) : 0)
extern "C" cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha, /* host or device pointer */ const void *A, cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta, /* host or device pointer */ const void *C, cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream) __attribute__((weak));
#define cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream) (cublasLtMatmul ? cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream) : 0)
extern "C" cublasStatus_t cublasLtMatmulAlgoGetHeuristic( cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int *returnAlgoCount) __attribute__((weak));
#define cublasLtMatmulAlgoGetHeuristic(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount) (cublasLtMatmulAlgoGetHeuristic ? cublasLtMatmulAlgoGetHeuristic(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount) : 0)
extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) __attribute__((weak));
#define cublasSetStream_v2(handle, streamId) (cublasSetStream_v2 ? cublasSetStream_v2(handle, streamId) : 0)
extern "C" cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) __attribute__((weak));
#define cublasSetMathMode(handle, mode) (cublasSetMathMode ? cublasSetMathMode(handle, mode) : 0)
extern "C" cublasStatus_t   cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) __attribute__((weak));
#define cublasGetMathMode(handle, mode) (cublasGetMathMode ? cublasGetMathMode(handle, mode) : 0)

extern "C" cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double * x, int incx, const double * y, int incy, double * result) __attribute__((weak));
#define cublasDdot_v2(handle, n, x, incx, y, incy, result) (cublasDdot_v2 ? cublasDdot_v2(handle, n, x, incx, y, incy, result) : 0)

extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) __attribute__((weak));
#define cublasDestroy_v2(handle) (cublasDestroy_v2 ? cublasDestroy_v2(handle) : 0)

extern "C" cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double * alpha, const double * x, int incx, double * y, int incy) __attribute__((weak));
#define cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy) (cublasDaxpy_v2 ? cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy) : 0)

extern "C" cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) __attribute__((weak));
#define cublasDasum_v2(handle, n, x, incx, result) (cublasDasum_v2 ? cublasDasum_v2(handle, n, x, incx, result) : 0)

extern "C" cublasStatus_t cublasDgemm_v2 (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) __attribute__((weak));
#define cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasDgemm_v2 ? cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" cublasStatus_t cublasDgemv_v2 (cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) __attribute__((weak));
#define cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy) (cublasDgemv_v2 ? cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) __attribute__((weak));
#define cublasDnrm2_v2(handle, n, x, incx, result) (cublasDnrm2_v2 ? cublasDnrm2_v2(handle, n, x, incx, result) : 0)

extern "C" cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) __attribute__((weak));
#define cublasDscal_v2(handle, n, alpha, x, incx) (cublasDscal_v2 ? cublasDscal_v2(handle, n, alpha, x, incx) : 0)

extern "C" cublasStatus_t cublasDswap_v2 (cublasHandle_t handle, int n, double *x, int incx, double *y, int incy) __attribute__((weak));
#define cublasDswap_v2(handle, n, x, incx, y, incy) (cublasDswap_v2 ? cublasDswap_v2(handle, n, x, incx, y, incy) : 0)

extern "C" cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) __attribute__((weak));
#define cublasIdamax_v2(handle, n, x, incx, result) (cublasIdamax_v2 ? cublasIdamax_v2(handle, n, x, incx, result) : 0)

extern "C" cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) __attribute__((weak));
#define cublasSscal_v2(handle, n, alpha, x, incx) (cublasSscal_v2 ? cublasSscal_v2(handle, n, x, incx, result) : 0)

extern "C" cublasStatus_t cublasSetMatrix (int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb) __attribute__((weak));
#define cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb) (cublasSetMatrix ? cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb) : 0)

extern "C" cublasStatus_t cublasGetMatrix (int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb) __attribute__((weak));
#define cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb) (cublasGetMatrix ? cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb) : 0)

extern "C" cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream) __attribute__((weak));
#define cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) (cublasSetMatrixAsync ? cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) : 0)

extern "C" cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream) __attribute__((weak));
#define cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) (cublasGetMatrixAsync ? cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) : 0)

extern "C" cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) __attribute__((weak));
#define cublasSetVector(n, elemSize, x, incx, y, incy) (cublasSetVector ? cublasSetVector(n, elemSize, x, incx, y, incy) : 0)

extern "C" cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) __attribute__((weak));
#define cublasGetVector(n, elemSize, x, incx, y, incy) (cublasGetVector ? cublasGetVector(n, elemSize, x, incx, y, incy) : 0)

extern "C" cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream) __attribute__((weak));
#define cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream) (cublasSetVectorAsync ? cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream) : 0)

extern "C" cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream) __attribute__((weak));
#define cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream) (cublasGetVectorAsync ? cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream) : 0)

/***********by tian01.liu for llama, 2024.01.08 **************/
extern "C" cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                                   int n, int k, const void* alpha, /* host or device pointer */const void* A, cudaDataType Atype,
                                                   int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, /* host or device pointer */
                                                   void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo)__attribute__((weak));
#define cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo):(cublasGemmEx ? cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo) : 0)  


extern "C" cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,int m,int n, int k,
                        const void* alpha, /* host or device pointer */const void* A,cudaDataType Atype,int lda,long long int strideA, /* purposely signed */
                        const void* B,cudaDataType Btype, int ldb,long long int strideB,const void* beta, /* host or device pointer */void* C,
                        cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo)__attribute__((weak));
#define cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo):(cublasGemmStridedBatchedEx ? cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo) : 0)  


extern "C" cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) __attribute__((weak));
#define cusparseCreate(handle) (cusparseCreate ? cusparseCreate(handle) : 0)

extern "C" cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) __attribute__((weak));
#define cusparseSetStream(handle, streamId) (cusparseSetStream ? cusparseSetStream(handle, streamId) : 0)

extern "C" cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t *descrA) __attribute__((weak));
#define cusparseCreateMatDescr(descrA) (cusparseCreateMatDescr ? cusparseCreateMatDescr(descrA) : 0)

extern "C" cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) __attribute__((weak));
#define cusparseSetMatType(descrA, type) (cusparseSetMatType ? cusparseSetMatType(descrA, type) : 0)

extern "C" cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base) __attribute__((weak));
#define cusparseSetMatIndexBase(descrA, base) (cusparseSetMatIndexBase ? cusparseSetMatIndexBase(descrA, base) : 0)

extern "C" cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) __attribute__((weak));
#define cusparseDestroy(handle) (cusparseDestroy ? cusparseDestroy(handle) : 0)

extern "C" cusparseStatus_t cusparseDestroyMatDescr (cusparseMatDescr_t descrA) __attribute__((weak));
#define cusparseDestroyMatDescr(descrA) (cusparseDestroyMatDescr ? cusparseDestroyMatDescr(descrA) : 0)

extern "C" cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA) __attribute__((weak));
#define cusparseGetMatType(descrA) (cusparseGetMatType ? cusparseGetMatType(descrA) : 0)

extern "C" cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) __attribute__((weak));
#define cusparseSetMatFillMode(descrA, fillMode) (cusparseSetMatFillMode ? cusparseSetMatFillMode(descrA, fillMode) : 0)

extern "C" cusparseFillMode_t cusparseGetMatFillMode(const cusparseMatDescr_t descrA) __attribute__((weak));
#define cusparseGetMatFillMode(descrA) (cusparseGetMatFillMode ? cusparseGetMatFillMode(descrA) : 0)

extern "C" cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t  descrA, cusparseDiagType_t diagType) __attribute__((weak));
#define cusparseSetMatDiagType(descrA, diagType) (cusparseSetMatDiagType ? cusparseSetMatDiagType(descrA, diagType) : 0)

extern "C" cusparseDiagType_t cusparseGetMatDiagType(const cusparseMatDescr_t descrA) __attribute__((weak));
#define cusparseGetMatDiagType(descrA) (cusparseGetMatDiagType ? cusparseGetMatDiagType(descrA) : 0)

extern "C" cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) __attribute__((weak));
#define cusparseGetMatIndexBase(descrA) (cusparseGetMatIndexBase ? cusparseGetMatIndexBase(descrA) : 0)

extern "C" cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) __attribute__((weak));
#define cusparseSetPointerMode(handle, mode) (cusparseSetPointerMode ? cusparseSetPointerMode(handle, mode) : 0)

extern "C" cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle) __attribute__((weak));
#define cusolverDnCreate(handle) (cusolverDnCreate ? cusolverDnCreate(handle) : 0)

extern "C" cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) __attribute__((weak));
#define cusolverDnDestroy(handle) (cusolverDnDestroy ? cusolverDnDestroy(handle) : 0)

extern "C" cusolverStatus_t cusolverDnSetStream (cusolverDnHandle_t handle, cudaStream_t streamId) __attribute__((weak));
#define cusolverDnSetStream(handle, streamId) (cusolverDnSetStream ? cusolverDnSetStream(handle, streamId) : 0)

extern "C" cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId) __attribute__((weak));
#define cusolverDnGetStream(handle, streamId) (cusolverDnGetStream ? cusolverDnGetStream(handle, streamId) : 0)

extern "C" cusolverStatus_t cusolverDnDgetrf_bufferSize( cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork ) __attribute__((weak));
#define cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork) (cusolverDnDgetrf_bufferSize ? cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork) : 0)

extern "C" cusolverStatus_t cusolverDnDgetrf( cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo ) __attribute__((weak));
#define cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo) (cusolverDnDgetrf ? cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo) : 0)

extern "C" cusolverStatus_t cusolverDnDgetrs( cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, int lda, const int *devIpiv, double *B, int ldb, int *devInfo ) __attribute__((weak));
#define cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo) (cusolverDnDgetrs ? cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo) : 0)

extern "C" cusolverStatus_t cusolverDnDpotrf_bufferSize( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *Lwork ) __attribute__((weak));
#define cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork) (cusolverDnDpotrf_bufferSize ? cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork) : 0)

extern "C" cusolverStatus_t cusolverDnDpotrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, double *Workspace, int Lwork, int *devInfo ) __attribute__((weak));
#define cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo) (cusolverDnDpotrf ? cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo) : 0)

extern "C" cusolverStatus_t cusolverDnDpotrs( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo) __attribute__((weak));
#define cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) (cusolverDnDpotrs ? cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) : 0)

extern "C" CUresult cuInit ( unsigned int  Flags ) __attribute__((weak));
#define cuInit(Flags) (cuInit ? cuInit(Flags) : 0)

extern "C" CUresult cuDriverGetVersion ( int* driverVersion ) __attribute__((weak));
#define cuDriverGetVersion(driverVersion) (cuDriverGetVersion ? cuDriverGetVersion(driverVersion) : 0)

extern "C" CUresult cuDeviceGet ( CUdevice* device, int  ordinal ) __attribute__((weak));
#define cuDeviceGet(device, ordinal) (cuDeviceGet ? cuDeviceGet(device, ordinal) : 0)

extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) __attribute__((weak));
#define cuGetProcAddress(symbol, pfn, cudaVersion, flags) (cuGetProcAddress ? cuGetProcAddress(symbol, pfn, cudaVersion, flags) : 0)
extern "C" CUresult cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev ) __attribute__((weak));
#define cuDeviceGetAttribute(pi, attrib, dev) (cuDeviceGetAttribute ? cuDeviceGetAttribute(pi, attrib, dev) : 0)

extern "C" CUresult cuDeviceGetCount ( int* count ) __attribute__((weak));
#define cuDeviceGetCount(count) (cuDeviceGetCount ? cuDeviceGetCount(count) : 0)

extern "C" CUresult cuDeviceGetName ( char* name, int  len, CUdevice dev ) __attribute__((weak));
#define cuDeviceGetName(name, len, dev) (cuDeviceGetName ? cuDeviceGetName(name, len, dev) : 0)

extern "C" CUresult cuDeviceGetUuid ( CUuuid* uuid, CUdevice dev ) __attribute__((weak));
#define cuDeviceGetUuid(uuid, dev) (cuDeviceGetUuid ? cuDeviceGetUuid(uuid, dev) : 0)

extern "C" CUresult cuDeviceTotalMem_v2 ( size_t* bytes, CUdevice dev ) __attribute__((weak));
#define cuDeviceTotalMem_v2(bytes, dev) (cuDeviceTotalMem_v2 ? cuDeviceTotalMem_v2(bytes, dev) : 0)

extern "C" CUresult cuDeviceComputeCapability ( int* major, int* minor, CUdevice dev ) __attribute__((weak));
#define cuDeviceComputeCapability(major, minor, dev) (cuDeviceComputeCapability ? cuDeviceComputeCapability(major, minor, dev) : 0)

extern "C" CUresult cuDeviceGetProperties ( CUdevprop* prop, CUdevice dev ) __attribute__((weak));
#define cuDeviceGetProperties(prop, dev) (cuDeviceGetProperties ? cuDeviceGetProperties(prop, dev) : 0)

extern "C" CUresult cuDevicePrimaryCtxGetState ( CUdevice dev, unsigned int* flags, int* active ) __attribute__((weak));
#define cuDevicePrimaryCtxGetState(dev, flags, active) (cuDevicePrimaryCtxGetState ? cuDevicePrimaryCtxGetState(dev, flags, active) : 0)
extern "C" CUresult cuDevicePrimaryCtxRelease_v2 (CUdevice dev ) __attribute__((weak));
#define cuDevicePrimaryCtxRelease_v2(dev) (cuDevicePrimaryCtxRelease_v2 ? cuDevicePrimaryCtxRelease_v2(dev) : 0)

extern "C" CUresult cuDevicePrimaryCtxReset_v2 (CUdevice dev ) __attribute__((weak));
#define cuDevicePrimaryCtxReset_v2(dev) (cuDevicePrimaryCtxReset_v2 ? cuDevicePrimaryCtxReset_v2(dev) : 0)

extern "C" CUresult cuDevicePrimaryCtxRetain ( CUcontext* pctx, CUdevice dev ) __attribute__((weak));
#define cuDevicePrimaryCtxRetain(pctx, dev) (cuDevicePrimaryCtxRetain ? cuDevicePrimaryCtxRetain(pctx, dev) : 0)

extern "C" CUresult cuCtxCreate_v2 ( CUcontext* pctx, unsigned int  flags, CUdevice dev ) __attribute__((weak));
#define cuCtxCreate_v2(pctx, flags, dev) (cuCtxCreate_v2 ? cuCtxCreate_v2(pctx, flags, dev) : 0)

extern "C" CUresult cuCtxDestroy_v2 ( CUcontext ctx ) __attribute__((weak));
#define cuCtxDestroy_v2(ctx) (cuCtxDestroy_v2 ? cuCtxDestroy_v2(ctx) : 0)

extern "C" CUresult cuCtxGetApiVersion ( CUcontext ctx, unsigned int* version ) __attribute__((weak));
#define cuCtxGetApiVersion(ctx, version) (cuCtxGetApiVersion ? cuCtxGetApiVersion(ctx, version) : 0)

extern "C" CUresult cuCtxGetCacheConfig ( CUfunc_cache* pconfig ) __attribute__((weak));
#define cuCtxGetCacheConfig(pconfig) (cuCtxGetCacheConfig ? cuCtxGetCacheConfig(pconfig) : 0)

extern "C" CUresult cuCtxGetCurrent ( CUcontext* pctx ) __attribute__((weak));
#define cuCtxGetCurrent(pctx) (cuCtxGetCurrent ? cuCtxGetCurrent(pctx) : 0)

extern "C" CUresult cuCtxGetDevice ( CUdevice* device ) __attribute__((weak));
#define cuCtxGetDevice(device) (cuCtxGetDevice ? cuCtxGetDevice(device) : 0)

extern "C" CUresult cuCtxGetFlags ( unsigned int* flags ) __attribute__((weak));
#define cuCtxGetFlags(flags) (cuCtxGetFlags ? cuCtxGetFlags(flags) : 0)

extern "C" CUresult cuCtxGetLimit ( size_t* pvalue, CUlimit limit ) __attribute__((weak));
#define cuCtxGetLimit(pvalue, limit) (cuCtxGetLimit ? cuCtxGetLimit(pvalue, limit) : 0)

extern "C" CUresult cuCtxGetSharedMemConfig ( CUsharedconfig* pConfig ) __attribute__((weak));
#define cuCtxGetSharedMemConfig(pConfig) (cuCtxGetSharedMemConfig ? cuCtxGetSharedMemConfig(pConfig) : 0)

extern "C" CUresult cuCtxGetStreamPriorityRange ( int* leastPriority, int* greatestPriority ) __attribute__((weak));
#define cuCtxGetStreamPriorityRange(leastPriority, greatestPriority) (cuCtxGetStreamPriorityRange ? cuCtxGetStreamPriorityRange(leastPriority, greatestPriority) : 0)

extern "C" CUresult cuCtxPopCurrent_v2 ( CUcontext* pctx ) __attribute__((weak));
#define cuCtxPopCurrent_v2(pctx) (cuCtxPopCurrent_v2 ? cuCtxPopCurrent_v2(pctx) : 0)

extern "C" CUresult cuCtxPushCurrent_v2 ( CUcontext ctx ) __attribute__((weak));
#define cuCtxPushCurrent_v2(ctx) (cuCtxPushCurrent_v2 ? cuCtxPushCurrent_v2(ctx) : 0)

extern "C" CUresult cuCtxSetCacheConfig ( CUfunc_cache config ) __attribute__((weak));
#define cuCtxSetCacheConfig(config) (cuCtxSetCacheConfig ? cuCtxSetCacheConfig(config) : 0)

extern "C" CUresult cuCtxSetCurrent ( CUcontext ctx ) __attribute__((weak));
#define cuCtxSetCurrent(ctx) (cuCtxSetCurrent ? cuCtxSetCurrent(ctx) : 0)

extern "C" CUresult cuCtxSetLimit ( CUlimit limit, size_t value ) __attribute__((weak));
#define cuCtxSetLimit(limit, value) (cuCtxSetLimit ? cuCtxSetLimit(limit, value) : 0)

extern "C" CUresult cuCtxSetSharedMemConfig ( CUsharedconfig config ) __attribute__((weak));
#define cuCtxSetSharedMemConfig(config) (cuCtxSetSharedMemConfig ? cuCtxSetSharedMemConfig(config) : 0)

extern "C" CUresult cuCtxSynchronize () __attribute__((weak));
#define cuCtxSynchronize() (cuCtxSynchronize ? cuCtxSynchronize() : 0)

extern "C" CUresult cuCtxAttach ( CUcontext* pctx, unsigned int  flags ) __attribute__((weak));
#define cuCtxAttach(pctx, flags) (cuCtxAttach ? cuCtxAttach(pctx, flags) : 0)

extern "C" CUresult cuCtxDetach ( CUcontext ctx ) __attribute__((weak));
#define cuCtxDetach(ctx) (cuCtxDetach ? cuCtxDetach(ctx) : 0)

extern "C" CUresult cuLinkAddData_v2 ( CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues ) __attribute__((weak));
#define cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues) (cuLinkAddData_v2 ? cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues) : 0)

extern "C" CUresult cuLinkAddFile_v2 ( CUlinkState state, CUjitInputType type, const char* path, unsigned int  numOptions, CUjit_option* options, void** optionValues ) __attribute__((weak));
#define cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues) (cuLinkAddFile_v2 ? cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues) : 0)

extern "C" CUresult cuLinkComplete ( CUlinkState state, void** cubinOut, size_t* sizeOut ) __attribute__((weak));
#define cuLinkComplete(state, cubinOut, sizeOut) (cuLinkComplete ? cuLinkComplete(state, cubinOut, sizeOut) : 0)

extern "C" CUresult cuLinkCreate_v2 ( unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut ) __attribute__((weak));
#define cuLinkCreate_v2(numOptions, options, optionValues, stateOut) (cuLinkCreate_v2 ? cuLinkCreate_v2(numOptions, options, optionValues, stateOut) : 0)

extern "C" CUresult cuLinkDestroy ( CUlinkState state ) __attribute__((weak));
#define cuLinkDestroy(state) (cuLinkDestroy ? cuLinkDestroy(state) : 0)

extern "C" CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ) __attribute__((weak));
#define cuModuleGetFunction(hfunc, hmod, name) (cuModuleGetFunction ? cuModuleGetFunction(hfunc, hmod, name) : 0)

extern "C" CUresult cuModuleGetGlobal_v2 ( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name ) __attribute__((weak));
#define cuModuleGetGlobal_v2(dptr, bytes, hmod, name) (cuModuleGetGlobal_v2 ? cuModuleGetGlobal_v2(dptr, bytes, hmod, name) : 0)

extern "C" CUresult cuModuleGetSurfRef ( CUsurfref* pSurfRef, CUmodule hmod, const char* name ) __attribute__((weak));
#define cuModuleGetSurfRef(pSurfRef, hmod, name) (cuModuleGetSurfRef ? cuModuleGetSurfRef(pSurfRef, hmod, name) : 0)

extern "C" CUresult cuModuleGetTexRef ( CUtexref* pTexRef, CUmodule hmod, const char* name ) __attribute__((weak));
#define cuModuleGetTexRef(pTexRef, hmod, name) (cuModuleGetTexRef ? cuModuleGetTexRef(pTexRef, hmod, name) : 0)

extern "C" CUresult cuModuleLoad ( CUmodule* module, const char* fname ) __attribute__((weak));
#define cuModuleLoad(module, fname) (cuModuleLoad ? cuModuleLoad(module, fname) : 0)

extern "C" CUresult cuModuleLoadData ( CUmodule* module, const void* image ) __attribute__((weak));
#define cuModuleLoadData(module, image) (cuModuleLoadData ? cuModuleLoadData(module, image) : 0)

extern "C" CUresult cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues ) __attribute__((weak));
#define cuModuleLoadDataEx(module, image, numOptions, options, optionValues) (cuModuleLoadDataEx ? cuModuleLoadDataEx(module, image, numOptions, options, optionValues) : 0)

extern "C" CUresult cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin ) __attribute__((weak));
#define cuModuleLoadFatBinary(module, fatCubin) (cuModuleLoadFatBinary ? cuModuleLoadFatBinary(module, fatCubin) : 0)

extern "C" CUresult cuModuleUnload ( CUmodule hmod ) __attribute__((weak));
#define cuModuleUnload(hmod) (cuModuleUnload ? cuModuleUnload(hmod) : 0)

extern "C" CUresult cuArray3DCreate_v2 ( CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray ) __attribute__((weak));
#define cuArray3DCreate_v2(pHandle, pAllocateArray) (cuArray3DCreate_v2 ? cuArray3DCreate_v2(pHandle, pAllocateArray) : 0)

extern "C" CUresult cuArray3DGetDescriptor_v2 ( CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray ) __attribute__((weak));
#define cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray) (cuArray3DGetDescriptor_v2 ? cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray) : 0)

extern "C" CUresult cuArrayCreate_v2 ( CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray ) __attribute__((weak));
#define cuArrayCreate_v2(pHandle, pAllocateArray) (cuArrayCreate_v2 ? cuArrayCreate_v2(pHandle, pAllocateArray) : 0)

extern "C" CUresult cuArrayDestroy ( CUarray hArray ) __attribute__((weak));
#define cuArrayDestroy(hArray) (cuArrayDestroy ? cuArrayDestroy(hArray) : 0)

extern "C" CUresult cuArrayGetDescriptor_v2 ( CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray ) __attribute__((weak));
#define cuArrayGetDescriptor_v2(pArrayDescriptor, hArray) (cuArrayGetDescriptor_v2 ? cuArrayGetDescriptor_v2(pArrayDescriptor, hArray) : 0)

extern "C" CUresult cuDeviceGetByPCIBusId ( CUdevice* dev, const char* pciBusId ) __attribute__((weak));
#define cuDeviceGetByPCIBusId(dev, pciBusId) (cuDeviceGetByPCIBusId ? cuDeviceGetByPCIBusId(dev, pciBusId) : 0)

extern "C" CUresult cuDeviceGetPCIBusId ( char* pciBusId, int  len, CUdevice dev ) __attribute__((weak));
#define cuDeviceGetPCIBusId(pciBusId, len, dev) (cuDeviceGetPCIBusId ? cuDeviceGetPCIBusId(pciBusId, len, dev) : 0)

extern "C" CUresult cuIpcCloseMemHandle ( CUdeviceptr dptr ) __attribute__((weak));
#define cuIpcCloseMemHandle(dptr) (cuIpcCloseMemHandle ? cuIpcCloseMemHandle(dptr) : 0)

extern "C" CUresult cuIpcGetEventHandle ( CUipcEventHandle* pHandle, CUevent event ) __attribute__((weak));
#define cuIpcGetEventHandle(pHandle, event) (cuIpcGetEventHandle ? cuIpcGetEventHandle(pHandle, event) : 0)

extern "C" CUresult cuIpcGetMemHandle ( CUipcMemHandle* pHandle, CUdeviceptr dptr ) __attribute__((weak));
#define cuIpcGetMemHandle(pHandle, dptr) (cuIpcGetMemHandle ? cuIpcGetMemHandle(pHandle, dptr) : 0)

extern "C" CUresult cuIpcOpenEventHandle ( CUevent* phEvent, CUipcEventHandle handle ) __attribute__((weak));
#define cuIpcOpenEventHandle(phEvent, handle) (cuIpcOpenEventHandle ? cuIpcOpenEventHandle(phEvent, handle) : 0)

extern "C" CUresult cuMemAlloc_v2 ( CUdeviceptr* dptr, size_t bytesize ) __attribute__((weak));
#define cuMemAlloc_v2(dptr, bytesize) (cuMemAlloc_v2 ? cuMemAlloc_v2(dptr, bytesize) : 0)

extern "C" CUresult cuMemAllocHost_v2 ( void** pp, size_t bytesize ) __attribute__((weak));
#define cuMemAllocHost_v2(pp, bytesize) (cuMemAllocHost_v2 ? cuMemAllocHost_v2(pp, bytesize) : 0)

extern "C" CUresult cuMemAllocManaged ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags ) __attribute__((weak));
#define cuMemAllocManaged(dptr, bytesize, flags) (cuMemAllocManaged ? cuMemAllocManaged(dptr, bytesize, flags) : 0)

extern "C" CUresult cuMemAllocPitch_v2 ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes ) __attribute__((weak));
#define cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes) (cuMemAllocPitch_v2 ? cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes) : 0)

extern "C" CUresult cuMemFree_v2 ( CUdeviceptr dptr ) __attribute__((weak));
#define cuMemFree_v2(dptr) (cuMemFree_v2 ? cuMemFree_v2(dptr) : 0)

extern "C" CUresult cuMemFreeHost ( void* p ) __attribute__((weak));
#define cuMemFreeHost(p) (cuMemFreeHost ? cuMemFreeHost(p) : 0)

extern "C" CUresult cuMemGetAddressRange_v2 ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr ) __attribute__((weak));
#define cuMemGetAddressRange_v2(pbase, psize, dptr) (cuMemGetAddressRange_v2 ? cuMemGetAddressRange_v2(pbase, psize, dptr) : 0)

extern "C" CUresult cuMemGetInfo_v2 ( size_t* free, size_t* total ) __attribute__((weak));
#define cuMemGetInfo_v2(free, total) (cuMemGetInfo_v2 ? cuMemGetInfo_v2(free, total) : 0)

extern "C" CUresult cuMemHostAlloc ( void** pp, size_t bytesize, unsigned int  Flags ) __attribute__((weak));
#define cuMemHostAlloc(pp, bytesize, Flags) (cuMemHostAlloc ? cuMemHostAlloc(pp, bytesize, Flags) : 0)

extern "C" CUresult cuMemHostGetDevicePointer_v2 ( CUdeviceptr* pdptr, void* p, unsigned int  Flags ) __attribute__((weak));
#define cuMemHostGetDevicePointer_v2(pdptr, p, Flags) (cuMemHostGetDevicePointer_v2 ? cuMemHostGetDevicePointer_v2(pdptr, p, Flags) : 0)

extern "C" CUresult cuMemHostGetFlags ( unsigned int* pFlags, void* p ) __attribute__((weak));
#define cuMemHostGetFlags(pFlags, p) (cuMemHostGetFlags ? cuMemHostGetFlags(pFlags, p) : 0)

extern "C" CUresult cuMemHostRegister_v2 ( void* p, size_t bytesize, unsigned int  Flags ) __attribute__((weak));
#define cuMemHostRegister_v2(p, bytesize, Flags) (cuMemHostRegister_v2 ? cuMemHostRegister_v2(p, bytesize, Flags) : 0)

extern "C" CUresult cuMemHostUnregister ( void* p ) __attribute__((weak));
#define cuMemHostUnregister(p) (cuMemHostUnregister ? cuMemHostUnregister(p) : 0)

extern "C" CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount ) __attribute__((weak));
#define cuMemcpy(dst, src, ByteCount) (cuMemcpy ? cuMemcpy(dst, src, ByteCount) : 0)

extern "C" CUresult cuMemcpy2D_v2 ( const CUDA_MEMCPY2D* pCopy ) __attribute__((weak));
#define cuMemcpy2D_v2(pCopy) (cuMemcpy2D_v2 ? cuMemcpy2D_v2(pCopy) : 0)

extern "C" CUresult cuMemcpy2DAsync_v2 ( const CUDA_MEMCPY2D* pCopy, CUstream hStream ) __attribute__((weak));
#define cuMemcpy2DAsync_v2(pCopy, hStream) (cuMemcpy2DAsync_v2 ? cuMemcpy2DAsync_v2(pCopy, hStream) : 0)

extern "C" CUresult cuMemcpy2DUnaligned_v2 ( const CUDA_MEMCPY2D* pCopy ) __attribute__((weak));
#define cuMemcpy2DUnaligned_v2(pCopy) (cuMemcpy2DUnaligned_v2 ? cuMemcpy2DUnaligned_v2(pCopy) : 0)

extern "C" CUresult cuMemcpy3D_v2 ( const CUDA_MEMCPY3D* pCopy ) __attribute__((weak));
#define cuMemcpy3D_v2(pCopy) (cuMemcpy3D_v2 ? cuMemcpy3D_v2(pCopy) : 0)

extern "C" CUresult cuMemcpy3DAsync_v2 ( const CUDA_MEMCPY3D* pCopy, CUstream hStream ) __attribute__((weak));
#define cuMemcpy3DAsync_v2(pCopy, hStream) (cuMemcpy3DAsync_v2 ? cuMemcpy3DAsync_v2(pCopy, hStream) : 0)

extern "C" CUresult cuMemcpy3DPeer ( const CUDA_MEMCPY3D_PEER* pCopy ) __attribute__((weak));
#define cuMemcpy3DPeer(pCopy) (cuMemcpy3DPeer ? cuMemcpy3DPeer(pCopy) : 0)

extern "C" CUresult cuMemcpy3DPeerAsync ( const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream ) __attribute__((weak));
#define cuMemcpy3DPeerAsync(pCopy, hStream) (cuMemcpy3DPeerAsync ? cuMemcpy3DPeerAsync(pCopy, hStream) : 0)

extern "C" CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream ) __attribute__((weak));
#define cuMemcpyAsync(dst, src, ByteCount, hStream) (cuMemcpyAsync ? cuMemcpyAsync(dst, src, ByteCount, hStream) : 0)

extern "C" CUresult cuMemcpyAtoA_v2 ( CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount ) __attribute__((weak));
#define cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount) (cuMemcpyAtoA_v2 ? cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount) : 0)

extern "C" CUresult cuMemcpyAtoD_v2 ( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount ) __attribute__((weak));
#define cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount) (cuMemcpyAtoD_v2 ? cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount) : 0)

extern "C" CUresult cuMemcpyAtoH_v2 ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount ) __attribute__((weak));
#define cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount) (cuMemcpyAtoH_v2 ? cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount) : 0)

extern "C" CUresult cuMemcpyAtoHAsync_v2 ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream ) __attribute__((weak));
#define cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream) (cuMemcpyAtoHAsync_v2 ? cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream) : 0)

extern "C" CUresult cuMemcpyDtoA_v2 ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount ) __attribute__((weak));
#define cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount) (cuMemcpyDtoA_v2 ? cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount) : 0)

extern "C" CUresult cuMemcpyDtoD_v2 ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount ) __attribute__((weak));
#define cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount) (cuMemcpyDtoD_v2 ? cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount) : 0)

extern "C" CUresult cuMemcpyDtoDAsync_v2 ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ) __attribute__((weak));
#define cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream) (cuMemcpyDtoDAsync_v2 ? cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream) : 0)

extern "C" CUresult cuMemcpyDtoH_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount ) __attribute__((weak));
#define cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount) (cuMemcpyDtoH_v2 ? cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount) : 0)

extern "C" CUresult cuMemcpyDtoHAsync_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ) __attribute__((weak));
#define cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream) (cuMemcpyDtoHAsync_v2 ? cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream) : 0)

extern "C" CUresult cuMemcpyHtoA_v2 ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount ) __attribute__((weak));
#define cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount) (cuMemcpyHtoA_v2 ? cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount) : 0)

extern "C" CUresult cuMemcpyHtoAAsync_v2 ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream ) __attribute__((weak));
#define cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream) (cuMemcpyHtoAAsync_v2 ? cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream) : 0)

extern "C" CUresult cuMemcpyHtoD_v2 ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount ) __attribute__((weak));
#define cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount) (cuMemcpyHtoD_v2 ? cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount) : 0)

extern "C" CUresult cuMemcpyHtoDAsync_v2 ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream ) __attribute__((weak));
#define cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream) (cuMemcpyHtoDAsync_v2 ? cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream) : 0)

extern "C" CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount ) __attribute__((weak));
#define cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount) (cuMemcpyPeer ? cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount) : 0)

extern "C" CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream ) __attribute__((weak));
#define cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream) (cuMemcpyPeerAsync ? cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream) : 0)

extern "C" CUresult cuMemsetD16_v2 ( CUdeviceptr dstDevice, unsigned short us, size_t N ) __attribute__((weak));
#define cuMemsetD16_v2(dstDevice, us, N) (cuMemsetD16_v2 ? cuMemsetD16_v2(dstDevice, us, N) : 0)

extern "C" CUresult cuMemsetD16Async ( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream ) __attribute__((weak));
#define cuMemsetD16Async(dstDevice, us, N, hStream) (cuMemsetD16Async ? cuMemsetD16Async(dstDevice, us, N, hStream) : 0)

extern "C" CUresult cuMemsetD2D16_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height ) __attribute__((weak));
#define cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height) (cuMemsetD2D16_v2 ? cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height) : 0)

extern "C" CUresult cuMemsetD2D16Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream ) __attribute__((weak));
#define cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream) (cuMemsetD2D16Async ? cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream) : 0)

extern "C" CUresult cuMemsetD2D32_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height ) __attribute__((weak));
#define cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height) (cuMemsetD2D32_v2 ? cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height) : 0)

extern "C" CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream ) __attribute__((weak));
#define cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream) (cuMemsetD2D32Async ? cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream) : 0)

extern "C" CUresult cuMemsetD2D8_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height ) __attribute__((weak));
#define cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height) (cuMemsetD2D8_v2 ? cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height) : 0)

extern "C" CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream ) __attribute__((weak));
#define cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream) (cuMemsetD2D8Async ? cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream) : 0)

extern "C" CUresult cuMemsetD32_v2 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N ) __attribute__((weak));
#define cuMemsetD32_v2(dstDevice, ui, N) (cuMemsetD32_v2 ? cuMemsetD32_v2(dstDevice, ui, N) : 0)

extern "C" CUresult cuMemsetD32Async ( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream ) __attribute__((weak));
#define cuMemsetD32Async(dstDevice, ui, N, hStream) (cuMemsetD32Async ? cuMemsetD32Async(dstDevice, ui, N, hStream) : 0)

extern "C" CUresult cuMemsetD8_v2 ( CUdeviceptr dstDevice, unsigned char  uc, size_t N ) __attribute__((weak));
#define cuMemsetD8_v2(dstDevice, uc, N) (cuMemsetD8_v2 ? cuMemsetD8_v2(dstDevice, uc, N) : 0)

extern "C" CUresult cuMemsetD8Async ( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream ) __attribute__((weak));
#define cuMemsetD8Async(dstDevice, uc, N, hStream) (cuMemsetD8Async ? cuMemsetD8Async(dstDevice, uc, N, hStream) : 0)

extern "C" CUresult cuMipmappedArrayCreate ( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels ) __attribute__((weak));
#define cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels) (cuMipmappedArrayCreate ? cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels) : 0)

extern "C" CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray ) __attribute__((weak));
#define cuMipmappedArrayDestroy(hMipmappedArray) (cuMipmappedArrayDestroy ? cuMipmappedArrayDestroy(hMipmappedArray) : 0)

extern "C" CUresult cuMipmappedArrayGetLevel ( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level ) __attribute__((weak));
#define cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level) (cuMipmappedArrayGetLevel ? cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level) : 0)

extern "C" CUresult cuMemAdvise ( CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device ) __attribute__((weak));
#define cuMemAdvise(devPtr, count, advice, device) (cuMemAdvise ? cuMemAdvise(devPtr, count, advice, device) : 0)

extern "C" CUresult cuMemPrefetchAsync ( CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream ) __attribute__((weak));
#define cuMemPrefetchAsync(devPtr, count, dstDevice, hStream) (cuMemPrefetchAsync ? cuMemPrefetchAsync(devPtr, count, dstDevice, hStream) : 0)

extern "C" CUresult cuMemRangeGetAttribute ( void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count ) __attribute__((weak));
#define cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count) (cuMemRangeGetAttribute ? cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count) : 0)

extern "C" CUresult cuMemRangeGetAttributes ( void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count ) __attribute__((weak));
#define cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count) (cuMemRangeGetAttributes ? cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count) : 0)

extern "C" CUresult cuPointerGetAttribute ( void* data, CUpointer_attribute attribute, CUdeviceptr ptr ) __attribute__((weak));
#define cuPointerGetAttribute(data, attribute, ptr) (cuPointerGetAttribute ? cuPointerGetAttribute(data, attribute, ptr) : 0)

extern "C" CUresult cuPointerGetAttributes ( unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr ) __attribute__((weak));
#define cuPointerGetAttributes(numAttributes, attributes, data, ptr) (cuPointerGetAttributes ? cuPointerGetAttributes(numAttributes, attributes, data, ptr) : 0)

extern "C" CUresult cuPointerSetAttribute ( const void* value, CUpointer_attribute attribute, CUdeviceptr ptr ) __attribute__((weak));
#define cuPointerSetAttribute(value, attribute, ptr) (cuPointerSetAttribute ? cuPointerSetAttribute(value, attribute, ptr) : 0)

extern "C" CUresult cuStreamAddCallback ( CUstream hStream, CUstreamCallback callback, void* userData, unsigned int  flags ) __attribute__((weak));
#define cuStreamAddCallback(hStream, callback, userData, flags) (cuStreamAddCallback ? cuStreamAddCallback(hStream, callback, userData, flags) : 0)

extern "C" CUresult cuStreamAttachMemAsync ( CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int  flags ) __attribute__((weak));
#define cuStreamAttachMemAsync(hStream, dptr, length, flags) (cuStreamAttachMemAsync ? cuStreamAttachMemAsync(hStream, dptr, length, flags) : 0)

extern "C" CUresult cuStreamCreate ( CUstream* phStream, unsigned int  Flags ) __attribute__((weak));
#define cuStreamCreate(phStream, Flags) (cuStreamCreate ? cuStreamCreate(phStream, Flags) : 0)

extern "C" CUresult cuStreamCreateWithPriority ( CUstream* phStream, unsigned int  flags, int  priority ) __attribute__((weak));
#define cuStreamCreateWithPriority(phStream, flags, priority) (cuStreamCreateWithPriority ? cuStreamCreateWithPriority(phStream, flags, priority) : 0)

extern "C" CUresult cuStreamDestroy_v2 ( CUstream hStream ) __attribute__((weak));
#define cuStreamDestroy_v2(hStream) (cuStreamDestroy_v2 ? cuStreamDestroy_v2(hStream) : 0)

extern "C" CUresult cuStreamEndCapture ( CUstream hStream, CUgraph* phGraph ) __attribute__((weak));
#define cuStreamEndCapture(hStream, phGraph) (cuStreamEndCapture ? cuStreamEndCapture(hStream, phGraph) : 0)

extern "C" CUresult cuStreamGetCtx ( CUstream hStream, CUcontext* pctx ) __attribute__((weak));
#define cuStreamGetCtx(hStream, pctx) (cuStreamGetCtx ? cuStreamGetCtx(hStream, pctx) : 0)

extern "C" CUresult cuStreamGetFlags ( CUstream hStream, unsigned int* flags ) __attribute__((weak));
#define cuStreamGetFlags(hStream, flags) (cuStreamGetFlags ? cuStreamGetFlags(hStream, flags) : 0)

extern "C" CUresult cuStreamGetPriority ( CUstream hStream, int* priority ) __attribute__((weak));
#define cuStreamGetPriority(hStream, priority) (cuStreamGetPriority ? cuStreamGetPriority(hStream, priority) : 0)

extern "C" CUresult cuStreamIsCapturing ( CUstream hStream, CUstreamCaptureStatus* captureStatus ) __attribute__((weak));
#define cuStreamIsCapturing(hStream, captureStatus) (cuStreamIsCapturing ? cuStreamIsCapturing(hStream, captureStatus) : 0)

extern "C" CUresult cuStreamQuery ( CUstream hStream ) __attribute__((weak));
#define cuStreamQuery(hStream) (cuStreamQuery ? cuStreamQuery(hStream) : 0)

extern "C" CUresult cuStreamSynchronize ( CUstream hStream ) __attribute__((weak));
#define cuStreamSynchronize(hStream) (cuStreamSynchronize ? cuStreamSynchronize(hStream) : 0)

extern "C" CUresult cuStreamWaitEvent ( CUstream hStream, CUevent hEvent, unsigned int  Flags ) __attribute__((weak));
#define cuStreamWaitEvent(hStream, hEvent, Flags) (cuStreamWaitEvent ? cuStreamWaitEvent(hStream, hEvent, Flags) : 0)

extern "C" CUresult cuEventCreate ( CUevent* phEvent, unsigned int  Flags ) __attribute__((weak));
#define cuEventCreate(phEvent, Flags) (cuEventCreate ? cuEventCreate(phEvent, Flags) : 0)

extern "C" CUresult cuEventDestroy_v2 ( CUevent hEvent ) __attribute__((weak));
#define cuEventDestroy_v2(hEvent) (cuEventDestroy_v2 ? cuEventDestroy_v2(hEvent) : 0)

extern "C" CUresult cuEventElapsedTime ( float* pMilliseconds, CUevent hStart, CUevent hEnd ) __attribute__((weak));
#define cuEventElapsedTime(pMilliseconds, hStart, hEnd) (cuEventElapsedTime ? cuEventElapsedTime(pMilliseconds, hStart, hEnd) : 0)

extern "C" CUresult cuEventQuery ( CUevent hEvent ) __attribute__((weak));
#define cuEventQuery(hEvent) (cuEventQuery ? cuEventQuery(hEvent) : 0)

extern "C" CUresult cuEventRecord ( CUevent hEvent, CUstream hStream ) __attribute__((weak));
#define cuEventRecord(hEvent, hStream) (cuEventRecord ? cuEventRecord(hEvent, hStream) : 0)

extern "C" CUresult cuEventSynchronize ( CUevent hEvent ) __attribute__((weak));
#define cuEventSynchronize(hEvent) (cuEventSynchronize ? cuEventSynchronize(hEvent) : 0)

extern "C" CUresult cuDestroyExternalMemory ( CUexternalMemory extMem ) __attribute__((weak));
#define cuDestroyExternalMemory(extMem) (cuDestroyExternalMemory ? cuDestroyExternalMemory(extMem) : 0)

extern "C" CUresult cuDestroyExternalSemaphore ( CUexternalSemaphore extSem ) __attribute__((weak));
#define cuDestroyExternalSemaphore(extSem) (cuDestroyExternalSemaphore ? cuDestroyExternalSemaphore(extSem) : 0)

extern "C" CUresult cuExternalMemoryGetMappedBuffer ( CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc ) __attribute__((weak));
#define cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc) (cuExternalMemoryGetMappedBuffer ? cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc) : 0)

extern "C" CUresult cuExternalMemoryGetMappedMipmappedArray ( CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc ) __attribute__((weak));
#define cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc) (cuExternalMemoryGetMappedMipmappedArray ? cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc) : 0)

extern "C" CUresult cuImportExternalMemory ( CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc ) __attribute__((weak));
#define cuImportExternalMemory(extMem_out, memHandleDesc) (cuImportExternalMemory ? cuImportExternalMemory(extMem_out, memHandleDesc) : 0)

extern "C" CUresult cuImportExternalSemaphore ( CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc ) __attribute__((weak));
#define cuImportExternalSemaphore(extSem_out, semHandleDesc) (cuImportExternalSemaphore ? cuImportExternalSemaphore(extSem_out, semHandleDesc) : 0)

extern "C" CUresult cuSignalExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream ) __attribute__((weak));
#define cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream) (cuSignalExternalSemaphoresAsync ? cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream) : 0)

extern "C" CUresult cuWaitExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream ) __attribute__((weak));
#define cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream) (cuWaitExternalSemaphoresAsync ? cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream) : 0)

extern "C" CUresult cuStreamBatchMemOp ( CUstream stream, unsigned int  count, CUstreamBatchMemOpParams* paramArray, unsigned int  flags ) __attribute__((weak));
#define cuStreamBatchMemOp(stream, count, paramArray, flags) (cuStreamBatchMemOp ? cuStreamBatchMemOp(stream, count, paramArray, flags) : 0)

extern "C" CUresult cuStreamWaitValue32 ( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags ) __attribute__((weak));
#define cuStreamWaitValue32(stream, addr, value, flags) (cuStreamWaitValue32 ? cuStreamWaitValue32(stream, addr, value, flags) : 0)

extern "C" CUresult cuStreamWaitValue64 ( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags ) __attribute__((weak));
#define cuStreamWaitValue64(stream, addr, value, flags) (cuStreamWaitValue64 ? cuStreamWaitValue64(stream, addr, value, flags) : 0)

extern "C" CUresult cuStreamWriteValue32 ( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags ) __attribute__((weak));
#define cuStreamWriteValue32(stream, addr, value, flags) (cuStreamWriteValue32 ? cuStreamWriteValue32(stream, addr, value, flags) : 0)

extern "C" CUresult cuStreamWriteValue64 ( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags ) __attribute__((weak));
#define cuStreamWriteValue64(stream, addr, value, flags) (cuStreamWriteValue64 ? cuStreamWriteValue64(stream, addr, value, flags) : 0)

extern "C" CUresult cuFuncGetAttribute ( int* pi, CUfunction_attribute attrib, CUfunction hfunc ) __attribute__((weak));
#define cuFuncGetAttribute(pi, attrib, hfunc) (cuFuncGetAttribute ? cuFuncGetAttribute(pi, attrib, hfunc) : 0)

extern "C" CUresult cuFuncSetAttribute ( CUfunction hfunc, CUfunction_attribute attrib, int  value ) __attribute__((weak));
#define cuFuncSetAttribute(hfunc, attrib, value) (cuFuncSetAttribute ? cuFuncSetAttribute(hfunc, attrib, value) : 0)

extern "C" CUresult cuFuncSetCacheConfig ( CUfunction hfunc, CUfunc_cache config ) __attribute__((weak));
#define cuFuncSetCacheConfig(hfunc, config) (cuFuncSetCacheConfig ? cuFuncSetCacheConfig(hfunc, config) : 0)

extern "C" CUresult cuFuncSetSharedMemConfig ( CUfunction hfunc, CUsharedconfig config ) __attribute__((weak));
#define cuFuncSetSharedMemConfig(hfunc, config) (cuFuncSetSharedMemConfig ? cuFuncSetSharedMemConfig(hfunc, config) : 0)

extern "C" CUresult cuLaunchCooperativeKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams ) __attribute__((weak));
#define cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams) (cuLaunchCooperativeKernel ? cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams) : 0)

extern "C" CUresult cuLaunchCooperativeKernelMultiDevice ( CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int  numDevices, unsigned int  flags ) __attribute__((weak));
#define cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags) (cuLaunchCooperativeKernelMultiDevice ? cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags) : 0)

extern "C" CUresult cuLaunchHostFunc ( CUstream hStream, CUhostFn fn, void* userData ) __attribute__((weak));
#define cuLaunchHostFunc(hStream, fn, userData) (cuLaunchHostFunc ? cuLaunchHostFunc(hStream, fn, userData) : 0)

extern "C" CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ) __attribute__((weak));
#define cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra) (cuLaunchKernel ? cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra) : 0)

extern "C" CUresult cuFuncSetBlockShape ( CUfunction hfunc, int  x, int  y, int  z ) __attribute__((weak));
#define cuFuncSetBlockShape(hfunc, x, y, z) (cuFuncSetBlockShape ? cuFuncSetBlockShape(hfunc, x, y, z) : 0)

extern "C" CUresult cuFuncSetSharedSize ( CUfunction hfunc, unsigned int  bytes ) __attribute__((weak));
#define cuFuncSetSharedSize(hfunc, bytes) (cuFuncSetSharedSize ? cuFuncSetSharedSize(hfunc, bytes) : 0)

extern "C" CUresult cuLaunch ( CUfunction f ) __attribute__((weak));
#define cuLaunch(f) (cuLaunch ? cuLaunch(f) : 0)

extern "C" CUresult cuLaunchGrid ( CUfunction f, int  grid_width, int  grid_height ) __attribute__((weak));
#define cuLaunchGrid(f, grid_width, grid_height) (cuLaunchGrid ? cuLaunchGrid(f, grid_width, grid_height) : 0)

extern "C" CUresult cuLaunchGridAsync ( CUfunction f, int  grid_width, int  grid_height, CUstream hStream ) __attribute__((weak));
#define cuLaunchGridAsync(f, grid_width, grid_height, hStream) (cuLaunchGridAsync ? cuLaunchGridAsync(f, grid_width, grid_height, hStream) : 0)

extern "C" CUresult cuParamSetSize ( CUfunction hfunc, unsigned int  numbytes ) __attribute__((weak));
#define cuParamSetSize(hfunc, numbytes) (cuParamSetSize ? cuParamSetSize(hfunc, numbytes) : 0)

extern "C" CUresult cuParamSetTexRef ( CUfunction hfunc, int  texunit, CUtexref hTexRef ) __attribute__((weak));
#define cuParamSetTexRef(hfunc, texunit, hTexRef) (cuParamSetTexRef ? cuParamSetTexRef(hfunc, texunit, hTexRef) : 0)

extern "C" CUresult cuParamSetf ( CUfunction hfunc, int  offset, float  value ) __attribute__((weak));
#define cuParamSetf(hfunc, offset, value) (cuParamSetf ? cuParamSetf(hfunc, offset, value) : 0)

extern "C" CUresult cuParamSeti ( CUfunction hfunc, int  offset, unsigned int  value ) __attribute__((weak));
#define cuParamSeti(hfunc, offset, value) (cuParamSeti ? cuParamSeti(hfunc, offset, value) : 0)

extern "C" CUresult cuParamSetv ( CUfunction hfunc, int  offset, void* ptr, unsigned int  numbytes ) __attribute__((weak));
#define cuParamSetv(hfunc, offset, ptr, numbytes) (cuParamSetv ? cuParamSetv(hfunc, offset, ptr, numbytes) : 0)

extern "C" CUresult cuGraphCreate ( CUgraph* phGraph, unsigned int  flags ) __attribute__((weak));
#define cuGraphCreate(phGraph, flags) (cuGraphCreate ? cuGraphCreate(phGraph, flags) : 0)

extern "C" CUresult cuGraphDestroy ( CUgraph hGraph ) __attribute__((weak));
#define cuGraphDestroy(hGraph) (cuGraphDestroy ? cuGraphDestroy(hGraph) : 0)

extern "C" CUresult cuGraphDestroyNode ( CUgraphNode hNode ) __attribute__((weak));
#define cuGraphDestroyNode(hNode) (cuGraphDestroyNode ? cuGraphDestroyNode(hNode) : 0)

extern "C" CUresult cuGraphExecDestroy ( CUgraphExec hGraphExec ) __attribute__((weak));
#define cuGraphExecDestroy(hGraphExec) (cuGraphExecDestroy ? cuGraphExecDestroy(hGraphExec) : 0)

extern "C" CUresult cuGraphGetEdges ( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges ) __attribute__((weak));
#define cuGraphGetEdges(hGraph, from, to, numEdges) (cuGraphGetEdges ? cuGraphGetEdges(hGraph, from, to, numEdges) : 0)

extern "C" CUresult cuGraphGetNodes ( CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes ) __attribute__((weak));
#define cuGraphGetNodes(hGraph, nodes, numNodes) (cuGraphGetNodes ? cuGraphGetNodes(hGraph, nodes, numNodes) : 0)

extern "C" CUresult cuGraphGetRootNodes ( CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes ) __attribute__((weak));
#define cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes) (cuGraphGetRootNodes ? cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes) : 0)

extern "C" CUresult cuGraphHostNodeGetParams ( CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams ) __attribute__((weak));
#define cuGraphHostNodeGetParams(hNode, nodeParams) (cuGraphHostNodeGetParams ? cuGraphHostNodeGetParams(hNode, nodeParams) : 0)

extern "C" CUresult cuGraphHostNodeSetParams ( CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams ) __attribute__((weak));
#define cuGraphHostNodeSetParams(hNode, nodeParams) (cuGraphHostNodeSetParams ? cuGraphHostNodeSetParams(hNode, nodeParams) : 0)

extern "C" CUresult cuGraphKernelNodeGetParams ( CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams ) __attribute__((weak));
#define cuGraphKernelNodeGetParams(hNode, nodeParams) (cuGraphKernelNodeGetParams ? cuGraphKernelNodeGetParams(hNode, nodeParams) : 0)

extern "C" CUresult cuGraphKernelNodeSetParams ( CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams ) __attribute__((weak));
#define cuGraphKernelNodeSetParams(hNode, nodeParams) (cuGraphKernelNodeSetParams ? cuGraphKernelNodeSetParams(hNode, nodeParams) : 0)

extern "C" CUresult cuGraphLaunch ( CUgraphExec hGraphExec, CUstream hStream ) __attribute__((weak));
#define cuGraphLaunch(hGraphExec, hStream) (cuGraphLaunch ? cuGraphLaunch(hGraphExec, hStream) : 0)

extern "C" CUresult cuGraphMemcpyNodeGetParams ( CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams ) __attribute__((weak));
#define cuGraphMemcpyNodeGetParams(hNode, nodeParams) (cuGraphMemcpyNodeGetParams ? cuGraphMemcpyNodeGetParams(hNode, nodeParams) : 0)

extern "C" CUresult cuGraphMemcpyNodeSetParams ( CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams ) __attribute__((weak));
#define cuGraphMemcpyNodeSetParams(hNode, nodeParams) (cuGraphMemcpyNodeSetParams ? cuGraphMemcpyNodeSetParams(hNode, nodeParams) : 0)

extern "C" CUresult cuGraphMemsetNodeGetParams ( CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams ) __attribute__((weak));
#define cuGraphMemsetNodeGetParams(hNode, nodeParams) (cuGraphMemsetNodeGetParams ? cuGraphMemsetNodeGetParams(hNode, nodeParams) : 0)

extern "C" CUresult cuGraphMemsetNodeSetParams ( CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams ) __attribute__((weak));
#define cuGraphMemsetNodeSetParams(hNode, nodeParams) (cuGraphMemsetNodeSetParams ? cuGraphMemsetNodeSetParams(hNode, nodeParams) : 0)

extern "C" CUresult cuGraphNodeFindInClone ( CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph ) __attribute__((weak));
#define cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph) (cuGraphNodeFindInClone ? cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph) : 0)

extern "C" CUresult cuGraphNodeGetDependencies ( CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies ) __attribute__((weak));
#define cuGraphNodeGetDependencies(hNode, dependencies, numDependencies) (cuGraphNodeGetDependencies ? cuGraphNodeGetDependencies(hNode, dependencies, numDependencies) : 0)

extern "C" CUresult cuGraphNodeGetDependentNodes ( CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes ) __attribute__((weak));
#define cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes) (cuGraphNodeGetDependentNodes ? cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes) : 0)

extern "C" CUresult cuGraphNodeGetType ( CUgraphNode hNode, CUgraphNodeType* type ) __attribute__((weak));
#define cuGraphNodeGetType(hNode, type) (cuGraphNodeGetType ? cuGraphNodeGetType(hNode, type) : 0)

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize ) __attribute__((weak));
#define cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize) (cuOccupancyMaxActiveBlocksPerMultiprocessor ? cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize) : 0)

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags ) __attribute__((weak));
#define cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags) (cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ? cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags) : 0)

extern "C" CUresult cuOccupancyMaxPotentialBlockSize ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit ) __attribute__((weak));
#define cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit) (cuOccupancyMaxPotentialBlockSize ? cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit) : 0)

extern "C" CUresult cuOccupancyMaxPotentialBlockSizeWithFlags ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit, unsigned int  flags ) __attribute__((weak));
#define cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags) (cuOccupancyMaxPotentialBlockSizeWithFlags ? cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags) : 0)

extern "C" CUresult cuTexRefCreate ( CUtexref* pTexRef ) __attribute__((weak));
#define cuTexRefCreate(pTexRef) (cuTexRefCreate ? cuTexRefCreate(pTexRef) : 0)

extern "C" CUresult cuTexRefDestroy ( CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefDestroy(hTexRef) (cuTexRefDestroy ? cuTexRefDestroy(hTexRef) : 0)

extern "C" CUresult cuTexRefGetAddress_v2 ( CUdeviceptr* pdptr, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetAddress_v2(pdptr, hTexRef) (cuTexRefGetAddress_v2 ? cuTexRefGetAddress_v2(pdptr, hTexRef) : 0)

extern "C" CUresult cuTexRefGetAddressMode ( CUaddress_mode* pam, CUtexref hTexRef, int  dim ) __attribute__((weak));
#define cuTexRefGetAddressMode(pam, hTexRef, dim) (cuTexRefGetAddressMode ? cuTexRefGetAddressMode(pam, hTexRef, dim) : 0)

extern "C" CUresult cuTexRefGetArray ( CUarray* phArray, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetArray(phArray, hTexRef) (cuTexRefGetArray ? cuTexRefGetArray(phArray, hTexRef) : 0)

extern "C" CUresult cuTexRefGetBorderColor ( float* pBorderColor, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetBorderColor(pBorderColor, hTexRef) (cuTexRefGetBorderColor ? cuTexRefGetBorderColor(pBorderColor, hTexRef) : 0)

extern "C" CUresult cuTexRefGetFilterMode ( CUfilter_mode* pfm, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetFilterMode(pfm, hTexRef) (cuTexRefGetFilterMode ? cuTexRefGetFilterMode(pfm, hTexRef) : 0)

extern "C" CUresult cuTexRefGetFlags ( unsigned int* pFlags, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetFlags(pFlags, hTexRef) (cuTexRefGetFlags ? cuTexRefGetFlags(pFlags, hTexRef) : 0)

extern "C" CUresult cuTexRefGetFormat ( CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetFormat(pFormat, pNumChannels, hTexRef) (cuTexRefGetFormat ? cuTexRefGetFormat(pFormat, pNumChannels, hTexRef) : 0)

extern "C" CUresult cuTexRefGetMaxAnisotropy ( int* pmaxAniso, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef) (cuTexRefGetMaxAnisotropy ? cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef) : 0)

extern "C" CUresult cuTexRefGetMipmapFilterMode ( CUfilter_mode* pfm, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetMipmapFilterMode(pfm, hTexRef) (cuTexRefGetMipmapFilterMode ? cuTexRefGetMipmapFilterMode(pfm, hTexRef) : 0)

extern "C" CUresult cuTexRefGetMipmapLevelBias ( float* pbias, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetMipmapLevelBias(pbias, hTexRef) (cuTexRefGetMipmapLevelBias ? cuTexRefGetMipmapLevelBias(pbias, hTexRef) : 0)

extern "C" CUresult cuTexRefGetMipmapLevelClamp ( float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef) (cuTexRefGetMipmapLevelClamp ? cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef) : 0)

extern "C" CUresult cuTexRefGetMipmappedArray ( CUmipmappedArray* phMipmappedArray, CUtexref hTexRef ) __attribute__((weak));
#define cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef) (cuTexRefGetMipmappedArray ? cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef) : 0)

extern "C" CUresult cuTexRefSetAddress_v2 ( size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes ) __attribute__((weak));
#define cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes) (cuTexRefSetAddress_v2 ? cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes) : 0)

extern "C" CUresult cuTexRefSetAddress2D_v3 ( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch ) __attribute__((weak));
#define cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch) (cuTexRefSetAddress2D_v3 ? cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch) : 0)

extern "C" CUresult cuTexRefSetAddressMode ( CUtexref hTexRef, int  dim, CUaddress_mode am ) __attribute__((weak));
#define cuTexRefSetAddressMode(hTexRef, dim, am) (cuTexRefSetAddressMode ? cuTexRefSetAddressMode(hTexRef, dim, am) : 0)

extern "C" CUresult cuTexRefSetArray ( CUtexref hTexRef, CUarray hArray, unsigned int  Flags ) __attribute__((weak));
#define cuTexRefSetArray(hTexRef, hArray, Flags) (cuTexRefSetArray ? cuTexRefSetArray(hTexRef, hArray, Flags) : 0)

extern "C" CUresult cuTexRefSetBorderColor ( CUtexref hTexRef, float* pBorderColor ) __attribute__((weak));
#define cuTexRefSetBorderColor(hTexRef, pBorderColor) (cuTexRefSetBorderColor ? cuTexRefSetBorderColor(hTexRef, pBorderColor) : 0)

extern "C" CUresult cuTexRefSetFilterMode ( CUtexref hTexRef, CUfilter_mode fm ) __attribute__((weak));
#define cuTexRefSetFilterMode(hTexRef, fm) (cuTexRefSetFilterMode ? cuTexRefSetFilterMode(hTexRef, fm) : 0)

extern "C" CUresult cuTexRefSetFlags ( CUtexref hTexRef, unsigned int  Flags ) __attribute__((weak));
#define cuTexRefSetFlags(hTexRef, Flags) (cuTexRefSetFlags ? cuTexRefSetFlags(hTexRef, Flags) : 0)

extern "C" CUresult cuTexRefSetFormat ( CUtexref hTexRef, CUarray_format fmt, int  NumPackedComponents ) __attribute__((weak));
#define cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents) (cuTexRefSetFormat ? cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents) : 0)

extern "C" CUresult cuTexRefSetMaxAnisotropy ( CUtexref hTexRef, unsigned int  maxAniso ) __attribute__((weak));
#define cuTexRefSetMaxAnisotropy(hTexRef, maxAniso) (cuTexRefSetMaxAnisotropy ? cuTexRefSetMaxAnisotropy(hTexRef, maxAniso) : 0)

extern "C" CUresult cuTexRefSetMipmapFilterMode ( CUtexref hTexRef, CUfilter_mode fm ) __attribute__((weak));
#define cuTexRefSetMipmapFilterMode(hTexRef, fm) (cuTexRefSetMipmapFilterMode ? cuTexRefSetMipmapFilterMode(hTexRef, fm) : 0)

extern "C" CUresult cuTexRefSetMipmapLevelBias ( CUtexref hTexRef, float  bias ) __attribute__((weak));
#define cuTexRefSetMipmapLevelBias(hTexRef, bias) (cuTexRefSetMipmapLevelBias ? cuTexRefSetMipmapLevelBias(hTexRef, bias) : 0)

extern "C" CUresult cuTexRefSetMipmapLevelClamp ( CUtexref hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp ) __attribute__((weak));
#define cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp) (cuTexRefSetMipmapLevelClamp ? cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp) : 0)

extern "C" CUresult cuTexRefSetMipmappedArray ( CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int  Flags ) __attribute__((weak));
#define cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags) (cuTexRefSetMipmappedArray ? cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags) : 0)

extern "C" CUresult cuSurfRefGetArray ( CUarray* phArray, CUsurfref hSurfRef ) __attribute__((weak));
#define cuSurfRefGetArray(phArray, hSurfRef) (cuSurfRefGetArray ? cuSurfRefGetArray(phArray, hSurfRef) : 0)

extern "C" CUresult cuSurfRefSetArray ( CUsurfref hSurfRef, CUarray hArray, unsigned int  Flags ) __attribute__((weak));
#define cuSurfRefSetArray(hSurfRef, hArray, Flags) (cuSurfRefSetArray ? cuSurfRefSetArray(hSurfRef, hArray, Flags) : 0)

extern "C" CUresult cuTexObjectCreate ( CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc ) __attribute__((weak));
#define cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc) (cuTexObjectCreate ? cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc) : 0)

extern "C" CUresult cuTexObjectDestroy ( CUtexObject texObject ) __attribute__((weak));
#define cuTexObjectDestroy(texObject) (cuTexObjectDestroy ? cuTexObjectDestroy(texObject) : 0)

extern "C" CUresult cuTexObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject ) __attribute__((weak));
#define cuTexObjectGetResourceDesc(pResDesc, texObject) (cuTexObjectGetResourceDesc ? cuTexObjectGetResourceDesc(pResDesc, texObject) : 0)

extern "C" CUresult cuTexObjectGetResourceViewDesc ( CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject ) __attribute__((weak));
#define cuTexObjectGetResourceViewDesc(pResViewDesc, texObject) (cuTexObjectGetResourceViewDesc ? cuTexObjectGetResourceViewDesc(pResViewDesc, texObject) : 0)

extern "C" CUresult cuTexObjectGetTextureDesc ( CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject ) __attribute__((weak));
#define cuTexObjectGetTextureDesc(pTexDesc, texObject) (cuTexObjectGetTextureDesc ? cuTexObjectGetTextureDesc(pTexDesc, texObject) : 0)

extern "C" CUresult cuSurfObjectCreate ( CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc ) __attribute__((weak));
#define cuSurfObjectCreate(pSurfObject, pResDesc) (cuSurfObjectCreate ? cuSurfObjectCreate(pSurfObject, pResDesc) : 0)

extern "C" CUresult cuSurfObjectDestroy ( CUsurfObject surfObject ) __attribute__((weak));
#define cuSurfObjectDestroy(surfObject) (cuSurfObjectDestroy ? cuSurfObjectDestroy(surfObject) : 0)

extern "C" CUresult cuSurfObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject ) __attribute__((weak));
#define cuSurfObjectGetResourceDesc(pResDesc, surfObject) (cuSurfObjectGetResourceDesc ? cuSurfObjectGetResourceDesc(pResDesc, surfObject) : 0)

extern "C" CUresult cuCtxDisablePeerAccess ( CUcontext peerContext ) __attribute__((weak));
#define cuCtxDisablePeerAccess(peerContext) (cuCtxDisablePeerAccess ? cuCtxDisablePeerAccess(peerContext) : 0)

extern "C" CUresult cuCtxEnablePeerAccess ( CUcontext peerContext, unsigned int  Flags ) __attribute__((weak));
#define cuCtxEnablePeerAccess(peerContext, Flags) (cuCtxEnablePeerAccess ? cuCtxEnablePeerAccess(peerContext, Flags) : 0)

extern "C" CUresult cuDeviceCanAccessPeer ( int* canAccessPeer, CUdevice dev, CUdevice peerDev ) __attribute__((weak));
#define cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev) (cuDeviceCanAccessPeer ? cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev) : 0)

extern "C" CUresult cuDeviceGetP2PAttribute ( int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice ) __attribute__((weak));
#define cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice) (cuDeviceGetP2PAttribute ? cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice) : 0)

extern "C" CUresult cuGraphicsMapResources ( unsigned int  count, CUgraphicsResource* resources, CUstream hStream ) __attribute__((weak));
#define cuGraphicsMapResources(count, resources, hStream) (cuGraphicsMapResources ? cuGraphicsMapResources(count, resources, hStream) : 0)

extern "C" CUresult cuGraphicsResourceGetMappedMipmappedArray ( CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource ) __attribute__((weak));
#define cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource) (cuGraphicsResourceGetMappedMipmappedArray ? cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource) : 0)

extern "C" CUresult cuGraphicsResourceGetMappedPointer_v2 ( CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource ) __attribute__((weak));
#define cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource) (cuGraphicsResourceGetMappedPointer_v2 ? cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource) : 0)

extern "C" CUresult cuGraphicsResourceSetMapFlags_v2 ( CUgraphicsResource resource, unsigned int  flags ) __attribute__((weak));
#define cuGraphicsResourceSetMapFlags_v2(resource, flags) (cuGraphicsResourceSetMapFlags_v2 ? cuGraphicsResourceSetMapFlags_v2(resource, flags) : 0)

extern "C" CUresult cuGraphicsSubResourceGetMappedArray ( CUarray* pArray, CUgraphicsResource resource, unsigned int  arrayIndex, unsigned int  mipLevel ) __attribute__((weak));
#define cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel) (cuGraphicsSubResourceGetMappedArray ? cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel) : 0)

extern "C" CUresult cuGraphicsUnmapResources ( unsigned int  count, CUgraphicsResource* resources, CUstream hStream ) __attribute__((weak));
#define cuGraphicsUnmapResources(count, resources, hStream) (cuGraphicsUnmapResources ? cuGraphicsUnmapResources(count, resources, hStream) : 0)

extern "C" CUresult cuGraphicsUnregisterResource ( CUgraphicsResource resource ) __attribute__((weak));
#define cuGraphicsUnregisterResource(resource) (cuGraphicsUnregisterResource ? cuGraphicsUnregisterResource(resource) : 0)

extern "C" void                                                                             __cudaRegisterFatBinaryEnd(void **fatCubinHandle) __attribute__((weak));
#define __cudaRegisterFatBinaryEnd(fatCubinHandle) (__cudaRegisterFatBinaryEnd ? __cudaRegisterFatBinaryEnd(fatCubinHandle) : 0)

extern "C" cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch) __attribute__((weak));
#define cufftPlan1d(plan, nx, type, batch) (cufftPlan1d ? cufftPlan1d(plan, nx, type, batch) : 0)

extern "C" cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type) __attribute__((weak));
#define cufftPlan2d(plan, nx, ny, type) (cufftPlan2d ? cufftPlan2d(plan, nx, ny, type) : 0)

extern "C" cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type) __attribute__((weak));
#define cufftPlan3d(plan, nx, ny, nz, type) (cufftPlan3d ? cufftPlan3d(plan, nx, ny, nz, type) : 0)

extern "C" cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch) __attribute__((weak));
#define cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch) (cufftPlanMany ? cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch) : 0)

extern "C" cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t *workSize) __attribute__((weak));
#define cufftMakePlan1d(plan, nx, type, batch, workSize) (cufftMakePlan1d ? cufftMakePlan1d(plan, nx, type, batch, workSize) : 0)

extern "C" cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t *workSize) __attribute__((weak));
#define cufftMakePlan2d(plan, nx, ny, type, workSize) (cufftMakePlan2d ? cufftMakePlan2d(plan, nx, ny, type, workSize) : 0)

extern "C" cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t *workSize) __attribute__((weak));
#define cufftMakePlan3d(plan, nx, ny, nz, type, workSize) (cufftMakePlan3d ? cufftMakePlan3d(plan, nx, ny, nz, type, workSize) : 0)

extern "C" cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workSize) __attribute__((weak));
#define cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize) (cufftMakePlanMany ? cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize) : 0)

extern "C" cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t * workSize) __attribute__((weak));
#define cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize) (cufftMakePlanMany64 ? cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize) : 0)

extern "C" cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t *workSize) __attribute__((weak));
#define cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize) (cufftGetSizeMany64 ? cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize) : 0)

extern "C" cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t *workSize) __attribute__((weak));
#define cufftEstimate1d(nx, type, batch, workSize) (cufftEstimate1d ? cufftEstimate1d(nx, type, batch, workSize) : 0)

extern "C" cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t *workSize) __attribute__((weak));
#define cufftEstimate2d(nx, ny, type, workSize) (cufftEstimate2d ? cufftEstimate2d(nx, ny, type, workSize) : 0)

extern "C" cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t *workSize) __attribute__((weak));
#define cufftEstimate3d(nx, ny, nz, type, workSize) (cufftEstimate3d ? cufftEstimate3d(nx, ny, nz, type, workSize) : 0)

extern "C" cufftResult cufftEstimateMany(int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workSize) __attribute__((weak));
#define cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize) (cufftEstimateMany ? cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize) : 0)

extern "C" cufftResult cufftCreate(cufftHandle * handle) __attribute__((weak));
#define cufftCreate(handle) (cufftCreate ? cufftCreate(handle) : 0)

extern "C" cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t *workSize ) __attribute__((weak));
#define cufftGetSize1d(handle, nx, type, batch, workSize) (cufftGetSize1d ? cufftGetSize1d(handle, nx, type, batch, workSize) : 0)

extern "C" cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t *workSize) __attribute__((weak));
#define cufftGetSize2d(handle, nx, ny, type, workSize) (cufftGetSize2d ? cufftGetSize2d(handle, nx, ny, type, workSize) : 0)

extern "C" cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t *workSize) __attribute__((weak));
#define cufftGetSize3d(handle, nx, ny, nz, type, workSize) (cufftGetSize3d ? cufftGetSize3d(handle, nx, ny, nz, type, workSize) : 0)

extern "C" cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workArea) __attribute__((weak));
#define cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea) (cufftGetSizeMany ? cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea) : 0)

extern "C" cufftResult cufftGetSize(cufftHandle handle, size_t *workSize) __attribute__((weak));
#define cufftGetSize(handle, workSize) (cufftGetSize ? cufftGetSize(handle, workSize) : 0)

extern "C" cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea) __attribute__((weak));
#define cufftSetWorkArea(plan, workArea) (cufftSetWorkArea ? cufftSetWorkArea(plan, workArea) : 0)

extern "C" cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) __attribute__((weak));
#define cufftSetAutoAllocation(plan, autoAllocate) (cufftSetAutoAllocation ? cufftSetAutoAllocation(plan, autoAllocate) : 0)

extern "C" cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction) __attribute__((weak));
#define cufftExecC2C(plan, idata, odata, direction) (cufftExecC2C ? cufftExecC2C(plan, idata, odata, direction) : 0)

extern "C" cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata) __attribute__((weak));
#define cufftExecR2C(plan, idata, odata) (cufftExecR2C ? cufftExecR2C(plan, idata, odata) : 0)

extern "C" cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata) __attribute__((weak));
#define cufftExecC2R(plan, idata, odata) (cufftExecC2R ? cufftExecC2R(plan, idata, odata) : 0)

extern "C" cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction) __attribute__((weak));
#define cufftExecZ2Z(plan, idata, odata, direction) (cufftExecZ2Z ? cufftExecZ2Z(plan, idata, odata, direction) : 0)

extern "C" cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata) __attribute__((weak));
#define cufftExecD2Z(plan, idata, odata) (cufftExecD2Z ? cufftExecD2Z(plan, idata, odata) : 0)

extern "C" cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleReal *odata) __attribute__((weak));
#define cufftExecZ2D(plan, idata, odata) (cufftExecZ2D ? cufftExecZ2D(plan, idata, odata) : 0)

extern "C" cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) __attribute__((weak));
#define cufftSetStream(plan, stream) (cufftSetStream ? cufftSetStream(plan, stream) : 0)

extern "C" cufftResult cufftDestroy(cufftHandle plan) __attribute__((weak));
#define cufftDestroy(plan) (cufftDestroy ? cufftDestroy(plan) : 0)

extern "C" cufftResult cufftGetVersion(int *version) __attribute__((weak));
#define cufftGetVersion(version) (cufftGetVersion ? cufftGetVersion(version) : 0)

extern "C" cufftResult cufftGetProperty(libraryPropertyType type, int *value) __attribute__((weak));
#define cufftGetProperty(type, value) (cufftGetProperty ? cufftGetProperty(type, value) : 0)


// for nccl 
extern "C" cudaError_t cudaStreamGetCaptureInfo_v2( cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned long long* id_out , cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) __attribute__((weak));
#define cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out) (cudaStreamGetCaptureInfo_v2 ? cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out) : 0)

extern "C" cudaError_t cudaUserObjectCreate( cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int  initialRefcount, unsigned int  flags ) __attribute__((weak));
#define cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags) (cudaUserObjectCreate ? cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)  : 0)


extern "C" cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int  count, unsigned int  flags) __attribute__((weak));
#define cudaGraphRetainUserObject(graph, object, count, flags) (cudaGraphRetainUserObject ? cudaGraphRetainUserObject(graph, object, count, flags)  : 0)

extern "C" cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int  flags) __attribute__((weak));
#define cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags) (cudaStreamUpdateCaptureDependencies ? cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags)  : 0)


extern "C" cudaError_t cudaGetDriverEntryPoint (const char* symbol, void** funcPtr, unsigned long long flags ) __attribute__((weak));
#define cudaGetDriverEntryPoint(symbol, funcPtr, flags) (cudaGetDriverEntryPoint ? cudaGetDriverEntryPoint(symbol, funcPtr, flags)  : 0)


extern "C" cudaError_t cudaFuncSetAttribute (const void* func, cudaFuncAttribute attr, int  value) __attribute__((weak));
#define cudaFuncSetAttribute(func, attr, value) (cudaFuncSetAttribute ? cudaFuncSetAttribute(func, attr, value)  : 0)


extern "C" cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) __attribute__((weak));
#define cudaIpcGetMemHandle(handle, devPtr) (cudaIpcGetMemHandle ? cudaIpcGetMemHandle(handle, devPtr)  : 0)


extern "C" cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags) __attribute__((weak));
#define cudaIpcOpenMemHandle(devPtr, handle, flags) (cudaIpcOpenMemHandle ? cudaIpcOpenMemHandle(devPtr, handle, flags)  : 0)


extern "C" cudaError_t cudaIpcCloseMemHandle(void* devPtr) __attribute__((weak));
#define cudaIpcCloseMemHandle(devPtr) (cudaIpcCloseMemHandle ? cudaIpcCloseMemHandle(devPtr)  : 0)


extern "C" cudaError_t cudaDriverGetVersion(int* driverVersion) __attribute__((weak));
#define cudaDriverGetVersion(driverVersion) (cudaDriverGetVersion ? cudaDriverGetVersion(driverVersion)  : 0)

extern "C" cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) __attribute__((weak));
#define cudaDeviceGetByPCIBusId(device, pciBusId) (cudaDeviceGetByPCIBusId ? cudaDeviceGetByPCIBusId(device, pciBusId)  : 0)


extern "C" cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode) __attribute__((weak));
#define cudaThreadExchangeStreamCaptureMode(mode) (cudaThreadExchangeStreamCaptureMode ? cudaThreadExchangeStreamCaptureMode(mode)  : 0)

extern "C" cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int  flags) __attribute__((weak));
#define cudaHostGetDevicePointer(pDevice, pHost, flags) (cudaHostGetDevicePointer ? cudaHostGetDevicePointer(pDevice, pHost, flags)  : 0)


extern "C" cudaError_t cudaHostUnregister(void* ptr) __attribute__((weak));
#define cudaHostUnregister(ptr) (cudaHostUnregister ? cudaHostUnregister(ptr)  : 0)


extern "C" cudaError_t cudaGraphAddEventWaitNode (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) __attribute__((weak));
#define cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event) (cudaGraphAddEventWaitNode ? cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)  : 0)


extern "C" cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) __attribute__((weak));
#define cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event) (cudaGraphAddEventRecordNode ? cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event)   : 0)


extern "C" cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) __attribute__((weak));
#define cudaLaunchHostFunc(stream, fn, userData) (cudaLaunchHostFunc ? cudaLaunchHostFunc(stream, fn, userData)  : 0)


extern "C" cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) __attribute__((weak));
#define cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams) (cudaGraphAddHostNode ? cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)  : 0)

extern "C" cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) __attribute__((weak));
#define cudaDeviceEnablePeerAccess(peerDevice, flags) (cudaDeviceEnablePeerAccess ? cudaDeviceEnablePeerAccess(peerDevice, flags)  : 0)


extern "C" cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) __attribute__((weak));
#define cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams) (cudaGraphAddKernelNode ? cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)  : 0)


extern "C" CUresult cuGetErrorName(CUresult error, const char** pStr) __attribute__((weak));
#define cuGetErrorName(error, pStr) (cuGetErrorName ? cuGetErrorName(error, pStr)  : 0)


extern "C" CUresult cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags) __attribute__((weak));
#define cuIpcOpenMemHandle_v2(pdptr, handle, Flags) (cuIpcOpenMemHandle_v2 ? cuIpcOpenMemHandle_v2(pdptr, handle, Flags) : 0)

// =================================PyTorch==========================
// CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count);
// CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags);
// cuUserObjectRelease
// cuUserObjectRetain
// cuUserObjectCreate
// cuGraphDebugDotPrint
// cuGraphKernelNodeSetAttribute  11000
// cuGraphKernelNodeGetAttribute, cudaVersion:11000
// cuGraphKernelNodeCopyAttributes, cudaVersion:11000
// cuGraphExecUpdate, cudaVersion:10020
// cuThreadExchangeStreamCaptureMode, cudaVersion:10010
// cuGraphExecEventWaitNodeSetEvent, cudaVersion:11010
// cuGraphExecEventRecordNodeSetEvent, cudaVersion:11010
// cuGraphExecChildGraphNodeSetParams, cudaVersion:11010
// cuGraphExecHostNodeSetParams, cudaVersion:10020
// cuGraphExecMemsetNodeSetParams, cudaVersion:10020
// cuGraphExecMemcpyNodeSetParams, cudaVersion:10020
// cuGraphExecKernelNodeSetParams, cudaVersion:10010
// cuStreamUpdateCaptureDependencies, cudaVersion:11030
// cuStreamGetCaptureInfo, cudaVersion:11030
// cuStreamBeginCapture, cudaVersion:10010
// cuGraphUpload, cudaVersion:11010
// cuGraphInstantiateWithFlags, cudaVersion:11040
// cuGraphInstantiate, cudaVersion:11000
// cuGraphRemoveDependencies, cudaVersion:10000
// cuGraphAddDependencies, cudaVersion:10000
// cuGraphClone, cudaVersion:10000
// cuDeviceSetGraphMemAttribute, cudaVersion:11040
// cuDeviceGetGraphMemAttribute, cudaVersion:11040
// cuDeviceGraphMemTrim, cudaVersion:11040
// cuGraphMemFreeNodeGetParams, cudaVersion:11040
// cuGraphAddMemFreeNode, cudaVersion:11040
// cuGraphMemAllocNodeGetParams, cudaVersion:11040
// cuGraphAddMemAllocNode, cudaVersion:11040
// cuGraphExecExternalSemaphoresWaitNodeSetParams, cudaVersion:11020
// cuGraphExecExternalSemaphoresSignalNodeSetParams, cudaVersion:11020
// cuGraphExternalSemaphoresWaitNodeSetParams, cudaVersion:11020
// cuGraphExternalSemaphoresWaitNodeGetParams, cudaVersion:11020
// cuGraphAddExternalSemaphoresWaitNode, cudaVersion:11020
// cuGraphExternalSemaphoresSignalNodeSetParams, cudaVersion:11020
// cuGraphExternalSemaphoresSignalNodeGetParams, cudaVersion:11020
// cuGraphAddExternalSemaphoresSignalNode, cudaVersion:11020
// cuGraphEventWaitNodeSetEvent, cudaVersion:11010
// cuGraphEventWaitNodeGetEvent, cudaVersion:11010
// cuGraphAddEventWaitNode, cudaVersion:11010
// cuGraphEventRecordNodeSetEvent, cudaVersion:11010
// cuGraphEventRecordNodeGetEvent, cudaVersion:11010
// cuGraphAddEventRecordNode, cudaVersion:11010
// cuGraphAddEmptyNode, cudaVersion:10000
// cuGraphChildGraphNodeGetGraph, cudaVersion:10000
// cuGraphAddChildGraphNode, cudaVersion:10000
// cuGraphAddHostNode, cudaVersion:10000
// cuGraphAddMemsetNode, cudaVersion:10000
// cuGraphAddMemcpyNode, cudaVersion:10000
// cuGraphAddKernelNode, cudaVersion:10000
// cuGetErrorName, cudaVersion:6000
// cuGetErrorString, cudaVersion:6000
// cuOccupancyMaxActiveClusters, cudaVersion:11070
// cuOccupancyMaxPotentialClusterSize, cudaVersion:11070
// cuOccupancyAvailableDynamicSMemPerBlock, cudaVersion:10020, flags:0
// cuGetExportTable, cudaVersion:3000, flags:0
// cuGraphicsVDPAURegisterOutputSurface, cudaVersion:3010, flags:0
// cuGraphicsVDPAURegisterVideoSurface, cudaVersion:3010, flags:0
// cuVDPAUCtxCreate, cudaVersion:3020, flags:0
// cuVDPAUGetDevice, cudaVersion:3010, flags:0
// cuProfilerStop, cudaVersion:4000, flags:0
// cuProfilerStart, cudaVersion:4000, flags:0
// cuProfilerInitialize, cudaVersion:4000, flags:0
// cuGraphicsResourceGetMappedPointer, cudaVersion:3020, flags:0
// cuGraphicsResourceSetMapFlags, cudaVersion:6050, flags:0
// cuEGLStreamConsumerConnectWithFlags, cudaVersion:8000, flags:0
// cuGraphicsResourceGetMappedEglFrame, cudaVersion:7000, flags:0
// cuEGLStreamProducerReturnFrame, cudaVersion:7000, flags:0
// cuEGLStreamProducerPresentFrame, cudaVersion:7000, flags:0
// cuEGLStreamProducerDisconnect, cudaVersion:7000, flags:0
// cuEGLStreamProducerConnect, cudaVersion:7000, flags:0
// cuEGLStreamConsumerReleaseFrame, cudaVersion:7000, flags:0
// cuEGLStreamConsumerAcquireFrame, cudaVersion:7000, flags:0
// cuEGLStreamConsumerDisconnect, cudaVersion:7000, flags:0
// cuEGLStreamConsumerConnect, cudaVersion:7000, flags:0
// cuGraphicsEGLRegisterImage, cudaVersion:7000, flags:0
// cuGraphicsGLRegisterBuffer, cudaVersion:3000, flags:0
// cuGraphicsGLRegisterImage, cudaVersion:3000, flags:0
// cuGLSetBufferObjectMapFlags, cudaVersion:2030, flags:0
// cuGLUnregisterBufferObject, cudaVersion:2000, flags:0
// cuGLUnmapBufferObjectAsync, cudaVersion:2030, flags:0
// cuGLUnmapBufferObject, cudaVersion:2000, flags:0
// cuGLMapBufferObjectAsync, cudaVersion:7000, flags:2
// cuGLMapBufferObjectAsync, cudaVersion:3020, flags:0
// cuGLMapBufferObject, cudaVersion:7000, flags:2
// cuGLMapBufferObject, cudaVersion:3020, flags:0
// cuGLRegisterBufferObject, cudaVersion:2000, flags:0
// cuGLGetDevices, cudaVersion:4010, flags:0
// cuGLInit, cudaVersion:2000, flags:0
// cuGLCtxCreate, cudaVersion:3020, flags:0
// cuIpcOpenMemHandle, cudaVersion:11000, flags:0
// cuIpcOpenMemHandle, cudaVersion:4010, flags:
// cuStreamSetAttribute, cudaVersion:11000, flags:2
// cuStreamGetAttribute, cudaVersion:11000, flags:2
// cuStreamGetAttribute, cudaVersion:11000, flags:0
// cuStreamCopyAttributes, cudaVersion:11000, flags:2
// cuStreamCopyAttributes, cudaVersion:11000, flags:0
// cuStreamDestroy, cudaVersion:4000, flags:0
// cuEventDestroy, cudaVersion:4000, flags:0
// cuEventRecordWithFlags, cudaVersion:11010, flags:2
// cuEventRecordWithFlags, cudaVersion:11010, flags:0
// cuLaunchKernelEx, cudaVersion:11060, flags:2
// cuLaunchKernelEx, cudaVersion:11060, flags:0
// cuDeviceGetNvSciSyncAttributes, cudaVersion:10020, flags:0
// cuTexRefSetAddress2D, cudaVersion:4010, flags:0
// cuTexRefSetAddress, cudaVersion:3020, flags:0
// cuMipmappedArrayGetMemoryRequirements, cudaVersion:11060, flags:0
// cuArrayGetMemoryRequirements, cudaVersion:11060, flags:0
// cuMipmappedArrayGetSparseProperties, cudaVersion:11010, flags:0
// cuArray3DGetDescriptor, cudaVersion:3020, flags:0
// cuArray3DCreate, cudaVersion:3020, flags:0
// cuArrayGetPlane, cudaVersion:11020, flags:0
// cuArrayGetSparseProperties, cudaVersion:11010, flags:0
// cuArrayGetDescriptor, cudaVersion:3020, flags:0
// cuArrayCreate, cudaVersion:3020, flags:0
// cuMemsetD2D8, cudaVersion:7000, flags:2
// cuMemsetD2D8, cudaVersion:3020, flags:0
// cuMemsetD8, cudaVersion:7000, flags:2
// cuMemsetD8, cudaVersion:3020, flags:0
// cuMemcpy3DAsync, cudaVersion:7000, flags:2
// cuMemcpy3DAsync, cudaVersion:3020, flags:0
// cuMemcpy3D, cudaVersion:7000, flags:2
// cuMemcpy3D, cudaVersion:3020, flags:0
// cuMemcpy2DAsync, cudaVersion:7000, flags:2
// cuMemcpy2DAsync, cudaVersion:3020, flags:0
// cuMemcpy2DUnaligned, cudaVersion:7000, flags:2
// cuMemcpy2DUnaligned, cudaVersion:3020, flags:0
// cuMemcpyDtoDAsync, cudaVersion:7000, flags:2
// cuMemcpyDtoDAsync, cudaVersion:3020, flags:0
// cuMemcpyDtoD, cudaVersion:7000, flags:2
// cuMemcpyDtoD, cudaVersion:3020, flags:0
// cuMemcpyDtoHAsync, cudaVersion:7000, flags:2
// cuMemcpyDtoHAsync, cudaVersion:3020, flags:0
// cuMemcpyDtoH, cudaVersion:7000, flags:2
// cuMemcpyDtoH, cudaVersion:3020, flags:0
// cuMemcpyHtoDAsync, cudaVersion:7000, flags:2
// cuMemcpyHtoDAsync, cudaVersion:3020, flags:0
// cuMemcpyHtoD, cudaVersion:7000, flags:2
// cuMemcpyHtoD, cudaVersion:3020, flags:0
// cuMemPoolImportPointer, cudaVersion:11020, flags:0
// cuMemPoolExportPointer, cudaVersion:11020, flags:0
// cuMemPoolImportFromShareableHandle, cudaVersion:11020, flags:0
// cuMemPoolExportToShareableHandle, cudaVersion:11020, flags:0
// cuMemPoolDestroy, cudaVersion:11020, flags:0
// cuMemPoolCreate, cudaVersion:11020, flags:0
// cuMemPoolGetAccess, cudaVersion:11020, flags:0
// cuMemPoolSetAccess, cudaVersion:11020, flags:0
// cuMemPoolGetAttribute, cudaVersion:11020, flags:0
// cuMemPoolSetAttribute, cudaVersion:11020, flags:0
// cuMemPoolTrimTo, cudaVersion:11020, flags:0
// cuMemFreeAsync, cudaVersion:11020, flags:2
// cuMemFreeAsync, cudaVersion:11020, flags:0
// CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);

// cuMemAllocFromPoolAsync, cudaVersion:11020, flags:2
// cuMemAllocFromPoolAsync, cudaVersion:11020, flags:0
// CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);

// cuMemAllocAsync, cudaVersion:11020, flags:2
// cuMemAllocAsync, cudaVersion:11020, flags:0
// CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
// cuMemHostRegister, cudaVersion:6050, flags:0
// CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
// cuMemHostGetDevicePointer, cudaVersion:3020, flags:0
// CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags);
// cuMemGetAddressRange, cudaVersion:3020, flags:0
// CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);
// cuMemFree, cudaVersion:3020, flags:0
// CUresult CUDAAPI cuMemFree(CUdeviceptr dptr);
// cuMemAllocPitch, cudaVersion:3020, flags:0
// CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
// cuMemAlloc, cudaVersion:3020, flags:0
// CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
// cuMemGetInfo, cudaVersion:3020, flags:0
// CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total);
// cuLinkAddFile, cudaVersion:5050, flags:0
// CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path,unsigned int numOptions, CUjit_option *options, void **optionValues);
// cuLinkAddData, cudaVersion:5050, flags:0
// CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name,unsigned int numOptions, CUjit_option *options, void **optionValues);
// cuLinkCreate, cudaVersion:5050, flags:0
// CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut);
// cuModuleGetGlobal, cudaVersion:3020, flags:0
// CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
// cuCtxPushCurrent, cudaVersion:4000, flags:0
// CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx);
// cuCtxPopCurrent, cudaVersion:4000, flags:0
// CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx);
// cuCtxResetPersistingL2Cache, cudaVersion:11000, flags:0
// CUresult CUDAAPI cuCtxResetPersistingL2Cache(void);
// cuCtxCreate, cudaVersion:3020, flags:0
// CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
// cuDevicePrimaryCtxReset, cudaVersion:7000, flags:0
// CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev);
// cuDevicePrimaryCtxSetFlags, cudaVersion:11000, flags:0  === ok
// CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
// cuDevicePrimaryCtxRelease, cudaVersion:7000, flags:0 === ok
// CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev);
// cuFlushGPUDirectRDMAWrites, cudaVersion:11030, flags:0 === ok
// cuDeviceGetMemPool, cudaVersion:11020, flags:0  === ok
// cuDeviceSetMemPool, cudaVersion:11020, flags:0  === ok
// cuDeviceGetDefaultMemPool, cudaVersion:11020, flags:0   === ok
// cuDeviceGetTexture1DLinearMaxWidth, cudaVersion:11010, flags:0 === ok
// cuDeviceTotalMem, cudaVersion:3020, flags:0 === ok


#endif // CUDA_AUTOGEN_WRAPPERS_H
