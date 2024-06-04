#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <map>  // add by tian01.liu 2023.1.28
#include "common.h"
#include "switch_context.h"
#include "upper-half-wrappers.h"
#include "upper-half-cuda-wrappers.h"
#include "cublas_v1_wrappers.h"
#include "log_and_replay.h"

typedef void* (*lhCUDAtype_t)(void *cuType, int type);

void check_cublas_map(void* newObject, void* cudaType, int type, lhCUDAtype_t getCUDAtype)
{
  void* pObj = getCUDAtype(cudaType, type);
  switch(type)
  {
    case CUDA_STREAM:
    {
      if (NULL == pObj)
      {
        //printf("Get cudaStream_t old object.\n");
        *(cudaStream_t*)newObject = *((cudaStream_t*)cudaType);
      }
      else
      {
        //printf("Get new object.\n");
        *(cudaStream_t*)newObject = *((cudaStream_t*)pObj);
      }
      break;
    }
    default:
      assert(false);
  }
}

//bia.xing@samsung.com add new cublas APIs
#undef cublasInit
extern "C" cublasStatus cublasInit () {
  typedef cublasStatus (*cublasInit_t)();
  cublasStatus ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasInit)();
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasShutdown
extern "C" cublasStatus cublasShutdown () {
  typedef cublasStatus (*cublasShutdown_t)();
  cublasStatus ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasShutdown)();
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasGetError
extern "C" cublasStatus cublasGetError () {
  typedef cublasStatus (*cublasGetError_t)();
  cublasStatus ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasGetError)();
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#if 0
#undef cublasAlloc
extern "C" cublasStatus cublasAlloc (int n, int elemSize, void **devicePtr) {
  typedef cublasStatus (*cublasAlloc_t)(int n, int elemSize, void **devicePtr);
  cublasStatus ret_val ;
  printf("Called at func1 '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasAlloc)(n, elemSize, devicePtr);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cublasAlloc, n, elemSize, devicePtr, ret_val);
  printf("devicePtr.......: %p\n", *devicePtr);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasFree
extern "C" cublasStatus cublasFree (void *devicePtr) {
  typedef cublasStatus (*cublasFree_t)(void *devicePtr);
  cublasStatus ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasFree)(devicePtr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
#else
#undef cublasAlloc
extern "C" cublasStatus cublasAlloc (int n, int elemSize, void **devicePtr) {
  // typedef cublasStatus (*cublasAlloc_t)(int n, int elemSize, void **devicePtr);
  cublasStatus ret_val = CUBLAS_STATUS_SUCCESS;
  // printf("Called at func1 '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  LhMMUAllocate_t func = (LhMMUAllocate_t)lhInfo.lhMMUAllocFptr;
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  *devicePtr = func(n * elemSize);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cublasAlloc, n, elemSize, devicePtr, ret_val);
  // printf("devicePtr.......: %p\n", *devicePtr);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasFree
extern "C" cublasStatus cublasFree (void *devicePtr) {
  // typedef cublasStatus (*cublasFree_t)(void *devicePtr);
  cublasStatus ret_val = CUBLAS_STATUS_SUCCESS;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  LhMMUFree_t func = (LhMMUFree_t)lhInfo.lhMMUFreeFptr;
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  func(devicePtr);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cublasFree, devicePtr, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
#endif

#undef cublasSetKernelStream
extern "C" cublasStatus cublasSetKernelStream (cudaStream_t stream) {
  typedef cublasStatus (*cublasSetKernelStream_t)(cudaStream_t stream);
  cublasStatus ret_val ;
//   printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  cudaStream_t newStream;
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cublas_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasSetKernelStream)(newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSnrm2
extern "C" float cublasSnrm2 (int n, const float *x, int incx) {
  typedef float (*cublasSnrm2_t)(int n, const float *x, int incx);
  float ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasSnrm2)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDnrm2
extern "C" double cublasDnrm2 (int n, const double *x, int incx) {
  typedef double (*cublasDnrm2_t)(int n, const double *x, int incx);
  double ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasDnrm2)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasScnrm2
extern "C" float cublasScnrm2 (int n, const cuComplex *x, int incx) {
  typedef float (*cublasScnrm2_t)(int n, const cuComplex *x, int incx);
  float ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasScnrm2)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDznrm2
extern "C" double cublasDznrm2 (int n, const cuDoubleComplex *x, int incx) {
  typedef double (*cublasDznrm2_t)(int n, const cuDoubleComplex *x, int incx);
  double ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasDznrm2)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSdot
extern "C" float cublasSdot (int n, const float *x, int incx, const float *y,  int incy) {
  typedef float (*cublasSdot_t)(int n, const float *x, int incx, const float *y,  int incy);
  float ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasSdot)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDdot
extern "C" double cublasDdot (int n, const double *x, int incx, const double *y,  int incy) {
  typedef double (*cublasDdot_t)(int n, const double *x, int incx, const double *y,  int incy);
  double ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasDdot)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasCdotu
extern "C" cuComplex cublasCdotu (int n, const cuComplex *x, int incx, const cuComplex *y,  int incy) {
  typedef cuComplex (*cublasCdotu_t)(int n, const cuComplex *x, int incx, const cuComplex *y,  int incy);
  cuComplex ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasCdotu)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasCdotc
extern "C" cuComplex cublasCdotc (int n, const cuComplex *x, int incx, const cuComplex *y,  int incy) {
  typedef cuComplex (*cublasCdotc_t)(int n, const cuComplex *x, int incx, const cuComplex *y,  int incy);
  cuComplex ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasCdotc)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasZdotu
extern "C" cuDoubleComplex cublasZdotu (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,  int incy) {
  typedef cuDoubleComplex (*cublasZdotu_t)(int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,  int incy);
  cuDoubleComplex ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasZdotu)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasZdotc
extern "C" cuDoubleComplex cublasZdotc (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,  int incy) {
  typedef cuDoubleComplex (*cublasZdotc_t)(int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,  int incy);
  cuDoubleComplex ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasZdotc)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSscal
extern "C" void cublasSscal (int n, float alpha, float *x, int incx) {
  typedef void (*cublasSscal_t)(int n, float alpha, float *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSscal)(n, alpha, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDscal
extern "C" void cublasDscal (int n, double alpha, double *x, int incx) {
  typedef void (*cublasDscal_t)(int n, double alpha, double *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDscal)(n, alpha, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCscal
extern "C" void cublasCscal (int n, cuComplex alpha, cuComplex *x, int incx) {
  typedef void (*cublasCscal_t)(int n, cuComplex alpha, cuComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCscal)(n, alpha, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZscal
extern "C" void cublasZscal (int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx) {
  typedef void (*cublasZscal_t)(int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZscal)(n, alpha, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCsscal
extern "C" void cublasCsscal (int n, float alpha, cuComplex *x, int incx) {
  typedef void (*cublasCsscal_t)(int n, float alpha, cuComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCsscal)(n, alpha, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZdscal
extern "C" void cublasZdscal (int n, double alpha, cuDoubleComplex *x, int incx) {
  typedef void (*cublasZdscal_t)(int n, double alpha, cuDoubleComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZdscal)(n, alpha, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSaxpy
extern "C" void cublasSaxpy (int n, float alpha, const float *x, int incx,  float *y, int incy) {
  typedef void (*cublasSaxpy_t)(int n, float alpha, const float *x, int incx,  float *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSaxpy)(n, alpha, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDaxpy
extern "C" void cublasDaxpy (int n, double alpha, const double *x,  int incx, double *y, int incy) {
  typedef void (*cublasDaxpy_t)(int n, double alpha, const double *x,  int incx, double *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDaxpy)(n, alpha, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCaxpy
extern "C" void cublasCaxpy (int n, cuComplex alpha, const cuComplex *x,  int incx, cuComplex *y, int incy) {
  typedef void (*cublasCaxpy_t)(int n, cuComplex alpha, const cuComplex *x,  int incx, cuComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCaxpy)(n, alpha, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZaxpy
extern "C" void cublasZaxpy (int n, cuDoubleComplex alpha, const cuDoubleComplex *x,  int incx, cuDoubleComplex *y, int incy) {
  typedef void (*cublasZaxpy_t)(int n, cuDoubleComplex alpha, const cuDoubleComplex *x,  int incx, cuDoubleComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZaxpy)(n, alpha, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasScopy
extern "C" void cublasScopy (int n, const float *x, int incx, float *y,  int incy) {
  typedef void (*cublasScopy_t)(int n, const float *x, int incx, float *y,  int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasScopy)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDcopy
extern "C" void cublasDcopy (int n, const double *x, int incx, double *y,  int incy) {
  typedef void (*cublasDcopy_t)(int n, const double *x, int incx, double *y,  int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDcopy)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCcopy
extern "C" void cublasCcopy (int n, const cuComplex *x, int incx, cuComplex *y, int incy) {
  typedef void (*cublasCcopy_t)(int n, const cuComplex *x, int incx, cuComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCcopy)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZcopy
extern "C" void cublasZcopy (int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
  typedef void (*cublasZcopy_t)(int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZcopy)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSswap
extern "C" void cublasSswap (int n, float *x, int incx, float *y, int incy) {
  typedef void (*cublasSswap_t)(int n, float *x, int incx, float *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSswap)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDswap
extern "C" void cublasDswap (int n, double *x, int incx, double *y, int incy) {
  typedef void (*cublasDswap_t)(int n, double *x, int incx, double *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDswap)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCswap
extern "C" void cublasCswap (int n, cuComplex *x, int incx, cuComplex *y, int incy) {
  typedef void (*cublasCswap_t)(int n, cuComplex *x, int incx, cuComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCswap)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZswap
extern "C" void cublasZswap (int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
  typedef void (*cublasZswap_t)(int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZswap)(n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasIsamax
extern "C" int cublasIsamax (int n, const float *x, int incx) {
  typedef int (*cublasIsamax_t)(int n, const float *x, int incx);
  int ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasIsamax)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasIdamax
extern "C" int cublasIdamax (int n, const double *x, int incx) {
  typedef int (*cublasIdamax_t)(int n, const double *x, int incx);
  int ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasIdamax)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasIcamax
extern "C" int cublasIcamax (int n, const cuComplex *x, int incx) {
  typedef int (*cublasIcamax_t)(int n, const cuComplex *x, int incx);
  int ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasIcamax)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasIzamax
extern "C" int cublasIzamax (int n, const cuDoubleComplex *x, int incx) {
  typedef int (*cublasIzamax_t)(int n, const cuDoubleComplex *x, int incx);
  int ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasIzamax)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasIsamin
extern "C" int cublasIsamin (int n, const float *x, int incx) {
  typedef int (*cublasIsamin_t)(int n, const float *x, int incx);
  int ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasIsamin)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasIdamin
extern "C" int cublasIdamin (int n, const double *x, int incx) {
  typedef int (*cublasIdamin_t)(int n, const double *x, int incx);
  int ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasIdamin)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasIcamin
extern "C" int cublasIcamin (int n, const cuComplex *x, int incx) {
  typedef int (*cublasIcamin_t)(int n, const cuComplex *x, int incx);
  int ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasIcamin)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasIzamin
extern "C" int cublasIzamin (int n, const cuDoubleComplex *x, int incx) {
  typedef int (*cublasIzamin_t)(int n, const cuDoubleComplex *x, int incx);
  int ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasIzamin)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSasum
extern "C" float cublasSasum (int n, const float *x, int incx) {
  typedef float (*cublasSasum_t)(int n, const float *x, int incx);
  float ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasSasum)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDasum
extern "C" double cublasDasum (int n, const double *x, int incx) {
  typedef double (*cublasDasum_t)(int n, const double *x, int incx);
  double ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasDasum)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasScasum
extern "C" float cublasScasum (int n, const cuComplex *x, int incx) {
  typedef float (*cublasScasum_t)(int n, const cuComplex *x, int incx);
  float ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasScasum)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDzasum
extern "C" double cublasDzasum (int n, const cuDoubleComplex *x, int incx) {
  typedef double (*cublasDzasum_t)(int n, const cuDoubleComplex *x, int incx);
  double ret_val = 0;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasDzasum)(n, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSrot
extern "C" void cublasSrot (int n, float *x, int incx, float *y, int incy,  float sc, float ss) {
  typedef void (*cublasSrot_t)(int n, float *x, int incx, float *y, int incy,  float sc, float ss);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSrot)(n, x, incx, y, incy, sc, ss);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDrot
extern "C" void cublasDrot (int n, double *x, int incx, double *y, int incy,  double sc, double ss) {
  typedef void (*cublasDrot_t)(int n, double *x, int incx, double *y, int incy,  double sc, double ss);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDrot)(n, x, incx, y, incy, sc, ss);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCrot
extern "C" void cublasCrot (int n, cuComplex *x, int incx, cuComplex *y,  int incy, float c, cuComplex s) {
  typedef void (*cublasCrot_t)(int n, cuComplex *x, int incx, cuComplex *y,  int incy, float c, cuComplex s);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCrot)(n, x, incx, y, incy, c, s);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZrot
extern "C" void cublasZrot (int n, cuDoubleComplex *x, int incx,  cuDoubleComplex *y, int incy, double sc,  cuDoubleComplex cs) {
  typedef void (*cublasZrot_t)(int n, cuDoubleComplex *x, int incx,  cuDoubleComplex *y, int incy, double sc,  cuDoubleComplex cs);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZrot)(n, x, incx, y, incy, sc, cs);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCsrot
extern "C" void cublasCsrot (int n, cuComplex *x, int incx, cuComplex *y, int incy, float c, float s) {
  typedef void (*cublasCsrot_t)(int n, cuComplex *x, int incx, cuComplex *y, int incy, float c, float s);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCsrot)(n, x, incx, y, incy, c, s);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZdrot
extern "C" void cublasZdrot (int n, cuDoubleComplex *x, int incx,  cuDoubleComplex *y, int incy, double c, double s) {
  typedef void (*cublasZdrot_t)(int n, cuDoubleComplex *x, int incx,  cuDoubleComplex *y, int incy, double c, double s);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZdrot)(n, x, incx, y, incy, c, s);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSrotg
extern "C" void cublasSrotg (float *sa, float *sb, float *sc, float *ss) {
  typedef void (*cublasSrotg_t)(float *sa, float *sb, float *sc, float *ss);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSrotg)(sa, sb, sc, ss);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDrotg
extern "C" void cublasDrotg (double *sa, double *sb, double *sc, double *ss) {
  typedef void (*cublasDrotg_t)(double *sa, double *sb, double *sc, double *ss);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDrotg)(sa, sb, sc, ss);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCrotg
extern "C" void cublasCrotg (cuComplex *ca, cuComplex cb, float *sc, cuComplex *cs) {
  typedef void (*cublasCrotg_t)(cuComplex *ca, cuComplex cb, float *sc, cuComplex *cs);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCrotg)(ca, cb, sc, cs);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZrotg
extern "C" void cublasZrotg (cuDoubleComplex *ca, cuDoubleComplex cb, double *sc, cuDoubleComplex *cs) {
  typedef void (*cublasZrotg_t)(cuDoubleComplex *ca, cuDoubleComplex cb, double *sc, cuDoubleComplex *cs);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZrotg)(ca, cb, sc, cs);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSrotm
extern "C" void cublasSrotm(int n, float *x, int incx, float *y, int incy,  const float* sparam) {
  typedef void (*cublasSrotm_t)(int n, float *x, int incx, float *y, int incy,  const float* sparam);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSrotm)(n, x, incx, y, incy, sparam);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDrotm
extern "C" void cublasDrotm(int n, double *x, int incx, double *y, int incy,  const double* sparam) {
  typedef void (*cublasDrotm_t)(int n, double *x, int incx, double *y, int incy,  const double* sparam);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDrotm)(n, x, incx, y, incy, sparam);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSrotmg
extern "C" void cublasSrotmg (float *sd1, float *sd2, float *sx1,  const float *sy1, float* sparam) {
  typedef void (*cublasSrotmg_t)(float *sd1, float *sd2, float *sx1,  const float *sy1, float* sparam);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSrotmg)(sd1, sd2, sx1, sy1, sparam);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDrotmg
extern "C" void cublasDrotmg (double *sd1, double *sd2, double *sx1,  const double *sy1, double* sparam) {
  typedef void (*cublasDrotmg_t)(double *sd1, double *sd2, double *sx1,  const double *sy1, double* sparam);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDrotmg)(sd1, sd2, sx1, sy1, sparam);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSgemv
extern "C" void cublasSgemv (char trans, int m, int n, float alpha, const float *A, int lda, const float *x, int incx, float beta, float *y, int incy) {
  typedef void (*cublasSgemv_t)(char trans, int m, int n, float alpha, const float *A, int lda, const float *x, int incx, float beta, float *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSgemv)(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDgemv
extern "C" void cublasDgemv (char trans, int m, int n, double alpha, const double *A, int lda, const double *x, int incx, double beta, double *y, int incy) {
  typedef void (*cublasDgemv_t)(char trans, int m, int n, double alpha, const double *A, int lda, const double *x, int incx, double beta, double *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDgemv)(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCgemv
extern "C" void cublasCgemv (char trans, int m, int n, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy) {
  typedef void (*cublasCgemv_t)(char trans, int m, int n, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCgemv)(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZgemv
extern "C" void cublasZgemv (char trans, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy) {
  typedef void (*cublasZgemv_t)(char trans, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZgemv)(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSgbmv
extern "C" void cublasSgbmv (char trans, int m, int n, int kl, int ku,  float alpha, const float *A, int lda,  const float *x, int incx, float beta, float *y,  int incy) {
  typedef void (*cublasSgbmv_t)(char trans, int m, int n, int kl, int ku,  float alpha, const float *A, int lda,  const float *x, int incx, float beta, float *y,  int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSgbmv)(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDgbmv
extern "C" void cublasDgbmv (char trans, int m, int n, int kl, int ku,  double alpha, const double *A, int lda,  const double *x, int incx, double beta, double *y,  int incy) {
  typedef void (*cublasDgbmv_t)(char trans, int m, int n, int kl, int ku,  double alpha, const double *A, int lda,  const double *x, int incx, double beta, double *y,  int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDgbmv)(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCgbmv
extern "C" void cublasCgbmv (char trans, int m, int n, int kl, int ku,  cuComplex alpha, const cuComplex *A, int lda,  const cuComplex *x, int incx, cuComplex beta, cuComplex *y,  int incy) {
  typedef void (*cublasCgbmv_t)(char trans, int m, int n, int kl, int ku,  cuComplex alpha, const cuComplex *A, int lda,  const cuComplex *x, int incx, cuComplex beta, cuComplex *y,  int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCgbmv)(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZgbmv
extern "C" void cublasZgbmv (char trans, int m, int n, int kl, int ku,  cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,  const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y,  int incy) {
  typedef void (*cublasZgbmv_t)(char trans, int m, int n, int kl, int ku,  cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,  const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y,  int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZgbmv)(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasStrmv
extern "C" void cublasStrmv (char uplo, char trans, char diag, int n,  const float *A, int lda, float *x, int incx) {
  typedef void (*cublasStrmv_t)(char uplo, char trans, char diag, int n,  const float *A, int lda, float *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasStrmv)(uplo, trans, diag, n, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDtrmv
extern "C" void cublasDtrmv (char uplo, char trans, char diag, int n,  const double *A, int lda, double *x, int incx) {
  typedef void (*cublasDtrmv_t)(char uplo, char trans, char diag, int n,  const double *A, int lda, double *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDtrmv)(uplo, trans, diag, n, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCtrmv
extern "C" void cublasCtrmv (char uplo, char trans, char diag, int n,  const cuComplex *A, int lda, cuComplex *x, int incx) {
  typedef void (*cublasCtrmv_t)(char uplo, char trans, char diag, int n,  const cuComplex *A, int lda, cuComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCtrmv)(uplo, trans, diag, n, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZtrmv
extern "C" void cublasZtrmv (char uplo, char trans, char diag, int n,  const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
  typedef void (*cublasZtrmv_t)(char uplo, char trans, char diag, int n,  const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZtrmv)(uplo, trans, diag, n, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasStbmv
extern "C" void cublasStbmv (char uplo, char trans, char diag, int n, int k,  const float *A, int lda, float *x, int incx) {
  typedef void (*cublasStbmv_t)(char uplo, char trans, char diag, int n, int k,  const float *A, int lda, float *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasStbmv)(uplo, trans, diag, n, k, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDtbmv
extern "C" void cublasDtbmv (char uplo, char trans, char diag, int n, int k,  const double *A, int lda, double *x, int incx) {
  typedef void (*cublasDtbmv_t)(char uplo, char trans, char diag, int n, int k,  const double *A, int lda, double *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDtbmv)(uplo, trans, diag, n, k, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCtbmv
extern "C" void cublasCtbmv (char uplo, char trans, char diag, int n, int k,  const cuComplex *A, int lda, cuComplex *x, int incx) {
  typedef void (*cublasCtbmv_t)(char uplo, char trans, char diag, int n, int k,  const cuComplex *A, int lda, cuComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCtbmv)(uplo, trans, diag, n, k, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZtbmv
extern "C" void cublasZtbmv (char uplo, char trans, char diag, int n, int k,  const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
  typedef void (*cublasZtbmv_t)(char uplo, char trans, char diag, int n, int k,  const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZtbmv)(uplo, trans, diag, n, k, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasStpmv
extern "C" void cublasStpmv (char uplo, char trans, char diag, int n, const float *AP, float *x, int incx) {
  typedef void (*cublasStpmv_t)(char uplo, char trans, char diag, int n, const float *AP, float *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasStpmv)(uplo, trans, diag, n, AP, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDtpmv
extern "C" void cublasDtpmv (char uplo, char trans, char diag, int n, const double *AP, double *x, int incx) {
  typedef void (*cublasDtpmv_t)(char uplo, char trans, char diag, int n, const double *AP, double *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDtpmv)(uplo, trans, diag, n, AP, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCtpmv
extern "C" void cublasCtpmv (char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
  typedef void (*cublasCtpmv_t)(char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCtpmv)(uplo, trans, diag, n, AP, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZtpmv
extern "C" void cublasZtpmv (char uplo, char trans, char diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
  typedef void (*cublasZtpmv_t)(char uplo, char trans, char diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZtpmv)(uplo, trans, diag, n, AP, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasStrsv
extern "C" void cublasStrsv (char uplo, char trans, char diag, int n, const float *A, int lda, float *x, int incx) {
  typedef void (*cublasStrsv_t)(char uplo, char trans, char diag, int n, const float *A, int lda, float *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasStrsv)(uplo, trans, diag, n, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDtrsv
extern "C" void cublasDtrsv (char uplo, char trans, char diag, int n, const double *A, int lda, double *x, int incx) {
  typedef void (*cublasDtrsv_t)(char uplo, char trans, char diag, int n, const double *A, int lda, double *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDtrsv)(uplo, trans, diag, n, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCtrsv
extern "C" void cublasCtrsv (char uplo, char trans, char diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) {
  typedef void (*cublasCtrsv_t)(char uplo, char trans, char diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCtrsv)(uplo, trans, diag, n, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZtrsv
extern "C" void cublasZtrsv (char uplo, char trans, char diag, int n, const cuDoubleComplex *A, int lda,  cuDoubleComplex *x, int incx) {
  typedef void (*cublasZtrsv_t)(char uplo, char trans, char diag, int n, const cuDoubleComplex *A, int lda,  cuDoubleComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZtrsv)(uplo, trans, diag, n, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasStpsv
extern "C" void cublasStpsv (char uplo, char trans, char diag, int n, const float *AP,  float *x, int incx) {
  typedef void (*cublasStpsv_t)(char uplo, char trans, char diag, int n, const float *AP,  float *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasStpsv)(uplo, trans, diag, n, AP, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDtpsv
extern "C" void cublasDtpsv (char uplo, char trans, char diag, int n, const double *AP, double *x, int incx) {
  typedef void (*cublasDtpsv_t)(char uplo, char trans, char diag, int n, const double *AP, double *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDtpsv)(uplo, trans, diag, n, AP, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCtpsv
extern "C" void cublasCtpsv (char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
  typedef void (*cublasCtpsv_t)(char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCtpsv)(uplo, trans, diag, n, AP, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZtpsv
extern "C" void cublasZtpsv (char uplo, char trans, char diag, int n, const cuDoubleComplex *AP,  cuDoubleComplex *x, int incx) {
  typedef void (*cublasZtpsv_t)(char uplo, char trans, char diag, int n, const cuDoubleComplex *AP,  cuDoubleComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZtpsv)(uplo, trans, diag, n, AP, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasStbsv
extern "C" void cublasStbsv (char uplo, char trans,  char diag, int n, int k, const float *A,  int lda, float *x, int incx) {
  typedef void (*cublasStbsv_t)(char uplo, char trans,  char diag, int n, int k, const float *A,  int lda, float *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasStbsv)(uplo, trans, diag, n, k, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDtbsv
extern "C" void cublasDtbsv (char uplo, char trans,  char diag, int n, int k, const double *A,  int lda, double *x, int incx) {
  typedef void (*cublasDtbsv_t)(char uplo, char trans,  char diag, int n, int k, const double *A,  int lda, double *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDtbsv)(uplo, trans, diag, n, k, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCtbsv
extern "C" void cublasCtbsv (char uplo, char trans,  char diag, int n, int k, const cuComplex *A,  int lda, cuComplex *x, int incx) {
  typedef void (*cublasCtbsv_t)(char uplo, char trans,  char diag, int n, int k, const cuComplex *A,  int lda, cuComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCtbsv)(uplo, trans, diag, n, k, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZtbsv
extern "C" void cublasZtbsv (char uplo, char trans,  char diag, int n, int k, const cuDoubleComplex *A,  int lda, cuDoubleComplex *x, int incx) {
  typedef void (*cublasZtbsv_t)(char uplo, char trans,  char diag, int n, int k, const cuDoubleComplex *A,  int lda, cuDoubleComplex *x, int incx);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZtbsv)(uplo, trans, diag, n, k, A, lda, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSsymv
extern "C" void cublasSsymv (char uplo, int n, float alpha, const float *A, int lda, const float *x, int incx, float beta,  float *y, int incy) {
  typedef void (*cublasSsymv_t)(char uplo, int n, float alpha, const float *A, int lda, const float *x, int incx, float beta,  float *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSsymv)(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDsymv
extern "C" void cublasDsymv (char uplo, int n, double alpha, const double *A, int lda, const double *x, int incx, double beta,  double *y, int incy) {
  typedef void (*cublasDsymv_t)(char uplo, int n, double alpha, const double *A, int lda, const double *x, int incx, double beta,  double *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDsymv)(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasChemv
extern "C" void cublasChemv (char uplo, int n, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex beta,  cuComplex *y, int incy) {
  typedef void (*cublasChemv_t)(char uplo, int n, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex beta,  cuComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasChemv)(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZhemv
extern "C" void cublasZhemv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta,  cuDoubleComplex *y, int incy) {
  typedef void (*cublasZhemv_t)(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta,  cuDoubleComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZhemv)(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSsbmv
extern "C" void cublasSsbmv (char uplo, int n, int k, float alpha,  const float *A, int lda, const float *x, int incx,  float beta, float *y, int incy) {
  typedef void (*cublasSsbmv_t)(char uplo, int n, int k, float alpha,  const float *A, int lda, const float *x, int incx,  float beta, float *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSsbmv)(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDsbmv
extern "C" void cublasDsbmv (char uplo, int n, int k, double alpha,  const double *A, int lda, const double *x, int incx,  double beta, double *y, int incy) {
  typedef void (*cublasDsbmv_t)(char uplo, int n, int k, double alpha,  const double *A, int lda, const double *x, int incx,  double beta, double *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDsbmv)(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasChbmv
extern "C" void cublasChbmv (char uplo, int n, int k, cuComplex alpha,  const cuComplex *A, int lda, const cuComplex *x, int incx,  cuComplex beta, cuComplex *y, int incy) {
  typedef void (*cublasChbmv_t)(char uplo, int n, int k, cuComplex alpha,  const cuComplex *A, int lda, const cuComplex *x, int incx,  cuComplex beta, cuComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasChbmv)(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZhbmv
extern "C" void cublasZhbmv (char uplo, int n, int k, cuDoubleComplex alpha,  const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,  cuDoubleComplex beta, cuDoubleComplex *y, int incy) {
  typedef void (*cublasZhbmv_t)(char uplo, int n, int k, cuDoubleComplex alpha,  const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,  cuDoubleComplex beta, cuDoubleComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZhbmv)(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSspmv
extern "C" void cublasSspmv (char uplo, int n, float alpha, const float *AP, const float *x, int incx, float beta, float *y, int incy) {
  typedef void (*cublasSspmv_t)(char uplo, int n, float alpha, const float *AP, const float *x, int incx, float beta, float *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSspmv)(uplo, n, alpha, AP, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDspmv
extern "C" void cublasDspmv(char uplo, int n, double alpha, const double *AP, const double *x, int incx, double beta, double *y, int incy) {
  typedef void (*cublasDspmv_t)(char uplo, int n, double alpha, const double *AP, const double *x, int incx, double beta, double *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDspmv)(uplo, n, alpha, AP, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasChpmv
extern "C" void cublasChpmv (char uplo, int n, cuComplex alpha, const cuComplex *AP, const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy) {
  typedef void (*cublasChpmv_t)(char uplo, int n, cuComplex alpha, const cuComplex *AP, const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasChpmv)(uplo, n, alpha, AP, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZhpmv
extern "C" void cublasZhpmv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy) {
  typedef void (*cublasZhpmv_t)(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZhpmv)(uplo, n, alpha, AP, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSger
extern "C" void cublasSger (int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
  typedef void (*cublasSger_t)(int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSger)(m, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDger
extern "C" void cublasDger (int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
  typedef void (*cublasDger_t)(int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDger)(m, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCgeru
extern "C" void cublasCgeru (int m, int n, cuComplex alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
  typedef void (*cublasCgeru_t)(int m, int n, cuComplex alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCgeru)(m, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCgerc
extern "C" void cublasCgerc (int m, int n, cuComplex alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
  typedef void (*cublasCgerc_t)(int m, int n, cuComplex alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCgerc)(m, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZgeru
extern "C" void cublasZgeru (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
  typedef void (*cublasZgeru_t)(int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZgeru)(m, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZgerc
extern "C" void cublasZgerc (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
  typedef void (*cublasZgerc_t)(int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZgerc)(m, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSsyr
extern "C" void cublasSsyr (char uplo, int n, float alpha, const float *x, int incx, float *A, int lda) {
  typedef void (*cublasSsyr_t)(char uplo, int n, float alpha, const float *x, int incx, float *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSsyr)(uplo, n, alpha, x, incx, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDsyr
extern "C" void cublasDsyr (char uplo, int n, double alpha, const double *x, int incx, double *A, int lda) {
  typedef void (*cublasDsyr_t)(char uplo, int n, double alpha, const double *x, int incx, double *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDsyr)(uplo, n, alpha, x, incx, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCher
extern "C" void cublasCher (char uplo, int n, float alpha,  const cuComplex *x, int incx, cuComplex *A, int lda) {
  typedef void (*cublasCher_t)(char uplo, int n, float alpha,  const cuComplex *x, int incx, cuComplex *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCher)(uplo, n, alpha, x, incx, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZher
extern "C" void cublasZher (char uplo, int n, double alpha,  const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda) {
  typedef void (*cublasZher_t)(char uplo, int n, double alpha,  const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZher)(uplo, n, alpha, x, incx, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSspr
extern "C" void cublasSspr (char uplo, int n, float alpha, const float *x, int incx, float *AP) {
  typedef void (*cublasSspr_t)(char uplo, int n, float alpha, const float *x, int incx, float *AP);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSspr)(uplo, n, alpha, x, incx, AP);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDspr
extern "C" void cublasDspr (char uplo, int n, double alpha, const double *x, int incx, double *AP) {
  typedef void (*cublasDspr_t)(char uplo, int n, double alpha, const double *x, int incx, double *AP);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDspr)(uplo, n, alpha, x, incx, AP);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasChpr
extern "C" void cublasChpr (char uplo, int n, float alpha, const cuComplex *x, int incx, cuComplex *AP) {
  typedef void (*cublasChpr_t)(char uplo, int n, float alpha, const cuComplex *x, int incx, cuComplex *AP);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasChpr)(uplo, n, alpha, x, incx, AP);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZhpr
extern "C" void cublasZhpr (char uplo, int n, double alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP) {
  typedef void (*cublasZhpr_t)(char uplo, int n, double alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZhpr)(uplo, n, alpha, x, incx, AP);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSsyr2
extern "C" void cublasSsyr2 (char uplo, int n, float alpha, const float *x,  int incx, const float *y, int incy, float *A,  int lda) {
  typedef void (*cublasSsyr2_t)(char uplo, int n, float alpha, const float *x,  int incx, const float *y, int incy, float *A,  int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSsyr2)(uplo, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDsyr2
extern "C" void cublasDsyr2 (char uplo, int n, double alpha, const double *x,  int incx, const double *y, int incy, double *A,  int lda) {
  typedef void (*cublasDsyr2_t)(char uplo, int n, double alpha, const double *x,  int incx, const double *y, int incy, double *A,  int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDsyr2)(uplo, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCher2
extern "C" void cublasCher2 (char uplo, int n, cuComplex alpha, const cuComplex *x,  int incx, const cuComplex *y, int incy, cuComplex *A,  int lda) {
  typedef void (*cublasCher2_t)(char uplo, int n, cuComplex alpha, const cuComplex *x,  int incx, const cuComplex *y, int incy, cuComplex *A,  int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCher2)(uplo, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZher2
extern "C" void cublasZher2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x,  int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A,  int lda) {
  typedef void (*cublasZher2_t)(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x,  int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A,  int lda);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZher2)(uplo, n, alpha, x, incx, y, incy, A, lda);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSspr2
extern "C" void cublasSspr2 (char uplo, int n, float alpha, const float *x,  int incx, const float *y, int incy, float *AP) {
  typedef void (*cublasSspr2_t)(char uplo, int n, float alpha, const float *x,  int incx, const float *y, int incy, float *AP);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSspr2)(uplo, n, alpha, x, incx, y, incy, AP);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDspr2
extern "C" void cublasDspr2 (char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy, double *AP) {
  typedef void (*cublasDspr2_t)(char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy, double *AP);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDspr2)(uplo, n, alpha, x, incx, y, incy, AP);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasChpr2
extern "C" void cublasChpr2 (char uplo, int n, cuComplex alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP) {
  typedef void (*cublasChpr2_t)(char uplo, int n, cuComplex alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasChpr2)(uplo, n, alpha, x, incx, y, incy, AP);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZhpr2
extern "C" void cublasZhpr2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP) {
  typedef void (*cublasZhpr2_t)(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZhpr2)(uplo, n, alpha, x, incx, y, incy, AP);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSgemm
extern "C" void cublasSgemm (char transa, char transb, int m, int n, int k,  float alpha, const float *A, int lda,  const float *B, int ldb, float beta, float *C,  int ldc) {
  typedef void (*cublasSgemm_t)(char transa, char transb, int m, int n, int k,  float alpha, const float *A, int lda,  const float *B, int ldb, float beta, float *C,  int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSgemm)(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDgemm
extern "C" void cublasDgemm (char transa, char transb, int m, int n, int k, double alpha, const double *A, int lda,  const double *B, int ldb, double beta, double *C,  int ldc) {
  typedef void (*cublasDgemm_t)(char transa, char transb, int m, int n, int k, double alpha, const double *A, int lda,  const double *B, int ldb, double beta, double *C,  int ldc);
//   printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDgemm)(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCgemm
extern "C" void cublasCgemm (char transa, char transb, int m, int n, int k,  cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc) {
  typedef void (*cublasCgemm_t)(char transa, char transb, int m, int n, int k,  cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCgemm)(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZgemm
extern "C" void cublasZgemm (char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
  typedef void (*cublasZgemm_t)(char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
//   printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZgemm)(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSsyrk
extern "C" void cublasSsyrk (char uplo, char trans, int n, int k, float alpha,  const float *A, int lda, float beta, float *C,  int ldc) {
  typedef void (*cublasSsyrk_t)(char uplo, char trans, int n, int k, float alpha,  const float *A, int lda, float beta, float *C,  int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSsyrk)(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDsyrk
extern "C" void cublasDsyrk (char uplo, char trans, int n, int k, double alpha, const double *A, int lda, double beta, double *C, int ldc) {
  typedef void (*cublasDsyrk_t)(char uplo, char trans, int n, int k, double alpha, const double *A, int lda, double beta, double *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDsyrk)(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCsyrk
extern "C" void cublasCsyrk (char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A, int lda, cuComplex beta, cuComplex *C, int ldc) {
  typedef void (*cublasCsyrk_t)(char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A, int lda, cuComplex beta, cuComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCsyrk)(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZsyrk
extern "C" void cublasZsyrk (char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
  typedef void (*cublasZsyrk_t)(char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZsyrk)(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCherk
extern "C" void cublasCherk (char uplo, char trans, int n, int k, float alpha, const cuComplex *A, int lda, float beta, cuComplex *C, int ldc) {
  typedef void (*cublasCherk_t)(char uplo, char trans, int n, int k, float alpha, const cuComplex *A, int lda, float beta, cuComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCherk)(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZherk
extern "C" void cublasZherk (char uplo, char trans, int n, int k, double alpha, const cuDoubleComplex *A, int lda, double beta, cuDoubleComplex *C, int ldc) {
  typedef void (*cublasZherk_t)(char uplo, char trans, int n, int k, double alpha, const cuDoubleComplex *A, int lda, double beta, cuDoubleComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZherk)(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSsyr2k
extern "C" void cublasSsyr2k (char uplo, char trans, int n, int k, float alpha,  const float *A, int lda, const float *B, int ldb,  float beta, float *C, int ldc) {
  typedef void (*cublasSsyr2k_t)(char uplo, char trans, int n, int k, float alpha,  const float *A, int lda, const float *B, int ldb,  float beta, float *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSsyr2k)(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDsyr2k
extern "C" void cublasDsyr2k (char uplo, char trans, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) {
  typedef void (*cublasDsyr2k_t)(char uplo, char trans, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDsyr2k)(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCsyr2k
extern "C" void cublasCsyr2k (char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc) {
  typedef void (*cublasCsyr2k_t)(char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCsyr2k)(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZsyr2k
extern "C" void cublasZsyr2k (char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
  typedef void (*cublasZsyr2k_t)(char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZsyr2k)(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCher2k
extern "C" void cublasCher2k (char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, float beta, cuComplex *C, int ldc) {
  typedef void (*cublasCher2k_t)(char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, float beta, cuComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCher2k)(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZher2k
extern "C" void cublasZher2k (char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, double beta, cuDoubleComplex *C, int ldc) {
  typedef void (*cublasZher2k_t)(char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, double beta, cuDoubleComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZher2k)(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSsymm
extern "C" void cublasSsymm (char side, char uplo, int m, int n, float alpha,  const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
  typedef void (*cublasSsymm_t)(char side, char uplo, int m, int n, float alpha,  const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasSsymm)(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDsymm
extern "C" void cublasDsymm (char side, char uplo, int m, int n, double alpha,  const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) {
  typedef void (*cublasDsymm_t)(char side, char uplo, int m, int n, double alpha,  const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDsymm)(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCsymm
extern "C" void cublasCsymm (char side, char uplo, int m, int n, cuComplex alpha,  const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc) {
  typedef void (*cublasCsymm_t)(char side, char uplo, int m, int n, cuComplex alpha,  const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCsymm)(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZsymm
extern "C" void cublasZsymm (char side, char uplo, int m, int n, cuDoubleComplex alpha,  const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
  typedef void (*cublasZsymm_t)(char side, char uplo, int m, int n, cuDoubleComplex alpha,  const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZsymm)(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasChemm
extern "C" void cublasChemm (char side, char uplo, int m, int n, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc) {
  typedef void (*cublasChemm_t)(char side, char uplo, int m, int n, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasChemm)(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZhemm
extern "C" void cublasZhemm (char side, char uplo, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
  typedef void (*cublasZhemm_t)(char side, char uplo, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZhemm)(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasStrsm
extern "C" void cublasStrsm (char side, char uplo, char transa, char diag, int m, int n, float alpha, const float *A, int lda, float *B, int ldb) {
  typedef void (*cublasStrsm_t)(char side, char uplo, char transa, char diag, int m, int n, float alpha, const float *A, int lda, float *B, int ldb);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasStrsm)(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDtrsm
extern "C" void cublasDtrsm (char side, char uplo, char transa, char diag, int m, int n, double alpha, const double *A, int lda, double *B, int ldb) {
  typedef void (*cublasDtrsm_t)(char side, char uplo, char transa, char diag, int m, int n, double alpha, const double *A, int lda, double *B, int ldb);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDtrsm)(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCtrsm
extern "C" void cublasCtrsm (char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, const cuComplex *A, int lda, cuComplex *B, int ldb) {
  typedef void (*cublasCtrsm_t)(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, const cuComplex *A, int lda, cuComplex *B, int ldb);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCtrsm)(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZtrsm
extern "C" void cublasZtrsm (char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb) {
  typedef void (*cublasZtrsm_t)(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZtrsm)(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasStrmm
extern "C" void cublasStrmm (char side, char uplo, char transa, char diag, int m, int n, float alpha, const float *A, int lda, float *B, int ldb) {
  typedef void (*cublasStrmm_t)(char side, char uplo, char transa, char diag, int m, int n, float alpha, const float *A, int lda, float *B, int ldb);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasStrmm)(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasDtrmm
extern "C" void cublasDtrmm (char side, char uplo, char transa, char diag, int m, int n, double alpha, const double *A, int lda, double *B, int ldb) {
  typedef void (*cublasDtrmm_t)(char side, char uplo, char transa, char diag, int m, int n, double alpha, const double *A, int lda, double *B, int ldb);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasDtrmm)(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasCtrmm
extern "C" void cublasCtrmm (char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, const cuComplex *A, int lda, cuComplex *B, int ldb) {
  typedef void (*cublasCtrmm_t)(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, const cuComplex *A, int lda, cuComplex *B, int ldb);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasCtrmm)(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasZtrmm
extern "C" void cublasZtrmm (char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb) {
  typedef void (*cublasZtrmm_t)(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb);
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  REAL_FNC(cublasZtrmm)(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cublasSetMatrix
extern "C" cublasStatus_t cublasSetMatrix (int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb) {
  typedef cublasStatus_t (*cublasSetMatrix_t)(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb);
  cublasStatus_t ret_val ;
//   printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasSetMatrix)(rows, cols, elemSize, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasGetMatrix
extern "C" cublasStatus_t cublasGetMatrix (int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb) {
  typedef cublasStatus_t (*cublasGetMatrix_t)(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb);
  cublasStatus_t ret_val ;
//   printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasGetMatrix)(rows, cols, elemSize, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSetMatrixAsync
extern "C" cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasSetMatrixAsync_t)(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream);
  cublasStatus_t ret_val ;
//   printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  cudaStream_t newStream;
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cublas_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasSetMatrixAsync)(rows, cols, elemSize, A, lda, B, ldb, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasGetMatrixAsync
extern "C" cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasGetMatrixAsync_t)(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream);
  cublasStatus_t ret_val ;
//   printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  cudaStream_t newStream;
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cublas_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasGetMatrixAsync)(rows, cols, elemSize, A, lda, B, ldb, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSetVector
extern "C" cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
  typedef cublasStatus_t (*cublasSetVector_t)(int n, int elemSize, const void *x, int incx, void *y, int incy);
  cublasStatus_t ret_val ;
//   printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasSetVector)(n, elemSize, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasGetVector
extern "C" cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
  typedef cublasStatus_t (*cublasGetVector_t)(int n, int elemSize, const void *x, int incx, void *y, int incy);
  cublasStatus_t ret_val ;
//   printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  ret_val = REAL_FNC(cublasGetVector)(n, elemSize, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSetVectorAsync
extern "C" cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasSetVectorAsync_t)(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream);
  cublasStatus_t ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  cudaStream_t newStream;
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cublas_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasSetVectorAsync)(n, elemSize, hostPtr, incx, devicePtr, incy, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasGetVectorAsync
extern "C" cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasGetVectorAsync_t)(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream);
  cublasStatus_t ret_val ;
  printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
  cudaStream_t newStream;
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cublas_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasGetVectorAsync)(n, elemSize, devicePtr, incx, hostPtr, incy, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* ----------- CUSPARSE APIs ------------ */