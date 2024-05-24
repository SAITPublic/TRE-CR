/****************************************************************************
 *   Copyright (C) 2019-2020 by Twinkle Jain, and Gene Cooperman            *
 *   jain.t@husky.neu.edu, gene@ccs.neu.edu                                 *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <assert.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <pthread.h> // test code, by tian01.liu
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <map>

#include "getmmap.h"
#include "common.h"
#include "mem-restore.h"

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_INTERNAL_COMPILATION__
// #define __CUDACC__
#endif
#include "crt/host_runtime.h"
#include "crt/device_functions.h"
#include "log_and_replay.h"
#include "procmapsutils.h"

#include <iostream>

void **new_fatCubinHandle = NULL;

typedef struct ptxLog
{
  void *addr;
  long len;
  void *ptx;
} ptxlog;
ptxlog ptxlg[10];

extern MtcpHeader mtcpHdr;

// by tian01.liu for ipc replay, 2023.8.15
extern bool g_finishRelay;

// by tian01.liu for nccl replay, 2023.9.6
bool isUniqueIdGet = false;
// key: original hash of uniqueId, value: the new uniqueId
std::map<unsigned long long, ncclUniqueId*> uniqueIdMap;

std::map<CUcontext, CUcontext *> contextmap;
std::map<CUmodule, CUmodule *> modulemap;
std::map<CUfunction, CUfunction *> functionmap;
std::map<CUevent, CUevent *> eventmap;
std::map<CUstream, CUstream *> streammap;
std::map<CUdeviceptr, CUdeviceptr *> deviceptrmap;
std::map<CUtexref, CUtexref *> texrefmap;
#define RCEVENT 0
#define RCMODULE 1

void *getCUtype(void *cuType, int type)
{
  switch (type)
  {
  case CU_DEV_PTR:
  {
    if (deviceptrmap.size() == 0)
      return NULL;
    return (void *)deviceptrmap[*(CUdeviceptr *)cuType];
  }
  case CU_CONTEXT:
  {
    if (contextmap.size() == 0)
      return NULL;
    return (void *)contextmap[*(CUcontext *)cuType];
  }
  case CU_MODULE:
  {
    if (modulemap.size() == 0)
      return NULL;
    return (void *)modulemap[*(CUmodule *)cuType];
  }
  case CU_FUNCTION:
  {
    if (functionmap.size() == 0)
      return NULL;
    return (void *)functionmap[*(CUfunction *)cuType];
  }
  case CU_EVENT:
  {
    if (eventmap.size() == 0)
      return NULL;
    return (void *)eventmap[*(CUevent *)cuType];
  }
  case CU_STREAM:
  {
    if (streammap.size() == 0)
      return NULL;
    return (void *)streammap[*(CUstream *)cuType];
  }
  case CU_TEXREF:
  {
    if (texrefmap.size() == 0)
      return NULL;
    return (void *)texrefmap[*(CUtexref *)cuType];
  }
  default:
    assert(false);
  }
}

void freeCUtype(void *cuType, int type)
{
  switch (type)
  {
  case CU_DEV_PTR:
  {
    if (deviceptrmap.size() == 0)
      return;
    delete deviceptrmap[*(CUdeviceptr *)cuType];
    deviceptrmap.erase(*(CUdeviceptr *)cuType);
    break;
  }
  case CU_CONTEXT:
  {
    if (contextmap.size() == 0)
      return;
    delete contextmap[*(CUcontext *)cuType];
    contextmap.erase(*(CUcontext *)cuType);
    break;
  }
  case CU_MODULE:
  {
    if (modulemap.size() == 0)
      return;
    delete modulemap[*(CUmodule *)cuType];
    modulemap.erase(*(CUmodule *)cuType);
    break;
  }
  case CU_FUNCTION:
  {
    if (functionmap.size() == 0)
      return;
    delete functionmap[*(CUfunction *)cuType];
    functionmap.erase(*(CUfunction *)cuType);
    break;
  }
  case CU_EVENT:
  {
    if (eventmap.size() == 0)
      return;
    delete eventmap[*(CUevent *)cuType];
    eventmap.erase(*(CUevent *)cuType);
    break;
  }
  case CU_STREAM:
  {
    if (streammap.size() == 0)
      return;
    delete streammap[*(CUstream *)cuType];
    streammap.erase(*(CUstream *)cuType);
    break;
  }
  case CU_TEXREF:
  {
    if (texrefmap.size() == 0)
      return;
    delete texrefmap[*(CUtexref *)cuType];
    texrefmap.erase(*(CUtexref *)cuType);
    break;
  }
  default:
    assert(false);
  }
}

std::map<cublasHandle_t, cublasHandle_t *> cublasHandleMap;
void *getCublasType(void *cublasType, int type)
{
  switch (type)
  {
  case CUBLAS_HANDLE:
  {
    if (cublasHandleMap.size() == 0)
      return NULL;
    return (void *)cublasHandleMap[*(cublasHandle_t *)cublasType];
    break;
  }
  default:
    assert(false);
  }
  return NULL;
}
void freeCublasType(void *cublasType, int type)
{
  switch (type)
  {
  case CUBLAS_HANDLE:
  {
    if (cublasHandleMap.size() == 0)
      return;
    delete cublasHandleMap[*(cublasHandle_t *)cublasType];
    cublasHandleMap.erase(*(cublasHandle_t *)cublasType);
    break;
  }
  default:
    assert(false);
  }
}
std::map<cudaStream_t, cudaStream_t *> cudaStreamMap;
std::map<cudaEvent_t, cudaEvent_t *> cudaEventMap;
void *getCudaType(void *cudaType, int type)
{
  switch (type)
  {
  case CUDA_STREAM:
  {
    if (cudaStreamMap.size() == 0)
      return NULL;
    return (void *)cudaStreamMap[*(cudaStream_t *)cudaType];
  }
  case CUDA_EVENT:
  {
    if (cudaEventMap.size() == 0)
      return NULL;
    return (void *)cudaEventMap[*(cudaEvent_t *)cudaType];
  }
  default:
    assert(false);
  }
}

void freeCudaType(void *cudaType, int type)
{
  switch (type)
  {
  case CUDA_STREAM:
  {
    if (cudaStreamMap.size() == 0)
      return;
    cudaStreamMap.erase(*(cudaStream_t *)cudaType);
    break;
  }
  case CUDA_EVENT:
  {
    if (cudaEventMap.size() == 0)
      return;
    cudaEventMap.erase(*(cudaEvent_t *)cudaType);
    break;
  }
  default:
    assert(false);
  }
}

/************test func*******/
uint64_t hashUniqueId(ncclUniqueId const &id) {
  char const *bytes = (char const*)&id;
  uint64_t h = 0xdeadbeef;
  for (int i = 0; i < (int)sizeof(ncclUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}
/************test func*******/

std::map<ncclComm_t, ncclComm_t> ncclCommMap;
void *getNickleType(void *ncclType, int type)
{
  switch (type)
  {
  case NCCL_COMM:
  {
    if (ncclCommMap.size() == 0)
      return NULL;
    return (void *)ncclCommMap[*((ncclComm_t*)ncclType)];
    break;
  }
  default:
    assert(false);
  }
  return NULL;
}
void freeNickleType(void *ncclType, int type)
{
  switch (type)
  {
  case NCCL_COMM:
  {
    if (ncclCommMap.size() == 0)
      return;

    ncclCommMap.erase(*(ncclComm_t *)ncclType);
    break;
  }
  default:
    assert(false);
  }
}

void writeInfo(void *oldmod, void *newmod, int type)
{
  int fd;
  if (type == RCMODULE)
    fd = open("moduleinfo.dat", O_CREAT | O_WRONLY | O_APPEND, 0600);
  else if (type == RCEVENT)
    fd = open("eventinfo.dat", O_CREAT | O_WRONLY | O_APPEND, 0600);
  write(fd, &oldmod, sizeof(oldmod));
  write(fd, &newmod, sizeof(newmod));
  close(fd);
}

extern void cudaFuncExec(Cuda_Fncs_t cuda_fnc_op, ...);
void replayAPI(CudaCallLog_t *l)
{
  Cuda_Fncs_t op;
  pid_t tid = l->thread_id;
  // printf("In replayAPI.....1, fnargs:%p\n", l->fncargs);
  memcpy(&op, l->fncargs, sizeof op);
  size_t chars_read = sizeof op;

  printf("In replayAPI %s, tid:%i :start\n", cuda_Fnc_to_str[op], tid);
  fflush(stdout);
  switch (op)
  {
  case GENERATE_ENUM(cudaMalloc):
  {
    break;
  }
  case GENERATE_ENUM(cuMemAlloc_v2):
  {
    break;
  }
  case GENERATE_ENUM(cudaMallocManaged):
  {
    void *oldDevPtr;
    memcpy(&oldDevPtr, l->fncargs + chars_read, sizeof oldDevPtr);
    chars_read += sizeof oldDevPtr;

    size_t len;
    memcpy(&len, l->fncargs + chars_read, sizeof len);
    chars_read += sizeof len;

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof flags;

    void *newDevPtr = NULL;
    cudaError_t ret = cudaMallocManaged(&newDevPtr, len, flags);

    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cuMemAllocManaged):
  {
    CUdeviceptr oldDevPtr;
    memcpy(&oldDevPtr, l->fncargs + chars_read, sizeof oldDevPtr);
    chars_read += sizeof oldDevPtr;

    size_t len;
    memcpy(&len, l->fncargs + chars_read, sizeof len);
    chars_read += sizeof len;

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof flags;

    CUdeviceptr *newDevPtr = new CUdeviceptr();
    CUresult ret = cuMemAllocManaged(newDevPtr, len, flags);

    assert(ret == CUDA_SUCCESS);
    if (oldDevPtr != *newDevPtr)
    {
      deviceptrmap[oldDevPtr] = newDevPtr;
    }
    // printf("oldDev: %llu, newDev: %llu,\n", oldDevPtr, *newDevPtr);
    break;
  }
  case GENERATE_ENUM(cudaFree):
  {
    break;
  }
  case GENERATE_ENUM(cuMemFree_v2):
  {
    break;
  }
  case GENERATE_ENUM(__cudaInitModule):
  {
    void *fatCubinHandle;
    memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof(void *));
    chars_read += sizeof(void *);
    __cudaInitModule(new_fatCubinHandle);
    // __cudaInitModule(new_fatCubinHandle);
    break;
  }
  case GENERATE_ENUM(__cudaPopCallConfiguration):
  {
    dim3 *gridDim;
    dim3 *blockDim;
    size_t *sharedMem;
    void *stream;
    memcpy(&gridDim, l->fncargs + chars_read, sizeof(gridDim));
    chars_read += sizeof gridDim;
    memcpy(&blockDim, l->fncargs + chars_read, sizeof(blockDim));
    chars_read += sizeof blockDim;
    memcpy(&sharedMem, l->fncargs + chars_read, sizeof sharedMem);
    chars_read += sizeof sharedMem;
    memcpy(&stream, l->fncargs + chars_read, sizeof(void *));
    chars_read += sizeof(void *);

    // replay
    // cudaError_t ret =
    // __cudaPopCallConfiguration(gridDim, blockDim,
    //                     sharedMem, stream);
    // JASSERT(ret == cudaSuccess)
    //   .Text("__cudaPopCallConfiguration replay failed");

    // printf("ret %d", ret);
    fflush(stdout);
    break;
  }
  case GENERATE_ENUM(__cudaPushCallConfiguration):
  {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    void *stream;
    memcpy(&gridDim, l->fncargs + chars_read, sizeof gridDim);
    chars_read += sizeof gridDim;
    memcpy(&blockDim, l->fncargs + chars_read, sizeof blockDim);
    chars_read += sizeof blockDim;
    memcpy(&sharedMem, l->fncargs + chars_read, sizeof sharedMem);
    chars_read += sizeof sharedMem;
    memcpy(&stream, l->fncargs + chars_read, sizeof(void *));
    chars_read += sizeof(void *);
    // replay
    // typedef unsigned int (*pushFptr_t)(dim3 gridDim, dim3 blockDim, size_t sharedMem, void * stream);
    // Cuda_Fncs_t fnc = Cuda_Fnc___cudaPushCallConfiguration;
    // pushFptr_t func = (pushFptr_t)lhDlsym(fnc);
    // func(gridDim, blockDim,sharedMem, stream);
    break;
  }
  case GENERATE_ENUM(__cudaRegisterFatBinary):
  {
    void *fatCubin;
    memcpy(&fatCubin, l->fncargs + chars_read, sizeof(void *));
    chars_read += sizeof(void *);
    // replay

    void **newRes = __cudaRegisterFatBinary(fatCubin);
    // void** newRes = 0;
    // cudaFuncExec(GENERATE_ENUM(__cudaRegisterFatBinary), tid, &newRes, fatCubin);
    // printf("\n old fatcubinhandle = %p\n", oldRes);
    // printf("fatcubinhandle = %p\n", newRes);
    new_fatCubinHandle = newRes;
    // JASSERT(memcmp(&oldRes, *newRes, sizeof(*newRes))!= 0)
    //   .Text("old and new results are not same!");
    break;
  }
  case GENERATE_ENUM(__cudaRegisterFatBinaryEnd):
  {
    // replay
    // This call was introduced in CUDA 10.2
    // CUDA 10.2 will fail without this call
    __cudaRegisterFatBinaryEnd(new_fatCubinHandle);
    // cudaFuncExec(GENERATE_ENUM(__cudaRegisterFatBinaryEnd), tid, new_fatCubinHandle);
    break;
  }

  case GENERATE_ENUM(__cudaUnregisterFatBinary):
  {
    __cudaUnregisterFatBinary(new_fatCubinHandle);
    // cudaFuncExec(GENERATE_ENUM(__cudaUnregisterFatBinary), tid, new_fatCubinHandle);
    // JTRACE(" __cudaUnregisterFatBinary replayed");
    break;
  }
  case GENERATE_ENUM(__cudaRegisterFunction):
  {
    void **fatCubinHandle;
    int thread_limit;
    uint3 *tid_1;
    uint3 *bid;
    dim3 *bDim;
    dim3 *gDim;
    int *wSize;

    memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof(void **));
    chars_read += sizeof(void **);

    char *hostFun;
    memcpy(&hostFun, l->fncargs + chars_read, sizeof(char *));
    chars_read += sizeof(char *);

    char *deviceFun;
    memcpy(&deviceFun, l->fncargs + chars_read, sizeof(char *));
    chars_read += sizeof(char *);

    char *deviceName;
    memcpy(&deviceName, l->fncargs + chars_read, sizeof(char *));
    chars_read += sizeof(char *);

    memcpy(&thread_limit, l->fncargs + chars_read, sizeof(thread_limit));
    chars_read += sizeof(thread_limit);

    memcpy(&tid_1, l->fncargs + chars_read, sizeof tid_1);
    chars_read += sizeof(tid_1);

    memcpy(&bid, l->fncargs + chars_read, sizeof bid);
    chars_read += sizeof(bid);

    memcpy(&bDim, l->fncargs + chars_read, sizeof bDim);
    chars_read += sizeof(bDim);

    memcpy(&gDim, l->fncargs + chars_read, sizeof gDim);
    chars_read += sizeof(gDim);

    memcpy(&wSize, l->fncargs + chars_read, sizeof wSize);
    chars_read += sizeof(wSize);

    // replay
    // __cudaRegisterFunction(&fatCubinHandle, hostFun, deviceFun,
    //   deviceName, thread_limit, &tid, &bid, &bDim, &gDim, &wSize);
    __cudaRegisterFunction(new_fatCubinHandle, hostFun, deviceFun,
                           deviceName, thread_limit, tid_1, bid, bDim, gDim, wSize);
    // cudaFuncExec(GENERATE_ENUM(__cudaRegisterFunction), tid, new_fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid_1, bid, bDim, gDim, wSize);
    break;
  }
  case GENERATE_ENUM(__cudaRegisterVar):
  {
    void **fatCubinHandle;
    char *hostVar;
    int ext;
    size_t size;
    int constant;
    int global;

    memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof(void **));
    chars_read += sizeof(void **);

    memcpy(&hostVar, l->fncargs + chars_read, sizeof(char *));
    chars_read += sizeof(char *);

    char *deviceAddress;
    memcpy(&deviceAddress, l->fncargs + chars_read, sizeof(char *));
    chars_read += sizeof(char *);

    char *deviceName;
    memcpy(&deviceName, l->fncargs + chars_read, sizeof(char *));
    chars_read += sizeof(char *);

    memcpy(&ext, l->fncargs + chars_read, sizeof ext);
    chars_read += sizeof(ext);

    memcpy(&size, l->fncargs + chars_read, sizeof size);
    chars_read += sizeof(size);

    memcpy(&constant, l->fncargs + chars_read, sizeof constant);
    chars_read += sizeof(constant);

    memcpy(&global, l->fncargs + chars_read, sizeof global);
    chars_read += sizeof(global);

    // replay
    __cudaRegisterVar(new_fatCubinHandle, hostVar,
                      deviceAddress, deviceName,
                      ext, size, constant, global);

    // cudaFuncExec(GENERATE_ENUM(__cudaRegisterVar), tid, new_fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
    // JTRACE("__cudaRegisterVar replayed");
    break;
  }
  case GENERATE_ENUM(__cudaRegisterManagedVar):
  {
    void **fatCubinHandle;
    void **hostVarPtrAddress;
    int ext;
    size_t size;
    int constant;
    int global;

    memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof(void **));
    chars_read += sizeof(void **);

    memcpy(&hostVarPtrAddress, l->fncargs + chars_read, sizeof(void **));
    chars_read += sizeof(void **);

    char *deviceAddress;
    memcpy(&deviceAddress, l->fncargs + chars_read, sizeof(char *));
    chars_read += sizeof(char *);

    char *deviceName;
    memcpy(&deviceName, l->fncargs + chars_read, sizeof(char *));
    chars_read += sizeof(char *);

    memcpy(&ext, l->fncargs + chars_read, sizeof ext);
    chars_read += sizeof(ext);

    memcpy(&size, l->fncargs + chars_read, sizeof size);
    chars_read += sizeof(size);

    memcpy(&constant, l->fncargs + chars_read, sizeof constant);
    chars_read += sizeof(constant);

    memcpy(&global, l->fncargs + chars_read, sizeof global);
    chars_read += sizeof(global);

    // replay
    __cudaRegisterManagedVar(new_fatCubinHandle, hostVarPtrAddress,
                             deviceAddress, deviceName,
                             ext, size, constant, global);
    // cudaFuncExec(GENERATE_ENUM(__cudaRegisterManagedVar), tid, new_fatCubinHandle, hostVarPtrAddress, deviceAddress, deviceName, ext, size, constant, global);
    // JTRACE("__cudaRegisterVar replayed");
    break;
  }
  case GENERATE_ENUM(__cudaRegisterTexture):
  {
    void **fatCubinHandle;
    struct textureReference *hostVar;
    const void **deviceAddress;
    int dim;
    int norm;
    int ext;
    memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof(void **));
    chars_read += sizeof(void **);

    memcpy(&hostVar, l->fncargs + chars_read, sizeof(hostVar));
    chars_read += sizeof(hostVar);

    memcpy(&deviceAddress, l->fncargs + chars_read, sizeof(void *));
    chars_read += sizeof(void *);

    char *deviceName;
    memcpy(&deviceName, l->fncargs + chars_read, sizeof(deviceName));
    chars_read += sizeof(deviceName);

    memcpy(&dim, l->fncargs + chars_read, sizeof dim);
    chars_read += sizeof(dim);

    memcpy(&norm, l->fncargs + chars_read, sizeof norm);
    chars_read += sizeof(norm);

    memcpy(&ext, l->fncargs + chars_read, sizeof ext);
    chars_read += sizeof(ext);

    // replay

    __cudaRegisterTexture(new_fatCubinHandle, hostVar, deviceAddress,
                          deviceName, dim, norm, ext);
    // cudaFuncExec(GENERATE_ENUM(__cudaRegisterTexture), tid, new_fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext);
    // JTRACE("__cudaRegisterTexture replayed");
    break;
  }
  case GENERATE_ENUM(__cudaRegisterSurface):
  {
    void **fatCubinHandle;
    struct surfaceReference *hostVar;
    const void **deviceAddress;
    int dim;
    int ext;
    memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof(void **));
    chars_read += sizeof(void **);

    memcpy(&hostVar, l->fncargs + chars_read, sizeof(hostVar));
    chars_read += sizeof(hostVar);

    memcpy(&deviceAddress, l->fncargs + chars_read, sizeof(void **));
    chars_read += sizeof(void **);

    char *deviceName;
    memcpy(&deviceName, l->fncargs + chars_read, sizeof(deviceName));
    chars_read += sizeof(deviceName);

    memcpy(&dim, l->fncargs + chars_read, sizeof dim);
    chars_read += sizeof(dim);

    memcpy(&ext, l->fncargs + chars_read, sizeof ext);
    chars_read += sizeof(ext);

    // replay
    __cudaRegisterSurface(new_fatCubinHandle, hostVar, deviceAddress,
                          deviceName, dim, ext);
    // cudaFuncExec(GENERATE_ENUM(__cudaRegisterSurface), tid, new_fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext);
    // JTRACE("__cudaRegisterSurface replayed");
    break;
  }
  case GENERATE_ENUM(cudaCreateTextureObject):
  {
    // args
    cudaTextureObject_t *pTexObject;
    memcpy(&pTexObject, l->fncargs + chars_read, sizeof(pTexObject));
    chars_read += sizeof(pTexObject);

    struct cudaResourceDesc *pResDesc;
    memcpy(&pResDesc, l->fncargs + chars_read, sizeof(pResDesc));
    chars_read += sizeof(pResDesc);

    struct cudaTextureDesc *pTexDesc;
    memcpy(&pTexDesc, l->fncargs + chars_read, sizeof(pTexDesc));
    chars_read += sizeof(pTexDesc);

    struct cudaResourceViewDesc *pResViewDesc;
    memcpy(&pResViewDesc, l->fncargs + chars_read, sizeof(pResViewDesc));
    chars_read += sizeof(pResViewDesc);

    cudaError_t ret = cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaDestroyTextureObject):
  {
    // args
    cudaTextureObject_t texObject;
    memcpy(&texObject, l->fncargs + chars_read, sizeof(texObject));
    chars_read += sizeof(texObject);

    cudaError_t ret = cudaDestroyTextureObject(texObject);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaBindTextureToArray):
  {
    textureReference *texref;
    memcpy(&texref, l->fncargs + chars_read, sizeof(texref));
    chars_read += sizeof(texref);

    cudaArray_const_t array;
    memcpy(&array, l->fncargs + chars_read, sizeof(array));
    chars_read += sizeof(array);

    cudaChannelFormatDesc *desc;
    memcpy(&desc, l->fncargs + chars_read, sizeof(desc));
    chars_read += sizeof(desc);
    cudaError_t ret = cudaBindTextureToArray(texref, array, desc);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaUnbindTexture):
  {
    struct textureReference *texref;
    memcpy(&texref, l->fncargs + chars_read, sizeof(texref));
    chars_read += sizeof(texref);
    cudaError_t ret = cudaUnbindTexture(texref);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaCreateChannelDesc):
  {
    int x, y, z, w;
    memcpy(&x, l->fncargs + chars_read, sizeof x);
    chars_read += sizeof x;
    memcpy(&y, l->fncargs + chars_read, sizeof y);
    chars_read += sizeof y;
    memcpy(&z, l->fncargs + chars_read, sizeof z);
    chars_read += sizeof z;
    memcpy(&w, l->fncargs + chars_read, sizeof w);
    chars_read += sizeof w;
    cudaChannelFormatKind f;
    memcpy(&f, l->fncargs + chars_read, sizeof f);
    chars_read += sizeof f;

    cudaCreateChannelDesc(x, y, z, w, f);
    // JASSERT(memcmp(&oldDesc, &newDesc, sizeof(oldDesc))!= 0)
    //   .Text("old and new desc are not same!");
    break;
  }
  case GENERATE_ENUM(cudaMallocArray):
  {
    // TODO : check the cudaMallocArray log and replay
    // args
    cudaArray_t *array;
    struct cudaChannelFormatDesc *desc;
    size_t width;
    size_t height;
    unsigned int flags;

    memcpy(&array, l->fncargs + chars_read, sizeof array);
    chars_read += sizeof(array);

    memcpy(&desc, l->fncargs + chars_read, sizeof desc);
    chars_read += sizeof(desc);

    memcpy(&width, l->fncargs + chars_read, sizeof width);
    chars_read += sizeof(width);

    memcpy(&height, l->fncargs + chars_read, sizeof height);
    chars_read += sizeof(height);

    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof(flags);

    cudaError_t ret = cudaMallocArray(array, desc, width, height, flags);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaFreeArray):
  {
    // args
    cudaArray_t *array;
    memcpy(&array, l->fncargs + chars_read, sizeof(array));
    chars_read += sizeof(array);

    cudaError_t ret = cudaFreeArray(*array);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaMallocHost):
  {
    break;
  }
  case GENERATE_ENUM(cuMemAllocHost_v2):
  {
    break;
  }
  case GENERATE_ENUM(cudaFreeHost):
  {
    break;
  }
  case GENERATE_ENUM(cuMemFreeHost):
  {
    break;
  }
  case GENERATE_ENUM(cudaHostAlloc):
  {
    break;
  }
  case GENERATE_ENUM(cuMemHostAlloc):
  {
    break;
  }
  case GENERATE_ENUM(cudaDeviceReset):
  {
    // no arguments to read from buffer
    cudaError_t ret = cudaDeviceReset();
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaDeviceSetLimit):
  {
    int limit;
    memcpy(&limit, l->fncargs + chars_read, sizeof limit);
    chars_read += sizeof(limit);

    size_t value;
    memcpy(&value, l->fncargs + chars_read, sizeof value);
    chars_read += sizeof(value);

    cudaError_t ret = cudaDeviceSetLimit((cudaLimit)limit, value);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaDeviceSetCacheConfig):
  {
    int cacheConfig;
    memcpy(&cacheConfig, l->fncargs + chars_read, sizeof cacheConfig);
    chars_read += sizeof(cacheConfig);

    cudaError_t ret = cudaDeviceSetCacheConfig((cudaFuncCache)cacheConfig);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaMallocPitch):
  {
    void **devPtr;
    memcpy(&devPtr, l->fncargs + chars_read, sizeof(devPtr));
    chars_read += sizeof(devPtr);

    size_t *pitch;
    memcpy(&pitch, l->fncargs + chars_read, sizeof pitch);
    chars_read += sizeof(pitch);

    size_t width;
    memcpy(&width, l->fncargs + chars_read, sizeof(width));
    chars_read += sizeof(width);

    size_t height;
    memcpy(&height, l->fncargs + chars_read, sizeof(height));
    chars_read += sizeof(height);
    cudaError_t ret = cudaMallocPitch(devPtr, pitch, width, height);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cuMemAllocPitch_v2):
  {
    CUdeviceptr *devPtr;
    memcpy(&devPtr, l->fncargs + chars_read, sizeof(devPtr));
    chars_read += sizeof(devPtr);

    size_t *pitch;
    memcpy(&pitch, l->fncargs + chars_read, sizeof pitch);
    chars_read += sizeof(pitch);

    size_t width;
    memcpy(&width, l->fncargs + chars_read, sizeof(width));
    chars_read += sizeof(width);

    size_t height;
    memcpy(&height, l->fncargs + chars_read, sizeof(height));
    chars_read += sizeof(height);

    unsigned int ElementSizeBytes;
    memcpy(&ElementSizeBytes, l->fncargs + chars_read, sizeof(ElementSizeBytes));
    chars_read += sizeof(ElementSizeBytes);
    CUresult ret = cuMemAllocPitch_v2(devPtr, pitch, width, height, ElementSizeBytes);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  /********************cuda ipc replay by tian01.liu--begin**************************/
  case GENERATE_ENUM(cudaIpcGetMemHandle):
  {
    cudaIpcMemHandle_t handle;
    memcpy(&handle, l->fncargs + chars_read, sizeof(cudaIpcMemHandle_t));
    chars_read += sizeof(cudaIpcMemHandle_t);

    void* devPtr = nullptr;
    memcpy(&devPtr, l->fncargs + chars_read, sizeof(void*));

    cudaError_t ret = cudaSuccess;
    cudaFuncExec(GENERATE_ENUM(cudaIpcGetMemHandle), tid, &ret, &handle, devPtr);

    break;
  }
  case GENERATE_ENUM(cudaIpcOpenMemHandle):
  {
    void* devPtr;
    memcpy(&devPtr, l->fncargs + chars_read, sizeof(void*));
    chars_read += sizeof(void*);

    cudaIpcMemHandle_t handle;
    memcpy(&handle, l->fncargs + chars_read, sizeof(cudaIpcMemHandle_t));
    chars_read += sizeof(cudaIpcMemHandle_t);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof(unsigned int));
    chars_read += sizeof(unsigned int);

    cudaError_t ret = cudaSuccess;
    cudaFuncExec(GENERATE_ENUM(cudaIpcOpenMemHandle), tid, &ret, &devPtr, handle, flags);
    break;
  }
  /********************cuda ipc replay by tian01.liu--end  **************************/
  case GENERATE_ENUM(cudaDeviceSynchronize):
  {
    // no arguments to read from buffer
    cudaError_t ret = cudaDeviceSynchronize();
    assert(ret == cudaSuccess);
    break;
  }
  // sth about cudaStreamCreate
  // if it is used in a program it will affect
  // the deterministism of cudaMalloc
  // mrCUDA
  case GENERATE_ENUM(cudaStreamCreate):
  {
    cudaStream_t stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof(stream));
    chars_read += sizeof(stream);

    cudaStream_t *pStream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
    cudaError_t ret = cudaStreamCreate(pStream);

    assert(ret == cudaSuccess);

    if (stream != NULL && (stream != *pStream))
    {
      cudaStreamMap[stream] = pStream;
    }
    break;
  }
  case GENERATE_ENUM(cuStreamCreate):
  {
    // CUstream *pStream;
    // memcpy(&pStream, l->fncargs + chars_read, sizeof pStream);
    // chars_read += sizeof(pStream);

    CUstream stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof stream);
    chars_read += sizeof(stream);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof(flags);

    CUstream *pStream = new CUstream();
    CUresult ret = cuStreamCreate(pStream, flags);
    assert(ret == CUDA_SUCCESS);
    if (stream != NULL && (stream != *pStream))
    {
      streammap[stream] = pStream;
    }
    // printf("[CREATE]oldStream: %p, newStream: %p. \n", stream, *pStream);
    break;
  }
  case GENERATE_ENUM(cudaStreamCreateWithFlags):
  {
    cudaStream_t stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof stream);
    chars_read += sizeof(stream);
    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof(flags);

    cudaStream_t *pStream = (cudaStream_t *)malloc(sizeof(cudaStream_t));

    cudaError_t ret = cudaStreamCreateWithFlags(pStream, flags);

    assert(ret == cudaSuccess);

    if (stream != NULL && (stream != *pStream))
    {
      cudaStreamMap[stream] = pStream;
    }
    break;
  }
  case GENERATE_ENUM(cudaStreamCreateWithPriority):
  {
    cudaStream_t stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof(cudaStream_t));
    chars_read += sizeof(cudaStream_t);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof(flags);

    int priority;
    memcpy(&priority, l->fncargs + chars_read, sizeof priority);
    chars_read += sizeof(priority);
    cudaStream_t *pStream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
    cudaError_t ret = cudaSuccess;

    ret = cudaStreamCreateWithPriority(pStream, flags, priority);
    assert(ret == cudaSuccess);

    // printf("cudaStreamCreateWithPriority, newStream:%p\n", *pStream);
    if (stream != NULL && (stream != *pStream))
    {
      cudaStreamMap[stream] = pStream;
    }
    break;
  }
  case GENERATE_ENUM(cuStreamCreateWithPriority):
  {
    CUstream stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof stream);
    chars_read += sizeof(stream);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof(flags);

    int priority;
    memcpy(&priority, l->fncargs + chars_read, sizeof priority);
    chars_read += sizeof(priority);

    CUstream *pStream = new CUstream();
    CUresult ret = cuStreamCreateWithPriority(pStream, flags, priority);
    assert(ret == CUDA_SUCCESS);
    // printf("oldstream = %p, newstream = %p\n", stream, *pStream);
    if (CUDA_SUCCESS == ret && (stream != NULL && stream != *pStream))
    {
      streammap[stream] = pStream;
    }
    break;
  }
  case GENERATE_ENUM(cuStreamQuery):
  {
    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    CUstream *pStream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    chars_read += sizeof(hStream);

    CUresult ret = CUDA_SUCCESS;
    if (pStream != NULL)
    {
        ret = cuStreamQuery(*pStream);
    }
    else
    {
        ret = cuStreamQuery(hStream);
    }
    assert(ret == CUDA_SUCCESS);
    // printf("[FREE]oldStream: %p, newStream: %p. \n", &hStream, pStream);
    break;
  }
  case GENERATE_ENUM(cuCtxGetSharedMemConfig):
  {
    CUsharedconfig config;
    memcpy(&config, l->fncargs + chars_read, sizeof config);
    chars_read += sizeof(config);

    CUresult ret = cuCtxGetSharedMemConfig(&config);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuCtxGetStreamPriorityRange):
  {
    int leastPriority;
    memcpy(&leastPriority, l->fncargs + chars_read, sizeof leastPriority);
    chars_read += sizeof(leastPriority);

    int greatestPriority;
    memcpy(&greatestPriority, l->fncargs + chars_read, sizeof greatestPriority);
    chars_read += sizeof(greatestPriority);

    CUresult ret = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuStreamGetCtx):
  {
    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    CUstream *pStream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    chars_read += sizeof(hStream);

    CUcontext pctx;
    memcpy(&pctx, l->fncargs + chars_read, sizeof pctx);
    chars_read += sizeof(pctx);

    CUresult ret = CUDA_SUCCESS;
    if (pStream != NULL)
    {
        ret = cuStreamGetCtx(*pStream, &pctx);
    }
    else
    {
        ret = cuStreamGetCtx(hStream, &pctx);
    }
    assert(ret == CUDA_SUCCESS);
    // printf("[FREE]oldStream: %p, newStream: %p. \n", &hStream, pStream);
    break;
  }
  case GENERATE_ENUM(cuStreamGetFlags):
  {
    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    CUstream *pStream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    chars_read += sizeof(hStream);

    unsigned int flags;
    CUresult ret = CUDA_SUCCESS;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof(flags);

    if (pStream != NULL)
    {
        ret = cuStreamGetFlags(*pStream, &flags);
    }
    else
    {
        ret = cuStreamGetFlags(*pStream, &flags);
    }
    assert(ret == CUDA_SUCCESS);
    //printf("[FREE]oldStream: %p, newStream: %p. \n", &hStream, pStream);
    break;
  }
  case GENERATE_ENUM(cuStreamGetPriority):
  {
    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    CUstream *pStream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    chars_read += sizeof(hStream);

    int priority;
    CUresult ret = CUDA_SUCCESS;
    memcpy(&priority, l->fncargs + chars_read, sizeof priority);
    chars_read += sizeof(priority);

    if (pStream != NULL)
    {
        ret = cuStreamGetPriority(*pStream, &priority);
    }
    else
    {
        ret = cuStreamGetPriority(hStream, &priority);
    }
    assert(ret == CUDA_SUCCESS);
    // printf("[FREE]oldStream: %p, newStream: %p. \n", hStream, *pStream);
    break;
  }
  case GENERATE_ENUM(cuStreamIsCapturing):
  {
    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    CUstream *pStream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    chars_read += sizeof(hStream);

    CUstreamCaptureStatus captureStatus;
    CUresult ret = CUDA_SUCCESS;
    memcpy(&captureStatus, l->fncargs + chars_read, sizeof captureStatus);
    chars_read += sizeof(captureStatus);

    if (pStream != NULL)
    {
        ret = cuStreamIsCapturing(*pStream, &captureStatus);
    }
    else
    {
        ret = cuStreamIsCapturing(hStream, &captureStatus);
    }
    assert(ret == CUDA_SUCCESS);
    // printf("[FREE]oldStream: %p, newStream: %p. \n", hStream, *pStream);
    break;
  }
  case GENERATE_ENUM(cudaStreamDestroy):
  {
    cudaError_t ret = cudaSuccess;
    cudaStream_t stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof(stream));
    chars_read += sizeof(stream);

    cudaStream_t *pStream = (cudaStream_t *)getCudaType((void *)&stream, CUDA_STREAM);
    if (NULL == pStream)
      ret = cudaStreamDestroy(stream);
    else
    {
        ret = cudaStreamDestroy(*pStream);

      if (cudaSuccess == ret)
      {
        cudaStreamMap.erase(stream);
        free(pStream);
      }
    }

    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cuStreamDestroy_v2):
  {
    CUstream hStream;
    CUresult ret = CUDA_SUCCESS;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    chars_read += sizeof(hStream);

    CUstream *pStream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    if (pStream != NULL)
    {
        ret = cuStreamDestroy_v2(*pStream);

      if (CUDA_SUCCESS == ret)
      {
        delete pStream;
        streammap.erase(hStream);
      }
    }
    else
    {
        ret = cuStreamDestroy_v2(hStream);
    }
    assert(ret == CUDA_SUCCESS);
    // printf("[FREE]oldStream: %p, newStream: %p. \n", hStream, *pStream);
    break;
  }
  case GENERATE_ENUM(cudaEventCreate):
  {
    cudaEvent_t event;
    memcpy(&event, l->fncargs + chars_read, sizeof(cudaEvent_t));
    chars_read += sizeof(event);

    cudaEvent_t *pEvent = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));

    cudaError_t ret = cudaEventCreate(pEvent);
    assert(ret == cudaSuccess);
    if (event !=NULL && (event != *pEvent))
      cudaEventMap[event] = pEvent;
    break;
  }
  case GENERATE_ENUM(cudaEventRecord):
  {
    cudaEvent_t event;
    memcpy(&event, l->fncargs + chars_read, sizeof(cudaEvent_t));
    chars_read += sizeof(event);
    cudaStream_t stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof(cudaStream_t));
    chars_read += sizeof(stream);

    cudaEvent_t *pEvent = (cudaEvent_t *)getCudaType((void *)&event, CUDA_EVENT);
    // cudaStream_t *pStream = (cudaStream_t *)getCudaType((void *)&stream, CUDA_STREAM);

    cudaEvent_t newEvent;
    // cudaStream_t newStream;
    if (pEvent != NULL)
    {
      newEvent = *pEvent;
    }
    else
      newEvent = event;

    // if(pStream != NULL)
    //   newStream = *pStream;
    // else
    //   newStream = stream;

    // printf("cudaEventRecord, Event:%p, stream:%p, tid:%i\n", newEvent, newStream, tid);
    // fflush(stdout);
    cudaError_t ret = cudaSuccess;
    ret = cudaEventRecord(newEvent, NULL);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cuEventCreate):
  {
    CUevent event;
    memcpy(&event, l->fncargs + chars_read, sizeof(event));
    chars_read += sizeof(event);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof(flags);

    CUevent *newEvent = new CUevent();
    CUresult ret = CUDA_SUCCESS;

      ret = cuEventCreate(newEvent, flags);

    // printf("cuEventCreate ret:%i\n", ret);
    assert(ret == CUDA_SUCCESS);
    eventmap[event] = newEvent;
    // writeInfo(event, newevent, RCEVENT);
    break;
  }
  case GENERATE_ENUM(cudaEventDestroy):
  {
    cudaEvent_t event;
    memcpy(&event, l->fncargs + chars_read, sizeof(event));
    chars_read += sizeof(event);
    cudaError_t ret = cudaSuccess;
    cudaEvent_t *pEvent = (cudaEvent_t *)getCudaType((void *)&event, CUDA_EVENT);
    // if(pEvent != NULL)
    //   printf("cudaEventDestroy, oldEvent:%p, newEvent:%p, tid:%i\n", event, *pEvent, tid);
    // else
    //   printf("cudaEventDestroy, oldEvent:%p, newEvent:nil, tid:%i\n", event, tid);
    fflush(stdout);
    if (pEvent != NULL)
    {

        ret = cudaEventDestroy(*pEvent);
      if (cudaSuccess == ret)
      {
        delete pEvent;
        cudaEventMap.erase(event);
      }
    }
    // else
    // {
    //     ret = cudaEventDestroy(event);
    //     //printf("cudaEventDestroy in replay, ret:%i\n", ret);
    // }
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cuEventDestroy_v2):
  {
    CUevent hEvent;
    CUresult ret = CUDA_SUCCESS;
    memcpy(&hEvent, l->fncargs + chars_read, sizeof(hEvent));
    chars_read += sizeof(hEvent);

    /*bool inmap = false;
    std::map<CUevent, CUevent*>::iterator iter = eventmap.begin();
    while(iter != eventmap.end())
    {
      CUevent oldevent = iter->first;
      printf("event %p vs %p\n", oldevent, event);
      if(oldevent == event)
      {
          CUevent* newevent = iter->second;
          CUresult res = cuEventDestroy_v2(*newevent);
          printf("fengtao.xie replay cuEventDestroy_v2 %d\n", res);
          eventmap.erase(iter);
          inmap = true;
          break;
      }
      iter++;
    }
    if(!inmap)
    {
      //printf("fengtao.xie event address in cuEventDestroy_v2 %p\n", event);
      CUresult res = cuEventDestroy_v2(event);
      printf("fengtao.xie replay cuEventDestroy_v2 org %p, %d\n", event, res);
      //printf("fengtao.xie cuEventDestroy_v2 ok\n");
    }*/
    CUevent *pEvent = (CUevent *)getCUtype((void *)&hEvent, CU_EVENT);
    if (NULL != pEvent)
    {
        ret = cuEventDestroy_v2(*pEvent);
      if (CUDA_SUCCESS == ret)
      {
        delete pEvent;
        eventmap.erase(hEvent);
      }
    }
    else
        ret = cuEventDestroy_v2(hEvent);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cudaEventSynchronize):
  {
    cudaEvent_t event;
    memcpy(&event, l->fncargs + chars_read, sizeof(event));
    chars_read += sizeof(event);
    cudaEvent_t* pCudaEvent = (cudaEvent_t*)getCudaType((void*)&event, CUDA_EVENT);
    cudaError_t ret = cudaEventSynchronize(pCudaEvent != NULL ? *pCudaEvent : event);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaEventCreateWithFlags):
  {
    cudaEvent_t event;
    memcpy(&event, l->fncargs + chars_read, sizeof(event));
    chars_read += sizeof(event);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof(flags);
 
    cudaEvent_t *pEvent = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
    cudaError_t ret = cudaSuccess;

      ret = cudaEventCreateWithFlags(pEvent, flags);

    assert(ret == cudaSuccess);
    if (event !=NULL && (event != *pEvent))
      cudaEventMap[event] = pEvent;
    // printf("cudaEventCreateWithFlags in replay, oldEvent:%p, new Event:%p, tid:%i\n", event, *pEvent, tid);
    fflush(stdout);
    break;
  }
  case GENERATE_ENUM(cuDestroyExternalMemory):
  {
    CUexternalMemory *extMem;
    memcpy(&extMem, l->fncargs + chars_read, sizeof(extMem));
    chars_read += sizeof(extMem);
    CUresult ret = cuDestroyExternalMemory(*extMem);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuDestroyExternalSemaphore):
  {
    CUexternalSemaphore *extSem;
    memcpy(&extSem, l->fncargs + chars_read, sizeof(extSem));
    chars_read += sizeof(extSem);
    CUresult ret = cuDestroyExternalSemaphore(*extSem);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuGraphCreate):
  {
    CUgraph *phGraph;
    memcpy(&phGraph, l->fncargs + chars_read, sizeof(phGraph));
    chars_read += sizeof(phGraph);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof flags);
    chars_read += sizeof(flags);
    CUresult ret = cuGraphCreate(phGraph, flags);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuGraphDestroy):
  {
    CUgraph *hGraph;
    memcpy(&hGraph, l->fncargs + chars_read, sizeof(hGraph));
    chars_read += sizeof(hGraph);
    CUresult ret = cuGraphDestroy(*hGraph);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuGraphDestroyNode):
  {
    CUgraphNode *hNode;
    memcpy(&hNode, l->fncargs + chars_read, sizeof(hNode));
    chars_read += sizeof(hNode);
    CUresult ret = cuGraphDestroyNode(*hNode);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuGraphExecDestroy):
  {
    CUgraphExec *hGraphExec;
    memcpy(&hGraphExec, l->fncargs + chars_read, sizeof(hGraphExec));
    chars_read += sizeof(hGraphExec);
    CUresult ret = cuGraphExecDestroy(*hGraphExec);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuTexRefCreate):
  {
    CUtexref *pTexRef;
    memcpy(&pTexRef, l->fncargs + chars_read, sizeof(pTexRef));
    chars_read += sizeof(pTexRef);
    CUresult ret = cuTexRefCreate(pTexRef);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuTexRefDestroy):
  {
    CUtexref *hTexRef;
    memcpy(&hTexRef, l->fncargs + chars_read, sizeof(hTexRef));
    chars_read += sizeof(hTexRef);
    CUresult ret = cuTexRefDestroy(*hTexRef);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuTexObjectCreate):
  {
    CUtexObject *pTexObject;
    memcpy(&pTexObject, l->fncargs + chars_read, sizeof(pTexObject));
    chars_read += sizeof(pTexObject);

    CUDA_RESOURCE_DESC *pResDesc;
    memcpy(&pResDesc, l->fncargs + chars_read, sizeof(pResDesc));
    chars_read += sizeof(pResDesc);

    CUDA_TEXTURE_DESC *pTexDesc;
    memcpy(&pTexDesc, l->fncargs + chars_read, sizeof(pTexDesc));
    chars_read += sizeof(pTexDesc);

    CUDA_RESOURCE_VIEW_DESC *pResViewDesc;
    memcpy(&pResViewDesc, l->fncargs + chars_read, sizeof(pResViewDesc));
    chars_read += sizeof(pResViewDesc);
    CUresult ret = cuTexObjectCreate(pTexObject, pResDesc, (const CUDA_TEXTURE_DESC *)pTexDesc, (const CUDA_RESOURCE_VIEW_DESC *)pResViewDesc);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuTexObjectDestroy):
  {
    CUtexObject *pTexObject;
    memcpy(&pTexObject, l->fncargs + chars_read, sizeof(pTexObject));
    chars_read += sizeof(pTexObject);
    CUresult ret = cuTexObjectDestroy(*pTexObject);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuSurfObjectCreate):
  {
    CUsurfObject *pSurfObject;
    memcpy(&pSurfObject, l->fncargs + chars_read, sizeof(pSurfObject));
    chars_read += sizeof(pSurfObject);

    CUDA_RESOURCE_DESC *pResDesc;
    memcpy(&pResDesc, l->fncargs + chars_read, sizeof(pResDesc));
    chars_read += sizeof(pResDesc);
    CUresult ret = cuSurfObjectCreate(pSurfObject, pResDesc);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuSurfObjectDestroy):
  {
    CUsurfObject *pSurfObject;
    memcpy(&pSurfObject, l->fncargs + chars_read, sizeof(pSurfObject));
    chars_read += sizeof(pSurfObject);
    CUresult ret = cuSurfObjectDestroy(*pSurfObject);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cublasCreate_v2):
  {
    cublasHandle_t handle;
    cublasStatus_t stat;
    memcpy(&handle, l->fncargs + chars_read, sizeof(cublasHandle_t));
    chars_read += sizeof(cublasHandle_t);
    cublasHandle_t *pCublasHandle = (cublasHandle_t *)malloc(sizeof(cublasHandle_t));

      stat = cublasCreate_v2(pCublasHandle);

    assert(stat == CUBLAS_STATUS_SUCCESS);
    if (handle != NULL && (handle != *pCublasHandle))
    {
      cublasHandleMap[handle] = pCublasHandle;
    }

    break;
  }
  case GENERATE_ENUM(cublasSetStream_v2):
  {
    cublasHandle_t handle;
    cublasStatus_t stat;
    memcpy(&handle, l->fncargs + chars_read, sizeof(cublasHandle_t));
    chars_read += sizeof(cublasHandle_t);
    cudaStream_t stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof(cudaStream_t));
    chars_read += sizeof(cudaStream_t);
    // printf("cublasSetStream_v2 replay, stream:%p\n", stream);
    if (stream == NULL)
      return;

    cublasHandle_t *pCublasHandle = (cublasHandle_t *)getCublasType((void *)&handle, CUBLAS_HANDLE);
    cudaStream_t *pStream = (cudaStream_t *)getCudaType((void*)&stream, CUDA_STREAM);
    cublasHandle_t newHandle;
    cudaStream_t newStream;
    if (pCublasHandle == NULL)
      newHandle = handle;
    else
      newHandle = *pCublasHandle;
    if (pStream == NULL)
      newStream = stream;
    else
      newStream = *pStream;
    stat = cublasSetStream_v2(newHandle, newStream);
    assert(stat == CUBLAS_STATUS_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cublasDestroy_v2):
  {
    cublasStatus_t stat;
    cublasHandle_t handle;
    memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
    chars_read += sizeof(handle);

    cublasHandle_t *pCublasHandle = (cublasHandle_t *)getCublasType((void *)&handle, CUBLAS_HANDLE);

    if (NULL == pCublasHandle)
    {
        stat = cublasDestroy_v2(handle);
    }
    else
    {
      stat = cublasDestroy_v2(*pCublasHandle);
      if (stat == CUBLAS_STATUS_SUCCESS)
      {
        cublasHandleMap.erase(handle);
        free(pCublasHandle);
      }
    }
    assert(stat == CUBLAS_STATUS_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cublasAlloc):
  {
    int n;
    memcpy(&n, l->fncargs + chars_read, sizeof n);
    chars_read += sizeof n;

    int elemSize;
    memcpy(&elemSize, l->fncargs + chars_read, sizeof elemSize);
    chars_read += sizeof elemSize;

    void *devicePtr;
    memcpy(&devicePtr, l->fncargs + chars_read, sizeof devicePtr);
    chars_read += sizeof devicePtr;
    // printf("n = %d, elemSize = %d, old_devicePtr.......: %p\n", n, elemSize, devicePtr);

    void *newDevicePtr = NULL;
    cublasStatus ret = cublasAlloc(n, elemSize, &newDevicePtr);

    assert(ret == CUBLAS_STATUS_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cufftPlan3d):
  {
    cufftHandle *plan;
    memcpy(&plan, l->fncargs + chars_read, sizeof(plan));
    chars_read += sizeof(plan);

    int nx;
    memcpy(&nx, l->fncargs + chars_read, sizeof(nx));
    chars_read += sizeof(nx);

    int ny;
    memcpy(&ny, l->fncargs + chars_read, sizeof(ny));
    chars_read += sizeof(ny);

    int nz;
    memcpy(&nz, l->fncargs + chars_read, sizeof(nz));
    chars_read += sizeof(nz);

    int type;
    memcpy(&type, l->fncargs + chars_read, sizeof(type));
    chars_read += sizeof(type);
    cufftResult ret = cufftPlan3d(plan, nx, ny, nz, (cufftType)type);
    assert(ret == CUFFT_SUCCESS);
    break;
  }
    case GENERATE_ENUM(cufftPlanMany):
    {
      cufftHandle *plan;
      memcpy(&plan, l->fncargs + chars_read, sizeof(plan));
      chars_read += sizeof(plan);

      int rank;
      memcpy(&rank, l->fncargs + chars_read, sizeof(rank));
      chars_read += sizeof(rank);

      int *n;
      memcpy(&n, l->fncargs + chars_read, sizeof(n));
      chars_read += sizeof(n);

      int *inembed;
      memcpy(&inembed, l->fncargs + chars_read, sizeof(inembed));
      chars_read += sizeof(inembed);

      int istride;
      memcpy(&istride, l->fncargs + chars_read, sizeof(istride));
      chars_read += sizeof(istride);

      int idist;
      memcpy(&idist, l->fncargs + chars_read, sizeof(idist));
      chars_read += sizeof(rank);

      int *onembed;
      memcpy(&onembed, l->fncargs + chars_read, sizeof(onembed));
      chars_read += sizeof(onembed);

      int ostride;
      memcpy(&ostride, l->fncargs + chars_read, sizeof(ostride));
      chars_read += sizeof(ostride);

      int odist;
      memcpy(&odist, l->fncargs + chars_read, sizeof(odist));
      chars_read += sizeof(odist);

      int type;
      memcpy(&type, l->fncargs + chars_read, sizeof(type));
      chars_read += sizeof(type);

      int batch;
      memcpy(&batch, l->fncargs + chars_read, sizeof(batch));
      chars_read += sizeof(batch);
      cufftResult ret = cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, (cufftType)type, batch);
      assert(ret == CUFFT_SUCCESS);
      break;
    }
  case GENERATE_ENUM(cufftSetStream):
  {
    cufftHandle plan;
    memcpy(&plan, l->fncargs + chars_read, sizeof(plan));
    chars_read += sizeof(plan);

    cudaStream_t stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof(stream));
    chars_read += sizeof(stream);

    cudaStream_t newStream;
    void *pStream = getCudaType((void *)&stream, CUDA_STREAM);
    if (NULL == pStream)
    {
      newStream = stream;
    }
    else
    {
      newStream = *((cudaStream_t *)pStream);
    }

    cufftResult ret = cufftSetStream(plan, newStream);
    assert(ret == CUFFT_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cusparseCreate):
  {
    cusparseHandle_t *handle;
    memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
    chars_read += sizeof(handle);
      cusparseCreate(handle);
    break;
  }
  case GENERATE_ENUM(cusparseDestroy):
  {
    cusparseHandle_t *handle;
    memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
    chars_read += sizeof(handle);
    cusparseDestroy(*handle);
    break;
  }
  case GENERATE_ENUM(cusparseCreateMatDescr):
  {
    cusparseMatDescr_t *handle;
    memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
    chars_read += sizeof(handle);
    cusparseCreateMatDescr(handle);
    break;
  }
  case GENERATE_ENUM(cusparseDestroyMatDescr):
  {
    cusparseMatDescr_t *handle;
    memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
    chars_read += sizeof(handle);
    cusparseDestroyMatDescr(*handle);
    break;
  }
  case GENERATE_ENUM(cusolverDnCreate):
  {
    cusolverDnHandle_t *handle;
    memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
    chars_read += sizeof(handle);
    cusolverDnCreate(handle);
    break;
  }
  case GENERATE_ENUM(cusolverDnDestroy):
  {
    cusolverDnHandle_t *handle;
    memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
    chars_read += sizeof(handle);
    cusolverDnDestroy(*handle);
    break;
  }
    case GENERATE_ENUM(cuDevicePrimaryCtxRelease_v2):
    {
      CUdevice dev;
      memcpy(&dev, l->fncargs + chars_read, sizeof(dev));
      chars_read += sizeof(dev);

      CUresult ret = cuDevicePrimaryCtxRelease_v2(dev);
      // Not erase the primary context from the contextmap, this should be fixed.
      assert(ret == CUDA_SUCCESS);
      break;
    }
    case GENERATE_ENUM(cuDevicePrimaryCtxReset_v2):
    {
      CUdevice dev;
      memcpy(&dev, l->fncargs + chars_read, sizeof(dev));
      chars_read += sizeof(dev);

      CUresult ret = cuDevicePrimaryCtxReset_v2(dev);
      assert(ret == CUDA_SUCCESS);
      break;
    }
    case GENERATE_ENUM(cuDevicePrimaryCtxRetain):
    {
      CUcontext context;
      memcpy(&context, l->fncargs + chars_read, sizeof(context));
      chars_read += sizeof(context);

      CUdevice dev;
      memcpy(&dev, l->fncargs + chars_read, sizeof(dev));
      chars_read += sizeof(dev);
      // printf("dev=%i\n", dev);

      CUcontext *pContext = new CUcontext();
      CUresult ret = cuDevicePrimaryCtxRetain(pContext, dev);
      if (context != NULL && (context != *pContext))
      {
          contextmap[context] = pContext;
      }
      // printf("cuDevicePrimaryCtxRetain, ret = %d, ctx = %p, newCtx = %p\n", ret, context, *pContext);
      assert(ret == CUDA_SUCCESS);
      break;
    }
  case GENERATE_ENUM(cuCtxCreate_v2):
  {
    CUcontext context;
    memcpy(&context, l->fncargs + chars_read, sizeof(context));
    chars_read += sizeof(context);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof(flags));
    chars_read += sizeof(flags);

    CUdevice dev;
    memcpy(&dev, l->fncargs + chars_read, sizeof(dev));
    chars_read += sizeof(dev);

    CUcontext *pContext = new CUcontext();
    CUresult ret = cuCtxCreate_v2(pContext, flags, dev);
    assert(ret == CUDA_SUCCESS);
    if (context != NULL && (context != *pContext))
    {
      contextmap[context] = pContext;
    }
    break;
  }
  case GENERATE_ENUM(cuCtxDestroy_v2):
  {
    CUcontext context;
    CUresult ret = CUDA_SUCCESS;
    memcpy(&context, l->fncargs + chars_read, sizeof(context));
    chars_read += sizeof(context);

    CUcontext *pContext = (CUcontext *)getCUtype((void *)&context, CU_CONTEXT);
    if (pContext != NULL)
    {
      ret = cuCtxDestroy_v2(*pContext);
      if (CUDA_SUCCESS == ret)
      {
        delete pContext;
        contextmap.erase(context);
      }
    }
    else
      ret = cuCtxDestroy_v2(context);

    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuCtxGetApiVersion):
  {
    CUcontext context;
    memcpy(&context, l->fncargs + chars_read, sizeof(context));
    chars_read += sizeof(context);

    unsigned int version;
    memcpy(&version, l->fncargs + chars_read, sizeof(version));
    chars_read += sizeof(version);

    CUresult ret = CUDA_SUCCESS;
    CUcontext *pContext = (CUcontext *)getCUtype((void *)&context, CU_CONTEXT);
    if (pContext != NULL)
    {
      ret = cuCtxGetApiVersion(*pContext, &version);
    }
    else
      ret = cuCtxGetApiVersion(context, &version);

    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuCtxGetCacheConfig):
  {
    CUfunc_cache *config;
    memcpy(&config, l->fncargs + chars_read, sizeof(config));
    chars_read += sizeof(config);

    CUresult ret = cuCtxGetCacheConfig(config);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuCtxGetCurrent):
  {
    CUcontext *context;
    memcpy(&context, l->fncargs + chars_read, sizeof(context));
    chars_read += sizeof(context);

    CUresult ret = cuCtxGetCurrent(context);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuCtxSetCurrent):
  {
    CUcontext context;
    memcpy(&context, l->fncargs + chars_read, sizeof(context));
    chars_read += sizeof(context);
    CUcontext* pContext = (CUcontext*)getCUtype((void*)&context, CU_CONTEXT);
    CUcontext ctx;
    if (pContext == NULL)
      ctx = context;
    else
      ctx = *pContext;
    CUresult ret = cuCtxSetCurrent(ctx);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuCtxGetDevice):
  {
    CUdevice *device;
    memcpy(&device, l->fncargs + chars_read, sizeof(device));
    chars_read += sizeof(device);

    CUresult ret = cuCtxGetDevice(device);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuCtxGetLimit):
  {
    size_t *pvalue;
    memcpy(&pvalue, l->fncargs + chars_read, sizeof(pvalue));
    chars_read += sizeof(pvalue);

    CUlimit limit;
    memcpy(&limit, l->fncargs + chars_read, sizeof(limit));
    chars_read += sizeof(limit);

    CUresult ret = cuCtxGetLimit(pvalue, limit);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuCtxSetLimit):
  {
    CUlimit limit;
    memcpy(&limit, l->fncargs + chars_read, sizeof(limit));
    chars_read += sizeof(limit);

    size_t value;
    memcpy(&value, l->fncargs + chars_read, sizeof(value));
    chars_read += sizeof(value);

    CUresult ret = cuCtxSetLimit(limit, value);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuLinkCreate_v2):
  {
    unsigned int numOptions;
    memcpy(&numOptions, l->fncargs + chars_read, sizeof(numOptions));
    chars_read += sizeof(numOptions);

    CUjit_option *options;
    memcpy(&options, l->fncargs + chars_read, sizeof(options));
    chars_read += sizeof(options);

    void **optionValues;
    memcpy(&optionValues, l->fncargs + chars_read, sizeof(optionValues));
    chars_read += sizeof(optionValues);
    // out parameter
    CUlinkState *stateOut;
    memcpy(&stateOut, l->fncargs + chars_read, sizeof(stateOut));
    chars_read += sizeof(stateOut);
    CUresult ret = cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuLinkDestroy):
  {
    CUlinkState *stateOut;
    memcpy(&stateOut, l->fncargs + chars_read, sizeof(stateOut));
    chars_read += sizeof(stateOut);
    CUresult ret = cuLinkDestroy(*stateOut);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuArray3DCreate_v2):
  {
    CUarray *pHandle;
    memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
    chars_read += sizeof(pHandle);

    CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray;
    memcpy(&pAllocateArray, l->fncargs + chars_read, sizeof(pAllocateArray));
    chars_read += sizeof(pAllocateArray);
    CUresult ret = cuArray3DCreate_v2(pHandle, pAllocateArray);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuArrayCreate_v2):
  {
    CUarray *pHandle;
    memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
    chars_read += sizeof(pHandle);

    CUDA_ARRAY_DESCRIPTOR *pAllocateArray;
    memcpy(&pAllocateArray, l->fncargs + chars_read, sizeof(pAllocateArray));
    chars_read += sizeof(pAllocateArray);
    CUresult ret = cuArrayCreate_v2(pHandle, pAllocateArray);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuArrayDestroy):
  {
    CUarray *pHandle;
    memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
    chars_read += sizeof(pHandle);
    CUresult ret =  cuArrayDestroy(*pHandle);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuMipmappedArrayCreate):
  {
    CUmipmappedArray *pHandle;
    memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
    chars_read += sizeof(pHandle);

    CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc;
    memcpy(&pMipmappedArrayDesc, l->fncargs + chars_read, sizeof(pMipmappedArrayDesc));
    chars_read += sizeof(pMipmappedArrayDesc);

    unsigned int numMipmapLevels;
    memcpy(&numMipmapLevels, l->fncargs + chars_read, sizeof(numMipmapLevels));
    chars_read += sizeof(numMipmapLevels);
    CUresult ret = cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuMipmappedArrayDestroy):
  {
    CUmipmappedArray *pHandle;
    memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
    chars_read += sizeof(pHandle);
    CUresult ret =  cuMipmappedArrayDestroy(*pHandle);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuInit):
  {
    unsigned int flag;
    memcpy(&flag, l->fncargs + chars_read, sizeof(flag));
    chars_read += sizeof(flag);
    CUresult ret = cuInit(0);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuDeviceGet):
  {
    CUdevice *cudevice;
    memcpy(&cudevice, l->fncargs + chars_read, sizeof(cudevice));
    chars_read += sizeof(cudevice);

    int ordinal;
    memcpy(&ordinal, l->fncargs + chars_read, sizeof(ordinal));
    chars_read += sizeof(ordinal);

    CUresult ret = cuDeviceGet(cudevice, ordinal);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cudaGetDevice):
  {
    int *device;
    memcpy(&device, l->fncargs + chars_read, sizeof(device));
    chars_read += sizeof(device);

    cudaError_t ret = cudaGetDevice(device);

    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cudaSetDevice):
  {
    int device;
    memcpy(&device, l->fncargs + chars_read, sizeof(device));
    chars_read += sizeof(device);
    cudaError_t ret = cudaSuccess;

    ret = cudaSetDevice(device);

    // unsigned int flag;
    // int inactive = 0;
    // CUdevice dev = device;
    // cuDevicePrimaryCtxGetState(dev, &flag, &inactive);
    // if (!inactive)
    // {
    //   CUcontext context;
    //   cuDevicePrimaryCtxRetain(&context, dev);
    // }
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(cuMemHostRegister_v2):
  {
    void *p;
    memcpy(&p, l->fncargs + chars_read, sizeof(void *));
    chars_read += sizeof(void *);
    size_t bytesize;
    memcpy(&bytesize, l->fncargs + chars_read, sizeof(size_t));
    chars_read += sizeof(size_t);
    unsigned int flag;
    memcpy(&flag, l->fncargs + chars_read, sizeof(unsigned int));
    chars_read += sizeof(unsigned int);
    CUresult ret = cuMemHostRegister_v2(p, bytesize, flag);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuMemHostUnregister):
  {
    void *p;
    memcpy(&p, l->fncargs + chars_read, sizeof(void *));
    chars_read += sizeof(void *);
    CUresult ret = cuMemHostUnregister(p);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuDeviceTotalMem_v2):
  {
    size_t bytes;
    memcpy(&bytes, l->fncargs + chars_read, sizeof(bytes));
    chars_read += sizeof(bytes);

    CUdevice cudevice;
    memcpy(&cudevice, l->fncargs + chars_read, sizeof(cudevice));
    chars_read += sizeof(cudevice);

    CUresult ret = cuDeviceTotalMem_v2(&bytes, cudevice);
    assert(ret == CUDA_SUCCESS);
    break;
  }

  case GENERATE_ENUM(cuDeviceComputeCapability):
  {
    int major;
    memcpy(&major, l->fncargs + chars_read, sizeof(major));
    chars_read += sizeof(major);

    int minor;
    memcpy(&minor, l->fncargs + chars_read, sizeof(minor));
    chars_read += sizeof(minor);

    CUdevice cudevice;
    memcpy(&cudevice, l->fncargs + chars_read, sizeof(cudevice));
    chars_read += sizeof(cudevice);

    CUresult ret = cuDeviceComputeCapability(&major, &minor, cudevice);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuDeviceGetProperties):
  {
    CUdevprop prop;
    memcpy(&prop, l->fncargs + chars_read, sizeof(CUdevprop));
    chars_read += sizeof(CUdevprop);

    CUdevice cudevice;
    memcpy(&cudevice, l->fncargs + chars_read, sizeof(cudevice));
    chars_read += sizeof(cudevice);

    CUresult ret = cuDeviceGetProperties(&prop, cudevice);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuDevicePrimaryCtxGetState):
  {
    CUdevice cudevice;
    memcpy(&cudevice, l->fncargs + chars_read, sizeof(cudevice));
    chars_read += sizeof(cudevice);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof(flags));
    chars_read += sizeof(flags);

    int active;
    memcpy(&active, l->fncargs + chars_read, sizeof(active));
    chars_read += sizeof(active);

    CUresult ret = CUDA_SUCCESS;

    ret = cuDevicePrimaryCtxGetState(cudevice, &flags, &active);

    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuDeviceGetUuid):
  {
    CUuuid uuid;
    memcpy(&uuid, l->fncargs + chars_read, sizeof(CUuuid));
    chars_read += sizeof(CUuuid);

    CUdevice cudevice;
    memcpy(&cudevice, l->fncargs + chars_read, sizeof(cudevice));
    chars_read += sizeof(cudevice);

    CUresult ret = cuDeviceGetUuid(&uuid, cudevice);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuEventElapsedTime):
  {
    float *pMilliseconds;
    memcpy(&pMilliseconds, l->fncargs + chars_read, sizeof(pMilliseconds));
    chars_read += sizeof(pMilliseconds);

    CUevent hStart;
    memcpy(&hStart, l->fncargs + chars_read, sizeof(hStart));
    chars_read += sizeof(hStart);

    CUevent hEnd;
    memcpy(&hEnd, l->fncargs + chars_read, sizeof(hEnd));
    chars_read += sizeof(hEnd);

    CUevent *start = (CUevent *)getCUtype((void *)&hStart, CU_EVENT);
    CUevent *end = (CUevent *)getCUtype((void *)&hEnd, CU_EVENT);
    CUresult ret = cuEventElapsedTime(pMilliseconds,
                               start != NULL ? *start : hStart,
                               end != NULL ? *end : hEnd);
    assert(ret == CUDA_SUCCESS);
    break;
  }

  case GENERATE_ENUM(cuEventSynchronize):
  {
    CUevent hEvent;
    memcpy(&hEvent, l->fncargs + chars_read, sizeof(hEvent));
    chars_read += sizeof(hEvent);

    CUevent *event = (CUevent *)getCUtype((void *)&hEvent, CU_EVENT);
    CUresult ret =  cuEventSynchronize(event != NULL ? *event : hEvent);

    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuModuleLoadData):
  {
    CUmodule module;
    memcpy(&module, l->fncargs + chars_read, sizeof(module));
    chars_read += sizeof(module);

    void *image;
    memcpy(&image, l->fncargs + chars_read, sizeof(image));
    chars_read += sizeof(image);
    // printf("image = %p\n", image);

    for (int i = 0; i < 10; i++)
    {
      if (ptxlg[i].addr == image)
      {
        // printf("we got the image info %ld\n", ptxlg[i].len);
        image = ptxlg[i].ptx;
        break;
      }
    }

    CUmodule *mymodule = new CUmodule();
    CUresult err = cuModuleLoadData(mymodule, image);

    if (module != NULL && (module != *mymodule))
    {
      modulemap[module] = mymodule;
    }
    // printf("replay cuModuleLoadData %d\n", err);
    assert(err == CUDA_SUCCESS);
    break;
  }

  case GENERATE_ENUM(cuModuleLoadDataEx):
  {
    CUmodule module;
    memcpy(&module, l->fncargs + chars_read, sizeof(module));
    chars_read += sizeof(module);

    void *image;
    memcpy(&image, l->fncargs + chars_read, sizeof(image));
    chars_read += sizeof(image);
    // printf("fengtao.xie image = %p\n", image);
    for (int i = 0; i < 10; i++)
    {
      //  printf("fengtao.xie we got the image[%d]= %p\n", i, ptxlg[i].addr);
      if (ptxlg[i].addr == image)
      {
        // printf("fengtao.xie we got the image info %ld\n", ptxlg[i].len);
        image = ptxlg[i].ptx;
        break;
      }
    }

    /*      unsigned int numOptions;
          memcpy(&numOptions, l->fncargs + chars_read, sizeof(numOptions));
          chars_read += sizeof(numOptions);

         CUjit_option * options;
          memcpy(&options, l->fncargs + chars_read, sizeof(options));
          chars_read += sizeof(options);

          void** optionValues;
          memcpy(&optionValues, l->fncargs + chars_read, sizeof(optionValues));
          chars_read += sizeof(optionValues);
    */

    CUmodule *mymodule = new CUmodule();
    // writeInfo(module, mymodule, RCMODULE);
    const unsigned int num_opts = 2;
    CUjit_option options[num_opts];
    void *values[num_opts];

    // set up size of compilation log buffer
    options[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    values[0] = (void *)(int)10240;
    // set up pointer to the compilation log buffer
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    char clog[10240];
    values[1] = clog;

    // modulemap.insert(std::pair<CUmodule, CUmodule*>(module, mymodule));
    modulemap[module] = mymodule;
    // CUresult err = cuModuleLoadDataEx(&mymodule, (void*)imagestr, numOptions, options, optionValues);
    CUresult err = cuModuleLoadDataEx(mymodule, image, num_opts, options, (void **)values);

    // printf("fengtao.xie replay cuModuleLoadDataEx %d\n", err);
    assert(err == CUDA_SUCCESS);
    break;
  }

  case GENERATE_ENUM(cuModuleGetFunction):
  {
    // CUfunction * hfunc;
    // memcpy(&hfunc, l->fncargs + chars_read, sizeof(hfunc));
    // chars_read += sizeof(hfunc);
    CUfunction func;
    memcpy(&func, l->fncargs + chars_read, sizeof(func));
    chars_read += sizeof(func);

    CUmodule hmod;
    memcpy(&hmod, l->fncargs + chars_read, sizeof(hmod));
    chars_read += sizeof(hmod);

    int len;
    memcpy(&len, l->fncargs + chars_read, sizeof(len));
    chars_read += sizeof(len);

    char *name = (char *)malloc(len + 1);
    memset(name, 0, len + 1);

    memcpy(name, l->fncargs + chars_read, len);
    chars_read += len;

    CUfunction *newfunc = new CUfunction();
    CUmodule *moduleptr = (CUmodule *)getCUtype((void *)&hmod, CU_MODULE);
    CUresult ret = CUDA_SUCCESS;
    ret = cuModuleGetFunction(newfunc, moduleptr != NULL ? *moduleptr : hmod, name);

    functionmap[func] = newfunc;
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuMemcpyHtoD_v2):
  {
    CUdeviceptr dstDevice;
    memcpy(&dstDevice, l->fncargs + chars_read, sizeof(dstDevice));
    chars_read += sizeof(dstDevice);

    // Need to logged
    // void* srcHost;
    // memcpy(&srcHost, l->fncargs + chars_read, sizeof(srcHost));
    // chars_read += sizeof(srcHost);

    size_t ByteCount;
    memcpy(&ByteCount, l->fncargs + chars_read, sizeof(ByteCount));
    chars_read += sizeof(char) * ByteCount;

    char *srcHostData = (char *)malloc(ByteCount * sizeof(char));
    memcpy(srcHostData, l->fncargs + chars_read, sizeof(char) * ByteCount);
    chars_read += sizeof(char) * ByteCount;

    CUdeviceptr *devptr = (CUdeviceptr *)getCUtype((void *)&dstDevice, CU_DEV_PTR);
    CUresult ret = CUDA_SUCCESS;
    ret = cuMemcpyHtoD_v2(devptr != NULL ? *devptr : dstDevice, (const void *)srcHostData, ByteCount);

    if (NULL != srcHostData)
      free(srcHostData);
    assert(ret == CUDA_SUCCESS);

    break;
  }
  case GENERATE_ENUM(cuMemcpyHtoDAsync_v2):
  {
    CUdeviceptr dstDevice;
    memcpy(&dstDevice, l->fncargs + chars_read, sizeof(dstDevice));
    chars_read += sizeof(dstDevice);

    // memcpy(&srcHost, l->fncargs + chars_read, sizeof(srcHost));
    // chars_read += sizeof(srcHost);

    size_t ByteCount;
    memcpy(&ByteCount, l->fncargs + chars_read, sizeof(ByteCount));
    chars_read += sizeof(ByteCount);

    char *srcHostData = (char *)malloc(ByteCount * sizeof(char));
    memcpy(srcHostData, l->fncargs + chars_read, sizeof(char) * ByteCount);
    chars_read += sizeof(char) * ByteCount;

    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    chars_read += sizeof(hStream);

    CUdeviceptr *devptr = (CUdeviceptr *)getCUtype((void *)&dstDevice, CU_DEV_PTR);
    CUstream *stream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    CUresult ret = CUDA_SUCCESS;
    ret = cuMemcpyHtoDAsync_v2(devptr != NULL ? *devptr : dstDevice, (const void *)srcHostData, ByteCount, stream != NULL ? *stream : hStream);

    if (NULL != srcHostData)
      free(srcHostData);
    // printf("cuMemcpyHtoDAsync_v2 ret = %d \n", ret);
    assert(ret == CUDA_SUCCESS);
    // Free the srcHostData
    break;
  }
  case GENERATE_ENUM(cuMemcpyDtoH_v2):
  {
    void *dstHost;
    memcpy(&dstHost, l->fncargs + chars_read, sizeof(dstHost));
    chars_read += sizeof(dstHost);

    CUdeviceptr srcDevice;
    memcpy(&srcDevice, l->fncargs + chars_read, sizeof(srcDevice));
    chars_read += sizeof(srcDevice);

    size_t ByteCount;
    memcpy(&ByteCount, l->fncargs + chars_read, sizeof(ByteCount));
    chars_read += sizeof(ByteCount);

    CUdeviceptr *devptr = (CUdeviceptr *)getCUtype((void *)&srcDevice, CU_DEV_PTR);
    CUresult ret = CUDA_SUCCESS;
    ret = cuMemcpyDtoH(dstHost, devptr != NULL ? *devptr : srcDevice, ByteCount);

    // printf("cuMemcpyDtoH_v2 ret = %d \n", ret);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuMemcpyDtoHAsync_v2):
  {
    void *dstHost;
    memcpy(&dstHost, l->fncargs + chars_read, sizeof(dstHost));
    chars_read += sizeof(dstHost);

    CUdeviceptr srcDevice;
    memcpy(&srcDevice, l->fncargs + chars_read, sizeof(srcDevice));
    chars_read += sizeof(srcDevice);

    size_t ByteCount;
    memcpy(&ByteCount, l->fncargs + chars_read, sizeof(ByteCount));
    chars_read += sizeof(ByteCount);

    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    chars_read += sizeof(hStream);

    CUstream *stream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    CUdeviceptr *devptr = (CUdeviceptr *)getCUtype((void *)&srcDevice, CU_DEV_PTR);
    // printf("cuMemcpyDtoHAsync_v2 ... \n");
    // printf("oldDevice = %llu, newDevice = %p\n", srcDevice, devptr);
    // printf("oldStream = %p, newStream = %p, dstHost = %p \n", hStream, *stream, dstHost);
    CUresult ret = cuMemcpyDtoHAsync_v2(dstHost, devptr != NULL ? *devptr : srcDevice, ByteCount, stream != NULL ? *stream : hStream);
    // printf("cuMemcpyDtoHAsync_v2 ret = %d \n", ret);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuMemcpyDtoDAsync_v2):
  {
    CUdeviceptr dstDevice;
    memcpy(&dstDevice, l->fncargs + chars_read, sizeof(dstDevice));
    chars_read += sizeof(dstDevice);

    CUdeviceptr srcDevice;
    memcpy(&srcDevice, l->fncargs + chars_read, sizeof(srcDevice));
    chars_read += sizeof(srcDevice);

    size_t ByteCount;
    memcpy(&ByteCount, l->fncargs + chars_read, sizeof(ByteCount));
    chars_read += sizeof(ByteCount);

    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    chars_read += sizeof(hStream);

    CUstream *stream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    CUdeviceptr *dstDevPtr = (CUdeviceptr *)getCUtype((void *)&dstDevice, CU_DEV_PTR);
    CUdeviceptr *srcDevPtr = (CUdeviceptr *)getCUtype((void *)&srcDevice, CU_DEV_PTR);
    CUresult ret = CUDA_SUCCESS;
    ret = cuMemcpyDtoDAsync_v2(dstDevPtr != NULL ? *dstDevPtr : dstDevice,
                                srcDevPtr != NULL ? *srcDevPtr : srcDevice, ByteCount, stream != NULL ? *stream : hStream);
    // printf("cuMemcpyDtoDAsync_v2 ret = %d \n", ret);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuEventRecord):
  {
    // printf("fengtao.xie in GENERATE_ENUM(cuEventRecord)\n");
    CUevent hEvent;
    memcpy(&hEvent, l->fncargs + chars_read, sizeof(hEvent));
    chars_read += sizeof(hEvent);

    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    chars_read += sizeof(hStream);

    CUevent *event = (CUevent *)getCUtype((void *)&hEvent, CU_EVENT);
    CUstream *stream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);

    CUresult ret = CUDA_SUCCESS;
    ret = cuEventRecord(event != NULL ? *event : hEvent, stream != NULL ? *stream : hStream);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuModuleGetTexRef):
  {
    CUtexref oldTexRef;
    memcpy(&oldTexRef, l->fncargs + chars_read, sizeof(oldTexRef));
    chars_read += sizeof(oldTexRef);

    CUmodule hmod;
    memcpy(&hmod, l->fncargs + chars_read, sizeof(hmod));
    chars_read += sizeof(hmod);

    int len;
    memcpy(&len, l->fncargs + chars_read, sizeof(len));
    chars_read += sizeof(len);

    char *name = (char *)malloc(len * sizeof(char));
    memcpy(name, l->fncargs + chars_read, len * sizeof(char));
    chars_read += len * sizeof(char);

    CUmodule *newhmod = (CUmodule *)getCUtype((void *)&hmod, CU_MODULE);
    CUtexref *pTexRef = new CUtexref();
    CUresult ret = CUDA_SUCCESS;
    ret = cuModuleGetTexRef(pTexRef, newhmod != NULL ? *newhmod : hmod, name);
    assert(ret == CUDA_SUCCESS);
    texrefmap[oldTexRef] = pTexRef;
    free(name);
    break;
  }
  case GENERATE_ENUM(cuTexRefSetAddress_v2):
  {
    /*
     *This parameter is nullptr in lammps, not store this parameter now.
     */
    size_t *ByteOffset;
    memcpy(&ByteOffset, l->fncargs + chars_read, sizeof(ByteOffset));
    chars_read += sizeof(ByteOffset);

    CUtexref hTexRef;
    memcpy(&hTexRef, l->fncargs + chars_read, sizeof(hTexRef));
    chars_read += sizeof(hTexRef);

    CUdeviceptr dptr;
    memcpy(&dptr, l->fncargs + chars_read, sizeof(dptr));
    chars_read += sizeof(dptr);

    size_t bytes;
    memcpy(&bytes, l->fncargs + chars_read, sizeof(bytes));
    chars_read += sizeof(bytes);

    CUtexref *newTexRef = (CUtexref *)getCUtype((void *)&hTexRef, CU_TEXREF);
    CUdeviceptr *devptr = (CUdeviceptr *)getCUtype((void *)&dptr, CU_DEV_PTR);
    CUresult ret = CUDA_SUCCESS;
    ret = cuTexRefSetAddress_v2(ByteOffset,
                                newTexRef != NULL ? *newTexRef : hTexRef,
                                devptr != NULL ? *devptr : dptr, bytes);
    assert(ret == CUDA_SUCCESS);

    break;
  }
  case GENERATE_ENUM(cuTexRefSetFormat):
  {
    CUtexref hTexRef;
    memcpy(&hTexRef, l->fncargs + chars_read, sizeof(hTexRef));
    chars_read += sizeof(hTexRef);

    CUarray_format fmt;
    memcpy(&fmt, l->fncargs + chars_read, sizeof(fmt));
    chars_read += sizeof(fmt);

    int NumPackedComponents;
    memcpy(&NumPackedComponents, l->fncargs + chars_read, sizeof(NumPackedComponents));
    chars_read += sizeof(NumPackedComponents);

    CUtexref *newTexRef = (CUtexref *)getCUtype((void *)&hTexRef, CU_TEXREF);
    CUresult ret = CUDA_SUCCESS;
    ret = cuTexRefSetFormat(newTexRef != NULL ? *newTexRef : hTexRef, fmt, NumPackedComponents);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuDriverGetVersion):
  {
    printf("cuDriverGetVersion not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuDeviceGetCount):
  {
    printf("cuDeviceGetCount not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuDeviceGetName):
  {
    char name;
    memcpy(&name, l->fncargs + chars_read, sizeof(name));
    chars_read += sizeof(name);

    int len;
    memcpy(&len, l->fncargs + chars_read, sizeof(len));
    chars_read += sizeof(len);

    CUdevice cudevice;
    memcpy(&cudevice, l->fncargs + chars_read, sizeof(cudevice));
    chars_read += sizeof(cudevice);

    CUresult ret = cuDeviceGetName(&name, len, cudevice);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuDeviceGetAttribute):
  {
    printf("cuDeviceGetAttribute not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuCtxSynchronize):
  {
    CUresult ret = CUDA_SUCCESS;
    ret = cuCtxSynchronize();
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuPointerGetAttributes):
  {
    unsigned int numAttributes;
    memcpy(&numAttributes, l->fncargs + chars_read, sizeof(numAttributes));
    chars_read += sizeof(numAttributes);
    CUpointer_attribute* attributes = NULL;
    attributes = (CUpointer_attribute*)calloc(1, sizeof(CUpointer_attribute));
    for (unsigned int i = 0; i < numAttributes; i++)
    {
      memcpy(&attributes[i], l->fncargs + chars_read, sizeof(attributes[i]));
      chars_read += sizeof(attributes[i]);
    }
    CUcontext dataValue;
    memcpy(&dataValue, l->fncargs + chars_read, sizeof(dataValue));
    chars_read += sizeof(dataValue);
    CUdeviceptr ptr;
    memcpy(&ptr, l->fncargs + chars_read, sizeof(ptr));
    chars_read += sizeof(ptr);
    CUresult ret = CUDA_SUCCESS;

	CUcontext** data = NULL;
    data = (CUcontext**)calloc(1, sizeof(CUcontext*));
    for (unsigned int i = 0; i < numAttributes; i++) {
      data[i] = (CUcontext *)calloc(1, sizeof(CUcontext));
    }

    int ctxIdx = -1;
    for (unsigned int i = 0; i < numAttributes; i++)
    {
      if (attributes[i] == CU_POINTER_ATTRIBUTE_CONTEXT) {
        // TODO: find the context in lower-half, only for device pointer
        ctxIdx = i;
        break;
      }
    }
    ret = cuPointerGetAttributes(numAttributes, attributes, (void **)data, ptr);
    assert(ret == CUDA_SUCCESS);

    // TODO: if the attributes has the mem context
    LhGetMemCtx_t ctxFnc = (LhGetMemCtx_t)(lhInfo.lhGetMemCtxFptr);
    CUcontext *pContext = new CUcontext();
    if (ctxIdx >= 0)
    {
      void* ctx = ctxFnc((void*)ptr);
      // printf("find ctx.....\n");
      // fflush(stdout);
      if (ctx != 0) {
        // printf("ctx: %p, pid:%i, tid:%i\n", ctx, pid, tid);
        memcpy(pContext, &ctx, sizeof(CUcontext));
      }
      if (dataValue != NULL && (dataValue != *pContext))
      {
         contextmap[dataValue] = pContext;
      }
    }
    break;
  }
  case GENERATE_ENUM(cuModuleLoad):
  {
    // Not call
    printf("cuModuleLoad not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuMemGetInfo_v2):
  {
    printf("cuMemGetInfo_v2 not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuMemcpy2D_v2):
  {
    // Not call
    printf("cuMemcpy2D_v2 not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuMemcpy2DAsync_v2):
  {
    // Not call
    printf("cuMemcpy2DAsync_v2 not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuMemsetD16Async):
  {
    // Not call
    printf("cuMemsetD16Async not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuMemsetD32Async):
  {
    // Not call
    printf("cuMemsetD32Async not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuMemsetD8Async):
  {
    CUdeviceptr dstDevice;
    memcpy(&dstDevice, l->fncargs + chars_read, sizeof(dstDevice));
    chars_read += sizeof(dstDevice);

    int uc;
    memcpy(&uc, l->fncargs + chars_read, sizeof(uc));
    chars_read += sizeof(uc);

    size_t N;
    memcpy(&N, l->fncargs + chars_read, sizeof(N));
    chars_read += sizeof(N);

    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    chars_read += sizeof(hStream);

    CUdeviceptr *devptr = (CUdeviceptr *)getCUtype((void *)&dstDevice, CU_DEV_PTR);
    CUstream *stream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    CUresult ret = CUDA_SUCCESS;
    ret = cuMemsetD8Async(devptr != NULL ? *devptr : dstDevice,
                          (unsigned char)uc, N, stream != NULL ? *stream : hStream);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuStreamSynchronize):
  {

    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    chars_read += sizeof(hStream);

    CUstream *stream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);
    CUresult ret = CUDA_SUCCESS;
    ret = cuStreamSynchronize(stream != NULL ? *stream : hStream);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  case GENERATE_ENUM(cuStreamWaitEvent):
  {
    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    chars_read += sizeof(hStream);

    CUevent hEvent;
    memcpy(&hEvent, l->fncargs + chars_read, sizeof(hEvent));
    chars_read += sizeof(hEvent);

    CUevent *event = (CUevent *)getCUtype((void *)&hEvent, CU_EVENT);
    CUstream *stream = (CUstream *)getCUtype((void *)&hStream, CU_STREAM);

    unsigned int flag;
    memcpy(&flag, l->fncargs + chars_read, sizeof(flag));
    chars_read += sizeof(flag);

    CUresult ret = CUDA_SUCCESS;
    ret = cuStreamWaitEvent(stream != NULL ? *stream : hStream, event != NULL ? *event : hEvent, flag);
    assert(ret == CUDA_SUCCESS);
    break;
  }
  /*case GENERATE_ENUM(cuStreamAddCallback):
{
    CUstream hStream;
    memcpy(&hStream, l->fncargs + chars_read, sizeof(hStream));
    chars_read += sizeof(hStream);

    CUstreamCallback callback;
    memcpy(&callback, l->fncargs + chars_read, sizeof(callback));
    chars_read += sizeof(callback);

    int userData;
    memcpy(&userData, l->fncargs + chars_read, sizeof(userData));
    chars_read += sizeof(userData);

    unsigned int flags;
    memcpy(&flags, l->fncargs + chars_read, sizeof(flags));
    chars_read += sizeof(flags);

    CUstream* stream = (CUstream*)getCUtype((void*)&hStream,CU_STREAM);
    CUresult ret = cuStreamAddCallback(stream != NULL ? *stream : hStream, callback, &userData, flags);

    assert(ret == CUDA_SUCCESS);
    break;
  }*/
  case GENERATE_ENUM(cuFuncSetBlockShape):
  {
    // Not call
    printf("cuFuncSetBlockShape not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuLaunchGridAsync):
  {
    // CUDA_VERSION < 4000
    printf("cuLaunchGridAsync not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuParamSetSize):
  {
    // CUDA_VERSION < 4000
    printf("cuParamSetSize not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuParamSetTexRef):
  {
    // CUDA_VERSION < 4000
    printf("cuParamSetTexRef not implemented\n");
    break;
  }
  case GENERATE_ENUM(cuParamSetv):
  {
    // CUDA_VERSION < 4000
    printf("cuParamSetv not implemented\n");
    break;
  }
  case GENERATE_ENUM(cudaLaunchKernel):
  {
    void *func_addr;
    void **args;
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    memcpy(&func_addr, l->fncargs + chars_read, sizeof(void *));
    chars_read += sizeof(void *);

    memcpy(&gridDim, l->fncargs + chars_read, sizeof gridDim);
    chars_read += sizeof gridDim;

    memcpy(&blockDim, l->fncargs + chars_read, sizeof blockDim);
    chars_read += sizeof blockDim;

    memcpy(&args, l->fncargs + chars_read, sizeof(void *));
    chars_read += sizeof(void *);

    memcpy(&sharedMem, l->fncargs + chars_read, sizeof sharedMem);
    chars_read += sizeof sharedMem;

    memcpy(&stream, l->fncargs + chars_read, sizeof(void *));

    cudaError_t ret = cudaSuccess;
    // replay
    ret = cudaLaunchKernel(func_addr, gridDim, blockDim, args, sharedMem, stream);
    assert(ret == cudaSuccess);
    break;
  }
  case GENERATE_ENUM(ncclCommInitRank):
  {
    ncclComm_t* comm;
    memcpy(&comm, l->fncargs + chars_read, sizeof(ncclComm_t*));
    chars_read += sizeof(ncclComm_t*);

    int nranks;
    memcpy(&nranks, l->fncargs + chars_read, sizeof(int));
    chars_read += sizeof(int);

    ncclUniqueId commId;
    memcpy(&commId, l->fncargs + chars_read, sizeof(ncclUniqueId));
    chars_read += sizeof(ncclUniqueId);

    int rank;
    memcpy(&rank, l->fncargs + chars_read, sizeof(int));
    chars_read += sizeof(int);

    ncclComm_t orignalComm;
    memcpy(&orignalComm, l->fncargs + chars_read, sizeof(ncclComm_t));

    // printf("ncclCommInitRank commId in replay:0x%llx, rank:%i, nranks:%i, tid:%i\n", (unsigned long long)hashUniqueId(commId), rank, nranks, tid);
    // fflush(stdout);

    unsigned long long uniqueIdHash =  hashUniqueId(commId);
    if (isUniqueIdGet)
    {
      commId = *(uniqueIdMap[uniqueIdHash]);
    }
    else
    {
      ncclUniqueId *newUniqueId = (ncclUniqueId*)malloc(sizeof(ncclUniqueId));
      ncclResult_t retTmp = ncclGetUniqueId(newUniqueId);
      if (retTmp != ncclSuccess)
        printf("get uniqueId from ncclCommInitRank failure....\n");
      else
      {
        commId = *newUniqueId;
        uniqueIdMap[uniqueIdHash] = newUniqueId;
      }
    }

    ncclResult_t ret = ncclSuccess;

    ret = ncclCommInitRank(comm, nranks, commId, rank);


    assert(ret == ncclSuccess);
    // update the ncclComm map
    ncclCommMap[orignalComm] = *comm;
    break;
  }
  case GENERATE_ENUM(ncclCommInitAll):
  {
    ncclComm_t* comm;
    memcpy(&comm, l->fncargs + chars_read, sizeof(ncclComm_t*));
    chars_read += sizeof(ncclComm_t*);

    int ndev;
    memcpy(&ndev, l->fncargs + chars_read, sizeof(int));
    chars_read += sizeof(int);

    int* devlist;
    memcpy(&devlist, l->fncargs + chars_read, sizeof(int*));
    chars_read += sizeof(int*);

    ncclComm_t* tmp = (ncclComm_t*)malloc(sizeof(ncclComm_t) * ndev);
    for (int i = 0; i < ndev; i++)
    {
      memcpy(tmp + i, l->fncargs + chars_read, sizeof(ncclComm_t));
      chars_read += sizeof(ncclComm_t);
    }

    ncclResult_t ret = ncclSuccess;
    ret = ncclCommInitAll(comm, ndev, devlist);

    // TODO: build map for new ncclComm and old ncclComm
    for (int i = 0; i < ndev; i++)
    {
      // printf("comm %i in replay,new:%p old:%p\n", i, comm[i], tmp[i]);
      // fflush(stdout);
      ncclCommMap[tmp[i]] = comm[i];
    }

    assert(ret == ncclSuccess);
    break;
  }
  // case GENERATE_ENUM(ncclCommFinalize):
  // {
  //   ncclComm_t comm;
  //   memcpy(&comm, l->fncargs + chars_read, sizeof(ncclComm_t));
  //   chars_read += sizeof(ncclComm_t*);

  //   ncclComm_t newComm;
  //   if(ncclCommMap.count(comm))
  //     newComm = ncclCommMap[comm];
  //   else
  //     newComm = comm;

  //   ncclResult_t ret = ncclSuccess;
  //   if( tid % 1000 == 0)
  //     ret = ncclCommFinalize(newComm);
  //   else
  //     cudaFuncExec(GENERATE_ENUM(ncclCommFinalize), tid, &ret, newComm);

  //   assert(ret == ncclSuccess);

  //   break;
  // }
  case GENERATE_ENUM(ncclCommDestroy):
  {
    ncclComm_t* comm;
    memcpy(&comm, l->fncargs + chars_read, sizeof(ncclComm_t*));
    chars_read += sizeof(ncclComm_t*);

    ncclResult_t ret = ncclSuccess;
    ret = ncclCommDestroy(*comm);

    assert(ret == ncclSuccess);
    break;
  }
  case GENERATE_ENUM(ncclCommAbort):
  {
    ncclComm_t* comm;
    memcpy(&comm, l->fncargs + chars_read, sizeof(ncclComm_t*));
    chars_read += sizeof(ncclComm_t*);

    ncclResult_t ret = ncclSuccess;
    ret = ncclCommAbort(*comm);

    assert(ret == ncclSuccess);
    break;
  }
  case GENERATE_ENUM(ncclCommCount):
  {
    // TODO: need to implement
    break;
  }
  case GENERATE_ENUM(ncclCommCuDevice):
  {
    // TODO: need to implement
    break;
  }
  case GENERATE_ENUM(ncclCommUserRank):
  {
    // TODO: need to implement
    break;
  }
  // case GENERATE_ENUM(ncclRedOpCreatePreMulSum):
  // {
  //   // TODO: need to implement
  //   break;
  // }
  // case GENERATE_ENUM(ncclRedOpDestroy):
  // {
  //   // TODO: need to implement
  //   break;
  // }
  case GENERATE_ENUM(ncclReduce):
  {
    // TODO: need to implement
    break;
  }
  case GENERATE_ENUM(ncclBcast):
  {
    // TODO: need to implement
    break;
  }
  case GENERATE_ENUM(ncclBroadcast):
  {
    // TODO: need to implement
    break;
  }
  case GENERATE_ENUM(ncclAllReduce):
  {
    void* sendbuff;
    memcpy(&sendbuff, l->fncargs + chars_read, sizeof(void*));
    chars_read += sizeof(void*);

    void* recvbuff;
    memcpy(&recvbuff, l->fncargs + chars_read, sizeof(void*));
    chars_read += sizeof(void*);

    size_t count = 0;
    memcpy(&count, l->fncargs + chars_read, sizeof(size_t));
    chars_read += sizeof(size_t);

    int datatype;
    memcpy(&datatype, l->fncargs + chars_read, sizeof(int));
    chars_read += sizeof(int);

    int op;
    memcpy(&op, l->fncargs + chars_read, sizeof(int));
    chars_read += sizeof(int);

    ncclComm_t comm;
    memcpy(&comm, l->fncargs + chars_read, sizeof(ncclComm_t));
    chars_read += sizeof(ncclComm_t);

    cudaStream_t stream;
    memcpy(&stream, l->fncargs + chars_read, sizeof(cudaStream_t));
    chars_read += sizeof(cudaStream_t);

    ncclResult_t ret = ncclSuccess;
    cudaStream_t *pStream = (cudaStream_t *)getCudaType((void *)&stream, CUDA_STREAM);
    if (NULL != pStream)
    {
      stream = *pStream;
    }

    ncclComm_t newComm = (ncclComm_t)getNickleType((void*)&comm, NCCL_COMM);
    if (newComm != NULL) {
      //printf("new Comm:%p, saved Comm:%p\n", newComm, comm);
      comm = newComm;
    }

    // printf("comm:%p stream:%p.... 1\n",comm, stream);
    // fflush(stdout);

      ret = ncclAllReduce(sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op, comm, stream);
    assert(ret == ncclSuccess);
    break;
  }
  case GENERATE_ENUM(ncclReduceScatter):
  {
    // TODO: need to implement
    break;
  }
  case GENERATE_ENUM(ncclAllGather):
  {
    // TODO: need to implement
    break;
  }
  case GENERATE_ENUM(ncclSend):
  {
    // TODO: need to implement
    break;
  }
  case GENERATE_ENUM(ncclRecv):
  {
    // TODO: need to implement
    break;
  }
  case GENERATE_ENUM(ncclGroupStart):
  {
    ncclResult_t ret = ncclSuccess;
    ret = ncclGroupStart();
    assert(ret == ncclSuccess);
    break;
  }
  case GENERATE_ENUM(ncclGroupEnd):
  {
    ncclResult_t ret = ncclSuccess;

      ret = ncclGroupEnd();

    assert(ret == ncclSuccess);
    break;
  }
  case GENERATE_ENUM(ncclGetUniqueId):
  {
    ncclUniqueId uniqueId;
    memcpy(&uniqueId, l->fncargs + chars_read, sizeof(ncclUniqueId));
    chars_read += sizeof(ncclUniqueId);

    unsigned long long uniqueIdHash = hashUniqueId(uniqueId);

    ncclUniqueId* newUniqueId = (ncclUniqueId*)malloc(sizeof(ncclUniqueId));
    ncclResult_t ret = ncclSuccess;

    ret = ncclGetUniqueId(newUniqueId);

    assert(ret == ncclSuccess);
    isUniqueIdGet = true;
    uniqueIdMap[uniqueIdHash] = newUniqueId;
    // printf("ncclGetUniqueId in replay:0x%llx, original id:0x%llx\n", (unsigned long long)hashUniqueId(*newUniqueId), uniqueIdHash);
    break;
  }
  case GENERATE_ENUM(ncclGetVersion):
  {
    break;
  }
  default:
    assert(false);
    break;
    // JASSERT(false)(op).Text("Replaying unknown op code");
  }
}
// getter for fatCubinHandle generated by replayed __cudaRegisterFatBinary
void **fatHandle()
{
  void **fatCubinHandle = new_fatCubinHandle;
  new_fatCubinHandle = NULL;
  return fatCubinHandle;
}
// This function iterates over the CUDA calls log and calls the given
// function on each call log object
// void logs_read_and_apply(void (*apply)(CudaCallLog_t *l))

void callbackPtx()
{
  int id = 0, ret = 0;
  /*
   *Read kernel data from this file, the format of kernel data
   *is addr, len, data | addr, len, data | ...
   */
  char filename[100];
  pid_t orig_pid = getpid();
  snprintf(filename, 100, "./moduledata_%d.dat", orig_pid);
  int fd = open(filename, O_RDONLY, 0600);
  while (1)
  {
    int size = read(fd, &ptxlg[id].addr, sizeof(void *));
    // printf("size = %d \n", size);
    if (size <= 0)
    {
      break; // Read finished
    }
    size = read(fd, &ptxlg[id].len, sizeof(long));
    if (0 == size)
    {
      ret = 1;
      break;
    }

    // printf("replay size is %ld\n", ptxlg[id].len);
    ptxlg[id].ptx = malloc(ptxlg[id].len);
    size = read(fd, ptxlg[id].ptx, ptxlg[id].len);
    if (0 == size)
    {
      ret = 1;
      break;
    }

    id++;
  }

  if (1 == ret)
  {
    printf("Read moduledata failed!!!\n");
  }
  else
  {
    printf("Read moduledata succeed...\n");
  }
}

void logs_read_and_apply()
{
  // Read ptx data from file to memory
  //  callbackPtx();

  // TODO: get the cuda logs before main functions
  char filename[100];
  pid_t orig_pid = mtcpHdr.orig_pid;
  snprintf(filename, 100, "./uhInfo_cuda_log_%d", orig_pid);
  // printf("[LT] uhInfo file name:%s\n", filename);
  int fd = open(filename, O_RDONLY, 0644);
  assert(fd != -1);
  size_t unlogSize = 0;
  size_t rc = read(fd, &unlogSize, sizeof(size_t));
  assert(rc >= sizeof(size_t));
  for (size_t i = 0; i < unlogSize; i++)
  {
    CudaCallLog_t tmpLog;
    rc = read(fd, &tmpLog, sizeof(CudaCallLog_t));
    assert(rc >= sizeof(CudaCallLog_t));
    replayAPI(&tmpLog);
  }

  GetCudaCallsLogFptr_t fnc = (GetCudaCallsLogFptr_t)uhInfo.cudaLogVectorFptr;
  std::vector<CudaCallLog_t> &cudaCallsLog = fnc();
  size_t count = 0;
  for (auto it = cudaCallsLog.begin(); it != cudaCallsLog.end(); it++)
  {
    count++;
    replayAPI(&(*it));
  }

  GetPageLockInfosFptr_t fnc_1 = (GetPageLockInfosFptr_t)uhInfo.uhPageLockMapFptr;
  std::map<void *, page_lock_info_t> &pageLockMap = fnc_1();
  for (auto page_lock_info : pageLockMap)
  {
    unsigned int flag = page_lock_info.second.flags;
    if (flag == 0x04)
      flag = 0x01;
    cudaError_t ret =  cudaHostRegister((void *)(page_lock_info.first), page_lock_info.second.size, flag);
    printf("register cuda pin memory, ret:%i, ptr:%p\n", ret, page_lock_info.first);
  }

  // by tian01.liu for ipc replay, 2023.8.15
  g_finishRelay = true;
}

// Added by biao.xing@samsung.com in 2024/1/15 for offline refilling solution
typedef struct __handle_info
{
    uint64_t value;  // value of CU handle
    uint64_t location; // address of CU handle in segment
} HandleInfo;
std::list<HandleInfo> gHandleList;

int openCuHandleInfoFile()
{
  char filename[100];
  pid_t orig_pid = getUhPid();
  snprintf(filename, 100, "./handle_info_%d", orig_pid);
  int fd = open(filename, O_RDONLY);
  if (fd < 0)
  {
    printf("Could not open upper-half file for reading. %s \n", filename);
    exit(-1);
  }

  return fd;
}

void refillNewHandle()
{
  Area area;
  std::list<HandleInfo>::iterator it = gHandleList.begin();
  int mapsfd = open("/proc/self/maps", O_RDONLY);
  while (readMapsLine(mapsfd, &area))
  {
    if (it == gHandleList.end())
    {
      // Refill finish, exit
      break;
    }

    if ((VA)(((HandleInfo)*it).location) < area.addr || (VA)(((HandleInfo)*it).location) > area.endAddr)
    {
      continue;
    }

    int ret = mprotect(area.addr, area.size, area.prot | PROT_WRITE);
    if (ret < 0)
    {
      fprintf(stderr, "Failed to add write permissions for memory region (%s)"
        "at: %p of : %zu bytes. Error: %s\n", area.name, area.addr, area.size, strerror(errno));
    }

    while ((VA)(((HandleInfo)*it).location) >= area.addr && (VA)(((HandleInfo)*it).location) <= area.endAddr)
    {
      // Find CU handle in this segment
      uint64_t handle = ((HandleInfo)*it).value;
      // fprintf(stdout, "REFILL OK, location : 0x%zx, re-created handle : 0x%zx\n", ((HandleInfo)*it).location, handle);
      memcpy((VA)((HandleInfo)*it).location, &handle, sizeof(uint64_t));
      it++;
    }
    ret = mprotect(area.addr, area.size, area.prot);
    if (ret < 0)
    {
      fprintf(stderr, "Failed to add revert permissions for memory region (%s)"
        "at: %p of : %zu bytes. Error: %s\n", area.name, area.addr, area.size, strerror(errno));
    }
  }

  close(mapsfd);
}

uint64_t findNewHandle(uint64_t handle)
{
  for (auto &iter : contextmap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(*(iter.second));
  }

  for (auto &iter : modulemap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(*(iter.second));
  }

  for (auto &iter : functionmap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(*(iter.second));
  }

  for (auto &iter : eventmap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(*(iter.second));
  }

  for (auto &iter : streammap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(*(iter.second));
  }

  for (auto &iter : texrefmap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(*(iter.second));
  }

  for (auto &iter : cudaStreamMap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(*(iter.second));
  }

  for (auto &iter : cudaEventMap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(*(iter.second));
  }

  for (auto &iter : cublasHandleMap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(*(iter.second));
  }

  for (auto &iter : ncclCommMap)
  {
    if(handle == (uint64_t)(iter.first)) return (uint64_t)(iter.second);
  }
  // Not find new handle, so return original handle directly
  return 0; //handle;
}

void printf_maps()
{
  for (auto &iter : contextmap)
  {
    fprintf(stdout,"ctx, first = %p, ctx, second = %p.\n", iter.first, *(iter.second));
  }

  for (auto &iter : modulemap)
  {
    fprintf(stdout,"mod, first = %p, mod, second = %p.\n", iter.first, *(iter.second));
  }

  for (auto &iter : functionmap)
  {
    fprintf(stdout,"fnc, first = %p, fnc, second = %p.\n", iter.first, *(iter.second));
  }

  for (auto &iter : eventmap)
  {
    fprintf(stdout,"evt, first = %p, evt, second = %p.\n", iter.first, *(iter.second));
  }

  for (auto &iter : streammap)
  {
    fprintf(stdout,"stream, first = %p, stream, second = %p.\n", iter.first, *(iter.second));
  }

  for (auto &iter : texrefmap)
  {
    fprintf(stdout,"texref, first = %p, txref, second = %p.\n", iter.first, *(iter.second));
  }

  for (auto &iter : cudaStreamMap)
  {
    fprintf(stdout,"cudaStream, first = %p, cudaStream, second = %p.\n", iter.first, *(iter.second));
  }

  for (auto &iter : cudaEventMap)
  {
    fprintf(stdout,"cudaEvent, first = %p, cudaEvent, second = %p.\n", iter.first, *(iter.second));
  }

  for (auto &iter : cublasHandleMap)
  {
    fprintf(stdout,"cublasHandle, first = %p, cublas, second = %p.\n", iter.first, *(iter.second));
  }

  for (auto &iter : ncclCommMap)
  {
    fprintf(stdout,"cublasHandle, first = %p, ncclComm, second = %p.\n", iter.first, iter.second);
  }
}

void replaceOrigHandle(int fd)
{
  HandleInfo handle_info;
  // printf_maps();
  // Read and refill
  while (read(fd, &handle_info, sizeof(HandleInfo)))
  {
    //Replace original hanle with re-created handle
    handle_info.value = findNewHandle(handle_info.value);
    // fprintf(stdout, "REPLACE OK : 0x%zx, ori handle : 0x%zx, re-created handle : 0x%zx\n",
    //    handle_info.location, ori_handle, handle_info.value);
    // Insert the new handles to list
    gHandleList.push_back(handle_info);
  }
}

void refill_handles()
{
  clock_t startRefill, endRefill;
  startRefill = clock();
  int fd = openCuHandleInfoFile();
  if (fd < 0)
  {
    printf("Could not open handle info file for reading \n");
    exit(-1);
  }

  // Replace original handles with re-created handles
  replaceOrigHandle(fd);

  // Refill re-created handles to the location of original handles
  refillNewHandle();

  // Clear the handles list after refilling finish
  gHandleList.clear();

  endRefill = clock();
  fprintf(stdout,"Refill finish. time = %.6f\n", (double)(endRefill - startRefill) / CLOCKS_PER_SEC * 1000);
}

