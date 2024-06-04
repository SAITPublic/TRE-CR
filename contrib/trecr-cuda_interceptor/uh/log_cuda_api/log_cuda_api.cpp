/****************************************************************************
 *  Copyright (C) 2019-2020 by Twinkle Jain, Rohan garg, and Gene Cooperman *
 *  jain.t@husky.neu.edu, rohgarg@ccs.neu.edu, gene@ccs.neu.edu             *
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

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>   // test code, by tian01.liu
#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <unordered_set> // by tian01.liu 2023.9.4
#include <stack>         // by tian01.liu 2023.9.4
#include "cudart_apis.h"

#include "jassert.h"
#include "log_and_replay.h"


using namespace std;
static bool disable_logging = false;
extern "C" pid_t doCheckpoint(int i); // support timing by huiru.deng
// struct CudaCallLog_t {
//   void *fncargs;
//   size_t size;
//   void *results;
// };

// enum pages_t {
//   CUDA_MALLOC_PAGE = 0,
//   CUDA_UVM_PAGE,
//   ....
// };

// typedef struct Lhckpt_pages_t {
//   pages_t mem_type;
//   size_t mem_len;
// }lhckpt_pages_t;
extern pthread_mutex_t mutex_for_map;
extern pthread_mutex_t mutex_for_log;
extern pthread_mutex_t mutex_for_logtag;

map<void *, lhckpt_pages_t> lh_pages_map;

map<void *, lhckpt_pages_t> &
getLhPageMaps()
{
  return lh_pages_map;
}

std::vector<CudaCallLog_t> cudaCallsLog;

std::vector<CudaCallLog_t> &
getCudaCallsLog()
{
  return cudaCallsLog;
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

// add by tian01.liu 2023.9.4, store all the nccl fnc ops
unordered_set<Cuda_Fncs_t> ncclOpSets = {Cuda_Fnc_ncclCommInitRank,
                                         Cuda_Fnc_ncclCommInitAll,
                                         Cuda_Fnc_ncclCommDestroy,
                                        /*Cuda_Fnc_ncclCommFinalize,*/
                                         Cuda_Fnc_ncclAllGather,
                                         Cuda_Fnc_ncclAllReduce,
                                         Cuda_Fnc_ncclBcast,
                                         Cuda_Fnc_ncclBroadcast,
                                         Cuda_Fnc_ncclCommAbort,
                                         Cuda_Fnc_ncclCommCount,
                                         Cuda_Fnc_ncclCommCuDevice,
                                         Cuda_Fnc_ncclCommGetAsyncError,
                                         Cuda_Fnc_ncclCommUserRank,
                                         Cuda_Fnc_ncclGetErrorString,
                                        /*  Cuda_Fnc_ncclGetLastError, */
                                         Cuda_Fnc_ncclGetUniqueId,
                                         Cuda_Fnc_ncclGetVersion,
                                         Cuda_Fnc_ncclGroupEnd,
                                         Cuda_Fnc_ncclGroupStart,
                                         Cuda_Fnc_ncclRecv,
                                        /*  Cuda_Fnc_ncclRedOpCreatePreMulSum,
                                         Cuda_Fnc_ncclRedOpDestroy,*/
                                         Cuda_Fnc_ncclReduce,
                                         Cuda_Fnc_ncclReduceScatter,
                                         Cuda_Fnc_ncclSend};

bool isNickleApi(Cuda_Fncs_t op)
{
  return ncclOpSets.count(op);
}

void display_map()
{
  for (auto lh_page : lh_pages_map) {
    printf("\n Address = %p with size = %lu", \
     lh_page.first, lh_page.second.mem_len);
  }
}

void
enableLogging()
{
  disable_logging = false;
}

void
disableLogging()
{
  disable_logging = true;
}

bool
isLoggingDisabled()
{
  return disable_logging;
}

extern __thread__ pid_t tid;
// This function does in-memory logging of CUDA calls that are specified
// using the @LogReplay decorator.
void
logAPI(Cuda_Fncs_t cuda_fnc_op, ...)
{
  if (isLoggingDisabled())
  {
    return;
  }
  // printf("In logAPI\n");
  // pid_t tid = syscall(SYS_gettid);
  va_list arglist;
  va_start(arglist, cuda_fnc_op);
  CudaCallLog_t log;
  // test code, by tian01.liu
  log.thread_id = tid;


  // fengtao.xie change 4096 to 40960
  char buf[4096*10];
  size_t chars_wrote = 0;
  // fill the cuda function op fisrtmem_typeto the buf
  memcpy(buf + chars_wrote, &cuda_fnc_op, sizeof cuda_fnc_op);
  chars_wrote += sizeof cuda_fnc_op;
  // printf("==== In logAPI %s:start\n", cuda_Fnc_to_str[cuda_fnc_op]);
  // fflush(stdout);

  switch (cuda_fnc_op) {
    case GENERATE_ENUM(cudaMalloc):
    {
      // args
      /**** begin: by tian01.liu for new virtual address space architecture 2023.1.9 */
      void **pointer = va_arg(arglist, void **);
      // memcpy(buf + chars_wrote, pointer, sizeof (void *));
      // chars_wrote += sizeof (void *);

      size_t size = va_arg(arglist, size_t);
      // memcpy(buf + chars_wrote, &size, sizeof size);
      // chars_wrote += sizeof (size);
      /**** end: by tian01.liu for new virtual address space architecture 2023.1.9 */

      int deviceId = -1;
      cudaGetDevice(&deviceId);
      // update the map
      lhckpt_pages_t page = {CUDA_MALLOC_PAGE, *pointer, size, 0, deviceId, false};
      // printf("[lt] log_api, cudaMalloc key:%p\n", *pointer);
      pthread_mutex_lock(&mutex_for_map);
      lh_pages_map[*pointer] = page;
      pthread_mutex_unlock(&mutex_for_map);
      // display_map();
      return;  // by tian01.liu, No log this api in checkpoint.from break to return
    }
    case GENERATE_ENUM(cuMemAlloc_v2):
    {
      // args
       /**** begin: by tian01.liu for new virtual address space architecture 2023.1.9 */
      CUdeviceptr *pointer = va_arg(arglist, CUdeviceptr *);

      // memcpy(buf + chars_wrote, pointer, sizeof (CUdeviceptr));
      // chars_wrote += sizeof (CUdeviceptr);

      size_t size = va_arg(arglist, size_t);
      // memcpy(buf + chars_wrote, &size, sizeof size);
      // chars_wrote += sizeof (size);
      /**** end: by tian01.liu for new virtual address space architecture 2023.1.9 */
      int deviceId = -1;
      cudaGetDevice(&deviceId);

      // update the map
      lhckpt_pages_t page = {CUDA_MALLOC_PAGE, (void*)(*pointer), size, 0, deviceId, false};
      pthread_mutex_lock(&mutex_for_map);
      lh_pages_map[(void*)(*pointer)] = page;
      pthread_mutex_unlock(&mutex_for_map);

      // used to gpu compress
      // lhckpt_pages_t page = {CUMEM_ALLOC_PAGE, (void*)(*pointer), size};
      // lh_pages_map[(void*)(*pointer)] = page;
      // display_map();
      return; // by tian01.liu, No log this api in checkpoint. from break to return
    }
    case GENERATE_ENUM(cudaFree):
    {
      // args
      /**** begin: by tian01.liu for new virtual address space architecture 2023.1.9 */
      void *pointer = va_arg(arglist, void *);
      // memcpy(buf + chars_wrote, &pointer, sizeof (void *));
      // chars_wrote += sizeof (void *);
      /**** end: by tian01.liu for new virtual address space architecture 2023.1.9 */

      // remove from maps
      // printf("[lt] log api, cudaFree key:%p\n", pointer);
      pthread_mutex_lock(&mutex_for_map);
      lh_pages_map.erase(pointer);
      pthread_mutex_unlock(&mutex_for_map);
      return; // by tian01.liu, No log this api in checkpoint. from break to return
    }
    case GENERATE_ENUM(cuMemFree_v2):
    {
      // args
      /**** begin: by tian01.liu for new virtual address space architecture 2023.1.9 */
      CUdeviceptr pointer = va_arg(arglist, CUdeviceptr);
      // memcpy(buf + chars_wrote, &pointer, sizeof(CUdeviceptr));
      // chars_wrote += sizeof (CUdeviceptr);
      /**** end: by tian01.liu for new virtual address space architecture 2023.1.9 */

      // remove from maps
      pthread_mutex_lock(&mutex_for_map);
      lh_pages_map.erase((void*)pointer);
      pthread_mutex_unlock(&mutex_for_map);

      // used to gpu compress
      // lh_pages_map.erase((void*)pointer);
      return; // by tian01.liu, No log this api in checkpoint. from break to return
    }
    case GENERATE_ENUM(__cudaInitModule):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, fatCubinHandle, sizeof (void *));
      chars_wrote += sizeof (void *);

      break;
    }
    case GENERATE_ENUM(__cudaPushCallConfiguration):
    {
      // args
      dim3 gridDim = va_arg(arglist, dim3);
      memcpy(buf + chars_wrote, &gridDim, sizeof (gridDim));
      chars_wrote += sizeof (gridDim);

      dim3 blockDim = va_arg(arglist, dim3);
      memcpy(buf + chars_wrote, &blockDim, sizeof (blockDim));
      chars_wrote += sizeof (blockDim);

      size_t sharedMem = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &sharedMem, sizeof (sharedMem));
      chars_wrote += sizeof (sharedMem);

      void *stream = va_arg(arglist, void *);
      memcpy(buf + chars_wrote, &stream, sizeof (void *));
      chars_wrote += sizeof (void *);

      log.res_size = sizeof(unsigned int);
      log.results = (char *)JALLOC_MALLOC(log.res_size + 1);

      unsigned int res = va_arg(arglist, unsigned int);
      memcpy(log.results, &res, sizeof (unsigned int));

      break;
    }
    case GENERATE_ENUM(__cudaPopCallConfiguration):
    {
      // args
      dim3 *gridDim = va_arg(arglist, dim3 *);
      memcpy(buf + chars_wrote, &gridDim, sizeof (gridDim));
      chars_wrote += sizeof (gridDim);

      dim3 *blockDim = va_arg(arglist, dim3 *);
      memcpy(buf + chars_wrote, &blockDim, sizeof (blockDim));
      chars_wrote += sizeof (blockDim);

      size_t *sharedMem = va_arg(arglist, size_t *);
      memcpy(buf + chars_wrote, &sharedMem, sizeof (sharedMem));
      chars_wrote += sizeof (sharedMem);

      void *stream = va_arg(arglist, void *);
      memcpy(buf + chars_wrote, &stream, sizeof (void *));
      chars_wrote += sizeof (void *);

      break;
    }
    case GENERATE_ENUM(__cudaRegisterFatBinary):
    {
      // args
      void *fatCubin = va_arg(arglist, void *);
      memcpy(buf + chars_wrote, &fatCubin, sizeof (void *));
      chars_wrote += sizeof (void *);
      // result
      void **res = va_arg(arglist, void **);
      // memcpy(buf + chars_wrote, &res, sizeof (res));
      // chars_wrote += sizeof (res);

      log.res_size = sizeof(*res);
      log.results = (char *)JALLOC_MALLOC(log.res_size + 1);
      memcpy(log.results, res, sizeof (*res));
      break;
    }
    // new
    case GENERATE_ENUM(__cudaRegisterFatBinaryEnd):
    {
      // args
      // void **fatCubinHandle = va_arg(arglist, void **);
      // memcpy(buf + chars_wrote, &fatCubinHandle, sizeof (void *));
      // chars_wrote += sizeof (void *);
      break;
    }

    case GENERATE_ENUM(__cudaUnregisterFatBinary):
    {
      // args
      // void **fatCubinHandle = va_arg(arglist, void **);
      // memcpy(buf + chars_wrote, fatCubinHandle, sizeof (*fatCubinHandle));
      // chars_wrote += sizeof (*fatCubinHandle);

      break;
    }
    case GENERATE_ENUM(__cudaRegisterFunction):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      char *hostFun = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &hostFun, sizeof(char *));
      chars_wrote += sizeof(char *);


      char *deviceFun = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceFun, sizeof(char *));
      chars_wrote += sizeof(char *);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceName, sizeof(char *));
      chars_wrote += sizeof(char *);

      int thread_limit = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &thread_limit, sizeof thread_limit);
      chars_wrote += sizeof thread_limit;

      uint3 *tid = va_arg(arglist, uint3 *);
      memcpy(buf + chars_wrote, &tid, sizeof (uint3 *));
      chars_wrote += sizeof (uint3 *);

      uint3 *bid = va_arg(arglist, uint3 *);
      memcpy(buf + chars_wrote, &bid, sizeof (uint3 *));
      chars_wrote += sizeof (uint3 *);

      dim3 *bDim = va_arg(arglist, dim3 *);
      memcpy(buf + chars_wrote, &bDim, sizeof (dim3 *));
      chars_wrote += sizeof (dim3 *);

      dim3 *gDim = va_arg(arglist, dim3 *);
      memcpy(buf + chars_wrote, &gDim, sizeof (dim3 *));
      chars_wrote += sizeof (dim3 *);

      int *wSize = va_arg(arglist, int*);
      memcpy(buf + chars_wrote, &wSize, sizeof (int *));
      chars_wrote += sizeof (int *);
      break;
    }
    case GENERATE_ENUM(__cudaRegisterVar):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      char *hostVar = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &hostVar, sizeof(char *));
      chars_wrote += sizeof(char *);

      char *deviceAddress = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceAddress, sizeof(char *));
      chars_wrote += sizeof(char *);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceName, sizeof(char *));
      chars_wrote += sizeof(char *);

      int ext = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ext, sizeof ext);
      chars_wrote += sizeof ext;

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof (size));
      chars_wrote += sizeof (size);

      int constant = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &constant, sizeof constant);
      chars_wrote += sizeof constant;

      int global = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &global, sizeof global);
      chars_wrote += sizeof global;
      break;
    }
    case GENERATE_ENUM(__cudaRegisterManagedVar):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      void **hostVarPtrAddress = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &hostVarPtrAddress, \
      sizeof (*hostVarPtrAddress));
      chars_wrote += sizeof (*hostVarPtrAddress);

      char *deviceAddress = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceAddress, sizeof(char *));
      chars_wrote += sizeof(char *);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceName, sizeof(char *));
      chars_wrote += sizeof(char *);

      int ext = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ext, sizeof ext);
      chars_wrote += sizeof ext;

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof (size));
      chars_wrote += sizeof (size);

      int constant = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &constant, sizeof constant);
      chars_wrote += sizeof constant;

      int global = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &global, sizeof global);
      chars_wrote += sizeof global;
      break;
    }
    case GENERATE_ENUM(__cudaRegisterTexture):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      struct textureReference *hostVar = va_arg(arglist,
                                                struct textureReference *);
      memcpy(buf + chars_wrote, &hostVar, sizeof(hostVar));
      chars_wrote += sizeof(hostVar);

      void **deviceAddress = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &deviceAddress, sizeof(deviceAddress));
      chars_wrote += sizeof(deviceAddress);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceName, sizeof(deviceName));
      chars_wrote += sizeof(deviceName);

      int dim = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &dim, sizeof dim);
      chars_wrote += sizeof dim;

      int norm = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &norm, sizeof norm);
      chars_wrote += sizeof norm;

      int ext = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ext, sizeof ext);
      chars_wrote += sizeof ext;
      break;
    }
    case GENERATE_ENUM(__cudaRegisterSurface):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      struct surfaceReference *hostVar = va_arg(arglist,
                                                struct surfaceReference *);
      memcpy(buf + chars_wrote, &hostVar, sizeof(hostVar));
      chars_wrote += sizeof(hostVar);

      void **deviceAddress = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &deviceAddress, sizeof(deviceAddress));
      chars_wrote += sizeof(deviceAddress);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceName, sizeof(deviceName));
      chars_wrote += sizeof(deviceName);

      int dim = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &dim, sizeof dim);
      chars_wrote += sizeof dim;

      int ext = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ext, sizeof ext);
      chars_wrote += sizeof ext;
      break;
    }
    case GENERATE_ENUM(cudaCreateTextureObject):
    {
      // args
      cudaTextureObject_t * pTexObject = va_arg(arglist, cudaTextureObject_t *);
      memcpy(buf + chars_wrote, &pTexObject, sizeof(pTexObject));
      chars_wrote += sizeof(pTexObject);

      struct cudaResourceDesc * pResDesc = va_arg(arglist,
                                                  struct cudaResourceDesc *);
      memcpy(buf + chars_wrote, &pResDesc, sizeof(pResDesc));
      chars_wrote += sizeof(pResDesc);

      struct cudaTextureDesc * pTexDesc = va_arg(arglist,
                                                 struct cudaTextureDesc *);
      memcpy(buf + chars_wrote, &pTexDesc, sizeof(pTexDesc));
      chars_wrote += sizeof(pTexDesc);

      struct cudaResourceViewDesc * pResViewDesc = va_arg(arglist,
                                                 struct cudaResourceViewDesc *);
      memcpy(buf + chars_wrote, &pResViewDesc, sizeof(pResViewDesc));
      chars_wrote += sizeof(pResViewDesc);
      break;
    }
    case GENERATE_ENUM(cudaDestroyTextureObject):
    {
      // args
      cudaTextureObject_t texObject = va_arg(arglist, cudaTextureObject_t);
      memcpy(buf + chars_wrote, &texObject, sizeof(texObject));
      chars_wrote += sizeof(texObject);
      break;
    }
    case GENERATE_ENUM(cudaBindTextureToArray):
    {
      textureReference* texref = va_arg(arglist, textureReference *);
      memcpy(buf + chars_wrote, &texref, sizeof(texref));
      chars_wrote += sizeof(texref);

      cudaArray_const_t array = va_arg(arglist, cudaArray_const_t);
      memcpy(buf + chars_wrote, &array, sizeof(array));
      chars_wrote += sizeof(array);

      cudaChannelFormatDesc * desc = va_arg(arglist, cudaChannelFormatDesc *);
      memcpy(buf + chars_wrote, &desc, sizeof(desc));
      chars_wrote += sizeof(desc);
      break;
    }
    case GENERATE_ENUM(cudaUnbindTexture):
    {
      struct textureReference * texref = va_arg(arglist, textureReference *);
      memcpy(buf + chars_wrote, &texref, sizeof(texref));
      chars_wrote += sizeof(texref);
      break;
    }
    case GENERATE_ENUM(cudaCreateChannelDesc):
    {
      // args
      int x = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &x, sizeof(x));
      chars_wrote += sizeof(x);

      int y = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &y, sizeof(y));
      chars_wrote += sizeof(y);

      int z = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &z, sizeof(z));
      chars_wrote += sizeof(z);

      int w = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &w, sizeof(w));
      chars_wrote += sizeof(w);

      cudaChannelFormatDesc f = va_arg(arglist, cudaChannelFormatDesc);
      memcpy(buf + chars_wrote, &f, sizeof(f));
      chars_wrote += sizeof(f);

      // result
      cudaChannelFormatDesc res = va_arg(arglist, cudaChannelFormatDesc);
      log.res_size = sizeof(res);
      log.results = (char *)JALLOC_MALLOC(log.res_size + 1);
      memcpy(log.results, &res, sizeof (res));
      break;
    }
    case GENERATE_ENUM(cudaMallocArray):
    {
      // args
      cudaArray_t ** array = va_arg(arglist, cudaArray_t **);
      memcpy(buf + chars_wrote, &array, sizeof(array));
      chars_wrote += sizeof(array);

      struct cudaChannelFormatDesc * desc = va_arg(arglist,
                                              struct cudaChannelFormatDesc *);
      memcpy(buf + chars_wrote, &desc, sizeof(desc));
      chars_wrote += sizeof(desc);

      size_t width = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &width, sizeof(width));
      chars_wrote += sizeof(width);

      size_t height = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &height, sizeof(height));
      chars_wrote += sizeof(height);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cudaFreeArray):
    {
      // args
      cudaArray_t array = va_arg(arglist, cudaArray_t);
      cudaArray_t *arrayptr = &array;
      memcpy(buf + chars_wrote, &arrayptr, sizeof(arrayptr));
      chars_wrote += sizeof(arrayptr);
      break;
    }
    case GENERATE_ENUM(cudaMallocHost):
    {
      // args
      void ** ptr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &ptr, sizeof(void **));
      chars_wrote += sizeof(void **);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;
      break;
    }
    case GENERATE_ENUM(cuMemAllocHost_v2):
    {
      // args
      void ** ptr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &ptr, sizeof(void **));
      chars_wrote += sizeof(void **);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;
      break;
    }
    case GENERATE_ENUM(cudaFreeHost):
    {
      // args
      void * ptr = va_arg(arglist, void *);
      void ** ptrptr = &ptr;
      memcpy(buf + chars_wrote, &ptrptr, sizeof(ptrptr));
      chars_wrote += sizeof(ptrptr);
      break;
    }
    case GENERATE_ENUM(cuMemFreeHost):
    {
      // args
      void * ptr = va_arg(arglist, void *);
      void ** ptrptr = &ptr;
      memcpy(buf + chars_wrote, &ptrptr, sizeof(void **));
      chars_wrote += sizeof(void **);
      break;
    }
    case GENERATE_ENUM(cudaHostAlloc):
    {
      // args
      void ** ptr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &ptr, sizeof(ptr));
      chars_wrote += sizeof(ptr);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cuMemHostAlloc):
    {
      // args
      void ** ptr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &ptr, sizeof(ptr));
      chars_wrote += sizeof(ptr);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cudaDeviceReset):
    {
      break;
    }
    case GENERATE_ENUM(cudaDeviceSetLimit):
    {
      int limit = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &limit, sizeof(limit));
      chars_wrote += sizeof(limit);

      size_t value = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &value, sizeof value);
      chars_wrote += sizeof(value);
      break;
    }
    case GENERATE_ENUM(cudaDeviceSetCacheConfig):
    {
      int cacheConfig = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &cacheConfig, sizeof(cacheConfig));
      chars_wrote += sizeof(cacheConfig);
      break;
    }
    case GENERATE_ENUM(cudaMallocPitch):
    {
      void ** devPtr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &devPtr, sizeof(devPtr));
      chars_wrote += sizeof(devPtr);

      size_t * pitch = va_arg(arglist, size_t *);
      memcpy(buf + chars_wrote, &pitch, sizeof pitch);
      chars_wrote += sizeof(pitch);

      size_t width = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &width, sizeof(width));
      chars_wrote += sizeof(width);

      size_t height = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &height, sizeof(height));
      chars_wrote += sizeof(height);
      break;
    }
    case GENERATE_ENUM(cuMemAllocPitch_v2):
    {
      CUdeviceptr* devPtr = va_arg(arglist, CUdeviceptr*);
      memcpy(buf + chars_wrote, &devPtr, sizeof(devPtr));
      chars_wrote += sizeof(devPtr);

      size_t * pitch = va_arg(arglist, size_t *);
      memcpy(buf + chars_wrote, &pitch, sizeof pitch);
      chars_wrote += sizeof(pitch);

      size_t width = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &width, sizeof(width));
      chars_wrote += sizeof(width);

      size_t height = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &height, sizeof(height));
      chars_wrote += sizeof(height);

      unsigned int ElementSizeBytes = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &ElementSizeBytes, sizeof(ElementSizeBytes));
      chars_wrote += sizeof(ElementSizeBytes);
      break;
    }
    case GENERATE_ENUM(cudaDeviceSynchronize):
    {
      break;
    }
    /**************begin:by tian01.liu for cuda ipc********************/
    case GENERATE_ENUM(cudaIpcGetMemHandle):
    {
      cudaIpcMemHandle_t *handle = va_arg(arglist, cudaIpcMemHandle_t*);
      memcpy(buf + chars_wrote, handle, sizeof(cudaIpcMemHandle_t));
      chars_wrote += sizeof(cudaIpcMemHandle_t);

      void* devPtr = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &devPtr, sizeof(void*));
      chars_wrote += sizeof(void*);

      break;
    }
    case GENERATE_ENUM(cudaIpcOpenMemHandle):
    {
      // TODO: the section used to replay api
      void* ptr = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &ptr, sizeof(void*));
      chars_wrote += sizeof(void*);

      cudaIpcMemHandle_t handle = va_arg(arglist, cudaIpcMemHandle_t);
      memcpy(buf + chars_wrote, &handle, sizeof(cudaIpcMemHandle_t));
      chars_wrote += sizeof(cudaIpcMemHandle_t);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(unsigned int));
      chars_wrote += sizeof(unsigned int);

      // TODO: this section used to save the data.
      size_t size = va_arg(arglist, size_t);
      unsigned int devId = va_arg(arglist, unsigned int);

      lhckpt_pages_t page = {CUMEM_ALLOC_PAGE, ptr, size, 0, (int)devId, true};
      lh_pages_map[ptr] = page;
      break;
    }
    /**************end  :by tian01.liu for cuda ipc********************/
    case GENERATE_ENUM(cudaMallocManaged):
    {
      // args
      /*********begin: by tian01.liu for new virtual address space architecture 2023.1.9 ************/
      void ** devPtr = va_arg(arglist, void **);
      // memcpy(buf + chars_wrote, devPtr, sizeof(void *));
      // chars_wrote += sizeof(void *);

      size_t size = va_arg(arglist, size_t);
      // memcpy(buf + chars_wrote, &size, sizeof size);
      // chars_wrote += sizeof size;

      // unsigned int flags = va_arg(arglist, unsigned int);
      // memcpy(buf + chars_wrote, &flags, sizeof(flags));
      // chars_wrote += sizeof(flags);
      /*********end: by tian01.liu for new virtual address space architecture 2023.1.9 ************/

      // update the map
      lhckpt_pages_t page = {CUDA_UVM_PAGE, *devPtr, size, 0, 0, false};
      lh_pages_map[*devPtr] = page;
      // display_map();
      return; // by tian01.liu 2023.1.9, from break to return
    }
    case GENERATE_ENUM(cuMemAllocManaged):
    {
      // args
      /*********begin: by tian01.liu for new virtual address space architecture 2023.1.9 ************/
      CUdeviceptr * devPtr = va_arg(arglist, CUdeviceptr*);
      // memcpy(buf + chars_wrote, devPtr, sizeof(CUdeviceptr));
      // chars_wrote += sizeof(CUdeviceptr);

      size_t size = va_arg(arglist, size_t);
      // memcpy(buf + chars_wrote, &size, sizeof size);
      // chars_wrote += sizeof size;

      // unsigned int flags = va_arg(arglist, unsigned int);
      // memcpy(buf + chars_wrote, &flags, sizeof(flags));
      // chars_wrote += sizeof(flags);
      /*********end: by tian01.liu for new virtual address space architecture 2023.1.9 ************/

      // update the map
      lhckpt_pages_t page = {CUDA_UVM_PAGE, devPtr, size, 0, 0, false};
      lh_pages_map[devPtr] = page;
      display_map();
      return; // by tian01.liu from break to return
    }
    case GENERATE_ENUM(cudaStreamCreate):
    {
      // args
      // Log the cudaStream_t, not cudaStream_t *
      cudaStream_t *pStream = va_arg(arglist, cudaStream_t *);
      memcpy(buf + chars_wrote, pStream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);
      break;
    }
    case GENERATE_ENUM(cuStreamCreate):
    {
      // args
      CUstream *phStream = va_arg(arglist, CUstream *);
      memcpy(buf + chars_wrote, phStream, sizeof(CUstream));
      chars_wrote += sizeof(CUstream);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cuMemHostRegister_v2):
    {
      void* p = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &p, sizeof(void*));
      chars_wrote += sizeof(void*);
      size_t bytesize = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &bytesize, sizeof(size_t));
      chars_wrote += sizeof(size_t);
      unsigned int flag = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flag, sizeof(unsigned int));
      chars_wrote += sizeof(unsigned int);
      break;
    }
    case GENERATE_ENUM(cuMemHostUnregister):
    {
      void* p = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &p, sizeof(void*));
      chars_wrote += sizeof(void*);
      break;
    }
    case GENERATE_ENUM(cudaStreamCreateWithFlags):
    {
      // args
      cudaStream_t *pStream = va_arg(arglist, cudaStream_t *);
      memcpy(buf + chars_wrote, pStream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cudaStreamCreateWithPriority):
    {
      // args
      cudaStream_t *pStream = va_arg(arglist, cudaStream_t *);
      memcpy(buf + chars_wrote, pStream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);

      int priority = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &priority, sizeof(priority));
      chars_wrote += sizeof(priority);
      break;
    }
    case GENERATE_ENUM(cuStreamCreateWithPriority):
    {
      // args
      CUstream *phStream = va_arg(arglist, CUstream *);
      memcpy(buf + chars_wrote, phStream, sizeof(CUstream));
      chars_wrote += sizeof(CUstream);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);

      int priority = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &priority, sizeof(priority));
      chars_wrote += sizeof(priority);
      break;
    }
    case GENERATE_ENUM(cuStreamQuery):
    {
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);
      break;
    }
    case GENERATE_ENUM(cuCtxGetSharedMemConfig):
    {
      CUsharedconfig *pConfig = va_arg(arglist, CUsharedconfig*);
      memcpy(buf + chars_wrote, pConfig, sizeof(CUsharedconfig));
      chars_wrote += sizeof(CUsharedconfig);
      break;
    }
    case GENERATE_ENUM(cuCtxGetStreamPriorityRange):
    {
      int* leastPriority = va_arg(arglist, int*);
      memcpy(buf + chars_wrote, leastPriority, sizeof(int));
      chars_wrote += sizeof(int);

      int* greatestPriority = va_arg(arglist, int*);
      memcpy(buf + chars_wrote, greatestPriority, sizeof(int));
      chars_wrote += sizeof(int);
      break;
	  }
    case GENERATE_ENUM(cuStreamGetCtx):
	  {
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);

      CUcontext* pctx = va_arg(arglist, CUcontext*);
      memcpy(buf + chars_wrote, pctx, sizeof(CUcontext));
      chars_wrote += sizeof(CUcontext);
      break;
    }
    case GENERATE_ENUM(cuStreamGetFlags):
    {
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);

      unsigned int* flags = va_arg(arglist, unsigned int*);
      memcpy(buf + chars_wrote, flags, sizeof(unsigned int));
      chars_wrote += sizeof(unsigned int);
      break;
    }
    case GENERATE_ENUM(cuStreamGetPriority):
    {
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);

      int* priority = va_arg(arglist, int*);
      memcpy(buf + chars_wrote, priority, sizeof(int));
      chars_wrote += sizeof(int);
      break;
    }
    case GENERATE_ENUM(cuStreamIsCapturing):
    {
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);

      CUstreamCaptureStatus *captureStatus = va_arg(arglist, CUstreamCaptureStatus*);
      memcpy(buf + chars_wrote, captureStatus, sizeof(CUstreamCaptureStatus));
      chars_wrote += sizeof(CUstreamCaptureStatus);
      break;
    }
    case GENERATE_ENUM(cudaStreamDestroy):
    {
      // args
      cudaStream_t pStream = va_arg(arglist, cudaStream_t);

      memcpy(buf + chars_wrote, &pStream, sizeof(cudaStream_t));
      chars_wrote += sizeof(pStream);
      break;
    }
    case GENERATE_ENUM(cuStreamDestroy_v2):
    {
      // args
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);
      break;
    }
    case GENERATE_ENUM(cudaEventCreate):
    {
      // leeyy_Note[epic=bugfix,seq=1282] cudaEventCreate log api
      cudaEvent_t * event = va_arg(arglist, cudaEvent_t *);
      memcpy(buf + chars_wrote, event, sizeof(cudaEvent_t));
      chars_wrote += sizeof(cudaEvent_t);
      break;
    }
    case GENERATE_ENUM(cuEventCreate):
    {
      CUevent *event = va_arg(arglist, CUevent *);
      memcpy(buf + chars_wrote, event, sizeof(CUevent));
      chars_wrote += sizeof(CUevent);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cudaEventDestroy):
    {
      cudaEvent_t event = va_arg(arglist, cudaEvent_t );
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);
      break;
    }
    case GENERATE_ENUM(cuEventDestroy_v2):
    {
      CUevent event = va_arg(arglist, CUevent);
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);
      break;
    }
    case GENERATE_ENUM(cudaEventRecord):
    {
      cudaEvent_t event = va_arg(arglist, cudaEvent_t);
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);
      cudaStream_t stream = va_arg(arglist, cudaStream_t);
      memcpy(buf + chars_wrote, &stream, sizeof(stream));
      break;
    }
    case GENERATE_ENUM(cudaEventSynchronize):
    {
      cudaEvent_t event = va_arg(arglist, cudaEvent_t);
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);
      break;
    }
    case GENERATE_ENUM(cudaEventCreateWithFlags):
    {
      cudaEvent_t * event = va_arg(arglist, cudaEvent_t *);
      memcpy(buf + chars_wrote, event, sizeof(cudaEvent_t));
      chars_wrote += sizeof(cudaEvent_t);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cuDestroyExternalMemory):
    {
      CUexternalMemory extMem = va_arg(arglist, CUexternalMemory);
      CUexternalMemory * extMemptr = &extMem;
      memcpy(buf + chars_wrote, &extMemptr, sizeof(extMemptr));
      chars_wrote += sizeof(extMemptr);
      break;
    }
    case GENERATE_ENUM(cuDestroyExternalSemaphore):
    {
      CUexternalSemaphore extSem = va_arg(arglist, CUexternalSemaphore);
      CUexternalSemaphore *extSemptr = &extSem;
      memcpy(buf + chars_wrote, &extSemptr, sizeof(extSemptr));
      chars_wrote += sizeof(extSemptr);
      break;
    }
    case GENERATE_ENUM(cuGraphCreate):
    {
      CUgraph * phGraph = va_arg(arglist, CUgraph *);
      memcpy(buf + chars_wrote, &phGraph, sizeof(phGraph));
      chars_wrote += sizeof(phGraph);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cuGraphDestroy):
    {
      CUgraph hGraph = va_arg(arglist, CUgraph );
      CUgraph *hGraphptr = &hGraph;
      memcpy(buf + chars_wrote, &hGraphptr, sizeof(hGraphptr));
      chars_wrote += sizeof(hGraphptr);
      break;
    }
    case GENERATE_ENUM(cuGraphDestroyNode):
    {
      CUgraphNode hNode = va_arg(arglist, CUgraphNode );
      CUgraphNode *hNodeptr = &hNode;
      memcpy(buf + chars_wrote, &hNodeptr, sizeof(hNodeptr));
      chars_wrote += sizeof(hNodeptr);
      break;
    }
    case GENERATE_ENUM(cuGraphExecDestroy):
    {
      CUgraphExec hGraphExec = va_arg(arglist, CUgraphExec);
      CUgraphExec *hGraphExecptr = &hGraphExec;
      memcpy(buf + chars_wrote, &hGraphExecptr, sizeof(hGraphExecptr));
      chars_wrote += sizeof(hGraphExecptr);
      break;
    }
    case GENERATE_ENUM(cuTexRefCreate):
    {
      CUtexref *pTexRef = va_arg(arglist, CUtexref *);
      memcpy(buf + chars_wrote, &pTexRef, sizeof(pTexRef));
      chars_wrote += sizeof(pTexRef);
      break;
    }
    case GENERATE_ENUM(cuTexRefDestroy):
    {
      CUtexref hTexRef = va_arg(arglist, CUtexref);
      CUtexref *hTexRefptr = &hTexRef;
      memcpy(buf + chars_wrote, &hTexRefptr, sizeof(hTexRefptr));
      chars_wrote += sizeof(hTexRefptr);
      break;
    }
    case GENERATE_ENUM(cuTexObjectCreate):
    {
      CUtexObject *pTexObject = va_arg(arglist, CUtexObject *);
      memcpy(buf + chars_wrote, &pTexObject, sizeof(pTexObject));
      chars_wrote += sizeof(pTexObject);

      CUDA_RESOURCE_DESC* pResDesc = va_arg(arglist, CUDA_RESOURCE_DESC *);
      memcpy(buf + chars_wrote, &pResDesc, sizeof(pResDesc));
      chars_wrote += sizeof(pResDesc);

      CUDA_TEXTURE_DESC* pTexDesc = va_arg(arglist, CUDA_TEXTURE_DESC *);
      memcpy(buf + chars_wrote, &pTexDesc, sizeof(pTexDesc));
      chars_wrote += sizeof(pTexDesc);

      CUDA_RESOURCE_VIEW_DESC* pResViewDesc = \
      va_arg(arglist, CUDA_RESOURCE_VIEW_DESC *);
      memcpy(buf + chars_wrote, &pResViewDesc, sizeof(pResViewDesc));
      chars_wrote += sizeof(pResViewDesc);
      break;
    }
    case GENERATE_ENUM(cuTexObjectDestroy):
    {
      CUtexObject pTexObject = va_arg(arglist, CUtexObject );
      CUtexObject *pTexObjectptr = &pTexObject;
      memcpy(buf + chars_wrote, &pTexObjectptr, sizeof(pTexObjectptr));
      chars_wrote += sizeof(pTexObjectptr);
      break;
    }
    case GENERATE_ENUM(cuSurfObjectCreate):
    {
      CUsurfObject *pSurfObject = va_arg(arglist, CUsurfObject *);
      memcpy(buf + chars_wrote, &pSurfObject, sizeof(pSurfObject));
      chars_wrote += sizeof(pSurfObject);

      CUDA_RESOURCE_DESC *pResDesc = va_arg(arglist, CUDA_RESOURCE_DESC *);
      memcpy(buf + chars_wrote, &pResDesc, sizeof(pResDesc));
      chars_wrote += sizeof(pResDesc);
      break;
    }
    case GENERATE_ENUM(cuSurfObjectDestroy):
    {
      CUsurfObject pSurfObject = va_arg(arglist, CUsurfObject);
      CUsurfObject *pSurfObjectptr = &pSurfObject;
      memcpy(buf + chars_wrote, &pSurfObjectptr, sizeof(pSurfObjectptr));
      chars_wrote += sizeof(pSurfObjectptr);
      break;
    }
    case GENERATE_ENUM(cublasCreate_v2):
    {
      cublasHandle_t *handle = va_arg(arglist, cublasHandle_t *);
      memcpy(buf + chars_wrote, handle, sizeof(cublasHandle_t));
      chars_wrote += sizeof(cublasHandle_t);
      break;
    }
    case GENERATE_ENUM(cublasDestroy_v2):
    {
      cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
      cublasHandle_t *handleptr = &handle;
      memcpy(buf + chars_wrote, handleptr, sizeof(cublasHandle_t));
      chars_wrote += sizeof(cublasHandle_t);
      break;
    }
    case GENERATE_ENUM(cublasSetStream_v2):
    {
      cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      cudaStream_t stream = va_arg(arglist, cudaStream_t);
      memcpy(buf + chars_wrote, &stream, sizeof(cudaStream_t));
      chars_wrote += sizeof(stream);
      break;
    }
    case GENERATE_ENUM(cublasAlloc):
    {
      int n = va_arg(arglist, int);
      // memcpy(buf + chars_wrote, &n, sizeof(n));
      // chars_wrote += sizeof(n);

      int elemSize = va_arg(arglist, int);
      // memcpy(buf + chars_wrote, &elemSize, sizeof(elemSize));
      // chars_wrote += sizeof(elemSize);

      void **devicePtr = va_arg(arglist, void **);
      // memcpy(buf + chars_wrote, devicePtr, sizeof (void *));
      // chars_wrote += sizeof (void *);

      // get the device id
      int deviceId = -1;
      cudaGetDevice(&deviceId);
      // update the map
      lhckpt_pages_t page = {CUMEM_ALLOC_PAGE, *devicePtr, (size_t)(n * elemSize), 0, deviceId, false};
      // printf("[lt] log_api, cudaMalloc key:%p\n", *pointer);
      lh_pages_map[*devicePtr] = page;
      return;
    }
    case GENERATE_ENUM(cublasFree):
    {
      void *devicePtr = va_arg(arglist, void *);
      // memcpy(buf + chars_wrote, devicePtr, sizeof (void *));
      // chars_wrote += sizeof (void *);

      // remove from map
      lh_pages_map.erase(devicePtr);
      return;
    }
    case GENERATE_ENUM(cufftPlan3d):
    {
      cufftHandle *plan = va_arg(arglist, cufftHandle *);
      memcpy(buf + chars_wrote, &plan, sizeof(plan));
      chars_wrote += sizeof(plan);

      int nx = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &nx, sizeof(nx));
      chars_wrote += sizeof(nx);

      int ny = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ny, sizeof(ny));
      chars_wrote += sizeof(ny);

      int nz = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &nz, sizeof(nz));
      chars_wrote += sizeof(nz);

      int type = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &type, sizeof(type));
      chars_wrote += sizeof(type);
      break;
    }
    case GENERATE_ENUM(cufftPlanMany):
    {
      cufftHandle *plan = va_arg(arglist, cufftHandle *);
      memcpy(buf + chars_wrote, &plan, sizeof(plan));
      chars_wrote += sizeof(plan);

      int rank = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &rank, sizeof(rank));
      chars_wrote += sizeof(rank);

      int *n = va_arg(arglist, int *);
      memcpy(buf + chars_wrote, &n, sizeof(n));
      chars_wrote += sizeof(n);

      int *inembed = va_arg(arglist, int *);
      memcpy(buf + chars_wrote, &inembed, sizeof(inembed));
      chars_wrote += sizeof(inembed);

      int istride = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &istride, sizeof(istride));
      chars_wrote += sizeof(istride);

      int idist = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &idist, sizeof(idist));
      chars_wrote += sizeof(idist);

      int *onembed = va_arg(arglist, int *);
      memcpy(buf + chars_wrote, &onembed, sizeof(onembed));
      chars_wrote += sizeof(onembed);

      int ostride = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ostride, sizeof(ostride));
      chars_wrote += sizeof(ostride);

      int odist = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &odist, sizeof(odist));
      chars_wrote += sizeof(odist);

      int type = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &type, sizeof(type));
      chars_wrote += sizeof(type);

      int batch = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &batch, sizeof(batch));
      chars_wrote += sizeof(batch);
      break;
    }
    case GENERATE_ENUM(cufftSetStream):
    {
      cufftHandle plan = va_arg(arglist, cufftHandle);
      memcpy(buf + chars_wrote, &plan, sizeof(plan));
      chars_wrote += sizeof(plan);

      cudaStream_t pStream = va_arg(arglist, cudaStream_t);
      // cudaStream_t *pStreamptr = &pStream;
      memcpy(buf + chars_wrote, &pStream, sizeof(pStream));
      chars_wrote += sizeof(pStream);
      break;
    }
    case GENERATE_ENUM(cusparseCreate):
    {
      cusparseHandle_t *handle = va_arg(arglist, cusparseHandle_t *);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusparseDestroy):
    {
      cusparseHandle_t handle = va_arg(arglist, cusparseHandle_t);
      cusparseHandle_t *handleptr = &handle;
      memcpy(buf + chars_wrote, &handleptr, sizeof(handleptr));
      chars_wrote += sizeof(handleptr);
      break;
    }
    case GENERATE_ENUM(cusparseCreateMatDescr):
    {
      cusparseMatDescr_t *handle = va_arg(arglist, cusparseMatDescr_t *);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusparseDestroyMatDescr):
    {
      cusparseMatDescr_t handle = va_arg(arglist, cusparseMatDescr_t );
      cusparseMatDescr_t* handleptr = &handle;
      memcpy(buf + chars_wrote, &handleptr, sizeof(handleptr));
      chars_wrote += sizeof(handleptr);
      break;
    }
    case GENERATE_ENUM(cusolverDnCreate):
    {
      cusolverDnHandle_t *handle = va_arg(arglist, cusolverDnHandle_t *);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusolverDnDestroy):
    {
      cusolverDnHandle_t handle = va_arg(arglist, cusolverDnHandle_t);
      cusolverDnHandle_t *handleptr = &handle;
      memcpy(buf + chars_wrote, &handleptr, sizeof(handleptr));
      chars_wrote += sizeof(handleptr);
      break;
    }
    case GENERATE_ENUM(cuDevicePrimaryCtxRelease_v2):
    {
      CUdevice dev = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &dev, sizeof(dev));
      chars_wrote += sizeof(dev);
      break;
    }
    case GENERATE_ENUM(cuDevicePrimaryCtxReset_v2):
    {
      CUdevice dev = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &dev, sizeof(dev));
      chars_wrote += sizeof(dev);
      break;
    }
    case GENERATE_ENUM(cuDevicePrimaryCtxRetain):
    {
      CUcontext *pctx = va_arg(arglist, CUcontext *);
      // CUcontext ** pctxptr = &pctx;
      memcpy(buf + chars_wrote, pctx, sizeof(CUcontext));
      chars_wrote += sizeof(CUcontext);

      CUdevice dev = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &dev, sizeof(dev));
      chars_wrote += sizeof(dev);
      break;
    }
    case GENERATE_ENUM(cuCtxCreate_v2):
    {
      CUcontext *pctx = va_arg(arglist, CUcontext *);
      // CUcontext ** pctxptr = &pctx;

      memcpy(buf + chars_wrote, pctx, sizeof(CUcontext));
      chars_wrote += sizeof(CUcontext);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);

      CUdevice dev = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &dev, sizeof(dev));
      chars_wrote += sizeof(dev);
      break;
    }
    case GENERATE_ENUM(cuCtxDestroy_v2):
    {
      CUcontext pctx = va_arg(arglist, CUcontext );
      memcpy(buf + chars_wrote, &pctx, sizeof(pctx));
      chars_wrote += sizeof(pctx);
      break;
    }
    case GENERATE_ENUM(cuCtxGetApiVersion):
    {
      CUcontext pctx = va_arg(arglist, CUcontext );
      memcpy(buf + chars_wrote, &pctx, sizeof(pctx));
      chars_wrote += sizeof(pctx);

      unsigned int *version = va_arg(arglist, unsigned int*);
      memcpy(buf + chars_wrote, version, sizeof(unsigned int));
      chars_wrote += sizeof(unsigned int);
      break;
    }
    case GENERATE_ENUM(cuCtxGetCacheConfig):
    {
      CUfunc_cache *config = va_arg(arglist, CUfunc_cache*);
      memcpy(buf + chars_wrote, &config, sizeof(CUfunc_cache));
      chars_wrote += sizeof(CUfunc_cache);
      break;
    }
    case GENERATE_ENUM(cuCtxSetCurrent):
    {
      CUcontext ctx = va_arg(arglist, CUcontext);
      memcpy(buf + chars_wrote, &ctx, sizeof(CUcontext));
      chars_wrote += sizeof(CUcontext);
      break;
    }
    case GENERATE_ENUM(cuCtxGetCurrent):
    {
      CUcontext* pctx = va_arg(arglist, CUcontext*);
      memcpy(buf + chars_wrote, &pctx, sizeof(CUcontext));
      chars_wrote += sizeof(CUcontext);
      break;
    }
    case GENERATE_ENUM(cuCtxGetDevice):
    {
      CUdevice* pdevice = va_arg(arglist, CUdevice*);
      memcpy(buf + chars_wrote, &pdevice, sizeof(CUdevice));
      chars_wrote += sizeof(CUdevice);
      break;
    }
    case GENERATE_ENUM(cuCtxSetLimit):
    {
      CUlimit limit = (CUlimit)va_arg(arglist, int);
      memcpy(buf + chars_wrote, &limit, sizeof(CUlimit));
      chars_wrote += sizeof(CUlimit);

      size_t value = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &value, sizeof(size_t));
      chars_wrote += sizeof(size_t);
      break;
    }
    case GENERATE_ENUM(cuCtxGetLimit):
    {
      size_t *pvalue = va_arg(arglist, size_t*);
      memcpy(buf + chars_wrote, &pvalue, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      CUlimit limit = (CUlimit)va_arg(arglist, int);
      memcpy(buf + chars_wrote, &limit, sizeof(CUlimit));
      chars_wrote += sizeof(CUlimit);
      break;
    }
    case GENERATE_ENUM(cuLinkCreate_v2):
    {
      unsigned int numOptions = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &numOptions, sizeof(numOptions));
      chars_wrote += sizeof(numOptions);

      CUjit_option * options = va_arg(arglist, CUjit_option *);
      memcpy(buf + chars_wrote, &options, sizeof(options));
      chars_wrote += sizeof(options);

      void** optionValues = va_arg(arglist, void**);
      memcpy(buf + chars_wrote, &optionValues, sizeof(optionValues));
      chars_wrote += sizeof(optionValues);
      // out parameter
      CUlinkState * stateOut = va_arg(arglist, CUlinkState *);
      memcpy(buf + chars_wrote, &stateOut, sizeof(stateOut));
      chars_wrote += sizeof(stateOut);
      break;
    }
    case GENERATE_ENUM(cuLinkDestroy):
    {
      CUlinkState  stateOut = va_arg(arglist, CUlinkState);
      CUlinkState  *stateOutptr = &stateOut;
      memcpy(buf + chars_wrote, &stateOutptr, sizeof(stateOutptr));
      chars_wrote += sizeof(stateOutptr);
      break;
    }
    case GENERATE_ENUM(cuArray3DCreate_v2):
    {
      CUarray* pHandle = va_arg(arglist, CUarray*);
      memcpy(buf + chars_wrote, &pHandle, sizeof(pHandle));
      chars_wrote += sizeof(pHandle);

      CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray = \
            va_arg(arglist, CUDA_ARRAY3D_DESCRIPTOR *);
      memcpy(buf + chars_wrote, &pAllocateArray, sizeof(pAllocateArray));
      chars_wrote += sizeof(pAllocateArray);
      break;
    }
    case GENERATE_ENUM(cuArrayCreate_v2):
    {
      CUarray* pHandle = va_arg(arglist, CUarray*);
      memcpy(buf + chars_wrote, &pHandle, sizeof(pHandle));
      chars_wrote += sizeof(pHandle);

      CUDA_ARRAY_DESCRIPTOR* pAllocateArray = \
            va_arg(arglist, CUDA_ARRAY_DESCRIPTOR *);
      memcpy(buf + chars_wrote, &pAllocateArray, sizeof(pAllocateArray));
      chars_wrote += sizeof(pAllocateArray);
      break;
    }
    case GENERATE_ENUM(cuArrayDestroy):
    {
      CUarray pHandle = va_arg(arglist, CUarray);
      CUarray *pHandleptr = &pHandle;
      memcpy(buf + chars_wrote, &pHandleptr, sizeof(pHandleptr));
      chars_wrote += sizeof(pHandleptr);
      break;
    }
    case GENERATE_ENUM(cuMipmappedArrayCreate):
    {
      CUmipmappedArray* pHandle = va_arg(arglist, CUmipmappedArray*);
      memcpy(buf + chars_wrote, &pHandle, sizeof(pHandle));
      chars_wrote += sizeof(pHandle);

      CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc = \
            va_arg(arglist, CUDA_ARRAY3D_DESCRIPTOR *);
      memcpy(buf + chars_wrote, &pMipmappedArrayDesc, \
      sizeof(pMipmappedArrayDesc));
      chars_wrote += sizeof(pMipmappedArrayDesc);

      unsigned int numMipmapLevels = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &numMipmapLevels, sizeof(numMipmapLevels));
      chars_wrote += sizeof(numMipmapLevels);
      break;
    }
    case GENERATE_ENUM(cuMipmappedArrayDestroy):
    {
      CUmipmappedArray pHandle = va_arg(arglist, CUmipmappedArray);
      CUmipmappedArray *pHandleptr = &pHandle;
      memcpy(buf + chars_wrote, &pHandleptr, sizeof(pHandleptr));
      chars_wrote += sizeof(pHandleptr);
      break;
    }
    case GENERATE_ENUM(cuInit):
    {
      unsigned int flag = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flag, sizeof(flag));
      chars_wrote += sizeof(flag);
      break;
    }

    case GENERATE_ENUM(cuDeviceGet):
    {
        CUdevice* device = va_arg(arglist, CUdevice* );
        memcpy(buf + chars_wrote, &device, sizeof(device));
        chars_wrote += sizeof(device);

        int ordinal = va_arg(arglist, int);
        memcpy(buf + chars_wrote, &ordinal, sizeof(ordinal));
        chars_wrote += sizeof(ordinal);
        break;
    }
    case GENERATE_ENUM(cudaGetDevice):
    {
        int* device = va_arg(arglist, int*);
        memcpy(buf + chars_wrote, &device, sizeof(device));
        chars_wrote += sizeof(device);
        break;
    }
    case GENERATE_ENUM(cudaSetDevice):
    {
        int device = va_arg(arglist, int);
        memcpy(buf + chars_wrote, &device, sizeof(device));
        chars_wrote += sizeof(device);
        break;
    }
    case GENERATE_ENUM(cuDeviceTotalMem_v2):
    {
      size_t* bytes = va_arg(arglist, size_t *);
      memcpy(buf + chars_wrote, bytes, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      CUdevice device = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &device, sizeof(device));
      chars_wrote += sizeof(device);
      break;
    }

    case GENERATE_ENUM(cuDeviceComputeCapability):
    {
      int* major = va_arg(arglist, int *);
      memcpy(buf + chars_wrote, major, sizeof(int));
      chars_wrote += sizeof(int);
	  int* minor = va_arg(arglist, int *);
      memcpy(buf + chars_wrote, minor, sizeof(int));
      chars_wrote += sizeof(int);

      CUdevice device = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &device, sizeof(device));
      chars_wrote += sizeof(device);
      break;
    }
    case GENERATE_ENUM(cuDeviceGetProperties):
    {
      CUdevprop *prop = va_arg(arglist, CUdevprop *);
      memcpy(buf + chars_wrote, prop, sizeof(CUdevprop));
      chars_wrote += sizeof(CUdevprop);

      CUdevice device = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &device, sizeof(device));
      chars_wrote += sizeof(device);
      break;
    }
    case GENERATE_ENUM(cuDevicePrimaryCtxGetState):
    {
      CUdevice device = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &device, sizeof(device));
      chars_wrote += sizeof(device);

      unsigned int *flags = va_arg(arglist, unsigned int *);
      memcpy(buf + chars_wrote, flags, sizeof(unsigned int));
      chars_wrote += sizeof(unsigned int);

      int *active = va_arg(arglist, int *);
      memcpy(buf + chars_wrote, active, sizeof(int));
      chars_wrote += sizeof(int);
      break;
    }
    case GENERATE_ENUM(cuDeviceGetName):
    {
      char *name = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, name, sizeof(char));
      chars_wrote += sizeof(char);

      int len = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &len, sizeof(len));
      chars_wrote += sizeof(len);

      CUdevice device = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &device, sizeof(device));
      chars_wrote += sizeof(device);
      break;
    }
    case GENERATE_ENUM(cuDeviceGetUuid):
    {
        CUuuid *uuid = va_arg(arglist, CUuuid *);
        memcpy(buf + chars_wrote, uuid, sizeof(CUuuid));
        chars_wrote += sizeof(CUuuid);

        CUdevice device = va_arg(arglist, CUdevice);
        memcpy(buf + chars_wrote, &device, sizeof(device));
        chars_wrote += sizeof(device);
        break;
    }
    case GENERATE_ENUM(cuEventElapsedTime ):
    {

      float *pMilliseconds = va_arg(arglist, float*);
      memcpy(buf + chars_wrote, &pMilliseconds, sizeof(pMilliseconds));
      chars_wrote += sizeof(pMilliseconds);

      CUevent hsevent = va_arg(arglist, CUevent);
      memcpy(buf + chars_wrote, &hsevent, sizeof(hsevent));
      chars_wrote += sizeof(hsevent);

      CUevent heevent = va_arg(arglist, CUevent);
      memcpy(buf + chars_wrote, &heevent, sizeof(heevent));
      chars_wrote += sizeof(heevent);
      break;
    }

    case GENERATE_ENUM(cuEventSynchronize):
    {
      CUevent hsevent = va_arg(arglist, CUevent);
      memcpy(buf + chars_wrote, &hsevent, sizeof(hsevent));
      chars_wrote += sizeof(hsevent);
      break;
    }

    case GENERATE_ENUM(cuModuleLoadData):
	  {
      CUmodule *module = va_arg(arglist, CUmodule*);
      CUmodule modobj = *module;
      memcpy(buf + chars_wrote, &modobj, sizeof(modobj));
      chars_wrote += sizeof(modobj);

      void *image = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &image, sizeof(image));
      chars_wrote += sizeof(image);

      break;
    }

    case GENERATE_ENUM(cuModuleLoadDataEx ):
    {
      CUmodule *module = va_arg(arglist, CUmodule*);
      CUmodule modobj = *module;
      memcpy(buf + chars_wrote, &modobj, sizeof(modobj));
      chars_wrote += sizeof(modobj);

      void *image = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &image, sizeof(image));
      chars_wrote += sizeof(image);

      // printf("the image name is %s\n", (char*)image);

      unsigned int numOptions = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &numOptions, sizeof(numOptions));
      chars_wrote += sizeof(numOptions);

      CUjit_option *options = va_arg(arglist, CUjit_option*);
      memcpy(buf + chars_wrote, &options, sizeof(options));
      chars_wrote += sizeof(options);

      void** optionValues = va_arg(arglist, void**);
      memcpy(buf + chars_wrote, &optionValues, sizeof(optionValues));
      chars_wrote += sizeof(optionValues);

      break;
    }

    case GENERATE_ENUM(cuModuleGetFunction ):
    {
      CUfunction *hfunc = va_arg(arglist, CUfunction*);
      // memcpy(buf + chars_wrote, &hfunc, sizeof(hfunc));
      // chars_wrote += sizeof(hfunc);

        CUfunction func = *hfunc;
        memcpy(buf + chars_wrote, &func, sizeof(func));
        chars_wrote += sizeof(func);

      CUmodule hmod = va_arg(arglist, CUmodule);
      memcpy(buf + chars_wrote, &hmod, sizeof(hmod));
      chars_wrote += sizeof(hmod);

      char* name = va_arg(arglist, char *);
      int len = strlen(name);
      // fengtao.xie added

      memcpy(buf + chars_wrote, &len, sizeof(len));
      chars_wrote += sizeof(len);

      memcpy(buf + chars_wrote, name, len);
      chars_wrote += len;

      // printf("logapi func %p, hmod %p, name %s\n", hfunc, hmod, name);
      break;
    }
    case GENERATE_ENUM(cuMemcpyHtoD_v2):
    {
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr);
        memcpy(buf + chars_wrote, &dstDevice, sizeof(dstDevice));
        chars_wrote += sizeof(dstDevice);

        // memcpy(buf + chars_wrote, &srcHost, sizeof(srcHost));
        // chars_wrote += sizeof(srcHost);

        size_t ByteCount = va_arg(arglist, size_t);
        memcpy(buf + chars_wrote, &ByteCount, sizeof(ByteCount));
        chars_wrote += sizeof(ByteCount);

        // Copy srcHostData to buffer
        const void* srcHost = va_arg(arglist, void*);
        memcpy(buf + chars_wrote, srcHost, sizeof(char) * ByteCount);
		    chars_wrote += sizeof(char) * ByteCount;
        break;
    }
    case GENERATE_ENUM(cuMemcpyHtoDAsync_v2):
    {
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr);
        memcpy(buf + chars_wrote, &dstDevice, sizeof(dstDevice));
        chars_wrote += sizeof(dstDevice);

        // memcpy(buf + chars_wrote, &srcHost, sizeof(srcHost));
        // chars_wrote += sizeof(srcHost);

        size_t ByteCount = va_arg(arglist, size_t);
        memcpy(buf + chars_wrote, &ByteCount, sizeof(ByteCount));
        chars_wrote += sizeof(ByteCount);

        // Copy srcHostData to buffer
        const void* srcHost = va_arg(arglist, const void*);
        memcpy(buf + chars_wrote, srcHost, sizeof(char) * ByteCount);
        chars_wrote += sizeof(char) * ByteCount;

        CUstream hStream = va_arg(arglist, CUstream);
        memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
        chars_wrote += sizeof(hStream);
        break;
    }
    case GENERATE_ENUM(cuMemcpyDtoH_v2):
    {
        void* dstHost = va_arg(arglist, void*);
        memcpy(buf + chars_wrote, &dstHost, sizeof(dstHost));
        chars_wrote += sizeof(dstHost);

        CUdeviceptr srcDevice = va_arg(arglist, CUdeviceptr);
        memcpy(buf + chars_wrote, &srcDevice, sizeof(srcDevice));
        chars_wrote += sizeof(srcDevice);

        size_t ByteCount = va_arg(arglist, size_t);
        memcpy(buf + chars_wrote, &ByteCount, sizeof(ByteCount));
        chars_wrote += sizeof(ByteCount);
        break;
    }
    case GENERATE_ENUM(cuMemcpyDtoHAsync_v2):
    {
        void* dstHost = va_arg(arglist, void*);
        memcpy(buf + chars_wrote, &dstHost, sizeof(dstHost));
        chars_wrote += sizeof(dstHost);

        CUdeviceptr srcDevice = va_arg(arglist, CUdeviceptr);
        memcpy(buf + chars_wrote, &srcDevice, sizeof(srcDevice));
        chars_wrote += sizeof(srcDevice);

        size_t ByteCount = va_arg(arglist, size_t);
        memcpy(buf + chars_wrote, &ByteCount, sizeof(ByteCount));
        chars_wrote += sizeof(ByteCount);

        CUstream hStream = va_arg(arglist, CUstream);
        memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
        chars_wrote += sizeof(hStream);
        break;
    }
    case GENERATE_ENUM(cuEventRecord):
    {
     // printf("fengtao.xie in GENERATE_ENUM(cuEventRecord)\n");
      CUevent event = va_arg(arglist, CUevent);
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);

      CUstream stream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &stream, sizeof(stream));
      chars_wrote += sizeof(stream);
      break;
    }
    case GENERATE_ENUM(cuModuleGetTexRef):
    {
      // Store CUtexref to buf
      CUtexref* pTexRef = va_arg(arglist, CUtexref*);
      memcpy(buf + chars_wrote, pTexRef, sizeof(CUtexref));
      chars_wrote += sizeof(CUtexref);

      CUmodule hmod = va_arg(arglist, CUmodule);
      memcpy(buf + chars_wrote, &hmod, sizeof(hmod));
      chars_wrote += sizeof(hmod);

      const char* name = va_arg(arglist, const char*);
      int len = strlen(name);

      memcpy(buf + chars_wrote, &len, sizeof(len));
      chars_wrote += sizeof(len);

      memcpy(buf + chars_wrote, name, sizeof(char*) * len);
      chars_wrote += len;
      break;
    }
    case GENERATE_ENUM(cuTexRefSetAddress_v2):
    {
      size_t* ByteOffset = va_arg(arglist, size_t*);
      memcpy(buf + chars_wrote, &ByteOffset, sizeof(ByteOffset));
      chars_wrote += sizeof(CUtexref);

      CUtexref hTexRef = va_arg(arglist, CUtexref);
      memcpy(buf + chars_wrote, &hTexRef, sizeof(hTexRef));
      chars_wrote += sizeof(hTexRef);

      CUdeviceptr dptr = va_arg(arglist, CUdeviceptr);
      memcpy(buf + chars_wrote, &dptr, sizeof(dptr));
      chars_wrote += sizeof(dptr);

      size_t bytes = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &bytes, sizeof(bytes));
      chars_wrote += sizeof(bytes);
      break;
    }
    case GENERATE_ENUM(cuTexRefSetFormat):
    {
      CUtexref hTexRef = va_arg(arglist, CUtexref);
      memcpy(buf + chars_wrote, &hTexRef, sizeof(hTexRef));
      chars_wrote += sizeof(hTexRef);

      // The type of CUarray_format is enum
      int fmt = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &fmt, sizeof(fmt));
      chars_wrote += sizeof(fmt);

      int NumPackedComponents = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &NumPackedComponents, sizeof(NumPackedComponents));
      chars_wrote += sizeof(NumPackedComponents);
      break;
    }
    case GENERATE_ENUM(cuCtxSynchronize):
    {
      // Not store any parameters
      break;
    }
    case GENERATE_ENUM(cuStreamSynchronize):
    {
      // args
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);
      break;
    }
    case GENERATE_ENUM(cuStreamWaitEvent):
    {
      // args
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);

      CUevent hEvent = va_arg(arglist, CUevent);
      memcpy(buf + chars_wrote, &hEvent, sizeof(hEvent));
      chars_wrote += sizeof(hEvent);

      unsigned int flag = va_arg(arglist,   unsigned int);
      memcpy(buf + chars_wrote, &flag, sizeof(flag));
      chars_wrote += sizeof(flag);
      break;
    }
    /*case GENERATE_ENUM(cuStreamAddCallback):
    {
      // args
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);

      CUstreamCallback callback = va_arg(arglist, CUstreamCallback);
      memcpy(buf + chars_wrote, &callback, sizeof(callback));
      chars_wrote += sizeof(callback);

      void* userData = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, userData, sizeof(void));
      chars_wrote += sizeof(void);

      unsigned int flags = va_arg(arglist,   unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }*/
    case GENERATE_ENUM(cuMemsetD8Async):
    {
      CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr);
      memcpy(buf + chars_wrote, &dstDevice, sizeof(dstDevice));
      chars_wrote += sizeof(dstDevice);

      // Compiling error: ‘unsigned char’ is promoted to ‘int’ when passed through ‘...’
      int uc = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &uc, sizeof(uc));
      chars_wrote += sizeof(uc);

      size_t N = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &N, sizeof(N));
      chars_wrote += sizeof(N);

      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);
      break;
    }
    case GENERATE_ENUM(cuPointerGetAttributes):
    {
      // args
      unsigned int numAttributes = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &numAttributes, sizeof(numAttributes));
      chars_wrote += sizeof(numAttributes);

      CUpointer_attribute* attributes = va_arg(arglist, CUpointer_attribute*);
      for (unsigned int i = 0; i < numAttributes; i++) {
          memcpy(buf + chars_wrote, &attributes[i], sizeof(attributes[i]));
          chars_wrote += sizeof(attributes[i]);
      }
      void** data = va_arg(arglist, void**);
      for (unsigned int i = 0; i < numAttributes; i++) {
          if (attributes[i] == CU_POINTER_ATTRIBUTE_CONTEXT) {
              CUcontext* dataPtr = (CUcontext*)data[i];
              CUcontext dataValue = *dataPtr;
              memcpy(buf + chars_wrote, &dataValue, sizeof(dataValue));
              chars_wrote += sizeof(dataValue);
              break;
          }
      }
      CUdeviceptr ptr = va_arg(arglist, CUdeviceptr);
      memcpy(buf + chars_wrote, &ptr, sizeof(ptr));
      chars_wrote += sizeof(ptr);
      break;
    }
    case GENERATE_ENUM(ncclCommInitRank):
    {
      ncclComm_t* comm = va_arg(arglist, ncclComm_t*);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t*));
      chars_wrote += sizeof(ncclComm_t*);

      int nranks = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &nranks, sizeof(int));
      chars_wrote += sizeof(int);

      // TODO: store the address of ncclUnqiueId variant, for in the migration situation, the ncclUnqiueId will changed
      ncclUniqueId commId = va_arg(arglist, ncclUniqueId);
      memcpy(buf + chars_wrote, &commId, sizeof(ncclUniqueId));
      chars_wrote += sizeof(ncclUniqueId);
      printf("ncclCommInitRank, commId:0x%llx\n", (unsigned long long)hashUniqueId(commId));
      fflush(stdout);

      int rank = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &rank, sizeof(int));
      chars_wrote += sizeof(int);

      // store the original ncclComm
      memcpy(buf+chars_wrote, comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);
      break;
    }
    case GENERATE_ENUM(ncclCommInitAll):
    {
      ncclComm_t* comm = va_arg(arglist, ncclComm_t*);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t*));
      chars_wrote += sizeof(ncclComm_t*);

      int ndev = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ndev, sizeof(int));
      chars_wrote += sizeof(int);

      const int* devlist = va_arg(arglist, const int*);
      memcpy(buf + chars_wrote, &devlist, sizeof(int*));
      chars_wrote += sizeof(int*);

      // store the ncclComm_t, used to build
      for (int i = 0; i < ndev; i++)
      {
        memcpy(buf + chars_wrote, comm + i, sizeof(ncclComm_t));
        chars_wrote += sizeof(ncclComm_t);
      }

      break;
    }
    // case GENERATE_ENUM(ncclCommFinalize):
    // {
    //   ncclComm_t* comm = va_arg(arglist, ncclComm_t*);
    //   memcpy(buf + chars_wrote, comm, sizeof(ncclComm_t));
    //   chars_wrote += sizeof(ncclComm_t);
    //   break;
    // }
    case GENERATE_ENUM(ncclCommDestroy):
    {
      ncclComm_t* comm = va_arg(arglist, ncclComm_t*);
      memcpy(buf + chars_wrote, comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);
      break;
    }
    case GENERATE_ENUM(ncclCommAbort):
    {
      ncclComm_t* comm = va_arg(arglist, ncclComm_t*);
      memcpy(buf + chars_wrote, comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);
      break;
    }
    case GENERATE_ENUM(ncclGetErrorString):
    {
      // TODO: Not Log for temperary
      break;
    }
    // case GENERATE_ENUM(ncclGetLastError):
    // {
    //   // TODO: Not Log for temperary
    //   break;
    // }
    case GENERATE_ENUM(ncclCommGetAsyncError):
    {
      // TODO:Not Log for temperary
      break;
    }
    case GENERATE_ENUM(ncclCommCount):
    {
      // TODO: Not Log for temperary
      break;
    }
    case GENERATE_ENUM(ncclCommCuDevice):
    {
      // TODO: Not Log for temperary
      break;
    }
    case GENERATE_ENUM(ncclCommUserRank):
    {
      // TODO: Not Log for temperary
      break;
    }
    // case GENERATE_ENUM(ncclRedOpCreatePreMulSum):
    // {
    //   // TODO: Not Log for temperary
    //   break;
    // }
    // case GENERATE_ENUM(ncclRedOpDestroy):
    // {
    //   // TODO: Not Log for temperary
    //   break;
    // }
    case GENERATE_ENUM(ncclReduce):
    {
      // TODO: Not Log for temperary
      const void* sendbuff = va_arg(arglist, const void*);
      memcpy(buf + chars_wrote, &sendbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      void* recvbuff = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &recvbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      size_t count = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &count, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      int datatype = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &datatype, sizeof(int));
      chars_wrote += sizeof(int);

      int op = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &op, sizeof(int));
      chars_wrote += sizeof(int);

      int root = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &root, sizeof(int));
      chars_wrote += sizeof(int);

      // TODO: it is a pointer in actual, so we only need to store its value 
      ncclComm_t comm = va_arg(arglist, ncclComm_t);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);

      cudaStream_t stream = va_arg(arglist, cudaStream_t); 
      memcpy(buf + chars_wrote, &stream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);
      break;
    }
    case GENERATE_ENUM(ncclBcast):
    {
      // TODO: Not Log for temperary
      void* buff = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &buff, sizeof(void*));
      chars_wrote += sizeof(void*);

      size_t count = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &count, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      int datatype = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &datatype, sizeof(int));
      chars_wrote += sizeof(int);

      int root = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &root, sizeof(int));
      chars_wrote += sizeof(int);

      // TODO: it is a pointer in actual, so we only need to store its value 
      ncclComm_t comm = va_arg(arglist, ncclComm_t);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);

      cudaStream_t stream = va_arg(arglist, cudaStream_t); 
      memcpy(buf + chars_wrote, &stream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);
      break;
    }
    case GENERATE_ENUM(ncclBroadcast):
    {
      // TODO: Not Log for temperary
      const void* sendbuff = va_arg(arglist, const void*);
      memcpy(buf + chars_wrote, &sendbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      void* recvbuff = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &recvbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      size_t count = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &count, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      int datatype = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &datatype, sizeof(int));
      chars_wrote += sizeof(int);

      int root = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &root, sizeof(int));
      chars_wrote += sizeof(int);

      // TODO: it is a pointer in actual, so we only need to store its value 
      ncclComm_t comm = va_arg(arglist, ncclComm_t);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);

      cudaStream_t stream = va_arg(arglist, cudaStream_t); 
      memcpy(buf + chars_wrote, &stream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);
      break;
    }
    case GENERATE_ENUM(ncclAllReduce):
    {
      const void* sendbuff = va_arg(arglist, const void*);
      memcpy(buf + chars_wrote, &sendbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      void* recvbuff = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &recvbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      size_t count = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &count, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      int datatype = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &datatype, sizeof(int));
      chars_wrote += sizeof(int);

      int op = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &op, sizeof(int));
      chars_wrote += sizeof(int);

      // TODO: it is a pointer in actual, so we only need to store its value 
      ncclComm_t comm = va_arg(arglist, ncclComm_t);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);

      cudaStream_t stream = va_arg(arglist, cudaStream_t); 
      memcpy(buf + chars_wrote, &stream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);
      break;
    }
    case GENERATE_ENUM(ncclReduceScatter):
    {
      const void* sendbuff = va_arg(arglist, const void*);
      memcpy(buf + chars_wrote, &sendbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      void* recvbuff = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &recvbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      size_t recvcount = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &recvcount, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      int datatype = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &datatype, sizeof(int));
      chars_wrote += sizeof(int);

      int op = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &op, sizeof(int));
      chars_wrote += sizeof(int);

      // TODO: it is a pointer in actual, so we only need to store its value 
      ncclComm_t comm = va_arg(arglist, ncclComm_t);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);

      cudaStream_t stream = va_arg(arglist, cudaStream_t); 
      memcpy(buf + chars_wrote, &stream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);
      break;
    }
    case GENERATE_ENUM(ncclAllGather):
    {
      const void* sendbuff = va_arg(arglist, const void*);
      memcpy(buf + chars_wrote, &sendbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      void* recvbuff = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &recvbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      size_t count = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &count, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      int datatype = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &datatype, sizeof(int));
      chars_wrote += sizeof(int);

      int op = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &op, sizeof(int));
      chars_wrote += sizeof(int);

      // TODO: it is a pointer in actual, so we only need to store its value 
      ncclComm_t comm = va_arg(arglist, ncclComm_t);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);

      cudaStream_t stream = va_arg(arglist, cudaStream_t); 
      memcpy(buf + chars_wrote, &stream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);
      break;
    }
    case GENERATE_ENUM(ncclSend):
    {
      const void* sendbuff = va_arg(arglist, const void*);
      memcpy(buf + chars_wrote, &sendbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      size_t count = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &count, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      int datatype = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &datatype, sizeof(int));
      chars_wrote += sizeof(int);

      int peer = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &peer, sizeof(int));
      chars_wrote += sizeof(int);

      // TODO: it is a pointer in actual, so we only need to store its value 
      ncclComm_t comm = va_arg(arglist, ncclComm_t);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);

      cudaStream_t stream = va_arg(arglist, cudaStream_t); 
      memcpy(buf + chars_wrote, &stream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);
      break;
    }
    case GENERATE_ENUM(ncclRecv):
    {
      void* recvbuff = va_arg(arglist, void*);
      memcpy(buf + chars_wrote, &recvbuff, sizeof(void*));
      chars_wrote += sizeof(void*);

      size_t count = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &count, sizeof(size_t));
      chars_wrote += sizeof(size_t);

      int datatype = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &datatype, sizeof(int));
      chars_wrote += sizeof(int);

      int peer = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &peer, sizeof(int));
      chars_wrote += sizeof(int);

      // TODO: it is a pointer in actual, so we only need to store its value 
      ncclComm_t comm = va_arg(arglist, ncclComm_t);
      memcpy(buf + chars_wrote, &comm, sizeof(ncclComm_t));
      chars_wrote += sizeof(ncclComm_t);

      cudaStream_t stream = va_arg(arglist, cudaStream_t); 
      memcpy(buf + chars_wrote, &stream, sizeof(cudaStream_t));
      chars_wrote += sizeof(cudaStream_t);
      break;
    }
    case GENERATE_ENUM(ncclGroupStart):
    {
      // TODO: No paramter need to store
      break;
    }
    case GENERATE_ENUM(ncclGroupEnd):
    {
      // TODO: if ncclGroupEnd is called, then remove the previous nccl api logs until ncclGroupStart
      //       while the ncclGroupEnd does not push to the log vector
      stack<CudaCallLog_t> tmpLogStack;
      while (!tmpLogStack.empty())
        tmpLogStack.pop();

      // TODO: check the logs from end to begin until meet the first ncclGroupStart
      //       if nccl api, delete from the vector
      //       if cuda api, then push it to the stack and later re-push to the vector
      //       For the scenarios with multiple threads, may be we need a mutex to protect the vector, implement later

      bool needRmv = true;
      while (!cudaCallsLog.empty())
      {
        CudaCallLog_t lastlog = cudaCallsLog.back(); 
        
        Cuda_Fncs_t op = *((Cuda_Fncs_t*)(lastlog.fncargs));
        bool finish = false;

        if (tid == lastlog.thread_id) // nccl api with same thread id
        {
          if (op == Cuda_Fnc_ncclGroupStart) {
            finish = true;
          }

          if (op == Cuda_Fnc_ncclCommInitAll || op == Cuda_Fnc_ncclCommInitRank) {
            needRmv = false; 
          }
        }
        
        tmpLogStack.push(lastlog);

        cudaCallsLog.pop_back();
        
        if (finish || !needRmv)
        {
          // printf("find the ncclGroupStart or do not need remove operation in group..\n");
          break;
        }
      }

      // printf("now ready to refill other api to vector...\n");
      while (!tmpLogStack.empty())
      {
        CudaCallLog_t logTmp = tmpLogStack.top();
        tmpLogStack.pop();
        if (tid != logTmp.thread_id)
          cudaCallsLog.push_back(logTmp);
        else
        {
          if (needRmv)
            JALLOC_FREE(logTmp.fncargs);
          else
            cudaCallsLog.push_back(logTmp);
        }
      }
      
      if (needRmv)
        return; // ncclGroupEnd is not need to store, so return.
      
      break;
    }
    case GENERATE_ENUM(ncclGetUniqueId):
    {
      ncclUniqueId* uniqueId = va_arg(arglist, ncclUniqueId*);
      memcpy(buf + chars_wrote, uniqueId, sizeof(ncclUniqueId));
      chars_wrote += sizeof(ncclUniqueId);

      // printf("ncclGetUniqueId in log api: 0x%llx\n", (unsigned long long)hashUniqueId(*uniqueId));
      // fflush(stdout);
      break;
    }
    case GENERATE_ENUM(ncclGetVersion):
    {
      break;
    }
    default:
    {
      JNOTE("log API not implemented") (cuda_fnc_op);
      break;
    }
  }
  // common for every API
  log.fncargs = (char *)JALLOC_MALLOC(chars_wrote);
  memcpy(log.fncargs, buf, chars_wrote);
  log.size = chars_wrote;

  // push_back fails/segfaults when a lot of cuda Calls are made
  // To avoid the segfault we can resize cudaCallsLog
  // However this will be destructive at restart;
  // lets use reserve for Now...
  // cudaCallsLog.resize(log.size);
  pthread_mutex_lock(&mutex_for_log);
  cudaCallsLog.reserve(log.size);
  cudaCallsLog.push_back(log);
  pthread_mutex_unlock(&mutex_for_log);
  // printf("insert log item successfully,func name:%s, cudaCallsLog.size:%ld, tid:%ld, vect_ptr:%p\n", cuda_Fnc_to_str[cuda_fnc_op], cudaCallsLog.size(), tid, &cudaCallsLog);
  // fflush(stdout);
  va_end(arglist);
}

// support timing by huiru.deng
void sendMsgToDmtcp(int i)
{
  JNOTE("send msg to dmtcp.") (i);
  doCheckpoint(i);
}
