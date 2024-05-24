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

#ifndef COMMON_H
#define COMMON_H

#include <link.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <asm/prctl.h>
#include <linux/limits.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <cufft.h>
#include <cublasLt.h>
#include "cusolverDn.h"
#include "lower_half_cuda_if.h"
#include "nccl.h"
typedef char* VA;  /* VA = virtual address */

// Based on the entries in /proc/<pid>/stat as described in `man 5 proc`
enum Procstat_t
{
  PID = 1,
  COMM,   // 2
  STATE,  // 3
  PPID,   // 4
  NUM_THREADS = 19,
  STARTSTACK = 27,
};

#define PAGE_SIZE 0x1000LL

// FIXME: 0x1000 is one page; Use sysconf(PAGESIZE) instead.
#define ROUND_DOWN(x) ((unsigned long long)(x) \
                      & ~(unsigned long long)(PAGE_SIZE - 1))
#define ROUND_UP(x)  (((unsigned long long)(x) + PAGE_SIZE - 1) & \
                      ~(PAGE_SIZE - 1))
#define PAGE_OFFSET(x)  ((x) & (PAGE_SIZE - 1))

// TODO: This is very x86-64 specific; support other architectures??
#define eax rax
#define ebx rbx
#define ecx rcx
#define edx rax
#define ebp rbp
#define esi rsi
#define edi rdi
#define esp rsp
#define CLEAN_FOR_64_BIT_HELPER(args ...) # args
#define CLEAN_FOR_64_BIT(args ...)        CLEAN_FOR_64_BIT_HELPER(args)

typedef struct __LowerHalfInfo
{
  void *lhSbrk;
  void *lhMmap;
  void *lhMunmap;

  void *lhRealloc; // add by tian01.liu 2023.1.19

  void *lhDlsym;
  unsigned long lhFsAddr;
  void *lhMmapListFptr;
  void *uhEndofHeapFptr;
  void *lhGetDeviceHeapFptr;
  void *lhCopyToCudaPtrFptr;
  void *lhDeviceHeap;
  void *getFatCubinHandle;
  // biao.xing added
  void *lhCUtypeFptr;
  void *lhFreeCUtypeFptr;

  void *lhCublasTypeFptr;
  void *lhFreeCublasTypeFptr;
  // add by tian01.liu 2022.7.5
  void *lhCompressFptr;
  void *lhCompressMSFptr;
  // add by tian01.liu 2022.9.17
  void *lhDecompressFptr;
  // add by yueyang.li 2023.3.1
  void *lhIncreamentalCkpt;
  void *lhComputeHash;
  // test func ptr
  void *lhGetTimeRestore;

  void *lhCUDAtypeFptr;
  void *lhFreeCUDATypeFptr;
  void *readUhInfoAddrHandle;
  void* logs_read_and_apply_handle;
  void* copy_lower_half_data_handle;
  void *refill_handles;
  // added by tian01.liu, 2023.1.9
  void* realloc_gpu_blocks_handle;
  void* lhGetMMUProfileFptr;
  void* lhMMUAllocFptr;
  void* lhMMUFreeFptr;
  void* lhReleaseMMUFptr;
  void* lhReInitMMUFptr;

  // added by tian01.liu for fixed the bug of cuda
  void* lhPLMPGetFptr;
  void* lhAllocHostFptr;
  void* lhFreeHostFptr;
  void* lhRegisterHostFptr;
  void* lhInitPageLockPoolFptr;

  void* lhExecApiFptr;
  void* lhGetMemCtxFptr;
  void* lhGetIpcInfoFptr; // by tian01.liu for cuda ipc

  // add by tian01.liu 2023.9.5
  void* lhNcclTypeFptr;
  void* lhFreeNcclTypeFptr;

  void* lhIpcGetApiEx;   // used to replace cudaIpcGetMemHandle in main thread
  void* lhIpcOpenApiEx;  // used to replace cudaIpcOpenMemHandle in main thread
  void* lhIpcCloseApiEx; // used to replace cudaIpcCloseMemHandle in main thread
 
  // add by tian01.liu for store application stack segment addr
  void *lhGetAppStackSegFptr;
  // add by tian01.liu 2023.10.28
  void* lhNewThreadFptr;
  void* lhGetFsFptr;
  void* lhWakeFptr;
  void* lhWaitFnhFptr;

  // lu77.wei added
  bool isReplay;
} LowerHalfInfo_t;

typedef struct __UpperHalfInfo
{
  void *uhEndofHeap;
  void *lhPagesRegion;
  void *cudaLogVectorFptr;
  bool is_gpu_compress;

  // add by tian01.liu for store application stack segment addr
  void *appStackAddr;

  // by tian01.liu, save the pointer of lhPagesMap and used to realloc gpu blocks in restore workflow, 2023.1.9
  void *cudaBlockMapFptr;
  void*                   mmu_reserve_big_addr;
  size_t                  mmu_reserve_big_size;
  void*                   mmu_reserve_small_addr;
  size_t                  mmu_reserve_small_size;
  size_t                  mmu_reserve_allign;
  unsigned long long      mmu_reserve_flags;

  // by tian01.liu fix the bug of page-lock memory
  void* lhPageLockRegion;
  void* uhPageLockMapFptr;
  void* lhPageLockPoolAddr;
  size_t lhPageLockSize;
} UpperHalfInfo_t;

typedef struct __MmapInfo
{
  void *addr;
  size_t len;
} MmapInfo_t;

typedef struct __CudaCallLog {
  char *fncargs;
  size_t size;
  char *results;
  size_t res_size;
  pid_t thread_id;
} CudaCallLog_t;

extern LowerHalfInfo_t lhInfo;
extern UpperHalfInfo_t uhInfo;

#ifdef __cplusplus
extern "C" {
#endif
void* lhDlsym(Cuda_Fncs_t type);
void** fatHandle();

// biao.xing@samsung.com added
void *getCUtype(void* cuType, int type);
void freeCUtype(void* cuType, int type);
void *getCublasType(void* cublasType, int type);
void freeCublasType(void* cublasType, int type);

void *getCudaType(void* cudaType, int type);
void freeCudaType(void* cudaType, int type);

// add by tian01.liu 2023.9.5
void *getNickleType(void *ncclType, int type);
void freeNickleType(void *ncclType, int type);

#ifdef __cplusplus
}
#endif
typedef void* (*LhDlsym_t)(Cuda_Fncs_t type);
// getter function returning new_fatCubinHandle
// from the replay code
typedef void** (*fatHandle_t)();

// add by tian01.liu for upper-half get the application stack
typedef void* (*lhGetAppStack)();

// global_fatCubinHandle defined in cuda-plugin.cpp
// extern void ** global_fatCubinHandle;

// biao.xing@samsung.com added
typedef enum {
  CU_DEV_PTR,
  CU_CONTEXT,
  CU_MODULE,
  CU_FUNCTION,
  CU_EVENT,
  CU_STREAM,
  CU_TEXREF,
  CU_MAX
}CU_TYPE;

typedef enum {
  CUDA_STREAM,
  CUDA_MEMORY,
  CUDA_EVENT,
  CUDA_MAX
}CUDA_TYPE;

// add by weilu for cublas handle
typedef enum {
  CUBLAS_HANDLE,
  CUBLAS_MAX
}CUBLAS_TYPE;

// add by tian01.liu for ncclComm
typedef enum {
  NCCL_COMM,
  NCCL_MAX
}NCCL_TYPE;

// enum for types
typedef enum pages_t {
  CUDA_MALLOC_PAGE = 0,
  CUDA_UVM_PAGE,
  CUDA_HOST_ALLOC_PAGE,
  CUDA_HEAP_PAGE,
  CUMEM_ALLOC_PAGE
}PAGE_TYPE;

typedef struct Lhckpt_pages_t {
  PAGE_TYPE mem_type;
  void * mem_addr;
  size_t mem_len;
  size_t comp_len; // by tian01.liu 2022.7.7 used for restart

  int devId; // by tian01.liu 2023.1.9, for new gpu allocate architecture
  bool isIpcMem; // by tian01.liu 2o23.8.15, for cuda ipc
}lhckpt_pages_t;

// add by tian01.liu for pin memory
typedef struct page_lock_info_t
{
  unsigned int flags;
  size_t size;
}st_page_lock_info;

/* 
 * added by tian01.liu 2023.1.10, for new gpu memory management architecture
 * Function: used to store the mmu initial info, then rebuild mmu initilize status
*/
typedef struct Lhmmu_profile_t
{
  void*                   mmu_reserve_big_addr;
  size_t                  mmu_reserve_big_size;
  void*                   mmu_reserve_small_addr;
  size_t                  mmu_reserve_small_size;
  size_t                  mmu_reserve_allign;
  unsigned long long      mmu_reserve_flags;
}lhmmu_profile_t;

typedef int (*LhGetMMUProfile_t)(lhmmu_profile_t *profile);
typedef void* (*LhMMUAllocate_t)(/*void** devPtr,*/ size_t size);
typedef void (*LhMMUFree_t)(void* devPtr);
typedef void (*LhReleaseMMU_t)();
typedef clock_t (*LhGetTimeRestore_t)();
typedef void* (*LhGetPageLockMemAddr_t)();
typedef int (*LhAllocHost_t)(void** ptr, size_t size, unsigned int flag);
typedef int (*LhFreeHost_t)(void* ptr);
typedef void (*LhRegisterHost_t)(void* ptr, size_t size);
typedef void (*LhInitPageLockPool_t)();
typedef void (*LhReinitMMU_t)();
typedef void (*LhExecApi_t)(Cuda_Fncs_t fucOp, ...);

typedef void* (*LhGetMemCtx_t)(void* devicePtr);
typedef void (*LhGetIpcInfo_t)(void* devPtr, size_t *size, unsigned int *devId);
typedef unsigned int (*LhIpcGetApiEx_t)(void* handlePtr, void* devPtr);
typedef unsigned int (*LhIpcOpenApiEx_t)(void** devPtr, void* handlePtr, unsigned int flags);
typedef unsigned int (*LhIpcCloseApiEx_t)(void* devPtr);
typedef void (*LhNewThread_t)(pid_t uhTid, int shm_key);
typedef void (*LhGetFs_t)(pid_t uhTid, unsigned long *fsAddr);
typedef void (*LhWake_t)(pid_t uhTid);
typedef void (*LhWaitFinish_t)(pid_t uhTid);
typedef void (*LhFreeCudaType_t)(void* cudaType, int type);
// typedef void* (*lhCUtype_t)(void *cuType, int type);
// typedef void (*lhFreeCUtype_t)(void *cuType, int type);
#endif // ifndef COMMON_H
