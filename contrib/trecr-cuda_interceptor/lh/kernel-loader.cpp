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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#define CUDA 1
#endif
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

// add by tian01.liu
#include <pthread.h>
#include <stdarg.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>

// add by tian01.liu for supporting cudaIpcGetMemHandle/cudaIpcOpenMemHandle
#include <sys/socket.h>
#include <sys/un.h>

// add by tian01.liu for MT solution
#include <semaphore.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include "increamental/parsha256_src/parsha256.h"
#include "common.h"
#include "mem-restore.h"
#include "custom-loader.h"
#include "kernel-loader.h"
#include "logging.h"
#include "procmapsutils.h"
#include "trampoline_setup.h"
#include "utils.h"
#include "getmmap.h"
#include "log_and_replay.h"
#include "switch_context.h"
#include "../restart_plugin/mtcp_restart_plugin.h"
// For MPI
#include "../utils/mtcp_util.h"

#include "mmu/MMU.h"
// #include "device_heap_util.h"
#include "increamental/increamental.h"

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_INTERNAL_COMPILATION__
#endif
#include "crt/host_runtime.h"
#include "crt/device_functions.h"
#define USE_COMP 0

#if USE_COMP
/*****************by tian01.liu 2022.7.5******************************/
#include "nvcomp.hpp"
#include "nvcompManagerFactory.hpp"

using namespace nvcomp;
/*****************by tian01.liu 2022.7.5******************************/
#endif


LowerHalfInfo_t lhInfo;
UpperHalfInfo_t uhInfo;
pthread_mutex_t mutex_for_vector;

extern "C"
{
  void readUhInfoAddr();
}

#define UNUSED(expr) do {(void)(expr);} while(0)

void copy_lower_half_data();
extern void replayAPI(CudaCallLog_t *l);
// static void logs_read_and_apply();
//  Local function declarations
static void getProcStatField(enum Procstat_t, char *, size_t);
static void getStackRegion(Area *, Area *);
static void *deepCopyStack(void *, const void *, size_t,
                           const void *, const void *,
                           const DynObjInfo_t *);
static void *createNewStackForRtld(const DynObjInfo_t *, Area *heap);
static void *createNewHeapForRtld(const DynObjInfo_t *, Area *heap);
static void *getEntryPoint(DynObjInfo_t);
static void patchAuxv(ElfW(auxv_t) *, unsigned long,
                      unsigned long, unsigned long);
static int writeLhInfoToFile();
static int setupLowerHalfInfo(bool);
static void printUsage();
static void printRestartUsage();
// Global functions

/*************MMU Export Apis by tian01.liu 2023.01.06********************/
/**
 * used to setup mmu initialize infomation
 */
Lhmmu_profile_t lhMMUInfo;
static int setupMMUInfo(bool);
// static int writeLhMMUInfoToFile();
static void readMMUInfoFromUhFile();

// declear the MMU Allocator
class Allocator;
Allocator *g_allocator = nullptr;
bool g_bPageLockInit = false;

pthread_t g_curThread = 0;
Cuda_Fncs_t curOp = Cuda_Fnc_Invalid;
pthread_mutex_t thread_mutex;
va_list arglist;
bool bExecFinished = false;
std::map<pid_t, pthread_t> g_thread_maps;

int init_mmu_allocator();
void release_mmu_allocator();
void reinit_mmu_space();
// int  init_mmu_allocator_ex(MMUProfile *profile);
static int get_mmu_profile_from_lh(Lhmmu_profile_t *mmu_profile);
static void *allocate_gpu_mem(/*void** devPtr,*/ size_t length);
static void free_gpu_mem(void *devPtr);
static int realloc_gpu_blocks();
static void* getDevMemCtx(void* devPtr);

// by tian01.liu for ipc replay, 2023.8.15
static void getIpcInfo(void* devPtr, size_t *size, unsigned int *devId);
bool g_finishRelay = true;

// Restore MPI
static RestoreInfo rinfo;

// store the stack information of cuda application
void *g_stack_seg_addr = NULL;
size_t g_stack_seg_size;

Increamental inc_obj;

void *getApplicationStackAddr()
{
  return g_stack_seg_addr;
}

void* getDevMemCtx(void* devPtr)
{
  return g_allocator->get_ptr_ctx(devPtr);
}

void getIpcInfo(void* devPtr, size_t *size, unsigned int *devId)
{
    if (devPtr == nullptr || g_allocator == nullptr) {
      *size = 0;
      return;
    }

    Block* blk = nullptr;
    if (g_allocator->ptr_block_map.count(devPtr))
      blk = g_allocator->ptr_block_map[devPtr];

    if (blk == nullptr)
    {
      *size = 0;
      return;
    }

    *size = blk->size;
    *devId = blk->device;
}

void *getPinMemPoolAddr()
{
  return g_allocator->host_start_address;
}

void init_page_lock_pool()
{
  // cudaError_t ret;
  // if (!g_bPageLockInit)
  // {
    // CUcontext ctx;
    // CUdevice device_id = 0;
    // cuInit(0);
    // cuCtxCreate(&ctx, 0, device_id);
    // cuDevicePrimaryCtxRetain(&ctx, device_id);
    // /*CUresult ret = */cuMemHostRegister(g_allocator->host_start_address, 1024ULL * 1024 * 1024 * 4, CU_MEMHOSTREGISTER_PORTABLE);
    // cudaError_t ret = cudaHostRegister(g_allocator->host_start_address, 1024ULL * 1024 * 1024 * 4, CUDA_MEMHOSTRE)
    // printf("pool addr:%p, ret:%i\n", g_allocator->host_start_address, ret);
    // g_bPageLockInit = true;
  // }
  // printf("pool addr:%p, ret:%i\n", g_allocator->host_start_address, ret);
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
}

int allocate_page_lock_mem(void **ptr, size_t size, int flags)
{
  pthread_mutex_lock(&thread_mutex);
  int ret =  g_allocator->alloc_host(ptr, size);
  if (flags == 0x04)
    flags = 0x01;
  cudaHostRegister(*ptr, size, flags);
  pthread_mutex_unlock(&thread_mutex);
  return ret;
}

int free_page_lock_mem(void *ptr)
{
  cudaHostUnregister(ptr);
  return g_allocator->free_host(ptr);
  // return 0;
}

void restore_page_lock_pool(void *ptr_1, size_t size)
{
  void *ptr = uhInfo.lhPageLockRegion;
  if (ptr == nullptr)
  {
    fprintf(stderr, "restore_page_lock_pool empty addr\n");
    fflush(stderr);
    return;
  }

  size_t count = 0;
  int frag_cnt = 0;
  memcpy(&frag_cnt, (VA)ptr + count, sizeof(int));
  count += sizeof(int);

  for (int i = 0; i < frag_cnt; i++)
  {
    void *dst_addr = nullptr;
    st_page_lock_info blk_info;
    memcpy(&dst_addr, (VA)ptr + count, sizeof(void *));
    count += sizeof(void *);
    memcpy(&blk_info, (VA)ptr + count, sizeof(st_page_lock_info));
    count += sizeof(st_page_lock_info);

    memcpy((VA)dst_addr, (VA)ptr + count, blk_info.size);
    count += blk_info.size;

    // printf("dst_addr:%p, src_addr:%p\n", dst_addr, (VA)ptr + count);
    // TODO: update the status of page-lock memory pool
    g_allocator->update_pool_status(dst_addr, blk_info.size);
  }

  munmapWrapper(ptr, uhInfo.lhPageLockSize);
}

/*
 * ipc_handle: the file description that the cuMemExportShareableHandle Produced
 * udsPath: the unique path that the peer used to connect the uds server
 * bytes:
 * offset:
 */
typedef struct
{
  int ipc_handle;  // cuMemExport产生的文件描述符
  char udsPath[64];
  size_t bytes;  // handle对应的物理大小
  size_t offset; // IPC地址相对于handle起始地址的一个偏移量
  int deviceId; // the deviceId
}serv_args_t;

/*
* bytes: the size of CUmemGenericAllocationHandle
* offset: the offset that the real data from the start location of the handle
*/
typedef struct
{
  size_t bytes;  // handle对应的物理大小
  size_t offset;  // IPC地址相对于handle起始地址的一个偏移量
  int deviceId;
}cuda_handle_t;

void* udsServerFunc(void* server_args)
{
    serv_args_t *pServArg = (serv_args_t*)server_args;
    int    client;
    struct msghdr   msg;
    struct iovec    iov;
    char   cmsg_buf[CMSG_SPACE(sizeof(int))];
    struct cmsghdr *cmsg;
    cuda_handle_t   payload;

    int  sockfd;
    /*
      * open unix domain server socket
      */
    sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockfd < 0)
        printf("failed on socket(2):%m\n");

    struct sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    memcpy(addr.sun_path, pServArg->udsPath, 64);

    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
      printf("%u failed on bind(2): %m\n", getpid());

    int num_children = 0;
    if (listen(sockfd, num_children) < 0)
      printf("failed on listen(2): %m\n");

    for (;;)
    {
      client = accept(sockfd, NULL, NULL);
      if (client < 0)
          printf("failed on accept:%m\n");

      /* send a file descriptor using SCM_RIGHTS */
      memset(&msg, 0, sizeof(msg));
      msg.msg_control = cmsg_buf;
      msg.msg_controllen = sizeof(cmsg_buf);

      cmsg = CMSG_FIRSTHDR(&msg);
      cmsg->cmsg_len = CMSG_LEN(sizeof(int));
      cmsg->cmsg_level = SOL_SOCKET;
      cmsg->cmsg_type = SCM_RIGHTS;

      // pServArg->ipc_handle = 0;
      // printf("send ipc_handle:%i\n", pServArg->ipc_handle);
      fflush(stdout);
      memcpy(CMSG_DATA(cmsg), &(pServArg->ipc_handle), sizeof(int));

      memset(&payload, 0, sizeof(payload));
      payload.bytes = pServArg->bytes;
      payload.offset = pServArg->offset;
      payload.deviceId = pServArg->deviceId;

      iov.iov_base = &payload;
      iov.iov_len = sizeof(payload);
      msg.msg_iov = &iov;
      msg.msg_iovlen = 1;

      if (sendmsg(client, &msg, 0) < 0)
        printf("failed on sendmsg(2): %m\n");

      sleep(20);
      close(client);
    }
    // unlink(addr.sun_path);
    return NULL;
}

int recvIpcHandle(void* servArgs)
{
  serv_args_t* pSrvArgs = (serv_args_t*)servArgs;

  int client;
  struct msghdr msg;
  struct iovec  iov;
  struct cmsghdr *cmsg;
  char cmsg_buf[CMSG_SPACE(sizeof(int))];
  cuda_handle_t   payload;
  int ipc_handle = -1;

  struct sockaddr_un addr;
  addr.sun_family = AF_UNIX;
  memcpy(addr.sun_path, pSrvArgs->udsPath, 64);

  /* connect to the server socket */
  client = socket(AF_UNIX, SOCK_STREAM, 0);
  if (client < 0)
    printf("failed on socket(2): %m\n");

  // connect until uds file created by peer
  while (connect(client, (struct sockaddr *)(&addr), sizeof(struct sockaddr_un)) != 0) {
    usleep(50);
  }

  /* recv a file descriptor using SCM_RIGHTS */
  memset(&msg, 0, sizeof(msg));
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  iov.iov_base = &payload;
  iov.iov_len = sizeof(payload);

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  if (recvmsg(client, &msg, 0) <= 0)
    printf("failed on recvmsg(2): %m\n");

  // get file description from the control message
  cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg)
    printf("message has no control header\n");

  if (cmsg->cmsg_len == CMSG_LEN(sizeof(int)) && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
    memcpy(&ipc_handle, CMSG_DATA(cmsg), sizeof(int));
  else
    printf("unexpected control header \n");

  // TODO: need to get the src memory info from the comm message
  // payload.bytes;
  // payload.offset;
  pSrvArgs->deviceId = payload.deviceId;
  pSrvArgs->bytes = payload.bytes;
  pSrvArgs->offset = payload.offset;

  // printf("sun_path:%s, received ipc handle is:%i, deviceId:%i, size:%ld\n", addr.sun_path, ipc_handle, payload.deviceId, payload.bytes);

  remove(pSrvArgs->udsPath);

  return ipc_handle;
}

unsigned int cudaIpcGetMemHandleEx(void* handlePtr, void* devPtr)
{
  cudaError_t ret = cudaSuccess;

  return ret;
}

unsigned int cudaIpcOpenMemHandleEx(void** devPtr, void* handlePtr, unsigned int flags)
{
  cudaError_t ret = cudaSuccess;

  return ret;
}

std::vector<sem_t*> mutex_for_thread; // 存储每个线程对应的互斥锁,用于唤醒lh线程执行
std::vector<sem_t*> seg_for_uh;       // 存储每个线程对应的互斥锁，用户通知uh线程继续执行
// std::vector<pthread_cond_t*> cond_for_threads;  // 获取每个线程对应的条件变量
std::unordered_map<pid_t, int> idx_map_threads;// 线程ID对应的锁以及条件变量在对应vector中的索引
std::map<pid_t, unsigned long> fs_threads_map; // uh线程ID与lower half线程的FS对应关系

pthread_rwlock_t mutex_update_dt;

void setWakeupCondition(pid_t tid)
{
  int idx = idx_map_threads[tid];
  sem_post(mutex_for_thread[idx]);
}

// wait lower half 执行完成
void waitLhApiExecFinish(pid_t tid)
{
  int idx = idx_map_threads[tid];
  sem_wait(seg_for_uh[idx]);
}

// 通知upper half API已经执行完成
void notifyContinue(int idx)
{
  sem_post(seg_for_uh[idx]);
}

struct ThreadParams
{
  pid_t tid;
  int shm_key;
  int idx;
};
// shm size
#define BUFSZ 512

void *threadFunc_Ex(void* param)
{
  ThreadParams *threadParam = (ThreadParams*)param;
  pid_t tid = threadParam->tid;
  unsigned long fsAddr;
  int rc = syscall(SYS_arch_prctl, ARCH_GET_FS, &fsAddr);
  if (rc < 0)
  {
    DLOG(ERROR, "failed to get fs: %d\n", errno);
    exit(-1);
  }
  sem_t * sem = (sem_t*)malloc(sizeof(sem_t));
  sem_init(sem, 0, 0);


  // TODO:分配共享内存，用于upper-half与lower-half之间传递参数预计结果
  // int shmid;dd
  // char *shmadd;
  // //创建共享内存
  // DLOG(ERROR, "tid in lower-half:%i\n", tid);
  // shmid = shmget((key_t)(threadParam->shm_key), BUFSZ, IPC_CREAT | 0666);
  // if(shmid < 0)
  // {
  //   DLOG(ERROR, "shmget failure...\n");
  //   exit(-1);
  // }

  // //映射
  // shmadd = (char*)shmat(shmid, NULL, 0);
  // if(shmadd < 0)
  // {
  //     DLOG(ERROR, "shmat ");
  //     _exit(-1);
  // }
  pthread_rwlock_wrlock(&mutex_update_dt);
  fs_threads_map[tid] = fsAddr;
  pthread_rwlock_unlock(&mutex_update_dt);
  while (1)
  {
    sem_wait(sem);
    printf("test thread func in lower half....\n");

    // TODO: 从共享内存中读取参数并执行
    // size_t count = 0;
    // Cuda_Fncs_t op;
    // memcpy(&op, shmadd + count, sizeof(Cuda_Fncs_t));
    // count += sizeof(Cuda_Fncs_t);
    // DLOG(ERROR, "op:%i\n", op);
    // switch(op)
    // {
    //   case GENERATE_ENUM(cudaSetDevice):{
    //     cudaError_t ret;
    //     // memcpy(&ret, shmadd + count, sizeof(cudaError_t*));
    //     // count += sizeof(cudaError_t*);

    //     int devId;
    //     memcpy(&devId, shmadd + count, sizeof(int));
    //     count += sizeof(int);
    //     ret = cudaSetDevice(devId);
    //     printf("cudaSetDevice in thread:%i, device:%i, ret:%i\n", tid, devId, ret);

    //     UNUSED(ret);

    //     bzero(shmadd, BUFSZ); // 清空shm中数据
    //     break;
    //   }
    //   case GENERATE_ENUM(cudaGetDevice):{
    //     cudaError_t ret;
    //     // memcpy(&ret, shmadd + count, sizeof(cudaError_t*));
    //     // count += sizeof(cudaError_t*);

    //     int* devId;
    //     memcpy(&devId, shmadd + count, sizeof(int*));
    //     count += sizeof(int*);

    //     ret = cudaGetDevice(devId);
    //     UNUSED(ret);
    //     bzero(shmadd, BUFSZ);
    //     break;
    //   }
    //   case GENERATE_ENUM(cudaDeviceSynchronize):
    //   {
    //     cudaError_t ret;

    //     ret = cudaDeviceSynchronize();
    //     UNUSED(ret);
    //     bzero(shmadd, BUFSZ);
    //     break;
    //   }
    //   case GENERATE_ENUM(cudaMalloc):
    //   {
    //     cudaError_t ret;
    //     void** pointer;
    //     memcpy(&pointer, shmadd + count, sizeof(void**));
    //     count += sizeof(void**);
    //     size_t size;
    //     memcpy(&size, shmadd + count, sizeof(size_t));
    //     count += sizeof(size_t);
    //     pthread_mutex_lock(&mutex_update_dt);
    //     if(!g_allocator->is_gpu_inited){
    //       g_allocator->init_gpu_memory_pool(NULL, NULL);
    //       setupMMUInfo(false);
    //     }
    //     *pointer = g_allocator->malloc_gpu(size);
    //     pthread_mutex_unlock(&mutex_update_dt);
    //     UNUSED(ret);
    //     break;
    //   }
    //   case GENERATE_ENUM(cudaMemcpy):
    //   {
    //     cudaError_t ret;
    //     void* dst;
    //     memcpy(&dst, shmadd + count, sizeof(void*));
    //     count += sizeof(void*);
    //     void* src;
    //     memcpy(&src, shmadd + count, sizeof(void*));
    //     count += sizeof(void*);
    //     size_t count_1;
    //     memcpy(&count_1, shmadd + count, sizeof(size_t));
    //     count += sizeof(size_t);
    //     cudaMemcpyKind kind;
    //     memcpy(&kind, shmadd + count, sizeof(cudaMemcpyKind));
    //     count += sizeof(cudaMemcpyKind);

    //     ret = cudaMemcpy(dst, src, count_1, kind);
    //     UNUSED(ret);
    //     break;
    //   }
    //   case GENERATE_ENUM(cudaMemset):
    //   {
    //     cudaError_t ret;
    //     void* devPtr;
    //     memcpy(&devPtr, shmadd + count, sizeof(void*));
    //     count += sizeof(void*);
    //     int value;
    //     memcpy(&value, shmadd + count, sizeof(int));
    //     count += sizeof(int);
    //     size_t count_1;
    //     memcpy(&count_1, shmadd + count, sizeof(size_t));
    //     count += sizeof(size_t);
    //     ret = cudaMemset(devPtr, value, count_1);
    //     UNUSED(ret);
    //     break;
    //   }
    //   case GENERATE_ENUM(cudaStreamCreate):
    //   {
    //     cudaError_t ret;
    //     cudaStream_t* stream;
    //     memcpy(&stream, shmadd + count, sizeof(cudaStream_t*));
    //     count += sizeof(void*);

    //     ret = cudaStreamCreate(stream);
    //     UNUSED(ret);
    //     break;
    //   }
    //   case GENERATE_ENUM(cudaStreamCreateWithPriority):
    //   {
    //     cudaError_t ret;
    //     cudaStream_t* stream;
    //     memcpy(&stream, shmadd + count, sizeof(cudaStream_t*));
    //     count += sizeof(void*);
    //     unsigned int flag;
    //     memcpy(&flag, shmadd + count, sizeof(unsigned int));
    //     count += sizeof(unsigned int);
    //     int priority;
    //     memcpy(&priority, shmadd + count, sizeof(int));

    //     ret = cudaStreamCreateWithPriority(stream, flag, priority);
    //     UNUSED(ret);
    //     break;
    //   }
    //   case GENERATE_ENUM(cudaStreamCreateWithFlags):
    //   {
    //     cudaError_t ret;
    //     cudaStream_t* stream;
    //     memcpy(&stream, shmadd + count, sizeof(cudaStream_t*));
    //     count += sizeof(void*);
    //     unsigned int flag;
    //     memcpy(&flag, shmadd + count, sizeof(unsigned int));
    //     count += sizeof(unsigned int);

    //     ret = cudaStreamCreateWithFlags(stream, flag);
    //     UNUSED(ret);
    //     break;
    //   }
    //   default:
    //     break;
    // }
  }
}
// add by tian01.liu 2023.10.28
void newThread(pid_t uhTid, int shmKey)
{
  pthread_mutex_lock(&thread_mutex);

  pthread_t newThread;
  ThreadParams *param = (ThreadParams*)malloc(sizeof(ThreadParams));
  param->tid = uhTid;
  param->shm_key = shmKey;
  pthread_create(&newThread, NULL, &threadFunc_Ex, param);
  pthread_mutex_unlock(&thread_mutex);
}

void getFs(pid_t uhTid, unsigned long *fsAddr)
{
  while(!fs_threads_map.count(uhTid));
  pthread_rwlock_rdlock(&mutex_update_dt);
  // DLOG(ERROR, "tid:%i, fs:%ld\n", uhTid, fs_threads_map[uhTid]);
  *fsAddr = fs_threads_map[uhTid];
  pthread_rwlock_unlock(&mutex_update_dt);
}
unsigned int cudaIpcCloseMemHandleEx(void* devPtr)
{
  // printf("enter cudaIpcCloseMemHandleEx...%p\n", devPtr);
  // fflush(stdout);
  cudaError_t ret = cudaSuccess;
  g_allocator->free_ipc(devPtr);
  return ret;
}

void *threadFunc(void* param)
{
  pthread_t tid = pthread_self();
  while (1)
  {
    while (g_curThread != tid)
      usleep(10);
    switch (curOp)
    {
      case GENERATE_ENUM(cudaCreateTextureObject):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        cudaTextureObject_t * pTexObject = va_arg(arglist, cudaTextureObject_t*);
        const struct cudaResourceDesc * pResDesc = va_arg(arglist, struct cudaResourceDesc*);
        const struct cudaTextureDesc * pTexDesc = va_arg(arglist, struct cudaTextureDesc*);
        const struct cudaResourceViewDesc * pResViewDesc = va_arg(arglist, struct cudaResourceViewDesc*);
        *orig_ret = cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
        break;
      }
      case GENERATE_ENUM(cudaDestroyTextureObject):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        cudaTextureObject_t pTexObject = va_arg(arglist, cudaTextureObject_t);
        *orig_ret = cudaDestroyTextureObject(pTexObject);
        break;
      }
      case GENERATE_ENUM(cudaBindTexture):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        size_t* offset = va_arg(arglist, size_t*);
        const textureReference* texref = va_arg(arglist, textureReference*);
        const void* devPtr = va_arg(arglist, void*);
        const cudaChannelFormatDesc* desc = va_arg(arglist, cudaChannelFormatDesc*);
        size_t size = va_arg(arglist, size_t);
        *orig_ret = cudaBindTexture(offset, texref, devPtr, desc, size);
        break;
      }
      case GENERATE_ENUM(cudaBindTexture2D):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        size_t* offset = va_arg(arglist, size_t*);
        const textureReference* texref = va_arg(arglist, textureReference*);
        const void* devPtr = va_arg(arglist, void*);
        const cudaChannelFormatDesc* desc = va_arg(arglist, cudaChannelFormatDesc*);
        size_t width = va_arg(arglist, size_t);
        size_t height = va_arg(arglist, size_t);
        size_t pitch = va_arg(arglist, size_t);
        *orig_ret = cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
        break;
      }
      case GENERATE_ENUM(cudaBindTextureToArray):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        const textureReference* texref = va_arg(arglist, textureReference*);
        cudaArray_const_t array = va_arg(arglist, cudaArray_const_t);
        const cudaChannelFormatDesc* desc = va_arg(arglist, cudaChannelFormatDesc*);
        *orig_ret = cudaBindTextureToArray(texref, array, desc);
        break;
      }
      case GENERATE_ENUM(cudaUnbindTexture):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        const struct textureReference* texref = va_arg(arglist, struct textureReference*);
        *orig_ret = cudaUnbindTexture(texref);
        break;
      }
      case GENERATE_ENUM(cudaCreateChannelDesc):
      {
        cudaChannelFormatDesc *orig_ret = va_arg(arglist, cudaChannelFormatDesc*);
        int  x = va_arg(arglist, int);
        int  y = va_arg(arglist, int);
        int  z = va_arg(arglist, int);
        int  w = va_arg(arglist, int);
        int f = va_arg(arglist, int);
        *orig_ret = cudaCreateChannelDesc(x, y, z, w, (cudaChannelFormatKind)f);
        break;
      }
      case GENERATE_ENUM(cudaStreamGetCaptureInfo_v2):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t stream =  va_arg(arglist, cudaStream_t);
        cudaStreamCaptureStatus * captureStatus_out = va_arg(arglist, cudaStreamCaptureStatus *);
        unsigned long long* id_out = va_arg(arglist, unsigned long long *);
        cudaGraph_t* graph_out = va_arg(arglist, cudaGraph_t *);
        const cudaGraphNode_t** dependencies_out = va_arg(arglist, const cudaGraphNode_t**);
        size_t* numDependencies_out = va_arg(arglist, size_t *);
        *orig_ret = cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
        break;
      }
      case GENERATE_ENUM(cudaEventCreate):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        cudaEvent_t * event = va_arg(arglist, cudaEvent_t*);
        *orig_ret = cudaEventCreate(event);
        printf("cudaEventCreate, event:%p, ret:%i\n", *event, *orig_ret);
        break;
      }
      case GENERATE_ENUM(cudaEventCreateWithFlags):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        cudaEvent_t * event = va_arg(arglist, cudaEvent_t*);
        unsigned int flags = va_arg(arglist, unsigned int);
        *orig_ret = cudaEventCreateWithFlags(event, flags);
        // printf("cudaEventCreateWithFlags, event:%p, ret:%i\n", *event, *orig_ret);
        break;
      }
      case GENERATE_ENUM(cudaEventDestroy):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        cudaEvent_t event = va_arg(arglist, cudaEvent_t);
        *orig_ret = cudaEventDestroy(event);
        break;
      }
      case GENERATE_ENUM(cudaEventElapsedTime):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        float * ms = va_arg(arglist, float*);
        cudaEvent_t start = va_arg(arglist, cudaEvent_t);
        cudaEvent_t end = va_arg(arglist, cudaEvent_t);
        *orig_ret = cudaEventElapsedTime(ms, start, end);
        break;
      }
      case GENERATE_ENUM(cudaEventQuery):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        cudaEvent_t event = va_arg(arglist, cudaEvent_t);
        *orig_ret = cudaEventQuery(event);
        break;
      }
      case GENERATE_ENUM(cudaEventRecord):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        cudaEvent_t event = va_arg(arglist, cudaEvent_t);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *orig_ret = cudaEventRecord(event, stream);
        break;
      }
      case GENERATE_ENUM(cudaEventSynchronize):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*); 
        cudaEvent_t event = va_arg(arglist, cudaEvent_t);
        *orig_ret = cudaEventSynchronize(event);
        break;
      }
      case GENERATE_ENUM(cudaMalloc):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        void ** pointer = va_arg(arglist, void**);
        size_t size = va_arg(arglist, size_t);
        *pointer = allocate_gpu_mem(size);
        *orig_ret = cudaSuccess;
        break;
      }
      case GENERATE_ENUM(cudaFree):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        void * pointer = va_arg(arglist, void*);
        // printf("cudaFree before....,ptr:%p\n", pointer);
        free_gpu_mem(pointer);
        *orig_ret = cudaSuccess;
        break;
      }
      case GENERATE_ENUM(cudaMallocArray):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        struct cudaArray ** array = va_arg(arglist, struct cudaArray **);
        const struct cudaChannelFormatDesc * desc  = va_arg(arglist, struct cudaChannelFormatDesc *);
        size_t width = va_arg(arglist, size_t);
        size_t height = va_arg(arglist, size_t);
        unsigned int flags  = va_arg(arglist, unsigned int);
        *orig_ret = cudaMallocArray(array, desc, width, height, flags);
        break;
      }
      case GENERATE_ENUM(cudaFreeArray):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        struct cudaArray * array = va_arg(arglist, struct cudaArray *);
        *orig_ret = cudaFreeArray(array);
        break;
      }
      case GENERATE_ENUM(cudaHostRegister):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        void* ptr = va_arg(arglist, void*);
        size_t size = va_arg(arglist, size_t);
        unsigned int  flags = va_arg(arglist, unsigned int);
        cudaError_t ret = cudaHostRegister(ptr, size, flags);
        memcpy(orig_ret, &ret, sizeof(cudaError_t));
        break;
      }
      case GENERATE_ENUM(cudaHostGetDevicePointer):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        void** pDevice = va_arg(arglist, void**);
        void* pHost = va_arg(arglist, void*);
        unsigned int  flags = va_arg(arglist, unsigned int);
        cudaError_t ret = cudaHostGetDevicePointer(pDevice, pHost, flags);
        memcpy(orig_ret, &ret, sizeof(cudaError_t));
        fflush(stdout);
        break;
      }
      case GENERATE_ENUM(cudaDeviceGetAttribute):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        int* value = va_arg(arglist, int*);
        int attr = va_arg(arglist, int);
        int  device = va_arg(arglist, int);
        cudaError_t ret = cudaDeviceGetAttribute(value, (cudaDeviceAttr)attr, device);
        memcpy(orig_ret, &ret, sizeof(cudaError_t));
        break;
      }
      case GENERATE_ENUM(cudaMallocHost):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        void ** ptr = va_arg(arglist, void **);
        size_t size = va_arg(arglist, size_t);
        int ret = allocate_page_lock_mem(ptr, size, 0x01);
	      // cudaHostRegister(*ptr, size, 0x01);
        if (ret)
          *orig_ret = cudaErrorUnknown;
        else
          *orig_ret = cudaSuccess;
        // *orig_ret = cudaMallocHost(ptr, size);
	break;
      }
      case GENERATE_ENUM(cudaFreeHost):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        void * ptr = va_arg(arglist, void *);
        free_page_lock_mem(ptr);
        *orig_ret = cudaSuccess;
        break;
      }
      case GENERATE_ENUM(cudaHostAlloc):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        void ** ptr = va_arg(arglist, void **);
        size_t size = va_arg(arglist, size_t);
        unsigned int flags = va_arg(arglist, unsigned int);
        int ret = allocate_page_lock_mem(ptr, size, flags);
        // cudaError_t cudaRet = cudaHostRegister(*ptr, size, flags);
        // printf("cudaHostRegister, ret:%i\n", cudaRet);
        if (ret)
          *orig_ret = cudaErrorUnknown;
        else
          *orig_ret = cudaSuccess;
	      // *orig_ret = cudaHostAlloc(ptr, size, flags);
        break;
      }
      case GENERATE_ENUM(cudaMallocPitch):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        void ** devPtr = va_arg(arglist, void**);
        size_t * pitch = va_arg(arglist, size_t*);
        size_t width = va_arg(arglist, size_t);
        size_t height = va_arg(arglist, size_t);
        *orig_ret = cudaMallocPitch(devPtr, pitch, width, height);
        break;
      }
      case GENERATE_ENUM(cudaGetDevice):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        int* device = va_arg(arglist, int*);
        *orig_ret = cudaGetDevice(device);
        break;
      }
      case GENERATE_ENUM(cudaSetDevice):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        int device = va_arg(arglist, int);
        *orig_ret = cudaSetDevice(device);
        break;
      }
      case GENERATE_ENUM(cudaDeviceGetLimit):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        size_t* pValue = va_arg(arglist, size_t*);
        int limit = va_arg(arglist, int);
        *orig_ret = cudaDeviceGetLimit(pValue, (cudaLimit)limit);
        break;
      }
      case GENERATE_ENUM(cudaDeviceSetLimit):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        int limit = va_arg(arglist, int);
        size_t value = va_arg(arglist, size_t);
        *orig_ret = cudaDeviceSetLimit((cudaLimit)limit, value);
        break;
      }
      case GENERATE_ENUM(cudaGetDeviceCount):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        int * count = va_arg(arglist, int*);
        *orig_ret = cudaGetDeviceCount(count);
        break;
      }
      case GENERATE_ENUM(cudaDeviceSetCacheConfig):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        int cacheConfig = va_arg(arglist, int);
        *orig_ret = cudaDeviceSetCacheConfig((cudaFuncCache)cacheConfig);
        break;
      }
      case GENERATE_ENUM(cudaGetDeviceProperties):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        cudaDeviceProp* prop = va_arg(arglist, cudaDeviceProp*);
        int  device = va_arg(arglist, int);
        *orig_ret = cudaGetDeviceProperties(prop, device);
        break;
      }
      case GENERATE_ENUM(cudaDeviceCanAccessPeer):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        int * canAccessPeer = va_arg(arglist, int*);
        int  device = va_arg(arglist, int);
        int peerDevice = va_arg(arglist, int);
        *orig_ret = cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
        break;
      }
      case GENERATE_ENUM(cudaDeviceGetPCIBusId):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        char * pciBusId  = va_arg(arglist, char*);
        int len = va_arg(arglist, int);
        int  device = va_arg(arglist, int);
        *orig_ret = cudaDeviceGetPCIBusId(pciBusId, len, device);
        break;
      }
      case GENERATE_ENUM(cudaDeviceReset):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        *orig_ret = cudaDeviceReset();
        break;
      }
      case GENERATE_ENUM(cudaDeviceSynchronize):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        *orig_ret = cudaDeviceSynchronize();
        break;
      }
      case GENERATE_ENUM(cudaLaunchKernel):
      {
        cudaError_t *orig_ret = va_arg(arglist, cudaError_t*);
        const void* func = va_arg(arglist, void*); 
        dim3 gridDim = va_arg(arglist, dim3);
        dim3 blockDim = va_arg(arglist, dim3);
        void** args = va_arg(arglist, void**);
        size_t sharedMem = va_arg(arglist, size_t);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        // printf("cudaLaunchKernel, gridDim-x:%i -y:%i -z:%i\n", gridDim.x, gridDim.y, gridDim.z);
        // printf("cudaLaunchKernel, blockDim-x:%i -y:%i -z:%i\n", blockDim.x, blockDim.y, blockDim.z);
        // printf("cudaLaunchKernel, args:%p, %p, func:%p, stream:%p\n", args[0], args[1], func, stream);
        *orig_ret = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
        // printf("cudaLaunchKernel....ret:%i, args:%p\n", *orig_ret, args);
        break;
      }
      case GENERATE_ENUM(cudaMallocManaged):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void** devPtr = va_arg(arglist, void**);
        size_t size = va_arg(arglist, size_t);
        unsigned int  flags = va_arg(arglist, unsigned int);
        *orig_ret = cudaMallocManaged(devPtr, size, flags);
        break;
      }
      case GENERATE_ENUM(cudaMemcpy):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void* dst = va_arg(arglist, void*);
        const void* src = va_arg(arglist, void*);
        size_t count = va_arg(arglist, size_t);
        int kind = va_arg(arglist, int);
        *orig_ret = cudaMemcpy(dst, src, count, (cudaMemcpyKind)kind);
        break;
      }
      case GENERATE_ENUM(cudaMemcpy2D):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void* dst = va_arg(arglist, void*);
        size_t dpitch = va_arg(arglist, size_t);
        const void* src = va_arg(arglist, void*);
        size_t spitch = va_arg(arglist, size_t);
        size_t width = va_arg(arglist, size_t);
        size_t height = va_arg(arglist, size_t);
        int kind = va_arg(arglist, int);
        *orig_ret = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, (cudaMemcpyKind)kind);
        break;
      }
      case GENERATE_ENUM(cudaMemcpyToArray):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaArray_t dst = va_arg(arglist, cudaArray_t);
        size_t wOffset = va_arg(arglist, size_t);
        size_t hOffset = va_arg(arglist, size_t);
        const void* src = va_arg(arglist, void*);
        size_t count = va_arg(arglist, size_t);
        int kind = va_arg(arglist, int);
        *orig_ret = cudaMemcpyToArray(dst, wOffset, hOffset, src, count, (cudaMemcpyKind)kind);
        break;
      }
      case GENERATE_ENUM(cudaMemcpyToSymbol):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        const void* symbol = va_arg(arglist, void*);
        const void* src = va_arg(arglist, void*);
        size_t count = va_arg(arglist, size_t);
        size_t offset = va_arg(arglist, size_t);
        int kind = va_arg(arglist, int);
        *orig_ret = cudaMemcpyToSymbol(symbol, src, count, offset, (cudaMemcpyKind)kind);
        break;
      }
      case GENERATE_ENUM(cudaMemcpyToSymbolAsync):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        const void* symbol = va_arg(arglist, void*);
        const void* src = va_arg(arglist, void*);
        size_t count = va_arg(arglist, size_t);
        size_t offset = va_arg(arglist, size_t);
        int kind = va_arg(arglist, int);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *orig_ret = cudaMemcpyToSymbolAsync(symbol, src, count, offset, (cudaMemcpyKind)kind, stream);
        break;
      }
      case GENERATE_ENUM(cudaMemcpyAsync):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void* dst = va_arg(arglist, void*);
        const void* src = va_arg(arglist, void*);
        size_t count = va_arg(arglist, size_t);
        int kind = va_arg(arglist, int);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *orig_ret = cudaMemcpyAsync(dst, src, count, (cudaMemcpyKind)kind, stream);
        break;
      }
      case GENERATE_ENUM(cudaMemset):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void * devPtr = va_arg(arglist, void*);
        int value = va_arg(arglist, int);
        size_t count = va_arg(arglist, size_t);
        *orig_ret = cudaMemset(devPtr, value, count);
        break;
      }
      case GENERATE_ENUM(cudaMemset2D):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void * devPtr = va_arg(arglist, void*);
        size_t pitch  = va_arg(arglist, size_t);
        int value = va_arg(arglist, int);
        size_t width  = va_arg(arglist, size_t);
        size_t height  = va_arg(arglist, size_t);
        *orig_ret = cudaMemset2D(devPtr, pitch, value, width, height);
        break;
      }
      case GENERATE_ENUM(cudaMemsetAsync):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void * devPtr = va_arg(arglist, void*);
        int value = va_arg(arglist, int);
        size_t count = va_arg(arglist, size_t);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *orig_ret = cudaMemsetAsync(devPtr, value, count, stream);
        break;
      }
      case GENERATE_ENUM(cudaMemGetInfo):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        size_t * free = va_arg(arglist, size_t *);
        size_t * total = va_arg(arglist, size_t *);
        *orig_ret = cudaMemGetInfo(free, total);
        break;
      }
      case GENERATE_ENUM(cudaMemAdvise):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        const void * devPtr = va_arg(arglist, void *);
        size_t  count = va_arg(arglist, size_t);
        int advice = va_arg(arglist, int);
        int device = va_arg(arglist, int);
        *orig_ret = cudaMemAdvise(devPtr, count, (cudaMemoryAdvise)advice, device);
        break;
      }
      case GENERATE_ENUM(cudaMemPrefetchAsync):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        const void * devPtr = va_arg(arglist, void *);
        size_t  count = va_arg(arglist, size_t);
        int dstDevice = va_arg(arglist, int);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *orig_ret = cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
        break;
      }
      case GENERATE_ENUM(cudaStreamCreate):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t * pStream = va_arg(arglist, cudaStream_t*);
        *orig_ret = cudaStreamCreate(pStream);
        // printf("cudaStreamCreate, stream:%p, ret:%i\n", *pStream, *orig_ret);
        break;
      }
      case GENERATE_ENUM(cudaStreamCreateWithPriority):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t * pStream = va_arg(arglist, cudaStream_t*);
        unsigned int  flags = va_arg(arglist, unsigned int);
        int  priority  = va_arg(arglist, int);
        *orig_ret = cudaStreamCreateWithPriority(pStream, flags, priority);
        // printf("cudaStreamCreateWithPriority, stream:%p, ret:%i\n", *pStream, *orig_ret);
        break;
      }
      case GENERATE_ENUM(cudaStreamCreateWithFlags):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t * pStream = va_arg(arglist, cudaStream_t*);
        unsigned int  flags = va_arg(arglist, unsigned int);
        *orig_ret = cudaStreamCreateWithFlags(pStream, flags);
        // printf("cudaStreamCreateWithFlags, stream:%p, ret:%i\n", *pStream, *orig_ret);
        break;
      }
      case GENERATE_ENUM(cudaStreamDestroy):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t pStream = va_arg(arglist, cudaStream_t);
        *orig_ret = cudaStreamDestroy(pStream);
        break;
      }
      case GENERATE_ENUM(cudaStreamSynchronize):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t pStream = va_arg(arglist, cudaStream_t);
        *orig_ret = cudaStreamSynchronize(pStream);
        break;
      }
      case GENERATE_ENUM(cudaStreamWaitEvent):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t pStream = va_arg(arglist, cudaStream_t);
        cudaEvent_t event = va_arg(arglist, cudaEvent_t);
        unsigned int flags = va_arg(arglist, unsigned int);
        *orig_ret = cudaStreamWaitEvent(pStream, event, flags);
        // printf("[LT]#########cudaStreamWaitEvent:%i, pStream:%p, event:%p\n", *orig_ret, pStream, event);
        break;
      }
      case GENERATE_ENUM(cudaThreadSynchronize):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        *orig_ret = cudaThreadSynchronize();
        break;
      }
      case GENERATE_ENUM(cudaThreadExit):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        *orig_ret = cudaThreadExit();
        break;
      }
      case GENERATE_ENUM(cudaPointerGetAttributes):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaPointerAttributes* attributes = va_arg(arglist, cudaPointerAttributes*);
        const void* ptr = va_arg(arglist, void*);
        *orig_ret = cudaPointerGetAttributes(attributes, ptr);
        break;
      }
      case GENERATE_ENUM(cudaGetErrorString):
      {
        const char* orig_ret = va_arg(arglist, char*);
        int error = va_arg(arglist, int);
        orig_ret = cudaGetErrorString((cudaError_t)error);
        printf("error string:%s\n", orig_ret);
        break;
      }
      case GENERATE_ENUM(cudaGetErrorName):
      {
        const char* orig_ret = va_arg(arglist, char*);
        int error = va_arg(arglist, int);
        orig_ret = cudaGetErrorName((cudaError_t)error);
        printf("error name:%s\n", orig_ret);
        break;
      }
      case GENERATE_ENUM(cudaGetLastError):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        *orig_ret = cudaGetLastError();
        break;
      }
      case GENERATE_ENUM(cudaPeekAtLastError):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        *orig_ret = cudaPeekAtLastError();
        break;
      }
      case GENERATE_ENUM(cudaFuncSetCacheConfig):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        const void* func = va_arg(arglist, void*);
        int cacheConfig = va_arg(arglist, int);
        *orig_ret = cudaFuncSetCacheConfig(func, (cudaFuncCache)cacheConfig);
        break;
      }
     case GENERATE_ENUM(cudaIpcGetMemHandle):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaIpcMemHandle_t* handle = va_arg(arglist, cudaIpcMemHandle_t*);
        void* devPtr = va_arg(arglist, void*);

        size_t offset;
        CUmemGenericAllocationHandle mem_handle = g_allocator->get_allocation_handle(devPtr, &offset);
        // printf("offset %ld, mem_handle:%lld, devPtr:%p\n", offset, mem_handle, devPtr);
        // fflush(stdout);

        if (mem_handle == 0)
            *orig_ret = cudaErrorInvalidValue;
        else
        {
          if (g_finishRelay) // normal workflow
          {
            size_t handle_size = g_allocator->handle_size_map[mem_handle];
            int ipc_handle;
            CUresult rslt = cuMemExportToShareableHandle((void*)&ipc_handle, mem_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 );
            // printf("cuMemExportToShareableHandle, ret:%i, devPtr:%p, handle_size:%ld\n", rslt, devPtr, handle_size);

            // TODO: 创建UDS，启动线程等待客户端连接UDS,然后将文件cuMemExportToShareableHandle产生的Linux文件描述符传递给需要的进程
            pid_t curPid = getpid();
            char udsPath[64] = {0};
            sprintf(udsPath, "pid:%d-tid:%ld-%i", curPid, tid, ipc_handle);
            // memcpy(handle->reserved, udsPath, 64);
            // printf("udsPath in cudaIpcGetMemHandle:%s\n", udsPath);

            // TODO:创建线程发送文件描述符给对端进程。
            int deviceId;
            cudaGetDevice(&deviceId);

            serv_args_t *serv_args = (serv_args_t*)malloc(sizeof(serv_args_t));
            serv_args->offset = offset;
            serv_args->bytes = handle_size;
            serv_args->ipc_handle = ipc_handle;
            serv_args->deviceId = deviceId;
            memcpy(serv_args->udsPath, udsPath, 64);
            pthread_t tmpThread;
            pthread_create(&tmpThread, NULL, udsServerFunc, serv_args);

            if (rslt == CUDA_SUCCESS) {
              memcpy(handle->reserved, udsPath, 64);
              *orig_ret = cudaSuccess;
            }
          }
          else // replay workflow
          {
            // SCENE-0: A-EXPORT====>CHECK====>B-IMPORT
            // SCENE-1: A-EXPORT====>B-IMPORT====>CHECK

            size_t handle_size = g_allocator->handle_size_map[mem_handle];
            int ipc_handle;
            /*CUresult rslt = */cuMemExportToShareableHandle((void*)&ipc_handle, mem_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 );
            // printf("cuMemExportToShareableHandle, ret:%i, devPtr:%p, handle_size:%ld\n", rslt, devPtr, handle_size);

            int deviceId;
            cudaGetDevice(&deviceId);

            serv_args_t *serv_args = (serv_args_t*)malloc(sizeof(serv_args_t));
            serv_args->offset = offset;
            serv_args->bytes = handle_size;
            serv_args->ipc_handle = ipc_handle;
            serv_args->deviceId = deviceId;
            memcpy(serv_args->udsPath, handle->reserved, 64);
            // printf("handle->reserved:%s\n", handle->reserved);

            pthread_t tmpThread;
            pthread_create(&tmpThread, NULL, udsServerFunc, serv_args);
          }
        }
        break;
      }
      case GENERATE_ENUM(cudaUserObjectCreate):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaUserObject_t* object_out = va_arg(arglist, cudaUserObject_t*); 
        void* ptr = va_arg(arglist, void*);
        cudaHostFn_t destroy = va_arg(arglist, cudaHostFn_t);
        unsigned int  initialRefcount = va_arg(arglist, unsigned int);
        unsigned int  flags = va_arg(arglist, unsigned int);
        *orig_ret = cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
        break;
      }
      case GENERATE_ENUM(cudaGraphRetainUserObject):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaGraph_t graph = va_arg(arglist, cudaGraph_t);
        cudaUserObject_t object = va_arg(arglist, cudaUserObject_t);
        unsigned int  count = va_arg(arglist, unsigned int);
        unsigned int  flags = va_arg(arglist, unsigned int);
        *orig_ret = cudaGraphRetainUserObject(graph, object, count, flags);
        break;
      }
      case GENERATE_ENUM(cudaStreamUpdateCaptureDependencies):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        cudaGraphNode_t* dependencies = va_arg(arglist, cudaGraphNode_t*);
        size_t numDependencies = va_arg(arglist, size_t);
        unsigned int  flags = va_arg(arglist, unsigned int);
        *orig_ret = cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);
        break;
      }
      case GENERATE_ENUM(cudaGetDriverEntryPoint):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        const char* symbol = va_arg(arglist, char*);
        void** funcPtr = va_arg(arglist, void**);
        unsigned long long flags = va_arg(arglist, unsigned long long);
        *orig_ret = cudaGetDriverEntryPoint(symbol, funcPtr, flags);
        break;
      }
      case GENERATE_ENUM(cudaFuncSetAttribute):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        const void* func = va_arg(arglist, void*);
        int attr = va_arg(arglist, int);
        int  value = va_arg(arglist, int);
        *orig_ret = cudaFuncSetAttribute(func, (cudaFuncAttribute)attr, value);
        break;
      }
      case GENERATE_ENUM(cudaIpcOpenMemHandle):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void** devPtr = va_arg(arglist, void**);
        cudaIpcMemHandle_t handle = va_arg(arglist, cudaIpcMemHandle_t);
        unsigned int  flags = va_arg(arglist, unsigned int);
        UNUSED(flags);
        int ipc_handle;
        serv_args_t servArgs;
        memcpy(servArgs.udsPath, handle.reserved, 64);

        if (g_finishRelay) // normal workflow
        {
          ipc_handle = recvIpcHandle(&servArgs);
          size_t ipc_size = servArgs.bytes;
          size_t ipc_offset = servArgs.offset;
          int device_id = servArgs.deviceId;


          CUmemGenericAllocationHandle allocate_handle;
          /*CUresult rslt = */cuMemImportFromShareableHandle(&allocate_handle, (void*)(uintptr_t)ipc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
          if (!g_allocator->is_gpu_inited)
            g_allocator->init_gpu_memory_pool(0, 0);

          CUmemAllocationProp prop;
          /*rslt = */cuMemGetAllocationPropertiesFromHandle(&prop, allocate_handle);
          // printf("devId in prop:%i, received devId:%i\n", prop.location.id, device_id);
          // fflush(stdout);

          *devPtr = g_allocator->malloc_ipc(allocate_handle, ipc_size, device_id, ipc_offset);
          // printf("cuMemImportFromShareableHandle..1, rslt:%i, *devPtr:%p, flags:%i\n", rslt, *devPtr, flags);
        }
        else // replay workflow
        {
          // TODO:  get a new handle and then mapped to a specified address
          ipc_handle = recvIpcHandle(&servArgs);
          size_t ipc_size = servArgs.bytes;
          size_t ipc_offset = servArgs.offset;
          // int device_id = servArgs.deviceId;


          CUmemGenericAllocationHandle allocate_handle;
          /*CUresult rslt = */cuMemImportFromShareableHandle(&allocate_handle, (void*)(uintptr_t)ipc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

          CUmemAllocationProp prop;
          /*rslt = */cuMemGetAllocationPropertiesFromHandle(&prop, allocate_handle);
          // printf("devId in prop:%i, received devId:%i\n", prop.location.id, device_id);
          g_allocator->malloc_restart(*devPtr, ipc_size, prop.location.id, true);
          if (g_allocator->ptr_block_map.count(*devPtr))
          {
            // printf("find new blk of cudaIpcOpenMemHandle in restoration...\n");
            // fflush(stdout);
            Block* blk = g_allocator->ptr_block_map[*devPtr];

            /*rslt = */cuMemMap((CUdeviceptr)(blk->ptr), ipc_size, 0ULL, allocate_handle, 0ULL);
            // printf("cuMemMap, ret:%i\n", rslt);
            int deviceCnt = 1;
            cuDeviceGetCount(&deviceCnt);

            CUmemAccessDesc accessDesc;
            accessDesc.location = prop.location;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            for (int i = 0; i < deviceCnt; i++)
            {
              accessDesc.location.id = i;
              /*rslt = */cuMemSetAccess((CUdeviceptr)(blk->ptr), ipc_size, &accessDesc, 1);
              // printf("cuMemSetAccess, ret:%i\n", rslt);
            }

            g_allocator->handle_size_map[allocate_handle] = ipc_size;
            g_allocator->block_handle_map[blk] = allocate_handle;

            if (ipc_offset == 0)
            {
              g_allocator->ipc_ptr_map[blk->ptr] = blk->ptr;
            }
            else
            {
                void* actual_ptr = (void*) ((size_t)blk->ptr + ipc_offset);
                g_allocator->ipc_ptr_map[actual_ptr] = blk->ptr;
            }
          }
        }

        *orig_ret = cudaSuccess;
        break;
      }
      case GENERATE_ENUM(cudaIpcCloseMemHandle):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void* devPtr = va_arg(arglist, void*);
        // *orig_ret = cudaIpcCloseMemHandle(devPtr);
        *orig_ret = cudaSuccess;
        g_allocator->free_ipc(devPtr);
        break;
      }
      case GENERATE_ENUM(cudaDriverGetVersion):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        int* driverVersion = va_arg(arglist, int*);
        *orig_ret = cudaDriverGetVersion(driverVersion);
        break;
      }
      case GENERATE_ENUM(cudaDeviceGetByPCIBusId):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        int* device = va_arg(arglist, int*);
        const char* pciBusId = va_arg(arglist, char*);
        *orig_ret = cudaDeviceGetByPCIBusId(device, pciBusId);
        break;
      }
      case GENERATE_ENUM(cudaStreamIsCapturing):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t stream = va_arg(arglist, cudaStream_t); 
        enum cudaStreamCaptureStatus * pCaptureStatus = va_arg(arglist, enum cudaStreamCaptureStatus *);
        *orig_ret = cudaStreamIsCapturing(stream, pCaptureStatus);
        break;
      }
      case GENERATE_ENUM(cudaHostUnregister):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        void* ptr = va_arg(arglist, void*);
        *orig_ret = cudaHostUnregister(ptr);
        break;
      }
      case GENERATE_ENUM(cudaGraphAddEventWaitNode):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaGraphNode_t* pGraphNode = va_arg(arglist, cudaGraphNode_t*);
        cudaGraph_t graph = va_arg(arglist, cudaGraph_t);
        const cudaGraphNode_t* pDependencies = va_arg(arglist, cudaGraphNode_t*);
        size_t numDependencies = va_arg(arglist, size_t);
        cudaEvent_t event = va_arg(arglist, cudaEvent_t);
        *orig_ret = cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event);
        break;
      }
      case GENERATE_ENUM(cudaGraphAddEventRecordNode):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaGraphNode_t* pGraphNode = va_arg(arglist, cudaGraphNode_t*);
        cudaGraph_t graph = va_arg(arglist, cudaGraph_t);
        const cudaGraphNode_t* pDependencies = va_arg(arglist, cudaGraphNode_t*);
        size_t numDependencies = va_arg(arglist, size_t);
        cudaEvent_t event = va_arg(arglist, cudaEvent_t);
        *orig_ret = cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event);
        break;
      }
      case GENERATE_ENUM(cudaLaunchHostFunc):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        cudaHostFn_t fn = va_arg(arglist, cudaHostFn_t);
        void* userData = va_arg(arglist, void*);
        *orig_ret = cudaLaunchHostFunc(stream, fn, userData);
        break;
      }
      case GENERATE_ENUM(cudaGraphAddHostNode):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaGraphNode_t* pGraphNode = va_arg(arglist, cudaGraphNode_t*);
        cudaGraph_t graph = va_arg(arglist, cudaGraph_t);
        const cudaGraphNode_t* pDependencies = va_arg(arglist, cudaGraphNode_t*);
        size_t numDependencies = va_arg(arglist, size_t);
        const cudaHostNodeParams* pNodeParams = va_arg(arglist, cudaHostNodeParams*);
        *orig_ret = cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
        break;
      }
      case GENERATE_ENUM(cudaDeviceEnablePeerAccess):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        int peerDevice = va_arg(arglist, int);
        unsigned int flags = va_arg(arglist, unsigned int);
        *orig_ret = cudaDeviceEnablePeerAccess(peerDevice, flags);
        break;
      }
      case GENERATE_ENUM(cudaGraphAddKernelNode):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        cudaGraphNode_t* pGraphNode = va_arg(arglist, cudaGraphNode_t*);
        cudaGraph_t graph = va_arg(arglist, cudaGraph_t);
        const cudaGraphNode_t* pDependencies = va_arg(arglist, cudaGraphNode_t*);
        size_t numDependencies = va_arg(arglist, size_t);
        const cudaKernelNodeParams* pNodeParams = va_arg(arglist, cudaKernelNodeParams*);
        *orig_ret = cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
        break;
      }
      case GENERATE_ENUM(cuGetErrorName):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        int error = va_arg(arglist, int);
        const char** pStr = va_arg(arglist, const char**);
        *orig_ret = cuGetErrorName((CUresult)error, pStr);
        break;
      }
      case GENERATE_ENUM(__cudaInitModule):
      {
        char* orig_ret = va_arg(arglist, char*);
        void **fatCubinHandle = va_arg(arglist, void**);
        *orig_ret = __cudaInitModule(fatCubinHandle);
        break;
      }
      case GENERATE_ENUM(__cudaPopCallConfiguration):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        dim3 *gridDim = va_arg(arglist, dim3*);
        dim3 *blockDim = va_arg(arglist, dim3*);
        size_t *sharedMem = va_arg(arglist, size_t*);
        void *stream = va_arg(arglist, void*);
        *orig_ret = __cudaPopCallConfiguration(gridDim, blockDim, sharedMem, stream);
        break;
      }
      case GENERATE_ENUM(__cudaPushCallConfiguration):
      {
        unsigned int* orig_ret = va_arg(arglist, unsigned int*);
        dim3 gridDim = va_arg(arglist, dim3);
        dim3 blockDim = va_arg(arglist, dim3);
        size_t sharedMem = va_arg(arglist, size_t);
        void *stream = va_arg(arglist, void*);
        typedef unsigned int (*pushFptr_t)(dim3 gridDim, dim3 blockDim, size_t sharedMem, void * stream);
        pushFptr_t func = (pushFptr_t)lhDlsym(GENERATE_ENUM(__cudaPushCallConfiguration));
        *orig_ret = func(gridDim, blockDim, sharedMem, stream);
        break;
      }
      case GENERATE_ENUM(__cudaRegisterFatBinary):
      {
        void*** orig_ret = va_arg(arglist, void***);
        void *fatCubin = va_arg(arglist, void*);
        void **ret = __cudaRegisterFatBinary(fatCubin);
        memcpy(orig_ret, &ret, sizeof(void**));
        // printf("ret value:%p\n", ret);
        break;
      }
      case GENERATE_ENUM(__cudaUnregisterFatBinary):
      {
        void **fatCubinHandle = va_arg(arglist, void **);
         __cudaUnregisterFatBinary(fatCubinHandle);
        break;
      }
      case GENERATE_ENUM(__cudaRegisterFunction):
      {
        void **fatCubinHandle = va_arg(arglist, void **);
        const char *hostFun =  va_arg(arglist, char *);
        char *deviceFun = va_arg(arglist, char*);
        const char *deviceName  = va_arg(arglist, char*);
        int thread_limit =  va_arg(arglist, int);
        uint3 *tid = va_arg(arglist, uint3*);
        uint3 *bid = va_arg(arglist, uint3*);
        dim3 *bDim = va_arg(arglist, dim3*);
        dim3 *gDim = va_arg(arglist, dim3*);
        int *wSize = va_arg(arglist, int*);
        __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
        break;
      }
      case GENERATE_ENUM(__cudaRegisterManagedVar):
      {
        void **fatCubinHandle = va_arg(arglist, void **);
        void **hostVarPtrAddress = va_arg(arglist, void**);
        char  *deviceAddress = va_arg(arglist, char*);
        const char  *deviceName = va_arg(arglist, char*);
        int    ext = va_arg(arglist, int);
        size_t size = va_arg(arglist, size_t);
        int    constant = va_arg(arglist, int);
        int    global = va_arg(arglist, int);
        __cudaRegisterManagedVar(fatCubinHandle, hostVarPtrAddress, deviceAddress, deviceName, ext, size, constant, global);
        break;
      }
      case GENERATE_ENUM(__cudaRegisterTexture):
      {
        void **fatCubinHandle = va_arg(arglist, void **);
        const struct textureReference  *hostVar = va_arg(arglist, struct textureReference  *);
        const void **deviceAddress = va_arg(arglist, const void**);
        const char *deviceName = va_arg(arglist, char*);
        int dim = va_arg(arglist, int);
        int norm = va_arg(arglist, int);
        int ext = va_arg(arglist, int);
        __cudaRegisterTexture(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext);
        break;
      }
      case GENERATE_ENUM(__cudaRegisterSurface):
      {
        void **fatCubinHandle = va_arg(arglist, void **);
        const struct surfaceReference  *hostVar = va_arg(arglist, struct surfaceReference  *);
        const void **deviceAddress = va_arg(arglist, const void**);
        const char *deviceName = va_arg(arglist, char*);
        int dim = va_arg(arglist, int);
         int ext = va_arg(arglist, int);
        __cudaRegisterSurface(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext);
        break;
      }
      case GENERATE_ENUM(__cudaRegisterVar):
      {
        void **fatCubinHandle = va_arg(arglist, void **);
        char *hostVar = va_arg(arglist, char*);
        char  *deviceAddress = va_arg(arglist, char*);
        const char  *deviceName = va_arg(arglist, char*);
        int ext = va_arg(arglist, int);
        size_t size = va_arg(arglist, size_t);
        int constant = va_arg(arglist, int);
        int global = va_arg(arglist, int);
        __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
        break;
      }
      case GENERATE_ENUM(__cudaRegisterFatBinaryEnd):
      {
        void **fatCubinHandle = va_arg(arglist, void **);
        __cudaRegisterFatBinaryEnd(fatCubinHandle);
        break;
      }
      case GENERATE_ENUM(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        int* numBlocks = va_arg(arglist, int*);
        const void* func = va_arg(arglist, void*);
        int  blockSize = va_arg(arglist, int);
        size_t dynamicSMemSize = va_arg(arglist, size_t);
        unsigned int  flags = va_arg(arglist, unsigned int);
        *orig_ret = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
        break;
      }
      case GENERATE_ENUM(cudaFuncGetAttributes):
      {
        cudaError_t* orig_ret = va_arg(arglist, cudaError_t*);
        struct cudaFuncAttributes *attr = va_arg(arglist, struct cudaFuncAttributes*);
        const void *func = va_arg(arglist, void*);
        *orig_ret = cudaFuncGetAttributes(attr, func);
        break;
      }
      case GENERATE_ENUM(cuInit):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        *orig_ret = cuInit(0);
        break;
      }
      case GENERATE_ENUM(cuDriverGetVersion):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        int* driverVersion = va_arg(arglist, int*);
        *orig_ret = cuDriverGetVersion(driverVersion);

        break;
      }
      case GENERATE_ENUM(cuDeviceGet):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUdevice* device = va_arg(arglist, CUdevice*);
        int ordinal = va_arg(arglist, int);
        *orig_ret = cuDeviceGet(device, ordinal);
        break;
      }
      case GENERATE_ENUM(cuDeviceGetAttribute):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        int* pi = va_arg(arglist, int*);
        int attrib = va_arg(arglist, int);
        CUdevice dev = va_arg(arglist, CUdevice);
        *orig_ret = cuDeviceGetAttribute(pi, (CUdevice_attribute)attrib, dev);
        break;
      }
      case GENERATE_ENUM(cuDeviceGetCount):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        int* count = va_arg(arglist, int*);

        *orig_ret = cuDeviceGetCount(count);
        break;
      }
      case GENERATE_ENUM(cuDeviceGetName):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        char* name = va_arg(arglist, char*);
        int  len = va_arg(arglist, int);  
        CUdevice dev = va_arg(arglist, CUdevice);

        *orig_ret = cuDeviceGetName(name, len, dev);
        break;
      }
      case GENERATE_ENUM(cuDeviceGetUuid):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUuuid* uuid = va_arg(arglist, CUuuid*);
        CUdevice dev = va_arg(arglist, CUdevice);
   
        *orig_ret = cuDeviceGetUuid(uuid, dev);
        break;
      }
      case GENERATE_ENUM(cuDeviceTotalMem_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        size_t* bytes = va_arg(arglist, size_t*); 
        CUdevice dev = va_arg(arglist, CUdevice);
 
        *orig_ret = cuDeviceTotalMem_v2(bytes, dev);
        break;
      }
      case GENERATE_ENUM(cuDeviceComputeCapability):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        int* major = va_arg(arglist, int*);
        int* minor = va_arg(arglist, int*); 
        CUdevice dev = va_arg(arglist, CUdevice);

        *orig_ret = cuDeviceComputeCapability(major, minor, dev);
        break;
      }
      case GENERATE_ENUM(cuDeviceGetProperties):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUdevprop* prop = va_arg(arglist, CUdevprop*);
        CUdevice dev = va_arg(arglist, CUdevice);

        *orig_ret = cuDeviceGetProperties(prop, dev);
        break;
      }
      case GENERATE_ENUM(cuDevicePrimaryCtxGetState):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUdevice dev = va_arg(arglist, CUdevice);
        unsigned int* flags = va_arg(arglist, unsigned int*);
        int* active = va_arg(arglist, int*);

        *orig_ret = cuDevicePrimaryCtxGetState(dev, flags, active);
        break;
      }
      case GENERATE_ENUM(cuDevicePrimaryCtxRetain):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext* pctx = va_arg(arglist, CUcontext*);
        CUdevice dev = va_arg(arglist, CUdevice);

        *orig_ret = cuDevicePrimaryCtxRetain(pctx, dev);
        break;
      }
      case GENERATE_ENUM(cuCtxCreate_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext* pctx = va_arg(arglist, CUcontext*);
        unsigned int  flags = va_arg(arglist, unsigned int);
        CUdevice dev = va_arg(arglist, CUdevice);

        *orig_ret = cuCtxCreate_v2(pctx, flags, dev);
        break;
      }
      case GENERATE_ENUM(cuCtxDestroy_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext ctx = va_arg(arglist, CUcontext);

        *orig_ret = cuCtxDestroy_v2(ctx);
        break;
      }
      case GENERATE_ENUM(cuCtxGetApiVersion):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext ctx = va_arg(arglist, CUcontext);
        unsigned int* version = va_arg(arglist, unsigned int*);

        *orig_ret = cuCtxGetApiVersion(ctx, version);
        break;
      }
      case GENERATE_ENUM(cuCtxGetCacheConfig):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUfunc_cache* pconfig = va_arg(arglist, CUfunc_cache*);

        *orig_ret = cuCtxGetCacheConfig(pconfig);
        break;
      }
      case GENERATE_ENUM(cuCtxGetCurrent):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext* ctx = va_arg(arglist, CUcontext*);

        *orig_ret = cuCtxGetCurrent(ctx);
        break;
      }
      case GENERATE_ENUM(cuCtxGetDevice):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUdevice* device = va_arg(arglist, CUdevice*);

        *orig_ret = cuCtxGetDevice(device);
        break;
      }
      case GENERATE_ENUM(cuCtxGetFlags):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        unsigned int* flags = va_arg(arglist, unsigned int*);

        *orig_ret = cuCtxGetFlags(flags);
        break;
      }
      case GENERATE_ENUM(cuCtxGetLimit):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        size_t* pvalue = va_arg(arglist, size_t*);
        int limit = va_arg(arglist, int);

        *orig_ret = cuCtxGetLimit(pvalue, (CUlimit)limit);
        break;
      }
      case GENERATE_ENUM(cuCtxGetSharedMemConfig):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUsharedconfig* pConfig = va_arg(arglist, CUsharedconfig*);

        *orig_ret = cuCtxGetSharedMemConfig(pConfig);
        break;
      }
      case GENERATE_ENUM(cuCtxGetStreamPriorityRange):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        int* leastPriority = va_arg(arglist, int*);
        int* greatestPriority = va_arg(arglist, int*);

        *orig_ret = cuCtxGetStreamPriorityRange(leastPriority, greatestPriority);
        break;
      }
      case GENERATE_ENUM(cuCtxPopCurrent_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext* pctx = va_arg(arglist, CUcontext*);
        
        *orig_ret = cuCtxPopCurrent_v2(pctx);
        break;
      }
      case GENERATE_ENUM(cuCtxPushCurrent_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext pctx = va_arg(arglist, CUcontext); 
        
        *orig_ret = cuCtxPushCurrent_v2(pctx);
        break;
      }
      case GENERATE_ENUM(cuCtxSetCacheConfig):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        int config = va_arg(arglist, int); 
        
        *orig_ret = cuCtxSetCacheConfig((CUfunc_cache)config);
        break;
      }
      case GENERATE_ENUM(cuCtxSetCurrent):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext ctx = va_arg(arglist, CUcontext); 
        
        *orig_ret = cuCtxSetCurrent(ctx);
        break;
      }
      case GENERATE_ENUM(cuCtxSetLimit):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        int limit = va_arg(arglist, int); 
        size_t value = va_arg(arglist, size_t); 
        
        *orig_ret = cuCtxSetLimit((CUlimit)limit, value);
        break;
      }
      case GENERATE_ENUM(cuCtxSetSharedMemConfig):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        int config = va_arg(arglist, int); 
        
        *orig_ret = cuCtxSetSharedMemConfig((CUsharedconfig)config);
        break;
      }
      case GENERATE_ENUM(cuCtxSynchronize):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        
        *orig_ret = cuCtxSynchronize();
        break;
      }
      case GENERATE_ENUM(cuCtxAttach):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext* pctx = va_arg(arglist, CUcontext*); 
        unsigned int  flags = va_arg(arglist, unsigned int); 
        
        *orig_ret = cuCtxAttach(pctx, flags);
        break;
      }
      case GENERATE_ENUM(cuCtxDetach):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUcontext pctx = va_arg(arglist, CUcontext); 
        
        *orig_ret = cuCtxDetach(pctx);
        break;
      }
      case GENERATE_ENUM(cuLinkAddData_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUlinkState state = va_arg(arglist, CUlinkState); 
        int type = va_arg(arglist, int); 
        void* data = va_arg(arglist, void*); 
        size_t size = va_arg(arglist, size_t); 
        const char* name = va_arg(arglist, const char*); 
        unsigned int  numOptions = va_arg(arglist, unsigned int); 
        CUjit_option* options = va_arg(arglist, CUjit_option*); 
        void** optionValues = va_arg(arglist, void**); 
        
        *orig_ret = cuLinkAddData_v2(state, (CUjitInputType)type, data, size, name, numOptions, options, optionValues);
        break;
      }
      case GENERATE_ENUM(cuLinkAddFile_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUlinkState state = va_arg(arglist, CUlinkState); 
        int type = va_arg(arglist, int); 
        const char* path = va_arg(arglist, const char*); 
        unsigned int  numOptions = va_arg(arglist, unsigned int); 
        CUjit_option* options = va_arg(arglist, CUjit_option*); 
        void** optionValues = va_arg(arglist, void**); 
        
        *orig_ret = cuLinkAddFile_v2(state, (CUjitInputType)type, path, numOptions, options, optionValues);
        break;
      }
      case GENERATE_ENUM(cuLinkComplete):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUlinkState state = va_arg(arglist, CUlinkState); 
        void** cubinOut = va_arg(arglist, void**);  
        size_t* sizeOut = va_arg(arglist, size_t*); 
        
        *orig_ret = cuLinkComplete(state, cubinOut, sizeOut);
        break;
      }
      case GENERATE_ENUM(cuLinkCreate_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        unsigned int  numOptions = va_arg(arglist, unsigned int);    
        CUjit_option* options = va_arg(arglist, CUjit_option*);   
        void** optionValues = va_arg(arglist, void**);  
        CUlinkState* stateOut = va_arg(arglist, CUlinkState*); 
        
        *orig_ret = cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
        break;
      }
      case GENERATE_ENUM(cuLinkDestroy):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUlinkState state = va_arg(arglist, CUlinkState); 
        
        *orig_ret = cuLinkDestroy(state);
        break;
      }
      case GENERATE_ENUM(cuModuleGetFunction):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUfunction* hfunc = va_arg(arglist, CUfunction*);  
        CUmodule hmod = va_arg(arglist, CUmodule); 
        const char* name = va_arg(arglist, const char*); 
        
        *orig_ret = cuModuleGetFunction(hfunc, hmod, name);
        break;
      }
      case GENERATE_ENUM(cuModuleGetGlobal_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr* dptr = va_arg(arglist, CUdeviceptr*);  
        size_t* bytes = va_arg(arglist, size_t*);  
        CUmodule hmod = va_arg(arglist, CUmodule); 
        const char* name = va_arg(arglist, const char*); 
        
        *orig_ret = cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
        break;
      }
      case GENERATE_ENUM(cuModuleGetSurfRef):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUsurfref* pSurfRef = va_arg(arglist, CUsurfref*);  
        CUmodule hmod = va_arg(arglist, CUmodule); 
        const char* name = va_arg(arglist, const char*); 
        
        *orig_ret = cuModuleGetSurfRef(pSurfRef, hmod, name);
        break;
      }
      case GENERATE_ENUM(cuModuleGetTexRef):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUtexref* pTexRef = va_arg(arglist, CUtexref*);  
        CUmodule hmod = va_arg(arglist, CUmodule); 
        const char* name = va_arg(arglist, const char*); 
        
        *orig_ret = cuModuleGetTexRef(pTexRef, hmod, name);
        break;
      }
      case GENERATE_ENUM(cuModuleLoad):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUmodule* module = va_arg(arglist, CUmodule*); 
        const char* fname = va_arg(arglist, const char*); 
        
        *orig_ret = cuModuleLoad(module, fname);
        break;
      }
      case GENERATE_ENUM(cuModuleLoadData):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUmodule* module = va_arg(arglist, CUmodule*); 
        const void* image = va_arg(arglist, const void*); 
        
        *orig_ret = cuModuleLoadData(module, image);
        break;
      }
      case GENERATE_ENUM(cuModuleLoadDataEx):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUmodule* module = va_arg(arglist, CUmodule*); 
        const void* image = va_arg(arglist, const void*);
        unsigned int  numOptions = va_arg(arglist, unsigned int); 
        CUjit_option* options = va_arg(arglist, CUjit_option*); 
        void** optionValues = va_arg(arglist, void**);
        
        *orig_ret = cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
        break;
      }
      case GENERATE_ENUM(cuModuleLoadFatBinary):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUmodule* module = va_arg(arglist, CUmodule*); 
        const void* fatCubin = va_arg(arglist, const void*);
        
        *orig_ret = cuModuleLoadFatBinary(module, fatCubin);
        break;
      }
      case GENERATE_ENUM(cuModuleUnload):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUmodule module = va_arg(arglist, CUmodule); 
        
        *orig_ret = cuModuleUnload(module);
        break;
      }
      case GENERATE_ENUM(cuArray3DCreate_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUarray* pHandle = va_arg(arglist, CUarray*); 
        const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray = va_arg(arglist, const CUDA_ARRAY3D_DESCRIPTOR*); 
        
        *orig_ret = cuArray3DCreate_v2(pHandle, pAllocateArray);
        break;
      }
      case GENERATE_ENUM(cuArray3DGetDescriptor_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor = va_arg(arglist, CUDA_ARRAY3D_DESCRIPTOR*); 
        CUarray hArray = va_arg(arglist, CUarray); 
        
        *orig_ret = cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray);
        break;
      }
      case GENERATE_ENUM(cuArrayCreate_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUarray* pHandle = va_arg(arglist, CUarray*);
        const CUDA_ARRAY_DESCRIPTOR* pAllocateArray = va_arg(arglist, const CUDA_ARRAY_DESCRIPTOR*); 
        
        *orig_ret = cuArrayCreate_v2(pHandle, pAllocateArray);
        break;
      }
      case GENERATE_ENUM(cuArrayDestroy):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUarray hArray = va_arg(arglist, CUarray);
        
        *orig_ret = cuArrayDestroy(hArray);
        break;
      }
      case GENERATE_ENUM(cuArrayGetDescriptor_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor = va_arg(arglist, CUDA_ARRAY_DESCRIPTOR*); 
        CUarray hArray = va_arg(arglist, CUarray);
        
        *orig_ret = cuArrayGetDescriptor_v2(pArrayDescriptor, hArray);
        break;
      }
      case GENERATE_ENUM(cuDeviceGetByPCIBusId):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUdevice* dev = va_arg(arglist, CUdevice*); 
        const char* pciBusId = va_arg(arglist, const char*);
        
        *orig_ret = cuDeviceGetByPCIBusId(dev, pciBusId);
        break;
      }
      case GENERATE_ENUM(cuDeviceGetPCIBusId):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        char* pciBusId = va_arg(arglist, char*); 
        int  len = va_arg(arglist, int); 
        CUdevice dev = va_arg(arglist, CUdevice);
        
        *orig_ret = cuDeviceGetPCIBusId(pciBusId, len, dev);
        break;
      }
      case GENERATE_ENUM(cuIpcCloseMemHandle):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUdeviceptr dptr = va_arg(arglist, CUdeviceptr);
        
        *orig_ret = cuIpcCloseMemHandle(dptr);
        break;
      }
      case GENERATE_ENUM(cuIpcGetEventHandle):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUipcEventHandle* pHandle = va_arg(arglist, CUipcEventHandle*); 
        CUevent event = va_arg(arglist, CUevent);
        
        *orig_ret = cuIpcGetEventHandle(pHandle, event);
        break;
      }
      case GENERATE_ENUM(cuIpcGetMemHandle):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUipcMemHandle* pHandle = va_arg(arglist, CUipcMemHandle*); 
        CUdeviceptr dptr = va_arg(arglist, CUdeviceptr);
        
        *orig_ret = cuIpcGetMemHandle(pHandle, dptr);
        break;
      }
      case GENERATE_ENUM(cuIpcOpenEventHandle):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUevent* phEvent = va_arg(arglist, CUevent*); 
        CUipcEventHandle handle = va_arg(arglist, CUipcEventHandle);
        
        *orig_ret = cuIpcOpenEventHandle(phEvent, handle);
        break;
      }
      case GENERATE_ENUM(cuMemAlloc_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUdeviceptr* dptr = va_arg(arglist, CUdeviceptr*); 
        size_t bytesize = va_arg(arglist, size_t);
        
        *dptr = (CUdeviceptr)allocate_gpu_mem(bytesize);
        *orig_ret = CUDA_SUCCESS;
        break;
      }
      case GENERATE_ENUM(cuMemAllocHost_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);  
        void ** pp = va_arg(arglist, void **); 
        size_t bytesize = va_arg(arglist, size_t);
        int ret = allocate_page_lock_mem(pp, bytesize, 0x01);
	      // cudaHostRegister(*pp, bytesize, 0x01);
        if (ret)
          *orig_ret = CUDA_ERROR_UNKNOWN;
        else
          *orig_ret = CUDA_SUCCESS;
        break;
      }
      case GENERATE_ENUM(cuMemAllocManaged):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);  
        CUdeviceptr* dptr = va_arg(arglist, CUdeviceptr*); 
        size_t bytesize = va_arg(arglist, size_t);   
        unsigned int  flags = va_arg(arglist, unsigned int);
        *orig_ret = cuMemAllocManaged(dptr, bytesize, flags); 
        break;
      }
      case GENERATE_ENUM(cuMemAllocPitch_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);  
        CUdeviceptr* dptr = va_arg(arglist, CUdeviceptr*); 
        size_t* pPitch = va_arg(arglist, size_t*); 
        size_t WidthInBytes = va_arg(arglist, size_t); 
        size_t Height = va_arg(arglist, size_t); 
        unsigned int  ElementSizeBytes = va_arg(arglist, unsigned int);
        *orig_ret = cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
        break;
      }
      case GENERATE_ENUM(cuMemFree_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUdeviceptr dptr = va_arg(arglist, CUdeviceptr);
        free_gpu_mem((void*)dptr);
        *orig_ret = CUDA_SUCCESS;
        break;
      }
      case GENERATE_ENUM(cuMemFreeHost):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        void* p = va_arg(arglist, void*);
        free_page_lock_mem(p);
        *orig_ret = CUDA_SUCCESS;
        break;
      }
      case GENERATE_ENUM(cuMemGetAddressRange_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);   
        CUdeviceptr* pbase = va_arg(arglist, CUdeviceptr*); 
        size_t* psize = va_arg(arglist, size_t*); 
        CUdeviceptr dptr = va_arg(arglist, CUdeviceptr);
        *orig_ret = cuMemGetAddressRange_v2(pbase, psize, dptr);
        break;
      }
      case GENERATE_ENUM(cuMemGetInfo_v2):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        size_t* free = va_arg(arglist, size_t*);  
        size_t* total = va_arg(arglist, size_t*);
        *orig_ret = cuMemGetInfo_v2(free, total);
        break;
      }
      case GENERATE_ENUM(cuMemHostAlloc):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);  
        void ** pp = va_arg(arglist, void **); 
        size_t bytesize = va_arg(arglist, size_t);
        unsigned int  Flags = va_arg(arglist, unsigned int);
        int ret = allocate_page_lock_mem(pp, bytesize, Flags);
	      // cudaHostRegister(*pp, bytesize, Flags);
        if (ret)
          *orig_ret = CUDA_ERROR_UNKNOWN;
        else
          *orig_ret = CUDA_SUCCESS;
        break;
      }
      case GENERATE_ENUM(cuMemHostGetDevicePointer_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr* pdptr = va_arg(arglist, CUdeviceptr*); 
        void* p = va_arg(arglist, void*); 
        unsigned int  Flags = va_arg(arglist, unsigned int);
        *orig_ret = cuMemHostGetDevicePointer_v2(pdptr, p, Flags);
        break;
      }
      case GENERATE_ENUM(cuMemHostGetFlags):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        unsigned int* pFlags = va_arg(arglist, unsigned int*); 
        void* p = va_arg(arglist, void*);
        *orig_ret = cuMemHostGetFlags(pFlags, p);
        break;
      }
      case GENERATE_ENUM(cuMemHostRegister_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        void* p = va_arg(arglist, void*);  
        size_t bytesize = va_arg(arglist, size_t);  
        unsigned int  Flags = va_arg(arglist, unsigned int); 
        *orig_ret = cuMemHostRegister_v2(p, bytesize, Flags);
        break;
      }
      case GENERATE_ENUM(cuMemHostUnregister):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        void* p = va_arg(arglist, void*);
        *orig_ret = cuMemHostUnregister(p);
        break;
      }
      case GENERATE_ENUM(cuMemcpy):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dst = va_arg(arglist, CUdeviceptr);
        CUdeviceptr src = va_arg(arglist, CUdeviceptr); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpy(dst, src, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpy2D_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        const CUDA_MEMCPY2D* pCopy = va_arg(arglist, const CUDA_MEMCPY2D*);
        *orig_ret = cuMemcpy2D_v2(pCopy);
        break;
      }
      case GENERATE_ENUM(cuMemcpy2DAsync_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        const CUDA_MEMCPY2D* pCopy = va_arg(arglist, const CUDA_MEMCPY2D*);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpy2DAsync_v2(pCopy, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemcpy2DUnaligned_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        const CUDA_MEMCPY2D* pCopy = va_arg(arglist, const CUDA_MEMCPY2D*);
        *orig_ret = cuMemcpy2DUnaligned_v2(pCopy);
        break;
      }
      case GENERATE_ENUM(cuMemcpy3D_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        const CUDA_MEMCPY3D* pCopy = va_arg(arglist, const CUDA_MEMCPY3D*);
        *orig_ret = cuMemcpy3D_v2(pCopy);
        break;
      }
      case GENERATE_ENUM(cuMemcpy3DAsync_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        const CUDA_MEMCPY3D* pCopy = va_arg(arglist, const CUDA_MEMCPY3D*);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpy3DAsync_v2(pCopy, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemcpy3DPeer):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        const CUDA_MEMCPY3D_PEER* pCopy = va_arg(arglist, const CUDA_MEMCPY3D_PEER*);
        *orig_ret = cuMemcpy3DPeer(pCopy);
        break;
      }
      case GENERATE_ENUM(cuMemcpy3DPeerAsync):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        const CUDA_MEMCPY3D_PEER* pCopy = va_arg(arglist, const CUDA_MEMCPY3D_PEER*);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpy3DPeerAsync(pCopy, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemcpyAsync):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dst = va_arg(arglist, CUdeviceptr);
        CUdeviceptr src = va_arg(arglist, CUdeviceptr); 
        size_t ByteCount = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpyAsync(dst, src, ByteCount, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemcpyAtoA_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUarray dstArray = va_arg(arglist, CUarray); 
        size_t dstOffset = va_arg(arglist, size_t); 
        CUarray srcArray = va_arg(arglist, CUarray); 
        size_t srcOffset = va_arg(arglist, size_t); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpyAtoD_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        CUarray srcArray = va_arg(arglist, CUarray); 
        size_t srcOffset = va_arg(arglist, size_t); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpyAtoH_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        void* dstHost = va_arg(arglist, void*); 
        CUarray srcArray = va_arg(arglist, CUarray); 
        size_t srcOffset = va_arg(arglist, size_t); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpyAtoHAsync_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        void* dstHost = va_arg(arglist, void*); 
        CUarray srcArray = va_arg(arglist, CUarray); 
        size_t srcOffset = va_arg(arglist, size_t); 
        size_t ByteCount = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemcpyDtoA_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUarray dstArray = va_arg(arglist, CUarray); 
        size_t dstOffset = va_arg(arglist, size_t); 
        CUdeviceptr srcDevice = va_arg(arglist, CUdeviceptr); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpyDtoD_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        CUdeviceptr srcDevice = va_arg(arglist, CUdeviceptr); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpyDtoDAsync_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        CUdeviceptr srcDevice = va_arg(arglist, CUdeviceptr); 
        size_t ByteCount = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemcpyDtoH_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        void* dstHost = va_arg(arglist, void*); 
        CUdeviceptr srcDevice = va_arg(arglist, CUdeviceptr); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpyDtoHAsync_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        void* dstHost = va_arg(arglist, void*); 
        CUdeviceptr srcDevice = va_arg(arglist, CUdeviceptr); 
        size_t ByteCount = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemcpyHtoA_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUarray dstArray = va_arg(arglist, CUarray);
        size_t dstOffset = va_arg(arglist, size_t); 
        const void* srcHost = va_arg(arglist, const void*); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpyHtoAAsync_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUarray dstArray = va_arg(arglist, CUarray);
        size_t dstOffset = va_arg(arglist, size_t); 
        const void* srcHost = va_arg(arglist, const void*); 
        size_t ByteCount = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemcpyHtoD_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        const void* srcHost = va_arg(arglist, const void*); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpyHtoDAsync_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        const void* srcHost = va_arg(arglist, const void*); 
        size_t ByteCount = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemcpyPeer):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        CUcontext dstContext = va_arg(arglist, CUcontext); 
        CUdeviceptr srcDevice = va_arg(arglist, CUdeviceptr); 
        CUcontext srcContext = va_arg(arglist, CUcontext); 
        size_t ByteCount = va_arg(arglist, size_t);
        *orig_ret = cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
        break;
      }
      case GENERATE_ENUM(cuMemcpyPeerAsync):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        CUcontext dstContext = va_arg(arglist, CUcontext); 
        CUdeviceptr srcDevice = va_arg(arglist, CUdeviceptr); 
        CUcontext srcContext = va_arg(arglist, CUcontext); 
        size_t ByteCount = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemsetD16_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        unsigned short us = va_arg(arglist, int); 
        size_t N = va_arg(arglist, size_t);
        *orig_ret = cuMemsetD16_v2(dstDevice, us, N);
        break;
      }
      case GENERATE_ENUM(cuMemsetD16Async):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        unsigned short us = va_arg(arglist, int); 
        size_t N = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemsetD16Async(dstDevice, us, N, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemsetD2D16_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        size_t dstPitch = va_arg(arglist, size_t); 
        unsigned short us = va_arg(arglist, int); 
        size_t Width = va_arg(arglist, size_t); 
        size_t Height = va_arg(arglist, size_t);
        *orig_ret = cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);
        break;
      }
      case GENERATE_ENUM(cuMemsetD2D16Async):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        size_t dstPitch = va_arg(arglist, size_t); 
        unsigned short us = va_arg(arglist, int); 
        size_t Width = va_arg(arglist, size_t); 
        size_t Height = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemsetD2D32_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        size_t dstPitch = va_arg(arglist, size_t); 
        unsigned int us = va_arg(arglist, unsigned int); 
        size_t Width = va_arg(arglist, size_t); 
        size_t Height = va_arg(arglist, size_t);
        *orig_ret = cuMemsetD2D32_v2(dstDevice, dstPitch, us, Width, Height);
        break;
      }
      case GENERATE_ENUM(cuMemsetD2D32Async):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        size_t dstPitch = va_arg(arglist, size_t); 
        unsigned int us = va_arg(arglist, unsigned int); 
        size_t Width = va_arg(arglist, size_t); 
        size_t Height = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemsetD2D32Async(dstDevice, dstPitch, us, Width, Height, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemsetD2D8_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        size_t dstPitch = va_arg(arglist, size_t); 
        unsigned char us = va_arg(arglist, unsigned int); 
        size_t Width = va_arg(arglist, size_t); 
        size_t Height = va_arg(arglist, size_t);
        *orig_ret = cuMemsetD2D8_v2(dstDevice, dstPitch, us, Width, Height);
        break;
      }
      case GENERATE_ENUM(cuMemsetD2D8Async):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        size_t dstPitch = va_arg(arglist, size_t); 
        unsigned char us = va_arg(arglist, unsigned int); 
        size_t Width = va_arg(arglist, size_t); 
        size_t Height = va_arg(arglist, size_t);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemsetD2D8Async(dstDevice, dstPitch, us, Width, Height, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemsetD32_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        unsigned int  ui = va_arg(arglist, unsigned int); 
        size_t N = va_arg(arglist, size_t); 
        *orig_ret = cuMemsetD32_v2(dstDevice, ui, N);
        break;
      }
      case GENERATE_ENUM(cuMemsetD32Async):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        unsigned int  ui = va_arg(arglist, unsigned int); 
        size_t N = va_arg(arglist, size_t); 
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemsetD32Async(dstDevice, ui, N, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemsetD8_v2):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        unsigned char  ui = va_arg(arglist, unsigned int); 
        size_t N = va_arg(arglist, size_t); 
        *orig_ret = cuMemsetD8_v2(dstDevice, ui, N);
        break;
      }
      case GENERATE_ENUM(cuMemsetD8Async):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUdeviceptr dstDevice = va_arg(arglist, CUdeviceptr); 
        unsigned char  ui = va_arg(arglist, unsigned int); 
        size_t N = va_arg(arglist, size_t); 
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuMemsetD8Async(dstDevice, ui, N, hStream);
        break;
      }
      case GENERATE_ENUM(cuMipmappedArrayCreate):
      {
        CUresult *ret = va_arg(arglist, CUresult*);
        CUmipmappedArray* pHandle = va_arg(arglist, CUmipmappedArray*); 
        const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc = va_arg(arglist, const CUDA_ARRAY3D_DESCRIPTOR*); 
        unsigned int  numMipmapLevels = va_arg(arglist, unsigned int);
        *ret = cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels);
        break;
      }
      case GENERATE_ENUM(cuMipmappedArrayDestroy):
      {
        CUresult *ret = va_arg(arglist, CUresult*);
        CUmipmappedArray hMipmappedArray = va_arg(arglist, CUmipmappedArray);
        *ret = cuMipmappedArrayDestroy(hMipmappedArray);
        break;
      }
      case GENERATE_ENUM(cuMipmappedArrayGetLevel):
      {
        CUresult *ret = va_arg(arglist, CUresult*);
        CUarray* pLevelArray = va_arg(arglist, CUarray*); 
        CUmipmappedArray hMipmappedArray = va_arg(arglist, CUmipmappedArray); 
        unsigned int  level = va_arg(arglist, unsigned int);
        *ret = cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level);
        break;
      }
      case GENERATE_ENUM(cuMemAdvise):
      {
        CUresult *ret = va_arg(arglist, CUresult*);
        CUdeviceptr devPtr = va_arg(arglist, CUdeviceptr); 
        size_t count = va_arg(arglist, size_t); 
        int advice = va_arg(arglist, int); 
        CUdevice device = va_arg(arglist, CUdevice);
        *ret = cuMemAdvise(devPtr, count, (CUmem_advise)advice, device);
        break;
      }
      case GENERATE_ENUM(cuMemPrefetchAsync):
      {
        CUresult *ret = va_arg(arglist, CUresult*);
        CUdeviceptr devPtr = va_arg(arglist, CUdeviceptr); 
        size_t count = va_arg(arglist, size_t); 
        CUdevice dstDevice = va_arg(arglist, CUdevice);  
        CUstream hStream = va_arg(arglist, CUstream); 

        *ret = cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
        break;
      }
      case GENERATE_ENUM(cuMemRangeGetAttribute):
      {
        CUresult *ret = va_arg(arglist, CUresult*);
        void* data = va_arg(arglist, void*); 
        size_t dataSize = va_arg(arglist, size_t); 
        int attribute = va_arg(arglist, int); 
        CUdeviceptr devPtr = va_arg(arglist, CUdeviceptr); 
        size_t count = va_arg(arglist, size_t);
        *ret = cuMemRangeGetAttribute(data, dataSize, (CUmem_range_attribute)attribute, devPtr, count);
        break;
      }
      case GENERATE_ENUM(cuMemRangeGetAttributes):
      {
        CUresult *ret = va_arg(arglist, CUresult*);
        void** data = va_arg(arglist, void**); 
        size_t* dataSizes = va_arg(arglist, size_t*); 
        CUmem_range_attribute* attributes = va_arg(arglist, CUmem_range_attribute*); 
        size_t numAttributes = va_arg(arglist, size_t); 
        CUdeviceptr devPtr = va_arg(arglist, CUdeviceptr); 
        size_t count = va_arg(arglist, size_t);
        *ret = cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);
        break;
      }
      case GENERATE_ENUM(cuPointerGetAttribute):
      {
        CUresult *ret = va_arg(arglist, CUresult*);
        void* data = va_arg(arglist, void*); 
        int attribute = va_arg(arglist, int); 
        CUdeviceptr ptr = va_arg(arglist, CUdeviceptr);
        *ret = cuPointerGetAttribute(data, (CUpointer_attribute)attribute, ptr);
        break;
      }
      case GENERATE_ENUM(cuPointerGetAttributes):
      {
        CUresult* orig_ret = va_arg(arglist, CUresult*);
        unsigned int  numAttributes = va_arg(arglist, unsigned int);
        CUpointer_attribute* attributes = va_arg(arglist, CUpointer_attribute*); 
        void** data = va_arg(arglist, void**);
        CUdeviceptr ptr = va_arg(arglist, CUdeviceptr);  
        *orig_ret = cuPointerGetAttributes(numAttributes, attributes, data, ptr);
        break;
      }
      case GENERATE_ENUM(cuPointerSetAttribute):
      {
        CUresult* ret = va_arg(arglist, CUresult*);
        const void* value = va_arg(arglist, const void*); 
        int attribute = va_arg(arglist, int); 
        CUdeviceptr ptr = va_arg(arglist, CUdeviceptr);
        *ret = cuPointerSetAttribute(value, (CUpointer_attribute)attribute, ptr);
        break;
      }
      case GENERATE_ENUM(cuStreamAddCallback):
      {
        CUresult* ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream); 
        CUstreamCallback callback = va_arg(arglist, CUstreamCallback); 
        void* userData = va_arg(arglist, void*); 
        unsigned int  flags = va_arg(arglist, unsigned int);
        *ret = cuStreamAddCallback(hStream, callback, userData, flags);
        break;
      }
      case GENERATE_ENUM(cuStreamAttachMemAsync):
      {
        CUresult* ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream);  
        CUdeviceptr dptr = va_arg(arglist, CUdeviceptr);  
        size_t length = va_arg(arglist, size_t);  
        unsigned int  flags = va_arg(arglist, unsigned int);
        *ret = cuStreamAttachMemAsync(hStream, dptr, length, flags);
        break;
      }
      case GENERATE_ENUM(cuStreamCreate):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUstream* phStream = va_arg(arglist, CUstream*); 
        unsigned int  Flags = va_arg(arglist, unsigned int);
        *orig_ret = cuStreamCreate(phStream, Flags);
        break;
      }
      case GENERATE_ENUM(cuStreamCreateWithPriority):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUstream* phStream = va_arg(arglist, CUstream*); 
        unsigned int  Flags = va_arg(arglist, unsigned int);
        int  priority = va_arg(arglist, int);
        *orig_ret = cuStreamCreateWithPriority(phStream, Flags, priority);
        break;
      }
      case GENERATE_ENUM(cuStreamDestroy_v2):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret =  cuStreamDestroy_v2(hStream);
        break;
      }
      case GENERATE_ENUM(cuStreamEndCapture):
      {
        CUresult * ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream); 
        CUgraph* phGraph = va_arg(arglist, CUgraph*);
        *ret = cuStreamEndCapture(hStream, phGraph);
        break;
      }
      case GENERATE_ENUM(cuStreamGetCtx):
      {
        CUresult * ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream); 
        CUcontext* pctx = va_arg(arglist, CUcontext*);
        *ret = cuStreamGetCtx(hStream, pctx);
        break;
      }
      case GENERATE_ENUM(cuStreamGetFlags):
      {
        CUresult * ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream);
        unsigned int* flags = va_arg(arglist, unsigned int*);
        *ret = cuStreamGetFlags(hStream, flags);
        break;
      }
      case GENERATE_ENUM(cuStreamGetPriority):
      {
        CUresult * ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream);
        int* priority = va_arg(arglist, int*);
        *ret = cuStreamGetPriority(hStream, priority);
        break;
      }
      case GENERATE_ENUM(cuStreamIsCapturing):
      {
        CUresult * ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream);
        CUstreamCaptureStatus* captureStatus = va_arg(arglist, CUstreamCaptureStatus*);
        *ret = cuStreamIsCapturing(hStream, captureStatus);
        break;
      }
      case GENERATE_ENUM(cuStreamQuery):
      {
        CUresult * ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream);
        *ret = cuStreamQuery(hStream);
        break;
      }
      case GENERATE_ENUM(cuStreamSynchronize):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuStreamSynchronize(hStream);
        break;
      }
      case GENERATE_ENUM(cuStreamWaitEvent):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUstream hStream = va_arg(arglist, CUstream);
        CUevent hEvent = va_arg(arglist, CUevent); 
        unsigned int  Flags = va_arg(arglist, unsigned int);
        *orig_ret = cuStreamWaitEvent(hStream, hEvent, Flags);
        break;
      }
      case GENERATE_ENUM(cuEventCreate):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUevent* phEvent = va_arg(arglist, CUevent*); 
        unsigned int  Flags = va_arg(arglist, unsigned int);
        *orig_ret = cuEventCreate(phEvent, Flags);
        break;
      }
      case GENERATE_ENUM(cuEventDestroy_v2):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUevent hEvent = va_arg(arglist, CUevent);
        *orig_ret = cuEventDestroy_v2(hEvent);
        break;
      }
      case GENERATE_ENUM(cuEventElapsedTime):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        float* pMilliseconds = va_arg(arglist, float*); 
        CUevent hStart = va_arg(arglist, CUevent); 
        CUevent hEnd = va_arg(arglist, CUevent);
        *orig_ret = cuEventElapsedTime(pMilliseconds, hStart, hEnd);
        break;
      }
      case GENERATE_ENUM(cuEventQuery):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUevent hEvent = va_arg(arglist, CUevent);
        *orig_ret = cuEventQuery(hEvent);
        break;
      }
      case GENERATE_ENUM(cuEventRecord):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUevent hEvent = va_arg(arglist, CUevent);
        CUstream hStream = va_arg(arglist, CUstream);
        *orig_ret = cuEventRecord(hEvent, hStream);
        break;
      }
      case GENERATE_ENUM(cuEventSynchronize):
      {
        CUresult * orig_ret = va_arg(arglist, CUresult*);
        CUevent hEvent = va_arg(arglist, CUevent);
        *orig_ret = cuEventSynchronize(hEvent);
        break;
      }
      case GENERATE_ENUM(cuDestroyExternalMemory):
      {
        CUresult * ret = va_arg(arglist, CUresult*);
        CUexternalMemory extMem = va_arg(arglist, CUexternalMemory);
        *ret = cuDestroyExternalMemory(extMem);
        break;
      }
      case GENERATE_ENUM(cuDestroyExternalSemaphore):
      {
        CUresult * ret = va_arg(arglist, CUresult*);
        CUexternalSemaphore extSem = va_arg(arglist, CUexternalSemaphore);
        *ret = cuDestroyExternalSemaphore(extSem); 
        break;
      }
      case GENERATE_ENUM(cuExternalMemoryGetMappedBuffer):
      {
        CUresult * ret = va_arg(arglist, CUresult*);
        CUdeviceptr* devPtr = va_arg(arglist, CUdeviceptr*);
        CUexternalMemory extMem = va_arg(arglist, CUexternalMemory); 
        const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc  = va_arg(arglist, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*);
        *ret = cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);
        break;
      }
      case GENERATE_ENUM(cuExternalMemoryGetMappedMipmappedArray):
      {
        break;
      }
      case GENERATE_ENUM(cuImportExternalMemory):
      {
        break;
      }
      case GENERATE_ENUM(cuImportExternalSemaphore):
      {
        break;
      }
      case GENERATE_ENUM(cuSignalExternalSemaphoresAsync):
      {
        break;
      }
      case GENERATE_ENUM(cuWaitExternalSemaphoresAsync):
      {
        break;
      }
      case GENERATE_ENUM(cuStreamBatchMemOp):
      {
        break;
      }
      case GENERATE_ENUM(cuStreamWaitValue32):
      {
        break;
      }
      case GENERATE_ENUM(cuStreamWaitValue64):
      {
        break;
      }
      case GENERATE_ENUM(cuStreamWriteValue32):
      {
        break;
      }
      case GENERATE_ENUM(cuStreamWriteValue64):
      {
        break;
      }
      case GENERATE_ENUM(cuFuncGetAttribute):
      {
        break;
      }
      case GENERATE_ENUM(cuFuncSetAttribute):
      {
        break;
      }
      case GENERATE_ENUM(cuFuncSetCacheConfig):
      {
        break;
      }
      case GENERATE_ENUM(cuFuncSetSharedMemConfig):
      {
        break;
      }
      case GENERATE_ENUM(cuLaunchCooperativeKernel):
      {
        break;
      }
      case GENERATE_ENUM(cuLaunchCooperativeKernelMultiDevice):
      {
        break;
      }
      case GENERATE_ENUM(cuLaunchHostFunc):
      {
        break;
      }
      case GENERATE_ENUM(cuLaunchKernel):
      {
        CUresult *orig_ret = va_arg(arglist, CUresult*);
        CUfunction f = va_arg(arglist, CUfunction); 
        unsigned int  gridDimX = va_arg(arglist, unsigned int); 
        unsigned int  gridDimY = va_arg(arglist, unsigned int); 
        unsigned int  gridDimZ = va_arg(arglist, unsigned int); 
        unsigned int  blockDimX = va_arg(arglist, unsigned int); 
        unsigned int  blockDimY = va_arg(arglist, unsigned int); 
        unsigned int  blockDimZ = va_arg(arglist, unsigned int); 
        unsigned int  sharedMemBytes = va_arg(arglist, unsigned int); 
        CUstream hStream = va_arg(arglist, CUstream); 
        void** kernelParams = va_arg(arglist, void**); 
        void** extra = va_arg(arglist, void**);
        *orig_ret = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
        break;
      }
      case GENERATE_ENUM(cuFuncSetBlockShape):
      {
        break;
      }
      case GENERATE_ENUM(cuFuncSetSharedSize):
      {
        break;
      }
      case GENERATE_ENUM(cuLaunch):
      {
        break;
      }
      case GENERATE_ENUM(cuLaunchGrid):
      {
        break;
      }
      case GENERATE_ENUM(cuLaunchGridAsync):
      {
        break;
      }
      case GENERATE_ENUM(cuParamSetSize):
      {
        break;
      }
      case GENERATE_ENUM(cuParamSetTexRef):
      {
        break;
      }
      case GENERATE_ENUM(cuParamSetf):
      {
        break;
      }
      case GENERATE_ENUM(cuParamSeti):
      {
        break;
      }
      case GENERATE_ENUM(cuParamSetv):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphCreate):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphDestroy):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphDestroyNode):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphExecDestroy):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphGetEdges):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphGetNodes):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphGetRootNodes):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphHostNodeGetParams):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphHostNodeSetParams):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphKernelNodeGetParams):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphKernelNodeSetParams):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphLaunch):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphMemcpyNodeGetParams):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphMemcpyNodeSetParams):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphMemsetNodeGetParams):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphMemsetNodeSetParams):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphNodeFindInClone):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphNodeGetDependencies):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphNodeGetDependentNodes):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphNodeGetType):
      {
        break;
      }
      case GENERATE_ENUM(cuOccupancyMaxActiveBlocksPerMultiprocessor):
      {
        break;
      }
      case GENERATE_ENUM(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags):
      {
        break;
      }
      case GENERATE_ENUM(cuOccupancyMaxPotentialBlockSize):
      {
        break;
      }
      case GENERATE_ENUM(cuOccupancyMaxPotentialBlockSizeWithFlags):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefCreate):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefDestroy):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetAddress_v2):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetAddressMode):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetArray):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetBorderColor):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetFilterMode):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetFlags):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetFormat):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetMaxAnisotropy):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetMipmapFilterMode):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetMipmapLevelBias):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetMipmapLevelClamp):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefGetMipmappedArray):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetAddress_v2):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetAddress2D_v3):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetAddressMode):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetArray):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetBorderColor):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetFilterMode):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetFlags):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetFormat):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetMaxAnisotropy):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetMipmapFilterMode):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetMipmapLevelBias):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetMipmapLevelClamp):
      {
        break;
      }
      case GENERATE_ENUM(cuTexRefSetMipmappedArray):
      {
        break;
      }
      case GENERATE_ENUM(cuSurfRefGetArray):
      {
        break;
      }
      case GENERATE_ENUM(cuSurfRefSetArray):
      {
        break;
      }
      case GENERATE_ENUM(cuTexObjectCreate):
      {
        break;
      }
      case GENERATE_ENUM(cuTexObjectDestroy):
      {
        break;
      }
      case GENERATE_ENUM(cuTexObjectGetResourceDesc):
      {
        break;
      }
      case GENERATE_ENUM(cuTexObjectGetResourceViewDesc):
      {
        break;
      }
      case GENERATE_ENUM(cuTexObjectGetTextureDesc):
      {
        break;
      }
      case GENERATE_ENUM(cuSurfObjectCreate):
      {
        break;
      }
      case GENERATE_ENUM(cuSurfObjectDestroy):
      {
        break;
      }
      case GENERATE_ENUM(cuSurfObjectGetResourceDesc):
      {
        break;
      }
      case GENERATE_ENUM(cuCtxDisablePeerAccess):
      {
        break;
      }
      case GENERATE_ENUM(cuCtxEnablePeerAccess):
      {
        break;
      }
      case GENERATE_ENUM(cuDeviceCanAccessPeer):
      {
        break;
      }
      case GENERATE_ENUM(cuDeviceGetP2PAttribute):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphicsMapResources):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphicsResourceGetMappedMipmappedArray):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphicsResourceGetMappedPointer_v2):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphicsResourceSetMapFlags_v2):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphicsSubResourceGetMappedArray):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphicsUnmapResources):
      {
        break;
      }
      case GENERATE_ENUM(cuGraphicsUnregisterResource):
      {
        break;
      }
      case GENERATE_ENUM(cufftPlan1d):
      {
        break;
      }
      case GENERATE_ENUM(cufftPlan2d):
      {
        break;
      }
      case GENERATE_ENUM(cufftPlan3d):
      {
        break;
      }
      case GENERATE_ENUM(cufftPlanMany):
      {
        break;
      }
      case GENERATE_ENUM(cufftMakePlan1d):
      {
        break;
      }
      case GENERATE_ENUM(cufftMakePlan2d):
      {
        break;
      }
      case GENERATE_ENUM(cufftMakePlan3d):
      {
        break;
      }
      case GENERATE_ENUM(cufftMakePlanMany):
      {
        break;
      }
      case GENERATE_ENUM(cufftMakePlanMany64):
      {
        break;
      }
      case GENERATE_ENUM(cufftGetSizeMany64):
      {
        break;
      }
      case GENERATE_ENUM(cufftEstimate1d):
      {
        break;
      }
      case GENERATE_ENUM(cufftEstimate2d):
      {
        break;
      }
      case GENERATE_ENUM(cufftEstimate3d):
      {
        break;
      }
      case GENERATE_ENUM(cufftEstimateMany):
      {
        break;
      }
      case GENERATE_ENUM(cufftCreate):
      {
        break;
      }
      case GENERATE_ENUM(cufftGetSize1d):
      {
        break;
      }
      case GENERATE_ENUM(cufftGetSize2d):
      {
        break;
      }
      case GENERATE_ENUM(cufftGetSize3d):
      {
        break;
      }
      case GENERATE_ENUM(cufftGetSizeMany):
      {
        break;
      }
      case GENERATE_ENUM(cufftGetSize):
      {
        break;
      }
      case GENERATE_ENUM(cufftSetWorkArea):
      {
        break;
      }
      case GENERATE_ENUM(cufftSetAutoAllocation):
      {
        break;
      }
      case GENERATE_ENUM(cufftExecC2C):
      {
        break;
      }
      case GENERATE_ENUM(cufftExecR2C):
      {
        break;
      }
      case GENERATE_ENUM(cufftExecC2R):
      {
        break;
      }
      case GENERATE_ENUM(cufftExecZ2Z):
      {
        break;
      }
      case GENERATE_ENUM(cufftExecD2Z):
      {
        break;
      }
      case GENERATE_ENUM(cufftExecZ2D):
      {
        break;
      }
      case GENERATE_ENUM(cufftSetStream):
      {
        break;
      }
      case GENERATE_ENUM(cufftDestroy):
      {
        break;
      }
      case GENERATE_ENUM(cufftGetVersion):
      {
        break;
      }
      case GENERATE_ENUM(cufftGetProperty):
      {
        break;
      }
      case GENERATE_ENUM(cublasCreate_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t *handle = va_arg(arglist, cublasHandle_t *);
        *orig_ret = cublasCreate_v2(handle);
        break;
      }
      case GENERATE_ENUM(cublasSetStream_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *orig_ret = cublasSetStream_v2(handle, stream);
        break;
      }
      case GENERATE_ENUM(cublasSetMathMode):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
        int mode = va_arg(arglist, int);
        *orig_ret = cublasSetMathMode(handle, (cublasMath_t)mode);
        break;
      }
      case GENERATE_ENUM(cublasGetMathMode):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
        cublasMath_t * mode = va_arg(arglist, cublasMath_t *);
        *orig_ret = cublasGetMathMode(handle, mode);
        break;
      }
      case GENERATE_ENUM(cublasDdot_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t); 
        int n = va_arg(arglist, int); 
        const double * x = va_arg(arglist, const double*); 
        int incx = va_arg(arglist, int); 
        const double * y = va_arg(arglist, const double*); 
        int incy = va_arg(arglist, int); 
        double * result = va_arg(arglist, double*);
        *orig_ret = cublasDdot_v2(handle, n, x, incx, y, incy, result);
        break;
      }
      case GENERATE_ENUM(cublasDestroy_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
        *orig_ret = cublasDestroy_v2(handle);
        break;
      }
      case GENERATE_ENUM(cublasDaxpy_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t); 
        int n = va_arg(arglist, int);
        const double * alpha = va_arg(arglist, const double *); 
        const double * x = va_arg(arglist, const double *); 
        int incx = va_arg(arglist, int); 
        double * y = va_arg(arglist, double *); 
        int incy = va_arg(arglist, int);
        *orig_ret = cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDasum_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t);  
        int n = va_arg(arglist, int); 
        const double *x = va_arg(arglist, const double *); 
        int incx = va_arg(arglist, int); 
        double *result = va_arg(arglist, double *);
        *orig_ret = cublasDasum_v2(handle, n, x, incx, result);
        break;
      }
      case GENERATE_ENUM(cublasDgemm_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t); 
        int transa = va_arg(arglist, int); 
        int transb = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        const double *alpha = va_arg(arglist, const double *); 
        const double *A = va_arg(arglist, const double *); 
        int lda = va_arg(arglist, int); 
        const double *B = va_arg(arglist, const double *); 
        int ldb = va_arg(arglist, int); 
        const double *beta = va_arg(arglist, const double *); 
        double *C = va_arg(arglist, double *); 
        int ldc = va_arg(arglist, int);
        *orig_ret = cublasDgemm_v2(handle, (cublasOperation_t)transa, (cublasOperation_t)transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasDgemv_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t); 
        int trans = va_arg(arglist, int);  
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        const double *alpha = va_arg(arglist, const double *); 
        const double *A = va_arg(arglist, const double *); 
        int lda = va_arg(arglist, int); 
        const double *x = va_arg(arglist, const double *); 
        int incx = va_arg(arglist, int);
        const double *beta = va_arg(arglist, const double *);
        double *y = va_arg(arglist, double *); 
        int incy = va_arg(arglist, int);
        *orig_ret = cublasDgemv_v2(handle, (cublasOperation_t)trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDnrm2_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t); 
        int n = va_arg(arglist, int);
        const double *x = va_arg(arglist, const double*); 
        int incx = va_arg(arglist, int); 
        double *result = va_arg(arglist, double*);
        *orig_ret = cublasDnrm2_v2(handle, n, x, incx, result);
        break;
      }
      case GENERATE_ENUM(cublasDscal_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t); 
        int n = va_arg(arglist, int);
        const double *alpha = va_arg(arglist, const double*); 
        double *x = va_arg(arglist, double*);
        int incx = va_arg(arglist, int); 
        *orig_ret = cublasDscal_v2(handle, n, alpha, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDswap_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
        int n = va_arg(arglist, int);
        double *x = va_arg(arglist, double*);
        int incx = va_arg(arglist, int); 
        double *y = va_arg(arglist, double*);
        int incy = va_arg(arglist, int);
        *orig_ret = cublasDswap_v2(handle, n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasIdamax_v2):
      {
        cublasStatus_t *orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
        int n = va_arg(arglist, int);
        const double *x = va_arg(arglist, const double*);
        int incx = va_arg(arglist, int); 
        int *result = va_arg(arglist, int*);
        *orig_ret = cublasIdamax_v2(handle, n, x, incx, result);
        break;
      }
      case GENERATE_ENUM(cublasInit):
      {
        cublasStatus *orig_ret = va_arg(arglist, cublasStatus*);
        *orig_ret = cublasInit();
        break;
      }
      case GENERATE_ENUM(cublasShutdown):
      {
        cublasStatus *orig_ret = va_arg(arglist, cublasStatus*);
        *orig_ret = cublasShutdown();
        break;
      }
      case GENERATE_ENUM(cublasGetError):
      {
        cublasStatus *orig_ret = va_arg(arglist, cublasStatus*);
        *orig_ret  = cublasGetError();
        break;
      }
      case GENERATE_ENUM(cublasAlloc):
      {
        cublasStatus *orig_ret = va_arg(arglist, cublasStatus*);
        int n = va_arg(arglist, int); 
        int elemSize = va_arg(arglist, int); 
        void **devicePtr = va_arg(arglist, void **);
        *orig_ret = cublasAlloc(n, elemSize, devicePtr);
        break;
      }
      case GENERATE_ENUM(cublasFree):
      {
        cublasStatus *orig_ret = va_arg(arglist, cublasStatus*);
        void *devicePtr =  va_arg(arglist, void*);
        *orig_ret = cublasFree(devicePtr);
        break;
      }
      case GENERATE_ENUM(cublasSetKernelStream):
      {
        cublasStatus *orig_ret = va_arg(arglist, cublasStatus*);
        cudaStream_t stream =  va_arg(arglist, cudaStream_t);
        *orig_ret = cublasSetKernelStream(stream);
        break;
      }
      case GENERATE_ENUM(cublasSnrm2):
      {
        float *orig_ret = va_arg(arglist, float*);
        int n =  va_arg(arglist, int); 
        const float *x =  va_arg(arglist, const float *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasSnrm2(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDnrm2):
      {
        double *orig_ret = va_arg(arglist, double*);
        int n =  va_arg(arglist, int); 
        const double *x =  va_arg(arglist, const double *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasDnrm2(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasScnrm2):
      {
        float *orig_ret = va_arg(arglist, float*);
        int n =  va_arg(arglist, int); 
        const cuComplex *x =  va_arg(arglist, const cuComplex *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasScnrm2(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDznrm2):
      {
        double *orig_ret = va_arg(arglist, double*);
        int n =  va_arg(arglist, int); 
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasDznrm2(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasSdot):
      {
        float *orig_ret = va_arg(arglist, float*);
        int n =  va_arg(arglist, int); 
        const float *x =  va_arg(arglist, const float *); 
        int incx =  va_arg(arglist, int);
        const float *y = va_arg(arglist, const float *);  
        int incy =  va_arg(arglist, int);
        *orig_ret = cublasSdot(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDdot):
      {
        double *orig_ret = va_arg(arglist, double*);
        int n =  va_arg(arglist, int); 
        const double *x =  va_arg(arglist, const double *); 
        int incx =  va_arg(arglist, int);
        const double *y = va_arg(arglist, const double *);  
        int incy =  va_arg(arglist, int);
        *orig_ret = cublasDdot(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasCdotu):
      {
        cuComplex *orig_ret = va_arg(arglist, cuComplex*);
        int n =  va_arg(arglist, int); 
        const cuComplex *x =  va_arg(arglist, const cuComplex *); 
        int incx =  va_arg(arglist, int);
        const cuComplex *y = va_arg(arglist, const cuComplex *);  
        int incy =  va_arg(arglist, int);
        *orig_ret = cublasCdotu(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasCdotc):
      {
        cuComplex *orig_ret = va_arg(arglist, cuComplex*);
        int n =  va_arg(arglist, int); 
        const cuComplex *x =  va_arg(arglist, const cuComplex *); 
        int incx =  va_arg(arglist, int);
        const cuComplex *y = va_arg(arglist, const cuComplex *);  
        int incy =  va_arg(arglist, int);
        *orig_ret = cublasCdotc(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZdotu):
      {
        cuDoubleComplex *orig_ret = va_arg(arglist, cuDoubleComplex*);
        int n =  va_arg(arglist, int); 
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        const cuDoubleComplex *y = va_arg(arglist, const cuDoubleComplex *);  
        int incy =  va_arg(arglist, int);
        *orig_ret = cublasZdotu(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZdotc):
      {
        cuDoubleComplex *orig_ret = va_arg(arglist, cuDoubleComplex*);
        int n =  va_arg(arglist, int); 
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        const cuDoubleComplex *y = va_arg(arglist, const cuDoubleComplex *);  
        int incy =  va_arg(arglist, int);
        *orig_ret = cublasZdotc(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasSscal):
      {
        int n =  va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        cublasSscal(n, (float)alpha, (float*)x, incx); 
        break;
      }
      case GENERATE_ENUM(cublasDscal):
      {
        int n =  va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        cublasDscal(n, alpha, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasCscal):
      {
        int n =  va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cublasCscal(n, alpha, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasZscal):
      {
        int n =  va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cublasZscal(n, alpha, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasCsscal):
      {
        int n =  va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cublasCsscal(n, (float)alpha, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasZdscal):
      {
        int n =  va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cublasZdscal(n, alpha, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasSaxpy):
      {
        int n =  va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        const float *x =  va_arg(arglist, const float *); 
        int incx =  va_arg(arglist, int);
        float *y =  va_arg(arglist, float *); 
        int incy =  va_arg(arglist, int);
        cublasSaxpy(n, (float)alpha, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDaxpy):
      {
        int n =  va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        const double *x =  va_arg(arglist, const double *); 
        int incx =  va_arg(arglist, int);
        double *y =  va_arg(arglist, double *); 
        int incy =  va_arg(arglist, int);
        cublasDaxpy(n, alpha, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasCaxpy):
      {
        int n =  va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);
        const cuComplex *x =  va_arg(arglist, const cuComplex *); 
        int incx =  va_arg(arglist, int);
        cuComplex *y =  va_arg(arglist, cuComplex *); 
        int incy =  va_arg(arglist, int);
        cublasCaxpy(n, alpha, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZaxpy):
      {
        int n =  va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cuDoubleComplex *y =  va_arg(arglist, cuDoubleComplex *); 
        int incy =  va_arg(arglist, int);
        cublasZaxpy(n, alpha, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasScopy):
      {
        int n =  va_arg(arglist, int); 
        const float *x =  va_arg(arglist, const float *); 
        int incx =  va_arg(arglist, int);
        float *y =  va_arg(arglist, float *); 
        int incy =  va_arg(arglist, int);
        cublasScopy(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDcopy):
      {
        int n =  va_arg(arglist, int); 
        const double *x =  va_arg(arglist, const double *); 
        int incx =  va_arg(arglist, int);
        double *y =  va_arg(arglist, double *); 
        int incy =  va_arg(arglist, int);
        cublasDcopy(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasCcopy):
      {
        int n =  va_arg(arglist, int); 
        const cuComplex *x =  va_arg(arglist, const cuComplex *); 
        int incx =  va_arg(arglist, int);
        cuComplex *y =  va_arg(arglist, cuComplex *); 
        int incy =  va_arg(arglist, int);
        cublasCcopy(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZcopy):
      {
        int n =  va_arg(arglist, int); 
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cuDoubleComplex *y =  va_arg(arglist, cuDoubleComplex *); 
        int incy =  va_arg(arglist, int);
        cublasZcopy(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasSswap):
      {
        int n =  va_arg(arglist, int); 
        float *x =  va_arg(arglist, float *); 
        int incx =  va_arg(arglist, int);
        float *y =  va_arg(arglist, float *); 
        int incy =  va_arg(arglist, int);
        cublasSswap(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDswap):
      {
        int n =  va_arg(arglist, int); 
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        double *y =  va_arg(arglist, double *); 
        int incy =  va_arg(arglist, int);
        cublasDswap(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasCswap):
      {
        int n =  va_arg(arglist, int); 
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cuComplex *y =  va_arg(arglist, cuComplex *); 
        int incy =  va_arg(arglist, int);
        cublasCswap(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZswap):
      {
        int n =  va_arg(arglist, int); 
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cuDoubleComplex *y =  va_arg(arglist, cuDoubleComplex *); 
        int incy =  va_arg(arglist, int);
        cublasZswap(n, x, incx, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasIsamax):
      {
        int *orig_ret = va_arg(arglist, int*);
        int n =  va_arg(arglist, int); 
        const float *x =  va_arg(arglist, const float *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasIsamax(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasIdamax):
      {
        int *orig_ret = va_arg(arglist, int*);
        int n =  va_arg(arglist, int); 
        const double *x =  va_arg(arglist, const double *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasIdamax(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasIcamax):
      {
        int *orig_ret = va_arg(arglist, int*);
        int n =  va_arg(arglist, int); 
        const cuComplex *x =  va_arg(arglist, const cuComplex *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasIcamax(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasIzamax):
      {
        int *orig_ret = va_arg(arglist, int*);
        int n =  va_arg(arglist, int); 
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasIzamax(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasIsamin):
      {
        int *orig_ret = va_arg(arglist, int*);
        int n =  va_arg(arglist, int); 
        const float *x =  va_arg(arglist, const float *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasIsamin(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasIdamin):
      {
        int *orig_ret = va_arg(arglist, int*);
        int n =  va_arg(arglist, int); 
        const double *x =  va_arg(arglist, const double *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasIdamin(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasIcamin):
      {
        int *orig_ret = va_arg(arglist, int*);
        int n =  va_arg(arglist, int); 
        const cuComplex *x =  va_arg(arglist, const cuComplex *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasIcamin(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasIzamin):
      {
        int *orig_ret = va_arg(arglist, int*);
        int n =  va_arg(arglist, int); 
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasIzamin(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasSasum):
      {
        float *orig_ret = va_arg(arglist, float*);
        int n =  va_arg(arglist, int); 
        const float *x =  va_arg(arglist, const float *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasSasum(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDasum):
      {
        double *orig_ret = va_arg(arglist, double*);
        int n =  va_arg(arglist, int); 
        const double *x =  va_arg(arglist, const double *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasDasum(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasScasum):
      {
        float *orig_ret = va_arg(arglist, float*);
        int n =  va_arg(arglist, int); 
        const cuComplex *x =  va_arg(arglist, const cuComplex *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasScasum(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDzasum):
      {
        double *orig_ret = va_arg(arglist, double*);
        int n =  va_arg(arglist, int); 
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        *orig_ret = cublasDzasum(n, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasSrot):
      {
        int n =  va_arg(arglist, int); 
        float *x =  va_arg(arglist, float *); 
        int incx =  va_arg(arglist, int);
        float *y =  va_arg(arglist, float *); 
        int incy =  va_arg(arglist, int);
        double sc =  va_arg(arglist, double);
        double ss =  va_arg(arglist, double);
        cublasSrot(n, x, incx, y, incy, (float)sc, (float)ss);
        break;
      }
      case GENERATE_ENUM(cublasDrot):
      {
        int n =  va_arg(arglist, int); 
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        double *y =  va_arg(arglist, double *); 
        int incy =  va_arg(arglist, int);
        double sc =  va_arg(arglist, double);
        double ss =  va_arg(arglist, double);
        cublasDrot(n, x, incx, y, incy, sc, ss);
        break;
      }
      case GENERATE_ENUM(cublasCrot):
      {
        int n =  va_arg(arglist, int); 
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cuComplex *y =  va_arg(arglist, cuComplex *); 
        int incy =  va_arg(arglist, int);
        double sc =  va_arg(arglist, double);
        cuComplex ss =  va_arg(arglist, cuComplex);
        cublasCrot(n, x, incx, y, incy, (float)sc, ss);
        break;
      }
      case GENERATE_ENUM(cublasZrot):
      {
        int n =  va_arg(arglist, int); 
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cuDoubleComplex *y =  va_arg(arglist, cuDoubleComplex *); 
        int incy =  va_arg(arglist, int);
        double sc =  va_arg(arglist, double);
        cuDoubleComplex ss =  va_arg(arglist, cuDoubleComplex);
        cublasZrot(n, x, incx, y, incy, sc, ss);
        break;
      }
      case GENERATE_ENUM(cublasCsrot):
      {
        int n =  va_arg(arglist, int); 
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cuComplex *y =  va_arg(arglist, cuComplex *); 
        int incy =  va_arg(arglist, int);
        double sc =  va_arg(arglist, double);
        double ss =  va_arg(arglist, double);
        cublasCsrot(n, x, incx, y, incy, (float)sc, (float)ss);
        break;
      }
      case GENERATE_ENUM(cublasZdrot):
      {
        int n =  va_arg(arglist, int); 
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cuDoubleComplex *y =  va_arg(arglist, cuDoubleComplex *); 
        int incy =  va_arg(arglist, int);
        double sc =  va_arg(arglist, double);
        double ss =  va_arg(arglist, double);
        cublasZdrot(n, x, incx, y, incy, sc, ss);
        break;
      }
      case GENERATE_ENUM(cublasSrotg):
      {
        float* sa =  va_arg(arglist, float*);
        float* sb =  va_arg(arglist, float *); 
        float* sc =  va_arg(arglist, float*);
        float* ss =  va_arg(arglist, float*);
        cublasSrotg(sa, sb, sc, ss);
        break;
      }
      case GENERATE_ENUM(cublasDrotg):
      {
        double* sa =  va_arg(arglist, double*);
        double* sb =  va_arg(arglist, double *); 
        double* sc =  va_arg(arglist, double*);
        double* ss =  va_arg(arglist, double*);
        cublasDrotg(sa, sb, sc, ss);
        break;
      }
      case GENERATE_ENUM(cublasCrotg):
      {
        cuComplex* sa =  va_arg(arglist, cuComplex*);
        cuComplex sb =  va_arg(arglist, cuComplex); 
        float* sc =  va_arg(arglist, float*);
        cuComplex* ss =  va_arg(arglist, cuComplex*);
        cublasCrotg(sa, sb, sc, ss);
        break;
      }
      case GENERATE_ENUM(cublasZrotg):
      {
        cuDoubleComplex* sa =  va_arg(arglist, cuDoubleComplex*);
        cuDoubleComplex sb =  va_arg(arglist, cuDoubleComplex); 
        double* sc =  va_arg(arglist, double*);
        cuDoubleComplex* ss =  va_arg(arglist, cuDoubleComplex*);
        cublasZrotg(sa, sb, sc, ss);
        break;
      }
      case GENERATE_ENUM(cublasSrotm):
      {
        int n =  va_arg(arglist, int); 
        float *x =  va_arg(arglist, float *); 
        int incx =  va_arg(arglist, int);
        float *y =  va_arg(arglist, float *); 
        int incy =  va_arg(arglist, int);
        const float* sparam =  va_arg(arglist, const float*);
        cublasSrotm(n, x, incx, y, incy, sparam);
        break;
      }
      case GENERATE_ENUM(cublasDrotm):
      {
        int n =  va_arg(arglist, int); 
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        double *y =  va_arg(arglist, double *); 
        int incy =  va_arg(arglist, int);
        const double* sparam =  va_arg(arglist, const double*);
        cublasDrotm(n, x, incx, y, incy, sparam);
        break;
      }
      case GENERATE_ENUM(cublasSrotmg):
      {
        float *sd1 =  va_arg(arglist, float *); 
        float *sd2 =  va_arg(arglist, float *); 
        float *sx1 =  va_arg(arglist, float *); 
        const float *sx2 =  va_arg(arglist, const float *); 
        float* sparam =  va_arg(arglist, float*);
        cublasSrotmg(sd1, sd2, sx1, sx2, sparam);
        break;
      }
      case GENERATE_ENUM(cublasDrotmg):
      {
        double *sd1 =  va_arg(arglist, double *); 
        double *sd2 =  va_arg(arglist, double *); 
        double *sx1 =  va_arg(arglist, double *); 
        const double *sx2 =  va_arg(arglist, const double *); 
        double* sparam =  va_arg(arglist, double*);
        cublasDrotmg(sd1, sd2, sx1, sx2, sparam);
        break;
      }
      case GENERATE_ENUM(cublasSgemv):
      {
        int trans =  va_arg(arglist, int); 
        int m =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int); 
        double alpha =  va_arg(arglist, double); 
        const float *A =  va_arg(arglist, const float*); 
        int lda =  va_arg(arglist, int);
        const float *x =  va_arg(arglist, const float*); 
        int incx =  va_arg(arglist, int); 
        double beta =  va_arg(arglist, double); 
        float *y =  va_arg(arglist, float*); 
        int incy =  va_arg(arglist, int);
        cublasSgemv((char)trans, m, n, (float)alpha, A, lda, x, incx, (float)beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDgemv):
      {
        int trans =  va_arg(arglist, int); 
        int m =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int); 
        double alpha =  va_arg(arglist, double); 
        const double *A =  va_arg(arglist, const double*); 
        int lda =  va_arg(arglist, int);
        const double *x =  va_arg(arglist, const double*); 
        int incx =  va_arg(arglist, int); 
        double beta =  va_arg(arglist, double); 
        double *y =  va_arg(arglist, double*); 
        int incy =  va_arg(arglist, int);
        cublasDgemv((char)trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasCgemv):
      {
        int trans =  va_arg(arglist, int); 
        int m =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int); 
        cuComplex alpha =  va_arg(arglist, cuComplex); 
        const cuComplex *A =  va_arg(arglist, const cuComplex*); 
        int lda =  va_arg(arglist, int);
        const cuComplex *x =  va_arg(arglist, const cuComplex*); 
        int incx =  va_arg(arglist, int); 
        cuComplex beta =  va_arg(arglist, cuComplex); 
        cuComplex *y =  va_arg(arglist, cuComplex*); 
        int incy =  va_arg(arglist, int);
        cublasCgemv((char)trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZgemv):
      {
        int trans =  va_arg(arglist, int); 
        int m =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int); 
        cuDoubleComplex alpha =  va_arg(arglist, cuDoubleComplex); 
        const cuDoubleComplex *A =  va_arg(arglist, const cuDoubleComplex*); 
        int lda =  va_arg(arglist, int);
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex*); 
        int incx =  va_arg(arglist, int); 
        cuDoubleComplex beta =  va_arg(arglist, cuDoubleComplex); 
        cuDoubleComplex *y =  va_arg(arglist, cuDoubleComplex*); 
        int incy =  va_arg(arglist, int);
        cublasZgemv((char)trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasSgbmv):
      {
        int trans =  va_arg(arglist, int); 
        int m =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int); 
        int kl =  va_arg(arglist, int); 
        int ku =  va_arg(arglist, int);
        double alpha =  va_arg(arglist, double); 
        const float *A =  va_arg(arglist, const float*); 
        int lda =  va_arg(arglist, int);
        const float *x =  va_arg(arglist, const float*); 
        int incx =  va_arg(arglist, int); 
        double beta =  va_arg(arglist, double); 
        float *y =  va_arg(arglist, float*); 
        int incy =  va_arg(arglist, int);
        cublasSgbmv((char)trans, m, n, kl, ku, (float)alpha, A, lda, x, incx, (float)beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDgbmv):
      {
        int trans =  va_arg(arglist, int); 
        int m =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int); 
        int kl =  va_arg(arglist, int); 
        int ku =  va_arg(arglist, int);
        double alpha =  va_arg(arglist, double); 
        const double *A =  va_arg(arglist, const double*); 
        int lda =  va_arg(arglist, int);
        const double *x =  va_arg(arglist, const double*); 
        int incx =  va_arg(arglist, int); 
        double beta =  va_arg(arglist, double); 
        double *y =  va_arg(arglist, double*); 
        int incy =  va_arg(arglist, int);
        cublasDgbmv((char)trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasCgbmv):
      {
        int trans =  va_arg(arglist, int); 
        int m =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int); 
        int kl =  va_arg(arglist, int); 
        int ku =  va_arg(arglist, int);
        cuComplex alpha =  va_arg(arglist, cuComplex); 
        const cuComplex *A =  va_arg(arglist, const cuComplex*); 
        int lda =  va_arg(arglist, int);
        const cuComplex *x =  va_arg(arglist, const cuComplex*); 
        int incx =  va_arg(arglist, int); 
        cuComplex beta =  va_arg(arglist, cuComplex); 
        cuComplex *y =  va_arg(arglist, cuComplex*); 
        int incy =  va_arg(arglist, int);
        cublasCgbmv((char)trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZgbmv):
      {
        int trans =  va_arg(arglist, int); 
        int m =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int); 
        int kl =  va_arg(arglist, int); 
        int ku =  va_arg(arglist, int);
        cuDoubleComplex alpha =  va_arg(arglist, cuDoubleComplex); 
        const cuDoubleComplex *A =  va_arg(arglist, const cuDoubleComplex*); 
        int lda =  va_arg(arglist, int);
        const cuDoubleComplex *x =  va_arg(arglist, const cuDoubleComplex*); 
        int incx =  va_arg(arglist, int); 
        cuDoubleComplex beta =  va_arg(arglist, cuDoubleComplex); 
        cuDoubleComplex *y =  va_arg(arglist, cuDoubleComplex*); 
        int incy =  va_arg(arglist, int);
        cublasZgbmv((char)trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasStrmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const float *A =  va_arg(arglist, const float *); 
        int lda =  va_arg(arglist, int);
        float *x =  va_arg(arglist, float *); 
        int incx =  va_arg(arglist, int);
        cublasStrmv((char)uplo, (char)trans, (char)diag, n, A, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDtrmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const double *A =  va_arg(arglist, const double *); 
        int lda =  va_arg(arglist, int);
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        cublasDtrmv((char)uplo, (char)trans, (char)diag, n, A, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasCtrmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const cuComplex *A =  va_arg(arglist, const cuComplex *); 
        int lda =  va_arg(arglist, int);
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cublasCtrmv((char)uplo, (char)trans, (char)diag, n, A, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasZtrmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const cuDoubleComplex *A =  va_arg(arglist, const cuDoubleComplex *); 
        int lda =  va_arg(arglist, int);
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cublasZtrmv((char)uplo, (char)trans, (char)diag, n, A, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasStbmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        int k =  va_arg(arglist, int);
        const float *A =  va_arg(arglist, const float *); 
        int lda =  va_arg(arglist, int);
        float *x =  va_arg(arglist, float *); 
        int incx =  va_arg(arglist, int);
        cublasStbmv((char)uplo, (char)trans, (char)diag, n, k, A, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDtbmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        int k =  va_arg(arglist, int);
        const double *A =  va_arg(arglist, const double *); 
        int lda =  va_arg(arglist, int);
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        cublasDtbmv((char)uplo, (char)trans, (char)diag, n, k, A, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasCtbmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        int k =  va_arg(arglist, int);
        const cuComplex *A =  va_arg(arglist, const cuComplex *); 
        int lda =  va_arg(arglist, int);
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cublasCtbmv((char)uplo, (char)trans, (char)diag, n, k, A, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasZtbmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        int k =  va_arg(arglist, int);
        const cuDoubleComplex *A =  va_arg(arglist, const cuDoubleComplex *); 
        int lda =  va_arg(arglist, int);
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cublasZtbmv((char)uplo, (char)trans, (char)diag, n, k, A, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasStpmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const float *AP =  va_arg(arglist, const float *); 
        float *x =  va_arg(arglist, float *); 
        int incx =  va_arg(arglist, int);
        cublasStpmv((char)uplo,(char) trans, (char)diag, n, AP, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDtpmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const double *AP =  va_arg(arglist, const double *); 
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        cublasDtpmv((char)uplo, (char)trans, (char)diag, n, AP, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasCtpmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const cuComplex *AP =  va_arg(arglist, const cuComplex *); 
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cublasCtpmv((char)uplo, (char)trans, (char)diag, n, AP, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasZtpmv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const cuDoubleComplex *AP =  va_arg(arglist, const cuDoubleComplex *); 
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cublasZtpmv((char)uplo, (char)trans, (char)diag, n, AP, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasStrsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const float *AP =  va_arg(arglist, const float *); 
        int lda = va_arg(arglist, int);
        float *x =  va_arg(arglist, float *); 
        int incx =  va_arg(arglist, int);
        cublasStrsv((char)uplo, (char)trans, (char)diag, n, AP, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDtrsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const double *AP =  va_arg(arglist, const double *); 
        int lda = va_arg(arglist, int);
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        cublasDtrsv((char)uplo, (char)trans, (char)diag, n, AP, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasCtrsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const cuComplex *AP =  va_arg(arglist, const cuComplex *); 
        int lda = va_arg(arglist, int);
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cublasCtrsv((char)uplo, (char)trans, (char)diag, n, AP, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasZtrsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const cuDoubleComplex *AP =  va_arg(arglist, const cuDoubleComplex *); 
        int lda = va_arg(arglist, int);
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cublasZtrsv((char)uplo, (char)trans, (char)diag, n, AP, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasStpsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const float *AP =  va_arg(arglist, const float *); 
        float *x =  va_arg(arglist, float *); 
        int incx =  va_arg(arglist, int);
        cublasStpsv((char)uplo, (char)trans, (char)diag, n, AP, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDtpsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const double *AP =  va_arg(arglist, const double *); 
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        cublasDtpsv((char)uplo, (char)trans, (char)diag, n, AP, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasCtpsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const cuComplex *AP =  va_arg(arglist, const cuComplex *); 
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cublasCtpsv((char)uplo, (char)trans, (char)diag, n, AP, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasZtpsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        const cuDoubleComplex *AP =  va_arg(arglist, const cuDoubleComplex *); 
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cublasZtpsv((char)uplo, (char)trans, (char)diag, n, AP, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasStbsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        int k =  va_arg(arglist, int);
        const float *AP =  va_arg(arglist, const float *); 
        int lda =  va_arg(arglist, int);
        float *x =  va_arg(arglist, float *); 
        int incx =  va_arg(arglist, int);
        cublasStbsv((char)uplo, (char)trans, (char)diag, n, k, AP, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasDtbsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        int k =  va_arg(arglist, int);
        const double *AP =  va_arg(arglist, const double *); 
        int lda =  va_arg(arglist, int);
        double *x =  va_arg(arglist, double *); 
        int incx =  va_arg(arglist, int);
        cublasDtbsv((char)uplo, (char)trans, (char)diag, n, k, AP, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasCtbsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        int k =  va_arg(arglist, int);
        const cuComplex *AP =  va_arg(arglist, const cuComplex *); 
        int lda =  va_arg(arglist, int);
        cuComplex *x =  va_arg(arglist, cuComplex *); 
        int incx =  va_arg(arglist, int);
        cublasCtbsv((char)uplo, (char)trans, (char)diag, n, k, AP, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasZtbsv):
      {
        int uplo =  va_arg(arglist, int); 
        int trans =  va_arg(arglist, int);  
        int diag =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        int k =  va_arg(arglist, int);
        const cuDoubleComplex *AP =  va_arg(arglist, const cuDoubleComplex *); 
        int lda =  va_arg(arglist, int);
        cuDoubleComplex *x =  va_arg(arglist, cuDoubleComplex *); 
        int incx =  va_arg(arglist, int);
        cublasZtbsv((char)uplo, (char) trans, (char)diag, n, k, AP, lda, x, incx);
        break;
      }
      case GENERATE_ENUM(cublasSsymv):
      {
        int uplo =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        double alpha =  va_arg(arglist, double); 
        const float *A =  va_arg(arglist, const float *); 
        int lda =  va_arg(arglist, int);
        const float *x =  va_arg(arglist, const float *); 
        int incx =  va_arg(arglist, int);
        double beta =  va_arg(arglist, double); 
        float *y =  va_arg(arglist, float *); 
        int incy =  va_arg(arglist, int);
        cublasSsymv((char)uplo, n, (float)alpha, A, lda, x, incx, (float)beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDsymv):
      {
        int uplo =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        double alpha =  va_arg(arglist, double); 
        const double *A =  va_arg(arglist, const double *); 
        int lda =  va_arg(arglist, int);
        const double *x =  va_arg(arglist, const double *); 
        int incx =  va_arg(arglist, int);
        double beta =  va_arg(arglist, double); 
        double *y =  va_arg(arglist, double *); 
        int incy =  va_arg(arglist, int);
        cublasDsymv((char)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasChemv):
      {
        int uplo =  va_arg(arglist, int); 
        int n =  va_arg(arglist, int);
        cuComplex alpha =  va_arg(arglist, cuComplex); 
        const cuComplex *A =  va_arg(arglist, const cuComplex *); 
        int lda =  va_arg(arglist, int);
        const cuComplex *x =  va_arg(arglist, const cuComplex *); 
        int incx =  va_arg(arglist, int);
        cuComplex beta =  va_arg(arglist, cuComplex); 
        cuComplex *y =  va_arg(arglist, cuComplex *); 
        int incy =  va_arg(arglist, int);
        cublasChemv((char)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZhemv):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex); 
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex *); 
        int lda = va_arg(arglist, int); 
        const cuDoubleComplex *x = va_arg(arglist, const cuDoubleComplex *);
        int incx = va_arg(arglist, int); 
        cuDoubleComplex beta = va_arg(arglist, cuDoubleComplex);
        cuDoubleComplex *y = va_arg(arglist, cuDoubleComplex *);
        int incy = va_arg(arglist, int);
        cublasZhemv((char)uplo, n, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasSsbmv):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        const float *A = va_arg(arglist, const float*); 
        int lda = va_arg(arglist, int); 
        const float *x = va_arg(arglist, const float*); 
        int incx = va_arg(arglist, int);  
        double beta = va_arg(arglist, double); 
        float *y = va_arg(arglist, float*); 
        int incy = va_arg(arglist, int);
        cublasSsbmv((char)uplo, n, k, (float)alpha, A, lda, x, incx, (float)beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDsbmv):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        const double *A = va_arg(arglist, const double*); 
        int lda = va_arg(arglist, int); 
        const double *x = va_arg(arglist, const double*); 
        int incx = va_arg(arglist, int);  
        double beta = va_arg(arglist, double); 
        double *y = va_arg(arglist, double*); 
        int incy = va_arg(arglist, int);
        cublasDsbmv((char)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasChbmv):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);
        const cuComplex *A = va_arg(arglist, const cuComplex*); 
        int lda = va_arg(arglist, int); 
        const cuComplex *x = va_arg(arglist, const cuComplex*); 
        int incx = va_arg(arglist, int);  
        cuComplex beta = va_arg(arglist, cuComplex); 
        cuComplex *y = va_arg(arglist, cuComplex*); 
        int incy = va_arg(arglist, int);
        cublasChbmv((char)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZhbmv):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex*); 
        int lda = va_arg(arglist, int); 
        const cuDoubleComplex *x = va_arg(arglist, const cuDoubleComplex*); 
        int incx = va_arg(arglist, int);  
        cuDoubleComplex beta = va_arg(arglist, cuDoubleComplex); 
        cuDoubleComplex *y = va_arg(arglist, cuDoubleComplex*); 
        int incy = va_arg(arglist, int);
        cublasZhbmv((char)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasSspmv):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const float *AP = va_arg(arglist, const float *);  
        const float *x = va_arg(arglist, const float *);  
        int incx = va_arg(arglist, int);  
        double beta = va_arg(arglist, double);
        float *y = va_arg(arglist, float *);  
        int incy = va_arg(arglist, int);
        cublasSspmv((char)uplo, n, (float)alpha, AP, x, incx, (float)beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasDspmv):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const double *AP = va_arg(arglist, const double *);  
        const double *x = va_arg(arglist, const double *);  
        int incx = va_arg(arglist, int);  
        double beta = va_arg(arglist, double);
        double *y = va_arg(arglist, double *);  
        int incy = va_arg(arglist, int);
        cublasDspmv((char)uplo, n, alpha, AP, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasChpmv):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int);  
        cuComplex alpha = va_arg(arglist, cuComplex); 
        const cuComplex *AP = va_arg(arglist, const cuComplex *);  
        const cuComplex *x = va_arg(arglist, const cuComplex *);  
        int incx = va_arg(arglist, int);  
        cuComplex beta = va_arg(arglist, cuComplex);
        cuComplex *y = va_arg(arglist, cuComplex *);  
        int incy = va_arg(arglist, int);
        cublasChpmv((char)uplo, n, alpha, AP, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasZhpmv):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int);  
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex); 
        const cuDoubleComplex *AP = va_arg(arglist, const cuDoubleComplex *);  
        const cuDoubleComplex *x = va_arg(arglist, const cuDoubleComplex *);  
        int incx = va_arg(arglist, int);  
        cuDoubleComplex beta = va_arg(arglist, cuDoubleComplex);
        cuDoubleComplex *y = va_arg(arglist, cuDoubleComplex *);  
        int incy = va_arg(arglist, int);
        cublasZhpmv((char)uplo, n, alpha, AP, x, incx, beta, y, incy);
        break;
      }
      case GENERATE_ENUM(cublasSger):
      {
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        const float *x = va_arg(arglist, const float*); 
        int incx = va_arg(arglist, int); 
        const float *y = va_arg(arglist, const float*); 
        int incy = va_arg(arglist, int); 
        float *A = va_arg(arglist, float*); 
        int lda = va_arg(arglist, int);
        cublasSger(m, n, (float)alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasDger):
      {
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);
        const double *x = va_arg(arglist, const double*); 
        int incx = va_arg(arglist, int); 
        const double *y = va_arg(arglist, const double*); 
        int incy = va_arg(arglist, int); 
        double *A = va_arg(arglist, double*); 
        int lda = va_arg(arglist, int);
        cublasDger(m, n, alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasCgeru):
      {
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);
        const cuComplex *x = va_arg(arglist, const cuComplex*); 
        int incx = va_arg(arglist, int); 
        const cuComplex *y = va_arg(arglist, const cuComplex*); 
        int incy = va_arg(arglist, int); 
        cuComplex *A = va_arg(arglist, cuComplex*); 
        int lda = va_arg(arglist, int);
        cublasCgeru(m, n, alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasCgerc):
      {
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);
        const cuComplex *x = va_arg(arglist, const cuComplex*); 
        int incx = va_arg(arglist, int); 
        const cuComplex *y = va_arg(arglist, const cuComplex*); 
        int incy = va_arg(arglist, int); 
        cuComplex *A = va_arg(arglist, cuComplex*); 
        int lda = va_arg(arglist, int);
        cublasCgerc(m, n, alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasZgeru):
      {
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);
        const cuDoubleComplex *x = va_arg(arglist, const cuDoubleComplex*); 
        int incx = va_arg(arglist, int); 
        const cuDoubleComplex *y = va_arg(arglist, const cuDoubleComplex*); 
        int incy = va_arg(arglist, int); 
        cuDoubleComplex *A = va_arg(arglist, cuDoubleComplex*); 
        int lda = va_arg(arglist, int);
        cublasZgeru(m, n, alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasZgerc):
      {
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);
        const cuDoubleComplex *x = va_arg(arglist, const cuDoubleComplex*); 
        int incx = va_arg(arglist, int); 
        const cuDoubleComplex *y = va_arg(arglist, const cuDoubleComplex*); 
        int incy = va_arg(arglist, int); 
        cuDoubleComplex *A = va_arg(arglist, cuDoubleComplex*); 
        int lda = va_arg(arglist, int);
        cublasZgerc(m, n, alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasSsyr):
      {
        int uplo = va_arg(arglist, int);  
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const float *x = va_arg(arglist, const float*); 
        int incx = va_arg(arglist, int);  
        float *A = va_arg(arglist, float*); 
        int lda = va_arg(arglist, int);
        cublasSsyr((char)uplo, n, (float)alpha, x, incx, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasDsyr):
      {
        int uplo = va_arg(arglist, int);  
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const double *x = va_arg(arglist, const double*); 
        int incx = va_arg(arglist, int);  
        double *A = va_arg(arglist, double*); 
        int lda = va_arg(arglist, int);
        cublasDsyr((char)uplo, n, alpha, x, incx, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasCher):
      {
        int uplo = va_arg(arglist, int);  
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const cuComplex *x = va_arg(arglist, const cuComplex*); 
        int incx = va_arg(arglist, int);  
        cuComplex *A = va_arg(arglist, cuComplex*); 
        int lda = va_arg(arglist, int);
        cublasCher((char)uplo, n, (float)alpha, x, incx, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasZher):
      {
        int uplo = va_arg(arglist, int);  
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const cuDoubleComplex *x = va_arg(arglist, const cuDoubleComplex*); 
        int incx = va_arg(arglist, int);  
        cuDoubleComplex *A = va_arg(arglist, cuDoubleComplex*); 
        int lda = va_arg(arglist, int);
        cublasZher((char)uplo, n, alpha, x, incx, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasSspr):
      {
        int uplo = va_arg(arglist, int);  
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const float *x = va_arg(arglist, const float*); 
        int incx = va_arg(arglist, int);  
        float *AP = va_arg(arglist, float*); 
        
        cublasSspr((char)uplo, n, (float)alpha, x, incx, AP);
        break;
      }
      case GENERATE_ENUM(cublasDspr):
      {
        int uplo = va_arg(arglist, int);  
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const double *x = va_arg(arglist, const double*); 
        int incx = va_arg(arglist, int);  
        double *AP = va_arg(arglist, double*); 
        
        cublasDspr((char)uplo, n, alpha, x, incx, AP);
        break;
      }
      case GENERATE_ENUM(cublasChpr):
      {
        int uplo = va_arg(arglist, int);  
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const cuComplex *x = va_arg(arglist, const cuComplex*); 
        int incx = va_arg(arglist, int);  
        cuComplex *AP = va_arg(arglist, cuComplex*); 
        
        cublasChpr((char)uplo, n, (float)alpha, x, incx, AP);
        break;
      }
      case GENERATE_ENUM(cublasZhpr):
      {
        int uplo = va_arg(arglist, int);  
        int n = va_arg(arglist, int);  
        double alpha = va_arg(arglist, double); 
        const cuDoubleComplex *x = va_arg(arglist, const cuDoubleComplex*); 
        int incx = va_arg(arglist, int);  
        cuDoubleComplex *AP = va_arg(arglist, cuDoubleComplex*); 
        
        cublasZhpr((char)uplo, n, alpha, x, incx, AP);
        break;
      }
      case GENERATE_ENUM(cublasSsyr2):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double); 
        const float *x = va_arg(arglist, const float*);  
        int incx = va_arg(arglist, int); 
        const float *y = va_arg(arglist, const float*); 
        int incy = va_arg(arglist, int); 
        float *A = va_arg(arglist, float*);  
        int lda = va_arg(arglist, int);
        cublasSsyr2((char)uplo, n, (float)alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasDsyr2):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double); 
        const double *x = va_arg(arglist, const double*);  
        int incx = va_arg(arglist, int); 
        const double *y = va_arg(arglist, const double*); 
        int incy = va_arg(arglist, int); 
        double *A = va_arg(arglist, double*);  
        int lda = va_arg(arglist, int);
        cublasDsyr2((char)uplo, n, alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasCher2):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex); 
        const cuComplex *x = va_arg(arglist, const cuComplex*);  
        int incx = va_arg(arglist, int); 
        const cuComplex *y = va_arg(arglist, const cuComplex*); 
        int incy = va_arg(arglist, int); 
        cuComplex *A = va_arg(arglist, cuComplex*);  
        int lda = va_arg(arglist, int);
        cublasCher2((char)uplo, n, alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasZher2):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex); 
        const cuDoubleComplex *x = va_arg(arglist, const cuDoubleComplex*);  
        int incx = va_arg(arglist, int); 
        const cuDoubleComplex *y = va_arg(arglist, const cuDoubleComplex*); 
        int incy = va_arg(arglist, int); 
        cuDoubleComplex *A = va_arg(arglist, cuDoubleComplex*);  
        int lda = va_arg(arglist, int);
        cublasZher2((char)uplo, n, alpha, x, incx, y, incy, A, lda);
        break;
      }
      case GENERATE_ENUM(cublasSspr2):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double); 
        const float *x = va_arg(arglist, const float*);  
        int incx = va_arg(arglist, int); 
        const float *y = va_arg(arglist, const float*); 
        int incy = va_arg(arglist, int); 
        float *AP = va_arg(arglist, float*);  
        
        cublasSspr2((char)uplo, n, (float)alpha, x, incx, y, incy, AP);
        break;
      }
      case GENERATE_ENUM(cublasDspr2):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double); 
        const double *x = va_arg(arglist, const double*);  
        int incx = va_arg(arglist, int); 
        const double *y = va_arg(arglist, const double*); 
        int incy = va_arg(arglist, int); 
        double *AP = va_arg(arglist, double*);  
        
        cublasDspr2((char)uplo, n, alpha, x, incx, y, incy, AP);
        break;
      }
      case GENERATE_ENUM(cublasChpr2):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex); 
        const cuComplex *x = va_arg(arglist, const cuComplex*);  
        int incx = va_arg(arglist, int); 
        const cuComplex *y = va_arg(arglist, const cuComplex*); 
        int incy = va_arg(arglist, int); 
        cuComplex *AP = va_arg(arglist, cuComplex*);  
        
        cublasChpr2((char)uplo, n, alpha, x, incx, y, incy, AP);
        break;
      }
      case GENERATE_ENUM(cublasZhpr2):
      {
        int uplo = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex); 
        const cuDoubleComplex *x = va_arg(arglist, const cuDoubleComplex*);  
        int incx = va_arg(arglist, int); 
        const cuDoubleComplex *y = va_arg(arglist, const cuDoubleComplex*); 
        int incy = va_arg(arglist, int); 
        cuDoubleComplex *AP = va_arg(arglist, cuDoubleComplex*);  
        
        cublasZhpr2((char)uplo, n, alpha, x, incx, y, incy, AP);
        break;
      }
      case GENERATE_ENUM(cublasSgemm):
      {
        int transa = va_arg(arglist, int); 
        int transb = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int);
        int k = va_arg(arglist, int); 
        double alpha =  va_arg(arglist, double);
        const float *A =  va_arg(arglist, const float*); 
        int lda =  va_arg(arglist, int); 
        const float *B =  va_arg(arglist, const float*); 
        int ldb =  va_arg(arglist, int); 
        double beta =  va_arg(arglist, double);
        float *C =  va_arg(arglist, float*); 
        int ldc =  va_arg(arglist, int);
        cublasSgemm((char)transa, (char)transb, m, n, k, (float)alpha, A, lda, B, ldb, (float)beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasDgemm):
      {
        int transa = va_arg(arglist, int); 
        int transb = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int);
        int k = va_arg(arglist, int); 
        double alpha =  va_arg(arglist, double);
        const double *A =  va_arg(arglist, const double*); 
        int lda =  va_arg(arglist, int); 
        const double *B =  va_arg(arglist, const double*); 
        int ldb =  va_arg(arglist, int); 
        double beta =  va_arg(arglist, double);
        double *C =  va_arg(arglist, double*); 
        int ldc =  va_arg(arglist, int);
        cublasDgemm((char)transa, (char)transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasCgemm):
      {
        int transa = va_arg(arglist, int); 
        int transb = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int);
        int k = va_arg(arglist, int); 
        cuComplex alpha =  va_arg(arglist, cuComplex);
        const cuComplex *A =  va_arg(arglist, const cuComplex*); 
        int lda =  va_arg(arglist, int); 
        const cuComplex *B =  va_arg(arglist, const cuComplex*); 
        int ldb =  va_arg(arglist, int); 
        cuComplex beta =  va_arg(arglist, cuComplex);
        cuComplex *C =  va_arg(arglist, cuComplex*); 
        int ldc =  va_arg(arglist, int);
        cublasCgemm((char)transa, (char)transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasZgemm):
      {
        int transa = va_arg(arglist, int); 
        int transb = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int);
        int k = va_arg(arglist, int); 
        cuDoubleComplex alpha =  va_arg(arglist, cuDoubleComplex);
        const cuDoubleComplex *A =  va_arg(arglist, const cuDoubleComplex*); 
        int lda =  va_arg(arglist, int); 
        const cuDoubleComplex *B =  va_arg(arglist, const cuDoubleComplex*); 
        int ldb =  va_arg(arglist, int); 
        cuDoubleComplex beta =  va_arg(arglist, cuDoubleComplex);
        cuDoubleComplex *C =  va_arg(arglist, cuDoubleComplex*); 
        int ldc =  va_arg(arglist, int);
        cublasZgemm((char)transa, (char)transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasSgemm_v2):
      {
        cublasStatus_t* orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t); 
        int transa = va_arg(arglist, int); 
        int transb = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int);
        int k = va_arg(arglist, int); 
        const float *alpha =  va_arg(arglist, const float*);
        const float *A =  va_arg(arglist, const float*); 
        int lda =  va_arg(arglist, int); 
        const float *B =  va_arg(arglist, const float*); 
        int ldb =  va_arg(arglist, int); 
        const float *beta =  va_arg(arglist, const float*);
        float *C =  va_arg(arglist, float*); 
        int ldc =  va_arg(arglist, int);
        *orig_ret = cublasSgemm_v2(handle, (cublasOperation_t)transa, (cublasOperation_t)transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
	    case GENERATE_ENUM(cublasSgemmStridedBatched):
      {
        cublasStatus_t* orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
        int transa = va_arg(arglist, int);
        cublasOperation_t tmp_transa = (cublasOperation_t)transa;
        int transb = va_arg(arglist, int);
        cublasOperation_t tmp_transb = (cublasOperation_t)transb;
        int m = va_arg(arglist, int);
        int n = va_arg(arglist, int);
        int k = va_arg(arglist, int);
        const float *alpha = va_arg(arglist, const float *);
        const float *A = va_arg(arglist, const float *);
        int lda = va_arg(arglist, int);
        long long int strideA = va_arg(arglist, long long int);
        const float *B = va_arg(arglist, const float *);
        int ldb = va_arg(arglist, int);
        long long int strideB = va_arg(arglist, long long int);
        const float *beta = va_arg(arglist, const float *);
        float *C = va_arg(arglist, float *);
        int ldc = va_arg(arglist, int);
        long long int strideC = va_arg(arglist, long long int);
        int batchCount = va_arg(arglist, int);
        *orig_ret = cublasSgemmStridedBatched(handle, tmp_transa, tmp_transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
        break;
      }
      case GENERATE_ENUM(cublasLtCreate):
      {
        cublasStatus_t* orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasLtHandle_t* handle = va_arg(arglist, cublasLtHandle_t*);
        *orig_ret = cublasLtCreate(handle);
        break;
      }
      case GENERATE_ENUM(cublasLtDestroy):
      {
        cublasStatus_t* orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasLtHandle_t handle = va_arg(arglist, cublasLtHandle_t);
        *orig_ret = cublasLtDestroy(handle);
        break;
      }
      case GENERATE_ENUM(cublasLtMatmul):
      {
        cublasStatus_t* orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasLtHandle_t lightHandle = va_arg(arglist, cublasLtHandle_t); 
        cublasLtMatmulDesc_t computeDesc = va_arg(arglist, cublasLtMatmulDesc_t); 
        const void *alpha = va_arg(arglist, const void*); 
        const void *A = va_arg(arglist, const void*); 
        cublasLtMatrixLayout_t Adesc = va_arg(arglist, cublasLtMatrixLayout_t); 
        const void *B = va_arg(arglist, const void*); 
        cublasLtMatrixLayout_t Bdesc = va_arg(arglist, cublasLtMatrixLayout_t); 
        const void *beta = va_arg(arglist, const void*); 
        const void *C = va_arg(arglist, const void*); 
        cublasLtMatrixLayout_t Cdesc = va_arg(arglist, cublasLtMatrixLayout_t); 
        void *D = va_arg(arglist, void*); 
        cublasLtMatrixLayout_t Ddesc = va_arg(arglist, cublasLtMatrixLayout_t); 
        const cublasLtMatmulAlgo_t *algo = va_arg(arglist, const cublasLtMatmulAlgo_t*); 
        void *workspace = va_arg(arglist, void*); 
        size_t workspaceSizeInBytes = va_arg(arglist, size_t); 
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *orig_ret = cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
        break;
      }
	  case GENERATE_ENUM(cublasLtMatmulAlgoGetHeuristic):
      {
        cublasStatus_t* orig_ret = va_arg(arglist, cublasStatus_t*);
        cublasLtHandle_t lightHandle = va_arg(arglist, cublasLtHandle_t);
        cublasLtMatmulDesc_t operationDesc = va_arg(arglist, cublasLtMatmulDesc_t);
        cublasLtMatrixLayout_t Adesc = va_arg(arglist, cublasLtMatrixLayout_t);
        cublasLtMatrixLayout_t Bdesc = va_arg(arglist, cublasLtMatrixLayout_t);
        cublasLtMatrixLayout_t Cdesc = va_arg(arglist, cublasLtMatrixLayout_t);
        cublasLtMatrixLayout_t Ddesc = va_arg(arglist, cublasLtMatrixLayout_t);
        cublasLtMatmulPreference_t preference = va_arg(arglist, cublasLtMatmulPreference_t);
        int requestedAlgoCount = va_arg(arglist, int);
        cublasLtMatmulHeuristicResult_t* heuristicResultsArray = va_arg(arglist, cublasLtMatmulHeuristicResult_t*);
        int *returnAlgoCount = va_arg(arglist, int*);
        *orig_ret = cublasLtMatmulAlgoGetHeuristic(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount);
        break;
      }
      case GENERATE_ENUM(cublasSsyrk):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const float *A = va_arg(arglist, const float*); 
        int lda = va_arg(arglist, int); 
        double beta = va_arg(arglist, double); 
        float *C = va_arg(arglist, float*);  
        int ldc = va_arg(arglist, int);
        cublasSsyrk((char)uplo, (char)trans, n, k, (float)alpha, A, lda, (float)beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasDsyrk):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const double *A = va_arg(arglist, const double*); 
        int lda = va_arg(arglist, int); 
        double beta = va_arg(arglist, double); 
        double *C = va_arg(arglist, double*);  
        int ldc = va_arg(arglist, int);
        cublasDsyrk((char)uplo, (char)trans, n, k, alpha, A, lda, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasCsyrk):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);  
        const cuComplex *A = va_arg(arglist, const cuComplex*); 
        int lda = va_arg(arglist, int); 
        cuComplex beta = va_arg(arglist, cuComplex); 
        cuComplex *C = va_arg(arglist, cuComplex*);  
        int ldc = va_arg(arglist, int);
        cublasCsyrk((char)uplo, (char)trans, n, k, alpha, A, lda, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasZsyrk):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);  
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex*); 
        int lda = va_arg(arglist, int); 
        cuDoubleComplex beta = va_arg(arglist, cuDoubleComplex); 
        cuDoubleComplex *C = va_arg(arglist, cuDoubleComplex*);  
        int ldc = va_arg(arglist, int);
        cublasZsyrk((char)uplo, (char)trans, n, k, alpha, A, lda, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasCherk):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const cuComplex *A = va_arg(arglist, const cuComplex*); 
        int lda = va_arg(arglist, int); 
        double beta = va_arg(arglist, double); 
        cuComplex *C = va_arg(arglist, cuComplex*);  
        int ldc = va_arg(arglist, int);
        cublasCherk((char)uplo, (char)trans, n, k, (float)alpha, A, lda, (float)beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasZherk):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex*); 
        int lda = va_arg(arglist, int); 
        double beta = va_arg(arglist, double); 
        cuDoubleComplex *C = va_arg(arglist, cuDoubleComplex*);  
        int ldc = va_arg(arglist, int);
        cublasZherk((char)uplo, (char)trans, n, k, alpha, A, lda, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasSsyr2k):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const float *A = va_arg(arglist, const float*); 
        int lda = va_arg(arglist, int);
        const float *B = va_arg(arglist, const float*); 
        int ldb = va_arg(arglist, int);
        double beta = va_arg(arglist, double); 
        float *C = va_arg(arglist, float*);  
        int ldc = va_arg(arglist, int);
        cublasSsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasDsyr2k):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const double *A = va_arg(arglist, const double*); 
        int lda = va_arg(arglist, int);
        const double *B = va_arg(arglist, const double*); 
        int ldb = va_arg(arglist, int);
        double beta = va_arg(arglist, double); 
        double *C = va_arg(arglist, double*);  
        int ldc = va_arg(arglist, int);
        cublasDsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasCsyr2k):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);  
        const cuComplex *A = va_arg(arglist, const cuComplex*); 
        int lda = va_arg(arglist, int);
        const cuComplex *B = va_arg(arglist, const cuComplex*); 
        int ldb = va_arg(arglist, int);
        cuComplex beta = va_arg(arglist, cuComplex); 
        cuComplex *C = va_arg(arglist, cuComplex*);  
        int ldc = va_arg(arglist, int);
        cublasCsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasZsyr2k):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);  
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex*); 
        int lda = va_arg(arglist, int);
        const cuDoubleComplex *B = va_arg(arglist, const cuDoubleComplex*); 
        int ldb = va_arg(arglist, int);
        cuDoubleComplex beta = va_arg(arglist, cuDoubleComplex); 
        cuDoubleComplex *C = va_arg(arglist, cuDoubleComplex*);  
        int ldc = va_arg(arglist, int);
        cublasZsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasCher2k):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);  
        const cuComplex *A = va_arg(arglist, const cuComplex*); 
        int lda = va_arg(arglist, int);
        const cuComplex *B = va_arg(arglist, const cuComplex*); 
        int ldb = va_arg(arglist, int);
        double beta = va_arg(arglist, double); 
        cuComplex *C = va_arg(arglist, cuComplex*);  
        int ldc = va_arg(arglist, int);
        cublasCher2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasZher2k):
      {
        int uplo = va_arg(arglist, int); 
        int trans = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        int k = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);  
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex*); 
        int lda = va_arg(arglist, int);
        const cuDoubleComplex *B = va_arg(arglist, const cuDoubleComplex*); 
        int ldb = va_arg(arglist, int);
        double beta = va_arg(arglist, double); 
        cuDoubleComplex *C = va_arg(arglist, cuDoubleComplex*);  
        int ldc = va_arg(arglist, int);
        cublasZher2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasSsymm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const float *A = va_arg(arglist, const float*); 
        int lda = va_arg(arglist, int);
        const float *B = va_arg(arglist, const float*); 
        int ldb = va_arg(arglist, int);
        double beta = va_arg(arglist, double); 
        float *C = va_arg(arglist, float*);  
        int ldc = va_arg(arglist, int);
        cublasSsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasDsymm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const double *A = va_arg(arglist, const double*); 
        int lda = va_arg(arglist, int);
        const double *B = va_arg(arglist, const double*); 
        int ldb = va_arg(arglist, int);
        double beta = va_arg(arglist, double); 
        double *C = va_arg(arglist, double*);  
        int ldc = va_arg(arglist, int);
        cublasDsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasCsymm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);  
        const cuComplex *A = va_arg(arglist, const cuComplex*); 
        int lda = va_arg(arglist, int);
        const cuComplex *B = va_arg(arglist, const cuComplex*); 
        int ldb = va_arg(arglist, int);
        cuComplex beta = va_arg(arglist, cuComplex); 
        cuComplex *C = va_arg(arglist, cuComplex*);  
        int ldc = va_arg(arglist, int);
        cublasCsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasZsymm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);  
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex*); 
        int lda = va_arg(arglist, int);
        const cuDoubleComplex *B = va_arg(arglist, const cuDoubleComplex*); 
        int ldb = va_arg(arglist, int);
        cuDoubleComplex beta = va_arg(arglist, cuDoubleComplex); 
        cuDoubleComplex *C = va_arg(arglist, cuDoubleComplex*);  
        int ldc = va_arg(arglist, int);
        cublasZsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasChemm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);  
        const cuComplex *A = va_arg(arglist, const cuComplex*); 
        int lda = va_arg(arglist, int);
        const cuComplex *B = va_arg(arglist, const cuComplex*); 
        int ldb = va_arg(arglist, int);
        cuComplex beta = va_arg(arglist, cuComplex); 
        cuComplex *C = va_arg(arglist, cuComplex*);  
        int ldc = va_arg(arglist, int);
        cublasChemm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasZhemm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);  
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex*); 
        int lda = va_arg(arglist, int);
        const cuDoubleComplex *B = va_arg(arglist, const cuDoubleComplex*); 
        int ldb = va_arg(arglist, int);
        cuDoubleComplex beta = va_arg(arglist, cuDoubleComplex); 
        cuDoubleComplex *C = va_arg(arglist, cuDoubleComplex*);  
        int ldc = va_arg(arglist, int);
        cublasZhemm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      }
      case GENERATE_ENUM(cublasStrsm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int);
        int transa = va_arg(arglist, int); 
        int diag = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const float *A = va_arg(arglist, const float*); 
        int lda = va_arg(arglist, int);
        float *B = va_arg(arglist, float*); 
        int ldb = va_arg(arglist, int);
        cublasStrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasDtrsm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int);
        int transa = va_arg(arglist, int); 
        int diag = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const double *A = va_arg(arglist, const double*); 
        int lda = va_arg(arglist, int);
        double *B = va_arg(arglist, double*); 
        int ldb = va_arg(arglist, int);
        cublasDtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasCtrsm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int);
        int transa = va_arg(arglist, int); 
        int diag = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);  
        const cuComplex *A = va_arg(arglist, const cuComplex*); 
        int lda = va_arg(arglist, int);
        cuComplex *B = va_arg(arglist, cuComplex*); 
        int ldb = va_arg(arglist, int);
        cublasCtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasZtrsm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int);
        int transa = va_arg(arglist, int); 
        int diag = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);  
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex*); 
        int lda = va_arg(arglist, int);
        cuDoubleComplex *B = va_arg(arglist, cuDoubleComplex*); 
        int ldb = va_arg(arglist, int);
        cublasZtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasStrmm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int);
        int transa = va_arg(arglist, int); 
        int diag = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const float *A = va_arg(arglist, const float*); 
        int lda = va_arg(arglist, int);
        float *B = va_arg(arglist, float*); 
        int ldb = va_arg(arglist, int);
        cublasStrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasDtrmm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int);
        int transa = va_arg(arglist, int); 
        int diag = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        double alpha = va_arg(arglist, double);  
        const double *A = va_arg(arglist, const double*); 
        int lda = va_arg(arglist, int);
        double *B = va_arg(arglist, double*); 
        int ldb = va_arg(arglist, int);
        cublasDtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasCtrmm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int);
        int transa = va_arg(arglist, int); 
        int diag = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuComplex alpha = va_arg(arglist, cuComplex);  
        const cuComplex *A = va_arg(arglist, const cuComplex*); 
        int lda = va_arg(arglist, int);
        cuComplex *B = va_arg(arglist, cuComplex*); 
        int ldb = va_arg(arglist, int);
        cublasCtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasZtrmm):
      {
        int side = va_arg(arglist, int); 
        int uplo = va_arg(arglist, int);
        int transa = va_arg(arglist, int); 
        int diag = va_arg(arglist, int); 
        int m = va_arg(arglist, int); 
        int n = va_arg(arglist, int); 
        cuDoubleComplex alpha = va_arg(arglist, cuDoubleComplex);  
        const cuDoubleComplex *A = va_arg(arglist, const cuDoubleComplex*); 
        int lda = va_arg(arglist, int);
        cuDoubleComplex *B = va_arg(arglist, cuDoubleComplex*); 
        int ldb = va_arg(arglist, int);
        cublasZtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasSetMatrix):
      {
        cublasStatus_t * ret = va_arg(arglist, cublasStatus_t *);
        int rows = va_arg(arglist, int); 
        int cols = va_arg(arglist, int); 
        int elemSize = va_arg(arglist, int);  
        const void *A = va_arg(arglist, const void*); 
        int lda = va_arg(arglist, int); 
        void *B = va_arg(arglist, void *);  
        int ldb = va_arg(arglist, int);
        *ret = cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasGetMatrix):
      {
        cublasStatus_t * ret = va_arg(arglist, cublasStatus_t *);
        int rows = va_arg(arglist, int); 
        int cols = va_arg(arglist, int); 
        int elemSize = va_arg(arglist, int);  
        const void *A = va_arg(arglist, const void*); 
        int lda = va_arg(arglist, int); 
        void *B = va_arg(arglist, void *);  
        int ldb = va_arg(arglist, int);
        *ret = cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
        break;
      }
      case GENERATE_ENUM(cublasSetMatrixAsync):
      {
        cublasStatus_t * ret = va_arg(arglist, cublasStatus_t *);
        int rows = va_arg(arglist, int); 
        int cols = va_arg(arglist, int); 
        int elemSize = va_arg(arglist, int);  
        const void *A = va_arg(arglist, const void*); 
        int lda = va_arg(arglist, int); 
        void *B = va_arg(arglist, void *);  
        int ldb = va_arg(arglist, int);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
        break;
      }
      case GENERATE_ENUM(cublasGetMatrixAsync):
      {
        cublasStatus_t * ret = va_arg(arglist, cublasStatus_t *);
        int rows = va_arg(arglist, int); 
        int cols = va_arg(arglist, int); 
        int elemSize = va_arg(arglist, int);  
        const void *A = va_arg(arglist, const void*); 
        int lda = va_arg(arglist, int); 
        void *B = va_arg(arglist, void *);  
        int ldb = va_arg(arglist, int);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
        break;
      }
      case GENERATE_ENUM(cublasSetVector):
      {
        cublasStatus_t * ret = va_arg(arglist, cublasStatus_t *);
        int n = va_arg(arglist, int);  
        int elemSize = va_arg(arglist, int);  
        const void *x = va_arg(arglist, const void*);  
        int incx = va_arg(arglist, int);  
        void *y = va_arg(arglist, void*);  
        int incy = va_arg(arglist, int);
        *ret = cublasSetVector(n, elemSize, x, incx, y, incy); 
        break;
      }
      case GENERATE_ENUM(cublasGetVector):
      {
        cublasStatus_t * ret = va_arg(arglist, cublasStatus_t *);
        int n = va_arg(arglist, int);  
        int elemSize = va_arg(arglist, int);  
        const void *x = va_arg(arglist, const void*);  
        int incx = va_arg(arglist, int);  
        void *y = va_arg(arglist, void*);  
        int incy = va_arg(arglist, int);
        *ret = cublasGetVector(n, elemSize, x, incx, y, incy); 
        break;
      }
      case GENERATE_ENUM(cublasSetVectorAsync):
      {
        cublasStatus_t * ret = va_arg(arglist, cublasStatus_t *);
        int n = va_arg(arglist, int);  
        int elemSize = va_arg(arglist, int);  
        const void *hostPtr = va_arg(arglist, const void*);  
        int incx = va_arg(arglist, int);  
        void *devicePtr = va_arg(arglist, void*);  
        int incy = va_arg(arglist, int);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream); 
        break;
      }
      case GENERATE_ENUM(cublasGetVectorAsync):
      {
        cublasStatus_t * ret = va_arg(arglist, cublasStatus_t *);
        int n = va_arg(arglist, int);  
        int elemSize = va_arg(arglist, int);  
        const void *hostPtr = va_arg(arglist, const void*);  
        int incx = va_arg(arglist, int);  
        void *devicePtr = va_arg(arglist, void*);  
        int incy = va_arg(arglist, int);
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = cublasGetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
        break;
      }
      case GENERATE_ENUM(cusparseCreate):
      {
        break;
      }
      case GENERATE_ENUM(cusparseSetStream):
      {
        break;
      }
      case GENERATE_ENUM(cusparseCreateMatDescr):
      {
        break;
      }
      case GENERATE_ENUM(cusparseSetMatType):
      {
        break;
      }
      case GENERATE_ENUM(cusparseSetMatIndexBase):
      {
        break;
      }
      case GENERATE_ENUM(cusparseDestroy):
      {
        break;
      }
      case GENERATE_ENUM(cusparseDestroyMatDescr):
      {
        break;
      }
      case GENERATE_ENUM(cusparseGetMatType):
      {
        break;
      }
      case GENERATE_ENUM(cusparseSetMatFillMode):
      {
        break;
      }
      case GENERATE_ENUM(cusparseGetMatFillMode):
      {
        break;
      }
      case GENERATE_ENUM(cusparseSetMatDiagType):
      {
        break;
      }
      case GENERATE_ENUM(cusparseGetMatDiagType):
      {
        break;
      }
      case GENERATE_ENUM(cusparseGetMatIndexBase):
      {
        break;
      }
      case GENERATE_ENUM(cusparseSetPointerMode):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnCreate):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnDestroy):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnSetStream):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnGetStream):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnDgetrf_bufferSize):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnDgetrf):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnDgetrs):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnDpotrf_bufferSize):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnDpotrf):
      {
        break;
      }
      case GENERATE_ENUM(cusolverDnDpotrs):
      {
        break;
      }
      case GENERATE_ENUM(ncclCommInitRank):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        ncclComm_t* comm = va_arg(arglist, ncclComm_t*); 
        int nranks = va_arg(arglist, int); 
        ncclUniqueId commId = va_arg(arglist, ncclUniqueId); 
        int rank = va_arg(arglist, int);

        *ret = ncclCommInitRank(comm, nranks, commId, rank);
        break;
      }
      case GENERATE_ENUM(ncclCommInitAll):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        ncclComm_t* comms = va_arg(arglist, ncclComm_t*);  
        int ndev = va_arg(arglist, int);  
        const int* devlist = va_arg(arglist, const int*);
        *ret = ncclCommInitAll(comms, ndev, devlist);
        break;
      }
      // case GENERATE_ENUM(ncclCommFinalize):
      // {
      //   ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
      //   ncclComm_t comm = va_arg(arglist, ncclComm_t);
      //   *ret = ncclCommFinalize(comm);
      //   break;
      // }
      case GENERATE_ENUM(ncclCommDestroy):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        ncclComm_t comm = va_arg(arglist, ncclComm_t);
        *ret = ncclCommDestroy(comm);
        break;
      }
      case GENERATE_ENUM(ncclCommAbort):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        ncclComm_t comm = va_arg(arglist, ncclComm_t);
        *ret = ncclCommAbort(comm);
        break;
      }
      case GENERATE_ENUM(ncclGetErrorString):
      {
        char *ret = va_arg(arglist, char*);
        int result = va_arg(arglist, int);
        ret = const_cast<char*>(ncclGetErrorString((ncclResult_t)result));
        UNUSED(ret);
        break;
      }
      // case GENERATE_ENUM(ncclGetLastError):
      // {
      //   char *ret = va_arg(arglist, char*);
      //   ncclComm_t comm = va_arg(arglist, ncclComm_t);
      //   ret = const_cast<char*>(ncclGetLastError(comm));
      //   UNUSED(ret);
      //   break;
      // }
      case GENERATE_ENUM(ncclCommGetAsyncError):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        ncclComm_t comm = va_arg(arglist, ncclComm_t);
        ncclResult_t *asyncError = va_arg(arglist, ncclResult_t *);
        *ret = ncclCommGetAsyncError(comm, asyncError);
        break;
      }
      case GENERATE_ENUM(ncclCommCount):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        const ncclComm* comm = va_arg(arglist, const ncclComm*);
        int* count = va_arg(arglist, int*);
        *ret = ncclCommCount(const_cast<ncclComm_t>(comm), count);
        break;
      }
      case GENERATE_ENUM(ncclCommCuDevice):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        const ncclComm* comm = va_arg(arglist, const ncclComm*);
        int* device = va_arg(arglist, int*);
        *ret = ncclCommCuDevice(const_cast<ncclComm_t>(comm), device);
        break;
      }
      case GENERATE_ENUM(ncclCommUserRank):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        const ncclComm* comm = va_arg(arglist, const ncclComm*);
        int* rank = va_arg(arglist, int*);
        *ret = ncclCommUserRank(const_cast<ncclComm_t>(comm), rank);
        break;
      }
      // case GENERATE_ENUM(ncclRedOpCreatePreMulSum):
      // {
      //   ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
      //   ncclRedOp_t *op = va_arg(arglist, ncclRedOp_t*); 
      //   void *scalar = va_arg(arglist, void*); 
      //   int datatype = va_arg(arglist, int); 
      //   int residence = va_arg(arglist, int); 
      //   ncclComm_t comm = va_arg(arglist, ncclComm_t);
      //   *ret = ncclRedOpCreatePreMulSum(op, scalar, (ncclDataType_t)datatype, (ncclScalarResidence_t)residence, comm);
      //   break;
      // }
      // case GENERATE_ENUM(ncclRedOpDestroy):
      // {
      //   ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
      //   int op = va_arg(arglist, int); 
      //   ncclComm_t comm = va_arg(arglist, ncclComm_t);
      //   *ret =  ncclRedOpDestroy((ncclRedOp_t)op, comm);
      //   break;
      // }
      case GENERATE_ENUM(ncclReduce):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        const void* sendbuff = va_arg(arglist, const void*); 
        void* recvbuff = va_arg(arglist, void*); 
        size_t count = va_arg(arglist, size_t); 
        int datatype = va_arg(arglist, int); 
        int op = va_arg(arglist, int); 
        int root = va_arg(arglist, int); 
        ncclComm_t comm = va_arg(arglist, ncclComm_t); 
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = ncclReduce(sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op, root, comm, stream);
        break;
      }
      case GENERATE_ENUM(ncclBcast):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        void* buff = va_arg(arglist, void*); 
        size_t count = va_arg(arglist, size_t); 
        int datatype = va_arg(arglist, int); 
        int root = va_arg(arglist, int); 
        ncclComm_t comm = va_arg(arglist, ncclComm_t); 
        cudaStream_t stream = va_arg(arglist, cudaStream_t);

        *ret = ncclBcast(buff, count, (ncclDataType_t)datatype, root, comm, stream);
        break;
      }
      case GENERATE_ENUM(ncclBroadcast):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        const void* sendbuff = va_arg(arglist, const void*); 
        void* recvbuff = va_arg(arglist, void*); 
        size_t count = va_arg(arglist, size_t); 
        int datatype = va_arg(arglist, int); 
        int root = va_arg(arglist, int); 
        ncclComm_t comm = va_arg(arglist, ncclComm_t); 
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = ncclBroadcast(sendbuff, recvbuff, count, (ncclDataType_t)datatype, root, comm, stream);
        break;
      }
      case GENERATE_ENUM(ncclAllReduce):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        const void* sendbuff = va_arg(arglist, const void*); 
        void* recvbuff = va_arg(arglist, void*); 
        size_t count = va_arg(arglist, size_t); 
        int datatype = va_arg(arglist, int); 
        int op = va_arg(arglist, int); 
        ncclComm_t comm = va_arg(arglist, ncclComm_t); 
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = ncclAllReduce(sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op, comm, stream);
        break;
      }
      case GENERATE_ENUM(ncclReduceScatter):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        const void* sendbuff = va_arg(arglist, const void*); 
        void* recvbuff = va_arg(arglist, void*); 
        size_t recvcount = va_arg(arglist, size_t); 
        int datatype = va_arg(arglist, int); 
        int op = va_arg(arglist, int); 
        ncclComm_t comm = va_arg(arglist, ncclComm_t); 
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = ncclReduceScatter(sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op, comm, stream);
        break;
      }
      case GENERATE_ENUM(ncclAllGather):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        const void* sendbuff = va_arg(arglist, const void*); 
        void* recvbuff = va_arg(arglist, void*); 
        size_t sendcount = va_arg(arglist, size_t); 
        int datatype = va_arg(arglist, int); 
        ncclComm_t comm = va_arg(arglist, ncclComm_t); 
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = ncclAllGather(sendbuff, recvbuff, sendcount, (ncclDataType_t)datatype, comm, stream);
        break;
      }
      case GENERATE_ENUM(ncclSend):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        const void* sendbuff = va_arg(arglist, const void*); 
        size_t sendcount = va_arg(arglist, size_t); 
        int datatype = va_arg(arglist, int); 
        int peer = va_arg(arglist, int);
        ncclComm_t comm = va_arg(arglist, ncclComm_t); 
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = ncclSend(sendbuff, sendcount, (ncclDataType_t)datatype, peer, comm, stream);
        break;
      }
      case GENERATE_ENUM(ncclRecv):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        void* recvbuff = va_arg(arglist, void*); 
        size_t sendcount = va_arg(arglist, size_t); 
        int datatype = va_arg(arglist, int); 
        int peer = va_arg(arglist, int);
        ncclComm_t comm = va_arg(arglist, ncclComm_t); 
        cudaStream_t stream = va_arg(arglist, cudaStream_t);
        *ret = ncclRecv(recvbuff, sendcount, (ncclDataType_t)datatype, peer, comm, stream);
        break;
      }
      case GENERATE_ENUM(ncclGroupStart):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        *ret = ncclGroupStart();
        break;
      }
      case GENERATE_ENUM(ncclGroupEnd):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        *ret = ncclGroupEnd();
        break;
      }
      case GENERATE_ENUM(ncclGetUniqueId):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        ncclUniqueId* uniqueId = va_arg(arglist, ncclUniqueId*);
        *ret = ncclGetUniqueId(uniqueId);
        break;
      }
      case GENERATE_ENUM(ncclGetVersion):
      {
        ncclResult_t *ret = va_arg(arglist, ncclResult_t*);
        int *version = va_arg(arglist, int*);
        *ret = ncclGetVersion(version);
        break;
      }
      default:
        break;
    }
    g_curThread = 0;
    bExecFinished = true;
    // printf("bExecFinished true\n");
    curOp =  Cuda_Fnc_Invalid;   
  }
  return nullptr;
}

void cudaFuncExec(Cuda_Fncs_t op, ...)
{
  fprintf(stderr, "cudaFuncExec wrong entrance\n");
  pthread_mutex_lock(&thread_mutex);
  va_start(arglist, op);
  pid_t uhThreadId = va_arg(arglist, pid_t);
  if(g_thread_maps.find(uhThreadId) == g_thread_maps.end())
  {
    pthread_t newThread;
    pthread_create(&newThread, NULL, &threadFunc, NULL);
    g_thread_maps[uhThreadId] = newThread;
    // printf("create new thread, tid:%ld\n", newThread);
  }
  curOp = op;
  g_curThread = g_thread_maps[uhThreadId];
  while(!bExecFinished)
  {
    usleep(10);
  }
  bExecFinished = false;
  va_end(arglist);
  pthread_mutex_unlock(&thread_mutex);
}

#if USE_COMP
void copy_lower_half_data_gpu_compress(int fd);

// for compression by tian01.liu 2022.7.5
static void doGpuCompression(void *uncompressedBuf, size_t size, void *compressedBuf, size_t &compressedSize);
static void doGpuCompressionMS(void *srcBufs[], size_t *srcLens, void *dstBufs[], size_t *dstLens, /*char* hMems[],*/ size_t mem_frag_nums);

// decompression for restore phase by tian01.liu 2022.8.11
static void doGpuDecompression(void *compressedBuf, size_t compressedSize, void *decompressedBuf, size_t &decompressSize);
#endif
static void doComputeHash(void *mem_addr, size_t mem_size, int *gpu_fd, pid_t pid);

static void doGpuIncreamtalCkpt(void *mem_addr, size_t mem_size);

// This function loads in ld.so, sets up a separate stack for it, and jumps
// to the entry point of ld.so
void runRtld()
{
  int rc = -1;

  // Pointer to the ld.so entry point
  void *ldso_entrypoint = NULL;

  // Load RTLD (ld.so)
  char *ldname = getenv("TARGET_LD");
  char *uhpreload = getenv("UH_PRELOAD");
  if (!ldname || !uhpreload)
  {
    printUsage();
    return;
  }

  DynObjInfo_t ldso = safeLoadLib(ldname);
  if (ldso.baseAddr == NULL || ldso.entryPoint == NULL)
  {
    DLOG(ERROR, "Error loading the runtime loader (%s). Exiting...\n", ldname);
    return;
  }

  DLOG(INFO, "New ld.so loaded at: %p\n", ldso.baseAddr);
  ldso_entrypoint = getEntryPoint(ldso);
  DLOG(INFO, "New ld.so ldso_entrypoint: %p\n", ldso_entrypoint);
  // Create new stack region to be used by RTLD
  Area kernelHeap;
  void *newStack = createNewStackForRtld(&ldso, &kernelHeap);
  if (!newStack)
  {
    DLOG(ERROR, "Error creating new stack for RTLD. Exiting...\n");
    exit(-1);
  }
  DLOG(INFO, "New stack start at: %p\n", newStack);

  // Create new heap region to be used by RTLD, the new heap will be fixed above the heap of kernel-loader
  void *newHeap = createNewHeapForRtld(&ldso, &kernelHeap);
  if (!newHeap)
  {
    DLOG(ERROR, "Error creating new heap for RTLD. Exiting...\n");
    exit(-1);
  }
  DLOG(INFO, "New heap mapped at: %p\n", newHeap);

  // insert a trampoline from ldso mmap address to mmapWrapper
  rc = insertTrampoline(ldso.mmapAddr, (void *)&mmapWrapper);
  if (rc < 0)
  {
    DLOG(ERROR, "Error inserting trampoline for mmap. Exiting...\n");
    exit(-1);
  }
  // insert a trampoline from ldso sbrk address to sbrkWrapper
  rc = insertTrampoline(ldso.sbrkAddr, (void *)&sbrkWrapper);
  if (rc < 0)
  {
    DLOG(ERROR, "Error inserting trampoline for sbrk. Exiting...\n");
    exit(-1);
  }

  // Everything is ready, let's set up the lower-half info struct for the upper
  // half to read from
  rc = setupLowerHalfInfo(false);
  if (rc < 0)
  {
    DLOG(ERROR, "Failed to set up lhinfo for the upper half. Exiting...\n");
    exit(-1);
  }

  // added by tian01.liu 2023.1.10. for new gpu memory management architecure.
  // TODO: initialize MMU and get the profile info, then write the profile to file.
  rc = setupMMUInfo(false);
  if (rc < 0)
  {
    DLOG(ERROR, "Failed to set up mmuinfo.Exiting...\n");
    exit(-1);
  }

  if (getenv("MPI_SUPPORTED") != nullptr &&
      strcmp(getenv("MPI_SUPPORTED"), "1") == 0)
  {
    // Run lower half of mpi
    splitProcess();
  }

  DLOG(INFO, "[CRAC] ldso_entrypoint = %p, newStack = %p\n", ldso_entrypoint, newStack);
  DLOG(ERROR, "[CRAC] ~~~~~ exec application \n");
  // Change the stack pointer to point to the new stack and jump into ld.so
  // TODO: Clean up all the registers?
  asm volatile (CLEAN_FOR_64_BIT(mov %0, %%esp; )
                : : "g" (newStack) : "memory");
  asm volatile ("jmp *%0" : : "g" (ldso_entrypoint) : "memory");
  DLOG(INFO, "[CRAC] ldso_entrypoint = %p\n", ldso_entrypoint);
}

// Local functions

static void
printUsage()
{
  DLOG(ERROR, "Usage: UH_PRELOAD=/path/to/libupperhalfwrappers.so "
              "TARGET_LD=/path/to/ld.so ./kernel-loader "
              "<target-application> [application arguments ...]\n");
}

static void
printRestartUsage()
{
  DLOG(ERROR, "Usage: ./kernel-loader --restore /path/to/ckpt.img\n");
}

clock_t startRestore;

void *pin_mem_blk_addr = nullptr;

// shift args
#define shift argc--, argv++
static void
processArgs(int argc, char **argv, char **environ)
{
  rinfo.argc = argc;
  rinfo.argv = argv;
  rinfo.environ = environ;
  rinfo.restartDir = NULL;
  rinfo.minLibsStart = NULL;
  rinfo.maxLibsEnd = NULL;
  rinfo.minHighMemStart = NULL;

  /**
    * argv[0] = kernel-loader.exe.
    * argv[1] = --restore
    * Call shitf twice to move to the correct offset of argv
    */
  shift; shift;
  while (argc > 0) {
    if (strcmp(argv[0], "--restartdir") == 0) {
      rinfo.restartDir = argv[1];
      shift; shift;
    } else if (strcmp(argv[0], "--minLibsStart") == 0) {
      rinfo.minLibsStart = (VA) mtcp_strtol(argv[1]);
      shift; shift;
    } else if (strcmp(argv[0], "--maxLibsEnd") == 0) {
      rinfo.maxLibsEnd = (VA) mtcp_strtol(argv[1]);
      shift; shift;
    } else if (strcmp(argv[0], "--minHighMemStart") == 0) {
      rinfo.minHighMemStart = (VA) mtcp_strtol(argv[1]);
      shift; shift;
    } else if (argc == 1) {
      // We would use MTCP_PRINTF, but it's also for output of util/readdmtcp.sh
      DLOG(INFO, "Considering '%s' as a ckpt image.\n", argv[0]);
      strcpy(rinfo.ckptImage, argv[0]);
      break;
    } else {
      printf("MTCP Internal Error\n");
    }
  }
}

// unsigned long testLhFs = 0;
// void* testThreadFnc(void*args)
// {
//   printf("test thread in kernel-loader....\n");
//   cpu_set_t mask;// cpu核的集合u

//   CPU_ZERO(&mask);// 将集合置为空集
//   CPU_SET(10,&mask);// 设置亲和力值
//   if(sched_setaffinity(0,sizeof(cpu_set_t),&mask)==-1)// 设置线程cpu亲和力
//   {
//       printf("warning: could not set CPU affinity, continuing...\n");
//   }

//   syscall(SYS_arch_prctl, ARCH_GET_FS, &testLhFs);
//   printf("FS in test thread, FS:%p\n", (void*)testLhFs);
//   fflush(stdout);

//   while(1)
//   {
//     printf("[lt]sleep in testThreadFnc of kernel-loader.exe.....\n");
//     fflush(stdout);
//   }

//   return NULL;
// }

int
main(int argc, char *argv[], char **environ)
{
  if (argc < 2)
  {
    printUsage();
    return -1;
  }

  pthread_mutex_init(&thread_mutex, NULL);
  pthread_rwlock_init(&mutex_update_dt, NULL);

  // added by tian01.liu 2023.1.10, for new gpu memory management architecture. This code won't run during restore
  init_mmu_allocator();

  if (strstr(argv[1], "--restore"))
  {
    if (argc < 3)
    {
      printRestartUsage();
      return -1;
    }
    DLOG(ERROR, "Return to dmtcp ...\n");
    // int ckptFd = atoi(argv[2]);
    int ckptFd = -1;
    int rc = setupLowerHalfInfo(true);
    if (rc < 0)
    {
      DLOG(ERROR, "Failed to set up lhinfo for the upper half. Exiting...\n");
      exit(-1);
    }

    // by tian01.liu for cuda ipc, 2023.8.15
    g_finishRelay = false;

    // Porcess arguments.
    processArgs(argc, argv, environ);
    // Find the real ckpt file and open it.
    if (getenv("MPI_SUPPORTED") != nullptr &&
        strcmp(getenv("MPI_SUPPORTED"), "1") == 0) {
        mtcp_plugin_hook(&rinfo);
        ckptFd = rinfo.fd;
    } else {
        ckptFd = atoi(argv[2]);
    }

    DLOG(ERROR, "Get the checkpoint file fd...\n");
    /*
     restoreCheckpoint will
     1. read the MtcpHeader
     2. restore the memory region of the application from ckpt image.
     3. return to the plugin code of the checkpoint thread.
    */
    // startRestore = clock();
    restoreCheckpointImg(ckptFd);
    // clock_t tmpClk = clock();
    // printf("[lt] restore Img Time:%.3f\n", (double)(tmpClk - startRestore) / CLOCKS_PER_SEC * 1000);
    // added by tian01.liu for bug of page-lock memory 2023.2.9
    // readUhInfoAddr();
    restore_page_lock_pool(nullptr, 0);

    DLOG(ERROR, "Return to dmtcp ...\n");
    returnTodmtcp();
    // Following line should not be reached.
    // dprintf(stderr_fd, "Restore failed!");
  }

  setupMMUInfo(false);
  // DLOG(ERROR, "GPU RESERVED PTR:%p\n", (void *)g_allocator->big_block_info.reserved_start_ptr);

  runRtld();
  return 0;
}

#if USE_COMP

#define MAX_FRAG_NUMS 30
static void
doGpuCompressionMS(void *srcBufs[], size_t *srcLens, void *dstBufs[], size_t *dstLens, /*char* hMems[] ,*/ size_t mem_frag_nums)
{
  DLOG(ERROR, "Enter doGpuCompressionMS...\n");
  // return;
  if (!mem_frag_nums)
    return;

  // JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  DLOG(ERROR, "Success jump to lower half...\n");

  int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;
  std::string comp_format = "bitcomp";

  cudaStream_t stream[MAX_FRAG_NUMS];
  std::shared_ptr<nvcompManagerBase> manager[MAX_FRAG_NUMS];
  int gpu_num = 0;

  // clock_t start_all,end_all;
  // start_all = clock();
  for (size_t i = 0; i < mem_frag_nums; i++)
  {
    cudaStreamCreate(&stream[i]);

    if (comp_format == "lz4")
    {
      manager[i] = std::make_shared<LZ4Manager>(chunk_size, data_type, stream[i], gpu_num, NoComputeNoVerify);
    }
    else if (comp_format == "snappy")
    {
      manager[i] = std::make_shared<SnappyManager>(chunk_size, stream[i], gpu_num, NoComputeNoVerify);
    }
    else if (comp_format == "bitcomp")
    {
      manager[i] = std::make_shared<BitcompManager>(data_type, 0, stream[i], gpu_num, NoComputeNoVerify);
    }
    else if (comp_format == "ans")
    {
      manager[i] = std::make_shared<ANSManager>(chunk_size, stream[i], gpu_num, NoComputeNoVerify);
    }
    auto compress_config = manager[i]->configure_compression(srcLens[i]);
    // size_t comp_out_bytes = compress_config.max_compressed_buffer_size;
    // printf("init out bytes: %ld\n", comp_out_bytes);
    manager[i]->compress((uint8_t *)srcBufs[i], (uint8_t *)dstBufs[i], compress_config);
    // printf("finish compress\n");
  }

  // double d_copy_time = 0;

  // cudaStream_t streamCopy[MAX_FRAG_NUMS];
  for (size_t i = 0; i < mem_frag_nums; i++)
  {
    cudaStreamSynchronize(stream[i]);
    // printf("finish stream sync\n");
    size_t comp_out_bytes = manager[i]->get_compressed_output_size((uint8_t *)dstBufs[i]);
    // printf("real out bytes: %ld\n", comp_out_bytes);
    // printf("orignal size:%ld, after compressed size:%ld\n", srcLens[i], comp_out_bytes);
    dstLens[i] = comp_out_bytes;
    cudaStreamDestroy(stream[i]);

    // cudaStreamCreate(&streamCopy[i]);
    // clock_t start,end;
    // start = clock();
    // cudaMemcpyAsync(hMems[i], dstBufs[i], comp_out_bytes, cudaMemcpyDeviceToHost, streamCopy[i]);
    // end = clock();
    // d_copy_time += (end - start);
  }

  // for(size_t i = 0; i < mem_frag_nums; i++)
  //{
  // cudaStreamSynchronize(streamCopy[i]);
  // cudaStreamDestroy(streamCopy[i]);
  // }
  // end_all = clock();

  // double d_comp_time = end_all - start_all - d_copy_time;

  // printf("compress_time:%.3f  cpy_time:%.3f\n", d_comp_time / CLOCKS_PER_SEC * 1000, d_copy_time / CLOCKS_PER_SEC * 1000);

  // RETURN_TO_UPPER_HALF();
  DLOG(ERROR, "return to upper halt\n");
}

/********************************
 * this function used to do gpu compression
 * by tian01.liu 2022.7.5
 * *****************************/
static void
doGpuCompression(void *uncompressedBuf, size_t size, void *compressedBuf, size_t &compressedSize)
{
  // void *ret = MAP_FAILED;
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  // ret = __mmapWrapper(addr, length, prot, flags, fd, offset);
  // RETURN_TO_UPPER_HALF();
  // compress_func_wrapper(NULL, 0, NULL, 0, 0);

  DLOG(INFO, "Enter doGpuCompress in lower half...\n");
  if (!uncompressedBuf || !size)
    return;

  int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;
  std::string comp_format = "bitcomp";

  int gpu_num = 0;
  // cudaSetDevice(gpu_num);
  DLOG(INFO, "Finish cudaSetDevice\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // DLOG(INFO, "Finish cudaStreamCreate\n");
  std::shared_ptr<nvcompManagerBase> manager;
  if (comp_format == "lz4")
  {
    manager = std::make_shared<LZ4Manager>(chunk_size, data_type, stream, gpu_num, NoComputeNoVerify);
  }
  else if (comp_format == "snappy")
  {
    manager = std::make_shared<SnappyManager>(chunk_size, stream, gpu_num, NoComputeNoVerify);
  }
  else if (comp_format == "bitcomp")
  {
    manager = std::make_shared<BitcompManager>(data_type, 0, stream, gpu_num, NoComputeNoVerify);
    // DLOG(ERROR, "Finish bitcomp manager create...\n");
  }
  else if (comp_format == "ans")
  {
    // DLOG(ERROR, "Create ans manager...\n");
    manager = std::make_shared<ANSManager>(chunk_size, stream, gpu_num, NoComputeNoVerify);
  }

  // uint8_t* testPtr = 0;
  // size_t testSize = 102400;
  // cudaMalloc(&testPtr, testSize);
  // auto compress_config = manager->configure_compression(102400);
  // size_t comp_out_bytes = compress_config.max_compressed_buffer_size;

  auto compress_config = manager->configure_compression(size);
  size_t comp_out_bytes = compress_config.max_compressed_buffer_size;
  printf("init out bytes: %ld\n", comp_out_bytes);
  manager->compress((uint8_t *)uncompressedBuf, /*d_comp_out*/ (uint8_t *)compressedBuf, compress_config);
  DLOG(ERROR, "finish compress\n");
  cudaStreamSynchronize(stream);
  DLOG(ERROR, "finish stream sync\n");
  comp_out_bytes = manager->get_compressed_output_size(/*d_comp_out*/ (uint8_t *)compressedBuf);
  // printf("real out bytes: %ld\n", comp_out_bytes);
  // printf("orignal size:%ld, after compressed size:%ld\n", size, comp_out_bytes);
  compressedSize = comp_out_bytes;
  cudaStreamDestroy(stream);

  RETURN_TO_UPPER_HALF();
  DLOG(ERROR, "return to upper halt\n");
}

/********************************
 * this function used to do gpu decompression
 * by tian01.liu 2022.8.11
 * *****************************/
static void
doGpuDecompression(void *compressedBuf, size_t compressedSize, void *decompressedBuf, size_t &decompressSize)
{
  // JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  DLOG(INFO, "Enter doGpuDecompress in kernel-loader...\n");
  if (!compressedBuf || !compressedSize || !decompressedBuf)
    return;

  int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;
  std::string comp_format = "bitcomp";

  int gpu_num = 0;
  // cudaSetDevice(gpu_num);
  // DLOG(INFO, "Finish cudaSetDevice\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::shared_ptr<nvcompManagerBase> manager;
  if (comp_format == "lz4")
  {
    manager = std::make_shared<LZ4Manager>(chunk_size, data_type, stream, gpu_num, NoComputeNoVerify);
  }
  else if (comp_format == "snappy")
  {
    manager = std::make_shared<SnappyManager>(chunk_size, stream, gpu_num, NoComputeNoVerify);
  }
  else if (comp_format == "bitcomp")
  {
    manager = std::make_shared<BitcompManager>(data_type, 0, stream, gpu_num, NoComputeNoVerify);
  }
  else if (comp_format == "ans")
  {
    manager = std::make_shared<ANSManager>(chunk_size, stream, gpu_num, NoComputeNoVerify);
  }
  // printf("[LT] decompress object init success...\n");
  fflush(stdout);
  size_t comp_scratch_bytes = manager->get_required_scratch_buffer_size();
  uint8_t *d_comp_scratch;
  cudaMalloc(&d_comp_scratch, comp_scratch_bytes);
  manager->set_scratch_buffer(d_comp_scratch);

  auto decomp_config = manager->configure_decompression((uint8_t *)compressedBuf);
  // decompress size estimate
  const size_t decomp_bytes = decomp_config.decomp_data_size;
  // void* dst_ptr = 0;
  // cudaMalloc(&dst_ptr, decomp_bytes);
  decompressSize = decomp_bytes;
  // printf("[LT] decompress config init success...\n");
  fflush(stdout);
  manager->decompress((uint8_t *)decompressedBuf, (uint8_t *)compressedBuf, decomp_config);
  // printf("[LT] decompress success...\n");
  fflush(stdout);
  cudaStreamSynchronize(stream);

  // printf("[LT] stream synchronize success...\n");
  fflush(stdout);
  // cudaMemcpy(compressedBuf, dst_ptr, decomp_bytes, cudaMemcpyDeviceToDevice);

  cudaStreamDestroy(stream);
  // printf("[LT] stream destroy success...\n");
  fflush(stdout);

  // decompressSize = decomp_bytes;
  // printf("[LT] set decompressSize success...\n");
  fflush(stdout);
}

#endif

static void
doComputeHash(void *mem_addr, size_t mem_size, int *gpu_fd, pid_t pid)
{
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);

  inc_obj.address_size_map[mem_addr] = mem_size;
  std::string hash;
  hash = compute_hash((int *)mem_addr, mem_size);
  bool cond = inc_obj.UpdateGpuStatus(mem_addr, mem_size, hash);
  if (cond)
  {
    *gpu_fd = -1;
  }
  else
  {
    *gpu_fd = inc_obj.GetCkptFile(mem_addr, pid);
  }
  RETURN_TO_UPPER_HALF();
}

static void
doGpuIncreamtalCkpt(void *mem_addr, size_t mem_size)
{
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  // char *cpu_mem_addr = (char *)malloc(mem_size);
  // cuMemcpyDtoH(cpu_mem_addr, (CUdeviceptr)mem_addr, mem_size);
  // write(tmp_gpu_fd, &mem_addr, sizeof(mem_addr));
  // write(tmp_gpu_fd, &mem_size, sizeof(mem_size));
  // write(tmp_gpu_fd, cpu_mem_addr, mem_size);
  // free(cpu_mem_addr);
  // close(tmp_gpu_fd);
  RETURN_TO_UPPER_HALF();
}

// Returns the /proc/self/stat entry in the out string (of length len)
static void
getProcStatField(enum Procstat_t type, char *out, size_t len)
{
  const char *procPath = "/proc/self/stat";
  char sbuf[1024] = {0};
  int field_counter = 0;
  char *field_str = NULL;
  int fd, num_read;

  fd = open(procPath, O_RDONLY);
  if (fd < 0)
  {
    DLOG(ERROR, "Failed to open %s. Error: %s\n", procPath, strerror(errno));
    return;
  }

  num_read = read(fd, sbuf, sizeof sbuf - 1);
  close(fd);
  if (num_read <= 0)
    return;
  sbuf[num_read] = '\0';

  field_str = strtok(sbuf, " ");
  while (field_str && field_counter != type)
  {
    field_str = strtok(NULL, " ");
    field_counter++;
  }

  if (field_str)
  {
    strncpy(out, field_str, len);
  }
  else
  {
    DLOG(ERROR, "Failed to parse %s.\n", procPath);
  }
}

// Returns the [stack] area by reading the proc maps
static void
getStackRegion(Area *stack, Area *heap) // OUT
{
  Area area;
  int mapsfd = open("/proc/self/maps", O_RDONLY);
  while (readMapsLine(mapsfd, &area))
  {
    // Get the heap area of proxy
    if (strstr(area.name, "[heap]"))
    {
      *heap = area;
      continue;
    } 

    // Get the stack area of proxy, the second condition is useless.
    if (strstr(area.name, "[stack]") && area.endAddr >= (VA)&area)
    {
      *stack = area;
      break;
    }
  }
  close(mapsfd);
}

// Given a pointer to aux vector, parses the aux vector, and patches the
// following three entries: AT_PHDR, AT_ENTRY, and AT_PHNUM
static void
patchAuxv(ElfW(auxv_t) * av, unsigned long phnum,
          unsigned long phdr, unsigned long entry)
{
  for (; av->a_type != AT_NULL; ++av)
  {
    switch (av->a_type)
    {
    case AT_PHNUM:
      av->a_un.a_val = phnum;
      break;
    case AT_PHDR:
      av->a_un.a_val = phdr;
      break;
    case AT_ENTRY:
      av->a_un.a_val = entry;
      break;
    case AT_RANDOM:
      DLOG(NOISE, "AT_RANDOM value: 0%lx\n", av->a_un.a_val);
      break;
    default:
      break;
    }
  }
}

// Creates a deep copy of the stack region pointed to be `origStack` at the
// location pointed to be `newStack`. Returns the start-of-stack pointer
// in the new stack region.
static void *
deepCopyStack(void *newStack, const void *origStack, size_t len,
              const void *newStackEnd, const void *origStackEnd,
              const DynObjInfo_t *info)
{
  // This function assumes that this env var is set.
  assert(getenv("TARGET_LD"));
  assert(getenv("UH_PRELOAD"));

  // Return early if any pointer is NULL
  if (!newStack || !origStack ||
      !newStackEnd || !origStackEnd ||
      !info)
  {
    return NULL;
  }

  // First, we do a shallow copy, which is essentially, just copying the
  // bits from the original stack into the new stack.
  memcpy(newStack, origStack, len);

  // Next, turn the shallow copy into a deep copy.
  //
  // The main thing we need to do is to patch the argv and env vectors in
  // the new stack to point to addresses in the new stack region. Note that
  // the argv and env are simply arrays of pointers. The pointers point to
  // strings in other locations in the stack.

  void *origArgcAddr = (void *)GET_ARGC_ADDR(origStackEnd);
  int origArgc = *(int *)origArgcAddr;
  char **origArgv = (char **)GET_ARGV_ADDR(origStackEnd);
  const char **origEnv = (const char **)GET_ENV_ADDR(origArgv, origArgc);

  void *newArgcAddr = (void *)GET_ARGC_ADDR(newStackEnd);
  int newArgc = *(int *)newArgcAddr;
  char **newArgv = (char **)GET_ARGV_ADDR(newStackEnd);
  const char **newEnv = (const char **)GET_ENV_ADDR(newArgv, newArgc);
  ElfW(auxv_t) *newAuxv = GET_AUXV_ADDR(newEnv);

  // Patch the argv vector in the new stack
  //   First, set up the argv vector based on the original stack
  for (int i = 0; origArgv[i] != NULL; i++)
  {
    off_t argvDelta = (uintptr_t)origArgv[i] - (uintptr_t)origArgv;
    newArgv[i] = (char *)((uintptr_t)newArgv + (uintptr_t)argvDelta);
  }

  //   Next, we patch argv[0], the first argument, on the new stack
  //   to point to "/path/to/ld.so".
  //
  //   From the point of view of ld.so, it would appear as if it was called
  //   like this: $ /lib/ld.so /path/to/target.exe app-args ...
  //
  //   NOTE: The kernel loader needs to be called with at least two arguments
  //   to get a stack that is 16-byte aligned at the start. Since we want to
  //   be able to jump into ld.so with at least two arguments (ld.so and the
  //   target exe) on the new stack, we also need two arguments on the
  //   original stack.
  //
  //   If the original stack had just one argument, we would have inherited
  //   that alignment in the new stack. Trying to push in another argument
  //   (target exe) on the new stack would destroy the 16-byte alignment
  //   on the new stack. This would lead to a crash later on in ld.so.
  //
  //   The problem is that there are instructions (like, "movaps") in ld.so's
  //   code that operate on the stack memory region and require their
  //   operands to be 16-byte aligned. A non-16-byte-aligned operand (for
  //   example, the stack base pointer) leads to a general protection
  //   exception (#GP), which translates into a segfault for the user
  //   process.
  //
  //   The Linux kernel ensures that the start of stack is always 16-byte
  //   aligned. It seems like this is part of the Linux kernel x86-64 ABI.
  //   For example, see here:
  //
  //     https://elixir.bootlin.com/linux/v4.18.11/source/fs/binfmt_elf.c#L150
  //
  //     https://elixir.bootlin.com/linux/v4.18.11/source/fs/binfmt_elf.c#L288
  //
  //   (The kernel uses the STACK_ROUND macro to first set up the stack base
  //    at a 16-byte aligned address, and then pushes items on the stack.)
  //
  //   We could do something similar on the new stack region. But perhaps it's
  //   easier to just depend on the original stack having at least two args:
  //   "/path/to/kernel-loader" and "/path/to/target.exe".
  //
  //   NOTE: We don't need to patch newArgc, since the original stack,
  //   from where we would have inherited the data in the new stack, already
  //   had the correct value for origArgc. We just make argv[0] in the
  //   new stack to point to "/path/to/ld.so", instead of
  //   "/path/to/kernel-loader".
  off_t argvDelta = (uintptr_t)getenv("TARGET_LD") - (uintptr_t)origArgv;
  newArgv[0] = (char *)((uintptr_t)newArgv + (uintptr_t)argvDelta);

  // Patch the env vector in the new stack
  for (int i = 0; origEnv[i] != NULL; i++)
  {
    off_t envDelta = (uintptr_t)origEnv[i] - (uintptr_t)origEnv;
    newEnv[i] = (char *)((uintptr_t)newEnv + (uintptr_t)envDelta);
  }

  // Change "UH_PRELOAD" to "LD_PRELOAD". This way, upper half's ld.so
  // will preload the upper half wrapper library.
  char **newEnvPtr = (char **)newEnv;
  for (; *newEnvPtr; newEnvPtr++)
  {
    if (strstr(*newEnvPtr, "UH_PRELOAD"))
    {
      (*newEnvPtr)[0] = 'L';
      (*newEnvPtr)[1] = 'D';
      break;
    }
  }

  // The aux vector, which we would have inherited from the original stack,
  // has entries that correspond to the kernel loader binary. In particular,
  // it has these entries AT_PHNUM, AT_PHDR, and AT_ENTRY that correspond
  // to kernel-loader. So, we atch the aux vector in the new stack to
  // correspond to the new binary: the freshly loaded ld.so.
  patchAuxv(newAuxv, info->phnum,
            (uintptr_t)info->phdr,
            (uintptr_t)info->entryPoint);

  DLOG(INFO, "newArgv[-2]: 0x%zx \n", (unsigned long)&newArgv[-2]);

  // We clear out the rest of the new stack region just in case ...
  memset(newStack, 0, (size_t)((uintptr_t)&newArgv[-2] - (uintptr_t)newStack));

  // Return the start of new stack.
  return (void *)newArgcAddr;
}

// This function does three things:
//  1. Creates a new stack region to be used for initialization of RTLD (ld.so)
//  2. Deep copies the original stack (from the kernel) in the new stack region
//  3. Returns a pointer to the beginning of stack in the new stack region
static void *
createNewStackForRtld(const DynObjInfo_t *info, Area *heapArea)
{
  Area stack;
  char stackEndStr[20] = {0};
  /**
   * @brief
   * The new stack of target application is created by kernel-loader, the size of the new stack
   * is same as the kernel-loader's. When CRAC checkpoints the target application.
   * When the size of target application's stack exceeds kernel-loader's, the target application
   * can not running normally, so I increase the size of kernel-loader's stack to 6MB.
   */
  char strtab[7 * 1024 * 1024];
  memset(strtab, 0, 7 * 1024 * 1024);

  getStackRegion(&stack, heapArea);
  // stack.size = 8*1024*1024;
  // 1. Allocate new stack region
  // We go through the mmap wrapper function to ensure that this gets added
  // to the list of upper half regions to be checkpointed.
  void *newStack = mmapWrapper(NULL, stack.size, PROT_READ | PROT_WRITE,
                               MAP_GROWSDOWN | MAP_PRIVATE | MAP_ANONYMOUS,
                               -1, 0);
  if (newStack == MAP_FAILED)
  {
    DLOG(ERROR, "Failed to mmap new stack region: %s\n", strerror(errno));
    return NULL;
  }
  DLOG(INFO, "New stack mapped at: %p, size = 0x%zx\n", newStack, stack.size);
  // 3. Get pointer to the beginning of the stack in the new stack region
  // The idea here is to look at the beginning of stack in the original
  // stack region, and use that to index into the new memory region. The
  // same offsets are valid in both the stack regions.
  getProcStatField(STARTSTACK, stackEndStr, sizeof stackEndStr);

  // NOTE: The kernel sets up the stack in the following format.
  //      -1(%rsp)                       Stack end for application
  //      0(%rsp)                        argc (Stack start for application)
  //      LP_SIZE(%rsp)                  argv[0]
  //      (2*LP_SIZE)(%rsp)              argv[1]
  //      ...
  //      (LP_SIZE*(argc))(%rsp)         NULL
  //      (LP_SIZE*(argc+1))(%rsp)       envp[0]
  //      (LP_SIZE*(argc+2))(%rsp)       envp[1]
  //      ...
  //                                     NULL
  //
  // NOTE: proc-stat returns the address of argc on the stack.
  // argv[0] is 1 LP_SIZE ahead of argc, i.e., startStack + sizeof(void*)
  // Stack End is 1 LP_SIZE behind argc, i.e., startStack - sizeof(void*)
  // sizeof(unsigned long) == sizeof(void*) == 8 on x86-64

  DLOG(INFO, "stackEndStr: %s\n", stackEndStr);

  unsigned long origStackEnd = atol(stackEndStr) - sizeof(unsigned long);
  unsigned long origStackOffset = origStackEnd - (unsigned long)stack.addr;
  unsigned long newStackOffset = origStackOffset;
  void *newStackEnd = (void *)((unsigned long)newStack + newStackOffset);

  // add by tian01.liu for store the stack of application 2023.2.17
  g_stack_seg_addr = newStack;

  DLOG(INFO, "origStack: 0x%zx origStackOffset: 0x%zx OrigStackEnd: 0x%zx \n", (unsigned long)stack.addr, (unsigned long)origStackOffset, (unsigned long)origStackEnd);
  DLOG(INFO, "newStack: 0x%zx newStackOffset: 0x%zx newStackEnd: 0x%zx \n", (unsigned long)newStack, (unsigned long)newStackOffset, (unsigned long)newStackEnd);

  // 2. Deep copy stack
  newStackEnd = deepCopyStack(newStack, stack.addr, stack.size,
                              (void *)newStackEnd, (void *)origStackEnd,
                              info);

  return newStackEnd;
}

// This function allocates a new heap for (the possibly second) ld.so.
// The initial heap size is 1 page
//
// Returns the start address of the new heap on success, or NULL on
// failure.
static void *
createNewHeapForRtld(const DynObjInfo_t *info, Area *heap)
{
  const uint64_t heapSize = 200000 * PAGE_SIZE;
  
  // TODO: create the user heap near the top of kernel-loader heap and fixed its start address
  const uint64_t offset = 1024LL * 1024 * 100;
  VA newHeapAddr = heap->endAddr + offset;
  // We go through the mmap wrapper function to ensure that this gets added
  // to the list of upper half regions to be checkpointed.
  void *addr = mmapWrapper(newHeapAddr, heapSize, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  if (addr == MAP_FAILED)
  {
    DLOG(ERROR, "Failed to mmap region. Error: %s\n",
         strerror(errno));
    return NULL;
  }
  // Add a guard page before the start of heap; this protects
  // the heap from getting merged with a "previous" region.
  mprotect(addr, PAGE_SIZE, PROT_NONE);
  setUhBrk((void *)((VA)addr + PAGE_SIZE));
  setEndOfHeap((void *)((VA)addr + heapSize));
  return addr;
}

// This function returns the entry point of the ld.so executable given
// the library handle
static void *
getEntryPoint(DynObjInfo_t info)
{
  return info.entryPoint;
}

// Writes out the lhinfo global object to a file. Returns 0 on success,
// -1 on failure.
static int
writeLhInfoToFile()
{
  size_t rc = 0;
  char filename[100];
  snprintf(filename, 100, "./lhInfo_%d", getpid());
  int fd = open(filename, O_WRONLY | O_CREAT, 0644);
  if (fd < 0)
  {
    DLOG(ERROR, "Could not create addr.bin file. Error: %s", strerror(errno));
    return -1;
  }

  rc = write(fd, &lhInfo, sizeof(lhInfo));
  if (rc < sizeof(lhInfo))
  {
    DLOG(ERROR, "Wrote fewer bytes than expected to addr.bin. Error: %s",
         strerror(errno));
    rc = -1;
  }
  close(fd);
  return rc;
}

clock_t getTimeRestore()
{
  return startRestore;
}

// Sets up lower-half info struct for the upper half to read from. Returns 0
// on success, -1 otherwise
static int
setupLowerHalfInfo(bool isReplay)
{
  lhInfo.lhSbrk = (void *)&sbrkWrapper;
  lhInfo.lhMmap = (void *)&mmapWrapper;
  lhInfo.lhMunmap = (void *)&munmapWrapper;
  lhInfo.lhDlsym = (void *)&lhDlsym;

  // add by tian01.liu for bug of page-lock memory
  lhInfo.lhPLMPGetFptr = (void *)&getPinMemPoolAddr;
  lhInfo.lhAllocHostFptr = (void *)&allocate_page_lock_mem;
  lhInfo.lhFreeHostFptr = (void *)&free_page_lock_mem;
  lhInfo.lhRegisterHostFptr = (void *)&restore_page_lock_pool;
  lhInfo.lhInitPageLockPoolFptr = (void *)&init_page_lock_pool;

  lhInfo.lhMmapListFptr = (void *)&getMmappedList;
  lhInfo.uhEndofHeapFptr = (void *)&getEndOfHeap;
  lhInfo.readUhInfoAddrHandle = (void *)&readUhInfoAddr;
  lhInfo.logs_read_and_apply_handle = (void *)&replayAPI;
  lhInfo.copy_lower_half_data_handle = (void *)&copy_lower_half_data;
  lhInfo.refill_handles = (void *)&refill_handles;

  // added by tian01.liu 2023.1.10
  lhInfo.realloc_gpu_blocks_handle = (void *)&realloc_gpu_blocks;
  lhInfo.lhGetMMUProfileFptr = (void *)&get_mmu_profile_from_lh;
  lhInfo.lhMMUAllocFptr = (void *)&allocate_gpu_mem;
  lhInfo.lhMMUFreeFptr = (void *)&free_gpu_mem;
  lhInfo.lhReleaseMMUFptr = (void *)&release_mmu_allocator;
  lhInfo.lhReInitMMUFptr = (void *)&reinit_mmu_space;
  // lhInfo.lhGetTimeRestore = (void*)&getTimeRestore;
  lhInfo.lhGetAppStackSegFptr = (void *)&getApplicationStackAddr;
  lhInfo.lhExecApiFptr = (void*)&cudaFuncExec;
  lhInfo.lhGetMemCtxFptr = (void*)&getDevMemCtx;
  lhInfo.lhGetIpcInfoFptr = (void*)&getIpcInfo;
  lhInfo.lhIpcCloseApiEx = (void*)&cudaIpcCloseMemHandleEx;

  // add by tian01.liu 2023.9.5
  lhInfo.lhNcclTypeFptr = (void*)&getNickleType;
  lhInfo.lhFreeNcclTypeFptr = (void*)freeNickleType;


  // fengtao.xie added  7/8
  lhInfo.getFatCubinHandle = (void *)&fatHandle;
  lhInfo.lhCUtypeFptr = (void *)&getCUtype;
  lhInfo.lhFreeCUtypeFptr = (void *)&freeCUtype;
  lhInfo.lhCUDAtypeFptr = (void *)&getCudaType;
  lhInfo.lhFreeCUDATypeFptr = (void *)&freeCudaType;
  
  // by lu77.wei for ai
  lhInfo.lhCublasTypeFptr = (void*)&getCublasType;
  lhInfo.lhFreeCublasTypeFptr = (void*)&freeCublasType;

#if USE_COMP
  // add by tian01.liu 9/17
  lhInfo.lhCompressFptr = (void *)&doGpuCompression;
  lhInfo.lhCompressMSFptr = (void *)&doGpuCompressionMS;
  lhInfo.lhDecompressFptr = (void *)&doGpuDecompression;
#endif
  // for increamental ckpt
  lhInfo.lhIncreamentalCkpt = (void *)&doGpuIncreamtalCkpt;
  lhInfo.lhComputeHash = (void *)&doComputeHash;

  // add by tian01.liu 2023.10.28
  lhInfo.lhNewThreadFptr = (void*)&newThread;
  lhInfo.lhGetFsFptr = (void*)&getFs;
  lhInfo.lhWakeFptr = (void*)setWakeupCondition;
  lhInfo.lhWaitFnhFptr = (void*)waitLhApiExecFinish;
  lhInfo.isReplay = isReplay;

  if (syscall(SYS_arch_prctl, ARCH_GET_FS, &lhInfo.lhFsAddr) < 0)
  {
    DLOG(ERROR, "Could not retrieve lower half's fs. Error: %s. Exiting...\n",
         strerror(errno));
    return -1;
  }

  // FIXME: We'll just write out the lhInfo object to a file; the upper half
  // will read this file to figure out the wrapper addresses. This is ugly
  // but will work for now.
  int rc = writeLhInfoToFile();
  if (rc < 0)
  {
    DLOG(ERROR, "Error writing address of lhinfo to file. Exiting...\n");
    return -1;
  }
  return 0;
}

// fengtao.xie added
extern "C"
{
  void readUhInfoAddr()
  {
    char filename[100];
    // snprintf(filename, 100, "./uhInfo_%d", getpid());
    pid_t orig_pid = getUhPid();
    snprintf(filename, 100, "./uhInfo_%d", orig_pid);
    int fd = open(filename, O_RDONLY);
    if (fd < 0)
    {
      printf("Could not open upper-half file for reading. %s \n", filename);
      exit(-1);
    }
    ssize_t rc = read(fd, &uhInfo, sizeof(uhInfo));
    if (rc != (ssize_t)sizeof(uhInfo))
    {
      perror("Read fewer bytes than expected from uhaddr.bin.");
      exit(-1);
    }

    g_stack_seg_addr = uhInfo.appStackAddr;
    // printf("Finished read the UhInfo\n");
    //  unlink(UH_FILE_NAME);
    //  close(fd);
  }
}

#if USE_COMP

void copy_lower_half_data_gpu_compress(int fd)
{
  fprintf(stderr, "enter copy_lower_half_data_gpu_compress....fd:%i\n", fd);
  fflush(stderr);

  // TODO: modify the lower_half data handle method used
  if (fd == -1)
  {
    // printf("[LT] Enter lower half data handling...\n");
    char filename[100];
    pid_t orig_pid = getUhPid();
    snprintf(filename, 100, "./gpu_mem_data_%d", orig_pid);
    printf("gpu data filename is %s\n", filename);
    fd = open(filename, O_RDONLY);
  }
  double dCudaMallocTime = 0;
  double dCompTime = 0;
  double dMemcpyHtoD = 0;
  double dMemcpyDtoD = 0;
  double dReadFile = 0;
  double dMallocTime = 0;
  double dReadPageTime = 0;
  clock_t start, end;

  lhckpt_pages_t lhpage_info;
  start = clock();
  ssize_t rc = read(fd, &lhpage_info, sizeof(lhckpt_pages_t));
  end = clock();
  dReadPageTime += (end - start);
  while (rc == (ssize_t)sizeof(lhckpt_pages_t))
  {
    size_t orig_size = lhpage_info.mem_len;
    size_t comp_size = lhpage_info.comp_len;
    void *orig_addr = lhpage_info.mem_addr;
    printf("orig_size:%ld  comp_size:%ld orig_addr:%p\n", orig_size, comp_size, orig_addr);

    start = clock();
    char *compressed_data = (char *)malloc(comp_size);
    end = clock();
    dMallocTime += (end - start);
    start = clock();
    ssize_t data_size = read(fd, compressed_data, comp_size);
    end = clock();
    dReadFile += (end - start);
    printf("read data:%ld, need data:%ld\n", data_size, comp_size);

    // TODO: decompress the data and copy to original gpu address
    start = clock();
    void *src_ptr = 0;
    cudaError_t ret = cudaMalloc(&src_ptr, comp_size);
    end = clock();
    dCudaMallocTime += (end - start);
    // printf("[LT] cudaMalloc ret:%i, malloc size:%ld\n", ret, comp_size);
    start = clock();
    ret = cudaMemcpy(src_ptr, (void *)compressed_data, comp_size, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
      exit(1);
    end = clock();
    dMemcpyHtoD += (end - start);
    // printf("[LT] cudaMemcpy From Host To Device, ret:%i, size:%ld\n", ret, comp_size);

    void *dst_ptr = 0;
    start = clock();
    ret = cudaMalloc(&dst_ptr, orig_size);
    end = clock();
    dCudaMallocTime += (end - start);
    // printf("[LT] cudaMalloc Dst, ret:%i, size:%ld\n", ret, orig_size);
    fflush(stdout);
    size_t actual_decompress_size = 0;
    start = clock();
    doGpuDecompression(src_ptr, comp_size, (void *)dst_ptr, actual_decompress_size);
    end = clock();
    dCompTime += (end - start);

    if (actual_decompress_size != orig_size)
    {
      printf("mem_addr:%p decompress data is invalid!\n", orig_addr);
    }
    else
    {
      switch (lhpage_info.mem_type)
      {
      case (CUDA_UVM_PAGE):
      case (CUDA_MALLOC_PAGE):
      {
        // copy back the actual data
        cudaMemcpy(orig_addr, (void *)dst_ptr, orig_size, cudaMemcpyDeviceToDevice);
        break;
      }
      case (CUMEM_ALLOC_PAGE):
      {
        // copy back the actual data

        // CUdeviceptr actual_orig_addr = *(CUdeviceptr*)getCUtype((void*)&orig_addr, CU_DEV_PTR);
        start = clock();

        CUresult cu_ret = cuMemcpyDtoD((CUdeviceptr)orig_addr, (CUdeviceptr)dst_ptr, orig_size);
        if (cu_ret != CUDA_SUCCESS)
          exit(1);
        end = clock();
        dMemcpyDtoD += (end - start);
        // printf("[LT] cuMemcpyDtoD in replay, src addr:%p, ret:%i\n", orig_addr, cu_ret);
        break;
      }
      default:
        printf("page type not implemented\n");
        break;
      }
    }
    cudaFree(src_ptr);
    cudaFree(dst_ptr);

    // TODO: read next gpu memory block info
    start = clock();
    rc = read(fd, &lhpage_info, sizeof(lhckpt_pages_t));
    end = clock();
    dReadPageTime += (end - start);
  }
  close(fd);
  // printf("[LT] Finish lower half data handling...readPage time:%.3f, readFile time:%.3f, malloc time:%.3f, memcpyHtoD:%.3f, memcpyDtoD:%.3f, cudaMalloc time:%.3f, decompress time:%.3f\n", dReadPageTime / CLOCKS_PER_SEC * 1000, dReadFile / CLOCKS_PER_SEC * 1000, dMallocTime / CLOCKS_PER_SEC * 1000, dMemcpyHtoD / CLOCKS_PER_SEC * 1000, dMemcpyDtoD / CLOCKS_PER_SEC * 1000, dCudaMallocTime / CLOCKS_PER_SEC * 1000, dCompTime / CLOCKS_PER_SEC * 1000);
}
#endif

// by tian01.liu for new gpu memory manamement
static int
realloc_gpu_blocks()
{
  // TODO: read lhMMUInfo and then initialize mmu module
  GetCudaBlockMapsFptr_t func = (GetCudaBlockMapsFptr_t)uhInfo.cudaBlockMapFptr;
  if (func == NULL)
    return 0;

  // TODO: iterator the maps and call mmu api to realloc gpu blocks
  std::map<void *, lhckpt_pages_t> lhPageMaps = func();
  int nBlockCnt = lhPageMaps.size();
  int nBlkCntReal = 0;

  for (auto lh_page : lhPageMaps)
  {
    CUdeviceptr cuPtr = (CUdeviceptr)(lh_page.second.mem_addr);
    size_t mem_len = lh_page.second.mem_len;

    printf("in realloc_gpu_blocks, cuPtr:%p, mem_size:%ld, devId:%i, uhInfo:%i\n", (void*)cuPtr, mem_len, lh_page.second.devId, getUhPid());
    fflush(stdout);

    // TODO: memory with IPC does not reallocate here, it will be recreate through cudaIpcOpenMemHandle replay 
    if (lh_page.second.isIpcMem)
      continue;

    g_allocator->malloc_restart((void *)cuPtr, mem_len, lh_page.second.devId);

    nBlkCntReal++;
  }

  // Check whether all the gpu blocks realloc successful.
  if (nBlockCnt != nBlkCntReal)
    return 0;
  // end = clock();
  // printf("[lt] realloc gpu blks time:%.3f\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
  return 1;
}

// by tian01.liu for new gpu memory managemement
static int
setupMMUInfo(bool isReplay)
{
  lhMMUInfo.mmu_reserve_big_addr = (void *)g_allocator->big_block_info[0].reserved_start_ptr;
  lhMMUInfo.mmu_reserve_big_size = g_allocator->big_block_info[0].reserved_memory * 4;
  lhMMUInfo.mmu_reserve_small_addr = (void *)g_allocator->small_block_info.reserved_start_ptr;
  lhMMUInfo.mmu_reserve_small_size = g_allocator->small_block_info.reserved_memory;
  lhMMUInfo.mmu_reserve_allign = 0;
  lhMMUInfo.mmu_reserve_flags = 0;

  return 0;
}

/**
 * this function used to create mmu object which can manage all the virual address and gpu memory blocks
 */
int init_mmu_allocator()
{
  if (g_allocator != nullptr)
  {
    delete g_allocator;
  }

  g_allocator = new Allocator(); // init mmu with parameters, used in restore workflow
  // printf("[lt] host alloc start addr:%p\n", g_allocator->host_start_address);
  return g_allocator == nullptr ? 0 : 1;
}

void release_mmu_allocator()
{
  if (g_allocator != nullptr)
    delete g_allocator;

  return;
}

static int
writeLhMMUInfoToFile()
{
  size_t rc = 0;
  char filename[100];
  snprintf(filename, 100, "./lhMMUInfo_%d", getpid());
  int fd = open(filename, O_WRONLY | O_CREAT, 0644);
  if (fd < 0)
  {
    DLOG(ERROR, "Could not create addr.bin file. Error: %s", strerror(errno));
    return -1;
  }

  rc = write(fd, &lhMMUInfo, sizeof(lhMMUInfo));
  if (rc < sizeof(lhMMUInfo))
  {
    DLOG(ERROR, "Wrote fewer bytes than expected to addr.bin. Error: %s",
         strerror(errno));
    rc = -1;
  }
  close(fd);
  return rc;
}

static void
readMMUInfoFromUhFile()
{
  char filename[100];
  // snprintf(filename, 100, "./uhInfo_%d", getpid());
  pid_t orig_pid = getUhPid();
  snprintf(filename, 100, "./uhInfo_%d", orig_pid);
  int fd = open(filename, O_RDONLY);
  if (fd < 0)
  {
    printf("Could not open MMU upper-half file for reading. %s \n", filename);
    exit(-1);
  }
  ssize_t rc = read(fd, &uhInfo, sizeof(uhInfo));
  if (rc != (ssize_t)sizeof(uhInfo))
  {
    perror("Read fewer bytes than expected from uhaddr.bin. MMU");
    exit(-1);
  }
  printf("Finished read the UhInfo\n");

  lhMMUInfo.mmu_reserve_big_addr = uhInfo.mmu_reserve_big_addr;
  lhMMUInfo.mmu_reserve_big_size = uhInfo.mmu_reserve_big_size;
  lhMMUInfo.mmu_reserve_small_addr = uhInfo.mmu_reserve_small_addr;
  lhMMUInfo.mmu_reserve_small_size = uhInfo.mmu_reserve_small_size;
  lhMMUInfo.mmu_reserve_allign = uhInfo.mmu_reserve_allign;
  lhMMUInfo.mmu_reserve_flags = uhInfo.mmu_reserve_flags;
}

static int
get_mmu_profile_from_lh(Lhmmu_profile_t *mmu_profile)
{
  memcpy(mmu_profile, &lhMMUInfo, sizeof(lhMMUInfo));
  return 1;
}

static void *
allocate_gpu_mem(size_t length)
{
  // cudaError_t cudaRet = cudaSuccess;
  if (g_allocator == nullptr)
    return nullptr;

  pthread_mutex_lock(&thread_mutex);
  if (!g_allocator->is_gpu_inited)
  {
    g_allocator->init_gpu_memory_pool(0, 0);
    setupMMUInfo(false);
  }

  // DLOG(ERROR, "gpu mem alloc size:%ld\n", length);
  void* ptr = g_allocator->malloc_gpu(length);
  pthread_mutex_unlock(&thread_mutex);
  return ptr;
}

static void
free_gpu_mem(void *devPtr)
{
  // cudaError_t cudaRet = cudaSuccess;
  if (devPtr == nullptr)
    return;
  pthread_mutex_lock(&thread_mutex);
  g_allocator->free_gpu(devPtr);
  pthread_mutex_unlock(&thread_mutex);
}

void reinit_mmu_space()
{
  if (g_allocator == nullptr)
    g_allocator = new Allocator();

  DLOG(ERROR, "reserverd big addr:%p, small addr:%p in reinit_mmu_space\n", uhInfo.mmu_reserve_big_addr, uhInfo.mmu_reserve_small_addr);
  // printf("reserverd addr:%p in reinit_mmu_space\n", uhInfo.mmu_reserve_addr);
  // Todo
  g_allocator->init_gpu_memory_pool(uhInfo.mmu_reserve_big_addr, uhInfo.mmu_reserve_small_addr);
}

void copy_lower_half_data()
{

  /***************************gpu compress workflow**************************************/
#if USE_COMP
  bool is_gpu_compress;
  is_gpu_compress = uhInfo.is_gpu_compress;
  printf("is_gpu_compress in kernel_loader: %d\n", is_gpu_compress);
  if (is_gpu_compress && inc_obj.use_increamental)
  {
    std::vector<int> gpu_id_vector;
    inc_obj.ReadCkptFile(gpu_id_vector,  getUhPid());
    for (auto id : gpu_id_vector)
    {
      copy_lower_half_data_gpu_compress(id);
      close(id);
    }
  }
  else if (is_gpu_compress)
  {
    copy_lower_half_data_gpu_compress(-1);
  }

  char filename[100];
  snprintf(filename, 100, "./orig_stack_data_%d", getUhPid());
  int fd_1 = open(filename, O_RDWR, 0644);
  readAll(fd_1, g_stack_seg_addr, g_stack_seg_size);
  return;
#endif

  /***************************orignal workflow**************************************/
  // CUcontext ctx;
  // unsigned int ctx_create_flags = 0;
  // CUdevice device_id = 0;
  // cuCtxCreate(&ctx, ctx_create_flags, device_id);

  if (inc_obj.use_increamental)
  {
    inc_obj.ReadCkptFile(getUhPid());
  }
  else
  {
    void *lhpages_addr = uhInfo.lhPagesRegion;
    // read total entries count
    int total_entries = 0;
    size_t count = 0;

    memcpy(&total_entries, ((VA)lhpages_addr + count), sizeof(total_entries));
    count += sizeof(total_entries);

    for (int i = 0; i < total_entries; i++)
    {
      // read metadata of one entry
      lhckpt_pages_t lhpage_info;
      memcpy(&lhpage_info, ((VA)lhpages_addr + count), sizeof(lhpage_info));
      count += sizeof(lhpage_info);

      void *dest_addr = lhpage_info.mem_addr;
      size_t size = lhpage_info.mem_len;

      switch (lhpage_info.mem_type)
      {
      case (CUDA_UVM_PAGE):
      case (CUDA_MALLOC_PAGE):
      {
        // copy back the actual data
        cudaError_t ret_val = cudaSuccess;
	      cudaSetDevice(lhpage_info.devId);
        ret_val = cudaMemcpy(dest_addr, ((VA)lhpages_addr + count), size, cudaMemcpyHostToDevice);
        // printf("copy_lower_half_data, dstAddr:%p, ret:%i\n", dest_addr, cudaRet);
        assert(ret_val == cudaSuccess);
        count += size;
        break;
      }
      case (CUMEM_ALLOC_PAGE):
      {
        // printf("copy_lower_half_data_cu, dstAddr:%p, srcAddr:%p, ret:%i\n", dest_addr, ((VA)lhpages_addr+count), cuRet);
        CUresult ret = cuMemcpyHtoD((CUdeviceptr)dest_addr, ((VA)lhpages_addr + count), size);
        // printf("copy_lower_half_data_cu, dstAddr:%p, srcAddr:%p, ret:%i\n", dest_addr, ((VA)lhpages_addr+count), ret);
        printf("ret %i\n", ret); fflush(stdout);
        assert(ret == CUDA_SUCCESS);
        // printf("copy_lower_half_data_cu, dstAddr:%p, srcAddr:%p, ret:%i\n", dest_addr, ((VA)lhpages_addr+count), cuRet);
        count += size;
        break;
      }
        /*      case CUDA_HEAP_PAGE:
              {
                void *newDeviceHeapStart = (void *)ROUND_DOWN(getDeviceHeapPtr());
                void *__cudaPtr = NULL;
                void *oldDeviceHeapStart = dest_addr;
                if (oldDeviceHeapStart != newDeviceHeapStart) {
                  DLOG(ERROR, "New Device heap = %p is not same as Old device heap =%p\n",
                  newDeviceHeapStart, oldDeviceHeapStart);
                }
                cudaMalloc(&__cudaPtr, size);
                cudaMemcpy(__cudaPtr, ((VA)lhpages_addr+count), size, cudaMemcpyHostToDevice);
                copyFromCudaPtr(__cudaPtr, newDeviceHeapStart, size);
                char buf[8192];
                copyToCudaPtr(__cudaPtr, newDeviceHeapStart, size);
                cudaMemcpy(buf, __cudaPtr, size, cudaMemcpyDeviceToHost);
                cudaFree(__cudaPtr);
                cudaDeviceSynchronize();
                count += size;
                break;
              } */
      default:
        printf("page type not implemented\n");
        break;
      }
      // cuCtxDestroy(ctx);
    }
    // cuCtxDestroy(ctx);
    // this section mainly solve the problem of stach data changed
    char filename[100];
    snprintf(filename, 100, "./orig_stack_data_%d", getUhPid());
    int fd = open(filename, O_RDWR, 0644);
    /*size_t size = */ readAll(fd, g_stack_seg_addr, g_stack_seg_size);
    close(fd);
  }
}
