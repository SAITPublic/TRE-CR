#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>  // test code, by tian01.liu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <map>  // add by tian01.liu 2023.1.28
#include <unordered_map>
#include "common.h"
#include "switch_context.h"
#include "upper-half-wrappers.h"
#include "upper-half-cuda-wrappers.h"
#include "cuda_autogen_wrappers.h"
#include "log_and_replay.h"
#include <iostream>

#define NCCL_MIGRATION
// by tian01.liu 2023.10.30
#include <sys/shm.h>
#include <sys/ipc.h>
// shm size
#define BUFSZ 512

std::map<cudaStream_t, cudaStream_t*> stream_map;
typedef void (*readUhInfoAddr_t)();
typedef void (*logs_read_and_apply_t)();
typedef void (*copy_lower_half_data_t)();

clock_t start_finish_restore;
typedef int (*realloc_gpu_blocks_t)();

// class MapFile
// {
//   public:
//     static MapFile& GetInstance()
//     {
//       static MapFile mapFile;
//       return mapFile;
//     }
//     void DelItem(void* addr)
//     { 
//       if(page_lock_maps.count(addr) == 0){
//         return;
//       }

//       std::map<void*, st_page_lock_info>::iterator it = page_lock_maps.find(addr);
//       if(it != page_lock_maps.end()){
//         page_lock_maps.erase(it);
//       }
//     }
//     void AddItem(void* addr, st_page_lock_info size)
//     {
//       pthread_mutex_lock(&mutex_tmp);
//       page_lock_maps.insert(std::pair<void*, st_page_lock_info>(addr, size));
//       pthread_mutex_unlock(&mutex_tmp);
//     }

//     void* findItem(void* addr)
//     {
//       auto it = page_lock_maps.lower_bound(addr);
//       if(it == page_lock_maps.end())
//           return NULL;
//       return it->first;
//     }
//   private:
//     MapFile(){pthread_mutex_init(&mutex_tmp, NULL);};
//     std::map<void*, st_page_lock_info> page_lock_maps;
//     pthread_mutex_t mutex_tmp;
// };

// added by tian01.liu 2023.1.28. fix the bug of cudaMallocHost/cuMemAllocHost/cuMemHostAlloc
extern std::map<void*, st_page_lock_info> g_addrs_for_host;
extern bool g_pin_record;
extern pthread_mutex_t mutex_for_map;
extern pthread_mutex_t mutex_for_map_fs;
// add by tian01.liu for lower-half fs of each user thread
std::unordered_map<pid_t, unsigned long> fsThreadsMap;

typedef void* (*lhCUtype_t)(void *cuType, int type);
typedef void (*lhFreeCUtype_t)(void *cuType, int type);
typedef void* (*lhCublasType_t)(void *cublasType, int type);
typedef void* (*lhFreeCublasType_t)(void *cublasType, int type);
typedef void* (*lhCUDAtype_t)(void *cuType, int type);
typedef void (*lhFreeCUDAtype_t)(void *cuType, int type);

// add by tian01.liu 2023.9.5 
typedef void* (*lhNickleType_t)(void* ncclType, int type); 
typedef void (lhFreeNickleType_t)(void* ncclType, int type);

__thread__ pid_t tid = -1;
__thread__ pid_t pid = -1;

__thread__ unsigned long fsOfNewThread = 0;
__thread__ bool is_first = false; 
std::unordered_map<pid_t, unsigned long> uhFsMap;

bool g_b_suspend = false;

// #undef JUMP_TO_LOWER_HALF
#define JUMP_TO_LOWER_HALF_EX(lhFs) \
  if(uh, &uhFsFs == 0) {/*printf("before syscall\n");*/syscall(SYS_arch_prctl, ARCH_GET_FS, &uhFs);uhFsPtr = &uhFs; /*printf("after syscall, uhFs:%ld, lhFs:%ld\n", uhFs, lhFs);*/} \
  syscall(SYS_arch_prctl, ARCH_SET_FS, lhFs)

// #undef RETURN_TO_UPPER_HALF
#define RETURN_TO_UPPER_HALF_EX() \
  /*printf("upper-half fs:%ld\n", *uhFsPtr);*/ \
  syscall(SYS_arch_prctl, ARCH_SET_FS, *uhFsPtr)  

void **global_fatCubinHandle=NULL;

__thread int nccl_wrapper_count = 0;
void to_avoidwarning(int x) {;}
int dmtcp_plugin_disable_ckptthread(void);
int dmtcp_plugin_disable_ckpt(void);
#undef DMTCP_PLUGIN_DISABLE_CKPT
#define DMTCP_PLUGIN_DISABLE_CKPT() \
  int __dmtcp_plugin_ckpt_disabled = 0; \
  to_avoidwarning(__dmtcp_plugin_ckpt_disabled); \
  if(nccl_wrapper_count == 0) __dmtcp_plugin_ckpt_disabled = dmtcp_plugin_disable_ckpt()

void dmtcp_plugin_enable_ckptthread(void);
void dmtcp_plugin_enable_ckpt(void);
#undef DMTCP_PLUGIN_ENABLE_CKPT
#define DMTCP_PLUGIN_ENABLE_CKPT() \
  if(__dmtcp_plugin_ckpt_disabled) {dmtcp_plugin_enable_ckpt();/*JNOTE("Release wrapper exection rd lock....");*/}

void   reinitPageLockPool()
{
  LhInitPageLockPool_t releaseFunc = (LhInitPageLockPool_t)(lhInfo.lhInitPageLockPoolFptr);
  if(releaseFunc)
      releaseFunc();
}


clock_t replay_start, replay_end;
void replay_gpu_status()
{
  if(lhInfo.isReplay)
  {
      replay_start = clock();

      //fflush(stdout);
      lhInfo.isReplay = false;
      readUhInfoAddr_t fnc = (readUhInfoAddr_t)lhInfo.readUhInfoAddrHandle;

      LhReinitMMU_t fucReInit = (LhReinitMMU_t)lhInfo.lhReInitMMUFptr;
      realloc_gpu_blocks_t func3 = (realloc_gpu_blocks_t)lhInfo.realloc_gpu_blocks_handle;
      if(func3 == NULL){
        printf("realloc gpu blocks handle get failure\n");
        return;
      }
     copy_lower_half_data_t func2 = (copy_lower_half_data_t)lhInfo.copy_lower_half_data_handle;
     if(func2 == NULL){
        printf("null function is called\n");
	      return;
     }
     JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
     fnc();
     fucReInit();
     fprintf(stderr, "reinit mmu in replay....\n");
     fflush(stderr);
     func3();
     fprintf(stderr, "realloc gpu block in replay....\n");
      func2();
     RETURN_TO_UPPER_HALF();
      printf(" End replay gpu status, sleep\n");
      //fflush(stdout);
  }
}


void check_cu_map(void* newObject, void* cuType, int type)
{
  void* pObj = getCUtype(cuType, type);
  switch(type)
  {
    case CU_DEV_PTR:
    {
      if(NULL == pObj) {
        *(CUdeviceptr*)newObject = *((CUdeviceptr*)cuType);
      }
      else {
        *(CUdeviceptr*)newObject = *((CUdeviceptr*)pObj);
      }
      break;
    }
    case CU_CONTEXT:
    {
      if(NULL == pObj) {
        *(CUcontext*)newObject = *((CUcontext*)cuType);
      }
      else{
        *(CUcontext*)newObject = *((CUcontext*)pObj);
      }
      break;
    }
    case CU_MODULE:
    {
      if(NULL == pObj) {
        *(CUmodule*)newObject = *((CUmodule*)cuType);
      }
      else{
        *(CUmodule*)newObject = *((CUmodule*)pObj);
      }
      break;
    }
    case CU_FUNCTION:
    {
      if(NULL == pObj) {
        *(CUfunction*)newObject = *((CUfunction*)cuType);
      }
      else{
        *(CUfunction*)newObject = *((CUfunction*)pObj);
      }
      break;
    }
    case CU_EVENT:
    {
      if(NULL == pObj) {
        *(CUevent*)newObject = *((CUevent*)cuType);
      }
      else{
        *(CUevent*)newObject = *((CUevent*)pObj);
      }
      break;
    }
    case CU_STREAM:
    {
      if(NULL == pObj) {
        *(CUstream*)newObject = *((CUstream*)cuType);
      }
      else{
        *(CUstream*)newObject = *((CUstream*)pObj);
      }
      break;
    }
    case CU_TEXREF:
    {
      if(NULL == pObj) {
        *(CUtexref*)newObject = *((CUtexref*)cuType);
      }
      else{
        *(CUtexref*)newObject = *((CUtexref*)pObj);
      }
      break;
    }
    default:
      assert(false);
    }
}


void check_cu_map(void* newObject, void* cuType, int type, int& freeFlag)
{
 void* pObj = getCUtype(cuType, type);
  switch(type)
  {
    case CU_DEV_PTR:
    {
      if(NULL == pObj) {
        *(CUdeviceptr*)newObject = *((CUdeviceptr*)cuType);
      }
      else {
        *(CUdeviceptr*)newObject = *((CUdeviceptr*)pObj);
        freeFlag = 1;
      }
      break;
    }
    case CU_CONTEXT:
    {
      if(NULL == pObj) {
        *(CUcontext*)newObject = *((CUcontext*)cuType);
      }
      else{
        *(CUcontext*)newObject = *((CUcontext*)pObj);
        freeFlag = 1;
      }
      break;
    }
    case CU_MODULE:
    {
      if(NULL == pObj) {
        *(CUmodule*)newObject = *((CUmodule*)cuType);
      }
      else{
        *(CUmodule*)newObject = *((CUmodule*)pObj);
        freeFlag = 1;
      }
      break;
    }
    case CU_FUNCTION:
    {
      if(NULL == pObj) {
        *(CUfunction*)newObject = *((CUfunction*)cuType);
      }
      else{
        *(CUfunction*)newObject = *((CUfunction*)pObj);
        freeFlag = 1;
      }
      break;
    }
    case CU_EVENT:
    {
      if(NULL == pObj) {
        *(CUevent*)newObject = *((CUevent*)cuType);
      }
      else{
        *(CUevent*)newObject = *((CUevent*)pObj);
        freeFlag = 1;
      }
      break;
    }
    case CU_STREAM:
    {
      if(NULL == pObj) {
        *(CUstream*)newObject = *((CUstream*)cuType);
      }
      else{
        *(CUstream*)newObject = *((CUstream*)pObj);
        freeFlag = 1;
      }
      break;
    }
    case CU_TEXREF:
    {
      if(NULL == pObj) {
        *(CUtexref*)newObject = *((CUtexref*)cuType);
      }
      else{
        *(CUtexref*)newObject = *((CUtexref*)pObj);
        freeFlag = 1;
      }
      break;
    }
    default:
      assert(false);
    }
}




void check_cu_map(void* newObject, void* cuType, int type, lhCUtype_t getCUtype)
{
  void* pObj = getCUtype(cuType, type);
  switch(type)
  {
    case CU_DEV_PTR:
    {
      if(NULL == pObj) {
        *(CUdeviceptr*)newObject = *((CUdeviceptr*)cuType);
      }
      else {
        *(CUdeviceptr*)newObject = *((CUdeviceptr*)pObj);
      }
      break;
    }
    case CU_CONTEXT:
    {
      if(NULL == pObj) {
        *(CUcontext*)newObject = *((CUcontext*)cuType);
      }
      else{
        *(CUcontext*)newObject = *((CUcontext*)pObj);
      }
      break;
    }
    case CU_MODULE:
    {
      if(NULL == pObj) {
        *(CUmodule*)newObject = *((CUmodule*)cuType);
      }
      else{
        *(CUmodule*)newObject = *((CUmodule*)pObj);
      }
      break;
    }
    case CU_FUNCTION:
    {
      if(NULL == pObj) {
        *(CUfunction*)newObject = *((CUfunction*)cuType);
      }
      else{
        *(CUfunction*)newObject = *((CUfunction*)pObj);
      }
      break;
    }
    case CU_EVENT:
    {
      if(NULL == pObj) {
        *(CUevent*)newObject = *((CUevent*)cuType);
      }
      else{
        *(CUevent*)newObject = *((CUevent*)pObj);
      }
      break;
    }
    case CU_STREAM:
    {
      if(NULL == pObj) {
        *(CUstream*)newObject = *((CUstream*)cuType);
      }
      else{
        *(CUstream*)newObject = *((CUstream*)pObj);
      }
      break;
    }
    case CU_TEXREF:
    {
      if(NULL == pObj) {
        *(CUtexref*)newObject = *((CUtexref*)cuType);
      }
      else{
        *(CUtexref*)newObject = *((CUtexref*)pObj);
      }
      break;
    }
    default:
      assert(false);
    }
}
// extern void *getCudaType(void* cudaType, int type);
// extern void *getCublasType(void *cublasType, int type);
// extern void freeCublasType(void *cublasType, int type);

void check_cuda_map(void* newObject, void* cudaType, int type, lhCUDAtype_t getCUDAtype)
{
  void* pObj = getCUDAtype(cudaType, type);
  switch(type)
  {
    case CUDA_STREAM:
    {
      if (NULL == pObj)
      {
        // printf("Get cudaStream_t old object.\n");
        *(cudaStream_t*)newObject = *((cudaStream_t*)cudaType);
      }
      else
      {
        *(cudaStream_t*)newObject = *((cudaStream_t*)pObj);
      }
      break;
    }
    case CUDA_EVENT:
    {
      if (NULL == pObj)
      {
        *(cudaEvent_t*)newObject = *((cudaEvent_t*)cudaType);
      }
      else
      {
        *(cudaEvent_t*)newObject = *((cudaEvent_t*)pObj);
      }
      break;
    }
    default:
      assert(false);
  }
}

// add by tian01.liu 2023.9.5
void check_nccl_map(void* newObject, void* ncclType, int type, lhNickleType_t getNickletype)
{
  void* pObj = getNickletype(ncclType, type);
  switch(type)
  {
    case NCCL_COMM:
    {
      if (NULL == pObj)
      {
        // printf("Get cudaStream_t old object.\n");
        *(ncclComm_t*)newObject = (*(ncclComm_t*)ncclType);
      }
      else
      {
        *(ncclComm_t*)newObject = (ncclComm_t)pObj;
      }
      break;
    }
    default:
      assert(false);
  }
}

unsigned long getFs(pid_t tid, pid_t pid)
{
  // printf("pid %i , tid %i\n", pid, tid);fflush(stdout);
  if(pid == tid){
    //fsThreadsMap[tid] = lhInfo.lhFsAddr;
    return lhInfo.lhFsAddr;
  }

  unsigned long fsOfNewThread;
  if(!fsThreadsMap.count(tid))
  {
    // printf("!fsThreadsMap.count(tid) %i\n", tid);fflush(stdout);
    // TODO: �������̲߳����ظ�Upper-Half,���¶�Ӧ��fs map����������̵߳Ļ���Ҫͬ��������һ����overhead
    LhNewThread_t newThreadFunc = (LhNewThread_t)(lhInfo.lhNewThreadFptr);
    LhGetFs_t getFsFunc = (LhGetFs_t)(lhInfo.lhGetFsFptr);
    int shmKey = tid + dmtcp_virtual_to_real_pid(pid);
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
    newThreadFunc(tid, shmKey);
    getFsFunc(tid, &fsOfNewThread);
    RETURN_TO_UPPER_HALF();

    pthread_mutex_lock(&mutex_for_map_fs);
    fsThreadsMap[tid] = fsOfNewThread;
    pthread_mutex_unlock(&mutex_for_map_fs);  
  }
  else{
    // printf("fsThreadsMap.count(tid) %i \n", tid);fflush(stdout);
    fsOfNewThread = fsThreadsMap[tid];
    // printf("%ld\n", fsOfNewThread);
  }
    
  
  return fsOfNewThread;
}
// char * getShmAddr(pid_t tid)
// {
//   char * shmAddr = NULL;

//   if(shmThreadsMap.count(tid))
//     shmAddr = shmThreadsMap[tid];
//   else
//   {
//     // TODO: ����uh�̶߳�Ӧ�Ĺ����ڴ�ӳ��
//     int shmid;
//     int ret;
    
//     //���������ڴ�
//     shmid = shmget((key_t)tid, BUFSZ, IPC_CREAT|0666);    
//     if(shmid < 0)
//     {
//       printf("shmget failure...\n");
//       exit(-1);
//     }

//     //ӳ��
//     shmAddr = (char*)shmat(shmid, NULL, 0);
//     if(shmAddr < 0)
//     {
//         perror("shmat");
//         _exit(-1);
//     }

//     pthread_mutex_lock(&mutex_for_map);
//     shmThreadsMap[tid] = shmAddr;
//     pthread_mutex_unlock(&mutex_for_map);
//   }

//   printf("upper-half tid:%i, shmaddr:%p\n", tid, shmAddr);
//   return shmAddr;
// }

clock_t start_jump, end_jump;
double dTotal = 0;

/* ------------ Cuda APIs ------------- */
#undef cudaCreateTextureObject
extern "C" cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc) {
  typedef cudaError_t (*cudaCreateTextureObject_t)(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaCreateTextureObject)(pTexObject, pResDesc, pTexDesc, pResViewDesc);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaCreateTextureObject, pTexObject, pResDesc, pTexDesc, pResViewDesc, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaDestroyTextureObject
extern "C" cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
  typedef cudaError_t (*cudaDestroyTextureObject_t)(cudaTextureObject_t texObject);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaDestroyTextureObject)(texObject);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaDestroyTextureObject, texObject, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cudaBindTexture
extern "C" cudaError_t cudaBindTexture ( size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size) {
  typedef cudaError_t (*cudaBindTexture_t)( size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaBindTexture)(offset, texref, devPtr, desc, size);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaBindTexture2D
extern "C" cudaError_t cudaBindTexture2D ( size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t width, size_t height, size_t pitch ) {
  typedef cudaError_t (*cudaBindTexture2D_t)( size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t width, size_t height, size_t pitch );
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaBindTexture2D)(offset, texref, devPtr, desc, width, height, pitch);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  return ret_val;
}

/* @LogReplay */

#undef cudaBindTextureToArray
extern "C" cudaError_t cudaBindTextureToArray ( const textureReference* texref, cudaArray_const_t array, const cudaChannelFormatDesc* desc ) {
  typedef cudaError_t (*cudaBindTextureToArray_t)( const textureReference* texref, cudaArray_const_t array, const cudaChannelFormatDesc* desc );
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaBindTextureToArray)(texref, array, desc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* @LogReplay */

#undef cudaUnbindTexture
extern "C" cudaError_t cudaUnbindTexture(const struct textureReference * texref) {
  typedef cudaError_t (*cudaUnbindTexture_t)(const struct textureReference * texref);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaUnbindTexture)(texref);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaCreateChannelDesc
extern "C" cudaChannelFormatDesc cudaCreateChannelDesc ( int  x, int  y, int  z, int  w, cudaChannelFormatKind f ) {
  typedef cudaChannelFormatDesc (*cudaCreateChannelDesc_t)( int  x, int  y, int  z, int  w, cudaChannelFormatKind f );
  cudaChannelFormatDesc ret_val ;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaCreateChannelDesc)(x, y, z, w, f);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaCreateChannelDesc, x, y, z, w, f, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}

/* Cuda event related APIs */

#undef cudaEventCreate
extern "C" cudaError_t cudaEventCreate(cudaEvent_t * event) {
  typedef cudaError_t (*cudaEventCreate_t)(cudaEvent_t * event);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
 
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val =  REAL_FNC(cudaEventCreate)(event);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cudaEventCreate, event, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  // printf("cudaEventCreate new:%p, ret:%i, tid:%i\n", event, ret_val, tid);
  if(ret_val != cudaSuccess)
    printf("Called at func '%s', ret:%i\n", __FUNCTION__, ret_val);
  return ret_val;
}


#undef cudaEventCreateWithFlags
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags) {
  typedef cudaError_t (*cudaEventCreateWithFlags_t)(cudaEvent_t * event, unsigned int flags);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaEventCreateWithFlags)(event, flags);
  RETURN_TO_UPPER_HALF();
  // printf("cudaEventCreateWithFlags, ret:%i, event:%p, flags:%i\n", ret_val, *event, flags);
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaEventCreateWithFlags, event, flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  // printf("cudaEventCreateWithFlags new:%p, ret:%i, tid:%i\n", event, ret_val, tid);
  if(ret_val != cudaSuccess)
    printf("Called at func '%s', ret:%i\n", __FUNCTION__, ret_val);
  return ret_val;
}


#undef cudaEventDestroy
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
  typedef cudaError_t (*cudaEventDestroy_t)(cudaEvent_t event);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  DMTCP_PLUGIN_DISABLE_CKPT();

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cudaEvent_t newEvent;
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  LhFreeCudaType_t freeCudaType = (LhFreeCudaType_t)lhInfo.lhFreeCUDATypeFptr;
  // printf("cudaEventDestroy original:%p, tid:%i\n", event, tid);
  // fflush(stdout);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cuda_map((void*)&newEvent, (void*)&event, CUDA_EVENT, getCUDAtype);
  ret_val = REAL_FNC(cudaEventDestroy)(newEvent);

  //Erase the record from map, the key is ctx
  if (newEvent !=event)
    freeCudaType((void*)&event, CUDA_EVENT);

  RETURN_TO_UPPER_HALF();
  // printf("cudaEventDestroy new:%p, ret:%i, tid:%i\n", newEvent, ret_val, tid);
  // fflush(stdout);
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaEventDestroy, newEvent, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  if(ret_val != cudaSuccess)
    printf("Called at func '%s', ret:%i ,pid:%i, tid:%i.\n", __FUNCTION__, ret_val, pid, tid);
  return ret_val;
}


#undef cudaEventElapsedTime
extern "C" cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end) {
  typedef cudaError_t (*cudaEventElapsedTime_t)(float * ms, cudaEvent_t start, cudaEvent_t end);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  DMTCP_PLUGIN_DISABLE_CKPT();

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cudaEvent_t newStart;
  cudaEvent_t newEnd;

  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStart, (void*)&start, CUDA_EVENT, getCUDAtype);
  check_cuda_map((void*)&newEnd, (void*)&end, CUDA_EVENT, getCUDAtype);
    ret_val = REAL_FNC(cudaEventElapsedTime)(ms, newStart, newEnd);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  if(ret_val != cudaSuccess)
    printf("Called at func '%s', ret:%i\n", __FUNCTION__, ret_val);
  return ret_val;
}


#undef cudaEventQuery
extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
  typedef cudaError_t (*cudaEventQuery_t)(cudaEvent_t event);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  DMTCP_PLUGIN_DISABLE_CKPT();

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cudaEvent_t newEvent;


  // start_jump = clock();
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newEvent, (void*)&event, CUDA_EVENT, getCUDAtype);  
  ret_val = REAL_FNC(cudaEventQuery)(newEvent);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  // if(ret_val != cudaSuccess)
  //   printf("Called at func '%s', ret:%i ,pid:%i, tid:%i.\n", __FUNCTION__, ret_val, pid, tid);
  return ret_val;
}


#undef cudaEventRecord
extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  typedef cudaError_t (*cudaEventRecord_t)(cudaEvent_t event, cudaStream_t stream);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  int ret_val_before = cudaGetLastError();

  // printf("before Called at func '%s', ret:%d ,pid:%i, tid:%i.\n", __FUNCTION__, ret_val_before, pid, tid);

  DMTCP_PLUGIN_DISABLE_CKPT();

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cudaEvent_t newEvent;
  cudaStream_t newStream;

  // start_jump = clock();
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  // end_jump = clock();
  // dTotal += (end_jump - start_jump);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newEvent, (void*)&event, CUDA_EVENT, getCUDAtype);
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cudaEventRecord)(newEvent, newStream);
  
  // start_jump = clock();
  RETURN_TO_UPPER_HALF();
  // printf("cudaEventRecord, ret:%i, newEvent:%p, oldEvent:%p, newStream:%p, oldStream:%p\n", ret_val, newEvent, event, newStream, stream);
  // end_jump = clock();
  // dTotal += (end_jump - start_jump);
  DMTCP_PLUGIN_ENABLE_CKPT();
  if(ret_val != cudaSuccess)
    // printf("Called at func '%s', ret:%i ,pid:%i, tid:%i.\n", __FUNCTION__, ret_val, pid, tid);
  logAPI(Cuda_Fnc_cudaEventRecord, newEvent, newStream, ret_val);
  return ret_val;
}


#undef cudaEventSynchronize
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  typedef cudaError_t (*cudaEventSynchronize_t)(cudaEvent_t event);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cudaEvent_t newEvent;


  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newEvent, (void*)&event, CUDA_EVENT, getCUDAtype);
    ret_val = REAL_FNC(cudaEventSynchronize)(newEvent);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  if(ret_val != cudaSuccess)
  {
    // printf("Called at func '%s', ret:%i\n", __FUNCTION__, ret_val);
  }
  return ret_val;
}

/* ----- Malloc and Free related APIs ----- */

#undef cudaMalloc
extern "C" cudaError_t cudaMalloc(void ** pointer, size_t size) {
  // typedef cudaError_t (*cudaMalloc_t)(void ** pointer, size_t size);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, threadid:%i.\n", __FUNCTION__, __LINE__ , tid);
  LhMMUAllocate_t allocFptr = (LhMMUAllocate_t)lhInfo.lhMMUAllocFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  //printf("before jump to lowerhalf %ld.\n", fsOfNewThread);
  JUMP_TO_LOWER_HALF(fsOfNewThread); //replay_gpu_status();
  *pointer = allocFptr(size);
  // ret_val = REAL_FNC(cudaMalloc)(pointer, size);
  RETURN_TO_UPPER_HALF();
  //printf("Called at func '%s' in line %i, tid:%i, pointer:%p, size:%ld.\n", __FUNCTION__, __LINE__ , tid, *pointer, size);
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaMalloc, pointer, size, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cudaFree
extern "C" cudaError_t cudaFree ( void * pointer ) {
  cudaError_t ret_val = cudaSuccess;
  // typedef cudaError_t (*cudaFree_t)(void* pointer);
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  // LhExecApi_t fuc = (LhExecApi_t)(lhInfo.lhExecApiFptr);
  LhMMUFree_t freeFnc = (LhMMUFree_t)(lhInfo.lhMMUFreeFptr);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  freeFnc(pointer);
  // ret_val  = REAL_FNC(cudaFree)(pointer);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaFree, pointer, ret_val); // by tian01.liu, modify from newPointer to pointer
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* @LogReplay */

#undef cudaMallocArray
extern "C" cudaError_t cudaMallocArray(struct cudaArray ** array, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags) {
  typedef cudaError_t (*cudaMallocArray_t)(struct cudaArray ** array, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaMallocArray)(array, desc, width, height, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* @LogReplay */

#undef cudaFreeArray
extern "C" cudaError_t cudaFreeArray(struct cudaArray * array) {
  typedef cudaError_t (*cudaFreeArray_t)(struct cudaArray * array);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  // ret_val = REAL_FNC(cudaFreeArray)(array);
    ret_val = REAL_FNC(cudaFreeArray)(array);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* Host related APIs */

#undef cudaHostRegister
extern "C" cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int  flags ) {
  typedef cudaError_t (*cudaHostRegister_t)( void* ptr, size_t size, unsigned int  flags );
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);

  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaHostRegister)(ptr, size, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  return ret_val;
}


#undef cudaDeviceGetAttribute
extern "C" cudaError_t cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device ) {
  typedef cudaError_t (*cudaDeviceGetAttribute_t)( int* value, cudaDeviceAttr attr, int  device );
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i.ret:%p\n", __FUNCTION__, __LINE__, &ret_val );
  //fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaDeviceGetAttribute)(value, attr, device);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


/* @LogReplay */

#undef cudaMallocHost
extern "C" cudaError_t cudaMallocHost ( void ** ptr , size_t size ) {
  // typedef cudaError_t (*cudaMallocHost_t)(void ** ptr , size_t size);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i....pid:%i, tid:%i\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  LhAllocHost_t hostAllocFnc = (LhAllocHost_t)lhInfo.lhAllocHostFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  hostAllocFnc(ptr, size, 0x01);
  // ret_val = REAL_FNC(cudaMallocHost)(ptr, size);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  if(g_pin_record)
    g_addrs_for_host[*ptr] = {0x01, size};
  //printf("[lt] cudaMallocHost..1, ret:%i, addr:%p, pid:%i, tid:%i\n", ret_val, *ptr, pid, tid);
  return ret_val;
}

/* @LogReplay */

#undef cudaFreeHost
extern "C" cudaError_t cudaFreeHost ( void* ptr ) {
  // typedef cudaError_t (*cudaFreeHost_t)(void *ptr );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);

  LhFreeHost_t freeFnc = (LhFreeHost_t)lhInfo.lhFreeHostFptr;
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  freeFnc(ptr);
  // ret_val = REAL_FNC(cudaFreeHost)(ptr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  g_addrs_for_host.erase(ptr);
  return ret_val;
}

/* @LogReplay */

#undef cudaHostAlloc
extern "C" cudaError_t cudaHostAlloc ( void ** ptr , size_t size , unsigned int flags ) {
  // typedef cudaError_t (*cudaHostAlloc_t)(void ** ptr , size_t size , unsigned int flags);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  LhAllocHost_t hostAllocFnc = (LhAllocHost_t)lhInfo.lhAllocHostFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  hostAllocFnc(ptr, size, flags);
  // ret_val = REAL_FNC(cudaHostAlloc)(ptr, size, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  if(g_pin_record)
    g_addrs_for_host[*ptr] = {flags, size};
  return ret_val;
}

/* @LogReplay */

#undef cudaMallocPitch
extern "C" cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height) {
  typedef cudaError_t (*cudaMallocPitch_t)(void ** devPtr, size_t * pitch, size_t width, size_t height);
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMallocPitch)(devPtr, pitch, width, height);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

__thread__ int device_ = -1;
/* Device related APIs */
#undef cudaGetDevice
extern "C" cudaError_t cudaGetDevice(int * device) {
  typedef cudaError_t (*cudaGetDevice_t)(int * device);

  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}

  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%d, tid:%d.\n", __FUNCTION__, __LINE__, pid, tid );

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid); 

  // start_jump = clock();
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  // end_jump = clock();
  // dTotal += (end_jump - start_jump);

  ret_val = REAL_FNC(cudaGetDevice)(device);
  
  // start_jump = clock();
  RETURN_TO_UPPER_HALF();
  // end_jump = clock();
  // dTotal += (end_jump - start_jump);

  /* Insert logging code here */
  // logAPI(Cuda_Fnc_cudaGetDevice, device, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();

  return ret_val;
}

__thread bool is_real_api_exe = false;

#undef cudaSetDevice
extern "C" cudaError_t cudaSetDevice(int device) {
  typedef cudaError_t (*cudaSetDevice_t)(int device);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%d, tid:%d device: %d.\n", __FUNCTION__, __LINE__, pid, tid, device);

  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid); 
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaSetDevice)(device);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaSetDevice, device, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();

  return ret_val;
}


#undef cudaDeviceGetLimit
extern "C" cudaError_t cudaDeviceGetLimit ( size_t* pValue, cudaLimit limit ) {
  typedef cudaError_t (*cudaDeviceGetLimit_t)( size_t* pValue, cudaLimit limit );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%d, tid:%d.\n", __FUNCTION__, __LINE__, pid, tid );
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaDeviceGetLimit)(pValue, limit);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaDeviceSetLimit
extern "C" cudaError_t cudaDeviceSetLimit ( cudaLimit limit, size_t value ) {
  typedef cudaError_t (*cudaDeviceSetLimit_t)( cudaLimit limit, size_t value );
  cudaError_t ret_val = cudaSuccess;
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaDeviceSetLimit)(limit, value);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cudaDeviceSetLimit, (int)limit, value, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaGetDeviceCount
extern "C" cudaError_t cudaGetDeviceCount(int * count) {
  typedef cudaError_t (*cudaGetDeviceCount_t)(int * count);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaGetDeviceCount)(count);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaDeviceSetCacheConfig
extern "C" cudaError_t cudaDeviceSetCacheConfig ( cudaFuncCache cacheConfig ) {
  typedef cudaError_t (*cudaDeviceSetCacheConfig_t)( cudaFuncCache cacheConfig );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cudaDeviceSetCacheConfig)(cacheConfig);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cudaDeviceSetCacheConfig, cacheConfig, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaGetDeviceProperties
extern "C" cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device ) {
  typedef cudaError_t (*cudaGetDeviceProperties_t)( cudaDeviceProp* prop, int  device );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val= REAL_FNC(cudaGetDeviceProperties)(prop, device);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaDeviceCanAccessPeer
extern "C" cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice) {
  typedef cudaError_t (*cudaDeviceCanAccessPeer_t)(int * canAccessPeer, int device, int peerDevice);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaDeviceCanAccessPeer)(canAccessPeer, device, peerDevice);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaDeviceGetPCIBusId
extern "C" cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device) {
  typedef cudaError_t (*cudaDeviceGetPCIBusId_t)(char * pciBusId, int len, int device);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaDeviceGetPCIBusId)(pciBusId, len, device);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* @LogReplay */

#undef cudaDeviceReset
extern "C" cudaError_t cudaDeviceReset() {
  typedef cudaError_t (*cudaDeviceReset_t)();
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaDeviceReset)();
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaDeviceSynchronize
extern "C" cudaError_t cudaDeviceSynchronize() {
  typedef cudaError_t (*cudaDeviceSynchronize_t)();
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); //replay_gpu_status();
  ret_val = REAL_FNC(cudaDeviceSynchronize)();
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* Launches a device function Note: This function is deprecated as of CUDA 7.0 */
/* cudaError_t cudaLaunch(const void * func)*/

#undef cudaLaunchKernel
extern "C" cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {
  typedef cudaError_t (*cudaLaunchKernel_t)( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  cudaStream_t newStream;


  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cudaLaunchKernel)(func, gridDim, blockDim, args, sharedMem, newStream);
  
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)*/
/* Cuda UVM APIs */

#undef cudaMallocManaged
extern "C" cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags) {
  typedef cudaError_t (*cudaMallocManaged_t)( void** devPtr, size_t size, unsigned int  flags);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMallocManaged)(devPtr, size, flags);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaMallocManaged, devPtr, size, flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* Cuda Memory management realated APIs */

#undef cudaMemcpy
extern "C" cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) {
  typedef cudaError_t (*cudaMemcpy_t)( void* dst, const void* src, size_t count, cudaMemcpyKind kind );
  cudaError_t ret_val = cudaErrorUnknown;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMemcpy)(dst, src, count, kind);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}


#undef cudaMemcpy2D
extern "C" cudaError_t cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind ) {
  typedef cudaError_t (*cudaMemcpy2D_t)( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMemcpy2D)(dst, dpitch, src, spitch, width, height, kind);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaMemcpyToArray
extern "C" cudaError_t cudaMemcpyToArray (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind ) {
  typedef cudaError_t (*cudaMemcpyToArray_t)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMemcpyToArray)(dst, wOffset, hOffset, src, count, kind);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaMemcpyToSymbol
extern "C" cudaError_t cudaMemcpyToSymbol ( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind) {
  typedef cudaError_t (*cudaMemcpyToSymbol_t)( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMemcpyToSymbol)(symbol, src, count, offset, kind);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaMemcpyToSymbolAsync
extern "C" cudaError_t cudaMemcpyToSymbolAsync ( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
  typedef cudaError_t (*cudaMemcpyToSymbolAsync_t)( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  cudaStream_t newStream;

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cudaMemcpyToSymbolAsync)(symbol, src, count, offset, kind, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaMemcpyAsync
extern "C" cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  typedef cudaError_t (*cudaMemcpyAsync_t)( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  cudaStream_t newStream;

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  
  // start_jump = clock();
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  // end_jump = clock();
  // dTotal += (end_jump - start_jump);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cudaMemcpyAsync)(dst, src, count, kind, newStream);
  
  // start_jump = clock();
  RETURN_TO_UPPER_HALF();
  // end_jump = clock();
  // dTotal += (end_jump - start_jump);

  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}


#undef cudaMemset
extern "C" cudaError_t cudaMemset(void * devPtr, int value, size_t count) {
  typedef cudaError_t (*cudaMemset_t)(void * devPtr, int value, size_t count);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMemset)(devPtr, value, count);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaMemset2D
extern "C" cudaError_t cudaMemset2D ( void* devPtr, size_t pitch, int value, size_t width, size_t height ) {
  typedef cudaError_t (*cudaMemset2D_t)( void* devPtr, size_t pitch, int value, size_t width, size_t height );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMemset2D)(devPtr, pitch, value, width, height);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaMemsetAsync
extern "C" cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream) {
  typedef cudaError_t (*cudaMemsetAsync_t)(void * devPtr, int value, size_t count, cudaStream_t stream);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  cudaStream_t newStream;

  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cudaMemsetAsync)(devPtr, value, count, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaMemGetInfo
extern "C" cudaError_t cudaMemGetInfo(size_t * free, size_t * total) {
  typedef cudaError_t (*cudaMemGetInfo_t)(size_t * free, size_t * total);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMemGetInfo)(free, total);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaMemAdvise
extern "C" cudaError_t cudaMemAdvise(const void * devPtr, size_t count, enum cudaMemoryAdvise advice, int device) {
  typedef cudaError_t (*cudaMemAdvise_t)(const void * devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaMemAdvise)(devPtr, count, advice, device);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaMemPrefetchAsync
extern "C" cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream) {
  typedef cudaError_t (*cudaMemPrefetchAsync_t)(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  cudaStream_t newStream;
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
    ret_val = REAL_FNC(cudaMemPrefetchAsync)(devPtr, count, dstDevice, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* cudaError_t cudaSetupArgument(const void * arg, size_t size, size_t offset)*/
/* ---- Cuda Stream related APIs ---- */

#undef cudaStreamCreate
extern "C" cudaError_t cudaStreamCreate(cudaStream_t * pStream) {
  typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t * pStream);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaStreamCreate)(pStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaStreamCreate, pStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}


#undef cudaStreamCreateWithPriority
extern "C" cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority ) {
  typedef cudaError_t (*cudaStreamCreateWithPriority_t)( cudaStream_t* pStream, unsigned int  flags, int  priority );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);

  ret_val = REAL_FNC(cudaStreamCreateWithPriority)(pStream, flags, priority);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaStreamCreateWithPriority, pStream, flags, priority, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaStreamCreateWithFlags
extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int flags) {
  typedef cudaError_t (*cudaStreamCreateWithFlags_t)(cudaStream_t * pStream, unsigned int flags);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  //fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaStreamCreateWithFlags)(pStream, flags);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaStreamCreateWithFlags, pStream, flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cudaStreamIsCapturing
extern "C" cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) {
  typedef cudaError_t (*cudaStreamIsCapturing_t)(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaStreamIsCapturing)(stream, pCaptureStatus);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
  
#undef cudaStreamGetCaptureInfo
extern "C" cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId) {
  typedef cudaError_t (*cudaStreamGetCaptureInfo_t)(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId);
  cudaError_t ret_val = cudaSuccess;
  
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);

  cudaStream_t newStream;


  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cudaStreamGetCaptureInfo)(newStream, pCaptureStatus, pId);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cudaStreamDestroy
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  typedef cudaError_t (*cudaStreamDestroy_t)(cudaStream_t stream);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  cudaStream_t newStream;
  LhFreeCudaType_t freeCudaType = (LhFreeCudaType_t)lhInfo.lhFreeCUDATypeFptr;


  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cudaStreamDestroy)(newStream);
  if (newStream !=stream)
    freeCudaType((void*)&stream, CUDA_STREAM);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cudaStreamDestroy, newStream , ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  // printf("after Called at func '%s' in line %i, tid:%d, pid:%d.........\n", __FUNCTION__, __LINE__ , tid, pid);
  // //fflush(stdout);
  return ret_val;
}


#undef cudaStreamSynchronize
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  typedef cudaError_t (*cudaStreamSynchronize_t)(cudaStream_t stream);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  cudaStream_t newStream;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cudaStreamSynchronize)(newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cudaStreamWaitEvent
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  typedef cudaError_t (*cudaStreamWaitEvent_t)(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  cudaStream_t newStream;
  cudaEvent_t newEvent;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  check_cuda_map((void*)&newEvent, (void*)&event, CUDA_EVENT, getCUDAtype);
  ret_val = REAL_FNC(cudaStreamWaitEvent)(newStream, newEvent, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  if(ret_val != cudaSuccess)
    printf("Called at func '%s', ret:%i\n", __FUNCTION__, ret_val);  
  return ret_val;
}

/* Thread related APIs (deprecated) */

#undef cudaThreadSynchronize
extern "C" cudaError_t cudaThreadSynchronize () {
  typedef cudaError_t (*cudaThreadSynchronize_t)();
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaThreadSynchronize)();
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  if(ret_val != cudaSuccess)
    printf("Called at func '%s', ret:%i\n", __FUNCTION__, ret_val);
  return ret_val;
}


#undef cudaThreadExit
extern "C" cudaError_t cudaThreadExit () {
  typedef cudaError_t (*cudaThreadExit_t)();
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaThreadExit)();
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* Miscellaneous API */

#undef cudaPointerGetAttributes
extern "C" cudaError_t cudaPointerGetAttributes ( cudaPointerAttributes* attributes, const void* ptr ) {
  typedef cudaError_t (*cudaPointerGetAttributes_t)( cudaPointerAttributes* attributes, const void* ptr );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaPointerGetAttributes)(attributes, ptr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  // std::cout << "cudaPointerGetAttributes " << ret_val << std::endl;
  return ret_val;
}


#undef cudaGetErrorString
extern "C" const char* cudaGetErrorString ( cudaError_t error ) {
  typedef const char* (*cudaGetErrorString_t)( cudaError_t error );
  const char* ret_val = NULL;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaGetErrorString)(error);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaGetErrorName
extern "C" const char* cudaGetErrorName ( cudaError_t error ) {
  typedef const char* (*cudaGetErrorName_t)( cudaError_t error );
  const char* ret_val = NULL;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaGetErrorName)(error);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}


#undef cudaGetLastError
extern "C" cudaError_t cudaGetLastError() {
  typedef cudaError_t (*cudaGetLastError_t)();
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  // DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  // start_jump = clock();
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  // end_jump = clock();
  // dTotal += (end_jump - start_jump);
  
  ret_val = REAL_FNC(cudaGetLastError)();
  
  // start_jump = clock();
  RETURN_TO_UPPER_HALF();
  // end_jump = clock();
  // dTotal += (end_jump - start_jump);
  // DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaPeekAtLastError
extern "C" cudaError_t cudaPeekAtLastError() {
  typedef cudaError_t (*cudaPeekAtLastError_t)();
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaPeekAtLastError)();
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaFuncSetCacheConfig
extern "C" cudaError_t cudaFuncSetCacheConfig ( const void* func, cudaFuncCache cacheConfig ) {
  typedef cudaError_t (*cudaFuncSetCacheConfig_t)( const void* func, cudaFuncCache cacheConfig );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaFuncSetCacheConfig)(func, cacheConfig);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

/* ---------- CUDART INTERNAL APIs ----------- */

#undef __cudaInitModule
extern "C" char __cudaInitModule(void **fatCubinHandle) {
  typedef char (*__cudaInitModule_t)(void **fatCubinHandle);
  char ret_val = 0;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(!initialized)
    initialize_wrappers();
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(__cudaInitModule)(fatCubinHandle);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc___cudaInitModule, fatCubinHandle, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef __cudaPopCallConfiguration
extern "C" cudaError_t __cudaPopCallConfiguration( dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream ) {
  typedef cudaError_t (*__cudaPopCallConfiguration_t)( dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(!initialized)
    initialize_wrappers();
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  // cudaStream_t newStream;
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  // lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  // check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(__cudaPopCallConfiguration)(gridDim, blockDim, sharedMem, stream);

  RETURN_TO_UPPER_HALF();

  /* Insert logging code here */
  //logAPI(Cuda_Fnc___cudaPopCallConfiguration, gridDim, blockDim, sharedMem, stream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef __cudaPushCallConfiguration
extern "C" unsigned int __cudaPushCallConfiguration( dim3 gridDim, dim3 blockDim, size_t sharedMem, void * stream ) {
  typedef unsigned int (*__cudaPushCallConfiguration_t)( dim3 gridDim, dim3 blockDim, size_t sharedMem, void * stream );
  unsigned int ret_val = 0;

  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(!initialized)
    initialize_wrappers();
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  fsOfNewThread = getFs(tid, pid);
  cudaStream_t newStream;
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(__cudaPushCallConfiguration)(gridDim, blockDim, sharedMem, newStream);
  
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  //logAPI(Cuda_Fnc___cudaPushCallConfiguration, gridDim, blockDim, sharedMem, stream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}

extern std::vector<CudaCallLog_t> cudaCallsLog;

#undef __cudaRegisterFatBinary
extern "C" void** __cudaRegisterFatBinary(void *fatCubin) {
  typedef void** (*__cudaRegisterFatBinary_t)(void *fatCubin);
  void** ret_val = NULL;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  ////printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__);
  if(!initialized)
    initialize_wrappers();
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(__cudaRegisterFatBinary)(fatCubin);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  // printf("return to upper\n");
  // //fflush(stdout);
  logAPI(Cuda_Fnc___cudaRegisterFatBinary, fatCubin, ret_val);
  global_fatCubinHandle = ret_val;
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef __cudaUnregisterFatBinary
extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  typedef void (*__cudaUnregisterFatBinary_t)(void **fatCubinHandle);
  fatHandle_t fat= NULL;
  fat = (fatHandle_t)lhInfo.getFatCubinHandle;
  if(tid == -1){tid = syscall(SYS_gettid); pid = getpid();}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(!initialized)
    initialize_wrappers();
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  void** tmpFatBinary = fat();
  if (tmpFatBinary == NULL) {
  fatCubinHandle = global_fatCubinHandle;
  } else {
   fatCubinHandle =  tmpFatBinary;
  }
  
  // add by tian01.liu, for multiple unregister operation.
  if(fatCubinHandle == NULL)
      return;

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  REAL_FNC(__cudaUnregisterFatBinary)( fatCubinHandle );
  RETURN_TO_UPPER_HALF();

  global_fatCubinHandle = NULL;

  /* Insert logging code here */
  logAPI(Cuda_Fnc___cudaUnregisterFatBinary, fatCubinHandle);
  ////printf("Called at func '%s' in line %i before end.\n", __FUNCTION__, __LINE__ );
  DMTCP_PLUGIN_ENABLE_CKPT();
  ////printf("Called at func '%s' in line %i end.\n", __FUNCTION__, __LINE__ );
  
}


#undef __cudaRegisterFunction
extern "C" void __cudaRegisterFunction( void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize ) {
  typedef void (*__cudaRegisterFunction_t)( void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize );
  pid_t pid = getpid();pid_t tid1 = syscall(SYS_gettid);
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid1);
  if(!initialized)
    initialize_wrappers();
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  unsigned long fsOfNewThread = getFs(tid1, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  REAL_FNC(__cudaRegisterFunction)(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc___cudaRegisterFunction, fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
  // printf("cudaCallsLog size:%ld\n", cudaCallsLog.size());
  DMTCP_PLUGIN_ENABLE_CKPT();
  }


#undef __cudaRegisterManagedVar
extern "C" void __cudaRegisterManagedVar( void **fatCubinHandle, void **hostVarPtrAddress, char  *deviceAddress, const char  *deviceName, int    ext, size_t size, int    constant, int    global ) {
  typedef void (*__cudaRegisterManagedVar_t)( void **fatCubinHandle, void **hostVarPtrAddress, char  *deviceAddress, const char  *deviceName, int    ext, size_t size, int    constant, int    global );
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  if(!initialized)
    initialize_wrappers();
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  REAL_FNC(__cudaRegisterManagedVar)(fatCubinHandle, hostVarPtrAddress, deviceAddress, deviceName, ext, size, constant, global);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc___cudaRegisterManagedVar, fatCubinHandle, hostVarPtrAddress, deviceAddress, deviceName, ext, size, constant, global);
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef __cudaRegisterTexture
extern "C" void __cudaRegisterTexture( void  **fatCubinHandle, const struct textureReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext ) {
  typedef void (*__cudaRegisterTexture_t)( void  **fatCubinHandle, const struct textureReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext );
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  if(!initialized)
    initialize_wrappers();
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  REAL_FNC(__cudaRegisterTexture)(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc___cudaRegisterTexture, fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext);
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef __cudaRegisterSurface
extern "C" void __cudaRegisterSurface( void **fatCubinHandle, const struct surfaceReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext ) {
  typedef void (*__cudaRegisterSurface_t)( void **fatCubinHandle, const struct surfaceReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext );
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  if(!initialized)
    initialize_wrappers();

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  REAL_FNC(__cudaRegisterSurface)(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc___cudaRegisterSurface, fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext);
  DMTCP_PLUGIN_ENABLE_CKPT();
  
}


#undef __cudaRegisterVar
extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char  *deviceAddress, const char  *deviceName, int ext, size_t size, int constant, int global) {
  typedef void (*__cudaRegisterVar_t)(void **fatCubinHandle, char *hostVar, char  *deviceAddress, const char  *deviceName, int ext, size_t size, int constant, int global);
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  if(!initialized)
    initialize_wrappers();
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  REAL_FNC(__cudaRegisterVar)(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc___cudaRegisterVar, fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
  DMTCP_PLUGIN_ENABLE_CKPT();
}


#undef cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags ) {
  typedef cudaError_t (*cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_t)( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags );
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(numBlocks, func, blockSize, dynamicSMemSize, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}


#undef cudaFuncGetAttributes
extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
  typedef cudaError_t (*cudaFuncGetAttributes_t)(struct cudaFuncAttributes *attr, const void *func);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaFuncGetAttributes)(attr, func);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}

/* ----------- CUBLAS APIs ------------ */

#undef cublasCreate_v2
extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t * handle) {
  typedef cublasStatus_t (*cublasCreate_v2_t)(cublasHandle_t * handle);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i. pid:%i, tid:%i\n", __FUNCTION__, __LINE__, pid, tid);fflush(stdout);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasCreate_v2)(handle);
  RETURN_TO_UPPER_HALF();
  //printf("cublasCreate_v2, handle:%p\n", (void*)(*handle));
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cublasCreate_v2, handle, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

__thread__ cublasHandle_t handle_ = nullptr;

#undef cublasSgemm_v2
extern "C" cublasStatus_t cublasSgemm_v2 (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, /* host or device pointer */ const float *A, int lda, const float *B, int ldb, const float *beta, /* host or device pointer */ float *C, int ldc) {
  typedef cublasStatus_t (*cublasSgemm_v2_t)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, /* host or device pointer */ const float *A, int lda, const float *B, int ldb, const float *beta, /* host or device pointer */ float *C, int ldc);
  cublasStatus_t ret_val = CUBLAS_STATUS_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i, handle:%p, handle_:%p.\n", __FUNCTION__, __LINE__ , pid, tid, handle, handle_);
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cublasHandle_t newHandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }
  ret_val = REAL_FNC(cublasSgemm_v2)(newHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasGemmEx
extern "C" cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                                   int n, int k, const void* alpha, /* host or device pointer */const void* A, cudaDataType Atype,
                                                   int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, /* host or device pointer */
                                                   void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo)
{
  typedef cublasStatus_t (*cublasGemmEx_t)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                                   int n, int k, const void* alpha, /* host or device pointer */const void* A, cudaDataType Atype,
                                                   int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, /* host or device pointer */
                                                   void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  cublasStatus_t ret_val = CUBLAS_STATUS_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cublasHandle_t newHandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }
  ret_val = REAL_FNC(cublasGemmEx)(newHandle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasGemmStridedBatchedEx
extern "C" cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,int m,int n, int k,
                        const void* alpha, /* host or device pointer */const void* A,cudaDataType Atype,int lda,long long int strideA, /* purposely signed */
                        const void* B,cudaDataType Btype, int ldb,long long int strideB,const void* beta, /* host or device pointer */void* C,
                        cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo)
{
  typedef cublasStatus_t (*cublasGemmStridedBatchedEx_t)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,int m,int n, int k,
                        const void* alpha, const void* A,cudaDataType Atype,int lda,long long int strideA,
                        const void* B,cudaDataType Btype, int ldb,long long int strideB,const void* beta, void* C,
                        cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  cublasStatus_t ret_val = CUBLAS_STATUS_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cublasHandle_t newHandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }
  ret_val = REAL_FNC(cublasGemmStridedBatchedEx)(newHandle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val; 
}

#undef cublasSgemmStridedBatched
extern "C" cublasStatus_t cublasSgemmStridedBatched (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,  /* host or device pointer */ const float *A, int lda, long long int strideA,   /* purposely signed */ const float *B, int ldb, long long int strideB, const float *beta,   /* host or device pointer */ float *C, int ldc, long long int strideC, int batchCount) {
  typedef cublasStatus_t (*cublasSgemmStridedBatched_t)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,  /* host or device pointer */ const float *A, int lda, long long int strideA,   /* purposely signed */ const float *B, int ldb, long long int strideB, const float *beta,   /* host or device pointer */ float *C, int ldc, long long int strideC, int batchCount);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cublasHandle_t newHandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;

  JUMP_TO_LOWER_HALF(fsOfNewThread);
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }
  ret_val = REAL_FNC(cublasSgemmStridedBatched)(newHandle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasLtCreate
extern "C" cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle) {
  typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t *lightHandle);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasLtCreate)(lightHandle);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
#undef cublasLtDestroy
extern "C" cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
  typedef cublasStatus_t (*cublasLtDestroy_t)(cublasLtHandle_t lightHandle);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasLtDestroy)(lightHandle);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
#undef cublasLtMatmul
extern "C" cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha, /* host or device pointer */ const void *A, cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta, /* host or device pointer */ const void *C, cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasLtMatmul_t)(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha, /* host or device pointer */ const void *A, cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta, /* host or device pointer */ const void *C, cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i. lightHandle:%p, pid:%i, tid:%i \n", __FUNCTION__, __LINE__ ,(void*)lightHandle, pid, tid);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  
  cublasHandle_t* phandle = (cublasHandle_t*)&lightHandle;
  cublasLtHandle_t newLtHandle;
  cublasHandle_t newHandle;
  cudaStream_t newStream;

  
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  cublasHandle_t handle = *phandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }

  newLtHandle = *(cublasLtHandle_t*)&newHandle;
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasLtMatmul)(newLtHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasLtMatmulAlgoGetHeuristic
extern "C" cublasStatus_t cublasLtMatmulAlgoGetHeuristic( cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int *returnAlgoCount) {
  typedef cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic_t)( cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int *returnAlgoCount);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);

  cublasHandle_t* phandle = (cublasHandle_t*)&lightHandle;
  cublasLtHandle_t newLtHandle;
  cublasHandle_t newHandle;
  cublasHandle_t handle = *phandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }

  newLtHandle = *(cublasLtHandle_t*)&newHandle;
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasLtMatmulAlgoGetHeuristic)(newLtHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasSetStream_v2
extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasSetStream_v2_t)(cublasHandle_t handle, cudaStream_t stream);
  cublasStatus_t ret_val = CUBLAS_STATUS_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i, stream:%p.\n", __FUNCTION__, __LINE__ , pid, tid, stream);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);

  cublasHandle_t newHandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }

  cudaStream_t newStream;
  // lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  // check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM);

  JUMP_TO_LOWER_HALF(fsOfNewThread); //replay_gpu_status();
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasSetStream_v2)(newHandle, newStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  if(stream != NULL)
    logAPI(Cuda_Fnc_cublasSetStream_v2, newHandle, newStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasSetMathMode
extern "C" cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
  typedef cublasStatus_t (*cublasSetMathMode_t)(cublasHandle_t handle, cublasMath_t mode);
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i, handle:%p, mode:%i.\n", __FUNCTION__, __LINE__, pid, tid, handle, mode);
  cublasStatus_t ret_val = CUBLAS_STATUS_SUCCESS;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cublasHandle_t newHandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;

  JUMP_TO_LOWER_HALF(fsOfNewThread);
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }
  ret_val = REAL_FNC(cublasSetMathMode)(newHandle, mode);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasGetMathMode
extern "C" cublasStatus_t   cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
  typedef cublasStatus_t (*cublasGetMathMode_t)(cublasHandle_t handle, cublasMath_t *mode);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cublasHandle_t newHandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }

  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasGetMathMode)(newHandle, mode);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
#undef cublasDdot_v2
extern "C" cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double * x, int incx, const double * y, int incy, double * result) {
  typedef cublasStatus_t (*cublasDdot_v2_t)(cublasHandle_t handle, int n, const double * x, int incx, const double * y, int incy, double * result);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasDdot_v2)(handle, n, x, incx, y, incy, result);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cublasDestroy_v2
extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
  typedef cublasStatus_t (*cublasDestroy_v2_t)(cublasHandle_t handle);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  cublasHandle_t newHandle;
  lhCublasType_t getCublasType = (lhCublasType_t)lhInfo.lhCublasTypeFptr;
  void* pHandle = getCublasType((void*)&handle, CUBLAS_HANDLE);
  if (NULL == pHandle) {
    newHandle = handle;
  } else {
    newHandle = *((cublasHandle_t*)pHandle);
  }
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasDestroy_v2)(newHandle);
  RETURN_TO_UPPER_HALF();
  freeCublasType((void*)&handle, CUBLAS_HANDLE);
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cublasDestroy_v2, handle, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDaxpy_v2
extern "C" cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double * alpha, const double * x, int incx, double * y, int incy) {
  typedef cublasStatus_t (*cublasDaxpy_v2_t)(cublasHandle_t handle, int n, const double * alpha, const double * x, int incx, double * y, int incy);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasDaxpy_v2)(handle, n, alpha, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDasum_v2
extern "C" cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
  typedef cublasStatus_t (*cublasDasum_v2_t)(cublasHandle_t handle, int n, const double *x, int incx, double *result);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasDasum_v2)(handle, n, x, incx, result);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDgemm_v2
extern "C" cublasStatus_t cublasDgemm_v2 (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
  typedef cublasStatus_t (*cublasDgemm_v2_t)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasDgemm_v2)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDgemv_v2
extern "C" cublasStatus_t cublasDgemv_v2 (cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
  typedef cublasStatus_t (*cublasDgemv_v2_t)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasDgemv_v2)(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDnrm2_v2
extern "C" cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
  typedef cublasStatus_t (*cublasDnrm2_v2_t)(cublasHandle_t handle, int n, const double *x, int incx, double *result);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasDnrm2_v2)(handle, n, x, incx, result);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDscal_v2
extern "C" cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
  typedef cublasStatus_t (*cublasDscal_v2_t)(cublasHandle_t handle, int n, const double *alpha, double *x, int incx);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasDscal_v2)(handle, n, alpha, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasDswap_v2
extern "C" cublasStatus_t cublasDswap_v2 (cublasHandle_t handle, int n, double *x, int incx, double *y, int incy) {
  typedef cublasStatus_t (*cublasDswap_v2_t)(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasDswap_v2)(handle, n, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasIdamax_v2
extern "C" cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
  typedef cublasStatus_t (*cublasIdamax_v2_t)(cublasHandle_t handle, int n, const double *x, int incx, int *result);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasIdamax_v2)(handle, n, x, incx, result);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasSscal_v2
extern "C" cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
  typedef cublasStatus_t (*cublasSscal_v2_t)(cublasHandle_t handle, int n, const float *alpha, float *x, int incx);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasSscal_v2)(handle, n, alpha, x, incx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
#undef cublasZdotc_v2
extern "C" cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n,const cuDoubleComplex *x, int incx,const cuDoubleComplex *y, int incy, cuDoubleComplex *result) {
  typedef cublasStatus_t (*cublasZdotc_v2_t)(cublasHandle_t handle, int n,const cuDoubleComplex *x, int incx,const cuDoubleComplex *y, int incy, cuDoubleComplex *result);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasZdotc_v2)(handle, n, x, incx, y, incy, result);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

#undef cublasSetMatrix
extern "C" cublasStatus_t cublasSetMatrix (int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb) {
  typedef cublasStatus_t (*cublasSetMatrix_t)(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasSetMatrix)(rows, cols, elemSize, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasGetMatrix
extern "C" cublasStatus_t cublasGetMatrix (int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb) {
  typedef cublasStatus_t (*cublasGetMatrix_t)(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasGetMatrix)(rows, cols, elemSize, A, lda, B, ldb);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSetMatrixAsync
extern "C" cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasSetMatrixAsync_t)(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  cudaStream_t newStream;

  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasSetMatrixAsync)(rows, cols, elemSize, A, lda, B, ldb, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasGetMatrixAsync
extern "C" cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasGetMatrixAsync_t)(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  cudaStream_t newStream;

  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasGetMatrixAsync)(rows, cols, elemSize, A, lda, B, ldb, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSetVector
extern "C" cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
  typedef cublasStatus_t (*cublasSetVector_t)(int n, int elemSize, const void *x, int incx, void *y, int incy);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasSetVector)(n, elemSize, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasGetVector
extern "C" cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
  typedef cublasStatus_t (*cublasGetVector_t)(int n, int elemSize, const void *x, int incx, void *y, int incy);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cublasGetVector)(n, elemSize, x, incx, y, incy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasSetVectorAsync
extern "C" cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasSetVectorAsync_t)(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  cudaStream_t newStream;


  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasSetVectorAsync)(n, elemSize, hostPtr, incx, devicePtr, incy, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cublasGetVectorAsync
extern "C" cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream) {
  typedef cublasStatus_t (*cublasGetVectorAsync_t)(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream);
  cublasStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  cudaStream_t newStream;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cublasGetVectorAsync)(n, elemSize, devicePtr, incx, hostPtr, incy, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* ----------- CUSPARSE APIs ------------ */

#undef cusparseCreate
extern "C" cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) {
  typedef cusparseStatus_t (*cusparseCreate_t)(cusparseHandle_t *handle);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  LhExecApi_t fnc = (LhExecApi_t)(lhInfo.lhExecApiFptr);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseCreate)(handle);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cusparseCreate, handle, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseSetStream
extern "C" cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) {
  typedef cusparseStatus_t (*cusparseSetStream_t)(cusparseHandle_t handle, cudaStream_t streamId);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  cudaStream_t newStream;


  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&streamId, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cusparseSetStream)(handle, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseCreateMatDescr
extern "C" cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t *descrA) {
  typedef cusparseStatus_t (*cusparseCreateMatDescr_t)(cusparseMatDescr_t *descrA);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseCreateMatDescr)(descrA);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cusparseCreateMatDescr, descrA, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseSetMatType
extern "C" cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) {
  typedef cusparseStatus_t (*cusparseSetMatType_t)(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseSetMatType)(descrA, type);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseSetMatIndexBase
extern "C" cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base) {
  typedef cusparseStatus_t (*cusparseSetMatIndexBase_t)(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseSetMatIndexBase)(descrA, base);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseDestroy
extern "C" cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
  typedef cusparseStatus_t (*cusparseDestroy_t)(cusparseHandle_t handle);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseDestroy)(handle);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cusparseDestroy, handle, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseDestroyMatDescr
extern "C" cusparseStatus_t cusparseDestroyMatDescr (cusparseMatDescr_t descrA) {
  typedef cusparseStatus_t (*cusparseDestroyMatDescr_t)(cusparseMatDescr_t descrA);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseDestroyMatDescr)(descrA);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cusparseDestroyMatDescr, descrA, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseGetMatType
extern "C" cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA) {
  typedef cusparseMatrixType_t (*cusparseGetMatType_t)(const cusparseMatDescr_t descrA);
  cusparseMatrixType_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseGetMatType)(descrA);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseSetMatFillMode
extern "C" cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) {
  typedef cusparseStatus_t (*cusparseSetMatFillMode_t)(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseSetMatFillMode)(descrA, fillMode);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseGetMatFillMode
extern "C" cusparseFillMode_t cusparseGetMatFillMode(const cusparseMatDescr_t descrA) {
  typedef cusparseFillMode_t (*cusparseGetMatFillMode_t)(const cusparseMatDescr_t descrA);
  cusparseFillMode_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseGetMatFillMode)(descrA);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseSetMatDiagType
extern "C" cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t  descrA, cusparseDiagType_t diagType) {
  typedef cusparseStatus_t (*cusparseSetMatDiagType_t)(cusparseMatDescr_t  descrA, cusparseDiagType_t diagType);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseSetMatDiagType)(descrA, diagType);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseGetMatDiagType
extern "C" cusparseDiagType_t cusparseGetMatDiagType(const cusparseMatDescr_t descrA) {
  typedef cusparseDiagType_t (*cusparseGetMatDiagType_t)(const cusparseMatDescr_t descrA);
  cusparseDiagType_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseGetMatDiagType)(descrA);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseGetMatIndexBase
extern "C" cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) {
  typedef cusparseIndexBase_t (*cusparseGetMatIndexBase_t)(const cusparseMatDescr_t descrA);
  cusparseIndexBase_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseGetMatIndexBase)(descrA);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusparseSetPointerMode
extern "C" cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
  typedef cusparseStatus_t (*cusparseSetPointerMode_t)(cusparseHandle_t handle, cusparsePointerMode_t mode);
  cusparseStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusparseSetPointerMode)(handle, mode);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* ==== cusolverDn API ==== */

#undef cusolverDnCreate
extern "C" cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle) {
  typedef cusolverStatus_t (*cusolverDnCreate_t)(cusolverDnHandle_t *handle);
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusolverDnCreate)(handle);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cusolverDnCreate, handle, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusolverDnDestroy
extern "C" cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
  typedef cusolverStatus_t (*cusolverDnDestroy_t)(cusolverDnHandle_t handle);
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusolverDnDestroy)(handle);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cusolverDnDestroy, handle, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusolverDnSetStream
extern "C" cusolverStatus_t cusolverDnSetStream (cusolverDnHandle_t handle, cudaStream_t streamId) {
  typedef cusolverStatus_t (*cusolverDnSetStream_t)(cusolverDnHandle_t handle, cudaStream_t streamId);
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  cudaStream_t newStream;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);  
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&streamId, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cusolverDnSetStream)(handle, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusolverDnGetStream
extern "C" cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId) {
  typedef cusolverStatus_t (*cusolverDnGetStream_t)(cusolverDnHandle_t handle, cudaStream_t *streamId);
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusolverDnGetStream)(handle, streamId);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusolverDnDgetrf_bufferSize
extern "C" cusolverStatus_t cusolverDnDgetrf_bufferSize( cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork ) {
  typedef cusolverStatus_t (*cusolverDnDgetrf_bufferSize_t)( cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork );
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusolverDnDgetrf_bufferSize)(handle, m, n, A, lda, Lwork);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusolverDnDgetrf
extern "C" cusolverStatus_t cusolverDnDgetrf( cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo ) {
  typedef cusolverStatus_t (*cusolverDnDgetrf_t)( cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo );
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusolverDnDgetrf)(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusolverDnDgetrs
extern "C" cusolverStatus_t cusolverDnDgetrs( cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, int lda, const int *devIpiv, double *B, int ldb, int *devInfo ) {
  typedef cusolverStatus_t (*cusolverDnDgetrs_t)( cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, int lda, const int *devIpiv, double *B, int ldb, int *devInfo );
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusolverDnDgetrs)(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusolverDnDpotrf_bufferSize
extern "C" cusolverStatus_t cusolverDnDpotrf_bufferSize( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *Lwork ) {
  typedef cusolverStatus_t (*cusolverDnDpotrf_bufferSize_t)( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *Lwork );
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusolverDnDpotrf_bufferSize)(handle, uplo, n, A, lda, Lwork);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusolverDnDpotrf
extern "C" cusolverStatus_t cusolverDnDpotrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, double *Workspace, int Lwork, int *devInfo ) {
  typedef cusolverStatus_t (*cusolverDnDpotrf_t)( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, double *Workspace, int Lwork, int *devInfo );
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusolverDnDpotrf)(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cusolverDnDpotrs
extern "C" cusolverStatus_t cusolverDnDpotrs( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo) {
  typedef cusolverStatus_t (*cusolverDnDpotrs_t)( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo);
  cusolverStatus_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cusolverDnDpotrs)(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* ==== Device APIs ==== */
/* ==== Initialization ==== */

#undef cuInit
extern "C" CUresult cuInit ( unsigned int  Flags ) {
  typedef CUresult (*cuInit_t)( unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cuInit)(Flags);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuInit, Flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* ==== Version Management ==== */

#undef cuDriverGetVersion
extern "C" CUresult cuDriverGetVersion ( int* driverVersion ) {
  typedef CUresult (*cuDriverGetVersion_t)( int* driverVersion );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDriverGetVersion)(driverVersion);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Device Management */
/* Returns a handle to a compute device. */

#undef cuDeviceGet
extern "C" CUresult cuDeviceGet ( CUdevice* device, int  ordinal ) {
  typedef CUresult (*cuDeviceGet_t)( CUdevice* device, int  ordinal );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceGet)(device, ordinal);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDeviceGet, device, ordinal, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

void* handle_of_libcuda = nullptr;
#undef cuGetProcAddress
extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
  //typedef CUresult (*cuGetProcAddress_t)(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);
  CUresult ret_val = CUDA_SUCCESS;
  pid_t pid = getpid();
  pid_t tid = syscall(SYS_gettid);
  //printf("Called at func '%s' in line %i. symbol:%s, cudaVersion:%i, flags:%lu, tid:%d, pid:%d\n", __FUNCTION__, __LINE__, symbol, cudaVersion, flags, tid, pid);fflush(stdout);

  *pfn = dlsym(NULL, symbol);

  /*
  DMTCP_PLUGIN_DISABLE_CKPT(); 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGetProcAddress)(symbol, pfn, cudaVersion, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  // printf("Finish cuGetProcAddress.., symbol:%s pfn:%p\n", symbol, *pfn);
  */
  return ret_val;
}

/* Returns information about the device. */

#undef cuDeviceGetAttribute
extern "C" CUresult cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev ) {
  typedef CUresult (*cuDeviceGetAttribute_t)( int* pi, CUdevice_attribute attrib, CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceGetAttribute)(pi, attrib, dev);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns the number of compute-capable devices. */

#undef cuDeviceGetCount
extern "C" CUresult cuDeviceGetCount ( int* count ) {
  typedef CUresult (*cuDeviceGetCount_t)( int* count );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceGetCount)(count);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* CUresult cuDeviceGetLuid ( char* luid, unsigned int* deviceNodeMask, CUdevice dev ) */
/* Return an LUID and device node mask for the device. */

#undef cuDeviceGetName
extern "C" CUresult cuDeviceGetName ( char* name, int  len, CUdevice dev ) {
  typedef CUresult (*cuDeviceGetName_t)( char* name, int  len, CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceGetName)(name, len, dev);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDeviceGetName, name, len, dev, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  // printf("cuDeviceGetName return is %d\n", ret_val);
  // //fflush(stdout);
  return ret_val;
}

/* Returns an identifer string for the device. */

#undef cuDeviceGetUuid
extern "C" CUresult cuDeviceGetUuid ( CUuuid* uuid, CUdevice dev ) {
  typedef CUresult (*cuDeviceGetUuid_t)( CUuuid* uuid, CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceGetUuid)(uuid, dev);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDeviceGetUuid, uuid, dev, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Return an UUID for the device. */
/* Returns the total amount of memory on the device. */

#undef cuDeviceTotalMem_v2
extern "C" CUresult cuDeviceTotalMem_v2 ( size_t* bytes, CUdevice dev ) {
  typedef CUresult (*cuDeviceTotalMem_v2_t)( size_t* bytes, CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceTotalMem_v2)(bytes, dev);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDeviceTotalMem_v2, bytes, dev, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* ==== Device Management (deprecated) ==== */

#undef cuDeviceComputeCapability
extern "C" CUresult cuDeviceComputeCapability ( int* major, int* minor, CUdevice dev ) {
  typedef CUresult (*cuDeviceComputeCapability_t)( int* major, int* minor, CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceComputeCapability)(major, minor, dev);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDeviceComputeCapability, major, minor, dev, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns the compute capability of the device. */

#undef cuDeviceGetProperties
extern "C" CUresult cuDeviceGetProperties ( CUdevprop* prop, CUdevice dev ) {
  typedef CUresult (*cuDeviceGetProperties_t)( CUdevprop* prop, CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceGetProperties)(prop, dev);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDeviceGetProperties, prop, dev, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns properties for a selected device. */
/* ==== Primary Context Management ==== */

#undef cuDevicePrimaryCtxGetState
extern "C" CUresult cuDevicePrimaryCtxGetState ( CUdevice dev, unsigned int* flags, int* active ) {
  typedef CUresult (*cuDevicePrimaryCtxGetState_t)( CUdevice dev, unsigned int* flags, int* active );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDevicePrimaryCtxGetState)(dev, flags, active);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDevicePrimaryCtxGetState, dev, flags, active, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Get the state of the primary context. */

#undef cuDevicePrimaryCtxRelease_v2
extern "C" CUresult cuDevicePrimaryCtxRelease_v2 (CUdevice dev ) {
  typedef CUresult (*cuDevicePrimaryCtxRelease_v2_t)(CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDevicePrimaryCtxRelease_v2)(dev);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDevicePrimaryCtxRelease_v2, dev, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
/* Release the primary context on GPU. */

#undef cuDevicePrimaryCtxReset_v2
extern "C" CUresult cuDevicePrimaryCtxReset_v2 (CUdevice dev ) {
  typedef CUresult (*cuDevicePrimaryCtxReset_v2_t)(CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDevicePrimaryCtxReset_v2)(dev);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  fprintf(stdout, "[XB] dev = %i, at func '%s' in line %i.\n", dev, __FUNCTION__, __LINE__);
  logAPI(Cuda_Fnc_cuDevicePrimaryCtxReset_v2, dev, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
/* Reset the primary context on GPU. */

#undef cuDevicePrimaryCtxRetain
extern "C" CUresult cuDevicePrimaryCtxRetain ( CUcontext* pctx, CUdevice dev ) {
  typedef CUresult (*cuDevicePrimaryCtxRetain_t)( CUcontext* pctx, CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDevicePrimaryCtxRetain)(pctx, dev);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDevicePrimaryCtxRetain, pctx, dev, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Retain the primary context on the GPU. */
/* ==== Context Management ==== */

#undef cuCtxCreate_v2
extern "C" CUresult cuCtxCreate_v2 ( CUcontext* pctx, unsigned int  flags, CUdevice dev ) {
  typedef CUresult (*cuCtxCreate_v2_t)( CUcontext* pctx, unsigned int  flags, CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxCreate_v2)(pctx, flags, dev);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxCreate_v2, pctx, flags, dev, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Create a CUDA context. */

#undef cuCtxDestroy_v2
extern "C" CUresult cuCtxDestroy_v2 ( CUcontext ctx ) {
  typedef CUresult (*cuCtxDestroy_v2_t)( CUcontext ctx );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  int freeFlag = 0;
  CUcontext newContext;
  check_cu_map((void*)&newContext, (void*)&ctx, CU_CONTEXT, freeFlag);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxDestroy_v2)(newContext);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxDestroy_v2, newContext, ret_val);
  lhFreeCUtype_t freeCUtype = (lhFreeCUtype_t)lhInfo.lhFreeCUtypeFptr;
  if (1 == freeFlag) {
    //Erase the record from map, the key is ctx
    freeCUtype((void*)&ctx, CU_CONTEXT);
  }
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroy a CUDA context. */

#undef cuCtxGetApiVersion
extern "C" CUresult cuCtxGetApiVersion ( CUcontext ctx, unsigned int* version ) {
  typedef CUresult (*cuCtxGetApiVersion_t)( CUcontext ctx, unsigned int* version );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUcontext newContext;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newContext, (void*)&ctx, CU_CONTEXT, getCUtype);
  ret_val = REAL_FNC(cuCtxGetApiVersion)(newContext, version);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxGetApiVersion, newContext, version, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the context's API version. */

#undef cuCtxGetCacheConfig
extern "C" CUresult cuCtxGetCacheConfig ( CUfunc_cache* pconfig ) {
  typedef CUresult (*cuCtxGetCacheConfig_t)( CUfunc_cache* pconfig );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxGetCacheConfig)(pconfig);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxGetCacheConfig, pconfig, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns the preferred cache configuration for the current context. */

#undef cuCtxGetCurrent
extern "C" CUresult cuCtxGetCurrent ( CUcontext* pctx ) {
  typedef CUresult (*cuCtxGetCurrent_t)( CUcontext* pctx );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxGetCurrent)(pctx);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxGetCurrent, pctx, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns the CUDA context bound to the calling CPU thread. */

#undef cuCtxGetDevice
extern "C" CUresult cuCtxGetDevice ( CUdevice* device ) {
  typedef CUresult (*cuCtxGetDevice_t)( CUdevice* device );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxGetDevice)(device);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxGetDevice, device, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns the device ID for the current context. */

#undef cuCtxGetFlags
extern "C" CUresult cuCtxGetFlags ( unsigned int* flags ) {
  typedef CUresult (*cuCtxGetFlags_t)( unsigned int* flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxGetFlags)(flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns the flags for the current context. */

#undef cuCtxGetLimit
extern "C" CUresult cuCtxGetLimit ( size_t* pvalue, CUlimit limit ) {
  typedef CUresult (*cuCtxGetLimit_t)( size_t* pvalue, CUlimit limit );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxGetLimit)(pvalue, limit);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxGetLimit, pvalue, limit, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns resource limits. */

#undef cuCtxGetSharedMemConfig
extern "C" CUresult cuCtxGetSharedMemConfig ( CUsharedconfig* pConfig ) {
  typedef CUresult (*cuCtxGetSharedMemConfig_t)( CUsharedconfig* pConfig );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxGetSharedMemConfig)(pConfig);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxGetSharedMemConfig, pConfig, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns the current shared memory configuration for the current context. */

#undef cuCtxGetStreamPriorityRange
extern "C" CUresult cuCtxGetStreamPriorityRange ( int* leastPriority, int* greatestPriority ) {
  typedef CUresult (*cuCtxGetStreamPriorityRange_t)( int* leastPriority, int* greatestPriority );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxGetStreamPriorityRange)(leastPriority, greatestPriority);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxGetStreamPriorityRange, leastPriority, greatestPriority, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns numerical values that correspond to the least and greatest stream priorities. */

#undef cuCtxPopCurrent_v2
extern "C" CUresult cuCtxPopCurrent_v2 ( CUcontext* pctx ) {
  typedef CUresult (*cuCtxPopCurrent_v2_t)( CUcontext* pctx );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxPopCurrent_v2)(pctx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Pops the current CUDA context from the current CPU thread. */

#undef cuCtxPushCurrent_v2
extern "C" CUresult cuCtxPushCurrent_v2 ( CUcontext ctx ) {
  typedef CUresult (*cuCtxPushCurrent_v2_t)( CUcontext ctx );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxPushCurrent_v2)(ctx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Pushes a context on the current CPU thread. */

#undef cuCtxSetCacheConfig
extern "C" CUresult cuCtxSetCacheConfig ( CUfunc_cache config ) {
  typedef CUresult (*cuCtxSetCacheConfig_t)( CUfunc_cache config );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxSetCacheConfig)(config);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the preferred cache configuration for the current context. */

#undef cuCtxSetCurrent
extern "C" CUresult cuCtxSetCurrent ( CUcontext ctx ) {
  typedef CUresult (*cuCtxSetCurrent_t)( CUcontext ctx );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxSetCurrent)(ctx);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxSetCurrent, ctx, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Binds the specified CUDA context to the calling CPU thread. */

#undef cuCtxSetLimit
extern "C" CUresult cuCtxSetLimit ( CUlimit limit, size_t value ) {
  typedef CUresult (*cuCtxSetLimit_t)( CUlimit limit, size_t value );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxSetLimit)(limit, value);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxSetLimit, limit, value, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Set resource limits. */

#undef cuCtxSetSharedMemConfig
extern "C" CUresult cuCtxSetSharedMemConfig ( CUsharedconfig config ) {
  typedef CUresult (*cuCtxSetSharedMemConfig_t)( CUsharedconfig config );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxSetSharedMemConfig)(config);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the shared memory configuration for the current context. */

#undef cuCtxSynchronize
extern "C" CUresult cuCtxSynchronize () {
  typedef CUresult (*cuCtxSynchronize_t)();
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxSynchronize)();
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuCtxSynchronize, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Block for a context's tasks to complete. */
/* ==== Context Management [DEPRECATED] ==== */

#undef cuCtxAttach
extern "C" CUresult cuCtxAttach ( CUcontext* pctx, unsigned int  flags ) {
  typedef CUresult (*cuCtxAttach_t)( CUcontext* pctx, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxAttach)(pctx, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Increment a context's usage-count. */

#undef cuCtxDetach
extern "C" CUresult cuCtxDetach ( CUcontext ctx ) {
  typedef CUresult (*cuCtxDetach_t)( CUcontext ctx );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxDetach)(ctx);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Decrement a context's usage-count. */
/* ==== Module Management ==== */

#undef cuLinkAddData_v2
extern "C" CUresult cuLinkAddData_v2 ( CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues ) {
  typedef CUresult (*cuLinkAddData_v2_t)( CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLinkAddData_v2)(state, type, data, size, name, numOptions, options, optionValues);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Add an input to a pending linker invocation. */

#undef cuLinkAddFile_v2
extern "C" CUresult cuLinkAddFile_v2 ( CUlinkState state, CUjitInputType type, const char* path, unsigned int  numOptions, CUjit_option* options, void** optionValues ) {
  typedef CUresult (*cuLinkAddFile_v2_t)( CUlinkState state, CUjitInputType type, const char* path, unsigned int  numOptions, CUjit_option* options, void** optionValues );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLinkAddFile_v2)(state, type, path, numOptions, options, optionValues);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Add a file input to a pending linker invocation. */

#undef cuLinkComplete
extern "C" CUresult cuLinkComplete ( CUlinkState state, void** cubinOut, size_t* sizeOut ) {
  typedef CUresult (*cuLinkComplete_t)( CUlinkState state, void** cubinOut, size_t* sizeOut );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLinkComplete)(state, cubinOut, sizeOut);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Complete a pending linker invocation. */

#undef cuLinkCreate_v2
extern "C" CUresult cuLinkCreate_v2 ( unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut ) {
  typedef CUresult (*cuLinkCreate_v2_t)( unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLinkCreate_v2)(numOptions, options, optionValues, stateOut);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuLinkCreate_v2, numOptions, options, optionValues, stateOut, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Creates a pending JIT linker invocation. */

#undef cuLinkDestroy
extern "C" CUresult cuLinkDestroy ( CUlinkState state ) {
  typedef CUresult (*cuLinkDestroy_t)( CUlinkState state );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLinkDestroy)(state);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuLinkDestroy, state, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


/* Destroys state for a JIT linker invocation. */

#undef cuModuleGetFunction
extern "C" CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ) {
  typedef CUresult (*cuModuleGetFunction_t)( CUfunction* hfunc, CUmodule hmod, const char* name );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  LhExecApi_t fnc = (LhExecApi_t)(lhInfo.lhExecApiFptr);
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  CUmodule newModule;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newModule, (void*)&hmod, CU_MODULE, getCUtype);
  ret_val = REAL_FNC(cuModuleGetFunction)(hfunc, newModule, name);
  RETURN_TO_UPPER_HALF();
  // printf("%d............, ret_val = %d\n", __LINE__, ret_val);
/* Insert logging code here */
  logAPI(Cuda_Fnc_cuModuleGetFunction, hfunc, newModule, name, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a function handle. */

#undef cuModuleGetGlobal_v2
extern "C" CUresult cuModuleGetGlobal_v2 ( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name ) {
  typedef CUresult (*cuModuleGetGlobal_v2_t)( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuModuleGetGlobal_v2)(dptr, bytes, hmod, name);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a global pointer from a module. */

#undef cuModuleGetSurfRef
extern "C" CUresult cuModuleGetSurfRef ( CUsurfref* pSurfRef, CUmodule hmod, const char* name ) {
  typedef CUresult (*cuModuleGetSurfRef_t)( CUsurfref* pSurfRef, CUmodule hmod, const char* name );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuModuleGetSurfRef)(pSurfRef, hmod, name);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a handle to a surface reference. */

#undef cuModuleGetTexRef
extern "C" CUresult cuModuleGetTexRef ( CUtexref* pTexRef, CUmodule hmod, const char* name ) {
  typedef CUresult (*cuModuleGetTexRef_t)( CUtexref* pTexRef, CUmodule hmod, const char* name );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  CUmodule newModule;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newModule, (void*)&hmod, CU_MODULE, getCUtype);
  ret_val = REAL_FNC(cuModuleGetTexRef)(pTexRef, newModule, name);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  logAPI(Cuda_Fnc_cuModuleGetTexRef, pTexRef, newModule, name, ret_val);
  return ret_val;
}

/* Returns a handle to a texture reference. */

#undef cuModuleLoad
extern "C" CUresult cuModuleLoad ( CUmodule* module, const char* fname ) {
  typedef CUresult (*cuModuleLoad_t)( CUmodule* module, const char* fname );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuModuleLoad)(module, fname);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Loads a compute module. */

#undef cuModuleLoadData
extern "C" CUresult cuModuleLoadData ( CUmodule* module, const void* image ) {
  typedef CUresult (*cuModuleLoadData_t)( CUmodule* module, const void* image );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuModuleLoadData)(module, image);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuModuleLoadData, module, image, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Load a module's data. */

#undef cuModuleLoadDataEx
extern "C" CUresult cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues ) {
  typedef CUresult (*cuModuleLoadDataEx_t)( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuModuleLoadDataEx)(module, image, numOptions, options, optionValues);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuModuleLoadDataEx, module, image, numOptions, options, optionValues, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Load a module's data with options. */

#undef cuModuleLoadFatBinary
extern "C" CUresult cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin ) {
  typedef CUresult (*cuModuleLoadFatBinary_t)( CUmodule* module, const void* fatCubin );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuModuleLoadFatBinary)(module, fatCubin);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Load a module's data. */

#undef cuModuleUnload
extern "C" CUresult cuModuleUnload ( CUmodule hmod ) {
  typedef CUresult (*cuModuleUnload_t)( CUmodule hmod );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuModuleUnload)(hmod);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Unloads a module. */
/* ==== Memory Management ==== */

#undef cuArray3DCreate_v2
extern "C" CUresult cuArray3DCreate_v2 ( CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray ) {
  typedef CUresult (*cuArray3DCreate_v2_t)( CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuArray3DCreate_v2)(pHandle, pAllocateArray);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuArray3DCreate_v2, pHandle, pAllocateArray, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Creates a 3D CUDA array. */

#undef cuArray3DGetDescriptor_v2
extern "C" CUresult cuArray3DGetDescriptor_v2 ( CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray ) {
  typedef CUresult (*cuArray3DGetDescriptor_v2_t)( CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuArray3DGetDescriptor_v2)(pArrayDescriptor, hArray);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Get a 3D CUDA array descriptor. */

#undef cuArrayCreate_v2
extern "C" CUresult cuArrayCreate_v2 ( CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray ) {
  typedef CUresult (*cuArrayCreate_v2_t)( CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuArrayCreate_v2)(pHandle, pAllocateArray);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuArrayCreate_v2, pHandle, pAllocateArray, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Creates a 1D or 2D CUDA array. */

#undef cuArrayDestroy
extern "C" CUresult cuArrayDestroy ( CUarray hArray ) {
  typedef CUresult (*cuArrayDestroy_t)( CUarray hArray );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuArrayDestroy)(hArray);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuArrayDestroy, hArray, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys a CUDA array. */

#undef cuArrayGetDescriptor_v2
extern "C" CUresult cuArrayGetDescriptor_v2 ( CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray ) {
  typedef CUresult (*cuArrayGetDescriptor_v2_t)( CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuArrayGetDescriptor_v2)(pArrayDescriptor, hArray);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Get a 1D or 2D CUDA array descriptor. */

#undef cuDeviceGetByPCIBusId
extern "C" CUresult cuDeviceGetByPCIBusId ( CUdevice* dev, const char* pciBusId ) {
  typedef CUresult (*cuDeviceGetByPCIBusId_t)( CUdevice* dev, const char* pciBusId );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceGetByPCIBusId)(dev, pciBusId);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a handle to a compute device. */

#undef cuDeviceGetPCIBusId
extern "C" CUresult cuDeviceGetPCIBusId ( char* pciBusId, int  len, CUdevice dev ) {
  typedef CUresult (*cuDeviceGetPCIBusId_t)( char* pciBusId, int  len, CUdevice dev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceGetPCIBusId)(pciBusId, len, dev);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a PCI Bus Id string for the device. */

#undef cuIpcCloseMemHandle
extern "C" CUresult cuIpcCloseMemHandle ( CUdeviceptr dptr ) {
  typedef CUresult (*cuIpcCloseMemHandle_t)( CUdeviceptr dptr );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuIpcCloseMemHandle)(dptr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Close memory mapped with cuIpcOpenMemHandle. */

#undef cuIpcGetEventHandle
extern "C" CUresult cuIpcGetEventHandle ( CUipcEventHandle* pHandle, CUevent event ) {
  typedef CUresult (*cuIpcGetEventHandle_t)( CUipcEventHandle* pHandle, CUevent event );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuIpcGetEventHandle)(pHandle, event);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets an interprocess handle for a previously allocated event. */

#undef cuIpcGetMemHandle
extern "C" CUresult cuIpcGetMemHandle ( CUipcMemHandle* pHandle, CUdeviceptr dptr ) {
  typedef CUresult (*cuIpcGetMemHandle_t)( CUipcMemHandle* pHandle, CUdeviceptr dptr );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuIpcGetMemHandle)(pHandle, dptr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets an interprocess memory handle for an existing device memory allocation. */

#undef cuIpcOpenEventHandle
extern "C" CUresult cuIpcOpenEventHandle ( CUevent* phEvent, CUipcEventHandle handle ) {
  typedef CUresult (*cuIpcOpenEventHandle_t)( CUevent* phEvent, CUipcEventHandle handle );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuIpcOpenEventHandle)(phEvent, handle);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Opens an interprocess event handle for use in the current process. */
int num_cuMemAlloc = 0;
#undef cuMemAlloc_v2
extern "C" CUresult cuMemAlloc_v2 ( CUdeviceptr* dptr, size_t bytesize ) {
  //typedef CUresult (*cuMemAlloc_v2_t)( CUdeviceptr* dptr, size_t bytesize );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  LhMMUAllocate_t func = (LhMMUAllocate_t)lhInfo.lhMMUAllocFptr;
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  *dptr = (CUdeviceptr)func(bytesize);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuMemAlloc_v2, dptr, bytesize, ret_val);
  printf("cuMemAlloc_v2 return is %d, devPtr:%p, fsOfNewThread:%ld\n", ret_val, (void*)(*dptr), fsOfNewThread);
  // //fflush(stdout);
  DMTCP_PLUGIN_ENABLE_CKPT();
#ifdef TIMING
  num_cuMemAlloc++;
  int check_count = DMTCP_GET_COUNT();
  char* check_name = DMTCP_GET_NAME();
  std::string tem_check_name = check_name;

  if (check_count == num_cuMemAlloc && tem_check_name == "cuMemAlloc"){
    printf("checkpointing after cuMemAlloc...........................\n");
    sendMsgToDmtcp(0);
  }
#else
  printf("Disable timing execution in cuda api cuMemAlloc !\n");
#endif
  return ret_val;
}

/* Allocates device memory. */

#undef cuMemAllocHost_v2
extern "C" CUresult cuMemAllocHost_v2 ( void** pp, size_t bytesize ) {
  // typedef CUresult (*cuMemAllocHost_v2_t)( void** pp, size_t bytesize );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  LhAllocHost_t func = (LhAllocHost_t)lhInfo.lhAllocHostFptr;
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = (CUresult)func(pp, bytesize, 0x01);
  RETURN_TO_UPPER_HALF();
  // printf("[lt] cuMemAllocHost_v2, addr:%p, size:%ld\n", *pp, bytesize);
  if(g_pin_record)
    g_addrs_for_host[*pp] = {0x01, bytesize};;
  /* Insert logging code here */
  // logAPI(Cuda_Fnc_cuMemAllocHost_v2, pp, bytesize, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Allocates page-locked host memory. */
#undef cuMemAllocManaged
extern "C" CUresult cuMemAllocManaged ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags ) {
  typedef CUresult (*cuMemAllocManaged_t)( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemAllocManaged)(dptr, bytesize, flags);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuMemAllocManaged, dptr, bytesize, flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Allocates memory that will be automatically managed by the Unified Memory system. */

#undef cuMemAllocPitch_v2
extern "C" CUresult cuMemAllocPitch_v2 ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes ) {
  typedef CUresult (*cuMemAllocPitch_v2_t)( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemAllocPitch_v2)(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Allocates pitched device memory. */

#undef cuMemFree_v2
extern "C" CUresult cuMemFree_v2 ( CUdeviceptr dptr ) {
  //typedef CUresult (*cuMemFree_v2_t)( CUdeviceptr dptr );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  LhMMUFree_t func = (LhMMUFree_t)lhInfo.lhMMUFreeFptr;
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  func((void*)dptr);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuMemFree_v2, dptr, ret_val); // modify from newDevptr to dptr by tian01.liu 2023.1.9
  // printf("ret_val = %d 2\n", ret_val);
  // //fflush(stdout);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Frees device memory. */
#undef cuMemFreeHost
extern "C" CUresult cuMemFreeHost ( void* p ) {
  // typedef CUresult (*cuMemFreeHost_t)( void* p );
  CUresult ret_val = CUDA_SUCCESS;
  //printf("Called at func '%s' in line %i. ptr:%p\n", __FUNCTION__, __LINE__, p );
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  LhFreeHost_t func = (LhFreeHost_t)lhInfo.lhFreeHostFptr;
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = (CUresult)func(p);
  RETURN_TO_UPPER_HALF();
  g_addrs_for_host.erase(p);
  /* Insert logging code here */
  //logAPI(Cuda_Fnc_cuMemFreeHost, p, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Frees page-locked host memory. */

#undef cuMemGetAddressRange_v2
extern "C" CUresult cuMemGetAddressRange_v2 ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr ) {
  typedef CUresult (*cuMemGetAddressRange_v2_t)( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = REAL_FNC(cuMemGetAddressRange_v2)(pbase, psize, dptr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Get information on memory allocations. */

#undef cuMemGetInfo_v2
extern "C" CUresult cuMemGetInfo_v2 ( size_t* free, size_t* total ) {
  typedef CUresult (*cuMemGetInfo_v2_t)( size_t* free, size_t* total );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = REAL_FNC(cuMemGetInfo_v2)(free, total);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets free and total memory. */

#undef cuMemHostAlloc
extern "C" CUresult cuMemHostAlloc ( void** pp, size_t bytesize, unsigned int  Flags ) {
  // typedef CUresult (*cuMemHostAlloc_t)( void** pp, size_t bytesize, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  LhAllocHost_t func = (LhAllocHost_t)lhInfo.lhAllocHostFptr;
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = (CUresult)func(pp, bytesize, Flags);
  // ret_val = REAL_FNC(cuMemHostAlloc)(pp, bytesize, Flags);
  RETURN_TO_UPPER_HALF();
  // printf("[lt] cuMemHostAlloc, addr:%p, fsOfNewThread:%ld\n", *pp, fsOfNewThread);
  if(g_pin_record)
    g_addrs_for_host[*pp] = {Flags, bytesize};

  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Allocates page-locked host memory. */

#undef cuMemHostGetDevicePointer_v2
extern "C" CUresult cuMemHostGetDevicePointer_v2 ( CUdeviceptr* pdptr, void* p, unsigned int  Flags ) {
  typedef CUresult (*cuMemHostGetDevicePointer_v2_t)( CUdeviceptr* pdptr, void* p, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = REAL_FNC(cuMemHostGetDevicePointer_v2)(pdptr, p, Flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Passes back device pointer of mapped pinned memory. */

#undef cuMemHostGetFlags
extern "C" CUresult cuMemHostGetFlags ( unsigned int* pFlags, void* p ) {
  typedef CUresult (*cuMemHostGetFlags_t)( unsigned int* pFlags, void* p );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = REAL_FNC(cuMemHostGetFlags)(pFlags, p);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Passes back flags that were used for a pinned allocation. */

#undef cuMemHostRegister_v2
extern "C" CUresult cuMemHostRegister_v2 ( void* p, size_t bytesize, unsigned int  Flags ) {
  typedef CUresult (*cuMemHostRegister_v2_t)( void* p, size_t bytesize, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = REAL_FNC(cuMemHostRegister_v2)(p, bytesize, Flags);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuMemHostRegister_v2, p, bytesize, Flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Registers an existing host memory range for use by CUDA. */

#undef cuMemHostUnregister
extern "C" CUresult cuMemHostUnregister ( void* p ) {
  typedef CUresult (*cuMemHostUnregister_t)( void* p );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = REAL_FNC(cuMemHostUnregister)(p);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuMemHostUnregister, p, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Unregisters a memory range that was registered with cuMemHostRegister. */

#undef cuMemcpy
extern "C" CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount ) {
  typedef CUresult (*cuMemcpy_t)( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = REAL_FNC(cuMemcpy)(dst, src, ByteCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory. */

#undef cuMemcpy2D_v2
extern "C" CUresult cuMemcpy2D_v2 ( const CUDA_MEMCPY2D* pCopy ) {
  typedef CUresult (*cuMemcpy2D_v2_t)( const CUDA_MEMCPY2D* pCopy );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpy2D_v2)(pCopy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory for 2D arrays. */

#undef cuMemcpy2DAsync_v2
extern "C" CUresult cuMemcpy2DAsync_v2 ( const CUDA_MEMCPY2D* pCopy, CUstream hStream ) {
  typedef CUresult (*cuMemcpy2DAsync_v2_t)( const CUDA_MEMCPY2D* pCopy, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemcpy2DAsync_v2)(pCopy, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory for 2D arrays. */

#undef cuMemcpy2DUnaligned_v2
extern "C" CUresult cuMemcpy2DUnaligned_v2 ( const CUDA_MEMCPY2D* pCopy ) {
  typedef CUresult (*cuMemcpy2DUnaligned_v2_t)( const CUDA_MEMCPY2D* pCopy );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);;
  ret_val = REAL_FNC(cuMemcpy2DUnaligned_v2)(pCopy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory for 2D arrays. */

#undef cuMemcpy3D_v2
extern "C" CUresult cuMemcpy3D_v2 ( const CUDA_MEMCPY3D* pCopy ) {
  typedef CUresult (*cuMemcpy3D_v2_t)( const CUDA_MEMCPY3D* pCopy );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpy3D_v2)(pCopy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory for 3D arrays. */

#undef cuMemcpy3DAsync_v2
extern "C" CUresult cuMemcpy3DAsync_v2 ( const CUDA_MEMCPY3D* pCopy, CUstream hStream ) {
  typedef CUresult (*cuMemcpy3DAsync_v2_t)( const CUDA_MEMCPY3D* pCopy, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemcpy3DAsync_v2)(pCopy, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory for 3D arrays. */

#undef cuMemcpy3DPeer
extern "C" CUresult cuMemcpy3DPeer ( const CUDA_MEMCPY3D_PEER* pCopy ) {
  typedef CUresult (*cuMemcpy3DPeer_t)( const CUDA_MEMCPY3D_PEER* pCopy );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpy3DPeer)(pCopy);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory between contexts. */

#undef cuMemcpy3DPeerAsync
extern "C" CUresult cuMemcpy3DPeerAsync ( const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream ) {
  typedef CUresult (*cuMemcpy3DPeerAsync_t)( const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemcpy3DPeerAsync)(pCopy, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory between contexts asynchronously. */

#undef cuMemcpyAsync
extern "C" CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream ) {
  typedef CUresult (*cuMemcpyAsync_t)( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemcpyAsync)(dst, src, ByteCount, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory asynchronously. */

#undef cuMemcpyAtoA_v2
extern "C" CUresult cuMemcpyAtoA_v2 ( CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount ) {
  typedef CUresult (*cuMemcpyAtoA_v2_t)( CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyAtoA_v2)(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Array to Array. */

#undef cuMemcpyAtoD_v2
extern "C" CUresult cuMemcpyAtoD_v2 ( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount ) {
  typedef CUresult (*cuMemcpyAtoD_v2_t)( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyAtoD_v2)(dstDevice, srcArray, srcOffset, ByteCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Array to Device. */

#undef cuMemcpyAtoH_v2
extern "C" CUresult cuMemcpyAtoH_v2 ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount ) {
  typedef CUresult (*cuMemcpyAtoH_v2_t)( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyAtoH_v2)(dstHost, srcArray, srcOffset, ByteCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Array to Host. */

#undef cuMemcpyAtoHAsync_v2
extern "C" CUresult cuMemcpyAtoHAsync_v2 ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream ) {
  typedef CUresult (*cuMemcpyAtoHAsync_v2_t)( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyAtoHAsync_v2)(dstHost, srcArray, srcOffset, ByteCount, hStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Array to Host. */

#undef cuMemcpyDtoA_v2
extern "C" CUresult cuMemcpyDtoA_v2 ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount ) {
  typedef CUresult (*cuMemcpyDtoA_v2_t)( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyDtoA_v2)(dstArray, dstOffset, srcDevice, ByteCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Device to Array. */

#undef cuMemcpyDtoD_v2
extern "C" CUresult cuMemcpyDtoD_v2 ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount ) {
  typedef CUresult (*cuMemcpyDtoD_v2_t)( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyDtoD_v2)(dstDevice, srcDevice, ByteCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Device to Device. */

#undef cuMemcpyDtoDAsync_v2
extern "C" CUresult cuMemcpyDtoDAsync_v2 ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ) {
  typedef CUresult (*cuMemcpyDtoDAsync_v2_t)( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemcpyDtoDAsync_v2)(dstDevice, srcDevice, ByteCount, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Device to Device. */

#undef cuMemcpyDtoH_v2
extern "C" CUresult cuMemcpyDtoH_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount ) {
  typedef CUresult (*cuMemcpyDtoH_v2_t)( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyDtoH_v2)(dstHost, srcDevice, ByteCount);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  //logAPI(Cuda_Fnc_cuMemcpyDtoH_v2, dstHost, srcDevice, ByteCount, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Device to Host. */

#undef cuMemcpyDtoHAsync_v2
extern "C" CUresult cuMemcpyDtoHAsync_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ) {
  typedef CUresult (*cuMemcpyDtoHAsync_v2_t)( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemcpyDtoHAsync_v2)(dstHost, srcDevice, ByteCount, newStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  //logAPI(Cuda_Fnc_cuMemcpyDtoHAsync_v2, dstHost, srcDevice, ByteCount, newStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Host to Array. */
#undef cuMemcpyHtoA_v2
extern "C" CUresult cuMemcpyHtoA_v2 ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount ) {
  typedef CUresult (*cuMemcpyHtoA_v2_t)( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyHtoA_v2)(dstArray, dstOffset, srcHost, ByteCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Host to Array. */

#undef cuMemcpyHtoAAsync_v2
extern "C" CUresult cuMemcpyHtoAAsync_v2 ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream ) {
  typedef CUresult (*cuMemcpyHtoAAsync_v2_t)( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyHtoAAsync_v2)(dstArray, dstOffset, srcHost, ByteCount, hStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Host to Array. */

#undef cuMemcpyHtoD_v2
extern "C" CUresult cuMemcpyHtoD_v2 ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount ) {
  typedef CUresult (*cuMemcpyHtoD_v2_t)( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i. destDevice:%p, srcPtr:%p\n", __FUNCTION__, __LINE__ , pid, tid, (void*)dstDevice, srcHost);
  fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyHtoD_v2)(dstDevice, srcHost, ByteCount);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  //logAPI(Cuda_Fnc_cuMemcpyHtoD_v2, newDevptr, srcHost, ByteCount, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Host to Device. */

#undef cuMemcpyHtoDAsync_v2
extern "C" CUresult cuMemcpyHtoDAsync_v2 ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream ) {
  typedef CUresult (*cuMemcpyHtoDAsync_v2_t)( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);fflush(stdout);
  //fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  // printf("dstDevice %p srcHost %p ByteCount %ld stream %p", (void*)dstDevice, srcHost, ByteCount, newStream);fflush(stdout);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemcpyHtoDAsync_v2)(dstDevice, srcHost, ByteCount, newStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  //logAPI(Cuda_Fnc_cuMemcpyHtoDAsync_v2, newDevptr, srcHost, ByteCount, newStream, ret_val);
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);fflush(stdout);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies memory from Host to Device. */

#undef cuMemcpyPeer
extern "C" CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount ) {
  typedef CUresult (*cuMemcpyPeer_t)( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemcpyPeer)(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies device memory between two contexts. */

#undef cuMemcpyPeerAsync
extern "C" CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream ) {
  typedef CUresult (*cuMemcpyPeerAsync_t)( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemcpyPeerAsync)(dstDevice, dstContext, srcDevice, srcContext, ByteCount, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Copies device memory between two contexts asynchronously. */

#undef cuMemsetD16_v2
extern "C" CUresult cuMemsetD16_v2 ( CUdeviceptr dstDevice, unsigned short us, size_t N ) {
  typedef CUresult (*cuMemsetD16_v2_t)( CUdeviceptr dstDevice, unsigned short us, size_t N );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemsetD16_v2)(dstDevice, us, N);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Initializes device memory. */

#undef cuMemsetD16Async
extern "C" CUresult cuMemsetD16Async ( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream ) {
  typedef CUresult (*cuMemsetD16Async_t)( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemsetD16Async)(dstDevice, us, N, newStream);
  RETURN_TO_UPPER_HALF();
  //logAPI(Cuda_Fnc_cuMemsetD16Async, newDevptr, us, N, newStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets device memory. */

#undef cuMemsetD2D16_v2
extern "C" CUresult cuMemsetD2D16_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height ) {
  typedef CUresult (*cuMemsetD2D16_v2_t)( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemsetD2D16_v2)(dstDevice, dstPitch, us, Width, Height);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Initializes device memory. */

#undef cuMemsetD2D16Async
extern "C" CUresult cuMemsetD2D16Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream ) {
  typedef CUresult (*cuMemsetD2D16Async_t)( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemsetD2D16Async)(dstDevice, dstPitch, us, Width, Height, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets device memory. */

#undef cuMemsetD2D32_v2
extern "C" CUresult cuMemsetD2D32_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height ) {
  typedef CUresult (*cuMemsetD2D32_v2_t)( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemsetD2D32_v2)(dstDevice, dstPitch, ui, Width, Height);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Initializes device memory. */

#undef cuMemsetD2D32Async
extern "C" CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream ) {
  typedef CUresult (*cuMemsetD2D32Async_t)( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemsetD2D32Async)(dstDevice, dstPitch, ui, Width, Height, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets device memory. */

#undef cuMemsetD2D8_v2
extern "C" CUresult cuMemsetD2D8_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height ) {
  typedef CUresult (*cuMemsetD2D8_v2_t)( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemsetD2D8_v2)(dstDevice, dstPitch, uc, Width, Height);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Initializes device memory. */

#undef cuMemsetD2D8Async
extern "C" CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream ) {
  typedef CUresult (*cuMemsetD2D8Async_t)( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemsetD2D8Async)(dstDevice, dstPitch, uc, Width, Height, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets device memory. */

#undef cuMemsetD32_v2
extern "C" CUresult cuMemsetD32_v2 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N ) {
  typedef CUresult (*cuMemsetD32_v2_t)( CUdeviceptr dstDevice, unsigned int  ui, size_t N );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemsetD32_v2)(dstDevice, ui, N);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Initializes device memory. */

#undef cuMemsetD32Async
extern "C" CUresult cuMemsetD32Async ( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream ) {
  typedef CUresult (*cuMemsetD32Async_t)( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemsetD32Async)(dstDevice, ui, N, newStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  //logAPI(Cuda_Fnc_cuMemsetD32Async, dstDevice, ui, N, newStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets device memory. */

#undef cuMemsetD8_v2
extern "C" CUresult cuMemsetD8_v2 ( CUdeviceptr dstDevice, unsigned char  uc, size_t N ) {
  typedef CUresult (*cuMemsetD8_v2_t)( CUdeviceptr dstDevice, unsigned char  uc, size_t N );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemsetD8_v2)(dstDevice, uc, N);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Initializes device memory. */

#undef cuMemsetD8Async
extern "C" CUresult cuMemsetD8Async ( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream ) {
  typedef CUresult (*cuMemsetD8Async_t)( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemsetD8Async)(dstDevice, uc, N, newStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  // logAPI(Cuda_Fnc_cuMemsetD8Async, dstDevice, uc, N, newStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets device memory. */
#undef cuMipmappedArrayCreate
extern "C" CUresult cuMipmappedArrayCreate ( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels ) {
  typedef CUresult (*cuMipmappedArrayCreate_t)( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMipmappedArrayCreate)(pHandle, pMipmappedArrayDesc, numMipmapLevels);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuMipmappedArrayCreate, pHandle, pMipmappedArrayDesc, numMipmapLevels, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Creates a CUDA mipmapped array. */

#undef cuMipmappedArrayDestroy
extern "C" CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray ) {
  typedef CUresult (*cuMipmappedArrayDestroy_t)( CUmipmappedArray hMipmappedArray );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMipmappedArrayDestroy)(hMipmappedArray);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuMipmappedArrayDestroy, hMipmappedArray, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys a CUDA mipmapped array. */

#undef cuMipmappedArrayGetLevel
extern "C" CUresult cuMipmappedArrayGetLevel ( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level ) {
  typedef CUresult (*cuMipmappedArrayGetLevel_t)( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMipmappedArrayGetLevel)(pLevelArray, hMipmappedArray, level);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets a mipmap level of a CUDA mipmapped array. */
/* ==== Unified Addressing ==== */

#undef cuMemAdvise
extern "C" CUresult cuMemAdvise ( CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device ) {
  typedef CUresult (*cuMemAdvise_t)( CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemAdvise)(devPtr, count, advice, device);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Advise about the usage of a given memory range. */

#undef cuMemPrefetchAsync
extern "C" CUresult cuMemPrefetchAsync ( CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream ) {
  typedef CUresult (*cuMemPrefetchAsync_t)( CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuMemPrefetchAsync)(devPtr, count, dstDevice, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Prefetches memory to the specified destination device. */

#undef cuMemRangeGetAttribute
extern "C" CUresult cuMemRangeGetAttribute ( void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count ) {
  typedef CUresult (*cuMemRangeGetAttribute_t)( void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  ret_val = REAL_FNC(cuMemRangeGetAttribute)(data, dataSize, attribute, devPtr, count);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Query an attribute of a given memory range. */

#undef cuMemRangeGetAttributes
extern "C" CUresult cuMemRangeGetAttributes ( void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count ) {
  typedef CUresult (*cuMemRangeGetAttributes_t)( void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuMemRangeGetAttributes)(data, dataSizes, attributes, numAttributes, devPtr, count);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Query attributes of a given memory range. */

#undef cuPointerGetAttribute
extern "C" CUresult cuPointerGetAttribute ( void* data, CUpointer_attribute attribute, CUdeviceptr ptr ) {
  typedef CUresult (*cuPointerGetAttribute_t)( void* data, CUpointer_attribute attribute, CUdeviceptr ptr );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuPointerGetAttribute)(data, attribute, ptr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns information about a pointer. */

#undef cuPointerGetAttributes
extern "C" CUresult cuPointerGetAttributes ( unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr ) {
  typedef CUresult (*cuPointerGetAttributes_t)( unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  
  LhGetMemCtx_t ctxFnc = (LhGetMemCtx_t)(lhInfo.lhGetMemCtxFptr);
  int ctxIdx = -1;
  for(unsigned int i = 0; i < numAttributes; i++)
  {
      if(attributes[i] == CU_POINTER_ATTRIBUTE_CONTEXT)
      {
        // TODO: find the context in lower-half, only for device pointer
        ctxIdx = i;
        break;
      } 
  }

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);

  ret_val = REAL_FNC(cuPointerGetAttributes)(numAttributes, attributes, data, ptr);
  // TODO: if the attributes has the mem context
  if(ctxIdx >= 0)
  {
    void* ctx = ctxFnc((void*)ptr);
    if(ctx != 0){
      memcpy(data[ctxIdx], &ctx, sizeof(CUcontext));
    }
  }
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuPointerGetAttributes, numAttributes, attributes, data, ptr, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns information about a pointer. */

#undef cuPointerSetAttribute
extern "C" CUresult cuPointerSetAttribute ( const void* value, CUpointer_attribute attribute, CUdeviceptr ptr ) {
  typedef CUresult (*cuPointerSetAttribute_t)( const void* value, CUpointer_attribute attribute, CUdeviceptr ptr );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuPointerSetAttribute)(value, attribute, ptr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Set attributes on a previously allocated memory region. */
/* ==== Stream Management ==== */

#undef cuStreamAddCallback
extern "C" CUresult cuStreamAddCallback ( CUstream hStream, CUstreamCallback callback, void* userData, unsigned int  flags ) {
  typedef CUresult (*cuStreamAddCallback_t)( CUstream hStream, CUstreamCallback callback, void* userData, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamAddCallback)(newStream, callback, userData, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Add a callback to a compute stream. */

#undef cuStreamAttachMemAsync
extern "C" CUresult cuStreamAttachMemAsync ( CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int  flags ) {
  typedef CUresult (*cuStreamAttachMemAsync_t)( CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamAttachMemAsync)(newStream, dptr, length, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Attach memory to a stream asynchronously. */
/* CUresult cuStreamBeginCapture_v2 ( CUstream hStream )*/
/* Begins graph capture on a stream. */

#undef cuStreamCreate
extern "C" CUresult cuStreamCreate ( CUstream* phStream, unsigned int  Flags ) {
  typedef CUresult (*cuStreamCreate_t)( CUstream* phStream, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuStreamCreate)(phStream, Flags);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamCreate, phStream, Flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Create a stream. */

#undef cuStreamCreateWithPriority
extern "C" CUresult cuStreamCreateWithPriority ( CUstream* phStream, unsigned int  flags, int  priority ) {
  typedef CUresult (*cuStreamCreateWithPriority_t)( CUstream* phStream, unsigned int  flags, int  priority );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuStreamCreateWithPriority)(phStream, flags, priority);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamCreateWithPriority, phStream, flags, priority, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Create a stream with the given priority. */

#undef cuStreamDestroy_v2
extern "C" CUresult cuStreamDestroy_v2 ( CUstream hStream ) {
  typedef CUresult (*cuStreamDestroy_v2_t)( CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  int freeFlag = 0;
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamDestroy_v2)(newStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamDestroy_v2, newStream, ret_val);
  lhFreeCUtype_t freeCUtype = (lhFreeCUtype_t)lhInfo.lhFreeCUtypeFptr;
  if (1 == freeFlag) {
    //Erase the record from map, the key is hStream
    freeCUtype((void*)&hStream, CU_STREAM);
  }
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys a stream. */

#undef cuStreamEndCapture
extern "C" CUresult cuStreamEndCapture ( CUstream hStream, CUgraph* phGraph ) {
  typedef CUresult (*cuStreamEndCapture_t)( CUstream hStream, CUgraph* phGraph );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamEndCapture)(newStream, phGraph);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Ends capture on a stream, returning the captured graph. */
/* CUresult cuStreamGetCaptureInfo ( CUstream hStream, CUstreamCaptureStatus* captureStatus, cuuint64_t* id ) */
/* Query capture status of a stream. */

#undef cuStreamGetCtx
extern "C" CUresult cuStreamGetCtx ( CUstream hStream, CUcontext* pctx ) {
  typedef CUresult (*cuStreamGetCtx_t)( CUstream hStream, CUcontext* pctx );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamGetCtx)(newStream, pctx);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamGetCtx, newStream, pctx, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Query the context associated with a stream. */

#undef cuStreamGetFlags
extern "C" CUresult cuStreamGetFlags ( CUstream hStream, unsigned int* flags ) {
  typedef CUresult (*cuStreamGetFlags_t)( CUstream hStream, unsigned int* flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamGetFlags)(newStream, flags);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamGetFlags, newStream, flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Query the flags of a given stream. */

#undef cuStreamGetPriority
extern "C" CUresult cuStreamGetPriority ( CUstream hStream, int* priority ) {
  typedef CUresult (*cuStreamGetPriority_t)( CUstream hStream, int* priority );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamGetPriority)(newStream, priority);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamGetPriority, newStream, priority, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Query the priority of a given stream. */

#undef cuStreamIsCapturing
extern "C" CUresult cuStreamIsCapturing ( CUstream hStream, CUstreamCaptureStatus* captureStatus ) {
  typedef CUresult (*cuStreamIsCapturing_t)( CUstream hStream, CUstreamCaptureStatus* captureStatus );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamIsCapturing)(newStream, captureStatus);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamIsCapturing, newStream, captureStatus, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a stream's capture status. */

#undef cuStreamQuery
extern "C" CUresult cuStreamQuery ( CUstream hStream ) {
  typedef CUresult (*cuStreamQuery_t)( CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamQuery)(newStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamQuery, newStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Determine status of a compute stream. */

#undef cuStreamSynchronize
extern "C" CUresult cuStreamSynchronize ( CUstream hStream ) {
  typedef CUresult (*cuStreamSynchronize_t)( CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamSynchronize)(newStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamSynchronize, newStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Wait until a stream's tasks are completed. */

#undef cuStreamWaitEvent
extern "C" CUresult cuStreamWaitEvent ( CUstream hStream, CUevent hEvent, unsigned int  Flags ) {
  typedef CUresult (*cuStreamWaitEvent_t)( CUstream hStream, CUevent hEvent, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUevent newEvent;
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newEvent, (void*)&hEvent, CU_EVENT, getCUtype);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamWaitEvent)(newStream, newEvent, Flags);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuStreamWaitEvent, newStream, newEvent, Flags);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Make a compute stream wait on an event. */
/* CUresult cuThreadExchangeStreamCaptureMode ( CUstreamCaptureMode* mode ) */
/* Swaps the stream capture interaction mode for a thread. Cuda 10.1 API */
/* ==== Event Management ==== */

#undef cuEventCreate
extern "C" CUresult cuEventCreate ( CUevent* phEvent, unsigned int  Flags ) {
  typedef CUresult (*cuEventCreate_t)( CUevent* phEvent, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuEventCreate)(phEvent, Flags);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuEventCreate, phEvent, Flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Creates an event. */

#undef cuEventDestroy_v2
extern "C" CUresult cuEventDestroy_v2 ( CUevent hEvent ) {
  typedef CUresult (*cuEventDestroy_v2_t)( CUevent hEvent );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUevent newEvent;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  int freeFlag = 0;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newEvent, (void*)&hEvent, CU_EVENT, getCUtype);
  ret_val = REAL_FNC(cuEventDestroy_v2)(newEvent);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuEventDestroy_v2, newEvent, ret_val);
  lhFreeCUtype_t freeCUtype = (lhFreeCUtype_t)lhInfo.lhFreeCUtypeFptr;
  if (1 == freeFlag) {
    //Erase the record from map, the key is hEvent
    freeCUtype((void*)&hEvent, CU_EVENT);
  }
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys an event. */

#undef cuEventElapsedTime
extern "C" CUresult cuEventElapsedTime ( float* pMilliseconds, CUevent hStart, CUevent hEnd ) {
  typedef CUresult (*cuEventElapsedTime_t)( float* pMilliseconds, CUevent hStart, CUevent hEnd );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUevent newStart;
  CUevent newEnd;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStart, (void*)&hStart, CU_EVENT, getCUtype);
  check_cu_map((void*)&newEnd, (void*)&hEnd, CU_EVENT, getCUtype);
  ret_val = REAL_FNC(cuEventElapsedTime)(pMilliseconds, newStart, newEnd);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuEventElapsedTime, pMilliseconds, newStart, newEnd, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Computes the elapsed time between two events. */

#undef cuEventQuery
extern "C" CUresult cuEventQuery ( CUevent hEvent ) {
  typedef CUresult (*cuEventQuery_t)( CUevent hEvent );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUevent newEvent; 
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newEvent, (void*)&hEvent, CU_EVENT, getCUtype);
  ret_val = REAL_FNC(cuEventQuery)(newEvent);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Queries an event's status. */

#undef cuEventRecord
extern "C" CUresult cuEventRecord ( CUevent hEvent, CUstream hStream ) {
  typedef CUresult (*cuEventRecord_t)( CUevent hEvent, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUevent newEvent;
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newEvent, (void*)&hEvent, CU_EVENT, getCUtype);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuEventRecord)(newEvent, newStream);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuEventRecord, newEvent, newStream, ret_val);

  DMTCP_PLUGIN_ENABLE_CKPT();
//   printf("oldE = %p, newE = %p, oldS = %p, newS = %p, ret = %d\n", hEvent, newEvent, hStream, newStream, ret_val);
  return ret_val;
}

/* Records an event. */

#undef cuEventSynchronize
extern "C" CUresult cuEventSynchronize ( CUevent hEvent ) {
  typedef CUresult (*cuEventSynchronize_t)( CUevent hEvent );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUevent newEvent;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newEvent, (void*)&hEvent, CU_EVENT, getCUtype);
  ret_val = REAL_FNC(cuEventSynchronize)(newEvent);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuEventSynchronize, newEvent, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
//   printf("%s, ret_val = %d\n", __FUNCTION__, ret_val);
  return ret_val;
}

/* Waits for an event to complete. */
/* ==== External Resource Interoperability ==== */

#undef cuDestroyExternalMemory
extern "C" CUresult cuDestroyExternalMemory ( CUexternalMemory extMem ) {
  typedef CUresult (*cuDestroyExternalMemory_t)( CUexternalMemory extMem );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDestroyExternalMemory)(extMem);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDestroyExternalMemory, extMem, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys an external memory object. */

#undef cuDestroyExternalSemaphore
extern "C" CUresult cuDestroyExternalSemaphore ( CUexternalSemaphore extSem ) {
  typedef CUresult (*cuDestroyExternalSemaphore_t)( CUexternalSemaphore extSem );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDestroyExternalSemaphore)(extSem);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuDestroyExternalSemaphore, extSem, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys an external semaphore. */

#undef cuExternalMemoryGetMappedBuffer
extern "C" CUresult cuExternalMemoryGetMappedBuffer ( CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc ) {
  typedef CUresult (*cuExternalMemoryGetMappedBuffer_t)( CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuExternalMemoryGetMappedBuffer)(devPtr, extMem, bufferDesc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Maps a buffer onto an imported memory object. */

#undef cuExternalMemoryGetMappedMipmappedArray
extern "C" CUresult cuExternalMemoryGetMappedMipmappedArray ( CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc ) {
  typedef CUresult (*cuExternalMemoryGetMappedMipmappedArray_t)( CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuExternalMemoryGetMappedMipmappedArray)(mipmap, extMem, mipmapDesc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Maps a CUDA mipmapped array onto an external memory object. */

#undef cuImportExternalMemory
extern "C" CUresult cuImportExternalMemory ( CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc ) {
  typedef CUresult (*cuImportExternalMemory_t)( CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuImportExternalMemory)(extMem_out, memHandleDesc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Imports an external memory object. */

#undef cuImportExternalSemaphore
extern "C" CUresult cuImportExternalSemaphore ( CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc ) {
  typedef CUresult (*cuImportExternalSemaphore_t)( CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuImportExternalSemaphore)(extSem_out, semHandleDesc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Imports an external semaphore. */

#undef cuSignalExternalSemaphoresAsync
extern "C" CUresult cuSignalExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream ) {
  typedef CUresult (*cuSignalExternalSemaphoresAsync_t)( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuSignalExternalSemaphoresAsync)(extSemArray, paramsArray, numExtSems, stream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Signals a set of external semaphore objects. */

#undef cuWaitExternalSemaphoresAsync
extern "C" CUresult cuWaitExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream ) {
  typedef CUresult (*cuWaitExternalSemaphoresAsync_t)( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&stream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuWaitExternalSemaphoresAsync)(extSemArray, paramsArray, numExtSems, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Waits on a set of external semaphore objects. */
/*  ==== Stream memory operations  ==== */

#undef cuStreamBatchMemOp
extern "C" CUresult cuStreamBatchMemOp ( CUstream stream, unsigned int  count, CUstreamBatchMemOpParams* paramArray, unsigned int  flags ) {
  typedef CUresult (*cuStreamBatchMemOp_t)( CUstream stream, unsigned int  count, CUstreamBatchMemOpParams* paramArray, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&stream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamBatchMemOp)(newStream, count, paramArray, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Batch operations to synchronize the stream via memory operations. */

#undef cuStreamWaitValue32
extern "C" CUresult cuStreamWaitValue32 ( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags ) {
  typedef CUresult (*cuStreamWaitValue32_t)( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&stream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamWaitValue32)(newStream, addr, value, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Wait on a memory location. */

#undef cuStreamWaitValue64
extern "C" CUresult cuStreamWaitValue64 ( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags ) {
  typedef CUresult (*cuStreamWaitValue64_t)( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&stream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamWaitValue64)(newStream, addr, value, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Wait on a memory location. */

#undef cuStreamWriteValue32
extern "C" CUresult cuStreamWriteValue32 ( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags ) {
  typedef CUresult (*cuStreamWriteValue32_t)( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&stream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamWriteValue32)(newStream, addr, value, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Write a value to memory. */

#undef cuStreamWriteValue64
extern "C" CUresult cuStreamWriteValue64 ( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags ) {
  typedef CUresult (*cuStreamWriteValue64_t)( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newStream, (void*)&stream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuStreamWriteValue64)(newStream, addr, value, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Write a value to memory. */
/*  ==== Execution Control  ==== */

#undef cuFuncGetAttribute
extern "C" CUresult cuFuncGetAttribute ( int* pi, CUfunction_attribute attrib, CUfunction hfunc ) {
  typedef CUresult (*cuFuncGetAttribute_t)( int* pi, CUfunction_attribute attrib, CUfunction hfunc );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuFuncGetAttribute)(pi, attrib, hfunc);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns information about a function. */

#undef cuFuncSetAttribute
extern "C" CUresult cuFuncSetAttribute ( CUfunction hfunc, CUfunction_attribute attrib, int  value ) {
  typedef CUresult (*cuFuncSetAttribute_t)( CUfunction hfunc, CUfunction_attribute attrib, int  value );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuFuncSetAttribute)(hfunc, attrib, value);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets information about a function. */

#undef cuFuncSetCacheConfig
extern "C" CUresult cuFuncSetCacheConfig ( CUfunction hfunc, CUfunc_cache config ) {
  typedef CUresult (*cuFuncSetCacheConfig_t)( CUfunction hfunc, CUfunc_cache config );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuFuncSetCacheConfig)(hfunc, config);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the preferred cache configuration for a device function. */

#undef cuFuncSetSharedMemConfig
extern "C" CUresult cuFuncSetSharedMemConfig ( CUfunction hfunc, CUsharedconfig config ) {
  typedef CUresult (*cuFuncSetSharedMemConfig_t)( CUfunction hfunc, CUsharedconfig config );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuFuncSetSharedMemConfig)(hfunc, config);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the shared memory configuration for a device function. */

#undef cuLaunchCooperativeKernel
extern "C" CUresult cuLaunchCooperativeKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams ) {
  typedef CUresult (*cuLaunchCooperativeKernel_t)( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLaunchCooperativeKernel)(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Launches a CUDA function where thread blocks can cooperate and synchronize as they execute. */

#undef cuLaunchCooperativeKernelMultiDevice
extern "C" CUresult cuLaunchCooperativeKernelMultiDevice ( CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int  numDevices, unsigned int  flags ) {
  typedef CUresult (*cuLaunchCooperativeKernelMultiDevice_t)( CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int  numDevices, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLaunchCooperativeKernelMultiDevice)(launchParamsList, numDevices, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute. */

#undef cuLaunchHostFunc
extern "C" CUresult cuLaunchHostFunc ( CUstream hStream, CUhostFn fn, void* userData ) {
  typedef CUresult (*cuLaunchHostFunc_t)( CUstream hStream, CUhostFn fn, void* userData );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLaunchHostFunc)(hStream, fn, userData);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Enqueues a host function call in a stream. */

#undef cuLaunchKernel
extern "C" CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ) {
  typedef CUresult (*cuLaunchKernel_t)( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  CUfunction newFunction;
  CUstream newStream;
  // lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  //Get new CUfuntion

  DMTCP_PLUGIN_DISABLE_CKPT();

  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newFunction, (void*)&f, CU_FUNCTION, getCUtype);
  //Get new stream
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuLaunchKernel)(newFunction, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, newStream, kernelParams, extra);
  /// Should log ????????????
  RETURN_TO_UPPER_HALF();
  //printf("%d............, ret_val2 = %d\n", __LINE__, ret_val);
  //fflush(stdout);
  DMTCP_PLUGIN_ENABLE_CKPT();
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i end.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  return ret_val;
}

/* Launches a CUDA function. */
/* ==== Execution Control [DEPRECATED] ==== */

#undef cuFuncSetBlockShape
extern "C" CUresult cuFuncSetBlockShape ( CUfunction hfunc, int  x, int  y, int  z ) {
  typedef CUresult (*cuFuncSetBlockShape_t)( CUfunction hfunc, int  x, int  y, int  z );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUfunction newFunction;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newFunction, (void*)&hfunc, CU_FUNCTION, getCUtype);
  ret_val = REAL_FNC(cuFuncSetBlockShape)(newFunction, x, y, z);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuFuncSetBlockShape, newFunction, x, y, z, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the block-dimensions for the function. */

#undef cuFuncSetSharedSize
extern "C" CUresult cuFuncSetSharedSize ( CUfunction hfunc, unsigned int  bytes ) {
  typedef CUresult (*cuFuncSetSharedSize_t)( CUfunction hfunc, unsigned int  bytes );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuFuncSetSharedSize)(hfunc, bytes);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the dynamic shared-memory size for the function. */

#undef cuLaunch
extern "C" CUresult cuLaunch ( CUfunction f ) {
  typedef CUresult (*cuLaunch_t)( CUfunction f );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLaunch)(f);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Launches a CUDA function. */

#undef cuLaunchGrid
extern "C" CUresult cuLaunchGrid ( CUfunction f, int  grid_width, int  grid_height ) {
  typedef CUresult (*cuLaunchGrid_t)( CUfunction f, int  grid_width, int  grid_height );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuLaunchGrid)(f, grid_width, grid_height);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Launches a CUDA function. */

#undef cuLaunchGridAsync
extern "C" CUresult cuLaunchGridAsync ( CUfunction f, int  grid_width, int  grid_height, CUstream hStream ) {
  typedef CUresult (*cuLaunchGridAsync_t)( CUfunction f, int  grid_width, int  grid_height, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUfunction newFunction;
  CUstream newStream;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  //Get new CUfuntion

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newFunction, (void*)&f, CU_FUNCTION, getCUtype);
  check_cu_map((void*)&newStream, (void*)&hStream, CU_STREAM, getCUtype);
  ret_val = REAL_FNC(cuLaunchGridAsync)(newFunction, grid_width, grid_height, newStream);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuLaunchGridAsync, newFunction, grid_width, grid_height, newStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Launches a CUDA function. */

#undef cuParamSetSize
extern "C" CUresult cuParamSetSize ( CUfunction hfunc, unsigned int  numbytes ) {
  typedef CUresult (*cuParamSetSize_t)( CUfunction hfunc, unsigned int  numbytes );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUfunction newFunction;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newFunction, (void*)&hfunc, CU_FUNCTION);
  ret_val = REAL_FNC(cuParamSetSize)(newFunction, numbytes);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuParamSetSize, newFunction, numbytes, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the parameter size for the function. */

#undef cuParamSetTexRef
extern "C" CUresult cuParamSetTexRef ( CUfunction hfunc, int  texunit, CUtexref hTexRef ) {
  typedef CUresult (*cuParamSetTexRef_t)( CUfunction hfunc, int  texunit, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  CUfunction newFunction;
  lhCUtype_t getCUtype = (lhCUtype_t)lhInfo.lhCUtypeFptr;
  
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  check_cu_map((void*)&newFunction, (void*)&hfunc, CU_FUNCTION, getCUtype);
  ret_val = REAL_FNC(cuParamSetTexRef)(newFunction, texunit, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Adds a texture-reference to the function's argument list. */

#undef cuParamSetf
extern "C" CUresult cuParamSetf ( CUfunction hfunc, int  offset, float  value ) {
  typedef CUresult (*cuParamSetf_t)( CUfunction hfunc, int  offset, float  value );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuParamSetf)(hfunc, offset, value);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Adds a floating-point parameter to the function's argument list. */

#undef cuParamSeti
extern "C" CUresult cuParamSeti ( CUfunction hfunc, int  offset, unsigned int  value ) {
  typedef CUresult (*cuParamSeti_t)( CUfunction hfunc, int  offset, unsigned int  value );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuParamSeti)(hfunc, offset, value);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Adds an integer parameter to the function's argument list. */

#undef cuParamSetv
extern "C" CUresult cuParamSetv ( CUfunction hfunc, int  offset, void* ptr, unsigned int  numbytes ) {
  typedef CUresult (*cuParamSetv_t)( CUfunction hfunc, int  offset, void* ptr, unsigned int  numbytes );
  CUresult ret_val = CUDA_SUCCESS;
  //This function is called when CUDA_VERSION < 4000
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuParamSetv)(hfunc, offset, ptr, numbytes);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Adds arbitrary data to the function's argument list. */
/*  ==== Graph Management  ==== */
/* CUresult cuGraphAddChildGraphNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph )*/
/* Creates a child graph node and adds it to a graph. */
/* CUresult cuGraphAddDependencies ( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t numDependencies )*/
/* Adds dependency edges to a graph. */
/* CUresult cuGraphAddEmptyNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies )*/
/* Creates an empty node and adds it to a graph. */
/* CUresult cuGraphAddHostNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams )*/
/* Creates a host execution node and adds it to a graph. */
/* CUresult cuGraphAddKernelNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams )*/
/* Creates a kernel execution node and adds it to a graph. */
/* CUresult cuGraphAddMemcpyNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx )*/
/* Creates a memcpy node and adds it to a graph. */
/* CUresult cuGraphAddMemsetNode ( CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx )*/
/* Creates a memset node and adds it to a graph. */
/* CUresult cuGraphChildGraphNodeGetGraph ( CUgraphNode hNode, CUgraph* phGraph )*/
/* Gets a handle to the embedded graph of a child graph node. */
/* CUresult cuGraphClone ( CUgraph* phGraphClone, CUgraph originalGraph )*/
/* Clones a graph. */

#undef cuGraphCreate
extern "C" CUresult cuGraphCreate ( CUgraph* phGraph, unsigned int  flags ) {
  typedef CUresult (*cuGraphCreate_t)( CUgraph* phGraph, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphCreate)(phGraph, flags);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuGraphCreate, phGraph, flags, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Creates a graph. */

#undef cuGraphDestroy
extern "C" CUresult cuGraphDestroy ( CUgraph hGraph ) {
  typedef CUresult (*cuGraphDestroy_t)( CUgraph hGraph );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphDestroy)(hGraph);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuGraphDestroy, hGraph, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys a graph. */

#undef cuGraphDestroyNode
extern "C" CUresult cuGraphDestroyNode ( CUgraphNode hNode ) {
  typedef CUresult (*cuGraphDestroyNode_t)( CUgraphNode hNode );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphDestroyNode)(hNode);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuGraphDestroyNode, hNode, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Remove a node from the graph. */

#undef cuGraphExecDestroy
extern "C" CUresult cuGraphExecDestroy ( CUgraphExec hGraphExec ) {
  typedef CUresult (*cuGraphExecDestroy_t)( CUgraphExec hGraphExec );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphExecDestroy)(hGraphExec);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuGraphExecDestroy, hGraphExec, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys an executable graph. */

#undef cuGraphGetEdges
extern "C" CUresult cuGraphGetEdges ( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges ) {
  typedef CUresult (*cuGraphGetEdges_t)( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphGetEdges)(hGraph, from, to, numEdges);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a graph's dependency edges. */

#undef cuGraphGetNodes
extern "C" CUresult cuGraphGetNodes ( CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes ) {
  typedef CUresult (*cuGraphGetNodes_t)( CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphGetNodes)(hGraph, nodes, numNodes);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a graph's nodes. */

#undef cuGraphGetRootNodes
extern "C" CUresult cuGraphGetRootNodes ( CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes ) {
  typedef CUresult (*cuGraphGetRootNodes_t)( CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphGetRootNodes)(hGraph, rootNodes, numRootNodes);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a graph's root nodes. */

#undef cuGraphHostNodeGetParams
extern "C" CUresult cuGraphHostNodeGetParams ( CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams ) {
  typedef CUresult (*cuGraphHostNodeGetParams_t)( CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphHostNodeGetParams)(hNode, nodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a host node's parameters. */

#undef cuGraphHostNodeSetParams
extern "C" CUresult cuGraphHostNodeSetParams ( CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams ) {
  typedef CUresult (*cuGraphHostNodeSetParams_t)( CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphHostNodeSetParams)(hNode, nodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets a host node's parameters. */

#undef cuGraphKernelNodeGetParams
extern "C" CUresult cuGraphKernelNodeGetParams ( CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams ) {
  typedef CUresult (*cuGraphKernelNodeGetParams_t)( CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphKernelNodeGetParams)(hNode, nodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a kernel node's parameters. */

#undef cuGraphKernelNodeSetParams
extern "C" CUresult cuGraphKernelNodeSetParams ( CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams ) {
  typedef CUresult (*cuGraphKernelNodeSetParams_t)( CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphKernelNodeSetParams)(hNode, nodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets a kernel node's parameters. */

#undef cuGraphLaunch
extern "C" CUresult cuGraphLaunch ( CUgraphExec hGraphExec, CUstream hStream ) {
  typedef CUresult (*cuGraphLaunch_t)( CUgraphExec hGraphExec, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphLaunch)(hGraphExec, hStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Launches an executable graph in a stream. */

#undef cuGraphMemcpyNodeGetParams
extern "C" CUresult cuGraphMemcpyNodeGetParams ( CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams ) {
  typedef CUresult (*cuGraphMemcpyNodeGetParams_t)( CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphMemcpyNodeGetParams)(hNode, nodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a memcpy node's parameters. */

#undef cuGraphMemcpyNodeSetParams
extern "C" CUresult cuGraphMemcpyNodeSetParams ( CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams ) {
  typedef CUresult (*cuGraphMemcpyNodeSetParams_t)( CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphMemcpyNodeSetParams)(hNode, nodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets a memcpy node's parameters. */

#undef cuGraphMemsetNodeGetParams
extern "C" CUresult cuGraphMemsetNodeGetParams ( CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams ) {
  typedef CUresult (*cuGraphMemsetNodeGetParams_t)( CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphMemsetNodeGetParams)(hNode, nodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a memset node's parameters. */

#undef cuGraphMemsetNodeSetParams
extern "C" CUresult cuGraphMemsetNodeSetParams ( CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams ) {
  typedef CUresult (*cuGraphMemsetNodeSetParams_t)( CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphMemsetNodeSetParams)(hNode, nodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets a memset node's parameters. */

#undef cuGraphNodeFindInClone
extern "C" CUresult cuGraphNodeFindInClone ( CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph ) {
  typedef CUresult (*cuGraphNodeFindInClone_t)( CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphNodeFindInClone)(phNode, hOriginalNode, hClonedGraph);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Finds a cloned version of a node. */

#undef cuGraphNodeGetDependencies
extern "C" CUresult cuGraphNodeGetDependencies ( CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies ) {
  typedef CUresult (*cuGraphNodeGetDependencies_t)( CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphNodeGetDependencies)(hNode, dependencies, numDependencies);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a node's dependencies. */

#undef cuGraphNodeGetDependentNodes
extern "C" CUresult cuGraphNodeGetDependentNodes ( CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes ) {
  typedef CUresult (*cuGraphNodeGetDependentNodes_t)( CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphNodeGetDependentNodes)(hNode, dependentNodes, numDependentNodes);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a node's dependent nodes. */

#undef cuGraphNodeGetType
extern "C" CUresult cuGraphNodeGetType ( CUgraphNode hNode, CUgraphNodeType* type ) {
  typedef CUresult (*cuGraphNodeGetType_t)( CUgraphNode hNode, CUgraphNodeType* type );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphNodeGetType)(hNode, type);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a node's type. */
/* CUresult cuGraphRemoveDependencies ( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t numDependencies )*/
/* Removes dependency edges from a graph. */
/*  ==== Occupancy ====  */

#undef cuOccupancyMaxActiveBlocksPerMultiprocessor
extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize ) {
  typedef CUresult (*cuOccupancyMaxActiveBlocksPerMultiprocessor_t)( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuOccupancyMaxActiveBlocksPerMultiprocessor)(numBlocks, func, blockSize, dynamicSMemSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns occupancy of a function. */

#undef cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags ) {
  typedef CUresult (*cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_t)( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(numBlocks, func, blockSize, dynamicSMemSize, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns occupancy of a function. */

#undef cuOccupancyMaxPotentialBlockSize
extern "C" CUresult cuOccupancyMaxPotentialBlockSize ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit ) {
  typedef CUresult (*cuOccupancyMaxPotentialBlockSize_t)( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cuOccupancyMaxPotentialBlockSize)(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Suggest a launch configuration with reasonable occupancy. */

#undef cuOccupancyMaxPotentialBlockSizeWithFlags
extern "C" CUresult cuOccupancyMaxPotentialBlockSizeWithFlags ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit, unsigned int  flags ) {
  typedef CUresult (*cuOccupancyMaxPotentialBlockSizeWithFlags_t)( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cuOccupancyMaxPotentialBlockSizeWithFlags)(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Suggest a launch configuration with reasonable occupancy. */
/* ==== Texture Reference Management [DEPRECATED] ==== */

#undef cuTexRefCreate
extern "C" CUresult cuTexRefCreate ( CUtexref* pTexRef ) {
  typedef CUresult (*cuTexRefCreate_t)( CUtexref* pTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefCreate)(pTexRef);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuTexRefCreate, pTexRef, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Creates a texture reference. */

#undef cuTexRefDestroy
extern "C" CUresult cuTexRefDestroy ( CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefDestroy_t)( CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cuTexRefDestroy)(hTexRef);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuTexRefDestroy, hTexRef, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys a texture reference. */

#undef cuTexRefGetAddress_v2
extern "C" CUresult cuTexRefGetAddress_v2 ( CUdeviceptr* pdptr, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetAddress_v2_t)( CUdeviceptr* pdptr, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefGetAddress_v2)(pdptr, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the address associated with a texture reference. */

#undef cuTexRefGetAddressMode
extern "C" CUresult cuTexRefGetAddressMode ( CUaddress_mode* pam, CUtexref hTexRef, int  dim ) {
  typedef CUresult (*cuTexRefGetAddressMode_t)( CUaddress_mode* pam, CUtexref hTexRef, int  dim );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefGetAddressMode)(pam, hTexRef, dim);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the addressing mode used by a texture reference. */

#undef cuTexRefGetArray
extern "C" CUresult cuTexRefGetArray ( CUarray* phArray, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetArray_t)( CUarray* phArray, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefGetArray)(phArray, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the array bound to a texture reference. */

#undef cuTexRefGetBorderColor
extern "C" CUresult cuTexRefGetBorderColor ( float* pBorderColor, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetBorderColor_t)( float* pBorderColor, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefGetBorderColor)(pBorderColor, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the border color used by a texture reference. */

#undef cuTexRefGetFilterMode
extern "C" CUresult cuTexRefGetFilterMode ( CUfilter_mode* pfm, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetFilterMode_t)( CUfilter_mode* pfm, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cuTexRefGetFilterMode)(pfm, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the filter-mode used by a texture reference. */

#undef cuTexRefGetFlags
extern "C" CUresult cuTexRefGetFlags ( unsigned int* pFlags, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetFlags_t)( unsigned int* pFlags, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefGetFlags)(pFlags, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the flags used by a texture reference. */

#undef cuTexRefGetFormat
extern "C" CUresult cuTexRefGetFormat ( CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetFormat_t)( CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cuTexRefGetFormat)(pFormat, pNumChannels, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the format used by a texture reference. */

#undef cuTexRefGetMaxAnisotropy
extern "C" CUresult cuTexRefGetMaxAnisotropy ( int* pmaxAniso, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetMaxAnisotropy_t)( int* pmaxAniso, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cuTexRefGetMaxAnisotropy)(pmaxAniso, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the maximum anisotropy for a texture reference. */

#undef cuTexRefGetMipmapFilterMode
extern "C" CUresult cuTexRefGetMipmapFilterMode ( CUfilter_mode* pfm, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetMipmapFilterMode_t)( CUfilter_mode* pfm, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefGetMipmapFilterMode)(pfm, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the mipmap filtering mode for a texture reference. */

#undef cuTexRefGetMipmapLevelBias
extern "C" CUresult cuTexRefGetMipmapLevelBias ( float* pbias, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetMipmapLevelBias_t)( float* pbias, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefGetMipmapLevelBias)(pbias, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the mipmap level bias for a texture reference. */

#undef cuTexRefGetMipmapLevelClamp
extern "C" CUresult cuTexRefGetMipmapLevelClamp ( float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetMipmapLevelClamp_t)( float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefGetMipmapLevelClamp)(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the min/max mipmap level clamps for a texture reference. */

#undef cuTexRefGetMipmappedArray
extern "C" CUresult cuTexRefGetMipmappedArray ( CUmipmappedArray* phMipmappedArray, CUtexref hTexRef ) {
  typedef CUresult (*cuTexRefGetMipmappedArray_t)( CUmipmappedArray* phMipmappedArray, CUtexref hTexRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefGetMipmappedArray)(phMipmappedArray, hTexRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Gets the mipmapped array bound to a texture reference. */

#undef cuTexRefSetAddress_v2
extern "C" CUresult cuTexRefSetAddress_v2 ( size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes ) {
  typedef CUresult (*cuTexRefSetAddress_v2_t)( size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes );
  CUresult ret_val = CUDA_SUCCESS;
  //This function is called when CUDA_VERSION < 11000
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetAddress_v2)(ByteOffset, hTexRef, dptr, bytes);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cuTexRefSetAddress_v2, ByteOffset, hTexRef, dptr, bytes, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Binds an address as a texture reference. */

#undef cuTexRefSetAddress2D_v3
extern "C" CUresult cuTexRefSetAddress2D_v3 ( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch ) {
  typedef CUresult (*cuTexRefSetAddress2D_v3_t)( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetAddress2D_v3)(hTexRef, desc, dptr, Pitch);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Binds an address as a 2D texture reference. */

#undef cuTexRefSetAddressMode
extern "C" CUresult cuTexRefSetAddressMode ( CUtexref hTexRef, int  dim, CUaddress_mode am ) {
  typedef CUresult (*cuTexRefSetAddressMode_t)( CUtexref hTexRef, int  dim, CUaddress_mode am );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetAddressMode)(hTexRef, dim, am);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the addressing mode for a texture reference. */

#undef cuTexRefSetArray
extern "C" CUresult cuTexRefSetArray ( CUtexref hTexRef, CUarray hArray, unsigned int  Flags ) {
  typedef CUresult (*cuTexRefSetArray_t)( CUtexref hTexRef, CUarray hArray, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetArray)(hTexRef, hArray, Flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Binds an array as a texture reference. */

#undef cuTexRefSetBorderColor
extern "C" CUresult cuTexRefSetBorderColor ( CUtexref hTexRef, float* pBorderColor ) {
  typedef CUresult (*cuTexRefSetBorderColor_t)( CUtexref hTexRef, float* pBorderColor );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetBorderColor)(hTexRef, pBorderColor);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the border color for a texture reference. */

#undef cuTexRefSetFilterMode
extern "C" CUresult cuTexRefSetFilterMode ( CUtexref hTexRef, CUfilter_mode fm ) {
  typedef CUresult (*cuTexRefSetFilterMode_t)( CUtexref hTexRef, CUfilter_mode fm );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetFilterMode)(hTexRef, fm);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the filtering mode for a texture reference. */

#undef cuTexRefSetFlags
extern "C" CUresult cuTexRefSetFlags ( CUtexref hTexRef, unsigned int  Flags ) {
  typedef CUresult (*cuTexRefSetFlags_t)( CUtexref hTexRef, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetFlags)(hTexRef, Flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the flags for a texture reference. */

#undef cuTexRefSetFormat
extern "C" CUresult cuTexRefSetFormat ( CUtexref hTexRef, CUarray_format fmt, int  NumPackedComponents ) {
  typedef CUresult (*cuTexRefSetFormat_t)( CUtexref hTexRef, CUarray_format fmt, int  NumPackedComponents );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //This function is called when CUDA_VERSION < 11000
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetFormat)(hTexRef, fmt, NumPackedComponents);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuTexRefSetFormat, hTexRef, fmt, NumPackedComponents, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the format for a texture reference. */

#undef cuTexRefSetMaxAnisotropy
extern "C" CUresult cuTexRefSetMaxAnisotropy ( CUtexref hTexRef, unsigned int  maxAniso ) {
  typedef CUresult (*cuTexRefSetMaxAnisotropy_t)( CUtexref hTexRef, unsigned int  maxAniso );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cuTexRefSetMaxAnisotropy)(hTexRef, maxAniso);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the maximum anisotropy for a texture reference. */

#undef cuTexRefSetMipmapFilterMode
extern "C" CUresult cuTexRefSetMipmapFilterMode ( CUtexref hTexRef, CUfilter_mode fm ) {
  typedef CUresult (*cuTexRefSetMipmapFilterMode_t)( CUtexref hTexRef, CUfilter_mode fm );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetMipmapFilterMode)(hTexRef, fm);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the mipmap filtering mode for a texture reference. */

#undef cuTexRefSetMipmapLevelBias
extern "C" CUresult cuTexRefSetMipmapLevelBias ( CUtexref hTexRef, float  bias ) {
  typedef CUresult (*cuTexRefSetMipmapLevelBias_t)( CUtexref hTexRef, float  bias );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetMipmapLevelBias)(hTexRef, bias);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the mipmap level bias for a texture reference. */

#undef cuTexRefSetMipmapLevelClamp
extern "C" CUresult cuTexRefSetMipmapLevelClamp ( CUtexref hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp ) {
  typedef CUresult (*cuTexRefSetMipmapLevelClamp_t)( CUtexref hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetMipmapLevelClamp)(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the mipmap min/max mipmap level clamps for a texture reference. */

#undef cuTexRefSetMipmappedArray
extern "C" CUresult cuTexRefSetMipmappedArray ( CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int  Flags ) {
  typedef CUresult (*cuTexRefSetMipmappedArray_t)( CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexRefSetMipmappedArray)(hTexRef, hMipmappedArray, Flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Binds a mipmapped array to a texture reference. */
/* ==== Surface Reference Management [DEPRECATED] ==== */

#undef cuSurfRefGetArray
extern "C" CUresult cuSurfRefGetArray ( CUarray* phArray, CUsurfref hSurfRef ) {
  typedef CUresult (*cuSurfRefGetArray_t)( CUarray* phArray, CUsurfref hSurfRef );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuSurfRefGetArray)(phArray, hSurfRef);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Passes back the CUDA array bound to a surface reference. */

#undef cuSurfRefSetArray
extern "C" CUresult cuSurfRefSetArray ( CUsurfref hSurfRef, CUarray hArray, unsigned int  Flags ) {
  typedef CUresult (*cuSurfRefSetArray_t)( CUsurfref hSurfRef, CUarray hArray, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuSurfRefSetArray)(hSurfRef, hArray, Flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Sets the CUDA array for a surface reference. */
/* ==== Texture Object Management ==== */

#undef cuTexObjectCreate
extern "C" CUresult cuTexObjectCreate ( CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc ) {
  typedef CUresult (*cuTexObjectCreate_t)( CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexObjectCreate)(pTexObject, pResDesc, pTexDesc, pResViewDesc);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuTexObjectCreate, pTexObject, pResDesc, pTexDesc, pResViewDesc, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Creates a texture object. */

#undef cuTexObjectDestroy
extern "C" CUresult cuTexObjectDestroy ( CUtexObject texObject ) {
  typedef CUresult (*cuTexObjectDestroy_t)( CUtexObject texObject );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexObjectDestroy)(texObject);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuTexObjectDestroy, texObject, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys a texture object. */

#undef cuTexObjectGetResourceDesc
extern "C" CUresult cuTexObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject ) {
  typedef CUresult (*cuTexObjectGetResourceDesc_t)( CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexObjectGetResourceDesc)(pResDesc, texObject);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a texture object's resource descriptor. */

#undef cuTexObjectGetResourceViewDesc
extern "C" CUresult cuTexObjectGetResourceViewDesc ( CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject ) {
  typedef CUresult (*cuTexObjectGetResourceViewDesc_t)( CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexObjectGetResourceViewDesc)(pResViewDesc, texObject);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a texture object's resource view descriptor. */

#undef cuTexObjectGetTextureDesc
extern "C" CUresult cuTexObjectGetTextureDesc ( CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject ) {
  typedef CUresult (*cuTexObjectGetTextureDesc_t)( CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuTexObjectGetTextureDesc)(pTexDesc, texObject);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a texture object's texture descriptor. */
/* ==== Surface Object Management ==== */

#undef cuSurfObjectCreate
extern "C" CUresult cuSurfObjectCreate ( CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc ) {
  typedef CUresult (*cuSurfObjectCreate_t)( CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuSurfObjectCreate)(pSurfObject, pResDesc);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuSurfObjectCreate, pSurfObject, pResDesc, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Creates a surface object. */

#undef cuSurfObjectDestroy
extern "C" CUresult cuSurfObjectDestroy ( CUsurfObject surfObject ) {
  typedef CUresult (*cuSurfObjectDestroy_t)( CUsurfObject surfObject );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuSurfObjectDestroy)(surfObject);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cuSurfObjectDestroy, surfObject, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Destroys a surface object. */

#undef cuSurfObjectGetResourceDesc
extern "C" CUresult cuSurfObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject ) {
  typedef CUresult (*cuSurfObjectGetResourceDesc_t)( CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuSurfObjectGetResourceDesc)(pResDesc, surfObject);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Returns a surface object's resource descriptor. */
/* ==== Peer Context Memory Access  ==== */

#undef cuCtxDisablePeerAccess
extern "C" CUresult cuCtxDisablePeerAccess ( CUcontext peerContext ) {
  typedef CUresult (*cuCtxDisablePeerAccess_t)( CUcontext peerContext );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxDisablePeerAccess)(peerContext);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Disables direct access to memory allocations in a peer context and unregisters any registered allocations. */

#undef cuCtxEnablePeerAccess
extern "C" CUresult cuCtxEnablePeerAccess ( CUcontext peerContext, unsigned int  Flags ) {
  typedef CUresult (*cuCtxEnablePeerAccess_t)( CUcontext peerContext, unsigned int  Flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuCtxEnablePeerAccess)(peerContext, Flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Enables direct access to memory allocations in a peer context. */

#undef cuDeviceCanAccessPeer
extern "C" CUresult cuDeviceCanAccessPeer ( int* canAccessPeer, CUdevice dev, CUdevice peerDev ) {
  typedef CUresult (*cuDeviceCanAccessPeer_t)( int* canAccessPeer, CUdevice dev, CUdevice peerDev );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceCanAccessPeer)(canAccessPeer, dev, peerDev);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Queries if a device may directly access a peer device's memory. */

#undef cuDeviceGetP2PAttribute
extern "C" CUresult cuDeviceGetP2PAttribute ( int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice ) {
  typedef CUresult (*cuDeviceGetP2PAttribute_t)( int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuDeviceGetP2PAttribute)(value, attrib, srcDevice, dstDevice);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Queries attributes of the link between two devices. */
/* ==== Graphics Interoperability ==== */

#undef cuGraphicsMapResources
extern "C" CUresult cuGraphicsMapResources ( unsigned int  count, CUgraphicsResource* resources, CUstream hStream ) {
  typedef CUresult (*cuGraphicsMapResources_t)( unsigned int  count, CUgraphicsResource* resources, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphicsMapResources)(count, resources, hStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Map graphics resources for access by CUDA. */

#undef cuGraphicsResourceGetMappedMipmappedArray
extern "C" CUresult cuGraphicsResourceGetMappedMipmappedArray ( CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource ) {
  typedef CUresult (*cuGraphicsResourceGetMappedMipmappedArray_t)( CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphicsResourceGetMappedMipmappedArray)(pMipmappedArray, resource);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Get a mipmapped array through which to access a mapped graphics resource. */

#undef cuGraphicsResourceGetMappedPointer_v2
extern "C" CUresult cuGraphicsResourceGetMappedPointer_v2 ( CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource ) {
  typedef CUresult (*cuGraphicsResourceGetMappedPointer_v2_t)( CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphicsResourceGetMappedPointer_v2)(pDevPtr, pSize, resource);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Get a device pointer through which to access a mapped graphics resource. */

#undef cuGraphicsResourceSetMapFlags_v2
extern "C" CUresult cuGraphicsResourceSetMapFlags_v2 ( CUgraphicsResource resource, unsigned int  flags ) {
  typedef CUresult (*cuGraphicsResourceSetMapFlags_v2_t)( CUgraphicsResource resource, unsigned int  flags );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphicsResourceSetMapFlags_v2)(resource, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Set usage flags for mapping a graphics resource. */

#undef cuGraphicsSubResourceGetMappedArray
extern "C" CUresult cuGraphicsSubResourceGetMappedArray ( CUarray* pArray, CUgraphicsResource resource, unsigned int  arrayIndex, unsigned int  mipLevel ) {
  typedef CUresult (*cuGraphicsSubResourceGetMappedArray_t)( CUarray* pArray, CUgraphicsResource resource, unsigned int  arrayIndex, unsigned int  mipLevel );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphicsSubResourceGetMappedArray)(pArray, resource, arrayIndex, mipLevel);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Get an array through which to access a subresource of a mapped graphics resource. */

#undef cuGraphicsUnmapResources
extern "C" CUresult cuGraphicsUnmapResources ( unsigned int  count, CUgraphicsResource* resources, CUstream hStream ) {
  typedef CUresult (*cuGraphicsUnmapResources_t)( unsigned int  count, CUgraphicsResource* resources, CUstream hStream );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphicsUnmapResources)(count, resources, hStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Unmap graphics resources. */

#undef cuGraphicsUnregisterResource
extern "C" CUresult cuGraphicsUnregisterResource ( CUgraphicsResource resource ) {
  typedef CUresult (*cuGraphicsUnregisterResource_t)( CUgraphicsResource resource );
  CUresult ret_val = CUDA_SUCCESS;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGraphicsUnregisterResource)(resource);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}

/* Unregisters a graphics resource for access by CUDA. */
#undef __cudaRegisterFatBinaryEnd
extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  typedef void (*__cudaRegisterFatBinaryEnd_t)(void **fatCubinHandle);
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  if(!initialized)
    initialize_wrappers();
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  REAL_FNC(__cudaRegisterFatBinaryEnd)(fatCubinHandle);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc___cudaRegisterFatBinaryEnd, fatCubinHandle);
  // printf("cudaCallsLog size:%ld\n", cudaCallsLog.size());
  DMTCP_PLUGIN_ENABLE_CKPT();
}

/* ==== FFT Management ==== */

#undef cufftPlan1d
extern "C" cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch) {
  typedef cufftResult (*cufftPlan1d_t)(cufftHandle *plan, int nx, cufftType type, int batch);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftPlan1d)(plan, nx, type, batch);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftPlan2d
extern "C" cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type) {
  typedef cufftResult (*cufftPlan2d_t)(cufftHandle *plan, int nx, int ny, cufftType type);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftPlan2d)(plan, nx, ny, type);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftPlan3d
extern "C" cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type) {
  typedef cufftResult (*cufftPlan3d_t)(cufftHandle *plan, int nx, int ny, int nz, cufftType type);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftPlan3d)(plan, nx, ny, nz, type);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cufftPlan3d, plan, nx, ny, nz, type, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftPlanMany
extern "C" cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch) {
  typedef cufftResult (*cufftPlanMany_t)(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftPlanMany)(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cufftPlanMany, plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftMakePlan1d
extern "C" cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t *workSize) {
  typedef cufftResult (*cufftMakePlan1d_t)(cufftHandle plan, int nx, cufftType type, int batch, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftMakePlan1d)(plan, nx, type, batch, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftMakePlan2d
extern "C" cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t *workSize) {
  typedef cufftResult (*cufftMakePlan2d_t)(cufftHandle plan, int nx, int ny, cufftType type, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftMakePlan2d)(plan, nx, ny, type, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftMakePlan3d
extern "C" cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t *workSize) {
  typedef cufftResult (*cufftMakePlan3d_t)(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftMakePlan3d)(plan, nx, ny, nz, type, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftMakePlanMany
extern "C" cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workSize) {
  typedef cufftResult (*cufftMakePlanMany_t)(cufftHandle plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftMakePlanMany)(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftMakePlanMany64
extern "C" cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t * workSize) {
  typedef cufftResult (*cufftMakePlanMany64_t)(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t * workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftMakePlanMany64)(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftGetSizeMany64
extern "C" cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t *workSize) {
  typedef cufftResult (*cufftGetSizeMany64_t)(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftGetSizeMany64)(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftEstimate1d
extern "C" cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t *workSize) {
  typedef cufftResult (*cufftEstimate1d_t)(int nx, cufftType type, int batch, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftEstimate1d)(nx, type, batch, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftEstimate2d
extern "C" cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t *workSize) {
  typedef cufftResult (*cufftEstimate2d_t)(int nx, int ny, cufftType type, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftEstimate2d)(nx, ny, type, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftEstimate3d
extern "C" cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t *workSize) {
  typedef cufftResult (*cufftEstimate3d_t)(int nx, int ny, int nz, cufftType type, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftEstimate3d)(nx, ny, nz, type, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftEstimateMany
extern "C" cufftResult cufftEstimateMany(int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workSize) {
  typedef cufftResult (*cufftEstimateMany_t)(int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftEstimateMany)(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftCreate
extern "C" cufftResult cufftCreate(cufftHandle * handle) {
  typedef cufftResult (*cufftCreate_t)(cufftHandle * handle);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftCreate)(handle);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftGetSize1d
extern "C" cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t *workSize ) {
  typedef cufftResult (*cufftGetSize1d_t)(cufftHandle handle, int nx, cufftType type, int batch, size_t *workSize );
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftGetSize1d)(handle, nx, type, batch, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftGetSize2d
extern "C" cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t *workSize) {
  typedef cufftResult (*cufftGetSize2d_t)(cufftHandle handle, int nx, int ny, cufftType type, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftGetSize2d)(handle, nx, ny, type, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftGetSize3d
extern "C" cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t *workSize) {
  typedef cufftResult (*cufftGetSize3d_t)(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftGetSize3d)(handle, nx, ny, nz, type, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftGetSizeMany
extern "C" cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workArea) {
  typedef cufftResult (*cufftGetSizeMany_t)(cufftHandle handle, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workArea);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftGetSizeMany)(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftGetSize
extern "C" cufftResult cufftGetSize(cufftHandle handle, size_t *workSize) {
  typedef cufftResult (*cufftGetSize_t)(cufftHandle handle, size_t *workSize);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftGetSize)(handle, workSize);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftSetWorkArea
extern "C" cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea) {
  typedef cufftResult (*cufftSetWorkArea_t)(cufftHandle plan, void *workArea);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftSetWorkArea)(plan, workArea);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftSetAutoAllocation
extern "C" cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) {
  typedef cufftResult (*cufftSetAutoAllocation_t)(cufftHandle plan, int autoAllocate);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftSetAutoAllocation)(plan, autoAllocate);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftExecC2C
extern "C" cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction) {
  typedef cufftResult (*cufftExecC2C_t)(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftExecC2C)(plan, idata, odata, direction);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftExecR2C
extern "C" cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata) {
  typedef cufftResult (*cufftExecR2C_t)(cufftHandle plan, cufftReal *idata, cufftComplex *odata);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftExecR2C)(plan, idata, odata);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftExecC2R
extern "C" cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata) {
  typedef cufftResult (*cufftExecC2R_t)(cufftHandle plan, cufftComplex *idata, cufftReal *odata);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftExecC2R)(plan, idata, odata);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftExecZ2Z
extern "C" cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction) {
  typedef cufftResult (*cufftExecZ2Z_t)(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftExecZ2Z)(plan, idata, odata, direction);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftExecD2Z
extern "C" cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata) {
  typedef cufftResult (*cufftExecD2Z_t)(cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftExecD2Z)(plan, idata, odata);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftExecZ2D
extern "C" cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleReal *odata) {
  typedef cufftResult (*cufftExecZ2D_t)(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleReal *odata);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftExecZ2D)(plan, idata, odata);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftSetStream
extern "C" cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) {
  typedef cufftResult (*cufftSetStream_t)(cufftHandle plan, cudaStream_t stream);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  cudaStream_t newStream;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  ret_val = REAL_FNC(cufftSetStream)(plan, newStream);
  RETURN_TO_UPPER_HALF();
  /* Insert logging code here */
  logAPI(Cuda_Fnc_cufftSetStream, plan, newStream, ret_val);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftDestroy
extern "C" cufftResult cufftDestroy(cufftHandle plan) {
  typedef cufftResult (*cufftDestroy_t)(cufftHandle plan);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(cufftDestroy)(plan);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftGetVersion
extern "C" cufftResult cufftGetVersion(int *version) {
  typedef cufftResult (*cufftGetVersion_t)(int *version);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftGetVersion)(version);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


#undef cufftGetProperty
extern "C" cufftResult cufftGetProperty(libraryPropertyType type, int *value) {
  typedef cufftResult (*cufftGetProperty_t)(libraryPropertyType type, int *value);
  cufftResult ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cufftGetProperty)(type, value);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}


// For NCCL 

#undef cudaStreamGetCaptureInfo_v2
extern "C" cudaError_t cudaStreamGetCaptureInfo_v2( cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out , const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) {
  typedef cudaError_t (*cudaStreamGetCaptureInfo_v2_t)( cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out , const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out);
  cudaError_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  cudaStream_t newStream;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  ret_val = REAL_FNC(cudaStreamGetCaptureInfo_v2)(newStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}


#undef cudaUserObjectCreate
extern "C" cudaError_t cudaUserObjectCreate( cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int  initialRefcount, unsigned int  flags ){
  typedef cudaError_t (*cudaUserObjectCreate_t)( cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int  initialRefcount, unsigned int  flags );
  cudaError_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  ret_val = REAL_FNC(cudaUserObjectCreate)(object_out, ptr, destroy, initialRefcount, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaGraphRetainUserObject
extern "C" cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int  count, unsigned int  flags){
  typedef cudaError_t (*cudaGraphRetainUserObject_t)(cudaGraph_t graph, cudaUserObject_t object, unsigned int  count, unsigned int  flags);
  cudaError_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  ret_val = REAL_FNC(cudaGraphRetainUserObject)(graph, object, count, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaStreamUpdateCaptureDependencies
extern "C" cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int  flags){
  typedef cudaError_t (*cudaStreamUpdateCaptureDependencies_t)(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int  flags);
  cudaError_t ret_val ;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  ret_val = REAL_FNC(cudaStreamUpdateCaptureDependencies)(stream, dependencies, numDependencies, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaGetDriverEntryPoint
extern "C" cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags ){
  typedef cudaError_t (*cudaGetDriverEntryPoint_t)(const char* symbol, void** funcPtr, unsigned long long flags );
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  ret_val = REAL_FNC(cudaGetDriverEntryPoint)(symbol, funcPtr, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaFuncSetAttribute
extern "C" cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int  value){
  typedef cudaError_t (*cudaFuncSetAttribute_t)(const void* func, cudaFuncAttribute attr, int value);
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  ret_val = REAL_FNC(cudaFuncSetAttribute)(func, attr, value);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaIpcGetMemHandle
extern "C" cudaError_t cudaIpcGetMemHandle( cudaIpcMemHandle_t* handle, void* devPtr )
{
  // typedef cudaError_t (*cudaIpcGetMemHandle_t)(cudaIpcMemHandle_t* handle, void* devPtr);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  LhExecApi_t fnc = (LhExecApi_t)(lhInfo.lhExecApiFptr);
  
  LhIpcGetApiEx_t ipcGetFnc = (LhIpcGetApiEx_t)(lhInfo.lhIpcGetApiEx);

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  ret_val = (cudaError_t)ipcGetFnc(handle, devPtr);
  RETURN_TO_UPPER_HALF();
  logAPI(Cuda_Fnc_cudaIpcGetMemHandle, handle, devPtr);
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaIpcOpenMemHandle
extern "C" cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags){
  // typedef cudaError_t (*cudaIpcOpenMemHandle_t)(void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);

  // TODO: need to get the mapped block size and the device belongs to, by tian01.liu 2023.8.14
  LhIpcOpenApiEx_t ipcOpenFnc = (LhIpcOpenApiEx_t)(lhInfo.lhIpcOpenApiEx);
  size_t size = 0;
  unsigned int devId = 0;
  LhGetIpcInfo_t ipcFnc = (LhGetIpcInfo_t)(lhInfo.lhGetIpcInfoFptr);

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread); 
  ret_val = (cudaError_t)ipcOpenFnc(devPtr, &handle, flags);

  // by tian01.liu for cuda ipc
  if(ipcFnc)
    ipcFnc(*devPtr, &size, &devId);
  
  RETURN_TO_UPPER_HALF();

  // by tian01.liu for cuda ipc
  logAPI(Cuda_Fnc_cudaIpcOpenMemHandle, *devPtr, handle, flags, size, devId);   

  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}

#undef cudaIpcCloseMemHandle
extern "C" cudaError_t cudaIpcCloseMemHandle(void* devPtr){
  // typedef cudaError_t (*cudaIpcCloseMemHandle_t)(void* devPtr);
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  LhExecApi_t fnc = (LhExecApi_t)(lhInfo.lhExecApiFptr);
  
  LhIpcCloseApiEx_t ipcCloseFnc = (LhIpcCloseApiEx_t)(lhInfo.lhIpcCloseApiEx);

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = (cudaError_t)ipcCloseFnc(devPtr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaDriverGetVersion
extern "C" cudaError_t cudaDriverGetVersion(int* driverVersion){
  typedef cudaError_t (*cudaDriverGetVersion_t)(void* devPtr);
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaDriverGetVersion)(driverVersion);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaDeviceGetByPCIBusId
extern "C" cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId){
  typedef cudaError_t (*cudaDeviceGetByPCIBusId_t)(int* device, const char* pciBusId);
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaDeviceGetByPCIBusId)(device, pciBusId); 
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaThreadExchangeStreamCaptureMode
extern "C" cudaError_t cudaThreadExchangeStreamCaptureMode ( cudaStreamCaptureMode * mode )
{
  typedef cudaError_t (*cudaThreadExchangeStreamCaptureMode_t)(cudaStreamCaptureMode * mode);
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaThreadExchangeStreamCaptureMode)(mode);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}

#undef cudaHostGetDevicePointer
extern "C" cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int  flags) {
  typedef cudaError_t (*cudaHostGetDevicePointer_t)(void** pDevice, void* pHost, unsigned int  flags) ;
  cudaError_t ret_val = cudaSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaHostGetDevicePointer)(pDevice, pHost, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaHostUnregister
extern "C" cudaError_t cudaHostUnregister(void* ptr) {
  typedef cudaError_t (*cudaHostUnregister_t)(void* ptr) ;
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaHostUnregister)(ptr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaGraphAddEventWaitNode
extern "C" cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) {
  typedef cudaError_t (*cudaGraphAddEventWaitNode_t)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) ;
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaGraphAddEventWaitNode)(pGraphNode, graph, pDependencies, numDependencies, event);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaGraphAddEventRecordNode
extern "C" cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) {
  typedef cudaError_t (*cudaGraphAddEventRecordNode_t)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) ;
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaGraphAddEventRecordNode)(pGraphNode, graph, pDependencies, numDependencies, event);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaLaunchHostFunc
extern "C" cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
  typedef cudaError_t (*cudaLaunchHostFunc_t)(cudaStream_t stream, cudaHostFn_t fn, void* userData) ;
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  // cudaStream_t newStream;
  // check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaLaunchHostFunc)(stream, fn, userData);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaGraphAddHostNode
extern "C" cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) {
  typedef cudaError_t (*cudaGraphAddHostNode_t)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams);
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaGraphAddHostNode)(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}

#undef cudaDeviceEnablePeerAccess
extern "C" cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)  {
  typedef cudaError_t (*cudaDeviceEnablePeerAccess_t)(int peerDevice, unsigned int flags);
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaDeviceEnablePeerAccess)(peerDevice, flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  
  return ret_val;
}


#undef cudaGraphAddKernelNode
extern "C" cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams)  {
  typedef cudaError_t (*cudaGraphAddKernelNode_t)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams);
  cudaError_t ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cudaGraphAddKernelNode)(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
 
  return ret_val;
}


#undef cuGetErrorName
extern "C" CUresult cuGetErrorName(CUresult error, const char** pStr)  {
  typedef CUresult (*cuGetErrorName_t)(CUresult error, const char** pStr);
  CUresult ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  //fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuGetErrorName)(error, pStr);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
#undef cuIpcOpenMemHandle_v2
extern "C" CUresult cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags)
{
  typedef CUresult (*cuIpcOpenMemHandle_v2_t)(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags);
  CUresult ret_val;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%i, tid:%i.\n", __FUNCTION__, __LINE__ , pid, tid);fflush(stdout);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(cuIpcOpenMemHandle_v2)(pdptr, handle, Flags);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return ret_val;
}
// For nccl we need this MACRO to contrl 
#define NCCL_DISABLE_CKPT() \
  if (nccl_wrapper_count == 0) dmtcp_plugin_disable_ckptthread(); \
  DMTCP_PLUGIN_DISABLE_CKPT(); \
  nccl_wrapper_count++; \

#define NCCL_ENABLE_CKPT() \
  nccl_wrapper_count--; \
  if (nccl_wrapper_count == 0)  { \
  DMTCP_PLUGIN_ENABLE_CKPT(); \
  dmtcp_plugin_enable_ckptthread();} \

void *libNcclHandle = nullptr;
// NCCL wrapper
#undef ncclCommInitAll
extern "C" ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  typedef ncclResult_t (*ncclCommInitAll_t)(ncclComm_t* comms, int ndev, const int* devlist);
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );
  //fflush(stdout);
  ncclResult_t ret_val = ncclSuccess;
  
#ifdef NCCL_MIGRATION
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(ncclCommInitAll)(comms, ndev, devlist);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclCommInitAll, comms, ndev, devlist, ret_val);
#else
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclCommInitAll_t func = (ncclCommInitAll_t)dlsym(libNcclHandle, "ncclCommInitAll");
  ret_val = func(comms, ndev, devlist);
  NCCL_ENABLE_CKPT();
#endif
  
  return ret_val;
}

#undef ncclGroupEnd
extern "C" ncclResult_t  ncclGroupEnd()
{
  typedef ncclResult_t (*ncclGroupEnd_t)();
  // typedef cudaError_t (*cudaDeviceSynchronize_t)();
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );
  ncclResult_t ret_val = ncclSuccess;
  
#ifdef NCCL_MIGRATION
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(ncclGroupEnd)();
  
  // REAL_FNC(cudaDeviceSynchronize)();

  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  logAPI(Cuda_Fnc_ncclGroupEnd, ret_val);
#else
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclGroupEnd_t func = (ncclGroupEnd_t)dlsym(libNcclHandle, "ncclGroupEnd");
  ret_val = func();
  NCCL_ENABLE_CKPT();
#endif
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d end.\n", __FUNCTION__, __LINE__, tid, pid );
  return ret_val;
}

#undef ncclGroupStart
extern "C" ncclResult_t  ncclGroupStart() 
{
  typedef ncclResult_t (*ncclGroupStart_t)();
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );
  
#ifdef NCCL_MIGRATION
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(ncclGroupStart)();
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclGroupStart, ret_val);
#else
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclGroupStart_t func = (ncclGroupStart_t)dlsym(libNcclHandle, "ncclGroupStart");
  ret_val = func();
  NCCL_ENABLE_CKPT();
#endif

  return ret_val;
}

#undef ncclAllGather
extern "C" ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) 
{
  typedef ncclResult_t (*ncclAllGather_t)(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION  
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclAllGather_t func = (ncclAllGather_t)dlsym(libNcclHandle, "ncclAllGather");
  ret_val = func(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  NCCL_ENABLE_CKPT();
#else
  cudaStream_t newStream;
  ncclComm_t newComm;


  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
  ret_val = REAL_FNC(ncclAllGather)(sendbuff, recvbuff, sendcount, datatype, newComm, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  logAPI(Cuda_Fnc_ncclAllGather, sendbuff, recvbuff, sendcount, datatype, newComm, newStream, ret_val);
#endif

  return ret_val;
}

#undef ncclAllReduce
extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  typedef ncclResult_t (*ncclAllReduce_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, pid:%d, tid:%d.\n", __FUNCTION__, __LINE__, pid, tid );

#ifndef NCCL_MIGRATION  
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclAllReduce_t func = (ncclAllReduce_t)dlsym(libNcclHandle, "ncclAllReduce");
  ret_val = func(sendbuff, recvbuff, count, datatype, op, comm, stream);
  NCCL_ENABLE_CKPT();
#else
  cudaStream_t newStream;
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
  ret_val = REAL_FNC(ncclAllReduce)(sendbuff, recvbuff, count, datatype, op, newComm, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclAllReduce, sendbuff, recvbuff, count, (int)datatype, (int)op, newComm, newStream, ret_val);
#endif

  return ret_val;
}

#undef ncclRecv
extern "C" ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) 
{
  typedef ncclResult_t (*ncclRecv_t)(const void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );
#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclRecv_t func = (ncclRecv_t)dlsym(libNcclHandle, "ncclRecv");
  ret_val = func(recvbuff, count, datatype, peer, comm, stream);
  NCCL_ENABLE_CKPT();
#else
  cudaStream_t newStream;
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
  ret_val = REAL_FNC(ncclRecv)(recvbuff, count, datatype, peer, newComm, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclRecv, recvbuff, count, datatype, peer, newComm, newStream, ret_val);
#endif

  return ret_val;
}

#undef ncclSend
extern "C" ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) 
{
  typedef ncclResult_t (*ncclSend_t)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );
#ifndef NCCL_MIGRATION 
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclSend_t func = (ncclSend_t)dlsym(libNcclHandle, "ncclSend");
  ret_val = func(sendbuff, count, datatype, peer, comm, stream);
  NCCL_ENABLE_CKPT();
#else
  cudaStream_t newStream;
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
    ret_val = REAL_FNC(ncclSend)(sendbuff, count, datatype, peer, newComm, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclSend, sendbuff, count, datatype, peer, newComm, newStream, ret_val);
#endif

  return ret_val;
}

#undef ncclReduceScatter
extern "C" ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
  typedef ncclResult_t (*ncclReduceScatter_t)(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclReduceScatter_t func = (ncclSend_t)dlsym(libNcclHandle, "ncclReduceScatter");
  ret_val = func(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  NCCL_ENABLE_CKPT();
#else
  cudaStream_t newStream;
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
  ret_val = REAL_FNC(ncclReduceScatter)(sendbuff, recvbuff, recvcount, datatype, op, newComm, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclReduceScatter, sendbuff, recvbuff, recvcount, datatype, op, newComm, newStream, ret_val);
#endif

  return ret_val;
}

#undef ncclBroadcast
extern "C" ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
  typedef ncclResult_t (*ncclBroadcast_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclBroadcast_t func = (ncclSend_t)dlsym(libNcclHandle, "ncclBroadcast");
  ret_val = func(sendbuff, recvbuff, count, datatype, root, comm, stream);
  NCCL_ENABLE_CKPT();
#else
  cudaStream_t newStream;
  ncclComm_t newComm;
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
  ret_val = REAL_FNC(ncclBroadcast)(sendbuff, recvbuff, count, datatype, root, newComm, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclBroadcast, sendbuff, recvbuff, count, datatype, root, newComm, newStream, ret_val);
#endif
  return ret_val;
}

#undef ncclBcast
extern "C" ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
  typedef ncclResult_t (*ncclBcast_t)(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclBcast_t func = (ncclSend_t)dlsym(libNcclHandle, "ncclBcast");
  ret_val = func(buff, count, datatype, root, comm, stream);
  NCCL_ENABLE_CKPT();
#else
  cudaStream_t newStream;
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
  ret_val = REAL_FNC(ncclBcast)(buff, count, datatype, root, newComm, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclBcast, buff, count, datatype, root, newComm, newStream, ret_val);
#endif
  return ret_val;
}

#undef ncclReduce
extern "C" ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
  typedef ncclResult_t (*ncclReduce_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid ); 

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclReduce_t func = (ncclSend_t)dlsym(libNcclHandle, "ncclReduce");
  ret_val = func(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
  NCCL_ENABLE_CKPT();
#else
  cudaStream_t newStream;
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhCUDAtype_t getCUDAtype = (lhCUDAtype_t)lhInfo.lhCUDAtypeFptr;
  check_cuda_map((void*)&newStream, (void*)&stream, CUDA_STREAM, getCUDAtype);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
  ret_val = REAL_FNC(ncclReduce)(sendbuff, recvbuff, count, datatype, op, root, newComm, newStream);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclReduce, sendbuff, recvbuff, count, datatype, op, root, newComm, newStream, ret_val);
#endif

  return ret_val;
}

// #undef ncclRedOpDestroy
// extern "C" ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm)
// {
//   typedef ncclResult_t (*ncclRedOpDestroy_t)(ncclRedOp_t op, ncclComm_t comm);
//   ncclResult_t ret_val = ncclSuccess;
//   if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}


// #ifndef NCCL_MIGRATION
//   NCCL_DISABLE_CKPT();
//   if(libNcclHandle == nullptr)
//     libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
//   ncclRedOpDestroy_t func = (ncclSend_t)dlsym(libNcclHandle, "ncclRedOpDestroy");
//   ret_val = func(op, comm);
//   NCCL_ENABLE_CKPT();
// #else
//   ncclComm_t newComm;
//   check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM);
//   DMTCP_PLUGIN_DISABLE_CKPT();
////   if(fsOfNewThread==0) 
// fsOfNewThread = getFs(tid, pid);
//   JUMP_TO_LOWER_HALF(fsOfNewThread);
//     ret_val = REAL_FNC(ncclRedOpDestroy)(op, newComm);
//   RETURN_TO_UPPER_HALF();
//   DMTCP_PLUGIN_ENABLE_CKPT();

//   logAPI(Cuda_Fnc_ncclRedOpDestroy, op, newComm, ret_val);
// #endif

//   return ret_val;
// }

// #undef ncclRedOpCreatePreMulSum
// extern "C" ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm)
// {
//   typedef ncclResult_t (*ncclRedOpCreatePreMulSum_t)(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
//   ncclResult_t ret_val = ncclSuccess;
//   if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}


// #ifndef NCCL_MIGRATION 
//   NCCL_DISABLE_CKPT();
//   if(libNcclHandle == nullptr)
//     libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
//   ncclRedOpCreatePreMulSum_t func = (ncclRedOpCreatePreMulSum_t)dlsym(libNcclHandle, "ncclRedOpCreatePreMulSum");
//   ret_val = func(op, scalar, datatype, residence, comm);
//   NCCL_ENABLE_CKPT();
// #else
//   ncclComm_t newComm;
//   check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM);
//   DMTCP_PLUGIN_DISABLE_CKPT();
////   if(fsOfNewThread==0) 
// fsOfNewThread = getFs(tid, pid);
//   JUMP_TO_LOWER_HALF(fsOfNewThread);
//   ret_val = REAL_FNC(ncclRedOpCreatePreMulSum)(op, scalar, datatype, residence, newComm);
//   RETURN_TO_UPPER_HALF();
//   DMTCP_PLUGIN_ENABLE_CKPT();

//   logAPI(Cuda_Fnc_ncclRedOpCreatePreMulSum, op, scalar, datatype, residence, newComm, ret_val);
// #endif

//   return ret_val;
// }

#undef ncclCommUserRank
extern "C" ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank)
{
  typedef ncclResult_t (*ncclCommUserRank_t)(const ncclComm_t comm, int* rank);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION  
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclCommUserRank_t func = (ncclCommUserRank_t)dlsym(libNcclHandle, "ncclCommUserRank");
  ret_val = func(comm, rank);
  NCCL_ENABLE_CKPT();
#else
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
    ret_val = REAL_FNC(ncclCommUserRank)(newComm, rank);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclCommUserRank, newComm, rank, ret_val);
#endif

  return ret_val;
}

#undef ncclCommCuDevice
extern "C" ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device)
{
  typedef ncclResult_t (*ncclCommCuDevice_t)(const ncclComm_t comm, int* device);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION  
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclCommCuDevice_t func = (ncclCommCuDevice_t)dlsym(libNcclHandle, "ncclCommCuDevice");
  ret_val = func(comm, device);
  NCCL_ENABLE_CKPT();
#else
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
    ret_val = REAL_FNC(ncclCommCuDevice)(newComm, device);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclCommCuDevice, newComm, device, ret_val);
#endif

  return ret_val;
}

#undef ncclCommCount
extern "C" ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count)
{
  typedef ncclResult_t (*ncclCommCount_t)(const ncclComm_t comm, int* count);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );
  
#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclCommCount_t func = (ncclCommCount_t)dlsym(libNcclHandle, "ncclCommCount");
  ret_val = func(comm, count);
  NCCL_ENABLE_CKPT();
#else
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
    ret_val = REAL_FNC(ncclCommCount)(newComm, count);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclCommCount, newComm, count, ret_val);
#endif
  return ret_val;
}

#undef ncclCommGetAsyncError
extern "C" ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError)
{
  typedef ncclResult_t (*ncclCommGetAsyncError_t)(ncclComm_t comm, ncclResult_t *asyncError);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclCommGetAsyncError_t func = (ncclCommGetAsyncError_t)dlsym(libNcclHandle, "ncclCommGetAsyncError");
  ret_val = func(comm, asyncError);
  NCCL_ENABLE_CKPT();
#else
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
  ret_val = REAL_FNC(ncclCommGetAsyncError)(newComm, asyncError);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
#endif

  return ret_val;
}

// #undef ncclGetLastError
// extern "C" const char*  ncclGetLastError(ncclComm_t comm)
// {
//   typedef const char* (*ncclGetLastError_t)(ncclComm_t comm);
//   if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}

//   char* ret_val = nullptr;

// #ifndef NCCL_MIGRATION
//   NCCL_DISABLE_CKPT();
//   if(libNcclHandle == nullptr)
//     libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
//   ncclGetLastError_t func = (ncclGetLastError_t)dlsym(libNcclHandle, "ncclGetLastError");
//   ret_val = const_cast<char*>(func(comm));
//   NCCL_ENABLE_CKPT();
// #else
//   ncclComm_t newComm;
//   check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM);
//   DMTCP_PLUGIN_DISABLE_CKPT();
////   if(fsOfNewThread==0) 
// fsOfNewThread = getFs(tid, pid);
//   JUMP_TO_LOWER_HALF(fsOfNewThread);
//     ret_val = const_cast<char*>(REAL_FNC(ncclGetLastError)(newComm));
//   RETURN_TO_UPPER_HALF();
//   DMTCP_PLUGIN_ENABLE_CKPT();
// #endif

//   return ret_val;
// }
#include <iostream>
#undef ncclGetErrorString
const char*  ncclGetErrorString(ncclResult_t result) 
{
  typedef const char* (*ncclGetErrorString_t)(ncclResult_t result);
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );
  char* ret_val = nullptr;

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclGetErrorString_t func = (ncclGetErrorString_t)dlsym(libNcclHandle, "ncclGetErrorString");
  ret_val = const_cast<char*>(func(result));
  NCCL_ENABLE_CKPT();
#else
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = const_cast<char*>(REAL_FNC(ncclGetErrorString)(result));
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
#endif
    std::cout << "ncclGetErrorString " << ret_val << std::endl;

  return ret_val;
}

#undef ncclCommAbort
extern "C" ncclResult_t  ncclCommAbort(ncclComm_t comm)
{
  typedef ncclResult_t (*ncclCommAbort_t)(ncclComm_t comm);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclCommAbort_t func = (ncclCommAbort_t)dlsym(libNcclHandle, "ncclCommAbort");
  ret_val = func(comm);
  NCCL_ENABLE_CKPT();
#else
  ncclComm_t newComm;
  // check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM);
  DMTCP_PLUGIN_DISABLE_CKPT();
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
    ret_val = REAL_FNC(ncclCommAbort)(newComm);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclCommAbort, &newComm, ret_val);
#endif

  return ret_val;
}

#undef ncclCommDestroy
extern "C" ncclResult_t  ncclCommDestroy(ncclComm_t comm)
{
  typedef ncclResult_t (*ncclCommDestroy_t)(ncclComm_t comm);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION  
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclCommDestroy_t func = (ncclCommDestroy_t)dlsym(libNcclHandle, "ncclCommDestroy");
  ret_val = func(comm);
  NCCL_ENABLE_CKPT();
#else
  ncclComm_t newComm;

  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  lhNickleType_t getNickleType = (lhNickleType_t)lhInfo.lhNcclTypeFptr;
  check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM, getNickleType);
    ret_val = REAL_FNC(ncclCommDestroy)(newComm);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclCommDestroy, &newComm, ret_val);
#endif
  return ret_val;
}

// #undef ncclCommFinalize
// extern "C" ncclResult_t  ncclCommFinalize(ncclComm_t comm)
// {
//   typedef ncclResult_t (*ncclCommFinalize_t)(ncclComm_t comm);
//   ncclResult_t ret_val = ncclSuccess;
//   if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}


// #ifndef NCCL_MIGRATION
//   NCCL_DISABLE_CKPT();
//   if(libNcclHandle == nullptr)
//     libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
//   ncclCommFinalize_t func = (ncclCommFinalize_t)dlsym(libNcclHandle, "ncclCommFinalize");
//   ret_val = func(comm);
//   NCCL_ENABLE_CKPT();
// #else
//   ncclComm_t newComm;
//   check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM);
//   DMTCP_PLUGIN_DISABLE_CKPT();
////   if(fsOfNewThread==0) 
// fsOfNewThread = getFs(tid, pid);
//   JUMP_TO_LOWER_HALF(fsOfNewThread);
//     ret_val = REAL_FNC(ncclCommFinalize)(newComm);
//   RETURN_TO_UPPER_HALF();
//   DMTCP_PLUGIN_ENABLE_CKPT();

//   logAPI(Cuda_Fnc_ncclCommFinalize, &newComm, ret_val);
// #endif
//   return ret_val;
// }

#undef ncclCommInitRank
extern "C" ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)
{
  typedef ncclResult_t (*ncclCommInitRank_t)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclCommInitRank_t func = (ncclCommInitRank_t)dlsym(libNcclHandle, "ncclCommInitRank");
  ret_val = func(comm, nranks, commId, rank);
  NCCL_ENABLE_CKPT();
#else
  // ncclComm_t newComm;
  // check_nccl_map((void*)&newComm, (void*)&comm, NCCL_COMM);
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(ncclCommInitRank)(comm, nranks, commId, rank);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclCommInitRank, comm, nranks, commId, rank, ret_val);
#endif

  return ret_val;
}

#undef ncclGetUniqueId
extern "C" ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId)
{
  typedef ncclResult_t (*ncclGetUniqueId_t)(ncclUniqueId* uniqueId);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );
  //fflush(stdout);

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclGetUniqueId_t func = (ncclGetUniqueId_t)dlsym(libNcclHandle, "ncclGetUniqueId");
  ret_val = func(uniqueId);
  NCCL_ENABLE_CKPT();
#else
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(ncclGetUniqueId)(uniqueId);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclGetUniqueId, uniqueId, ret_val);
#endif

  return ret_val;
}

#undef ncclGetVersion
extern "C" ncclResult_t  ncclGetVersion(int *version)
{
  typedef ncclResult_t (*ncclGetVersion_t)(int *version);
  ncclResult_t ret_val = ncclSuccess;
  if(tid == -1){pid = getpid();tid = syscall(SYS_gettid);}
  //printf("Called at func '%s' in line %i, tid:%d, pid:%d start.\n", __FUNCTION__, __LINE__, tid, pid );

#ifndef NCCL_MIGRATION
  NCCL_DISABLE_CKPT();
  if(libNcclHandle == nullptr)
    libNcclHandle = dlopen("/home/tian01.liu/AI-workspace/nccl-master/build/lib/libnccl.so.2", RTLD_LAZY | RTLD_GLOBAL);
  ncclGetVersion_t func = (ncclGetVersion_t)dlsym(libNcclHandle, "ncclGetVersion");
  ret_val = func(version);
  NCCL_ENABLE_CKPT();
#else
  DMTCP_PLUGIN_DISABLE_CKPT();
  //if(fsOfNewThread==0) 
  fsOfNewThread = getFs(tid, pid);
  JUMP_TO_LOWER_HALF(fsOfNewThread);
  ret_val = REAL_FNC(ncclGetVersion)(version);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();

  logAPI(Cuda_Fnc_ncclGetVersion, version, ret_val);
#endif

  return ret_val;
}
