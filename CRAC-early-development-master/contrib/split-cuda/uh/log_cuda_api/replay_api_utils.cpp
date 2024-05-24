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
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <pthread.h> // test code, by tian01.liu
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include "switch_context.h"
#include "upper-half-wrappers.h"
#include "upper-half-cuda-wrappers.h"

#include <map>

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_INTERNAL_COMPILATION__
// #define __CUDACC__
#endif
#include "crt/host_runtime.h"
#include "crt/device_functions.h"
#include "log_and_replay.h"

typedef struct ptxLog {
void* addr;
long len;
void* ptx;
}ptxlog;
ptxlog ptxlg[10];

// add by tian01.liu 2023.11.1
extern std::map<pid_t, char*> shmThreadsMap;
void **new_fatCubinHandle = NULL;
std::map<CUcontext, CUcontext*> contextmap;
std::map<CUmodule, CUmodule*> modulemap;
std::map<CUfunction, CUfunction*> functionmap;
std::map<CUevent, CUevent*> eventmap;
std::map<CUstream, CUstream*> streammap;
std::map<CUdeviceptr, CUdeviceptr*> deviceptrmap;
std::map<CUtexref, CUtexref*> texrefmap;
#define RCEVENT 0
#define RCMODULE 1
int* test;
void *getCUtype(void* cuType, int type)
{
  switch (type)
  {
    case CU_DEV_PTR:
    {
      if (deviceptrmap.size() == 0) return NULL;
      return (void*)deviceptrmap[*(CUdeviceptr*)cuType];
    }
    case CU_CONTEXT:
    {
      if (contextmap.size() == 0) return NULL;
      return (void*)contextmap[*(CUcontext*)cuType];
    }
    case CU_MODULE:
    {
      if (modulemap.size() == 0) return NULL;
      return (void*)modulemap[*(CUmodule*)cuType];
    }
    case CU_FUNCTION:
    {
      if (functionmap.size() == 0) return NULL;
      return (void*)functionmap[*(CUfunction*)cuType];
    }
    case CU_EVENT:
    {
      if (eventmap.size() == 0) return NULL;
      return (void*)eventmap[*(CUevent*)cuType];
    }
    case CU_STREAM:
    {
      if (streammap.size() == 0) return NULL;
      return (void*)streammap[*(CUstream*)cuType];
    }
    case CU_TEXREF:
    {
      if (texrefmap.size() == 0) return NULL;
      return (void*)texrefmap[*(CUtexref*)cuType];
    }
    default:
    assert(false);
  }
}

void freeCUtype(void* cuType, int type)
{
  switch (type)
  {
    case CU_DEV_PTR:
    {
      if (deviceptrmap.size() == 0) return;
      delete deviceptrmap[*(CUdeviceptr*)cuType];
      deviceptrmap.erase(*(CUdeviceptr*)cuType);
      break;
    }
    case CU_CONTEXT:
    {
      if (contextmap.size() == 0) return;
      delete contextmap[*(CUcontext*)cuType];
      contextmap.erase(*(CUcontext*)cuType);
      break;
    }
    case CU_MODULE:
    {
      if (modulemap.size() == 0) return;
      delete modulemap[*(CUmodule*)cuType];
      modulemap.erase(*(CUmodule*)cuType);
      break;
    }
    case CU_FUNCTION:
    {
      if (functionmap.size() == 0) return;
      delete functionmap[*(CUfunction*)cuType];
      functionmap.erase(*(CUfunction*)cuType);
      break;
    }
    case CU_EVENT:
    {
      if (eventmap.size() == 0) return;
      delete eventmap[*(CUevent*)cuType];
      eventmap.erase(*(CUevent*)cuType);
      break;
    }
    case CU_STREAM:
    {
      if (streammap.size() == 0) return;
      delete streammap[*(CUstream*)cuType];
      streammap.erase(*(CUstream*)cuType);
      break;
    }
    case CU_TEXREF:
    {
      if (texrefmap.size() == 0) return;
      delete texrefmap[*(CUtexref*)cuType];
      texrefmap.erase(*(CUtexref*)cuType);
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

std::map<cudaStream_t, cudaStream_t*> cudaStreamMap;
std::map<cudaEvent_t, cudaEvent_t*> cudaEventMap;
void *getCudaType(void* cudaType, int type)
{
  switch (type)
  {
    case CUDA_STREAM:
    {
      if (cudaStreamMap.size() == 0) return NULL;
      return (void*)cudaStreamMap[*(cudaStream_t*)cudaType];
    }
    case CUDA_EVENT:
    {
      if (cudaEventMap.size() == 0) return NULL;
      return (void*)cudaEventMap[*(cudaEvent_t*)cudaType];
    }
    default:
    assert(false);
  }
}

void freeCudaType(void* cudaType, int type)
{
  switch (type)
  {
    case CUDA_STREAM:
    {
      if (cudaStreamMap.size() == 0) return;
      cudaStreamMap.erase(*(cudaStream_t*)cudaType);
      break;
    }
    case CUDA_EVENT:
    {
      if (cudaEventMap.size() == 0) return;
      cudaEventMap.erase(*(cudaEvent_t*)cudaType);
      break;
    }
    default:
    assert(false);
  }
}

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

void writeInfo(void* oldmod, void* newmod, int type)
{
    int fd;
    if (type == RCMODULE)
        fd =  open("moduleinfo.dat", O_CREAT | O_WRONLY | O_APPEND, 0600);
    else if (type == RCEVENT)
        fd =  open("eventinfo.dat", O_CREAT | O_WRONLY | O_APPEND , 0600);
    write(fd, &oldmod, sizeof(oldmod));
    write(fd, &newmod, sizeof(newmod));
    close(fd);
}
typedef void (*replayAPI_t)(CudaCallLog_t *l);
extern LowerHalfInfo_t lhInfo;
void replayAPI(CudaCallLog_t *l, unsigned long threadFs, pid_t tid)
{
  // Todo: need find the root cause, the modification only skip the segment fault, by tian01.liu 2023.12.13
  if (tid == 0)
    return;

  replayAPI_t repaly_func = (replayAPI_t)lhInfo.logs_read_and_apply_handle;
  JUMP_TO_LOWER_HALF(threadFs);
  repaly_func(l);
  RETURN_TO_UPPER_HALF();
}


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
        int size = read(fd, &ptxlg[id].addr, sizeof(void*));
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
