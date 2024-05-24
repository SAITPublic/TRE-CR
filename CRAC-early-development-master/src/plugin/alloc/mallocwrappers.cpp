/****************************************************************************
 *   Copyright (C) 2006-2013 by Jason Ansel, Kapil Arya, and Gene Cooperman *
 *   jansel@csail.mit.edu, kapil@ccs.neu.edu, gene@ccs.neu.edu              *
 *                                                                          *
 *   This file is part of the dmtcp/src module of DMTCP (DMTCP:dmtcp/src).  *
 *                                                                          *
 *  DMTCP:dmtcp/src is free software: you can redistribute it and/or        *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP:dmtcp/src is distributed in the hope that it will be useful,      *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#include <stdlib.h>
#include <unistd.h>
#include "alloc.h"
#include "dmtcp.h"
#include <sys/syscall.h>
#include <iostream>

// added by tian01.liu, fix bug of realloc 2023.1.20
#include "procmapsarea.h"
#include <map>
#include <utility>
#include <pthread.h>
// std::map<void*, size_t> mapRealloc;

LIB_PRIVATE pid_t
dmtcp_gettid()
{
  return syscall(SYS_gettid);
}

EXTERNC int
dmtcp_alloc_enabled() { return 1; }
__thread int is_locked = 0;
__thread int need_to_unlock = 1;
class MapFile
{
  public:
    static MapFile& GetInstance()
    {
      static MapFile mapFile;
      return mapFile;
    }
    void DelItem(void* addr)
    { 
      if(reallocItems.count(addr) == 0){
        return;
      }
      if (is_locked == 0){
      pthread_mutex_lock(&mutex_tmp);
        is_locked = 1;
        need_to_unlock = 1;
      }else{
        need_to_unlock = 0;
      }
      std::map<void*, size_t>::iterator it = reallocItems.find(addr);
      if(it != reallocItems.end()){
        reallocItems.erase(it);
      }
      if (need_to_unlock == 1){
      pthread_mutex_unlock(&mutex_tmp);
        is_locked = 0;
      }else{
        need_to_unlock = 1;
      }
    }
    void AddItem(void* addr, size_t size)
    {
      if (is_locked == 0){
      pthread_mutex_lock(&mutex_tmp);
        is_locked = 1;
        need_to_unlock = 1;
      }else{
        need_to_unlock = 0;
      }
      reallocItems.insert(std::pair<void*, size_t>(addr, size));
      if (need_to_unlock == 1){
      pthread_mutex_unlock(&mutex_tmp);
        is_locked = 0;
      }else{
        need_to_unlock = 1;
      }
    }

    void* findItem(void* addr)
    {
      auto it = reallocItems.lower_bound(addr);
      if(it == reallocItems.end())
          return NULL;
      return it->first;
    }
  private:
    MapFile(){pthread_mutex_init(&mutex_tmp, NULL);};
    std::map<void*, size_t> reallocItems;
    pthread_mutex_t mutex_tmp;
};

/*
  This function checks whether we should skip the region or checkpoint fully or
  Partially.
  The idea is that we are recording each mmap by upper-half. So, all the
  ckpt'ble area
*/

#undef dmtcp_skip_memory_region_ckpting_ex
EXTERNC int
dmtcp_skip_memory_region_ckpting_ex(ProcMapsArea *area, int fd, int stack_was_seen)
{
    void* ptrInArea = (void*)(area->addr);
    size_t size = area->size;

    // the first equal or great than the area start address, and the address not beyond the area, then return 0
    // void* ptrFind = it->first;
    MapFile& mapFile = MapFile::GetInstance();
    void* ptrFind = mapFile.findItem(ptrInArea);
    if (NULL == ptrFind) return 1;
    if((long int)ptrFind <= (long int)ptrInArea + size)
        return 0;

    return 1;
}

extern "C" void *calloc(size_t nmemb, size_t size)
{
  DMTCP_PLUGIN_DISABLE_CKPT();
  void *retval = _real_calloc(nmemb, size);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return retval;
}

extern "C" void *malloc(size_t size)
{
  DMTCP_PLUGIN_DISABLE_CKPT();
  void *retval = _real_malloc(size);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return retval;
}

extern "C" void *memalign(size_t boundary, size_t size)
{
  DMTCP_PLUGIN_DISABLE_CKPT();
  void *retval = _real_memalign(boundary, size);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return retval;
}

extern "C" int
posix_memalign(void **memptr, size_t alignment, size_t size)
{
  DMTCP_PLUGIN_DISABLE_CKPT();
  int retval = _real_posix_memalign(memptr, alignment, size);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return retval;
}

extern "C" void *valloc(size_t size)
{
  DMTCP_PLUGIN_DISABLE_CKPT();
  void *retval = _real_valloc(size);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return retval;
}

extern "C" void
free(void *ptr)
{
  DMTCP_PLUGIN_DISABLE_CKPT();
  _real_free(ptr);
  
  // add by tian01.liu 2023.1.28
  if(ptr){
    MapFile& mapFile = MapFile::GetInstance();
    mapFile.DelItem(ptr);
  }

  DMTCP_PLUGIN_ENABLE_CKPT();
}

extern "C" void *realloc(void *ptr, size_t size)
{
  DMTCP_PLUGIN_DISABLE_CKPT();
  void *retval = _real_realloc(ptr, size);

  
  MapFile& mapFile = MapFile::GetInstance();
  // Notes: has bug on delItem with pytorch-gptj, delete temp. I will find the final solution, by tian01.liu 2023.9.18
  if(ptr){
    mapFile.DelItem(ptr);
  }
  mapFile.AddItem(retval, size);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return retval;
}
