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

#include <vector>
#include <algorithm>
#include <map>
#include <set> // add by tian01.liu 2023.1.28
#include <unordered_set>
#include <unordered_map>

#include "common.h"
#include "dmtcp.h"
#include "config.h"
#include "jassert.h"
#include "procmapsarea.h"
#include "getmmap.h"
#include "util.h"
#include "log_and_replay.h"
#include "mmap-wrapper.h"
#include "upper-half-wrappers.h"
#include "switch_context.h"

#define DEV_NVIDIA_STR "/dev/nvidia"

pthread_mutex_t mutex_for_map;
pthread_mutex_t mutex_for_map_fs;
pthread_mutex_t mutex_for_log;
pthread_mutex_t mutex_for_logtag;
clock_t start_finish_check, end;
extern clock_t start_finish_restore;
extern pthread_rwlock_t thread_lock;
using namespace dmtcp;
std::map<void *, lhckpt_pages_t> lh_pages_maps;
void *lh_ckpt_mem_addr = NULL;
size_t lh_ckpt_mem_size = 0;
int pagesize = sysconf(_SC_PAGESIZE);
GetMmappedListFptr_t fnc = NULL;
dmtcp::vector<MmapInfo_t> merged_uhmaps;
UpperHalfInfo_t uhInfo;

// by tian01.liu 2023.11.1
extern std::unordered_map<pid_t, unsigned long> fsThreadsMap;
extern std::map<pid_t, char*> shmThreadsMap; // 存储每个线程对应的共享内存地址，用于参数传递
extern std::vector<CudaCallLog_t> cudaCallsLog;
// by tian01.liu 2022.8.15
char *hPinMem = 0;
bool g_pin_record = true;
bool is_gpu_compress = false;
bool use_increamental = false;
// added by tian01.liu 2023.1.28. fix the bug of cudaMallocHost/cuMemAllocHost/cuMemHostAlloc
std::map<void *, st_page_lock_info> g_addrs_for_host;
void *lh_ckpt_page_lock_addr = NULL;
size_t lh_ckpt_page_lock_size = -1;

std::map<void*, page_lock_info_t>& getPageLockInfos()
{
    return g_addrs_for_host;
}

extern LowerHalfInfo_t lhInfo;
typedef void (*fp)(void *pointer, size_t size, void *dstpointer, size_t &compressedSize);
void doGpuCompression(void *pointer, size_t size, void *dstpointer, size_t &compressedSize)
{
  static __typeof__(fp) lowerHalfCompressWrapper = (__typeof__(fp))-1;
  // void *lowerHalf = -1;
  if (!initialized)
  {
    initialize_wrappers();
  }

  if (lowerHalfCompressWrapper == (__typeof__(fp))-1)
  {
    lowerHalfCompressWrapper = (__typeof__(fp))lhInfo.lhCompressFptr;
  }
  // TODO: Switch fs context
  lowerHalfCompressWrapper(pointer, size, dstpointer, compressedSize);
}

typedef void (*fp_ms)(void *srcAddrs[], size_t *srcLens, void *dstAddrs[], size_t *dstLens, /*char* hMems[],*/ size_t fragNums);
void doGpuCompressionMS(void *srcAddrs[], size_t *srcLens, void *dstAddrs[], size_t *dstLens, /*char* hMems[],*/ size_t fragNums)
{
  static __typeof__(fp_ms) lowerHalfCompressMSWrapper = (__typeof__(fp_ms))-1;
  // void *lowerHalf = -1;
  if (!initialized)
  {
    initialize_wrappers();
  }

  if (lowerHalfCompressMSWrapper == (__typeof__(fp_ms))-1)
  {
    lowerHalfCompressMSWrapper = (__typeof__(fp_ms))lhInfo.lhCompressMSFptr;
  }
  JTRACE("lowerHalfCompressMSWrapper")
  ((void *)lowerHalfCompressMSWrapper);

  // TODO: Switch fs context

  JTRACE("[WEILU]before lowerHalfCompressMSWrapper");
  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
  lowerHalfCompressMSWrapper(srcAddrs, srcLens, dstAddrs, dstLens, /*hMems,*/ fragNums);
  RETURN_TO_UPPER_HALF();
  JTRACE("[WEILU]after lowerHalfCompressMSWrapper");
}

typedef void (*inc_ckpt)(void *mem_addr, size_t mem_size);
void GpuIncreamtalCkpt(void *mem_addr, size_t mem_size)
{
  static __typeof__(inc_ckpt) lowerHalfGpuIncreamtalCkpt = (__typeof__(inc_ckpt))-1;
  // void *lowerHalf = -1;
  if (!initialized)
  {
    initialize_wrappers();
  }
  if (lowerHalfGpuIncreamtalCkpt == (__typeof__(inc_ckpt))-1)
  {
    lowerHalfGpuIncreamtalCkpt = (__typeof__(inc_ckpt))lhInfo.lhIncreamentalCkpt;
  }
  lowerHalfGpuIncreamtalCkpt(mem_addr, mem_size);
}

typedef void (*cp_hash)(void *mem_addr, size_t mem_size, int *gpu_fd, pid_t pid);
void ComputeHash(void *mem_addr, size_t mem_size, int *gpu_fd, pid_t pid)
{
  static __typeof__(cp_hash) lowerHalfComputeHash = (__typeof__(cp_hash))-1;
  // void *lowerHalf = -1;
  if (!initialized)
  {
    initialize_wrappers();
  }
  if (lowerHalfComputeHash == (__typeof__(cp_hash))-1)
  {
    lowerHalfComputeHash = (__typeof__(cp_hash))lhInfo.lhComputeHash;
  }
  lowerHalfComputeHash(mem_addr, mem_size, gpu_fd, pid);
}

static bool skipWritingTextSegments = false;
extern "C" pid_t dmtcp_get_real_pid();
/* This function returns a range of zero or non-zero pages. If the first page
 * is non-zero, it searches for all contiguous non-zero pages and returns them.
 * If the first page is all-zero, it searches for contiguous zero pages and
 * returns them.
 */
static void
mtcp_get_next_page_range(Area *area, size_t *size, int *is_zero)
{
  char *pg;
  char *prevAddr;
  size_t count = 0;
  const size_t one_MB = (1024 * 1024);

  if (area->size < one_MB)
  {
    *size = area->size;
    *is_zero = 0;
    return;
  }
  *size = one_MB;
  *is_zero = Util::areZeroPages(area->addr, one_MB / MTCP_PAGE_SIZE);
  prevAddr = area->addr;
  for (pg = area->addr + one_MB;
       pg < area->addr + area->size;
       pg += one_MB)
  {
    size_t minsize = MIN(one_MB, (size_t)(area->addr + area->size - pg));
    if (*is_zero != Util::areZeroPages(pg, minsize / MTCP_PAGE_SIZE))
    {
      break;
    }
    *size += minsize;
    if (*is_zero && ++count % 10 == 0)
    { // madvise every 10MB
      if (madvise(prevAddr, area->addr + *size - prevAddr,
                  MADV_DONTNEED) == -1)
      {
        JNOTE("error doing madvise(..., MADV_DONTNEED)")
        (JASSERT_ERRNO)((void *)area->addr)((int)*size);
        prevAddr = pg;
      }
    }
  }
}

static void
mtcp_write_non_rwx_and_anonymous_pages(int fd, Area *orig_area)
{
  Area area = *orig_area;

  /* Now give read permission to the anonymous/[heap]/[stack]/[stack:XXX] pages
   * that do not have read permission. We should remove the permission
   * as soon as we are done writing the area to the checkpoint image
   *
   * NOTE: Changing the permission here can results in two adjacent memory
   * areas to become one (merged), if they have similar permissions. This can
   * results in a modified /proc/self/maps file. We shouldn't get affected by
   * the changes because we are going to remove the PROT_READ later in the
   * code and that should reset the /proc/self/maps files to its original
   * condition.
   */

  JASSERT(orig_area->name[0] == '\0' || (strcmp(orig_area->name, "[heap]") == 0) ||
          (strcmp(orig_area->name, "[stack]") == 0) ||
          (Util::strStartsWith(area.name, "[stack:XXX]")));

  if ((orig_area->prot & PROT_READ) == 0)
  {
    JASSERT(mprotect(orig_area->addr, orig_area->size,
                     orig_area->prot | PROT_READ) == 0)
    (JASSERT_ERRNO)(orig_area->size)((void *)orig_area->addr)
        .Text("error adding PROT_READ to mem region");
  }

  while (area.size > 0)
  {
    size_t size;
    int is_zero;
    Area a = area;
    if (dmtcp_infiniband_enabled && dmtcp_infiniband_enabled())
    {
      size = area.size;
      is_zero = 0;
    }
    else
    {
      mtcp_get_next_page_range(&a, &size, &is_zero);
    }

    a.properties = is_zero ? DMTCP_ZERO_PAGE : 0;
    a.size = size;

    Util::writeAll(fd, &a, sizeof(a));
    if (!is_zero)
    {
      Util::writeAll(fd, a.addr, a.size);
    }
    else
    {
      if (madvise(a.addr, a.size, MADV_DONTNEED) == -1)
      {
        JNOTE("error doing madvise(..., MADV_DONTNEED)")
        (JASSERT_ERRNO)((void *)a.addr)((int)a.size);
      }
    }
    area.addr += size;
    area.size -= size;
  }

  /* Now remove the PROT_READ from the area if it didn't have it originally
   */
  if ((orig_area->prot & PROT_READ) == 0)
  {
    JASSERT(mprotect(orig_area->addr, orig_area->size, orig_area->prot) == 0)
    (JASSERT_ERRNO)((void *)orig_area->addr)(orig_area->size)
        .Text("error removing PROT_READ from mem region.");
  }
}

static void
writememoryarea(int fd, Area *area, int stack_was_seen)
{
  void *addr = area->addr;

  if (!(area->flags & MAP_ANONYMOUS))
  {
    JTRACE("save region")
    (addr)(area->size)(area->name)(area->offset);
  }
  else if (area->name[0] == '\0')
  {
    JTRACE("save anonymous")
    (addr)(area->size);
  }
  else
  {
    JTRACE("save anonymous")
    (addr)(area->size)(area->name)(area->offset);
  }

  if ((area->name[0]) == '\0')
  {
    char *brk = (char *)sbrk(0);
    if (brk > area->addr && brk <= area->addr + area->size)
    {
      strcpy(area->name, "[heap]");
    }
  }

  if (area->size == 0)
  {
    /* Kernel won't let us munmap this.  But we don't need to restore it. */
    JTRACE("skipping over [stack] segment (not the orig stack)")
    (addr)(area->size);
  }
  else if (0 == strcmp(area->name, "[vsyscall]") ||
           0 == strcmp(area->name, "[vectors]") ||
           0 == strcmp(area->name, "[vvar]") ||
           0 == strcmp(area->name, "[vdso]"))
  {
    JTRACE("skipping over memory special section")
    (area->name)(addr)(area->size);
  }
  else if (area->prot == 0 ||
           (area->name[0] == '\0' &&
            ((area->flags & MAP_ANONYMOUS) != 0) &&
            ((area->flags & MAP_PRIVATE) != 0)))
  {
    /* Detect zero pages and do not write them to ckpt image.
     * Currently, we detect zero pages in non-rwx mapping and anonymous
     * mappings only
     */
    mtcp_write_non_rwx_and_anonymous_pages(fd, area);
  }
  else
  {
    /* Anonymous sections need to have their data copied to the file,
     *   as there is no file that contains their data
     * We also save shared files to checkpoint file to handle shared memory
     *   implemented with backing files
     */
    JASSERT((area->flags & MAP_ANONYMOUS) || (area->flags & MAP_SHARED));

    if (skipWritingTextSegments && (area->prot & PROT_EXEC))
    {
      area->properties |= DMTCP_SKIP_WRITING_TEXT_SEGMENTS;
      Util::writeAll(fd, area, sizeof(*area));
      JTRACE("Skipping over text segments")
      (area->name)((void *)area->addr);
    }
    else
    {
      Util::writeAll(fd, area, sizeof(*area));
      Util::writeAll(fd, area->addr, area->size);
    }
  }
}

// Returns true if needle is in the haystack
static inline int
regionContains(const void *haystackStart,
               const void *haystackEnd,
               const void *needleStart,
               const void *needleEnd)
{
  return needleStart >= haystackStart && needleEnd <= haystackEnd;
}

void getAndMergeUhMaps()
{
  if (lhInfo.lhMmapListFptr && fnc == NULL)
  {
    fnc = (GetMmappedListFptr_t)lhInfo.lhMmapListFptr;
    int numUhRegions = 0;
    std::vector<MmapInfo_t> uh_mmaps = fnc(&numUhRegions);

    // add by tian01.liu, reason: multi-times checkpoint
    merged_uhmaps.clear();

    // merge the entries if two entries are continous
    merged_uhmaps.push_back(uh_mmaps[0]);
    for (size_t i = 1; i < uh_mmaps.size(); i++)
    {
      MmapInfo_t last_merged = merged_uhmaps.back();
      void *uhMmapStart = uh_mmaps[i].addr;
      void *uhMmapEnd = (VA)uh_mmaps[i].addr + uh_mmaps[i].len;
      void *lastmergedStart = last_merged.addr;
      void *lastmergedEnd = (VA)last_merged.addr + last_merged.len;
      MmapInfo_t merged_item;
      if (regionContains(uhMmapStart, uhMmapEnd,
                         lastmergedStart, lastmergedEnd))
      {
        merged_uhmaps.pop_back();
        merged_uhmaps.push_back(uh_mmaps[i]);
      }
      else if (regionContains(lastmergedStart, lastmergedEnd,
                              uhMmapStart, uhMmapEnd))
      {
        continue;
      }
      else if (lastmergedStart > uhMmapStart && uhMmapEnd >= lastmergedStart)
      {
        merged_item.addr = uhMmapStart;
        merged_item.len = (VA)lastmergedEnd - (VA)uhMmapStart;
        merged_uhmaps.pop_back();
        merged_uhmaps.push_back(merged_item);
      }
      else if (lastmergedStart < uhMmapStart && lastmergedEnd >= uhMmapStart)
      {
        merged_item.addr = lastmergedStart;
        merged_item.len = (VA)uhMmapEnd - (VA)lastmergedStart;
        merged_uhmaps.pop_back();
        merged_uhmaps.push_back(merged_item);
      }
      else
      {
        // insert uh_maps[i] to the merged list as a new item
        merged_uhmaps.push_back(uh_mmaps[i]);
      }
    }
    // TODO: print the content once
  }
}

/*
  This function checks whether we should skip the region or checkpoint fully or
  Partially.
  The idea is that we are recording each mmap by upper-half. So, all the
  ckpt'ble area
*/

// add by tian01.liu for fix the bug of page-lock memory
void *g_page_lock_pool_addr = NULL;
void *g_stack_addr = NULL;

#undef dmtcp_skip_memory_region_ckpting
EXTERNC int
dmtcp_skip_memory_region_ckpting(ProcMapsArea *area, int fd, int stack_was_seen)
{
  JNOTE("In skip area")
  ((void *)area->addr);
  ssize_t rc = 1;
  if (strstr(area->name, "vvar") ||
      strstr(area->name, "vdso") ||
      strstr(area->name, "vsyscall") ||
      strstr(area->name, DEV_NVIDIA_STR)
      /*(strstr(area->name, DEV_NVIDIA_STR) && NULL == strstr(area->name, "nvidiactl"))*/)
  {
    return rc; // skip this region
  }

  if (g_stack_addr == NULL)
  {
    // todo: get the start address of page-lock pool
    lhGetAppStack funcGetStack = (lhGetAppStack)lhInfo.lhGetAppStackSegFptr;
    g_stack_addr = (void *)funcGetStack();
  }

  if (area->addr == g_stack_addr)
  {
    pid_t pid = getpid();
    char filename[100];
    snprintf(filename, 100, "./orig_stack_data_%d", pid);

    int fd_stack = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    write(fd_stack, area->addr, area->size);
    close(fd_stack);
  }

  // add by tian01.liu for bug of page-lock memory
  if (g_page_lock_pool_addr == NULL)
  {
    // todo: get the start address of page-lock pool
    LhGetPageLockMemAddr_t func = (LhGetPageLockMemAddr_t)lhInfo.lhPLMPGetFptr;
    if (func)
      g_page_lock_pool_addr = func();

    JNOTE("g_page_lock_pool_addr is NULL");
  }
  // JNOTE("==========In page-lock pool=======") ((void*)area->addr) (g_page_lock_pool_addr);
  if (area->addr == g_page_lock_pool_addr)
  {
    JNOTE("==========In page-lock pool======= SKIP");
    return 1;
  }

  // get and merge uh maps
  getAndMergeUhMaps();

  // smaller than smallest uhmaps or greater than largest address
  if ((area->endAddr < merged_uhmaps[0].addr) ||
      (area->addr > (void *)((VA)merged_uhmaps.back().addr +
                             merged_uhmaps.back().len)))
  {
    return rc;
  }

  // Don't skip the lh_ckpt_region
  // [leeyy] GPU部分直接跳过
  if (lh_ckpt_mem_addr && area->addr == lh_ckpt_mem_addr)
  {
    if (use_increamental)
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }

  size_t i = 0;
  // if(!Util::strEndsWith(area->name, "nvidiactl")){
  while (i < merged_uhmaps.size())
  {
    void *uhMmapStart = merged_uhmaps[i].addr;
    void *uhMmapEnd = (VA)merged_uhmaps[i].addr + merged_uhmaps[i].len;

    if (regionContains(uhMmapStart, uhMmapEnd, area->addr, area->endAddr))
    {
      JNOTE("Case 1 detected")
      ((void *)area->addr)((void *)area->endAddr)(uhMmapStart)(uhMmapEnd);
      return 0; // checkpoint this region
    }
    else if ((area->addr < uhMmapStart) && (uhMmapStart < area->endAddr) && (area->endAddr <= uhMmapEnd))
    {
      JNOTE("Case 2 detected")
      ((void *)area->addr)((void *)area->endAddr)(uhMmapStart)(uhMmapEnd);

      // skip the region above the uhMmapStart
      area->addr = (VA)uhMmapStart;
      area->size = area->endAddr - area->addr;
      JNOTE("Case 2: area to checkpoint")
      ((void *)area->addr)((void *)area->endAddr)(area->size);
      return 0; // checkpoint but values changed
    }
    else if ((uhMmapStart <= area->addr) && (area->addr < uhMmapEnd))
    {
      // check the next element in the merged list if it contains in the area
      // TODO: handle that case
      //
      //  traverse until uhmap's start addr is bigger than the area -> endAddr
      //  rc = number of the item in the  array
      //
      JNOTE("Case 3: detected")
      ((void *)area->addr)((void *)area->endAddr)(area->size);
      //      int dummy=1; while(dummy);
      ProcMapsArea newArea = *area;
      newArea.endAddr = (VA)uhMmapEnd;
      newArea.size = newArea.endAddr - newArea.addr;
      writememoryarea(fd, &newArea, stack_was_seen);
      // whiteAreas[count++] = newArea;
      while (i < merged_uhmaps.size() - 1 && merged_uhmaps[++i].addr < area->endAddr)
      {
        // TODO: Update the area after each writememoryarea
        // remove the merged uhmaps node when it's ckpt'ed
        ProcMapsArea newArea = *area;
        uhMmapStart = merged_uhmaps[i].addr;
        uhMmapEnd = (VA)merged_uhmaps[i].addr + merged_uhmaps[i].len;
        if (regionContains(area->addr, area->endAddr, uhMmapStart, uhMmapEnd))
        {
          newArea.addr = (VA)uhMmapStart;
          newArea.endAddr = (VA)uhMmapEnd;
          newArea.size = newArea.endAddr - newArea.addr;
          writememoryarea(fd, &newArea, stack_was_seen);
        }
        else
        {
          newArea.addr = (VA)uhMmapStart;
          newArea.size = newArea.endAddr - newArea.addr;
          writememoryarea(fd, &newArea, stack_was_seen);
          return 1;
        }
      }
    }
    else if (regionContains(area->addr, area->endAddr,
                            uhMmapStart, uhMmapEnd))
    {
      JNOTE("Case 4: detected")
      ((void *)area->addr)((void *)area->endAddr)(area->size);
      fflush(stdout);
      // TODO find out what is going on here
      // TODO: this usecase is not completed; fix it later
      // int dummy = 1; while(dummy);
      // JNOTE("skipping the region partially ") (area->addr) (area->endAddr)
      //   (area->size) (array[i].len);
      area->addr = (VA)uhMmapStart;
      area->endAddr = (VA)uhMmapEnd;
      area->size = area->endAddr - area->addr;
      // rc = 2; // skip partially
      return 0;
      // break;
    }
    i++;
  }
  // }
  // JNOTE("[lt]++++++unexpect branch++++++");
  return 1;
}
/*
void init()
{
  // typedef void (*cudaRegisterAllFptr_t) ();
  void * dlsym_handle = _real_dlopen(NULL, RTLD_NOW);
  JASSERT(dlsym_handle) (_real_dlerror());
  void * cudaRegisterAllFptr = _real_dlsym(dlsym_handle, "_ZL24__sti____cudaRegisterAllv");
  JASSERT(cudaRegisterAllFptr) (_real_dlerror());
  JNOTE("found symbol") (cudaRegisterAllFptr);
}
*/
#define USE_COMP 0
#define MAX_FRAG_NUM 30
#define BLK_DIVIDED_SIZE 1024LL * 1024 * 1024 * 3
#define LARGE_BLK_SIZE 1024LL * 1024 * 1024 * 3

#define SIZE_ONE_SUB_BLK (1024LL * 1024 * 1024 * 2)

#if USE_COMP
/* this function used multi-stream to decrease compress time*/
static void *
gpu_mem_handle_ms_thread(void *dummy)
{

  JNOTE("In gpu_mem_handle_ms_thread");

  lh_pages_maps = getLhPageMaps();
  size_t frag_cnt = lh_pages_maps.size();
  if (frag_cnt == 0)
    return NULL;

  double total_len = 0;
  for (auto lh_page : lh_pages_maps)
  {
    total_len += lh_page.second.mem_len;
  }

  pid_t pid = getpid();


  int fd_for_comp = -1;
  if (!use_increamental)
  {
    char filename[100];
    snprintf(filename, 100, "./gpu_mem_data_%d", pid);
    fd_for_comp = open(filename, O_WRONLY | O_CREAT, 0644);
  }

  clock_t startComp, endComp;
  clock_t startMove, endMove;
  clock_t startWrite, endWrite;

  double dCompTime = 0;
  double dMoveTime = 0;
  double dWriteTime = 0;
  double comp_size = 0;

  void *srcAddrs[MAX_FRAG_NUM]; // store uncompressed mem addr
  void *dstAddrs[MAX_FRAG_NUM]; // store compressed mem addr
  size_t srcLens[MAX_FRAG_NUM]; // store uncompressed mem len
  size_t dstLens[MAX_FRAG_NUM]; // store compressed mem len
  int gpuID[MAX_FRAG_NUM];
  // set used to store large blocks, which would cudaMalloc fail
  set<void *> largeBlocks;
  largeBlocks.clear();

  memset(dstLens, 0, sizeof(size_t) * MAX_FRAG_NUM);
  size_t index = 0;
  size_t max_size = 0;

  size_t curTotalSize = 0;
  for (auto lh_page : lh_pages_maps)
  {
    void *mem_addr = lh_page.second.mem_addr;
    size_t mem_len = lh_page.second.mem_len;
    // leeyy increamental
    if (use_increamental) // compress and incremental
    {
      int gpu_fd = -1;
      ComputeHash((void *)(mem_addr), mem_len, &gpu_fd, pid);

      if (gpu_fd == -1)
        continue;

      gpuID[index] = gpu_fd;

      // todo: malloc an area to store the compressed gpu data
      void *d_compressed_data = nullptr;
      cudaError_t ret_val = cudaSuccess;
      ret_val = cudaMalloc(&d_compressed_data, mem_len + 4000000);
      if (ret_val != cudaSuccess) // cudaMalloc fail,then add to set, later to handle.
      {
        largeBlocks.insert(mem_addr);
        close(gpu_fd);
        continue;
      }

      max_size = max_size > mem_len ? max_size : mem_len;
      // init info of each memory fragment
      srcAddrs[index] = (void *)mem_addr;
      srcLens[index] = mem_len;
      dstAddrs[index] = d_compressed_data;

      // keyAddrs[index] = lh_page.first; // first == second.mem_addr
      curTotalSize += mem_len;
      index++;

      // TODO: beyond 18 fragments or current blocks size great than 3G
      if (index > 18 || curTotalSize > BLK_DIVIDED_SIZE)
      {
        startComp = clock();
        doGpuCompressionMS(srcAddrs, srcLens, dstAddrs, dstLens, /*hMems,*/ index);
        endComp = clock();
        dCompTime += (double)(endComp - startComp);
        double dTmpMove = 0;
        // TODO: update compressed memory data to file
        // cudaStream_t stream[MAX_FRAG_NUM];
        for (size_t i = 0; i < index; i++)
        {
          lhckpt_pages_t stPage = lh_pages_maps[srcAddrs[i]];
          stPage.comp_len = dstLens[i];
          comp_size += dstLens[i];
          startMove = clock();
          cudaError_t ret_val = cudaMemcpy((void *)hPinMem, dstAddrs[i], dstLens[i], cudaMemcpyDeviceToHost);
          if (ret_val != cudaSuccess)
            exit(EXIT_FAILURE);

          endMove = clock();
          dMoveTime += (double)(endMove - startMove);
          dTmpMove += (double)(endMove - startMove);

          startWrite = clock();
          /*int ret = */ write(gpuID[i], (void *)&stPage, sizeof(lhckpt_pages_t));
          size_t write_size = 0;
          while (write_size < dstLens[i])
          {
            write_size += write(gpuID[i], hPinMem + write_size, dstLens[i] - write_size);
          }
          endWrite = clock();
          dWriteTime += (double)(endWrite - startWrite);
          // printf("write compressed data time:%.3f\n", (double)(endWrite - startWrite) / CLOCKS_PER_SEC * 1000);
          cudaFree(dstAddrs[i]);
          close(gpuID[i]);
        }

        index = 0;
        max_size = 0;
        curTotalSize = 0;
      }
    }
    else // only compress
    {
      void *d_compressed_data = nullptr;
      cudaError_t ret_val = cudaSuccess;
      ret_val = cudaMalloc(&d_compressed_data, mem_len + 4000000);
      if (ret_val != cudaSuccess) // cudaMalloc fail,then add to set, later to handle.
      {
        largeBlocks.insert(mem_addr);
        continue;
      }

      max_size = max_size > mem_len ? max_size : mem_len;
      // init info of each memory fragment
      srcAddrs[index] = (void *)mem_addr;
      srcLens[index] = mem_len;
      dstAddrs[index] = d_compressed_data;

      // keyAddrs[index] = lh_page.first; // first == second.mem_addr
      curTotalSize += mem_len;
      index++;

      // TODO: beyond 18 fragments or current blocks size great than 3G
      if (index > 18 || curTotalSize > BLK_DIVIDED_SIZE)
      {
        startComp = clock();
        doGpuCompressionMS(srcAddrs, srcLens, dstAddrs, dstLens, /*hMems,*/ index);
        endComp = clock();
        dCompTime += (double)(endComp - startComp);
        double dTmpMove = 0;
        // TODO: update compressed memory data to file
        // cudaStream_t stream[MAX_FRAG_NUM];
        for (size_t i = 0; i < index; i++)
        {
          lhckpt_pages_t stPage = lh_pages_maps[srcAddrs[i]];
          stPage.comp_len = dstLens[i];
          comp_size += dstLens[i];
          startMove = clock();
          cudaError_t ret_val = cudaMemcpy((void *)hPinMem, dstAddrs[i], dstLens[i], cudaMemcpyDeviceToHost);
          if (ret_val != cudaSuccess)
            exit(EXIT_FAILURE);

          endMove = clock();
          dMoveTime += (double)(endMove - startMove);
          dTmpMove += (double)(endMove - startMove);

          startWrite = clock();
          /*int ret = */ write(fd_for_comp, (void *)&stPage, sizeof(lhckpt_pages_t));
          size_t write_size = 0;
          while (write_size < dstLens[i])
          {
            write_size += write(fd_for_comp, hPinMem + write_size, dstLens[i] - write_size);
          }
          endWrite = clock();
          dWriteTime += (double)(endWrite - startWrite);
          // printf("write compressed data time:%.3f\n", (double)(endWrite - startWrite) / CLOCKS_PER_SEC * 1000);
          cudaFree(dstAddrs[i]);
        }

        index = 0;
        max_size = 0;
        curTotalSize = 0;
      }
    }
  }

  // TODO: pass memory info to lower half to execute compress
  if (curTotalSize)
  {
    clock_t start, end;
    start = clock();
    doGpuCompressionMS(srcAddrs, srcLens, dstAddrs, dstLens, /*hMems,*/ index);
    end = clock();
    dCompTime += (double)(end - start);
    // printf("gpu compress time is %.3f\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    start = clock();
    // TODO: update compressed memory data to file
    for (size_t i = 0; i < index; i++)
    {
      if (use_increamental) {
        comp_size += dstLens[i];
        startMove = clock();
        cudaError_t ret_val = cudaMemcpy((void *)hPinMem, dstAddrs[i], dstLens[i], cudaMemcpyDeviceToHost);
        endMove = clock();
        printf("cudaMemcpy result:%i\n", ret_val);
        dMoveTime += (double)(endMove - startMove);

        lhckpt_pages_t stPage = lh_pages_maps[srcAddrs[i]];
        stPage.comp_len = dstLens[i];

        startWrite = clock();
        write(gpuID[i], (void *)&stPage, sizeof(lhckpt_pages_t));
        size_t write_size = 0;
        while (write_size < dstLens[i])
        {
          write_size += write(gpuID[i], hPinMem + write_size, dstLens[i] - write_size);
        }
        // int ret = write(fd/*pipe_hds[1]*/, (void *)&stPage, sizeof(lhckpt_pages_t));
        // ret = write(fd/*pipe_hds[1]*/, hPinMem, dstLens[i]);
        endWrite = clock();
        dWriteTime += (double)(endWrite - startWrite);
        // printf("write compressed data time:%.3f\n", (double)(endWrite - startWrite) / CLOCKS_PER_SEC * 1000);
        // printf("orig addr:%p mem_addr:%p mem_len:%ld\n", keyAddrs[i], stPage.mem_addr, stPage.mem_len);
        cudaFree(dstAddrs[i]);
        close(gpuID[i]);
      }
      else
      {
        comp_size += dstLens[i];
        startMove = clock();
        cudaError_t ret_val = cudaMemcpy((void *)hPinMem, dstAddrs[i], dstLens[i], cudaMemcpyDeviceToHost);
        endMove = clock();
        printf("cudaMemcpy result:%i\n", ret_val);
        dMoveTime += (double)(endMove - startMove);

        lhckpt_pages_t stPage = lh_pages_maps[srcAddrs[i]];
        stPage.comp_len = dstLens[i];

        startWrite = clock();
        write(fd_for_comp, (void *)&stPage, sizeof(lhckpt_pages_t));
        size_t write_size = 0;
        while (write_size < dstLens[i])
        {
          write_size += write(fd_for_comp, hPinMem + write_size, dstLens[i] - write_size);
        }
        endWrite = clock();
        dWriteTime += (double)(endWrite - startWrite);
        // printf("write compressed data time:%.3f\n", (double)(endWrite - startWrite) / CLOCKS_PER_SEC * 1000);
        // printf("orig addr:%p mem_addr:%p mem_len:%ld\n", keyAddrs[i], stPage.mem_addr, stPage.mem_len);
        cudaFree(dstAddrs[i]);
      }
    }
    end = clock();
    printf("move time is %.6f\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
  }

  // printf("compress total time:%.3f, copy time:%.3f\n", dCompTime/CLOCKS_PER_SEC * 1000, dMoveTime/CLOCKS_PER_SEC*1000);
  //  TODO: handle large blocks, if can reach here and use_incremental, then must can incremental
  for (auto largeBlock : largeBlocks)
  {
    void *blkAddr = (void *)largeBlock;
    lhckpt_pages_t stPage = lh_pages_maps[blkAddr];
    printf("blkAddr:%p, blkLen:%ld\n", blkAddr, stPage.mem_len);
    // TODO: large block will divided into 4 subblocks
    size_t blkSize = stPage.mem_len;
    void *d_compressed_data = nullptr;
    size_t compressedSize = 0;
    cudaError_t ret_val = cudaSuccess;

    // startComp = clock();
    ret_val = cudaMalloc(&d_compressed_data, blkSize + 400000);
    // printf("cudaMalloc ret:%i addr:%p\n", ret_val, d_compressed_data);
    if (ret_val != cudaSuccess)
      continue;

    startComp = clock();
    doGpuCompression(blkAddr, blkSize, d_compressed_data, compressedSize);
    endComp = clock();
    comp_size += compressedSize;
    dCompTime += (double)(endComp - startComp);

    printf("gpu compress 11G time is %.6f\n", (double)(endComp - startComp) / CLOCKS_PER_SEC * 1000);

    // startMove = clock();
    // char* hMem = (char*)malloc(compressedSize);
    startMove = clock();
    ret_val = cudaMemcpy((void *)hPinMem, d_compressed_data, compressedSize, cudaMemcpyDeviceToHost);
    endMove = clock();
    dMoveTime += (double)(endMove - startMove);

    // lhckpt_pages_t stPage = lh_pages_maps[keyAddrs[i]];
    stPage.comp_len = compressedSize;

    startWrite = clock();
    int ret = 0;
    if (!use_increamental) {
      ret = write(fd_for_comp, (void *)&stPage, sizeof(lhckpt_pages_t));
      size_t write_size = 0;
      while (write_size < compressedSize)
      {
        write_size += write(fd_for_comp, hPinMem + write_size, compressedSize - write_size);
      }
    }
    else
    {
      char tmp_file_path[1000];
      snprintf(tmp_file_path, 1000, "./dmtcp_gpu_ckpt_%d_%llu", pid, (unsigned long long)blkAddr);
      int fd_tmp =  open(tmp_file_path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
      ret = write(fd_tmp, (void *)&stPage, sizeof(lhckpt_pages_t));
      size_t write_size = 0;
      while (write_size < compressedSize)
      {
        write_size += write(fd_tmp, hPinMem + write_size, compressedSize - write_size);
      }
      close(fd_tmp);
    }
    endWrite = clock();
    dWriteTime += (double)(endWrite - startWrite);

    printf("write compressed data, ret = %d, copy result:%i\n", ret, ret_val);
    cudaFree(d_compressed_data);
  }

  if (fd_for_comp != -1)
    close(fd_for_comp);

  // cudaFreeHost(hPinMem);
  // g_bMemHandleFinished = true;
  printf("fragments cnt:%ld compress total time:%.3f, copy time:%.3f, io time:%.3f\n", frag_cnt, dCompTime / CLOCKS_PER_SEC * 1000, dMoveTime / CLOCKS_PER_SEC * 1000, dWriteTime / CLOCKS_PER_SEC * 1000);
  printf("compress_size:%.3f compression ratio is:%.3f\n", comp_size, total_len / comp_size);
  return NULL;
}
#endif

/*************add by tian01.liu 2023.2.8,begin**************/
void save_page_lock_to_memory()
{
  size_t total_size = sizeof(int);
  for (auto lock_page : g_addrs_for_host)
  {
    if (hPinMem != nullptr && (void*)lock_page.first == (void*)hPinMem)
    {
        continue;
    }

    total_size += (sizeof(st_page_lock_info) + sizeof(void *) + lock_page.second.size);
  }

  // printf("total size:%ld, blks:%ld\n", total_size - sizeof(int), g_addrs_for_host.size());
  if (total_size > 0)
  {
    total_size = ((total_size + pagesize - 1) & ~(pagesize - 1));
    // mmap a region in the process address space big enough for the structure
    // + data + initial_guard_page
    void *addr = mmap(NULL, pagesize + total_size + pagesize,
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    JASSERT(addr != MAP_FAILED)
    (addr)(JASSERT_ERRNO);
    JASSERT(mprotect(addr, pagesize, PROT_EXEC) != -1)
    (addr)(JASSERT_ERRNO);
    addr = (void *)((VA)addr + pagesize);
    JASSERT(mprotect((void *)((VA)addr + total_size), pagesize, PROT_EXEC) != -1)
    (addr)(JASSERT_ERRNO)(total_size)(pagesize);

    lh_ckpt_page_lock_addr = addr;
    lh_ckpt_page_lock_size = total_size;

    size_t count = 0;
    int total_entries = g_addrs_for_host.size();
    memcpy(((VA)addr + count), &total_entries, sizeof total_entries);
    count += sizeof(total_entries);
    // printf("[lt] point 0....\n");
    for (auto lock_page : g_addrs_for_host)
    {
      void *mem_addr = lock_page.first;
      if (hPinMem != nullptr && mem_addr == (void*)hPinMem)
      {
          continue;
      }

      st_page_lock_info mem_len = lock_page.second;
      // copy the metadata and data to the new mmap'ed region
      void *dest = memcpy(((VA)addr + count), (void *)&mem_addr,
                          sizeof(void *));
      // printf("[lt] point 1....\n");
      JASSERT(dest == ((VA)addr + count))
      ("memcpy failed...0");
      count += sizeof(void *);

      dest = memcpy(((VA)addr + count), (void *)&mem_len,
                    sizeof(st_page_lock_info));
      // printf("[lt] point 2....\n");
      JASSERT(dest == ((VA)addr + count))
      ("memcpy failed...1");
      count += sizeof(st_page_lock_info);

      // printf("mem_addr:%p\n", mem_addr);
      dest = memcpy(((VA)addr + count), mem_addr, mem_len.size);
      // printf("[lt] point 3....\n");
      JASSERT(dest == ((VA)addr + count))
      ("memcpy failed...2");
      count += mem_len.size;
    }
  }
}
/*************add by tian01.liu 2023.2.8,end****************/

void save_lh_pages_to_memory()
{
/*************************gpu compress workflow**********************************/
#if USE_COMP
  printf("Before gpu compress is_gpu_compress: %d\n", is_gpu_compress);
  if (is_gpu_compress)
  {
    gpu_mem_handle_ms_thread(NULL);
  }
  return;
#endif

  /*************************original workflow**********************************/
  // get the Lower-half page maps
  // printf("[LT] enter save_lh_pages_to_memory\n");
  lh_pages_maps = getLhPageMaps();
  /*
  // add the device heap entry to lh_pages_maps
  size_t cudaDeviceHeapSize = 0;
  cudaDeviceGetLimit(&cudaDeviceHeapSize, cudaLimitMallocHeapSize);

  JASSERT(lhInfo.lhGetDeviceHeapFptr) ("GetDeviceHeapFptr is not set up");
  GetDeviceHeapPtrFptr_t func = (GetDeviceHeapPtrFptr_t) lhInfo.lhGetDeviceHeapFptr;
  void *mallocPtr = func();

  size_t actualHeapSize = (size_t)((VA)ROUND_UP(mallocPtr) - (VA)lhInfo.lhDeviceHeap);
  JASSERT(actualHeapSize > 0) (mallocPtr) (lhInfo.lhDeviceHeap);
  lhckpt_pages_t page = {CUDA_HEAP_PAGE, lhInfo.lhDeviceHeap, actualHeapSize};
  lh_pages_maps[lhInfo.lhDeviceHeap] = page; */

  // yueyang.li for increamental ckpt
  // inc_obj.mem_set.clear();

  if (use_increamental)
  {
    for (auto lh_page : lh_pages_maps)
    {
      void *mem_addr = lh_page.second.mem_addr;
      size_t mem_size = lh_page.second.mem_len;
      int gpu_fd = -1;
      ComputeHash((void *)(mem_addr), mem_size, &gpu_fd, getpid());
      if (gpu_fd == -1)
      {
        continue;
      }
      else
      {
        char *cpu_mem_addr = (char *)malloc(mem_size);
        cuMemcpyDtoH(cpu_mem_addr, (CUdeviceptr)mem_addr, mem_size);
        write(gpu_fd, &mem_addr, sizeof(mem_addr));
        write(gpu_fd, &mem_size, sizeof(mem_size));
        write(gpu_fd, cpu_mem_addr, mem_size);
        free(cpu_mem_addr);
        close(gpu_fd);
      }
    }
    return;
  }
  size_t total_size = sizeof(int);
  for (auto lh_page : lh_pages_maps)
  {
    // printf("\n Address = %p with size = %lu", lh_page.first,
    // lh_page.second.mem_len);
    // lhckpt_pages_t
    total_size += lh_page.second.mem_len + sizeof(lh_page.second);
  }
  if (total_size > 0)
  {
    // round up to the page size
    total_size = ((total_size + pagesize - 1) & ~(pagesize - 1));
    // mmap a region in the process address space big enough for the structure
    // + data + initial_guard_page
    void *addr = mmap(NULL, pagesize + total_size + pagesize,
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    JASSERT(addr != MAP_FAILED)
    (addr)(JASSERT_ERRNO);
    JASSERT(mprotect(addr, pagesize, PROT_EXEC) != -1)
    (addr)(JASSERT_ERRNO);
    addr = (void *)((VA)addr + pagesize);
    JASSERT(mprotect((void *)((VA)addr + total_size), pagesize, PROT_EXEC) != -1)
    (addr)(JASSERT_ERRNO)(total_size)(pagesize);

    // make this address and size available to dmtcp_skip_memory_region
    lh_ckpt_mem_addr = addr;
    lh_ckpt_mem_size = total_size;

    size_t count = 0;
    int total_entries = lh_pages_maps.size();
    memcpy(((VA)addr + count), &total_entries, sizeof total_entries);
    count += sizeof(total_entries);
    // mprotect with read permission on the page
    /* Que: should we change the read permission of the entire cuda malloc'ed
      region at once? So that when we mprotect back to the ---p permission, we
      don't see many entries in /proc/pid/maps file even if the perms are same.
    */
    for (auto lh_page : lh_pages_maps)
    {
      void *mem_addr = lh_page.second.mem_addr;
      size_t mem_len = lh_page.second.mem_len;
      // copy the metadata and data to the new mmap'ed region
      void *dest = memcpy(((VA)addr + count), (void *)&lh_page.second,
                          sizeof(lh_page.second));
      JASSERT(dest == ((VA)addr + count))
      ("memcpy failed")(addr)(dest)(count)(sizeof(lh_page.second))(JASSERT_ERRNO);
      count += sizeof(lh_page.second);
      // copy the actual data
      switch (lh_page.second.mem_type)
      {
      case (CUDA_MALLOC_PAGE):
      case (CUDA_UVM_PAGE):
      {
	      cudaSetDevice(lh_page.second.devId);
        cudaMemcpy(((VA)addr + count), mem_addr, mem_len,
                   cudaMemcpyDeviceToHost);
	// printf("cudaMemcpy in pre_ckpt, ret:%i\n", ret);
        break;
      }
      case (CUMEM_ALLOC_PAGE):
      {
        // copy back the actual data
        cuMemcpyDtoH(((VA)addr + count), (CUdeviceptr)mem_addr, mem_len);
        break;
      }
      /*
      case (CUDA_HEAP_PAGE):
      {
        void *__cudaPtr = NULL;
        void * deviceHeapStart = mem_addr;
        size_t heapSize = mem_len;
        cudaMalloc(&__cudaPtr, heapSize);
        JASSERT(lhInfo.lhCopyToCudaPtrFptr)
               ("copyFromCudaPtrFptr is not set up");
        CopyToCudaPtrFptr_t func1 =
                 (CopyToCudaPtrFptr_t) lhInfo.lhCopyToCudaPtrFptr;
        func1(__cudaPtr, deviceHeapStart, heapSize);
        cudaMemcpy(((VA)addr + count),
                    __cudaPtr,
                    mem_len, cudaMemcpyDeviceToHost);
        cudaFree(__cudaPtr);
        break;
      } */
      default:
      {
        JASSERT(false)
        ("page type unkown");
      }
      }
      // JASSERT(dest == (void *)((uint64_t)addr + count))("memcpy failed")
      //  (addr) (count) (mem_addr) (mem_len);
      count += mem_len;
    }
  }
}

void pre_ckpt()
{
  /**/
  // printf("[lt] begin sleep in pre_ckpt\n");
  // sleep(1000);
  // clock_t start, end;
  // start = clock();
  disableLogging();
  cudaDeviceSynchronize();
  save_lh_pages_to_memory();
  enableLogging();
  // end = clock();
  // printf("merge gpu data time:%.3f\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
  // // added by tian01.liu
  // start = clock();
  save_page_lock_to_memory();
  // end = clock();
  // printf("merge page-lock data:%.3f\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
  fflush(stdout);
}

// Writes out the lhinfo global object to a file. Returns 0 on success,
// -1 on failure.
static void
writeUhInfoToFile()
{
  char filename[100];
  snprintf(filename, 100, "./uhInfo_%d", getpid());
  int fd = open(filename, O_WRONLY | O_CREAT, 0644);
  JASSERT(fd != -1)
  ("Could not create uhaddr.bin file.")(JASSERT_ERRNO);

  size_t rc = write(fd, &uhInfo, sizeof(uhInfo));
  JASSERT(rc >= sizeof(uhInfo))
  ("Wrote fewer bytes than expected to uhaddr.bin")(JASSERT_ERRNO);
  close(fd);
}

// sets up upper-half info for the lower-half to use on the restart
static void
setupUpperHalfInfo()
{
  GetEndOfHeapFptr_t func = (GetEndOfHeapFptr_t)lhInfo.uhEndofHeapFptr;
  uhInfo.uhEndofHeap = (void *)func();
  uhInfo.lhPagesRegion = (void *)lh_ckpt_mem_addr;
  uhInfo.cudaLogVectorFptr = (void *)&getCudaCallsLog;
  // leeyy 0306
  uhInfo.is_gpu_compress = false;

  // added by tian01.liu 2023.1.9, for new gpu memory alloc architecture
  uhInfo.cudaBlockMapFptr = (void *)&getLhPageMaps;
  Lhmmu_profile_t mmuProfile;
  LhGetMMUProfile_t func1 = (LhGetMMUProfile_t)lhInfo.lhGetMMUProfileFptr;
  func1(&mmuProfile);
  uhInfo.mmu_reserve_big_addr = mmuProfile.mmu_reserve_big_addr;
  uhInfo.mmu_reserve_big_size = mmuProfile.mmu_reserve_big_size;
  uhInfo.mmu_reserve_small_addr = mmuProfile.mmu_reserve_small_addr;
  uhInfo.mmu_reserve_small_size = mmuProfile.mmu_reserve_small_size;
  uhInfo.mmu_reserve_allign = mmuProfile.mmu_reserve_allign;
  uhInfo.mmu_reserve_flags = mmuProfile.mmu_reserve_flags;

  // by tian01.liu fixed the bug of page-lock memory
  uhInfo.lhPageLockRegion = (void *)lh_ckpt_page_lock_addr;
  uhInfo.lhPageLockSize = lh_ckpt_page_lock_size;
  uhInfo.uhPageLockMapFptr = (void*)&getPageLockInfos;
  LhGetPageLockMemAddr_t func2 = (LhGetPageLockMemAddr_t)lhInfo.lhPLMPGetFptr;
  uhInfo.lhPageLockPoolAddr = func2();

  lhGetAppStack funcGetStack = (lhGetAppStack)lhInfo.lhGetAppStackSegFptr;
  uhInfo.appStackAddr = (void *)funcGetStack();

  // FIXME: We'll just write out the uhInfo object to a file; the lower half
  // will read this file to figure out the information. This is ugly
  // but will work for now.
  unsigned long addr = 0;
  syscall(SYS_arch_prctl, ARCH_GET_FS, &addr);
  JNOTE("upper-half FS")
  ((void *)addr);
  JNOTE("uhInfo")
  ((void *)&uhInfo)(uhInfo.uhEndofHeap)(uhInfo.lhPagesRegion)(uhInfo.cudaLogVectorFptr);
  writeUhInfoToFile();
}

void resume()
{
  // unmap the region we mapped it earlier
  if (lh_ckpt_mem_addr != NULL && lh_ckpt_mem_size > 0)
  {
    JASSERT(munmap(lh_ckpt_mem_addr, lh_ckpt_mem_size) != -1)
    ("munmap failed!")(lh_ckpt_mem_addr)(lh_ckpt_mem_size);
    JASSERT(munmap((VA)lh_ckpt_mem_addr - pagesize, pagesize) != -1)
    ("munmap failed!")((VA)lh_ckpt_mem_addr - pagesize)(pagesize);
  }
  // by tian01.liu fixed the bug of page-lock memory
  else if (lh_ckpt_page_lock_addr != NULL && lh_ckpt_page_lock_size >= 0)
  {
    JASSERT(munmap(lh_ckpt_page_lock_addr, lh_ckpt_page_lock_size) != -1)
    ("munmap failed!")(lh_ckpt_page_lock_addr)(lh_ckpt_page_lock_size);
    JASSERT(munmap((VA)lh_ckpt_page_lock_addr - pagesize, pagesize) != -1)
    ("munmap failed!")((VA)lh_ckpt_page_lock_addr - pagesize)(pagesize);
  }
  else
  {
    JTRACE("no memory region was allocated earlier")
    (lh_ckpt_mem_addr)(lh_ckpt_mem_size);
  }

  setupUpperHalfInfo();

  // add by tian01.liu fixed bug
  fnc = NULL;
}

void restart()
{
  reset_wrappers();
  initialize_wrappers();
  // fix lower-half fs
  unsigned long addr = 0;
  syscall(SYS_arch_prctl, ARCH_GET_FS, &addr);
  // We copy the upper-half's FS register's magic number (addr+40) into
  // the lower-half's FS register magic number (lhFsAddr+40)
  // The lhInfo.lhFsAddr+40 contains an old magic number from the previous
  // program execution. This old magic number should be updated to the new one
  // Otherwise context switching would fail.
  memcpy((long *)((VA)lhInfo.lhFsAddr + 40), (long *)(addr + 40), sizeof(long));
  JNOTE("upper-half FS")
  ((void *)addr);
  JNOTE("lower-half FS")
  ((void *)lhInfo.lhFsAddr);

  // add by tian01.liu, reason: multi-times checkpoint
  fnc = NULL;
}

clock_t total_start;
clock_t clk_restart;
extern clock_t replay_start, replay_end;
extern void replay_gpu_status();
extern std::map<cudaStream_t, cudaStream_t*> cudaStreamMap;
extern std::map<cudaEvent_t, cudaEvent_t*> cudaEventMap;
extern void replayAPI(CudaCallLog_t *l, unsigned long threadFs, pid_t tid);

extern pthread_rwlock_t thread_lock;

static void
cuda_plugin_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data)
{
  switch (event)
  {
  case DMTCP_EVENT_PRE_EXEC:
  {
    JNOTE("*** DMTCP_EVENT_PRE_EXEC");
    pthread_mutex_init(&mutex_for_map, NULL);
    pthread_mutex_init(&mutex_for_log, NULL);
    pthread_mutex_init(&mutex_for_logtag, NULL);
    break;
  }
  case DMTCP_EVENT_INIT:
  {
    // JTRACE("*** DMTCP_EVENT_INIT");
    // JTRACE("Plugin intialized");
    // printf("*** DMTCP_EVENT_INIT...0, cudaCallsLog size:%ld\n", cudaCallsLog.size());
    // leeyy
    pthread_rwlockattr_t attr;
    pthread_rwlockattr_init(&attr);
    pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);

    initialize_wrappers();
    pthread_mutex_init(&mutex_for_map, NULL);
    pthread_mutex_init(&mutex_for_map_fs, NULL);
    pthread_mutex_init(&mutex_for_log, NULL);
    pthread_mutex_init(&mutex_for_logtag, NULL);
    pthread_rwlock_init(&thread_lock, &attr);
    total_start = clock();

    // TODO: save the cuda logs before main
    char filename[100];
    snprintf(filename, 100, "./uhInfo_cuda_log_%d", getpid());
    int fd = open(filename, O_WRONLY | O_CREAT, 0644);
    JASSERT(fd != -1)
    ("Could not create uhInfo_cuda_log file.")(JASSERT_ERRNO);
    size_t cudaLogSize = cudaCallsLog.size();
    // JNOTE("cudaLogSize:")(cudaLogSize);
    size_t rc = write(fd, &cudaLogSize, sizeof(size_t));
    JASSERT(rc >= sizeof(size_t))
    ("Wrote fewer bytes than expected to uhaddr.bin")(JASSERT_ERRNO);

    for (size_t i = 0; i < cudaLogSize; i++)
    {
      CudaCallLog_t log = cudaCallsLog[i];
      rc = write(fd, &log, sizeof(CudaCallLog_t));
      JASSERT(rc >= sizeof(CudaCallLog_t))
      ("Wrote fewer bytes than expected to uhaddr.bin")(JASSERT_ERRNO);
    }
    close(fd);
    is_gpu_compress = false;
    if (is_gpu_compress && hPinMem == nullptr)
    {
      g_pin_record = false;
      cudaMallocHost((void **)&hPinMem, 1024LL * 1024 * 1024 * 2);
      g_pin_record = true;
    }
    break;
  }
  case DMTCP_EVENT_EXIT:
  {
    JTRACE("*** DMTCP_EVENT_EXIT");
    // printf("In DMTCP_EVENT_EXIT is_gpu_compress: %d\n", is_gpu_compress);
    // end = clock();
    // printf("[lt] Total time from resume to exit;%.3f\n", (double)(end - start_finish_check) / CLOCKS_PER_SEC * 1000);
    // printf("[lt] Total time from restore to exit;%.3f\n", (double)(end - clk_restart - (replay_end - replay_start)) / CLOCKS_PER_SEC * 1000);
    // printf("[lt] Total time from init to exit;%.3f\n", (double)(end - total_start) / CLOCKS_PER_SEC * 1000);
    break;
  }
  case DMTCP_EVENT_PRESUSPEND:
  {
    break;
  }
  case DMTCP_EVENT_PRECHECKPOINT:
  {
    JNOTE("enter precheck of crac");
    // for(auto item : fsThreadsMap){
    //   fprintf(stderr,"item key %i, item value\n", item.first);
    // }
    pre_ckpt();

    break;
  }
  case DMTCP_EVENT_RESUME:
  {
    resume();
    break;
  }
  case DMTCP_EVENT_RESTART:
  {
    restart();
    LhInitPageLockPool_t releaseFunc = (LhInitPageLockPool_t)(lhInfo.lhInitPageLockPoolFptr);

    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
    if (releaseFunc)
      releaseFunc();
    replay_gpu_status();
    RETURN_TO_UPPER_HALF();

    // TODO:recreate threads of lower-half
    LhNewThread_t newThPtr = (LhNewThread_t)lhInfo.lhNewThreadFptr;
    LhGetFs_t getFsFunc = (LhGetFs_t)(lhInfo.lhGetFsFptr);
    pid_t pid = getpid();
    for (auto item : fsThreadsMap)
    {
      unsigned long tmpFs = 0;
      pid_t tid = item.first;
      key_t key = tid + dmtcp_virtual_to_real_pid(pid);
      if (pid == tid)
        continue;

      JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
      newThPtr(tid, key);
      getFsFunc(tid, &tmpFs);
      RETURN_TO_UPPER_HALF();
      fsThreadsMap[item.first] = tmpFs;
    }

    fsThreadsMap[pid] = lhInfo.lhFsAddr;

    // TODO: 清空stream和event的map, 此两个容器是给replay阶段用
    // cudaStreamMap.clear();
    // cudaEventMap.clear();
    // fprintf(stderr,"after RETURN_TO_UPPER_HALF\n");
    // fflush(stdout);
    // LhWake_t wakeFunc = (LhWake_t)(lhInfo.lhWakeFptr);
    // LhWaitFinish_t waitFunc = (LhWaitFinish_t)(lhInfo.lhWaitFnhFptr);
    // TODO: replayAPI
    for (size_t i = 0; i < cudaCallsLog.size(); i++)
    {
      CudaCallLog_t logItem = cudaCallsLog[i];
      pid_t tid = logItem.thread_id;
      replayAPI(&logItem, fsThreadsMap[tid], tid);
    }

    break;
  }
  default:
    break;
  }
}

/*
static DmtcpBarrier cudaPluginBarriers[] = {
  { DMTCP_GLOBAL_BARRIER_PRE_CKPT, pre_ckpt, "checkpoint" },
  { DMTCP_GLOBAL_BARRIER_RESUME, resume, "resume" },
  { DMTCP_GLOBAL_BARRIER_RESTART, restart, "restart" }
};
*/
DmtcpPluginDescriptor_t cuda_plugin = {
    DMTCP_PLUGIN_API_VERSION,
    PACKAGE_VERSION,
    "cuda_plugin",
    "DMTCP",
    "dmtcp@ccs.neu.edu",
    "Cuda Split Plugin",
    cuda_plugin_event_hook};
//  DMTCP_DECL_BARRIERS(cudaPluginBarriers),

DMTCP_DECL_PLUGIN(cuda_plugin);
