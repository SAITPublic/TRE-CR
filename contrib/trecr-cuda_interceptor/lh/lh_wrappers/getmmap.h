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

#include <vector>
#include <map> // by tian01.liu
#include <list>
#include "common.h"

std::vector<MmapInfo_t>& getMmappedList(int *num);
typedef std::vector<MmapInfo_t>& (*GetMmappedListFptr_t)(int *num);

std::vector<CudaCallLog_t>& getCudaCallsLog();
typedef std::vector<CudaCallLog_t>& (*GetCudaCallsLogFptr_t)();

// Added by biao.xing 2024.1.11
std::list<uint64_t> & getCUHandles();
void clearCUHandles(void);

// added by tian01.liu 2023.1.9, for new gpu memory blocks alloc architecture.
std::map<void *, lhckpt_pages_t>& getCudaBlockMaps();
typedef std::map<void *, lhckpt_pages_t>& (*GetCudaBlockMapsFptr_t)();

std::map<void*, page_lock_info_t>& getPageLockInfos();
typedef std::map<void *, page_lock_info_t>& (*GetPageLockInfosFptr_t)();

void * getEndOfHeap();
typedef void *(*GetEndOfHeapFptr_t)();
