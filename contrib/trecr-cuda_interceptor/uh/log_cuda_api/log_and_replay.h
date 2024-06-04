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

#ifndef LOG_AND_REPLAY_H
#define LOG_AND_REPLAY_H

#include <map>
#include <vector>
#include "lower_half_cuda_if.h"
#include "common.h"

/*****begin: move to this code to common.h by tian01.liu******/
// // enum for types
// enum pages_t {
//   CUDA_MALLOC_PAGE = 0,
//   CUDA_UVM_PAGE,
//   CUDA_HOST_ALLOC_PAGE,
//   CUDA_HEAP_PAGE,
//   CUMEM_ALLOC_PAGE
// };

// typedef struct Lhckpt_pages_t {
//   pages_t mem_type;
//   void * mem_addr;
//   size_t mem_len;
//   size_t comp_len; // by tian01.liu 2022.7.7 used for restart
// }lhckpt_pages_t;
/*****end: move to this code to common.h by tian01.liu******/

void logAPI(Cuda_Fncs_t cuda_fnc, ...);
// void replayAPI(CudaCallLog_t *l);
void logs_read_and_apply();
void refill_handles();

void disableLogging();
void enableLogging();
bool isLoggingDisabled();
void sendMsgToDmtcp(int i); // support timing by huiru.deng

std::vector<CudaCallLog_t>& getCudaCallsLog();
std::map<void*, page_lock_info_t>& getCudaHostCallsLog();
std::map<void *, lhckpt_pages_t> & getLhPageMaps();
#endif
