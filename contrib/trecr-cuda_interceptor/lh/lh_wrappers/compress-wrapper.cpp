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
#define _GNU_SOURCE // For MAP_ANONYMOUS
#endif
#include <errno.h>
#include <stddef.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <vector>
#include <algorithm>

#include "common.h"
#include "logging.h"
#include "kernel-loader.h"
#include "utils.h"
#include "switch_context.h"
/*****************by tian01.liu 2022.7.5******************************/
#include "nvcomp.hpp"
#include "nvcompManagerFactory.hpp"
using namespace nvcomp;
/*****************by tian01.liu 2022.7.5******************************/


// #if 1

// #define MAX_FRAG_NUMS 30
// static void
// doGpuCompressionMS(void * srcBufs[], size_t *srcLens, void * dstBufs[], size_t *dstLens, /*char* hMems[] ,*/size_t mem_frag_nums)
// {
//     DLOG(ERROR, "Enter doGpuCompressionMS...\n");
// 	//return;

//     if(!mem_frag_nums)
// 	return;

//     JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
//     DLOG(ERROR, "Success jump to lower half...\n");

//     int chunk_size = 1 << 16;
//     nvcompType_t data_type = NVCOMP_TYPE_CHAR;
//     std::string comp_format = "bitcomp";

//     cudaStream_t stream[MAX_FRAG_NUMS];
//     std::shared_ptr<nvcompManagerBase> manager[MAX_FRAG_NUMS];
//     int gpu_num = 0;

//     //clock_t start_all,end_all;
//     //start_all = clock();
//     for(size_t i = 0; i < mem_frag_nums; i++)
//     {
//         cudaStreamCreate(&stream[i]);

//         if (comp_format == "lz4") {
//               manager[i] = std::make_shared<LZ4Manager>(chunk_size, data_type, stream[i], gpu_num, NoComputeNoVerify);
//          } else if (comp_format == "snappy") {
//               manager[i] = std::make_shared<SnappyManager>(chunk_size, stream[i], gpu_num, NoComputeNoVerify);
//          } else if (comp_format == "bitcomp") {
//               manager[i] = std::make_shared<BitcompManager>(data_type, 0, stream[i], gpu_num, NoComputeNoVerify);
//          } else if (comp_format == "ans") {
//               manager[i] = std::make_shared<ANSManager>(chunk_size, stream[i], gpu_num, NoComputeNoVerify);
//          }
//         auto compress_config = manager[i]->configure_compression(srcLens[i]);
//         size_t comp_out_bytes = compress_config.max_compressed_buffer_size;
//         printf("init out bytes: %ld\n", comp_out_bytes);
//         manager[i]->compress((uint8_t*)srcBufs[i], (uint8_t*)dstBufs[i], compress_config);
//         printf("finish compress\n");
//     }

//     //double d_copy_time = 0;

//     //cudaStream_t streamCopy[MAX_FRAG_NUMS];
//     for(size_t i=0; i<mem_frag_nums; i++)
//     {
//         cudaStreamSynchronize(stream[i]);
//         printf("finish stream sync\n");
//         size_t comp_out_bytes = manager[i]->get_compressed_output_size((uint8_t*)dstBufs[i]);
//         printf("real out bytes: %ld\n", comp_out_bytes);
//         printf("orignal size:%ld, after compressed size:%ld\n", srcLens[i], comp_out_bytes);
//         dstLens[i] = comp_out_bytes;
//         cudaStreamDestroy(stream[i]);

//         //cudaStreamCreate(&streamCopy[i]);
// 	//clock_t start,end;
// 	//start = clock();
// 	//cudaMemcpyAsync(hMems[i], dstBufs[i], comp_out_bytes, cudaMemcpyDeviceToHost, streamCopy[i]);
// 	//end = clock();
// 	//d_copy_time += (end - start);

//     }

//     //for(size_t i = 0; i < mem_frag_nums; i++)
//     //{
// 	//cudaStreamSynchronize(streamCopy[i]);
// 	//cudaStreamDestroy(streamCopy[i]);
//     //}
//     //end_all = clock();

//     //double d_comp_time = end_all - start_all - d_copy_time;

//     //printf("compress_time:%.3f  cpy_time:%.3f\n", d_comp_time / CLOCKS_PER_SEC * 1000, d_copy_time / CLOCKS_PER_SEC * 1000);

//     RETURN_TO_UPPER_HALF();
//     DLOG(ERROR, "return to upper halt\n");
// }


// /********************************
//  * this function used to do gpu compression
//  * by tian01.liu 2022.7.5
//  * *****************************/
// static void
// doGpuCompression(void* uncompressedBuf, size_t size, void* compressedBuf, size_t &compressedSize)
// {
//     //void *ret = MAP_FAILED;
//     JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);
//     //ret = __mmapWrapper(addr, length, prot, flags, fd, offset);
//     //RETURN_TO_UPPER_HALF();
//     //compress_func_wrapper(NULL, 0, NULL, 0, 0);

//     DLOG(INFO, "Enter doGpuCompress in lower half...\n");
//     if(!uncompressedBuf || !size)
//         return;

//     int chunk_size = 1 << 16;
//     nvcompType_t data_type = NVCOMP_TYPE_CHAR;
//     std::string comp_format = "bitcomp";

//     int gpu_num = 0;
//     //cudaSetDevice(gpu_num);
//     DLOG(INFO, "Finish cudaSetDevice\n");
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     //DLOG(INFO, "Finish cudaStreamCreate\n");
//     std::shared_ptr<nvcompManagerBase> manager;
//     if (comp_format == "lz4") {
//       manager = std::make_shared<LZ4Manager>(chunk_size, data_type, stream, gpu_num, NoComputeNoVerify);
//     } else if (comp_format == "snappy") {
//       manager = std::make_shared<SnappyManager>(chunk_size, stream, gpu_num, NoComputeNoVerify);
//     } else if (comp_format == "bitcomp") {
//       manager = std::make_shared<BitcompManager>(data_type, 0, stream, gpu_num, NoComputeNoVerify);
//       //DLOG(ERROR, "Finish bitcomp manager create...\n");
//     } else if (comp_format == "ans") {
//       //DLOG(ERROR, "Create ans manager...\n");
//       manager = std::make_shared<ANSManager>(chunk_size, stream, gpu_num, NoComputeNoVerify);
//     }

//     //uint8_t* testPtr = 0;
//     //size_t testSize = 102400;
//     //cudaMalloc(&testPtr, testSize);
//     //auto compress_config = manager->configure_compression(102400);
//     //size_t comp_out_bytes = compress_config.max_compressed_buffer_size;

//     auto compress_config = manager->configure_compression(size);
//     size_t comp_out_bytes = compress_config.max_compressed_buffer_size;
//     printf("init out bytes: %ld\n", comp_out_bytes);
//     manager->compress((uint8_t*)uncompressedBuf, /*d_comp_out*/(uint8_t*)compressedBuf, compress_config);
//     DLOG(ERROR, "finish compress\n");
//     cudaStreamSynchronize(stream);
//     DLOG(ERROR, "finish stream sync\n");
//     comp_out_bytes = manager->get_compressed_output_size(/*d_comp_out*/(uint8_t*)compressedBuf);
//     printf("real out bytes: %ld\n", comp_out_bytes);
//     printf("orignal size:%ld, after compressed size:%ld\n", size, comp_out_bytes);
//     compressedSize = comp_out_bytes;
//     cudaStreamDestroy(stream);

//     RETURN_TO_UPPER_HALF();
//     DLOG(ERROR, "return to upper halt\n");
// }
// #endif
