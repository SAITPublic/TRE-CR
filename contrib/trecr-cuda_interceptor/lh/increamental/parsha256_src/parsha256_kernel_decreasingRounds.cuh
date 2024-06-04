//
// Created by neville on 15.11.20.
//

#ifndef SHA_ON_GPU_PARSHA256_KERNEL_DECREASINGROUNDS_CUH
#define SHA_ON_GPU_PARSHA256_KERNEL_DECREASINGROUNDS_CUH

#include "parsha256_sha256.h"

__global__ void parsha256_kernel_gpu_decreasingRound(int *__restrict__ input_message, int *__restrict__ buf_read, int *__restrict__ buf_write) {

    const int id = threadIdx.x + blockIdx.x * blockDim.x; // My id

    const int child1 = 2 * id;
    const int child2 = 2 * id + 1;

    int total_threads = blockDim.x * gridDim.x; // total thread in grid


    int *buf_write_me = buf_write + id * 8; // My Buffer where to write
    int *buf_read_me = buf_read + id * 8; // My Buffer where to write


    int *buf_read_child1 = buf_read + child1 * 8; // child1 bufffer, where to read
    int *buf_read_child2 = buf_read + child2 * 8; // child2 bufffer, where to read



    int *buffer1_read;
    int *buffer2_read;
    int *buffer3_read;


    if (id < total_threads / 2) { // Compute result

        input_message += 8 * id;

        buffer1_read = buf_read_child1;
        buffer2_read = buf_read_child2;
        buffer3_read = input_message;


        parsha256_sha256(buffer1_read, buffer2_read, buffer3_read, buf_write_me);


    } else { // Copy buffers

#pragma unroll
        for (int j = 0; j < 8; j++) {
            buf_write_me[j] = buf_read_me[j];
        }

    }
}

#endif //SHA_ON_GPU_PARSHA256_KERNEL_DECREASINGROUNDS_CUH
