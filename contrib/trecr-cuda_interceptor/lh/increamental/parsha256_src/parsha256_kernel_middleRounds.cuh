//
// Created by neville on 15.11.20.
//

#ifndef SHA_ON_GPU_PARSHA256_KERNEL_MIDDLEROUNDS_CUH
#define SHA_ON_GPU_PARSHA256_KERNEL_MIDDLEROUNDS_CUH

#include "parsha256_sha256.h"

__global__ void parsha256_kernel_gpu_middleRound(int *__restrict__ input_message, int *__restrict__ buf_read, int *__restrict__ buf_write) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x; // My id
    const int child1 = 2 * id;
    const int child2 = 2 * id + 1;
    int total_threads = blockDim.x * gridDim.x; // total thread in grid
    int *buf_write_me = buf_write + id * 8; // My Buffer where to write
    int *buf_read_child1 = buf_read + child1 * 8; // child1 bufffer, where to read
    int *buf_read_child2 = buf_read + child2 * 8; // child2 bufffer, where to read
// temp variables for later
    int *buffer1_read;
    int *buffer2_read;
    int *buffer3_read;
    if (id < total_threads / 2) {
        input_message += 8 * id; // Offset for non leaf threads
    } else {
        input_message += 8 * (total_threads / 2) + (id - (total_threads / 2)) * 24; // Offset for leaf threads
    }
    // We set the pointer variales in a if else statement and do not perform the whole sha-256 computation in an if else
    if (id < total_threads / 2) {
        buffer1_read = buf_read_child1;
        buffer2_read = buf_read_child2;
        buffer3_read = input_message;
    } else {
        buffer1_read = input_message;
        buffer2_read = input_message + 8;
        buffer3_read = input_message + 16;
    }
    parsha256_sha256(buffer1_read, buffer2_read, buffer3_read, buf_write_me);
}

#endif //SHA_ON_GPU_PARSHA256_KERNEL_MIDDLEROUNDS_CUH
