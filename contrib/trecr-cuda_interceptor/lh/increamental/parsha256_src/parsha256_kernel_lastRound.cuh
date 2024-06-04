//
// Created by neville on 15.11.20.
//

#ifndef SHA_ON_GPU_PARSHA256_KERNEL_LASTROUND_CUH
#define SHA_ON_GPU_PARSHA256_KERNEL_LASTROUND_CUH

#include "parsha256_sha256.h"

__global__ void parsha256_kernel_gpu_lastRound(int *__restrict__ input_message, int *__restrict__ buf_read, int *__restrict__ buf_write, int *__restrict__ out, const int b, const int L) {

    const int id = threadIdx.x + blockIdx.x * blockDim.x; // My id

    const int child1 = 2 * id;
    // const int child2 = 2 * id + 1;


    int *buf_read_child1 = buf_read + child1 * 8; // child1 bufffer, where to read
    // int *buf_read_child2 = buf_read + child2 * 8; // child2 bufffer, where to read




//    if (b > 0) {
//        parsha256_sha256(buf_read_child1, buf_read_child2, input_message, out);
//
//    } else {
    // Padding
    for (int i = 0; i < 15; i++) {
        buf_read_child1[8 + i] = 0;
    }
    buf_read_child1[23] = L;
    parsha256_sha256(buf_read_child1, buf_read_child1 + 8, buf_read_child1 + 16, out);
//    }
}

#endif //SHA_ON_GPU_PARSHA256_KERNEL_LASTROUND_CUH
