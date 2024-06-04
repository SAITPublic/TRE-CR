//
// Created by neville on 15.11.20.
//

#include "parsha256_sha256.h"

#ifndef SHA_ON_GPU_PARSHA256_KERNEL_firstRound_CUH
#define SHA_ON_GPU_PARSHA256_KERNEL_firstRound_CUH

// The kernel for the first round
__global__ void parsha256_kernel_gpu_firstRound(int *__restrict__ input_message, int *__restrict__ buf_write) {

    const int id = threadIdx.x + blockIdx.x * blockDim.x; // My id

    int *buf_write_me = buf_write + id * 8; // my buffer where to write


    const int *first_round_in = input_message + id * 24;
    parsha256_sha256(first_round_in, first_round_in + 8, first_round_in + 16, buf_write_me);

}

#endif //SHA_ON_GPU_PARSHA256_KERNEL_firstRound_CUH
