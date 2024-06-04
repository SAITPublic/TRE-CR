//
// Created by neville on 15.11.20.
//

#include "parsha256_sha256.h"

#ifndef SHA_ON_GPU_PARSHA256_KERNEL_singleInvocation_CUH
#define SHA_ON_GPU_PARSHA256_KERNEL_singleInvocation_CUH

// The kernel for the first round
__global__ void parsha256_kernel_gpu_singleInvocation(int *__restrict__ input_message, int *__restrict__ buf_write) {

    parsha256_sha256(input_message, input_message + 8, input_message + 16, buf_write);

}

#endif //SHA_ON_GPU_PARSHA256_KERNEL_firstRound_CUH
