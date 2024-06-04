#include <stdio.h>
#include <stdlib.h>
#include<cstdlib>
#include<unistd.h>
#include<sys/time.h>
#include <cuda.h>
#include <builtin_types.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "scan_on_gpu.h"

// const global variables
const static int THREAD_NUM = 1024;
const static int THREAD_NUM_PER_WRAP = 32;

// global variables
uint64_t *handles_in_gpu = NULL;
size_t orig_handle_num = 0;
unsigned long *gpu_buff = NULL;

#define ROUND_UP(len) ((unsigned long)(len + THREAD_NUM - 1) & ~(THREAD_NUM - 1))

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors( cudaError_t err, const char *file, const int line )
{
    if(err != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

__global__ void scan_handles(unsigned long *A,
                             unsigned int len1,
                             unsigned long *B,
                             unsigned int len2,
                             unsigned long *ori_addr,
                             unsigned long *handle_buff,
                             unsigned long *handle_num,
                             int32_t *lock)
{
    unsigned int ti = blockDim.x * blockIdx.x + threadIdx.x;
    /**
     * @brief Length of each block is thread_num * 8bytes, because the 
     *        length of pointer is 8 bytes, and the len1 is the length
     *        of original buffer with bytes.
     */
    unsigned int block_num = ROUND_UP(len1) / THREAD_NUM;
    unsigned int offset = 0;
    unsigned long offset_array[THREAD_NUM];
    unsigned int id = 0;
    __shared__ unsigned int offset_cnt;

    for (unsigned int block_i = 0; block_i < block_num; block_i++)
    {
        offset = ti + block_i * THREAD_NUM;
        if (offset >= len1)
        {
            // Exit the loop, if excceed the max length
            break;
        }

        for (unsigned int i = 0; i < len2; i++)
        {
            /**
             * @brief In B array, the data type is : original handle, re-created handle;
             *        so assgined re-created handle to the area which stores original handle
             */
            if(A[offset] == B[i])
            {
                offset_array[id++] = *(A + offset); // original handle value
                offset_array[id++] = *ori_addr + offset * 8; // original handle offset
                // printf("Find handle handle: %lu, offset: %lu, id = %u.\n", A[offset], *ori_addr + offset * 8, id);
                break;
            }
        }
    }

    offset_cnt = 0;
    // Wait for all threads scan finish.
    __syncthreads();
    for (int j = 0; j < THREAD_NUM_PER_WRAP; j++)
    {
        if (ti % THREAD_NUM_PER_WRAP != j)
        {
            continue;
        }

        while(atomicExch(lock, 0) == 0);
        for (int k = 0; k < id; k++)
        {
            handle_buff[offset_cnt++] = offset_array[k];
        }

        //Release the lock
        *lock = 1;
    }

    // Wait for all threads fill finish.
    __syncthreads();
    *handle_num = offset_cnt;
}

void copy_handles_to_gpu(uint64_t* handles_buff, size_t num)
{
    // fprintf(stdout,"Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    orig_handle_num = num;
    checkCudaErrors(cudaMalloc((void **)&handles_in_gpu, num * sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(handles_in_gpu, handles_buff,
                 num * sizeof(uint64_t), cudaMemcpyHostToDevice));
}

void free_handles_on_gpu()
{
    checkCudaErrors(cudaFree(handles_in_gpu));
}

uint64_t* scan_on_gpu(char *ori_data_buff, 
                      size_t ori_data_len,
                      uint64_t orig_addr,
                      size_t *phandle_num)
{
    //TicketLock
    int32_t *d_lock =         NULL;
    int32_t lock =               1;
    uint64_t *o_addr =        NULL;
    uint64_t *handle_buff_g = NULL;
    uint64_t *handle_num_g =  NULL;
    uint64_t handle_num_h =      0;
    uint64_t *handle_buff_h = NULL;

    // Copy ticket lock to device side
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lock), sizeof(int32_t)));
    checkCudaErrors(cudaMemcpy(d_lock, &lock, sizeof(int32_t), cudaMemcpyHostToDevice));

    // Copy original buffer to device side
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&gpu_buff), ori_data_len));
    checkCudaErrors(cudaMemcpy(gpu_buff, ori_data_buff, ori_data_len, cudaMemcpyHostToDevice));

    // Copy original base address of memory region to device side
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&o_addr), sizeof(uint64_t)));
    checkCudaErrors(cudaMemcpy(o_addr, &orig_addr, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Allocate result buffer which stores offset to deivce side
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&handle_buff_g), THREAD_NUM * sizeof(uint64_t)));

    // Store number of handles
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&handle_num_g), sizeof(uint64_t)));

    // Launch kernel, covert byte to uint64_t, so ori_len / sizeof(uint64_t)
    scan_handles <<< 1, THREAD_NUM >>>(gpu_buff,
                                       ori_data_len / sizeof(uint64_t), 
                                       handles_in_gpu,
                                       orig_handle_num,
                                       o_addr,
                                       handle_buff_g,
                                       handle_num_g,
                                       d_lock);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&handle_num_h, handle_num_g, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Copy the found handles from device to host
    if (handle_num_h > 0)
    {
        // fprintf(stderr, "Scan on gpu ... kernel runs finish, handle_num: %lu\n", handle_num_h / 2);
        handle_buff_h = (uint64_t*)malloc(handle_num_h * sizeof(uint64_t));
        cudaMemcpy(handle_buff_h, handle_buff_g, handle_num_h * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    // Return number of handles to output
    *phandle_num = handle_num_h / 2;
    checkCudaErrors(cudaFree(handle_num_g));
    checkCudaErrors(cudaFree(handle_buff_g));
    checkCudaErrors(cudaFree(o_addr));
    checkCudaErrors(cudaFree(gpu_buff));
    checkCudaErrors(cudaFree(d_lock));

    return handle_buff_h;
}
