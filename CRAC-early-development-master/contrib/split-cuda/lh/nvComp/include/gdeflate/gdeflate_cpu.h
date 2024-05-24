/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cstdint>

namespace gdeflate {

/**
 * @brief Perform decompression on the CPU.
 *
 * @param in_ptr The pointers on the CPU, to the compressed chunks.
 * @param batch_size The number of batch items.
 * @param out_ptr The pointers on the CPU, to where to uncompress each chunk (output).
 * @param out_bytes The pointers on the CPU to store the uncompressed sizes (output).
 *
 */
void decompressCPU(
    const void* const* in_ptr,
    size_t batch_size,
    void* const* out_ptr,
    size_t* out_bytes);

/**
 * @brief Perform compression on the CPU.
 *
 * @param in_ptr The pointers on the CPU, to uncompressed batched items.
 * @param in_bytes The size of each uncompressed batch item on the CPU.
 * @param max_chunk_size The maximum size of a chunk.
 * @param batch_size The number of batch items.
 * @param out_ptr The pointers on the CPU, to the output location for each compressed batch item (output).
 * @param out_bytes The compressed size of each chunk on the CPU (output).
 *
 */
void compressCPU(
    const void* const* in_ptr,
    const size_t* in_bytes,
    const size_t max_chunk_size,
    size_t batch_size,
    void* const* out_ptr,
    size_t* out_bytes);

} // namespace gdeflate
