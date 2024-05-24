//
// Created by nevil on 25.11.2020.
//

#ifndef SHA256_BENCH_GPU_HELPER_CUH
#define SHA256_BENCH_GPU_HELPER_CUH

__device__ __host__ __inline__ uint32_t ch(const uint32_t x, const uint32_t y, const uint32_t z) {

    return (x & y) ^ (~x & z);
}

__device__ __host__ __inline__ uint32_t maj(const uint32_t x, const uint32_t y, const uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

template<unsigned int N>
__device__ __host__ __inline__ uint32_t ROTR(const uint32_t x) {
    return (x >> N) | (x << (32 - N));
}

template<unsigned int N>
__device__ __host__ __inline__ uint32_t SHR(const uint32_t x) {
    return x >> N;
}

__device__ __host__ __inline__ uint32_t sigma0(const uint32_t x) {
    return ROTR<7>(x) ^ ROTR<18>(x) ^ SHR<3>(x);
}

__device__ __host__ __inline__ uint32_t sigma1(const uint32_t x) {
    return ROTR<17>(x) ^ ROTR<19>(x) ^ SHR<10>(x);
}

__device__ __host__ __inline__ uint32_t Sigma1(const uint32_t x) {
    return ROTR<6>(x) ^ ROTR<11>(x) ^ ROTR<25>(x);
}

__device__ __host__ __inline__ uint32_t Sigma0(const uint32_t x) {
    return ROTR<2>(x) ^ ROTR<13>(x) ^ ROTR<22>(x);
}

uint64_t delta(const uint64_t i, const uint64_t m = 256, const uint64_t n = 768, const uint64_t l = 0) {
    uint64_t pow = std::pow(2, i);
    return pow * (2 * n - 2 * m - l) - (n - 2 * m);
}

uint64_t lambda(const uint64_t i, const uint64_t m = 256, const uint64_t n = 768, const uint64_t l = 0) {
    uint64_t pow = std::pow(2, i - 1);
    return pow * (2 * n - 2 * m - l);
}




#endif //SHA256_BENCH_GPU_HELPER_CUH
