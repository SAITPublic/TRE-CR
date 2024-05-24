#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <set>
#include <iostream>
#include <unordered_map>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>
using namespace std;

struct BlockPool;
struct Block;

#define CHECK(call)                                     \
    {                                                   \
        const CUresult error_code = call;               \
        if (error_code != CUDA_SUCCESS)                 \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            exit(1);                                    \
        }                                               \
    }

constexpr size_t MinBlockSize = 512; // samll block 512B的最小block size

constexpr size_t SmallSize = 1048576; // 1M大小

constexpr size_t SmallBlockSize = 2 * 1048576; // 1M大小

constexpr int gpu_num = 8;

enum block_type
{
    SMALL,
    BIG,
    HOST,
    IPC
};
struct Block
{
    int device;  // gpu
    size_t size; // block size in bytes
    BlockPool *pool{nullptr};
    void *ptr;      // memory address
    bool allocated; // in-use flag
    int type;
    bool is_start; // used for small block default value false
    Block *prev;   // prev block if split from a larger allocation
    Block *next;   // next block if split from a larger allocation
    Block(int device, size_t size, void *ptr, int type, bool allocated, bool is_start = false)
        : device(device),
          size(size),
          ptr(ptr),
          allocated(allocated),
          type(type),
          is_start(is_start),
          prev(nullptr),
          next(nullptr) {}

    Block(int device, size_t size)
        : device(device),
          size(size) {}
};

typedef bool (*Comparison)(const Block *, const Block *);

struct BlockPool
{
    BlockPool(Comparison comparator) : blocks(comparator) {}
    BlockPool() {}
    std::set<Block *, Comparison> blocks;
};

static bool BlockComparator(const Block *a, const Block *b)
{
    if (a->device != b->device)
    {
        return true;
    }
    if (a->size != b->size)
    {
        return a->size < b->size;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct AllocParams
{
    AllocParams(
        int device,
        size_t size,
        BlockPool *pool,
        size_t alloc_size,
        int type)
        : search_key(device, size),
          pool(pool),
          alloc_size(alloc_size),
          block(nullptr),
          type(type) {}

    Block search_key;
    BlockPool *pool;
    size_t alloc_size;
    Block *block;
    int type;
};

struct Block_info
{
    Block *start_block = nullptr;
    Block *tail_block = nullptr;
    size_t reserved_memory = 0;
    CUdeviceptr reserved_start_ptr;
    CUdeviceptr reserved_end_ptr;
    CUdeviceptr current_ptr;
};
class Allocator
{
public:
    BlockPool blocks_pool_free[gpu_num];

    BlockPool small_blocks_pool_free;

    BlockPool host_blocks_pool_free;
    // BlockPool blocks_pool_used;
    unordered_map<Block *, CUmemGenericAllocationHandle> block_handle_map;
    unordered_map<CUmemGenericAllocationHandle, size_t> handle_size_map;
    // by tian01.liu from unordered to ordered map
    // unordered_map<void *, Block *> ptr_block_map;
    unordered_map<void*, void*> ipc_ptr_map;
    map<void *, Block *> ptr_block_map;
    // by tian01.liu for multi-threads
    unordered_map<void *, CUcontext> ptr_ctx_map;

    Block *host_start_block;

    void *host_start_address; // start address for pinned host start address.

    Block_info big_block_info[gpu_num];

    Block_info small_block_info;

    size_t total_allocated_memory = 0;

    CUcontext ctx;

    size_t granularity;

    CUmemAllocationProp prop = {};

    CUmemAccessDesc accessDesc;

    bool is_gpu_inited;

    Allocator();

    bool init_gpu_memory_pool(void *requested_big, void *requested_small);
    //
    void *malloc_gpu(size_t size);

    void *malloc_ipc(CUmemGenericAllocationHandle ipc_handle, size_t size, int device, size_t ipc_offset);

    void free_ipc(void* ptr);

    CUcontext get_ptr_ctx(void *devPtr);

    void malloc_restart(void *address, size_t size, int device_id = 0, bool isIpc = false);

    void free_gpu(void *ptr);
    // 根据粒度调整实际大小

    void check();

    bool init_pin_memory_pool();

    int alloc_host(void **ptr, size_t size);

    int free_host(void *ptr);
    CUmemGenericAllocationHandle get_allocation_handle(void* devPtr, size_t* offset);

    void update_pool_status(void *ptr, size_t size);

private:
    void init_small(size_t reserve_size, Block_info &block_info, CUdeviceptr requested);

    void init_big(size_t reserve_size, Block_info* block_info, CUdeviceptr requested);

    void free_(void *ptr);

    size_t round_size(size_t size, bool is_small = false);

    // 获得free block中的最匹配block
    bool get_free_block(Block *&block, AllocParams params);
    // 申请一个大小适当的block
    bool alloc_block(Block *&block, AllocParams params);

    void map_handle(CUdeviceptr ptr, size_t alloc_size, Block *block, int device_id = 0);

    Block *split_block_left_allocate(Block *left, AllocParams params);

    Block *split_block(Block *left, AllocParams params);

    void merge_block(Block *left, Block *right);

    Block *alloc_new_block(size_t size, Block_info &block_info, AllocParams params, int devic_id = 0);

    void printHandleMap();
};
