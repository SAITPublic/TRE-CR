#include "MMU.h"
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "../lh_wrappers/mmap-wrapper.h"
// struct Block;
// struct BlockPool;

int pagesize = sysconf(_SC_PAGESIZE);

Allocator::Allocator() : small_blocks_pool_free(BlockComparator), host_blocks_pool_free(BlockComparator)
{
    for (int i = 0; i < gpu_num; i++) {
        blocks_pool_free[i] = BlockPool(BlockComparator);
    }
    is_gpu_inited = false;
    init_pin_memory_pool();
}

bool Allocator::init_gpu_memory_pool(void *requested_big, void *requested_small)
{
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)0;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.allocFlags.compressionType = 1;
    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK(cuInit(0));
    CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t big_size = 1024ULL * 1024 * 1024 * 80; // for multi-gpu
    size_t small_size = 1024ULL * 1024 * 1024 * 80;

    init_big(big_size, big_block_info, (CUdeviceptr)requested_big);
    init_small(small_size, small_block_info, (CUdeviceptr)requested_small);
    is_gpu_inited = true;
    return true;
}

void Allocator::init_small(size_t reserve_size, Block_info &block_info, CUdeviceptr requested)
{
    block_info.start_block = new Block(0, 0, nullptr, BIG, true, true);
    block_info.tail_block = block_info.start_block;
    block_info.reserved_memory = round_size(reserve_size);
    CHECK(cuMemAddressReserve(&block_info.reserved_start_ptr, block_info.reserved_memory, 0, requested, 0ULL));
    block_info.current_ptr = block_info.reserved_start_ptr;
    block_info.reserved_end_ptr = block_info.reserved_start_ptr + block_info.reserved_memory;
    block_info.start_block->ptr = (void *)block_info.reserved_start_ptr;
}

void Allocator::init_big(size_t reserve_size, Block_info* block_info, CUdeviceptr requested)
{
    size_t reserved_memory = round_size(reserve_size);
    CUdeviceptr current_ptr;
    CHECK(cuMemAddressReserve(&block_info[0].reserved_start_ptr, reserved_memory * gpu_num, 0, requested, 0ULL));

    current_ptr = block_info[0].reserved_start_ptr;
    for (int i = 0; i < gpu_num; i++) {
        block_info[i].start_block = new Block(0, 0, nullptr, BIG, true, true);
        block_info[i].tail_block = block_info[i].start_block;
        block_info[i].reserved_memory = reserved_memory;
        block_info[i].current_ptr = current_ptr;
        block_info[i].reserved_start_ptr = current_ptr;
        block_info[i].reserved_end_ptr = block_info[i].reserved_start_ptr + block_info[i].reserved_memory;
        current_ptr = block_info[i].reserved_end_ptr;
        block_info[i].start_block->ptr = (void *)block_info[i].reserved_start_ptr;
    }
}

void *Allocator::malloc_gpu(size_t size)
{
    Block *block;
    int type = BIG; // BIG SMALL or HOST
    int device = 0;
    cudaGetDevice(&device);
    auto &pool = (size <= SmallSize) ? small_blocks_pool_free : blocks_pool_free[device];
    if (size <= SmallSize) type = SMALL;
    size = round_size(size, size <= SmallSize);
    AllocParams params(device, size, &pool, size, type);
    bool block_found = get_free_block(block, params) || alloc_block(block, params);
    if (block_found == false)
    {
        cout << "malloc failure" << endl;
        exit(EXIT_FAILURE);
    }
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);
    ptr_ctx_map[block->ptr] = ctx;
    return block->ptr;
}

CUcontext Allocator::get_ptr_ctx(void *devPtr)
{
    if (ptr_ctx_map.empty()) {
        return 0;
    }
    auto itFind = ptr_block_map.lower_bound(devPtr);
    if (itFind == ptr_block_map.end()) {
        return 0;
    }
    if (itFind->first == devPtr)
        return ptr_ctx_map[devPtr];
    else
    {
        if (itFind == ptr_block_map.begin())
            return NULL;
        itFind--;
        void *ptr = itFind->first;
        Block* blkInner = itFind->second;
        if (devPtr > ptr && devPtr < (char*)ptr + blkInner->size)
        {
            return ptr_ctx_map[ptr];
        }       
    }
    return NULL;
}
bool Allocator::get_free_block(Block *&block, AllocParams params)
{
    auto &info = (params.type == SMALL) ? small_block_info : big_block_info[params.search_key.device];
    auto it = params.pool->blocks.lower_bound(&params.search_key);
    if (it == params.pool->blocks.end())
    {
        return false;
    }
    Block *find_block = *it;
    params.pool->blocks.erase(find_block);
    if (find_block->size > params.alloc_size)
    {
        Block *new_block = split_block_left_allocate(find_block, params);
        if (params.type != HOST && new_block->next == nullptr)
        {
            info.tail_block = new_block;
        }
    }
    if (params.type == BIG)
        map_handle((CUdeviceptr)(find_block->ptr), params.alloc_size, find_block, params.search_key.device);
    find_block->allocated = true;
    find_block->pool = nullptr;
    find_block->device = params.search_key.device;
    block = find_block;
    return true;
}

bool Allocator::alloc_block(Block *&block, AllocParams params)
{
    size_t alloc_size = (params.type == SMALL) ? SmallBlockSize : params.alloc_size;

    auto &info = (params.type == SMALL) ? small_block_info : big_block_info[params.search_key.device];

    block = alloc_new_block(alloc_size, info, params, params.search_key.device);
    block->device = params.search_key.device;
    if (params.type == SMALL)
        split_block_left_allocate(block, params);
    return true;
}

// 申请新的block
Block *Allocator::alloc_new_block(size_t size, Block_info &block_info, AllocParams params, int devic_id)
{
    // bool cond = (params.type==SMALL) ? true : false;
    Block *block = new Block(devic_id, size, (void *)block_info.current_ptr, params.type, true, params.type == SMALL);
    // block->is_start = cond;
    if (params.type != IPC)
        map_handle(block_info.current_ptr, size, block, devic_id);

    block->prev = block_info.tail_block;
    block_info.tail_block->next = block;
    block_info.tail_block = block;
    ptr_block_map[block->ptr] = block;
    block_info.current_ptr += size;
    return block;
}

void Allocator::free_gpu(void *ptr)
{
    free_(ptr);
}

bool Allocator::init_pin_memory_pool()
{
    host_start_block = new Block(0, 0, nullptr, HOST, true, true);
    host_start_block->is_start = true;
    // 主机端申请
    size_t total_size = 1024ULL * 1024 * 1024 * 4;
    total_size = ((total_size + pagesize - 1) & ~(pagesize - 1));
    void *addr = mmap(NULL, pagesize + total_size + pagesize, PROT_READ | PROT_WRITE,
                      MAP_ANONYMOUS | MAP_SHARED, 0, 0);

    mprotect(addr, pagesize, PROT_EXEC);
    host_start_address = (void *)((uint8_t *)addr + pagesize);
    mprotect((void *)((uint8_t *)host_start_address + total_size), pagesize, PROT_EXEC);

    Block *first_block = new Block(0, 1024ULL * 1024 * 1024 * 4, host_start_address, HOST, false, false);
    host_start_block->next = first_block;
    first_block->prev = host_start_block;
    host_blocks_pool_free.blocks.insert(first_block);
    first_block->pool = &host_blocks_pool_free;
    ptr_block_map[first_block->ptr] = first_block;
    if (host_start_address == NULL)
    {
        printf("Error allocating pinned host memory\n");
        exit(EXIT_FAILURE);
    }
    return true;
}

// find,split
int Allocator::alloc_host(void **ptr, size_t size)
{
    Block *block;
    size_t real_size = round_size(size, true);
    AllocParams param(0, real_size, &host_blocks_pool_free, real_size, HOST);
    get_free_block(block, param);
    *ptr = block->ptr;
    return 0;
}

int Allocator::free_host(void *ptr)
{
    free_(ptr);
    return 0;
}

// small big host free的统一处理逻辑
void Allocator::free_(void *ptr)
{
    Block *block = ptr_block_map[ptr];
    /* 
      if(block == nullptr){
        free(ptr);
        return;
    }*/
    BlockPool *pool;
    int block_type = block->type;

    if (block_type == BIG ||  block_type == IPC)
    {
        CHECK(cuMemUnmap(CUdeviceptr(block->ptr), block->size));
        CHECK(cuMemRelease(block_handle_map[block]));
        pool = &blocks_pool_free[block->device];
        handle_size_map.erase(block_handle_map[block]);
        block_handle_map.erase(block);
    }
    else if (block_type == SMALL)
        pool = &small_blocks_pool_free;
    else
        pool = &host_blocks_pool_free;

    if (block->next != nullptr && block->next->allocated == false && block->next->is_start == false)
    {
        Block *tmp_block = block->next;
        pool->blocks.erase(tmp_block);
        merge_block(block, tmp_block);
        delete (tmp_block);
    }
    if (block->is_start == false && block->prev != nullptr && block->prev->allocated == false)
    {
        Block *tmp_block = block;
        block = block->prev;
        pool->blocks.erase(block);
        merge_block(block, tmp_block);
        delete (tmp_block);
    }
    if (block_type != HOST && block->next == NULL)
    {
        auto &info = (block_type == SMALL) ? small_block_info : big_block_info[block->device];
        info.tail_block = block;
    }
    block->allocated = false;
    pool->blocks.insert(block);
    block->pool = pool;
}

// just split block did nothing else
Block *Allocator::split_block(Block *left, AllocParams params)
{
    Block *right = new Block(params.search_key.device, 0, nullptr, params.type, false);
    right->ptr = (void *)((unsigned long long)left->ptr + params.alloc_size);
    right->size = left->size - params.alloc_size;
    left->size = params.alloc_size;
    right->next = left->next;
    right->prev = left;
    if (left->next != nullptr)
    {
        left->next->prev = right;
    }
    left->next = right;
    ptr_block_map[right->ptr] = right;
    return right;
}

// split left allocate
Block *Allocator::split_block_left_allocate(Block *left, AllocParams params)
{
    left->allocated = true;
    left->pool = nullptr;
    Block *right = split_block(left, params);
    params.pool->blocks.insert(right);
    right->pool = params.pool;
    return right;
}

void Allocator::merge_block(Block *left, Block *right)
{
    left->size += right->size;
    left->next = right->next;
    if (right->next != nullptr)
        right->next->prev = left;
    ptr_block_map.erase(right->ptr);
}

void Allocator::map_handle(CUdeviceptr ptr, size_t alloc_size, Block *block, int device_id)
{
    CUmemGenericAllocationHandle handle;
    prop.location.id = device_id;
    accessDesc.location = prop.location;
    CHECK(cuMemCreate(&handle, alloc_size, &prop, 0));
    CHECK(cuMemMap(ptr, alloc_size, 0ULL, handle, 0ULL));
    int deviceCnt = 0;
    CHECK(cuDeviceGetCount(&deviceCnt));

    for (int i = 0; i < deviceCnt; i++) {
	    accessDesc.location.id = device_id;
    	CHECK(cuMemSetAccess(ptr, alloc_size, &accessDesc, 1));
    }
    block_handle_map[block] = handle;
    handle_size_map[handle] = alloc_size;
}

size_t Allocator::round_size(size_t size, bool is_small)
{
    size_t gra = (is_small) ? MinBlockSize : granularity;
    return (gra * ((size + gra - 1) / gra));
}

void Allocator::malloc_restart(void *address, size_t size, int device_id, bool isIpc)
{

    int type = (size <= SmallSize) ? SMALL : BIG;
 
    if (isIpc)
        type = IPC;

    auto &pool = (size <= SmallSize) ? small_blocks_pool_free : blocks_pool_free[device_id];
    auto &info = (size <= SmallSize) ? small_block_info : big_block_info[device_id];
    size_t real_size = (size <= SmallSize) ? round_size(size, true) : round_size(size);
    size_t alloc_size = (size <= SmallSize) ? SmallBlockSize : ((unsigned long long)address - (unsigned long long)info.current_ptr);
    AllocParams params(device_id, real_size, &pool, real_size, type);
    Block *new_block;
    int device = 0;
    cudaSetDevice(device_id);
    cudaGetDevice(&device);
    CUcontext context;
    cuCtxGetCurrent(&context);
    // printf("In malloc restart, address:%p, context:%p, device_id:%i\n", address, context, device_id);
    ptr_ctx_map[address] = context;

    if (type == BIG || type == IPC)
    {
        void *ptr = nullptr;
        if (address > (void *)info.current_ptr)
        {
            new_block = alloc_new_block(alloc_size, info, params, device_id);
            ptr = new_block->ptr;
        }
        new_block = alloc_new_block(real_size, info, params, device_id);
        new_block->allocated = true;
        if (ptr != 0)
        {
            free_(ptr);
        }
    }
    else
    {
        while (address >= (void *)info.current_ptr)
        {
            new_block = alloc_new_block(alloc_size, info, params, device_id);
            pool.blocks.insert(new_block);
            new_block->allocated = false;
        }
        Block *p = info.start_block->next;
        while (p->next != nullptr && p->next->ptr != nullptr && ((void *)((unsigned long long)p->next->ptr)) <= address)
        {
            p = p->next;
        }
        pool.blocks.erase(p);
        if (p->ptr != address)
        {
            // Block* Allocator::split_block(Block* left, AllocParams params)
            AllocParams tmp_params(device_id, real_size, &pool, (size_t)address - (size_t)p->ptr, type);
            Block *right = split_block(p, tmp_params);
            pool.blocks.insert(p);
            p = right;
        }
        if (p->size != real_size)
        {
            split_block_left_allocate(p, params);
        }
    }
}

void Allocator::update_pool_status(void *ptr, size_t size)
{
    Block *p = host_start_block->next;
    size_t real_size = round_size(size, true);
    AllocParams params(0, real_size, &host_blocks_pool_free, real_size, HOST);
    while (p->next != nullptr && p->next->ptr <= ptr)
    {
        p = p->next;
    }
    host_blocks_pool_free.blocks.erase(p);
    if (p->ptr != ptr)
    {
        p->allocated = false;
        AllocParams tmp_params(0, real_size, &host_blocks_pool_free, (unsigned long long)ptr - (unsigned long long)p->ptr, HOST);
        split_block(p, tmp_params);
        host_blocks_pool_free.blocks.insert(p);
        p = p->next;
    }
    if (p->size != real_size)
    {
        split_block_left_allocate(p, params);
    }
    p->allocated = true;
}


void *Allocator::malloc_ipc(CUmemGenericAllocationHandle ipc_handle, size_t size, int device, size_t ipc_offset)
{
    Block *block;
    int type = IPC; // BIG SMALL or HOST
    // cudaGetDevice(&device);
    auto &pool = blocks_pool_free[device];

    size = round_size(size, false);

    AllocParams params(device, size, &pool, size, type);
    bool block_found = get_free_block(block, params) || alloc_block(block, params);
    if (block_found == false)
    {
        cout << "malloc failure" << endl;
        exit(EXIT_FAILURE);
    }
    prop.location.id = device;
    accessDesc.location = prop.location;
    // printf("ptr:%p, size:%ld, handle:%lld\n", block->ptr, size, ipc_handle);
    CHECK(cuMemMap((CUdeviceptr)(block->ptr), size, 0ULL, ipc_handle, 0ULL));
    int deviceCnt = 1;
    CHECK(cuDeviceGetCount(&deviceCnt));

    for (int i = 0; i < deviceCnt; i++)
    {
        accessDesc.location.id = i;
        CHECK(cuMemSetAccess((CUdeviceptr)(block->ptr), size, &accessDesc, 1));
    }

    handle_size_map[ipc_handle] = size;
    block_handle_map[block] = ipc_handle;
    // return block->ptr;
    if (ipc_offset == 0) {
        ipc_ptr_map[block->ptr] = block->ptr;
        return block->ptr;
    } else {
        void* actual_ptr = (void*) ((size_t)block->ptr + ipc_offset);
        ipc_ptr_map[actual_ptr] = block->ptr;
        return actual_ptr;
    }
}

void Allocator::free_ipc(void *ptr) {
    free_(ipc_ptr_map[ptr]);
    ipc_ptr_map.erase(ptr);
}


void Allocator::printHandleMap()
{
    printf("block handle map items:\n");
    auto it = block_handle_map.begin();
    for (; it != block_handle_map.end(); it++)
    {
        printf("  block_ptr:%p  handle:%lld\n", it->first, it->second);
    }
    fflush(stdout);
}

CUmemGenericAllocationHandle Allocator::get_allocation_handle(void* devPtr, size_t* off)
{
    if (!ptr_block_map.count(devPtr))
    {
        printf("ptr_block_map not found!\n");
        fflush(stdout);
        return 0;
    }

    // printHandleMap();

    Block *block = ptr_block_map[devPtr];
    // printf("block ptr_0:%p, block->is_start:%i\n", block, block->is_start);

    if (block->size > SmallSize)
    {
        *off = 0;
    }
    else {
        void* ptr = block->ptr;
        void* start_ptr = NULL;
        while (block->is_start == false) {
            block = block->prev;
        }
        start_ptr = block->ptr;
        *off = (size_t)ptr-(size_t)start_ptr;
    }

    return block_handle_map[block];
}

