#include<stdio.h>
#include<stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <string.h>
#include <sys/mman.h>
#include <iostream>

#include<list>
#include<vector>

#include "procmapsarea.h"
#include "mtcp_header.h"
#include "scanner.h"
#include "scan_on_gpu.h"

using namespace std;

// #define SCAN_ON_CPU 1 // Swtich mode of scanning between CPU and GPU

// Global variables
#ifdef SCAN_ON_CPU
list<uint64_t> gHandleList;
vector<HandleInfo> handleInfos;
#else
uint64_t* g_handle_buff;
size_t g_handle_num;
#endif

MtcpHeader mtcpHdr;

int hand_info_fd = -1;

static void printUsage()
{
    fprintf(stdout, "Usage: ./scanner absolute_path/ckpt_rank_i/ckpt_kernel-loader.exe_xxxx.dmtcp"
                    "absolute_path/uhHandle_vpid\n");
}

int openFile(char *path, int flags)
{
    // Open ckpt file
    int ckpt_fd = open(path, flags);
    return ckpt_fd;
}

size_t readAll(int fd, void *buf, size_t count)
{
    char *ptr = (char *)buf;
    size_t num_read = 0;

    for (num_read = 0; num_read < count;)
    {
        ssize_t rc = read(fd, ptr + num_read, count - num_read);
        if (rc == -1)
        {
            if (errno == EINTR || errno == EAGAIN)
            {
                continue;
            } else {
                return -1;
            }
        }
        else if (rc == 0)
        {
            break;
        }
        else
        { // else rc > 0
            num_read += rc;
        }
    }

    return num_read;
}

#ifdef  SCAN_ON_CPU
// -1: open handle file failed
int openHandleFile(char *path)
{
    if (!path) return -1;
    int rc;
    size_t handle_num;
    uint64_t handle;

    // Open CU handle file
    int fd = open(path, O_RDONLY);
    rc = read(fd, &handle_num, sizeof(size_t));
    if (rc != sizeof(size_t)) return -1;

    for (size_t i = 0; i < handle_num; i++)
    {
        rc = read(fd, &handle, sizeof(uint64_t));
        if (rc != sizeof(uint64_t)) return -1;
        // fprintf(stdout, "handle : %lu\n", handle);
        gHandleList.push_back(handle);
    }

    close(fd);
    // Succeed
    return 0;
}

void writeHanleInfoToFile(size_t size)
{
    size_t rc = 0;
    HandleInfo hand_info;
    /**
     * @brief Write handle info to file, and the format in file is:
     *        orig_handle_1:address_1, orig_handle_2:address_2, ......
     */
    for (vector<HandleInfo>::iterator it = handleInfos.begin(); it != handleInfos.end(); it++)
    {
        rc = write(hand_info_fd, &(*it), sizeof(HandleInfo));
        if (rc != sizeof(HandleInfo))
        {
            fprintf(stderr, "Write handle info failed.\n");
            exit(-1);
        }
    }
}

// Scan handle location on host(CPU) side
void findHandle(char *addr, size_t size, uint64_t orig_addr)
{
    size_t chars_read = 0, handle_size = 0;
    uint64_t data;
    HandleInfo handle_info;
    // fprintf(stderr, "[XB] addr = %p, size = 0x%zx, orig_addr = 0x%zx\n", addr, size, orig_addr);
    handleInfos.clear();
    while (chars_read < size)
    {
        memcpy(&data, addr + chars_read, sizeof(uint64_t));
        for (std::list<uint64_t>::iterator it = gHandleList.begin(); it != gHandleList.end(); it++)
        {
            if (data == *it)
            {
                // fprintf(stderr, "Find the handle : 0x%zx, offset : %zu\n", data, chars_read);
                handle_info.value = data; // original handle
                handle_info.location = orig_addr + chars_read; // address (VA)
                // fprintf(stderr, "SCAN OK : 0x%zx, value : 0x%zx\n", handle_info.location, data);
                handleInfos.push_back(handle_info);
                break;
            }
        }
        chars_read += sizeof(uint64_t);
    }

    handle_size = handleInfos.size(); 
    if (handle_size > 0)
    {
        writeHanleInfoToFile(handle_size);
    }
}
#else // Scan on gpu
// -1: open handle file failed
int openHandleFile(char *path)
{
    if (!path) return -1;
    int rc;
    uint64_t handle;

    // Open CU handle file
    int fd = open(path, O_RDONLY);
    rc = read(fd, &g_handle_num, sizeof(size_t));
    if (rc != sizeof(size_t)) return -1;

    g_handle_buff = (uint64_t*)malloc(g_handle_num * sizeof(uint64_t));
    for (size_t i = 0; i < g_handle_num; i++)
    {
        rc = read(fd, &handle, sizeof(uint64_t));
        if (rc != sizeof(uint64_t)) return -1;
        // fprintf(stdout, "handle : %lu\n", handle);
        *(g_handle_buff + i) = handle;
    }

    close(fd);
    // Succeed
    return 0;
}

void freeHandles()
{
    free(g_handle_buff);
}

void writeHanleInfoToFile(uint64_t *handles, size_t num)
{
    fprintf(stderr, "Write handles to file, number : %lu\n", num);
    size_t rc = 0;
    /**
     * @brief Write handle info to file, and the format in file is:
     *        orig_handle_1:address_1, orig_handle_2:address_2, ......
     */
    for (int i = 0; i < num; i++)
    {
        rc = write(hand_info_fd, &(*(handles + 2 * i)), sizeof(uint64_t));
        if (rc != sizeof(uint64_t))
        {
            fprintf(stderr, "Write handle value failed.\n");
            exit(-1);
        }

        rc = write(hand_info_fd, &(*(handles + 2 * i + 1)), sizeof(uint64_t));
        if (rc != sizeof(uint64_t))
        {
            fprintf(stderr, "Write handle location failed.\n");
            exit(-1);
        }
        // fprintf(stderr, "SCAN OK : 0x%zx, value : 0x%zx\n", *(handles + 2 * i), *(handles + 2 * i + 1));
    }
}

// Scan handle location on device(GPU) side
void findHandle(char *addr, size_t size, uint64_t orig_addr)
{
    // fprintf(stderr, "[XB] addr = %p, size = 0x%zx\n", addr, size);
    size_t handle_num = 0;
    uint64_t *orig_handles = NULL;
    // fprintf(stderr, "Scan on gpu ... addr: %p, size: %lu, orig_addr: 0x%zx\n", addr, size, orig_addr);
    // Find location in GPU side, and store handles in the orig_handles buffer
    orig_handles = scan_on_gpu(addr, size, orig_addr, &handle_num);
    fprintf(stderr, "Scan on gpu finish for current region.\n");
    // Write found handles to file
    if (orig_handles != NULL && handle_num > 0)
    {
        writeHanleInfoToFile(orig_handles, handle_num);
    }

    free(orig_handles);
}
#endif

int scanMemoryRegion(int ckptfd, Area* area)
{
    size_t bytes = 0;
    void * addr;
    //TODO
    // Heap of UH
    // fprintf(stderr, "[XB] addr0 = %p, size = 0x%zx, flags = %d\n", area->addr, area->size, area->flags);
    if (area->flags & MAP_SHARED)
    {
        area->flags = area->flags ^ MAP_SHARED;
        area->flags = area->flags | MAP_PRIVATE | MAP_ANONYMOUS;
    }

    // Anonymous memory region
    if ((area->properties & 0x0001) != 0)
    {
        addr = mmap(area->addr, area->size,
                    area->prot,
                    area->flags | MAP_FIXED, -1, 0);

        if (addr != area->addr)
        {
            fprintf(stderr, "mapping %zx bytes at %p\n", area->size, area->addr);
            exit(-1);
        }
        munmap(addr, area->size);
    }
    // Text segment, data segment and anonmyous memory regions
    else if (((area->addr < TEXT_SEGMENT_OF_KERNELLOADER) && (area->addr >= TEXT_SEGMENT_OF_APP))
         || area->flags & MAP_ANONYMOUS)
    {
        addr = mmap(0, area->size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (addr == MAP_FAILED)
        {
            fprintf(stderr, "Mapping failed, size: 0x%zx. Error: %s\n", area->size, strerror(errno));
            exit(-1);
        }
        // Read in the data
        bytes = readAll(ckptfd, addr, area->size);
        if (bytes < area->size)
        {
            fprintf(stderr, "Read failed for memory region (%s) at: %p of: %zu bytes. "
                "Error: %s\n", area->name, addr, area->size, strerror(errno));
            return -1;        
        }
        // Find the offset of handle in the ckpt file
        findHandle((char *)addr, area->size, (uint64_t)(area->addr));
        munmap(addr, area->size);
    }
    return 0;
}

static int
readMtcpHeader(int ckptFd)
{
  int rc = -1;
  fprintf(stdout, "Read mtcp header\n");
  // This assumes that the MTCP header signature is unique.
  // We repeatedly look for mtcpHdr because the first header will be
  // for DMTCP.  So, we look deeper for the MTCP header.  The MTCP
  // header is guaranteed to start on an offset that's an integer
  // multiple of sizeof(mtcpHdr), which is currently 4096 bytes.
  do {
    rc = readAll(ckptFd, &mtcpHdr, sizeof mtcpHdr);
  } while (rc > 0 && strcmp(mtcpHdr.signature, MTCP_SIGNATURE) != 0);
  return 1;
}

int scanMemory(int ckptfd, string pid)
{
    int rc = 0;
    Area area = {0};
    string fileName = "handle_infos_" + pid;
    hand_info_fd = open(fileName.c_str(), O_WRONLY | O_CREAT, 0644);
    if (hand_info_fd == -1)
    {
        return -1;
    }

#ifndef SCAN_ON_CPU
    // Copy handle list to GPU side
    copy_handles_to_gpu(g_handle_buff, g_handle_num);
#endif

    while (!rc && readAll(ckptfd, &area, sizeof area))
    {
        rc = scanMemoryRegion(ckptfd, &area);
    }
#ifndef SCAN_ON_CPU
    free_handles_on_gpu();
#endif
    close(hand_info_fd);
    return rc;    
}

string splitString(const string str, const char split, int idx){
    vector<string> res;
    string strs = str + split;
    int pos = strs.find(split);
    while (pos!=strs.npos)
    {
        string temp = strs.substr(0, pos);
        res.push_back(temp);
        strs = strs.substr(pos+1, strs.size());
        pos = strs.find(split);
    }
    return res[idx];
}

int main(int argc, char *argv[])
{
    clock_t startRefill, endRefill;
    printf("argc = %d\n", argc);

    if (argc < 3 || !argv[1] || !argv[2])
    {
        printUsage();
        return -1;
    }

    // get pid from ckpt filename
    string pid = splitString(argv[1], '-', 2);

    // Open ckpt file
    int ckpt_fd = openFile(argv[1], O_RDONLY);
    if (ckpt_fd == -1)
    {
        fprintf(stderr, "Open ckpt file failed.\n");
        return -1;       
    }

    int rc = openHandleFile(argv[2]);
    if (rc == -1)
    {
        fprintf(stderr, "Open handle file failed.\n");
        return -1;
    }

    readMtcpHeader(ckpt_fd);
    fprintf(stderr, "Scan begin ...\n");
    startRefill = clock();
    scanMemory(ckpt_fd, pid);
    endRefill = clock();
    fprintf(stdout,"Scan finish. time = %.6f\n", (double)(endRefill - startRefill) / CLOCKS_PER_SEC * 1000);

#ifndef SCAN_ON_CPU
    free(g_handle_buff);
#endif
    close(ckpt_fd);
    return 0;
}
