#include "increamental.h"

#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

using namespace dmtcp;

bool Increamental::use_increamental = false;

std::unordered_map<void *, std::string> Increamental::address_hash_map;

std::unordered_map<void *, bool> Increamental::address_change_map;

std::unordered_map<void *, size_t> Increamental::address_size_map;

std::set<void *> Increamental::mem_set;

std::unordered_map<void *, void *> Increamental::gpu_cpu_map;

std::string Increamental::ckpt_filename = "dmtcp_gpu_ckpt_";

bool Increamental::UpdateGpuStatus(void *address, size_t len, std::string hash)
{
    if (address_hash_map.count(address) == 0 || address_hash_map[address] != hash)
    {
        address_hash_map[address] = hash;
        address_change_map[address] = false;
        return false;
    }
    else
    {
        address_hash_map[address] = hash;
        address_change_map[address] = true;
        return true;
    }
}

void Increamental::WriteMemMaptoFile()
{
    // 清空文件
    std::ofstream file_writer("MemMap.txt", std::ios_base::out);
    file_writer.close();
    std::ofstream file;
    file.open("MemMap.txt", std::ios::app);
    for (auto mem : mem_set)
    {
        file << (unsigned long long)mem << "\n"
             << (unsigned long long)((void *)gpu_cpu_map[mem]) << "\n"
             << address_size_map[mem] << "\n";
    }
    file.close();
}

void Increamental::ReadMemMapfromFile()
{
    ifstream infile;

    infile.open("MemMap.txt", std::ios::in);
    // unsigned long long data;
    std::string buf;
    unsigned long long gpu_mem;
    unsigned long long cpu_mem;
    unsigned long long size;
    int count = 0;
    while (getline(infile, buf))
    {
        if ((count % 3) == 0)
        {
            gpu_mem = strtoull(buf.c_str(), NULL, 10);
        }
        else if ((count % 3) == 1)
        {
            cpu_mem = strtoull(buf.c_str(), NULL, 10);
        }
        else
        {
            size = strtoull(buf.c_str(), NULL, 10);
            gpu_cpu_map[(void *)gpu_mem] = VA((void *)cpu_mem);
            address_size_map[(void *)gpu_mem] = size;
        }
        count++;
    }
    infile.close();
}

int Increamental::GetCkptFile(void *addr, pid_t pid)
{
    std::string file_name = ckpt_filename + std::to_string(pid) + "_" + std::to_string((unsigned long long)addr);
    int flags = O_CREAT | O_TRUNC | O_WRONLY;
    int fd = open(file_name.c_str(), flags, 0600);
    return fd;
}

// 读取ckpt文件到内存,并建立map信息
void Increamental::ReadCkptFile(std::vector<int> &gpu_fd_v, pid_t pid)
{
    std::string to_find_file = "dmtcp_gpu_ckpt_" + std::to_string(pid) + "_";
    DIR *dir = opendir((std::string("./")).c_str());
    struct dirent *entry;
    entry = readdir(dir);

    while ((entry = readdir(dir)) != NULL)
    {
        std::string file_name = std::string(entry->d_name);
        bool ret = strstr(file_name.c_str(), to_find_file.c_str());
        if (ret == 1)
        {
            int tmp_fd = open(file_name.c_str(), O_RDONLY);
            gpu_fd_v.push_back(tmp_fd);
        }
    }
}

void Increamental::ReadCkptFile(pid_t pid)
{
    std::string to_find_file = "dmtcp_gpu_ckpt_" + std::to_string(pid) + "_";
    DIR *dir = opendir((std::string("./")).c_str());
    struct dirent *entry;
    entry = readdir(dir);

    while ((entry = readdir(dir)) != NULL)
    {
        std::string file_name = std::string(entry->d_name);
        bool ret = strstr(file_name.c_str(), to_find_file.c_str());
        if (ret == 1)
        {
            size_t size;
            void *address;
            int tmp_fd = open(file_name.c_str(), O_RDONLY);
            read(tmp_fd, &address, sizeof(address));
            read(tmp_fd, &size, sizeof(size));
            char *cpu_mem_addr = (char *)malloc(size);
            read(tmp_fd, cpu_mem_addr, size);
            // gpu_cpu_map[address] = (void*)cpu_mem_addr;
            address_size_map[address] = size;
            cuMemcpyHtoD((CUdeviceptr)address, (void *)cpu_mem_addr, size);
            free(cpu_mem_addr);
        }
    }
}
// 释放掉所有的gpu对应的cpu mem 避免被重复ckpt
void Increamental::FreeAllCpuMem()
{
    for (auto mem_pair : gpu_cpu_map)
    {
        free(mem_pair.second);
    }
    gpu_cpu_map.clear();
}
