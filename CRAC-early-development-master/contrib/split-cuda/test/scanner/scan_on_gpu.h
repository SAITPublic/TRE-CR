#ifndef SCAN_ON_GPU_H
#define SCAN_ON_GPU_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void copy_handles_to_gpu(uint64_t* handles_buff, size_t num);
void free_handles_on_gpu();
uint64_t* scan_on_gpu(char *ori_buff,
                      size_t ori_len,
                      uint64_t orig_addr,
                      size_t *phandle_num);

#endif //SCAN_ON_GPU_H