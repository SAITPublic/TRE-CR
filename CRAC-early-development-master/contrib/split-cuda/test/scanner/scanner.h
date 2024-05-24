#ifndef SCANNER_H
#define SCANNER_H

#include <stdint.h>
#include "procmapsarea.h"

typedef struct __handle_info
{
    uint64_t value;  // value of CU handle
    uint64_t location; // address of CU handle in segment
} HandleInfo;

#define TEXT_SEGMENT_OF_APP  (VA)0x400000
#define TEXT_SEGMENT_OF_KERNELLOADER (VA)0x0ae00000

#endif // #ifndef SCANNER_H