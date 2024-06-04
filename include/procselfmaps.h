#ifndef __DMTCP_PROCSELFMAPS_H__
#define __DMTCP_PROCSELFMAPS_H__

#include "jalloc.h"
#include "dmtcp.h"
#include "procmapsarea.h"

namespace dmtcp
{
class ProcSelfMaps
{
  public:
#ifdef JALIB_ALLOCATOR
    static void *operator new(size_t nbytes, void *p) { return p; }

    static void *operator new(size_t nbytes) { JALLOC_HELPER_NEW(nbytes); }

    static void operator delete(void *p) { JALLOC_HELPER_DELETE(p); }
#endif // ifdef JALIB_ALLOCATOR

    ProcSelfMaps();
    ~ProcSelfMaps();

    size_t getNumAreas() const { return numAreas; }

    int getNextArea(ProcMapsArea *area);
    const char* getData() const { return data; }

  private:
    unsigned long int readDec();
    unsigned long int readHex();
    bool isValidData();

    char *data;
    size_t dataIdx;
    size_t numAreas;
    size_t numBytes;
    int fd;
    int numAllocExpands;
};
}
#endif // #ifndef __DMTCP_PROCSELFMAPS_H__

/*
 * This callback can be used by plugins to inform the DMTCP core memory
 * checkpointing engine that the specified region of memory should be
 * skipped. The skipped memory regions are not written out to the checkpoint
 * image.
 *
 * The callback should return 1 if the region should be skipped, 0 otherwise.
 * by tian01.liu for fix the bug produced by realloc, 2023.1.20
 */
EXTERNC int dmtcp_skip_memory_region_ckpting_ex(const ProcMapsArea *, int fd, int stack_was_seen)
__attribute((weak));

/*
 * This callback can be used by plugins to inform the DMTCP core memory
 * checkpointing engine that the specified region of memory should be
 * skipped. The skipped memory regions are not written out to the checkpoint
 * image.
 *
 * The callback should return 1 if the region should be skipped, 0 otherwise.
 */
EXTERNC int dmtcp_skip_memory_region_ckpting(const ProcMapsArea *, int fd, int stack_was_seen)
__attribute((weak));
