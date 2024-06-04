#ifndef __MPI_RESTORE_INFO_H__
#define __MPI_RESTORE_INFO_H__
#include <linux/limits.h>
//#include "mtcp_restart_plugin.h"
// #ifdef MTCP_PLUGIN_H
// #include MTCP_PLUGIN_H
// #else
// #define PluginInfo char
// #define mtcp_plugin_hook(args)
// #endif

/* The use of NO_OPTIMIZE is deprecated and will be removed, since we
 * compile mtcp_restart.c with the -O0 flag already.
 */
#ifdef __clang__
# define NO_OPTIMIZE __attribute__((optnone)) /* Supported only in late 2014 */
#else /* ifdef __clang__ */
# define NO_OPTIMIZE __attribute__((optimize(0)))
#endif /* ifdef __clang__ */

#define MAX_LH_REGIONS 500
typedef char *VA;  /* VA = virtual address */
typedef struct __MpiMemRange
{
  void *start;
  void *end;
} MemRange_t;

typedef struct __MpiMmapInfo
{
  void *addr;
  size_t len;
  int unmapped; 
  int guard;
} MpiMmapInfo_t;

typedef struct __MpiLhCoreRegions
{
  void *start_addr; // Start address of a LH memory segment
  void *end_addr; // End address
  int prot; // Protection flag
} LhCoreRegions_t;

/**
 * @brief Attention: please synchronize the modification to MPILowerHalfInfo_t,
 *        which is in "contrib/mpi-proxy-split/lower-half/lower_half_api.h"
 *
 */
typedef struct LowerHalfInfo
{
  void *startText; // Start address of text segment (R-X) of lower half
  void *endText;   // End address of text segmeent (R-X) of lower half
  void *endOfHeap; // Pointer to the end of heap segment of lower half
  void *libc_start_main; // Pointer to libc's __libc_start_main function in statically-linked lower half
  void *main;      // Pointer to the main() function in statically-linked lower half
  void *libc_csu_init; // Pointer to libc's __libc_csu_init() function in statically-linked lower half
  void *libc_csu_fini; // Pointer ot libc's __libc_csu_fini() function in statically-linked lower half
  void *fsaddr; // The base value of the FS register of the lower half
  uint64_t lh_AT_PHNUM; // The number of program headers (AT_PHNUM) from the auxiliary vector of the lower half
  uint64_t lh_AT_PHDR;  // The address of the program headers (AT_PHDR) from the auxiliary vector of the lower half
  void *g_appContext; // Pointer to ucontext_t of upper half application (defined in the lower half)
  void *lh_dlsym;     // Pointer to mydlsym() function in the lower half
  void *lh_pdlhandle; // Pointer to myhandle() function in the lower half
  void *getRankFptr;  // Pointer to getRank() function in the lower half
#ifdef SINGLE_CART_REORDER
  void *getCoordinatesFptr; // Pointer to getCoordinates() function in the lower half
  void *getCartesianCommunicatorFptr; // Pointer to getCartesianCommunicator() function in the lower half
#endif
  void *parentStackStart; // Address to the start of the stack of the parent process (FIXME: Not currently used anywhere)
  void *updateEnvironFptr; // Pointer to updateEnviron() function in the lower half
  void *getMmappedListFptr; // Pointer to getMmapedList() function in the lower half
  void *resetMmappedListFptr; // Pointer to resetMmapedList() function in the lower half
  int numCoreRegions; // total number of core regions in the lower half
  void *getLhRegionsListFptr; // Pointer to getLhRegionsList() function in the lower half
  MemRange_t memRange; // MemRange_t object in the lower half
} MpiLowerHalfInfo_t;

typedef struct _RestoreInfo {
  int fd;
  VA minLibsStart;
  VA maxLibsEnd;
  VA minHighMemStart;
  char* restartDir;
  int argc;
  char **argv;
  char **environ;
  
  MpiLowerHalfInfo_t pluginInfo;
  char ckptImage[PATH_MAX];
} RestoreInfo;

#endif // #ifndef __MPI_RESTORE_INFO_H__
