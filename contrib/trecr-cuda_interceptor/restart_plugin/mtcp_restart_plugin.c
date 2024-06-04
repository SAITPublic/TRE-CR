#include <stdio.h>
#include <stdlib.h>
#include <asm/prctl.h>
#define _GNU_SOURCE // needed for MREMAP_MAYMOVE
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/auxv.h>
//#include <mtcp_util.h>
#include <linux/limits.h>
#include <fcntl.h>
#include "../lh/restore_mem/mtcp_sys.h"
#include "../utils/mtcp_util.h"
#include "../../trecr-mpi_interceptor/mana_header.h"
#include "mtcp_split_process.h"
#include "mtcp_restart_plugin.h"

# define GB (uint64_t)(1024 * 1024 * 1024)
#define ROUNDADDRUP(addr, size) ((addr + size - 1) & ~(size - 1))

NO_OPTIMIZE
char*
getCkptImageByRank(int rank, char **argv)
{
  char *fname = NULL;
  if (rank >= 0) {
    fname = argv[rank];
  }
  return fname;
}

int itoa2(int value, char* result, int base) {
	// check that the base if valid
	if (base < 2 || base > 36) { *result = '\0'; return 0; }

	char* ptr = result, *ptr1 = result, tmp_char;
	int tmp_value;

	int len = 0;
	do {
		tmp_value = value;
		value /= base;
		*ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
		len++;
	} while ( value );

	// Apply negative sign
	if (tmp_value < 0) *ptr++ = '-';
	*ptr-- = '\0';
	while(ptr1 < ptr) {
		tmp_char = *ptr;
		*ptr--= *ptr1;
		*ptr1++ = tmp_char;
	}
	return len;
}

int my_memcmp(const void *buffer1, const void *buffer2, size_t len) {
  const uint8_t *bbuf1 = (const uint8_t *) buffer1;
  const uint8_t *bbuf2 = (const uint8_t *) buffer2;
  size_t i;
  for (i = 0; i < len; ++i) {
      if(bbuf1[i] != bbuf2[i]) return bbuf1[i] - bbuf2[i];
  }
  return 0;
}

// FIXME: Many style rules broken.  Code never reviewed by skilled programmer.
int getCkptImageByDir(RestoreInfo *rinfo, char *buffer, size_t buflen, int rank) {
  if(!rinfo->restartDir) {
    MTCP_PRINTF("***ERROR No restart directory found - cannot find checkpoint image by directory!");
    return -1;
  }

  size_t len = mtcp_strlen(rinfo->restartDir);
  if(len >= buflen){
    MTCP_PRINTF("***ERROR Restart directory would overflow given buffer!");
    return -1;
  }
  mtcp_strcpy(buffer, rinfo->restartDir); // start with directory

  // ensure directory ends with /
  if(buffer[len - 1] != '/') {
    if(len + 2 > buflen){ // Make room for buffer(strlen:len) + '/' + '\0'
      MTCP_PRINTF("***ERROR Restart directory would overflow given buffer!");
      return -1;
    }
    buffer[len] = '/';
    buffer[len+1] = '\0';
    len += 1;
  }

  if(len + 10 >= buflen){
    MTCP_PRINTF("***ERROR Ckpt directory would overflow given buffer!");
    return -1;
  }
  mtcp_strcpy(buffer + len, "ckpt_rank_");
  len += 10; // length of "ckpt_rank_"

  // "Add rank"
  len += itoa2(rank, buffer + len, 10); // TODO: this can theoretically overflow
  if(len + 10 >= buflen){
    MTCP_PRINTF("***ERROR Ckpt directory has overflowed the given buffer!");
    return -1;
  }

  // append '/'
  if(len + 1 >= buflen){
    MTCP_PRINTF("***ERROR Ckpt directory would overflow given buffer!");
    return -1;
  }
  buffer[len] = '/';
  buffer[len + 1] = '\0'; // keep null terminated for open call
  len += 1;

  int fd = mtcp_sys_open2(buffer, O_RDONLY | O_DIRECTORY);
  if(fd == -1) {
      return -1;
  }

  char ldirents[256];
  int found = 0;
  while(!found){
      int nread = mtcp_sys_getdents(fd, ldirents, 256);
      if(nread == -1) {
          MTCP_PRINTF("***ERROR reading directory entries from directory (%s); errno: %d\n",
                      buffer, mtcp_sys_errno);
          return -1;
      }
      if(nread == 0) return -1; // end of directory

      int bpos = 0;
      while(bpos < nread) {
        struct linux_dirent *entry = (struct linux_dirent *) (ldirents + bpos);
        int slen = mtcp_strlen(entry->d_name);
        // int slen = entry->d_reclen - 2 - offsetof(struct linux_dirent, d_name);
        if(slen > 6
            && my_memcmp(entry->d_name, "ckpt", 4) == 0
            && my_memcmp(entry->d_name + slen - 6, ".dmtcp", 6) == 0) {
          found = 1;
          if(len + slen >= buflen){
            MTCP_PRINTF("***ERROR Ckpt file name would overflow given buffer!");
            len = -1;
            break; // don't return or we won't close the file
          }
          mtcp_strcpy(buffer + len, entry->d_name);
          len += slen;
          break;
        }

        if(entry->d_reclen == 0) {
          MTCP_PRINTF("***ERROR Directory Entry struct invalid size of 0!");
          found = 1; // just to exit outer loop
          len = -1;
          break; // don't return or we won't close the file
        }
        bpos += entry->d_reclen;
      }
  }

  if(mtcp_sys_close(fd) == -1) {
      MTCP_PRINTF("***ERROR closing ckpt directory (%s); errno: %d\n",
                  buffer, mtcp_sys_errno);
      return -1;
  }

  return len;
}

void
set_header_filepath(char* full_filename, char* restartDir)
{
  char *header_filename = "ckpt_rank_0/header.mana";
  char restart_path[PATH_MAX] = {0};

  MTCP_ASSERT(mtcp_strlen(header_filename) +
              mtcp_strlen(restartDir) <= PATH_MAX - 2);

  if (mtcp_strlen(restartDir) == 0) {
    mtcp_strcpy(restart_path, "./");
  }
  else {
    mtcp_strcpy(restart_path, restartDir);
    restart_path[mtcp_strlen(restartDir)] = '/';
    restart_path[mtcp_strlen(restartDir)+1] = '\0';
  }
  mtcp_strcpy(full_filename, restart_path);
  mtcp_strncat(full_filename, header_filename, mtcp_strlen(header_filename));
}

void
mtcp_plugin_hook(RestoreInfo *rinfo)
{
  //remap_vdso_and_vvar_regions(rinfo);
  //mysetauxval(rinfo->environ, AT_SYSINFO_EHDR,
  //            (unsigned long int) rinfo->currentVdsoStart);

  // NOTE: We use mtcp_restart's original stack to initialize the lower
  // half. We need to do this in order to call MPI_Init() in the lower half,
  // which is required to figure out our rank, and hence, figure out which
  // checkpoint image to open for memory restoration.
  // The other assumption here is that we can only handle uncompressed
  // checkpoint images.
  // This creates the lower half and copies the bits to this address space
  splitProcess(rinfo);
  char full_filename[PATH_MAX] = {0};
  set_header_filepath(full_filename, rinfo->restartDir);
  ManaHeader m_header;
  // Read header.mana to extract init_flag
  // MTCP_PRINTF("[Rank] filename: %s\n", full_filename);
  int fd = mtcp_sys_open2(full_filename, O_RDONLY);
  // MTCP_PRINTF("[Rank] header.mana, fd: %d\n", fd);
  if (fd != -1)
  {
    mtcp_sys_read(fd, &m_header.init_flag, sizeof(int));
    mtcp_sys_close(fd);
  }
  else
  {
      m_header.init_flag = 1;
  }
  typedef int (*getRankFptr_t)(int);

  int rank = -1;
  //JUMP_TO_LOWER_HALF(rinfo->pluginInfo.fsaddr);
  unsigned long upperHalfFs;
  syscall(SYS_arch_prctl, ARCH_GET_FS, &upperHalfFs);
  syscall(SYS_arch_prctl, ARCH_SET_FS, rinfo->pluginInfo.fsaddr);
  // MPI_Init is called here. GNI memory areas will be loaded by MPI_Init.
  // MPI_Init will call mmap64 to map memory
  rank = ((getRankFptr_t)rinfo->pluginInfo.getRankFptr)(m_header.init_flag);
  syscall(SYS_arch_prctl, ARCH_SET_FS, upperHalfFs);
  //RETURN_TO_UPPER_HALF();
  if(getCkptImageByDir(rinfo, rinfo->ckptImage, 512, rank) == -1) {
      mtcp_strncpy(rinfo->ckptImage,  getCkptImageByRank(rank, rinfo->argv), PATH_MAX);
  }

  rinfo->fd = mtcp_sys_open2(rinfo->ckptImage, O_RDONLY);
  // MTCP_PRINTF("[Rank] rinfo->fd: %d\n", rinfo->fd);
}