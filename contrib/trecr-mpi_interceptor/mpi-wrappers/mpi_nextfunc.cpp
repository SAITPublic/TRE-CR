#include "mpi_plugin.h"
#include "config.h"
#include "dmtcp.h"
#include "util.h"
#include "jassert.h"
#include "jfilesystem.h"
#include "protectedfds.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "mpi_nextfunc.h"
#include "../virtual-ids.h"

int MPIinitialized = 0;
static void readMPILhInfoAddr();
extern "C" pid_t dmtcp_get_real_pid();

void
MPIinitialize_wrappers()
{
  if (!MPIinitialized) {
    // fprintf(stderr,"MPIinitialize_wrappers start\n");
    readMPILhInfoAddr();
    MPIinitialized = 1;
  }
}

void
readMPILhInfoAddr()
{
  char filename[100];
  snprintf(filename, 100, "./mpiLhInfo_%d", dmtcp_get_real_pid());
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    printf("Could not open %s for reading.", filename);
    exit(-1);
  }
  ssize_t rc = read(fd, &MPIlh_info, sizeof(MPIlh_info));
  if (rc != (ssize_t)sizeof(MPIlh_info)) {
    perror("Read fewer bytes than expected from addr.bin.");
    exit(-1);
  }
  pdlsym = (proxyDlsym_t)MPIlh_info.lh_dlsym;
  pdlhandle = (proxyHandle_t)MPIlh_info.lh_pdlhandle;
//  unlink(LH_FILE_NAME);
//  close(fd);
}
