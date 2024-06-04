/****************************************************************************
 *   Copyright (C) 2019-2021 by Gene Cooperman, Rohan Garg, Yao Xu          *
 *   gene@ccs.neu.edu, rohgarg@ccs.neu.edu, xu.yao1@northeastern.edu        *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License in the files COPYING and COPYING.LESSER.  If not, see           *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

// Needed for process_vm_readv
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif

#include <linux/version.h>
#include <asm/prctl.h>
#include <sys/prctl.h>
#include <sys/personality.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <libgen.h>
#include <limits.h>
#include <link.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <fcntl.h>


#include "jassert.h"
#include "lower_half_api.h"
#include "switch_context.h"
#include "procmapsutils.h"
#include "config.h" // for HAS_FSGSBASE
#include "constants.h"
#include "util.h"
#include "dmtcp.h"

MPILowerHalfInfo_t MPIlh_info;
#define lh_info MPIlh_info
proxyDlsym_t pdlsym;
proxyHandle_t pdlhandle; // initialized to (proxyHandle_t)lh_info.lh_dlhandle

#if 0
// TODO: Code to compile the lower half at runtime to adjust for differences
// in memory layout in different environments.
static Area getHighestAreaInMaps();
static char* proxyAddrInApp();
static void compileProxy();
#endif

bool FsGsBaseEnabled = false;

bool CheckAndEnableFsGsBase()
{
  const char *str = getenv(ENV_VAR_FSGSBASE_ENABLED);
  if (str != NULL && str[0] == '1') {
    FsGsBaseEnabled = true;
  }

  return FsGsBaseEnabled;
}

SwitchContext::SwitchContext(unsigned long lowerHalfFs)
{
  this->lowerHalfFs = lowerHalfFs;
  this->upperHalfFs = getFS();
  setFS(this->lowerHalfFs);
}

SwitchContext::~SwitchContext()
{
  setFS(this->upperHalfFs);
}

#if 0
static Area
getHighestAreaInMaps()
{
  Area area;
  int mapsfd = open("/proc/self/maps", O_RDONLY);
  Area highest_area;
  mtcp_readMapsLine(mapsfd, &highest_area);
  while (mtcp_readMapsLine(mapsfd, &area)) {
    if (area.endAddr > highest_area.endAddr) {
      highest_area = area;
    }
  }
  close(mapsfd);
  return highest_area;
}

static char*
proxyAddrInApp()
{
  Area highest_area = getHighestAreaInMaps();
  // two pages after after highest memory section?
  return highest_area.endAddr + (2*getpagesize());
}

static void
compileProxy()
{
  dmtcp::string cmd = "make lh_proxy PROXY_TXT_ADDR=0x";
  dmtcp::string safe_addr = proxyAddrInApp();
  cmd.append(safe_addr);
  int ret = system(cmd.c_str());
  if (ret == -1) {
      ret = system("make lh_proxy");
      if (ret == -1) {
        JWARNING(false)(JASSERT_ERRNO).Text("Proxy building failed!");
      }
  }
}
#endif
