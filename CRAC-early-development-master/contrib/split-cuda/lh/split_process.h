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

#ifndef _SPLIT_PROCESS_H
#define _SPLIT_PROCESS_H
#include <asm/prctl.h>
#include <linux/version.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include "jassert.h"
// #include "lower_half_api.h"
LhCoreRegions_t split_lh_regions_list[MAX_LH_REGIONS] = {0};

/* Defined in asm/hwcap.h */
#ifndef HWCAP2_FSGSBASE
#define HWCAP2_FSGSBASE        (1 << 1)
#endif
// initialized to (proxyDlsym_t)lh_info.lh_dlsym

/* The support to set and get FS base register in user-space has been merged in
 * Linux kernel v5.3 (see https://elixir.bootlin.com/linux/v5.3/C/ident/rdfsbase
 *  or https://www.phoronix.com/scan.php?page=news_item&px=Linux-5.3-FSGSBASE).
 *
 * MANA leverages this faster user-space switch on kernel version >= 5.3.
 */
#define ONEMB (uint64_t)(1024 * 1024)
#define ONEGB (uint64_t)(1024 * 1024 * 1024)

// Rounds the given address up to the nearest region size, given as an input.
#define ROUNDADDRUP(addr, size) ((addr + size - 1) & ~(size - 1))

#ifdef __clang__
# define NO_OPTIMIZE __attribute__((optnone))
#else /* ifdef __clang__ */
# define NO_OPTIMIZE __attribute__((optimize(0)))
#endif /* ifdef __clang__ */

// This function splits the process by initializing the lower half with the
// lh_proxy code. It returns 0 on success.

int splitProcess();

#endif // ifndef _SPLIT_PROCESS_H
