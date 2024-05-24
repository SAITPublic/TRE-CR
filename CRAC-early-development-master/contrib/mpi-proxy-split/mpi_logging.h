/****************************************************************************
 *  Copyright (C) 2019-2020 by Twinkle Jain, Rohan garg, and Gene Cooperman *
 *  jain.t@husky.neu.edu, rohgarg@ccs.neu.edu, gene@ccs.neu.edu             *
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
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#ifndef MPI_LOGGING_H
#define MPI_LOGGING_H

#include <stdio.h>
#include <string.h>

#include <linux/limits.h>

// Logging levels
#define NOISE 3 // Noise!
#define INFO  2 // Informational logs
#define ERROR 1 // Highest error/exception level

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

static const char *colors[] = {KNRM, KRED, KBLU, KGRN, KYEL};

#ifndef MPI_DEBUG_LEVEL
// Let's announce errors out loud
# define MPI_DEBUG_LEVEL 1
#endif // ifndef MPI_DEBUG_LEVEL

#define MPIDLOG(LOG_LEVEL, fmt, ...)                                           \
do {                                                                           \
  if (MPI_DEBUG_LEVEL) {                                                       \
    if (LOG_LEVEL <= MPI_DEBUG_LEVEL)                                          \
      fprintf(stderr, "%s[%s +%d]: " fmt KNRM, colors[LOG_LEVEL], __FILE__,    \
              __LINE__, ##__VA_ARGS__);                                        \
  }                                                                            \
} while(0)


#endif // ifndef MPI_LOGGING_H
