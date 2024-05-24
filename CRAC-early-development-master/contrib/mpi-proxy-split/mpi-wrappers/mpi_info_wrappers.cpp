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
#include "mpi_plugin.h"
#include "config.h"
#include "dmtcp.h"
#include "util.h"
#include "jassert.h"
#include "jfilesystem.h"
#include "protectedfds.h"
#include "mpi_nextfunc.h"
#include "virtual-ids.h"


#if (MANA_USE_OPENMPI)
USER_DEFINED_WRAPPER(MPI_Info, Info_f2c, (MPI_Fint) info)
{
  MPIDLOG(INFO, "Called at func '%s' in line %i, info = %d.\n", __FUNCTION__, __LINE__, info);
  MPI_Info lhRealInfo;
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lh_info.fsaddr);
  lhRealInfo = NEXT_FUNC(Info_f2c)(info);
  RETURN_TO_UPPER_HALF();
  MPI_Info virtualInfo = REAL_TO_VIRTUAL_INFO(lhRealInfo);
  DMTCP_PLUGIN_ENABLE_CKPT();
  return virtualInfo;
}

USER_DEFINED_WRAPPER(MPI_Fint, Info_c2f, (MPI_Info) info)
{
  MPIDLOG(INFO, "Called at func '%s' in line %i, info = %p.\n", __FUNCTION__, __LINE__ ,info);
  MPI_Fint fInfo;
  DMTCP_PLUGIN_DISABLE_CKPT();
  MPI_Info realInfo = VIRTUAL_TO_REAL_INFO(info);
  JUMP_TO_LOWER_HALF(lh_info.fsaddr);
#if (MANA_USE_OPENMPI)
  GET_LH_HANDLE(realInfo, (void**)&realInfo);
#endif
  fInfo = NEXT_FUNC(Info_c2f)(realInfo);
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  return fInfo;
}
#endif

#if (MANA_USE_OPENMPI)
PMPI_IMPL(MPI_Info, MPI_Info_f2c, MPI_Fint info)
PMPI_IMPL(MPI_Fint, MPI_Info_c2f, MPI_Info info)
#endif