#include "convert.h"
static void* UH_MPI_Handle_Ptrs[] = {
  NULL,
  FOREACH_HANDLE(GENERATE_HANDLE)
  NULL,
};

MPI_Handles
getMpiType(void* mpi_handle)
{
    enum MPI_Handles handle = MPI_Handle_Null;
    while (handle < MPI_Handle_Invalid)
    {
        if (UH_MPI_Handle_Ptrs[handle] == mpi_handle)
        {
            return handle;
        }
        handle = (enum MPI_Handles)(handle + 1);
    }

    return MPI_Handle_Invalid;
}