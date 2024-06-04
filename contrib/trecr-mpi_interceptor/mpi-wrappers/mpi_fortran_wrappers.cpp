// *** THIS FILE IS AUTO-GENERATED! DO 'make' TO UPDATE. ***

#include <mpi.h>
#include "dmtcp.h"
#include "jassert.h"
/*
EXTERNC int mpi_finalize_ (int *ierr) {
  *ierr = MPI_Finalize();
  return *ierr;
}

EXTERNC int mpi_finalized_ (int* flag, int *ierr) {
  *ierr = MPI_Finalized(flag);
  return *ierr;
}

EXTERNC int mpi_get_processor_name_ (char* name,  int* resultlen, int *ierr) {
  *ierr = MPI_Get_processor_name(name, resultlen);
  return *ierr;
}

EXTERNC double mpi_wtime_ (int *ierr) {
  return MPI_Wtime();
}

EXTERNC int mpi_initialized_ (int* flag, int *ierr) {
  *ierr = MPI_Initialized(flag);
  return *ierr;
}

EXTERNC int mpi_get_count_ (const MPI_Status* status,  MPI_Datatype* datatype,  int* count, int *ierr) {
  *ierr = MPI_Get_count(status, *datatype, count);
  return *ierr;
}

EXTERNC int mpi_bcast_ (char *buffer, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr) {
  MPI_Comm c_comm;
  MPI_Datatype c_type;
  c_comm = MPI_Comm_f2c(*comm);
  c_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_Bcast(buffer, *count, c_type, *root, c_comm);
  return *ierr;
}

EXTERNC int mpi_barrier_ (MPI_Fint *comm, MPI_Fint *ierr) {
  MPI_Comm c_comm;
  c_comm = MPI_Comm_f2c(*comm);
  *ierr = MPI_Barrier(c_comm);
  return *ierr;
}

EXTERNC int mpi_allreduce_ (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierr) {
  MPI_Comm c_comm;
  MPI_Datatype c_type;
  MPI_Op c_op;
  c_comm = MPI_Comm_f2c(*comm);
  c_type = MPI_Type_f2c(*datatype);
  c_op = MPI_Op_f2c(*op);
  *ierr = MPI_Allreduce(sendbuf, recvbuf, *count, c_type, c_op, c_comm);
  return *ierr;
}

EXTERNC int mpi_reduce_ (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr) {
  MPI_Datatype c_type;
  MPI_Op c_op;
  MPI_Comm c_comm;
  c_type = MPI_Type_f2c(*datatype);
  c_op = MPI_Op_f2c(*op);
  c_comm = MPI_Comm_f2c(*comm);
  *ierr = MPI_Reduce(sendbuf, recvbuf, *count, c_type, c_op, *root, c_comm);
  return *ierr;
}

EXTERNC int mpi_reduce_local_ (const void* inbuf,  void* inoutbuf,  int* count,  MPI_Datatype* datatype,  MPI_Op* op, int *ierr) {
  *ierr = MPI_Reduce_local(inbuf, inoutbuf, *count, *datatype, *op);
  return *ierr;
}

EXTERNC int mpi_reduce_scatter_ (const void* sendbuf,  void* recvbuf,  const int* recvcounts,  MPI_Datatype* datatype,  MPI_Op* op,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, *datatype, *op, *comm);
  return *ierr;
}

EXTERNC int mpi_alltoall_ (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr) {
  MPI_Comm c_comm;
  MPI_Datatype c_sendtype, c_recvtype;
  c_comm = MPI_Comm_f2c(*comm);
  c_sendtype = MPI_Type_f2c(*sendtype);
  c_recvtype = MPI_Type_f2c(*recvtype);
  *ierr = MPI_Alltoall(sendbuf, *sendcount, c_sendtype, recvbuf, *recvcount, c_recvtype, c_comm);
  return *ierr;
}

EXTERNC int mpi_alltoallv_ (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr) {
  MPI_Comm c_comm;
  MPI_Datatype c_sendtype, c_recvtype;
  c_comm = MPI_Comm_f2c(*comm);
  c_sendtype = MPI_Type_f2c(*sendtype);
  c_recvtype = MPI_Type_f2c(*recvtype);
  *ierr = MPI_Alltoallv(sendbuf, sendcounts, sdispls, c_sendtype, recvbuf, recvcounts, rdispls, c_recvtype, c_comm);
  return *ierr;
}

EXTERNC int mpi_allgather_ (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr) {
  MPI_Comm c_comm;
  MPI_Datatype c_sendtype, c_recvtype;
  c_comm = MPI_Comm_f2c(*comm);
  c_sendtype = MPI_Type_f2c(*sendtype);
  c_recvtype = MPI_Type_f2c(*recvtype);
  *ierr = MPI_Allgather(sendbuf, *sendcount, c_sendtype, recvbuf, *recvcount, c_recvtype, c_comm);
  return *ierr;
}

EXTERNC int mpi_allgatherv_ (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr) {
  MPI_Comm c_comm;
  MPI_Datatype c_sendtype, c_recvtype;
  c_comm = MPI_Comm_f2c(*comm);
  c_sendtype = MPI_Type_f2c(*sendtype);
  c_recvtype = MPI_Type_f2c(*recvtype);
  *ierr = MPI_Allgatherv(sendbuf, *sendcount, c_sendtype, recvbuf, recvcounts, displs, c_recvtype, c_comm);
  return *ierr;
}

EXTERNC int mpi_gather_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Gather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, *recvtype, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_gatherv_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  void* recvbuf,  const int* recvcounts,  const int* displs,  MPI_Datatype* recvtype,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Gatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts, displs, *recvtype, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_scatter_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Scatter(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, *recvtype, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_scatterv_ (const void* sendbuf,  const int* sendcounts,  const int* displs,  MPI_Datatype* sendtype,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Scatterv(sendbuf, sendcounts, displs, *sendtype, recvbuf, *recvcount, *recvtype, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_scan_ (const void* sendbuf,  void* recvbuf,  int* count,  MPI_Datatype* datatype,  MPI_Op* op,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Scan(sendbuf, recvbuf, *count, *datatype, *op, *comm);
  return *ierr;
}

EXTERNC int mpi_comm_size_ (MPI_Fint* comm,  MPI_Fint* world_size, MPI_Fint *ierr) {
  MPI_Comm c_comm = MPI_Comm_f2c( *comm );
  *ierr = MPI_Comm_size(c_comm, world_size);
  return *ierr;
}

EXTERNC int mpi_comm_rank_ (MPI_Fint* comm,  MPI_Fint* world_rank, MPI_Fint *ierr) {
  MPI_Comm c_comm = MPI_Comm_f2c( *comm );
  *ierr = MPI_Comm_rank(c_comm, world_rank);
  return *ierr;
}

EXTERNC int mpi_abort_ (MPI_Fint *comm, MPI_Fint *errorcode, MPI_Fint *ierr) {
  MPI_Comm c_comm = MPI_Comm_f2c(*comm);
  *ierr = MPI_Abort(c_comm, *errorcode);
  return *ierr;
}

EXTERNC int mpi_comm_split_ (MPI_Comm* comm,  int* color,  int* key,  MPI_Comm* newcomm, int *ierr) {
  *ierr = MPI_Comm_split(*comm, *color, *key, newcomm);
  return *ierr;
}

EXTERNC int mpi_comm_dup_ (MPI_Comm* comm,  MPI_Comm* newcomm, int *ierr) {
  *ierr = MPI_Comm_dup(*comm, newcomm);
  return *ierr;
}

EXTERNC int mpi_comm_create_ (MPI_Fint *comm, MPI_Fint *group, MPI_Fint *newcomm, MPI_Fint *ierr) {
  MPI_Comm c_newcomm;
  MPI_Comm c_comm = MPI_Comm_f2c (*comm);
  MPI_Group c_group = MPI_Group_f2c(*group);
  *ierr = MPI_Comm_create(c_comm, c_group, &c_newcomm);
  if (MPI_SUCCESS == *ierr) {
        *newcomm = MPI_Comm_c2f (c_newcomm);
    }
  return *ierr;
}

EXTERNC int mpi_comm_compare_ (MPI_Comm* comm1,  MPI_Comm* comm2,  int* result, int *ierr) {
  *ierr = MPI_Comm_compare(*comm1, *comm2, result);
  return *ierr;
}

EXTERNC int mpi_comm_free_ (MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Comm_free(comm);
  return *ierr;
}

EXTERNC int mpi_comm_set_errhandler_ (MPI_Comm* comm,  MPI_Errhandler* errhandler, int *ierr) {
  *ierr = MPI_Comm_set_errhandler(*comm, *errhandler);
  return *ierr;
}

EXTERNC int mpi_topo_test_ (MPI_Comm* comm,  int* status, int *ierr) {
  *ierr = MPI_Topo_test(*comm, status);
  return *ierr;
}

EXTERNC int mpi_comm_split_type_ (MPI_Fint *comm, MPI_Fint *split_type, MPI_Fint *key, MPI_Fint *info, MPI_Fint *newcomm, MPI_Fint *ierr) {
  MPI_Comm c_newcomm;
  MPI_Comm c_comm = MPI_Comm_f2c ( *comm );
  MPI_Info c_info;

  c_info = MPI_Info_f2c(*info);
  *ierr = MPI_Comm_split_type(c_comm, *split_type, *key, c_info, &c_newcomm);
  if (MPI_SUCCESS == *ierr) {
        *newcomm = MPI_Comm_c2f (c_newcomm);
    }
  return *ierr;
}

#ifndef MANA_USE_OPENMPI
EXTERNC int mpi_attr_get_ (MPI_Comm* comm,  int* keyval,  void* attribute_val,  int* flag, int *ierr) {
  *ierr = MPI_Attr_get(*comm, *keyval, attribute_val, flag);
  return *ierr;
}

EXTERNC int mpi_attr_delete_ (MPI_Comm* comm,  int* keyval, int *ierr) {
  *ierr = MPI_Attr_delete(*comm, *keyval);
  return *ierr;
}

EXTERNC int mpi_attr_put_ (MPI_Comm* comm,  int* keyval,  void* attribute_val, int *ierr) {
  *ierr = MPI_Attr_put(*comm, *keyval, attribute_val);
  return *ierr;
}

#endif
EXTERNC int mpi_comm_create_keyval_ (MPI_Comm_copy_attr_function* comm_copy_attr_fn,  MPI_Comm_delete_attr_function* comm_delete_attr_fn,  int* comm_keyval,  void* extra_state, int *ierr) {
  *ierr = MPI_Comm_create_keyval(comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state);
  return *ierr;
}

EXTERNC int mpi_comm_free_keyval_ (int* comm_keyval, int *ierr) {
  *ierr = MPI_Comm_free_keyval(comm_keyval);
  return *ierr;
}

EXTERNC int mpi_comm_create_group_ (MPI_Comm* comm,  MPI_Group* group,  int* tag,  MPI_Comm* newcomm, int *ierr) {
  *ierr = MPI_Comm_create_group(*comm, *group, *tag, newcomm);
  return *ierr;
}

EXTERNC int mpi_cart_coords_ (MPI_Comm* comm,  int* rank,  int* maxdims,  int* coords, int *ierr) {
  *ierr = MPI_Cart_coords(*comm, *rank, *maxdims, coords);
  return *ierr;
}

EXTERNC int mpi_cart_create_ (MPI_Fint *old_comm, MPI_Fint *ndims, MPI_Fint *dims, int *periods, int *reorder, MPI_Fint *comm_cart, MPI_Fint *ierr) {
  MPI_Comm c_comm1, c_comm2;
  c_comm1 = MPI_Comm_f2c(*old_comm);
  *ierr = MPI_Cart_create(c_comm1, *ndims, dims, periods, *reorder, &c_comm2);
  if (MPI_SUCCESS == *ierr) {
        *comm_cart = MPI_Comm_c2f(c_comm2);
    }
  return *ierr;
}

EXTERNC int mpi_cart_get_ (MPI_Comm* comm,  int* maxdims,  int* dims,  int* periods,  int* coords, int *ierr) {
  *ierr = MPI_Cart_get(*comm, *maxdims, dims, periods, coords);
  return *ierr;
}

EXTERNC int mpi_cart_map_ (MPI_Comm* comm,  int* ndims,  const int* dims,  const int* periods,  int* newrank, int *ierr) {
  *ierr = MPI_Cart_map(*comm, *ndims, dims, periods, newrank);
  return *ierr;
}

EXTERNC int mpi_cart_rank_ (MPI_Comm* comm,  const int* coords,  int* rank, int *ierr) {
  *ierr = MPI_Cart_rank(*comm, coords, rank);
  return *ierr;
}

EXTERNC int mpi_cart_shift_ (MPI_Comm* comm,  int* direction,  int* disp,  int* rank_source,  int* rank_dest, int *ierr) {
  *ierr = MPI_Cart_shift(*comm, *direction, *disp, rank_source, rank_dest);
  return *ierr;
}

EXTERNC int mpi_cart_sub_ (MPI_Fint *comm, int *remain_dims, MPI_Fint *new_comm, MPI_Fint *ierr) {
  MPI_Comm c_comm, c_new_comm;
  c_comm = MPI_Comm_f2c(*comm);
  c_new_comm = MPI_Comm_f2c(*new_comm);
  *ierr = MPI_Cart_sub(c_comm, remain_dims, &c_new_comm);
  if (MPI_SUCCESS == *ierr) {
        *new_comm = MPI_Comm_c2f(c_new_comm);
    }
  return *ierr;
}

EXTERNC int mpi_cartdim_get_ (MPI_Comm* comm,  int* ndims, int *ierr) {
  *ierr = MPI_Cartdim_get(*comm, ndims);
  return *ierr;
}

EXTERNC int mpi_dims_create_ (int* nnodes,  int* ndims,  int* dims, int *ierr) {
  *ierr = MPI_Dims_create(*nnodes, *ndims, dims);
  return *ierr;
}

EXTERNC int mpi_test_ (MPI_Request* request,  int* flag,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Test(request, flag, status);
  return *ierr;
}

EXTERNC int mpi_wait_ (MPI_Request* request,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Wait(request, status);
  return *ierr;
}

EXTERNC int mpi_iprobe_ (int* source,  int* tag,  MPI_Comm* comm,  int* flag,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Iprobe(*source, *tag, *comm, flag, status);
  return *ierr;
}

EXTERNC int mpi_probe_ (int* source,  int* tag,  MPI_Comm* comm,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Probe(*source, *tag, *comm, status);
  return *ierr;
}

EXTERNC int mpi_waitall_ (int* count,  MPI_Request* array_of_requests,  MPI_Status* array_of_statuses, int *ierr) {
  *ierr = MPI_Waitall(*count, array_of_requests, array_of_statuses);
  return *ierr;
}

EXTERNC int mpi_waitany_ (int* count,  MPI_Request* array_of_requests,  int* index,  MPI_Status* status, int *ierr) {
  int *local_index = index;
  *ierr = MPI_Waitany(*count, array_of_requests, index, status);
  if (*ierr == MPI_SUCCESS && *local_index != MPI_UNDEFINED) {
    *local_index = *local_index + 1;
  }
  return *ierr;
}

EXTERNC int mpi_testall_ (int* count,  MPI_Request* array_of_requests,  int* flag,  MPI_Status* array_of_statuses, int *ierr) {
  *ierr = MPI_Testall(*count, array_of_requests, flag, array_of_statuses);
  return *ierr;
}

EXTERNC int mpi_testany_ (int* count,  MPI_Request* array_of_requests,  int* index,  int* flag,  MPI_Status* status, int *ierr) {
  int *local_index = index;
  *ierr = MPI_Testany(*count, array_of_requests, index, flag, status);
  if (*ierr == MPI_SUCCESS && *local_index != MPI_UNDEFINED) {
    *local_index = *local_index + 1;
  }
  return *ierr;
}

EXTERNC int mpi_comm_group_ (MPI_Fint *comm, MPI_Fint *group, MPI_Fint *ierr) {
  MPI_Group c_group;
  MPI_Comm c_comm = MPI_Comm_f2c( *comm );
  *ierr = MPI_Comm_group(c_comm, &c_group);
  if (MPI_SUCCESS == *ierr) {
        *group = MPI_Group_c2f (c_group);
    }
  return *ierr;
}

EXTERNC int mpi_group_size_ (MPI_Group* group,  int* size, int *ierr) {
  *ierr = MPI_Group_size(*group, size);
  return *ierr;
}

EXTERNC int mpi_group_free_ (MPI_Group* group, int *ierr) {
  *ierr = MPI_Group_free(group);
  return *ierr;
}

EXTERNC int mpi_group_compare_ (MPI_Group* group1,  MPI_Group* group2,  int* result, int *ierr) {
  *ierr = MPI_Group_compare(*group1, *group2, result);
  return *ierr;
}

EXTERNC int mpi_group_rank_ (MPI_Fint *group, MPI_Fint *rank, MPI_Fint *ierr) {
  MPI_Group c_group;
  c_group = MPI_Group_f2c(*group);
  *ierr = MPI_Group_rank(c_group, rank);
  return *ierr;
}
*/

EXTERNC int mpi_group_incl_ (MPI_Fint *group, MPI_Fint *n, MPI_Fint *ranks, MPI_Fint *newgroup, MPI_Fint *ierr) {
  MPI_Group c_group, c_newgroup;
  c_group = MPI_Group_f2c(*group);
  *ierr = MPI_Group_incl(c_group, *n, ranks, &c_newgroup);
  if (MPI_SUCCESS == *ierr) {
    *newgroup = MPI_Group_c2f (c_newgroup);
    }
  return *ierr;
}

/*
EXTERNC int mpi_type_size_ (MPI_Datatype* datatype,  int* size, int *ierr) {
  *ierr = MPI_Type_size(*datatype, size);
  return *ierr;
}

EXTERNC int mpi_type_commit_ (MPI_Datatype* type, int *ierr) {
  *ierr = MPI_Type_commit(type);
  return *ierr;
}

EXTERNC int mpi_type_contiguous_ (int* count,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_contiguous(*count, *oldtype, newtype);
  return *ierr;
}

EXTERNC int mpi_type_free_ (MPI_Datatype* type, int *ierr) {
  *ierr = MPI_Type_free(type);
  return *ierr;
}

EXTERNC int mpi_type_vector_ (int* count,  int* blocklength,  int* stride,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_vector(*count, *blocklength, *stride, *oldtype, newtype);
  return *ierr;
}

#ifndef MANA_USE_OPENMPI
EXTERNC int mpi_type_hvector_ (int* count,  int* blocklength,  MPI_Aint* stride,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_hvector(*count, *blocklength, *stride, *oldtype, newtype);
  return *ierr;
}

#endif
EXTERNC int mpi_type_create_struct_ (int* count,  const int* array_of_blocklengths,  const MPI_Aint* array_of_displacements,  MPI_Datatype* array_of_types,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_create_struct(*count, array_of_blocklengths, array_of_displacements, array_of_types, newtype);
  return *ierr;
}

#ifndef MANA_USE_OPENMPI
EXTERNC int mpi_type_indexed_ (int* count,  const int* array_of_blocklengths,  const int* array_of_displacements,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_indexed(*count, array_of_blocklengths, array_of_displacements, *oldtype, newtype);
  return *ierr;
}

#endif
EXTERNC int mpi_type_get_extent_ (MPI_Datatype* type,  MPI_Aint* lb,  MPI_Aint* extent, int *ierr) {
  *ierr = MPI_Type_get_extent(*type, lb, extent);
  return *ierr;
}

EXTERNC int mpi_type_create_hvector_ (int* count,  int* blocklength,  MPI_Aint* stride,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_create_hvector(*count, *blocklength, *stride, *oldtype, newtype);
  return *ierr;
}

EXTERNC int mpi_type_create_hindexed_ (int* count,  const int* array_of_blocklengths,  const MPI_Aint* array_of_displacements,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_create_hindexed(*count, array_of_blocklengths, array_of_displacements, *oldtype, newtype);
  return *ierr;
}

EXTERNC int mpi_type_create_hindexed_block_ (int* count,  int* blocklength,  const MPI_Aint* array_of_displacements,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_create_hindexed_block(*count, *blocklength, array_of_displacements, *oldtype, newtype);
  return *ierr;
}

EXTERNC int mpi_type_create_resized_ (MPI_Datatype* oldtype,  MPI_Aint* lb,  MPI_Aint* extent,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_create_resized(*oldtype, *lb, *extent, newtype);
  return *ierr;
}

EXTERNC int mpi_type_dup_ (MPI_Datatype* type,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_dup(*type, newtype);
  return *ierr;
}

#ifndef MANA_USE_OPENMPI
#ifdef CRAY_MPICH_VERSION
EXTERNC int mpi_type_hindexed_ (int* count,
            const int* array_of_blocklengths,
            const MPI_Aint* array_of_displacements,
            MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
#else
EXTERNC int mpi_type_hindexed_ (int* count,
            int* array_of_blocklengths,  MPI_Aint* array_of_displacements,
            MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
#endif
    *ierr = MPI_Type_hindexed(*count, array_of_blocklengths,
            array_of_displacements, *oldtype, newtype);
  return *ierr;
}

#endif
EXTERNC int mpi_pack_size_ (int* incount,  MPI_Datatype* datatype,  MPI_Comm* comm,  int* size, int *ierr) {
  *ierr = MPI_Pack_size(*incount, *datatype, *comm, size);
  return *ierr;
}

EXTERNC int mpi_pack_ (const void* inbuf,  int* incount,  MPI_Datatype* datatype,  void* outbuf,  int* outsize,  int* position,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Pack(inbuf, *incount, *datatype, outbuf, *outsize, position, *comm);
  return *ierr;
}

EXTERNC int mpi_op_create_ (MPI_User_function* user_fn,  int* commute,  MPI_Op* op, int *ierr) {
  *ierr = MPI_Op_create(user_fn, *commute, op);
  return *ierr;
}

EXTERNC int mpi_op_free_ (MPI_Op* op, int *ierr) {
  *ierr = MPI_Op_free(op);
  return *ierr;
}

EXTERNC int mpi_send_ (const void* buf,  int* count,  MPI_Datatype* datatype,  int* dest,  int* tag,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Send(buf, *count, *datatype, *dest, *tag, *comm);
  return *ierr;
}

EXTERNC int mpi_isend_ (const void* buf,  int* count,  MPI_Datatype* datatype,  int* dest,  int* tag,  MPI_Comm* comm,  MPI_Request* request, int *ierr) {
  *ierr = MPI_Isend(buf, *count, *datatype, *dest, *tag, *comm, request);
  return *ierr;
}

EXTERNC int mpi_recv_ (void* buf,  int* count,  MPI_Datatype* datatype,  int* source,  int* tag,  MPI_Comm* comm,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Recv(buf, *count, *datatype, *source, *tag, *comm, status);
  return *ierr;
}

EXTERNC int mpi_irecv_ (void* buf,  int* count,  MPI_Datatype* datatype,  int* source,  int* tag,  MPI_Comm* comm,  MPI_Request* request, int *ierr) {
  *ierr = MPI_Irecv(buf, *count, *datatype, *source, *tag, *comm, request);
  return *ierr;
}

EXTERNC int mpi_sendrecv_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  int* dest,  int* sendtag,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  int* source,  int* recvtag,  MPI_Comm* comm,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Sendrecv(sendbuf, *sendcount, *sendtype, *dest, *sendtag, recvbuf, *recvcount, *recvtype, *source, *recvtag, *comm, status);
  return *ierr;
}

EXTERNC int mpi_sendrecv_replace_ (void* buf,  int* count,  MPI_Datatype* datatype,  int* dest,  int* sendtag,  int* source,  int* recvtag,  MPI_Comm* comm,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Sendrecv_replace(buf, *count, *datatype, *dest, *sendtag, *source, *recvtag, *comm, status);
  return *ierr;
}

EXTERNC int mpi_rsend_ (const void* ibuf,  int* count,  MPI_Datatype* datatype,  int* dest,  int* tag,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Rsend(ibuf, *count, *datatype, *dest, *tag, *comm);
  return *ierr;
}

EXTERNC int mpi_ibarrier_ (MPI_Comm* comm,  MPI_Request* request, int *ierr) {
  *ierr = MPI_Ibarrier(*comm, request);
  return *ierr;
}

EXTERNC int mpi_ibcast_ (void* buffer,  int* count,  MPI_Datatype* datatype,  int* root,  MPI_Comm* comm,  MPI_Request* request, int *ierr) {
  *ierr = MPI_Ibcast(buffer, *count, *datatype, *root, *comm, request);
  return *ierr;
}

EXTERNC int mpi_ireduce_ (const void* sendbuf,  void* recvbuf,  int* count,  MPI_Datatype* datatype,  MPI_Op* op,  int* root,  MPI_Comm* comm,  MPI_Request* request, int *ierr) {
  *ierr = MPI_Ireduce(sendbuf, recvbuf, *count, *datatype, *op, *root, *comm, request);
  return *ierr;
}

EXTERNC int mpi_group_translate_ranks_ (MPI_Group* group1,  int* n,  const int* ranks1,  MPI_Group* group2,  int* ranks2, int *ierr) {
  *ierr = MPI_Group_translate_ranks(*group1, *n, ranks1, *group2, ranks2);
  return *ierr;
}

EXTERNC int mpi_alloc_mem_ (MPI_Aint* size,  MPI_Info* info,  void* baseptr, int *ierr) {
  *ierr = MPI_Alloc_mem(*size, *info, baseptr);
  return *ierr;
}

EXTERNC int mpi_free_mem_ (void* base, int *ierr) {
  *ierr = MPI_Free_mem(base);
  return *ierr;
}

EXTERNC int mpi_error_string_ (int* errorcode,  char* string,  int* resultlen, int *ierr) {
  *ierr = MPI_Error_string(*errorcode, string, resultlen);
  return *ierr;
}

EXTERNC int mpi_file_open_ (MPI_Comm* comm,  const char* filename,  int* amode,  MPI_Info* info,  MPI_File* fh, int *ierr) {
  *ierr = MPI_File_open(*comm, filename, *amode, *info, fh);
  return *ierr;
}

EXTERNC int mpi_file_get_atomicity_ (MPI_File* fh,  int* flag, int *ierr) {
  *ierr = MPI_File_get_atomicity(*fh, flag);
  return *ierr;
}

EXTERNC int mpi_file_set_atomicity_ (MPI_File* fh,  int* flag, int *ierr) {
  *ierr = MPI_File_set_atomicity(*fh, *flag);
  return *ierr;
}

EXTERNC int mpi_file_set_size_ (MPI_File* fh,  MPI_Offset* size, int *ierr) {
  *ierr = MPI_File_set_size(*fh, *size);
  return *ierr;
}

EXTERNC int mpi_file_get_size_ (MPI_File* fh,  MPI_Offset* size, int *ierr) {
  *ierr = MPI_File_get_size(*fh, size);
  return *ierr;
}

EXTERNC int mpi_file_set_view_ (MPI_File* fh,  MPI_Offset* disp,  MPI_Datatype* etype,  MPI_Datatype* filetype,  const char* datarep,  MPI_Info* info, int *ierr) {
  *ierr = MPI_File_set_view(*fh, *disp, *etype, *filetype, datarep, *info);
  return *ierr;
}

EXTERNC int mpi_file_get_view_ (MPI_File* fh,  MPI_Offset* disp,  MPI_Datatype* etype,  MPI_Datatype* filetype,  char* datarep, int *ierr) {
  *ierr = MPI_File_get_view(*fh, disp, etype, filetype, datarep);
  return *ierr;
}

EXTERNC int mpi_file_read_ (MPI_File* fh,  void* buf,  int* count,  MPI_Datatype* datatype,  MPI_Status* status, int *ierr) {
  *ierr = MPI_File_read(*fh, buf, *count, *datatype, status);
  return *ierr;
}

EXTERNC int mpi_file_read_at_ (MPI_File* fh,  MPI_Offset* offset,  void* buf,  int* count,  MPI_Datatype* datatype,  MPI_Status* status, int *ierr) {
  *ierr = MPI_File_read_at(*fh, *offset, buf, *count, *datatype, status);
  return *ierr;
}

EXTERNC int mpi_file_read_at_all_ (MPI_File* fh,  MPI_Offset* offset,  void* buf,  int* count,  MPI_Datatype* datatype,  MPI_Status* status, int *ierr) {
  *ierr = MPI_File_read_at_all(*fh, *offset, buf, *count, *datatype, status);
  return *ierr;
}

EXTERNC int mpi_file_write_ (MPI_File* fh,  const void* buf,  int* count,  MPI_Datatype* datatype,  MPI_Status* status, int *ierr) {
  *ierr = MPI_File_write(*fh, buf, *count, *datatype, status);
  return *ierr;
}

EXTERNC int mpi_file_write_at_ (MPI_File* fh,  MPI_Offset* offset,  const void* buf,  int* count,  MPI_Datatype* datatype,  MPI_Status* status, int *ierr) {
  *ierr = MPI_File_write_at(*fh, *offset, buf, *count, *datatype, status);
  return *ierr;
}

EXTERNC int mpi_file_write_at_all_ (MPI_File* fh,  MPI_Offset* offset,  const void* buf,  int* count,  MPI_Datatype* datatype,  MPI_Status* status, int *ierr) {
  *ierr = MPI_File_write_at_all(*fh, *offset, buf, *count, *datatype, status);
  return *ierr;
}

EXTERNC int mpi_file_sync_ (MPI_File* fh, int *ierr) {
  *ierr = MPI_File_sync(*fh);
  return *ierr;
}

EXTERNC int mpi_file_get_position_ (MPI_File* fh,  MPI_Offset* offset, int *ierr) {
  *ierr = MPI_File_get_position(*fh, offset);
  return *ierr;
}

EXTERNC int mpi_file_seek_ (MPI_File* fh,  MPI_Offset* offset,  int* whence, int *ierr) {
  *ierr = MPI_File_seek(*fh, *offset, *whence);
  return *ierr;
}

EXTERNC int mpi_file_close_ (MPI_File* fh, int *ierr) {
  *ierr = MPI_File_close(fh);
  return *ierr;
}

EXTERNC int mpi_file_set_errhandler_ (MPI_File* fh,  MPI_Errhandler* errhandler, int *ierr) {
  *ierr = MPI_File_set_errhandler(*fh, *errhandler);
  return *ierr;
}

EXTERNC int mpi_file_get_errhandler_ (MPI_File* fh,  MPI_Errhandler* errhandler, int *ierr) {
  *ierr = MPI_File_get_errhandler(*fh, errhandler);
  return *ierr;
}

EXTERNC int mpi_file_delete_ (const char* filename,  MPI_Info* info, int *ierr) {
  *ierr = MPI_File_delete(filename, *info);
  return *ierr;
}

EXTERNC int mpi_get_library_version_ (char* version,  int* resultlen, int *ierr) {
  *ierr = MPI_Get_library_version(version, resultlen);
  return *ierr;
}

EXTERNC int mpi_get_address_ (const void* location,  MPI_Aint* address, int *ierr) {
  *ierr = MPI_Get_address(location, address);
  return *ierr;
}

EXTERNC int mpi_init_ (int *ierr) {
  int argc = 0;
  char **argv;
  *ierr = MPI_Init(&argc, &argv);
  return *ierr;
}
EXTERNC int mpi_init_thread_ (int* required, int* provided, int *ierr) {
  int argc = 0;
  char **argv;
  *ierr = MPI_Init_thread(&argc, &argv, *required, provided);
  return *ierr;
}
*/
