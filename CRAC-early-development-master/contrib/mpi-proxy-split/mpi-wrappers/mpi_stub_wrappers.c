// *** THIS FILE IS AUTO-GENERATED! DO 'make' TO UPDATE. ***

// This is used to generate libmpistub.so.  It should contain the same symbols
//   as libmpi.so.  This is to allow end users to compile an upper half linked
//   against libmpistub.so, so that it can be loaded even if the original
//   libmpi.so is no longer present.
// At compile-time, the Makefile of an MPI application may still compile against
//   libmpi.so, libmpich.so, etc.  The top-level lib/dmtcp directory has
//   symbolic links for these libraries, that link to libmpistub.so.
//   By convention, these executables have file type ./mana.exe .
//   Hence, as long as a lower-half (static) MPI library is present, we don't
//   need the dynamic MPI library to build the upper-half executable.
// At runtime, using mana_launch or dmtcp_launch, the library libmana.so will be
//   preloaded by setting LD_PRELOAD.  Normally, libmana.so includes all
//   symbols present in a libmpi.so.  So libmpistub.so normally is never
//   called ar tuntime.
//   (Note that mpi_unimplemented_wrappers.cpp is also compiled into libmana.so.)
// In principle, MANA could be launched with a native MPI executable (e.g., .exe)
//   instead of an executable of type .mana.exe.
//   FIXME:  Does it work to just use the .exe file?
// native signatures from local mpi.h conflict with this.
// #include <mpi.h>
#include <assert.h>
int mpl_cray_trmem = 0;
int MPI_Accumulate() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Accumulate () __attribute__ ((weak, alias ("MPI_Accumulate")));
int mpi_accumulate_ () __attribute__ ((weak, alias ("MPI_Accumulate")));

int MPI_Add_error_class() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Add_error_class () __attribute__ ((weak, alias ("MPI_Add_error_class")));
int mpi_add_error_class_ () __attribute__ ((weak, alias ("MPI_Add_error_class")));

int MPI_Add_error_code() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Add_error_code () __attribute__ ((weak, alias ("MPI_Add_error_code")));
int mpi_add_error_code_ () __attribute__ ((weak, alias ("MPI_Add_error_code")));

int MPI_Add_error_string() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Add_error_string () __attribute__ ((weak, alias ("MPI_Add_error_string")));
int mpi_add_error_string_ () __attribute__ ((weak, alias ("MPI_Add_error_string")));

int MPI_Address() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Address () __attribute__ ((weak, alias ("MPI_Address")));
int mpi_address_ () __attribute__ ((weak, alias ("MPI_Address")));

int MPI_Iallgather() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Iallgather () __attribute__ ((weak, alias ("MPI_Iallgather")));
int mpi_iallgather_ () __attribute__ ((weak, alias ("MPI_Iallgather")));

int MPI_Iallgatherv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Iallgatherv () __attribute__ ((weak, alias ("MPI_Iallgatherv")));
int mpi_iallgatherv_ () __attribute__ ((weak, alias ("MPI_Iallgatherv")));

int MPI_Alloc_mem() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Alloc_mem () __attribute__ ((weak, alias ("MPI_Alloc_mem")));
int mpi_alloc_mem_ () __attribute__ ((weak, alias ("MPI_Alloc_mem")));

int MPI_Iallreduce() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Iallreduce () __attribute__ ((weak, alias ("MPI_Iallreduce")));
int mpi_iallreduce_ () __attribute__ ((weak, alias ("MPI_Iallreduce")));

int MPI_Ialltoall() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ialltoall () __attribute__ ((weak, alias ("MPI_Ialltoall")));
int mpi_ialltoall_ () __attribute__ ((weak, alias ("MPI_Ialltoall")));

int MPI_Ialltoallv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ialltoallv () __attribute__ ((weak, alias ("MPI_Ialltoallv")));
int mpi_ialltoallv_ () __attribute__ ((weak, alias ("MPI_Ialltoallv")));

int MPI_Alltoallw() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Alltoallw () __attribute__ ((weak, alias ("MPI_Alltoallw")));
int mpi_alltoallw_ () __attribute__ ((weak, alias ("MPI_Alltoallw")));

int MPI_Ialltoallw() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ialltoallw () __attribute__ ((weak, alias ("MPI_Ialltoallw")));
int mpi_ialltoallw_ () __attribute__ ((weak, alias ("MPI_Ialltoallw")));

int MPI_Attr_delete() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Attr_delete () __attribute__ ((weak, alias ("MPI_Attr_delete")));
int mpi_attr_delete_ () __attribute__ ((weak, alias ("MPI_Attr_delete")));

int MPI_Attr_get() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Attr_get () __attribute__ ((weak, alias ("MPI_Attr_get")));
int mpi_attr_get_ () __attribute__ ((weak, alias ("MPI_Attr_get")));

int MPI_Attr_put() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Attr_put () __attribute__ ((weak, alias ("MPI_Attr_put")));
int mpi_attr_put_ () __attribute__ ((weak, alias ("MPI_Attr_put")));

int MPI_Ibarrier() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ibarrier () __attribute__ ((weak, alias ("MPI_Ibarrier")));
int mpi_ibarrier_ () __attribute__ ((weak, alias ("MPI_Ibarrier")));

int MPI_Bsend() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Bsend () __attribute__ ((weak, alias ("MPI_Bsend")));
int mpi_bsend_ () __attribute__ ((weak, alias ("MPI_Bsend")));

int MPI_Ibcast() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ibcast () __attribute__ ((weak, alias ("MPI_Ibcast")));
int mpi_ibcast_ () __attribute__ ((weak, alias ("MPI_Ibcast")));

int MPI_Bsend_init() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Bsend_init () __attribute__ ((weak, alias ("MPI_Bsend_init")));
int mpi_bsend_init_ () __attribute__ ((weak, alias ("MPI_Bsend_init")));

int MPI_Buffer_attach() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Buffer_attach () __attribute__ ((weak, alias ("MPI_Buffer_attach")));
int mpi_buffer_attach_ () __attribute__ ((weak, alias ("MPI_Buffer_attach")));

int MPI_Buffer_detach() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Buffer_detach () __attribute__ ((weak, alias ("MPI_Buffer_detach")));
int mpi_buffer_detach_ () __attribute__ ((weak, alias ("MPI_Buffer_detach")));

int MPI_Cancel() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Cancel () __attribute__ ((weak, alias ("MPI_Cancel")));
int mpi_cancel_ () __attribute__ ((weak, alias ("MPI_Cancel")));

int MPI_Close_port() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Close_port () __attribute__ ((weak, alias ("MPI_Close_port")));
int mpi_close_port_ () __attribute__ ((weak, alias ("MPI_Close_port")));

int MPI_Comm_accept() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_accept () __attribute__ ((weak, alias ("MPI_Comm_accept")));
int mpi_comm_accept_ () __attribute__ ((weak, alias ("MPI_Comm_accept")));

int MPI_Comm_call_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_call_errhandler () __attribute__ ((weak, alias ("MPI_Comm_call_errhandler")));
int mpi_comm_call_errhandler_ () __attribute__ ((weak, alias ("MPI_Comm_call_errhandler")));

int MPI_Comm_connect() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_connect () __attribute__ ((weak, alias ("MPI_Comm_connect")));
int mpi_comm_connect_ () __attribute__ ((weak, alias ("MPI_Comm_connect")));

int MPI_Comm_create_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_create_errhandler () __attribute__ ((weak, alias ("MPI_Comm_create_errhandler")));
int mpi_comm_create_errhandler_ () __attribute__ ((weak, alias ("MPI_Comm_create_errhandler")));

int MPI_Comm_create_keyval() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_create_keyval () __attribute__ ((weak, alias ("MPI_Comm_create_keyval")));
int mpi_comm_create_keyval_ () __attribute__ ((weak, alias ("MPI_Comm_create_keyval")));

int MPI_Comm_create_group() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_create_group () __attribute__ ((weak, alias ("MPI_Comm_create_group")));
int mpi_comm_create_group_ () __attribute__ ((weak, alias ("MPI_Comm_create_group")));

int MPI_Comm_delete_attr() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_delete_attr () __attribute__ ((weak, alias ("MPI_Comm_delete_attr")));
int mpi_comm_delete_attr_ () __attribute__ ((weak, alias ("MPI_Comm_delete_attr")));

int MPI_Comm_disconnect() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_disconnect () __attribute__ ((weak, alias ("MPI_Comm_disconnect")));
int mpi_comm_disconnect_ () __attribute__ ((weak, alias ("MPI_Comm_disconnect")));

int MPI_Comm_idup() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_idup () __attribute__ ((weak, alias ("MPI_Comm_idup")));
int mpi_comm_idup_ () __attribute__ ((weak, alias ("MPI_Comm_idup")));

int MPI_Comm_dup_with_info() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_dup_with_info () __attribute__ ((weak, alias ("MPI_Comm_dup_with_info")));
int mpi_comm_dup_with_info_ () __attribute__ ((weak, alias ("MPI_Comm_dup_with_info")));

int MPI_Comm_free_keyval() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_free_keyval () __attribute__ ((weak, alias ("MPI_Comm_free_keyval")));
int mpi_comm_free_keyval_ () __attribute__ ((weak, alias ("MPI_Comm_free_keyval")));

int MPI_Comm_get_attr() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_get_attr () __attribute__ ((weak, alias ("MPI_Comm_get_attr")));
int mpi_comm_get_attr_ () __attribute__ ((weak, alias ("MPI_Comm_get_attr")));

int MPI_Dist_graph_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Dist_graph_create () __attribute__ ((weak, alias ("MPI_Dist_graph_create")));
int mpi_dist_graph_create_ () __attribute__ ((weak, alias ("MPI_Dist_graph_create")));

int MPI_Dist_graph_create_adjacent() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Dist_graph_create_adjacent () __attribute__ ((weak, alias ("MPI_Dist_graph_create_adjacent")));
int mpi_dist_graph_create_adjacent_ () __attribute__ ((weak, alias ("MPI_Dist_graph_create_adjacent")));

int MPI_Dist_graph_neighbors() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Dist_graph_neighbors () __attribute__ ((weak, alias ("MPI_Dist_graph_neighbors")));
int mpi_dist_graph_neighbors_ () __attribute__ ((weak, alias ("MPI_Dist_graph_neighbors")));

int MPI_Dist_graph_neighbors_count() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Dist_graph_neighbors_count () __attribute__ ((weak, alias ("MPI_Dist_graph_neighbors_count")));
int mpi_dist_graph_neighbors_count_ () __attribute__ ((weak, alias ("MPI_Dist_graph_neighbors_count")));

int MPI_Comm_get_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_get_errhandler () __attribute__ ((weak, alias ("MPI_Comm_get_errhandler")));
int mpi_comm_get_errhandler_ () __attribute__ ((weak, alias ("MPI_Comm_get_errhandler")));

int MPI_Comm_get_info() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_get_info () __attribute__ ((weak, alias ("MPI_Comm_get_info")));
int mpi_comm_get_info_ () __attribute__ ((weak, alias ("MPI_Comm_get_info")));

int MPI_Comm_get_name() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_get_name () __attribute__ ((weak, alias ("MPI_Comm_get_name")));
int mpi_comm_get_name_ () __attribute__ ((weak, alias ("MPI_Comm_get_name")));

int MPI_Comm_get_parent() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_get_parent () __attribute__ ((weak, alias ("MPI_Comm_get_parent")));
int mpi_comm_get_parent_ () __attribute__ ((weak, alias ("MPI_Comm_get_parent")));

int MPI_Comm_join() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_join () __attribute__ ((weak, alias ("MPI_Comm_join")));
int mpi_comm_join_ () __attribute__ ((weak, alias ("MPI_Comm_join")));

int MPI_Comm_remote_group() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_remote_group () __attribute__ ((weak, alias ("MPI_Comm_remote_group")));
int mpi_comm_remote_group_ () __attribute__ ((weak, alias ("MPI_Comm_remote_group")));

int MPI_Comm_remote_size() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_remote_size () __attribute__ ((weak, alias ("MPI_Comm_remote_size")));
int mpi_comm_remote_size_ () __attribute__ ((weak, alias ("MPI_Comm_remote_size")));

int MPI_Comm_set_attr() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_set_attr () __attribute__ ((weak, alias ("MPI_Comm_set_attr")));
int mpi_comm_set_attr_ () __attribute__ ((weak, alias ("MPI_Comm_set_attr")));

int MPI_Comm_set_info() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_set_info () __attribute__ ((weak, alias ("MPI_Comm_set_info")));
int mpi_comm_set_info_ () __attribute__ ((weak, alias ("MPI_Comm_set_info")));

int MPI_Comm_set_name() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_set_name () __attribute__ ((weak, alias ("MPI_Comm_set_name")));
int mpi_comm_set_name_ () __attribute__ ((weak, alias ("MPI_Comm_set_name")));

int MPI_Comm_spawn() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_spawn () __attribute__ ((weak, alias ("MPI_Comm_spawn")));
int mpi_comm_spawn_ () __attribute__ ((weak, alias ("MPI_Comm_spawn")));

int MPI_Comm_spawn_multiple() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_spawn_multiple () __attribute__ ((weak, alias ("MPI_Comm_spawn_multiple")));
int mpi_comm_spawn_multiple_ () __attribute__ ((weak, alias ("MPI_Comm_spawn_multiple")));

int MPI_Comm_split_type() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_split_type () __attribute__ ((weak, alias ("MPI_Comm_split_type")));
int mpi_comm_split_type_ () __attribute__ ((weak, alias ("MPI_Comm_split_type")));

int MPI_Comm_test_inter() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_test_inter () __attribute__ ((weak, alias ("MPI_Comm_test_inter")));
int mpi_comm_test_inter_ () __attribute__ ((weak, alias ("MPI_Comm_test_inter")));

int MPI_Compare_and_swap() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Compare_and_swap () __attribute__ ((weak, alias ("MPI_Compare_and_swap")));
int mpi_compare_and_swap_ () __attribute__ ((weak, alias ("MPI_Compare_and_swap")));

int MPI_Dims_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Dims_create () __attribute__ ((weak, alias ("MPI_Dims_create")));
int mpi_dims_create_ () __attribute__ ((weak, alias ("MPI_Dims_create")));

int MPI_Errhandler_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Errhandler_create () __attribute__ ((weak, alias ("MPI_Errhandler_create")));
int mpi_errhandler_create_ () __attribute__ ((weak, alias ("MPI_Errhandler_create")));

int MPI_Errhandler_free() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Errhandler_free () __attribute__ ((weak, alias ("MPI_Errhandler_free")));
int mpi_errhandler_free_ () __attribute__ ((weak, alias ("MPI_Errhandler_free")));

int MPI_Errhandler_get() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Errhandler_get () __attribute__ ((weak, alias ("MPI_Errhandler_get")));
int mpi_errhandler_get_ () __attribute__ ((weak, alias ("MPI_Errhandler_get")));

int MPI_Errhandler_set() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Errhandler_set () __attribute__ ((weak, alias ("MPI_Errhandler_set")));
int mpi_errhandler_set_ () __attribute__ ((weak, alias ("MPI_Errhandler_set")));

int MPI_Error_class() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Error_class () __attribute__ ((weak, alias ("MPI_Error_class")));
int mpi_error_class_ () __attribute__ ((weak, alias ("MPI_Error_class")));

int MPI_Error_string() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Error_string () __attribute__ ((weak, alias ("MPI_Error_string")));
int mpi_error_string_ () __attribute__ ((weak, alias ("MPI_Error_string")));

int MPI_Exscan() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Exscan () __attribute__ ((weak, alias ("MPI_Exscan")));
int mpi_exscan_ () __attribute__ ((weak, alias ("MPI_Exscan")));

int MPI_Fetch_and_op() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Fetch_and_op () __attribute__ ((weak, alias ("MPI_Fetch_and_op")));
int mpi_fetch_and_op_ () __attribute__ ((weak, alias ("MPI_Fetch_and_op")));

int MPI_Iexscan() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Iexscan () __attribute__ ((weak, alias ("MPI_Iexscan")));
int mpi_iexscan_ () __attribute__ ((weak, alias ("MPI_Iexscan")));

int MPI_File_call_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_call_errhandler () __attribute__ ((weak, alias ("MPI_File_call_errhandler")));
int mpi_file_call_errhandler_ () __attribute__ ((weak, alias ("MPI_File_call_errhandler")));

int MPI_File_create_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_create_errhandler () __attribute__ ((weak, alias ("MPI_File_create_errhandler")));
int mpi_file_create_errhandler_ () __attribute__ ((weak, alias ("MPI_File_create_errhandler")));

int MPI_File_set_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_set_errhandler () __attribute__ ((weak, alias ("MPI_File_set_errhandler")));
int mpi_file_set_errhandler_ () __attribute__ ((weak, alias ("MPI_File_set_errhandler")));

int MPI_File_get_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_errhandler () __attribute__ ((weak, alias ("MPI_File_get_errhandler")));
int mpi_file_get_errhandler_ () __attribute__ ((weak, alias ("MPI_File_get_errhandler")));

int MPI_File_open() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_open () __attribute__ ((weak, alias ("MPI_File_open")));
int mpi_file_open_ () __attribute__ ((weak, alias ("MPI_File_open")));

int MPI_File_close() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_close () __attribute__ ((weak, alias ("MPI_File_close")));
int mpi_file_close_ () __attribute__ ((weak, alias ("MPI_File_close")));

int MPI_File_delete() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_delete () __attribute__ ((weak, alias ("MPI_File_delete")));
int mpi_file_delete_ () __attribute__ ((weak, alias ("MPI_File_delete")));

int MPI_File_set_size() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_set_size () __attribute__ ((weak, alias ("MPI_File_set_size")));
int mpi_file_set_size_ () __attribute__ ((weak, alias ("MPI_File_set_size")));

int MPI_File_preallocate() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_preallocate () __attribute__ ((weak, alias ("MPI_File_preallocate")));
int mpi_file_preallocate_ () __attribute__ ((weak, alias ("MPI_File_preallocate")));

int MPI_File_get_size() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_size () __attribute__ ((weak, alias ("MPI_File_get_size")));
int mpi_file_get_size_ () __attribute__ ((weak, alias ("MPI_File_get_size")));

int MPI_File_get_group() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_group () __attribute__ ((weak, alias ("MPI_File_get_group")));
int mpi_file_get_group_ () __attribute__ ((weak, alias ("MPI_File_get_group")));

int MPI_File_get_amode() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_amode () __attribute__ ((weak, alias ("MPI_File_get_amode")));
int mpi_file_get_amode_ () __attribute__ ((weak, alias ("MPI_File_get_amode")));

int MPI_File_set_info() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_set_info () __attribute__ ((weak, alias ("MPI_File_set_info")));
int mpi_file_set_info_ () __attribute__ ((weak, alias ("MPI_File_set_info")));

int MPI_File_get_info() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_info () __attribute__ ((weak, alias ("MPI_File_get_info")));
int mpi_file_get_info_ () __attribute__ ((weak, alias ("MPI_File_get_info")));

int MPI_File_set_view() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_set_view () __attribute__ ((weak, alias ("MPI_File_set_view")));
int mpi_file_set_view_ () __attribute__ ((weak, alias ("MPI_File_set_view")));

int MPI_File_get_view() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_view () __attribute__ ((weak, alias ("MPI_File_get_view")));
int mpi_file_get_view_ () __attribute__ ((weak, alias ("MPI_File_get_view")));

int MPI_File_read_at() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_at () __attribute__ ((weak, alias ("MPI_File_read_at")));
int mpi_file_read_at_ () __attribute__ ((weak, alias ("MPI_File_read_at")));

int MPI_File_read_at_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_at_all () __attribute__ ((weak, alias ("MPI_File_read_at_all")));
int mpi_file_read_at_all_ () __attribute__ ((weak, alias ("MPI_File_read_at_all")));

int MPI_File_write_at() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_at () __attribute__ ((weak, alias ("MPI_File_write_at")));
int mpi_file_write_at_ () __attribute__ ((weak, alias ("MPI_File_write_at")));

int MPI_File_write_at_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_at_all () __attribute__ ((weak, alias ("MPI_File_write_at_all")));
int mpi_file_write_at_all_ () __attribute__ ((weak, alias ("MPI_File_write_at_all")));

int MPI_File_iread_at() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iread_at () __attribute__ ((weak, alias ("MPI_File_iread_at")));
int mpi_file_iread_at_ () __attribute__ ((weak, alias ("MPI_File_iread_at")));

int MPI_File_iwrite_at() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iwrite_at () __attribute__ ((weak, alias ("MPI_File_iwrite_at")));
int mpi_file_iwrite_at_ () __attribute__ ((weak, alias ("MPI_File_iwrite_at")));

int MPI_File_iread_at_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iread_at_all () __attribute__ ((weak, alias ("MPI_File_iread_at_all")));
int mpi_file_iread_at_all_ () __attribute__ ((weak, alias ("MPI_File_iread_at_all")));

int MPI_File_iwrite_at_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iwrite_at_all () __attribute__ ((weak, alias ("MPI_File_iwrite_at_all")));
int mpi_file_iwrite_at_all_ () __attribute__ ((weak, alias ("MPI_File_iwrite_at_all")));

int MPI_File_read() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read () __attribute__ ((weak, alias ("MPI_File_read")));
int mpi_file_read_ () __attribute__ ((weak, alias ("MPI_File_read")));

int MPI_File_read_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_all () __attribute__ ((weak, alias ("MPI_File_read_all")));
int mpi_file_read_all_ () __attribute__ ((weak, alias ("MPI_File_read_all")));

int MPI_File_write() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write () __attribute__ ((weak, alias ("MPI_File_write")));
int mpi_file_write_ () __attribute__ ((weak, alias ("MPI_File_write")));

int MPI_File_write_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_all () __attribute__ ((weak, alias ("MPI_File_write_all")));
int mpi_file_write_all_ () __attribute__ ((weak, alias ("MPI_File_write_all")));

int MPI_File_iread() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iread () __attribute__ ((weak, alias ("MPI_File_iread")));
int mpi_file_iread_ () __attribute__ ((weak, alias ("MPI_File_iread")));

int MPI_File_iwrite() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iwrite () __attribute__ ((weak, alias ("MPI_File_iwrite")));
int mpi_file_iwrite_ () __attribute__ ((weak, alias ("MPI_File_iwrite")));

int MPI_File_iread_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iread_all () __attribute__ ((weak, alias ("MPI_File_iread_all")));
int mpi_file_iread_all_ () __attribute__ ((weak, alias ("MPI_File_iread_all")));

int MPI_File_iwrite_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iwrite_all () __attribute__ ((weak, alias ("MPI_File_iwrite_all")));
int mpi_file_iwrite_all_ () __attribute__ ((weak, alias ("MPI_File_iwrite_all")));

int MPI_File_seek() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_seek () __attribute__ ((weak, alias ("MPI_File_seek")));
int mpi_file_seek_ () __attribute__ ((weak, alias ("MPI_File_seek")));

int MPI_File_get_position() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_position () __attribute__ ((weak, alias ("MPI_File_get_position")));
int mpi_file_get_position_ () __attribute__ ((weak, alias ("MPI_File_get_position")));

int MPI_File_get_byte_offset() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_byte_offset () __attribute__ ((weak, alias ("MPI_File_get_byte_offset")));
int mpi_file_get_byte_offset_ () __attribute__ ((weak, alias ("MPI_File_get_byte_offset")));

int MPI_File_read_shared() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_shared () __attribute__ ((weak, alias ("MPI_File_read_shared")));
int mpi_file_read_shared_ () __attribute__ ((weak, alias ("MPI_File_read_shared")));

int MPI_File_write_shared() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_shared () __attribute__ ((weak, alias ("MPI_File_write_shared")));
int mpi_file_write_shared_ () __attribute__ ((weak, alias ("MPI_File_write_shared")));

int MPI_File_iread_shared() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iread_shared () __attribute__ ((weak, alias ("MPI_File_iread_shared")));
int mpi_file_iread_shared_ () __attribute__ ((weak, alias ("MPI_File_iread_shared")));

int MPI_File_iwrite_shared() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_iwrite_shared () __attribute__ ((weak, alias ("MPI_File_iwrite_shared")));
int mpi_file_iwrite_shared_ () __attribute__ ((weak, alias ("MPI_File_iwrite_shared")));

int MPI_File_read_ordered() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_ordered () __attribute__ ((weak, alias ("MPI_File_read_ordered")));
int mpi_file_read_ordered_ () __attribute__ ((weak, alias ("MPI_File_read_ordered")));

int MPI_File_write_ordered() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_ordered () __attribute__ ((weak, alias ("MPI_File_write_ordered")));
int mpi_file_write_ordered_ () __attribute__ ((weak, alias ("MPI_File_write_ordered")));

int MPI_File_seek_shared() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_seek_shared () __attribute__ ((weak, alias ("MPI_File_seek_shared")));
int mpi_file_seek_shared_ () __attribute__ ((weak, alias ("MPI_File_seek_shared")));

int MPI_File_get_position_shared() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_position_shared () __attribute__ ((weak, alias ("MPI_File_get_position_shared")));
int mpi_file_get_position_shared_ () __attribute__ ((weak, alias ("MPI_File_get_position_shared")));

int MPI_File_read_at_all_begin() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_at_all_begin () __attribute__ ((weak, alias ("MPI_File_read_at_all_begin")));
int mpi_file_read_at_all_begin_ () __attribute__ ((weak, alias ("MPI_File_read_at_all_begin")));

int MPI_File_read_at_all_end() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_at_all_end () __attribute__ ((weak, alias ("MPI_File_read_at_all_end")));
int mpi_file_read_at_all_end_ () __attribute__ ((weak, alias ("MPI_File_read_at_all_end")));

int MPI_File_write_at_all_begin() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_at_all_begin () __attribute__ ((weak, alias ("MPI_File_write_at_all_begin")));
int mpi_file_write_at_all_begin_ () __attribute__ ((weak, alias ("MPI_File_write_at_all_begin")));

int MPI_File_write_at_all_end() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_at_all_end () __attribute__ ((weak, alias ("MPI_File_write_at_all_end")));
int mpi_file_write_at_all_end_ () __attribute__ ((weak, alias ("MPI_File_write_at_all_end")));

int MPI_File_read_all_begin() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_all_begin () __attribute__ ((weak, alias ("MPI_File_read_all_begin")));
int mpi_file_read_all_begin_ () __attribute__ ((weak, alias ("MPI_File_read_all_begin")));

int MPI_File_read_all_end() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_all_end () __attribute__ ((weak, alias ("MPI_File_read_all_end")));
int mpi_file_read_all_end_ () __attribute__ ((weak, alias ("MPI_File_read_all_end")));

int MPI_File_write_all_begin() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_all_begin () __attribute__ ((weak, alias ("MPI_File_write_all_begin")));
int mpi_file_write_all_begin_ () __attribute__ ((weak, alias ("MPI_File_write_all_begin")));

int MPI_File_write_all_end() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_all_end () __attribute__ ((weak, alias ("MPI_File_write_all_end")));
int mpi_file_write_all_end_ () __attribute__ ((weak, alias ("MPI_File_write_all_end")));

int MPI_File_read_ordered_begin() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_ordered_begin () __attribute__ ((weak, alias ("MPI_File_read_ordered_begin")));
int mpi_file_read_ordered_begin_ () __attribute__ ((weak, alias ("MPI_File_read_ordered_begin")));

int MPI_File_read_ordered_end() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_read_ordered_end () __attribute__ ((weak, alias ("MPI_File_read_ordered_end")));
int mpi_file_read_ordered_end_ () __attribute__ ((weak, alias ("MPI_File_read_ordered_end")));

int MPI_File_write_ordered_begin() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_ordered_begin () __attribute__ ((weak, alias ("MPI_File_write_ordered_begin")));
int mpi_file_write_ordered_begin_ () __attribute__ ((weak, alias ("MPI_File_write_ordered_begin")));

int MPI_File_write_ordered_end() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_write_ordered_end () __attribute__ ((weak, alias ("MPI_File_write_ordered_end")));
int mpi_file_write_ordered_end_ () __attribute__ ((weak, alias ("MPI_File_write_ordered_end")));

int MPI_File_get_type_extent() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_type_extent () __attribute__ ((weak, alias ("MPI_File_get_type_extent")));
int mpi_file_get_type_extent_ () __attribute__ ((weak, alias ("MPI_File_get_type_extent")));

int MPI_File_set_atomicity() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_set_atomicity () __attribute__ ((weak, alias ("MPI_File_set_atomicity")));
int mpi_file_set_atomicity_ () __attribute__ ((weak, alias ("MPI_File_set_atomicity")));

int MPI_File_get_atomicity() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_get_atomicity () __attribute__ ((weak, alias ("MPI_File_get_atomicity")));
int mpi_file_get_atomicity_ () __attribute__ ((weak, alias ("MPI_File_get_atomicity")));

int MPI_File_sync() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_File_sync () __attribute__ ((weak, alias ("MPI_File_sync")));
int mpi_file_sync_ () __attribute__ ((weak, alias ("MPI_File_sync")));

int MPI_Free_mem() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Free_mem () __attribute__ ((weak, alias ("MPI_Free_mem")));
int mpi_free_mem_ () __attribute__ ((weak, alias ("MPI_Free_mem")));

int MPI_Igather() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Igather () __attribute__ ((weak, alias ("MPI_Igather")));
int mpi_igather_ () __attribute__ ((weak, alias ("MPI_Igather")));

int MPI_Igatherv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Igatherv () __attribute__ ((weak, alias ("MPI_Igatherv")));
int mpi_igatherv_ () __attribute__ ((weak, alias ("MPI_Igatherv")));

int MPI_Get_address() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Get_address () __attribute__ ((weak, alias ("MPI_Get_address")));
int mpi_get_address_ () __attribute__ ((weak, alias ("MPI_Get_address")));

int MPI_Get_elements() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Get_elements () __attribute__ ((weak, alias ("MPI_Get_elements")));
int mpi_get_elements_ () __attribute__ ((weak, alias ("MPI_Get_elements")));

int MPI_Get_elements_x() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Get_elements_x () __attribute__ ((weak, alias ("MPI_Get_elements_x")));
int mpi_get_elements_x_ () __attribute__ ((weak, alias ("MPI_Get_elements_x")));

int MPI_Get() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Get () __attribute__ ((weak, alias ("MPI_Get")));
int mpi_get_ () __attribute__ ((weak, alias ("MPI_Get")));

int MPI_Get_accumulate() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Get_accumulate () __attribute__ ((weak, alias ("MPI_Get_accumulate")));
int mpi_get_accumulate_ () __attribute__ ((weak, alias ("MPI_Get_accumulate")));

int MPI_Get_library_version() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Get_library_version () __attribute__ ((weak, alias ("MPI_Get_library_version")));
int mpi_get_library_version_ () __attribute__ ((weak, alias ("MPI_Get_library_version")));

int MPI_Get_version() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Get_version () __attribute__ ((weak, alias ("MPI_Get_version")));
int mpi_get_version_ () __attribute__ ((weak, alias ("MPI_Get_version")));

int MPI_Graph_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Graph_create () __attribute__ ((weak, alias ("MPI_Graph_create")));
int mpi_graph_create_ () __attribute__ ((weak, alias ("MPI_Graph_create")));

int MPI_Graph_get() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Graph_get () __attribute__ ((weak, alias ("MPI_Graph_get")));
int mpi_graph_get_ () __attribute__ ((weak, alias ("MPI_Graph_get")));

int MPI_Graph_map() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Graph_map () __attribute__ ((weak, alias ("MPI_Graph_map")));
int mpi_graph_map_ () __attribute__ ((weak, alias ("MPI_Graph_map")));

int MPI_Graph_neighbors_count() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Graph_neighbors_count () __attribute__ ((weak, alias ("MPI_Graph_neighbors_count")));
int mpi_graph_neighbors_count_ () __attribute__ ((weak, alias ("MPI_Graph_neighbors_count")));

int MPI_Graph_neighbors() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Graph_neighbors () __attribute__ ((weak, alias ("MPI_Graph_neighbors")));
int mpi_graph_neighbors_ () __attribute__ ((weak, alias ("MPI_Graph_neighbors")));

int MPI_Graphdims_get() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Graphdims_get () __attribute__ ((weak, alias ("MPI_Graphdims_get")));
int mpi_graphdims_get_ () __attribute__ ((weak, alias ("MPI_Graphdims_get")));

int MPI_Grequest_complete() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Grequest_complete () __attribute__ ((weak, alias ("MPI_Grequest_complete")));
int mpi_grequest_complete_ () __attribute__ ((weak, alias ("MPI_Grequest_complete")));

int MPI_Grequest_start() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Grequest_start () __attribute__ ((weak, alias ("MPI_Grequest_start")));
int mpi_grequest_start_ () __attribute__ ((weak, alias ("MPI_Grequest_start")));

int MPI_Group_difference() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_difference () __attribute__ ((weak, alias ("MPI_Group_difference")));
int mpi_group_difference_ () __attribute__ ((weak, alias ("MPI_Group_difference")));

int MPI_Group_excl() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_excl () __attribute__ ((weak, alias ("MPI_Group_excl")));
int mpi_group_excl_ () __attribute__ ((weak, alias ("MPI_Group_excl")));

int MPI_Group_intersection() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_intersection () __attribute__ ((weak, alias ("MPI_Group_intersection")));
int mpi_group_intersection_ () __attribute__ ((weak, alias ("MPI_Group_intersection")));

int MPI_Group_range_excl() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_range_excl () __attribute__ ((weak, alias ("MPI_Group_range_excl")));
int mpi_group_range_excl_ () __attribute__ ((weak, alias ("MPI_Group_range_excl")));

int MPI_Group_range_incl() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_range_incl () __attribute__ ((weak, alias ("MPI_Group_range_incl")));
int mpi_group_range_incl_ () __attribute__ ((weak, alias ("MPI_Group_range_incl")));

int MPI_Group_translate_ranks() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_translate_ranks () __attribute__ ((weak, alias ("MPI_Group_translate_ranks")));
int mpi_group_translate_ranks_ () __attribute__ ((weak, alias ("MPI_Group_translate_ranks")));

int MPI_Group_union() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_union () __attribute__ ((weak, alias ("MPI_Group_union")));
int mpi_group_union_ () __attribute__ ((weak, alias ("MPI_Group_union")));

int MPI_Ibsend() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ibsend () __attribute__ ((weak, alias ("MPI_Ibsend")));
int mpi_ibsend_ () __attribute__ ((weak, alias ("MPI_Ibsend")));

int MPI_Improbe() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Improbe () __attribute__ ((weak, alias ("MPI_Improbe")));
int mpi_improbe_ () __attribute__ ((weak, alias ("MPI_Improbe")));

int MPI_Imrecv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Imrecv () __attribute__ ((weak, alias ("MPI_Imrecv")));
int mpi_imrecv_ () __attribute__ ((weak, alias ("MPI_Imrecv")));

int MPI_Info_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Info_create () __attribute__ ((weak, alias ("MPI_Info_create")));
int mpi_info_create_ () __attribute__ ((weak, alias ("MPI_Info_create")));

int MPI_Info_delete() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Info_delete () __attribute__ ((weak, alias ("MPI_Info_delete")));
int mpi_info_delete_ () __attribute__ ((weak, alias ("MPI_Info_delete")));

int MPI_Info_dup() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Info_dup () __attribute__ ((weak, alias ("MPI_Info_dup")));
int mpi_info_dup_ () __attribute__ ((weak, alias ("MPI_Info_dup")));

int MPI_Info_free() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Info_free () __attribute__ ((weak, alias ("MPI_Info_free")));
int mpi_info_free_ () __attribute__ ((weak, alias ("MPI_Info_free")));

int MPI_Info_get() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Info_get () __attribute__ ((weak, alias ("MPI_Info_get")));
int mpi_info_get_ () __attribute__ ((weak, alias ("MPI_Info_get")));

int MPI_Info_get_nkeys() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Info_get_nkeys () __attribute__ ((weak, alias ("MPI_Info_get_nkeys")));
int mpi_info_get_nkeys_ () __attribute__ ((weak, alias ("MPI_Info_get_nkeys")));

int MPI_Info_get_nthkey() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Info_get_nthkey () __attribute__ ((weak, alias ("MPI_Info_get_nthkey")));
int mpi_info_get_nthkey_ () __attribute__ ((weak, alias ("MPI_Info_get_nthkey")));

int MPI_Info_get_valuelen() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Info_get_valuelen () __attribute__ ((weak, alias ("MPI_Info_get_valuelen")));
int mpi_info_get_valuelen_ () __attribute__ ((weak, alias ("MPI_Info_get_valuelen")));

int MPI_Info_set() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Info_set () __attribute__ ((weak, alias ("MPI_Info_set")));
int mpi_info_set_ () __attribute__ ((weak, alias ("MPI_Info_set")));

int MPI_Intercomm_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Intercomm_create () __attribute__ ((weak, alias ("MPI_Intercomm_create")));
int mpi_intercomm_create_ () __attribute__ ((weak, alias ("MPI_Intercomm_create")));

int MPI_Intercomm_merge() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Intercomm_merge () __attribute__ ((weak, alias ("MPI_Intercomm_merge")));
int mpi_intercomm_merge_ () __attribute__ ((weak, alias ("MPI_Intercomm_merge")));

int MPI_Irsend() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Irsend () __attribute__ ((weak, alias ("MPI_Irsend")));
int mpi_irsend_ () __attribute__ ((weak, alias ("MPI_Irsend")));

int MPI_Issend() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Issend () __attribute__ ((weak, alias ("MPI_Issend")));
int mpi_issend_ () __attribute__ ((weak, alias ("MPI_Issend")));

int MPI_Is_thread_main() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Is_thread_main () __attribute__ ((weak, alias ("MPI_Is_thread_main")));
int mpi_is_thread_main_ () __attribute__ ((weak, alias ("MPI_Is_thread_main")));

int MPI_Keyval_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Keyval_create () __attribute__ ((weak, alias ("MPI_Keyval_create")));
int mpi_keyval_create_ () __attribute__ ((weak, alias ("MPI_Keyval_create")));

int MPI_Keyval_free() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Keyval_free () __attribute__ ((weak, alias ("MPI_Keyval_free")));
int mpi_keyval_free_ () __attribute__ ((weak, alias ("MPI_Keyval_free")));

int MPI_Lookup_name() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Lookup_name () __attribute__ ((weak, alias ("MPI_Lookup_name")));
int mpi_lookup_name_ () __attribute__ ((weak, alias ("MPI_Lookup_name")));

int MPI_Mprobe() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Mprobe () __attribute__ ((weak, alias ("MPI_Mprobe")));
int mpi_mprobe_ () __attribute__ ((weak, alias ("MPI_Mprobe")));

int MPI_Mrecv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Mrecv () __attribute__ ((weak, alias ("MPI_Mrecv")));
int mpi_mrecv_ () __attribute__ ((weak, alias ("MPI_Mrecv")));

int MPI_Neighbor_allgather() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Neighbor_allgather () __attribute__ ((weak, alias ("MPI_Neighbor_allgather")));
int mpi_neighbor_allgather_ () __attribute__ ((weak, alias ("MPI_Neighbor_allgather")));

int MPI_Ineighbor_allgather() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ineighbor_allgather () __attribute__ ((weak, alias ("MPI_Ineighbor_allgather")));
int mpi_ineighbor_allgather_ () __attribute__ ((weak, alias ("MPI_Ineighbor_allgather")));

int MPI_Neighbor_allgatherv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Neighbor_allgatherv () __attribute__ ((weak, alias ("MPI_Neighbor_allgatherv")));
int mpi_neighbor_allgatherv_ () __attribute__ ((weak, alias ("MPI_Neighbor_allgatherv")));

int MPI_Ineighbor_allgatherv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ineighbor_allgatherv () __attribute__ ((weak, alias ("MPI_Ineighbor_allgatherv")));
int mpi_ineighbor_allgatherv_ () __attribute__ ((weak, alias ("MPI_Ineighbor_allgatherv")));

int MPI_Neighbor_alltoall() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Neighbor_alltoall () __attribute__ ((weak, alias ("MPI_Neighbor_alltoall")));
int mpi_neighbor_alltoall_ () __attribute__ ((weak, alias ("MPI_Neighbor_alltoall")));

int MPI_Ineighbor_alltoall() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ineighbor_alltoall () __attribute__ ((weak, alias ("MPI_Ineighbor_alltoall")));
int mpi_ineighbor_alltoall_ () __attribute__ ((weak, alias ("MPI_Ineighbor_alltoall")));

int MPI_Neighbor_alltoallv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Neighbor_alltoallv () __attribute__ ((weak, alias ("MPI_Neighbor_alltoallv")));
int mpi_neighbor_alltoallv_ () __attribute__ ((weak, alias ("MPI_Neighbor_alltoallv")));

int MPI_Ineighbor_alltoallv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ineighbor_alltoallv () __attribute__ ((weak, alias ("MPI_Ineighbor_alltoallv")));
int mpi_ineighbor_alltoallv_ () __attribute__ ((weak, alias ("MPI_Ineighbor_alltoallv")));

int MPI_Neighbor_alltoallw() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Neighbor_alltoallw () __attribute__ ((weak, alias ("MPI_Neighbor_alltoallw")));
int mpi_neighbor_alltoallw_ () __attribute__ ((weak, alias ("MPI_Neighbor_alltoallw")));

int MPI_Ineighbor_alltoallw() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ineighbor_alltoallw () __attribute__ ((weak, alias ("MPI_Ineighbor_alltoallw")));
int mpi_ineighbor_alltoallw_ () __attribute__ ((weak, alias ("MPI_Ineighbor_alltoallw")));

int MPI_Op_commutative() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Op_commutative () __attribute__ ((weak, alias ("MPI_Op_commutative")));
int mpi_op_commutative_ () __attribute__ ((weak, alias ("MPI_Op_commutative")));

int MPI_Open_port() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Open_port () __attribute__ ((weak, alias ("MPI_Open_port")));
int mpi_open_port_ () __attribute__ ((weak, alias ("MPI_Open_port")));

int MPI_Op_free() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Op_free () __attribute__ ((weak, alias ("MPI_Op_free")));
int mpi_op_free_ () __attribute__ ((weak, alias ("MPI_Op_free")));

int MPI_Pack_external() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Pack_external () __attribute__ ((weak, alias ("MPI_Pack_external")));
int mpi_pack_external_ () __attribute__ ((weak, alias ("MPI_Pack_external")));

int MPI_Pack_external_size() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Pack_external_size () __attribute__ ((weak, alias ("MPI_Pack_external_size")));
int mpi_pack_external_size_ () __attribute__ ((weak, alias ("MPI_Pack_external_size")));

int MPI_Pack() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Pack () __attribute__ ((weak, alias ("MPI_Pack")));
int mpi_pack_ () __attribute__ ((weak, alias ("MPI_Pack")));

int MPI_Pack_size() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Pack_size () __attribute__ ((weak, alias ("MPI_Pack_size")));
int mpi_pack_size_ () __attribute__ ((weak, alias ("MPI_Pack_size")));

int MPI_Publish_name() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Publish_name () __attribute__ ((weak, alias ("MPI_Publish_name")));
int mpi_publish_name_ () __attribute__ ((weak, alias ("MPI_Publish_name")));

int MPI_Put() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Put () __attribute__ ((weak, alias ("MPI_Put")));
int mpi_put_ () __attribute__ ((weak, alias ("MPI_Put")));

int MPI_Query_thread() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Query_thread () __attribute__ ((weak, alias ("MPI_Query_thread")));
int mpi_query_thread_ () __attribute__ ((weak, alias ("MPI_Query_thread")));

int MPI_Raccumulate() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Raccumulate () __attribute__ ((weak, alias ("MPI_Raccumulate")));
int mpi_raccumulate_ () __attribute__ ((weak, alias ("MPI_Raccumulate")));

int MPI_Recv_init() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Recv_init () __attribute__ ((weak, alias ("MPI_Recv_init")));
int mpi_recv_init_ () __attribute__ ((weak, alias ("MPI_Recv_init")));

int MPI_Ireduce() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ireduce () __attribute__ ((weak, alias ("MPI_Ireduce")));
int mpi_ireduce_ () __attribute__ ((weak, alias ("MPI_Ireduce")));

int MPI_Reduce_local() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Reduce_local () __attribute__ ((weak, alias ("MPI_Reduce_local")));
int mpi_reduce_local_ () __attribute__ ((weak, alias ("MPI_Reduce_local")));

int MPI_Reduce_scatter() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Reduce_scatter () __attribute__ ((weak, alias ("MPI_Reduce_scatter")));
int mpi_reduce_scatter_ () __attribute__ ((weak, alias ("MPI_Reduce_scatter")));

int MPI_Ireduce_scatter() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ireduce_scatter () __attribute__ ((weak, alias ("MPI_Ireduce_scatter")));
int mpi_ireduce_scatter_ () __attribute__ ((weak, alias ("MPI_Ireduce_scatter")));

int MPI_Reduce_scatter_block() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Reduce_scatter_block () __attribute__ ((weak, alias ("MPI_Reduce_scatter_block")));
int mpi_reduce_scatter_block_ () __attribute__ ((weak, alias ("MPI_Reduce_scatter_block")));

int MPI_Ireduce_scatter_block() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ireduce_scatter_block () __attribute__ ((weak, alias ("MPI_Ireduce_scatter_block")));
int mpi_ireduce_scatter_block_ () __attribute__ ((weak, alias ("MPI_Ireduce_scatter_block")));

int MPI_Register_datarep() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Register_datarep () __attribute__ ((weak, alias ("MPI_Register_datarep")));
int mpi_register_datarep_ () __attribute__ ((weak, alias ("MPI_Register_datarep")));

int MPI_Request_free() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Request_free () __attribute__ ((weak, alias ("MPI_Request_free")));
int mpi_request_free_ () __attribute__ ((weak, alias ("MPI_Request_free")));

int MPI_Request_get_status() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Request_get_status () __attribute__ ((weak, alias ("MPI_Request_get_status")));
int mpi_request_get_status_ () __attribute__ ((weak, alias ("MPI_Request_get_status")));

int MPI_Rget() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Rget () __attribute__ ((weak, alias ("MPI_Rget")));
int mpi_rget_ () __attribute__ ((weak, alias ("MPI_Rget")));

int MPI_Rget_accumulate() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Rget_accumulate () __attribute__ ((weak, alias ("MPI_Rget_accumulate")));
int mpi_rget_accumulate_ () __attribute__ ((weak, alias ("MPI_Rget_accumulate")));

int MPI_Rput() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Rput () __attribute__ ((weak, alias ("MPI_Rput")));
int mpi_rput_ () __attribute__ ((weak, alias ("MPI_Rput")));

int MPI_Rsend() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Rsend () __attribute__ ((weak, alias ("MPI_Rsend")));
int mpi_rsend_ () __attribute__ ((weak, alias ("MPI_Rsend")));

int MPI_Rsend_init() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Rsend_init () __attribute__ ((weak, alias ("MPI_Rsend_init")));
int mpi_rsend_init_ () __attribute__ ((weak, alias ("MPI_Rsend_init")));

int MPI_Iscan() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Iscan () __attribute__ ((weak, alias ("MPI_Iscan")));
int mpi_iscan_ () __attribute__ ((weak, alias ("MPI_Iscan")));

int MPI_Iscatter() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Iscatter () __attribute__ ((weak, alias ("MPI_Iscatter")));
int mpi_iscatter_ () __attribute__ ((weak, alias ("MPI_Iscatter")));

int MPI_Iscatterv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Iscatterv () __attribute__ ((weak, alias ("MPI_Iscatterv")));
int mpi_iscatterv_ () __attribute__ ((weak, alias ("MPI_Iscatterv")));

int MPI_Send_init() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Send_init () __attribute__ ((weak, alias ("MPI_Send_init")));
int mpi_send_init_ () __attribute__ ((weak, alias ("MPI_Send_init")));

int MPI_Ssend_init() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ssend_init () __attribute__ ((weak, alias ("MPI_Ssend_init")));
int mpi_ssend_init_ () __attribute__ ((weak, alias ("MPI_Ssend_init")));

int MPI_Ssend() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Ssend () __attribute__ ((weak, alias ("MPI_Ssend")));
int mpi_ssend_ () __attribute__ ((weak, alias ("MPI_Ssend")));

int MPI_Start() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Start () __attribute__ ((weak, alias ("MPI_Start")));
int mpi_start_ () __attribute__ ((weak, alias ("MPI_Start")));

int MPI_Startall() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Startall () __attribute__ ((weak, alias ("MPI_Startall")));
int mpi_startall_ () __attribute__ ((weak, alias ("MPI_Startall")));

int MPI_Status_set_cancelled() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Status_set_cancelled () __attribute__ ((weak, alias ("MPI_Status_set_cancelled")));
int mpi_status_set_cancelled_ () __attribute__ ((weak, alias ("MPI_Status_set_cancelled")));

int MPI_Status_set_elements() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Status_set_elements () __attribute__ ((weak, alias ("MPI_Status_set_elements")));
int mpi_status_set_elements_ () __attribute__ ((weak, alias ("MPI_Status_set_elements")));

int MPI_Status_set_elements_x() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Status_set_elements_x () __attribute__ ((weak, alias ("MPI_Status_set_elements_x")));
int mpi_status_set_elements_x_ () __attribute__ ((weak, alias ("MPI_Status_set_elements_x")));

int MPI_Testall() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Testall () __attribute__ ((weak, alias ("MPI_Testall")));
int mpi_testall_ () __attribute__ ((weak, alias ("MPI_Testall")));

int MPI_Testany() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Testany () __attribute__ ((weak, alias ("MPI_Testany")));
int mpi_testany_ () __attribute__ ((weak, alias ("MPI_Testany")));

int MPI_Test_cancelled() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Test_cancelled () __attribute__ ((weak, alias ("MPI_Test_cancelled")));
int mpi_test_cancelled_ () __attribute__ ((weak, alias ("MPI_Test_cancelled")));

int MPI_Testsome() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Testsome () __attribute__ ((weak, alias ("MPI_Testsome")));
int mpi_testsome_ () __attribute__ ((weak, alias ("MPI_Testsome")));

int MPI_Topo_test() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Topo_test () __attribute__ ((weak, alias ("MPI_Topo_test")));
int mpi_topo_test_ () __attribute__ ((weak, alias ("MPI_Topo_test")));

int MPI_Type_create_darray() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_darray () __attribute__ ((weak, alias ("MPI_Type_create_darray")));
int mpi_type_create_darray_ () __attribute__ ((weak, alias ("MPI_Type_create_darray")));

int MPI_Type_create_f90_complex() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_f90_complex () __attribute__ ((weak, alias ("MPI_Type_create_f90_complex")));
int mpi_type_create_f90_complex_ () __attribute__ ((weak, alias ("MPI_Type_create_f90_complex")));

int MPI_Type_create_f90_integer() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_f90_integer () __attribute__ ((weak, alias ("MPI_Type_create_f90_integer")));
int mpi_type_create_f90_integer_ () __attribute__ ((weak, alias ("MPI_Type_create_f90_integer")));

int MPI_Type_create_f90_real() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_f90_real () __attribute__ ((weak, alias ("MPI_Type_create_f90_real")));
int mpi_type_create_f90_real_ () __attribute__ ((weak, alias ("MPI_Type_create_f90_real")));

int MPI_Type_create_hindexed_block() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_hindexed_block () __attribute__ ((weak, alias ("MPI_Type_create_hindexed_block")));
int mpi_type_create_hindexed_block_ () __attribute__ ((weak, alias ("MPI_Type_create_hindexed_block")));

int MPI_Type_create_hindexed() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_hindexed () __attribute__ ((weak, alias ("MPI_Type_create_hindexed")));
int mpi_type_create_hindexed_ () __attribute__ ((weak, alias ("MPI_Type_create_hindexed")));

int MPI_Type_create_hvector() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_hvector () __attribute__ ((weak, alias ("MPI_Type_create_hvector")));
int mpi_type_create_hvector_ () __attribute__ ((weak, alias ("MPI_Type_create_hvector")));

int MPI_Type_create_keyval() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_keyval () __attribute__ ((weak, alias ("MPI_Type_create_keyval")));
int mpi_type_create_keyval_ () __attribute__ ((weak, alias ("MPI_Type_create_keyval")));

int MPI_Type_create_indexed_block() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_indexed_block () __attribute__ ((weak, alias ("MPI_Type_create_indexed_block")));
int mpi_type_create_indexed_block_ () __attribute__ ((weak, alias ("MPI_Type_create_indexed_block")));

int MPI_Type_create_struct() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_struct () __attribute__ ((weak, alias ("MPI_Type_create_struct")));
int mpi_type_create_struct_ () __attribute__ ((weak, alias ("MPI_Type_create_struct")));

int MPI_Type_create_subarray() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_subarray () __attribute__ ((weak, alias ("MPI_Type_create_subarray")));
int mpi_type_create_subarray_ () __attribute__ ((weak, alias ("MPI_Type_create_subarray")));

int MPI_Type_create_resized() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_create_resized () __attribute__ ((weak, alias ("MPI_Type_create_resized")));
int mpi_type_create_resized_ () __attribute__ ((weak, alias ("MPI_Type_create_resized")));

int MPI_Type_delete_attr() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_delete_attr () __attribute__ ((weak, alias ("MPI_Type_delete_attr")));
int mpi_type_delete_attr_ () __attribute__ ((weak, alias ("MPI_Type_delete_attr")));

int MPI_Type_dup() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_dup () __attribute__ ((weak, alias ("MPI_Type_dup")));
int mpi_type_dup_ () __attribute__ ((weak, alias ("MPI_Type_dup")));

int MPI_Type_extent() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_extent () __attribute__ ((weak, alias ("MPI_Type_extent")));
int mpi_type_extent_ () __attribute__ ((weak, alias ("MPI_Type_extent")));

int MPI_Type_free_keyval() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_free_keyval () __attribute__ ((weak, alias ("MPI_Type_free_keyval")));
int mpi_type_free_keyval_ () __attribute__ ((weak, alias ("MPI_Type_free_keyval")));

int MPI_Type_get_attr() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_get_attr () __attribute__ ((weak, alias ("MPI_Type_get_attr")));
int mpi_type_get_attr_ () __attribute__ ((weak, alias ("MPI_Type_get_attr")));

int MPI_Type_get_contents() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_get_contents () __attribute__ ((weak, alias ("MPI_Type_get_contents")));
int mpi_type_get_contents_ () __attribute__ ((weak, alias ("MPI_Type_get_contents")));

int MPI_Type_get_envelope() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_get_envelope () __attribute__ ((weak, alias ("MPI_Type_get_envelope")));
int mpi_type_get_envelope_ () __attribute__ ((weak, alias ("MPI_Type_get_envelope")));

int MPI_Type_get_extent() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_get_extent () __attribute__ ((weak, alias ("MPI_Type_get_extent")));
int mpi_type_get_extent_ () __attribute__ ((weak, alias ("MPI_Type_get_extent")));

int MPI_Type_get_extent_x() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_get_extent_x () __attribute__ ((weak, alias ("MPI_Type_get_extent_x")));
int mpi_type_get_extent_x_ () __attribute__ ((weak, alias ("MPI_Type_get_extent_x")));

int MPI_Type_get_name() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_get_name () __attribute__ ((weak, alias ("MPI_Type_get_name")));
int mpi_type_get_name_ () __attribute__ ((weak, alias ("MPI_Type_get_name")));

int MPI_Type_get_true_extent() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_get_true_extent () __attribute__ ((weak, alias ("MPI_Type_get_true_extent")));
int mpi_type_get_true_extent_ () __attribute__ ((weak, alias ("MPI_Type_get_true_extent")));

int MPI_Type_get_true_extent_x() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_get_true_extent_x () __attribute__ ((weak, alias ("MPI_Type_get_true_extent_x")));
int mpi_type_get_true_extent_x_ () __attribute__ ((weak, alias ("MPI_Type_get_true_extent_x")));

int MPI_Type_hindexed() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_hindexed () __attribute__ ((weak, alias ("MPI_Type_hindexed")));
int mpi_type_hindexed_ () __attribute__ ((weak, alias ("MPI_Type_hindexed")));

int MPI_Type_hvector() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_hvector () __attribute__ ((weak, alias ("MPI_Type_hvector")));
int mpi_type_hvector_ () __attribute__ ((weak, alias ("MPI_Type_hvector")));

int MPI_Type_indexed() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_indexed () __attribute__ ((weak, alias ("MPI_Type_indexed")));
int mpi_type_indexed_ () __attribute__ ((weak, alias ("MPI_Type_indexed")));

int MPI_Type_lb() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_lb () __attribute__ ((weak, alias ("MPI_Type_lb")));
int mpi_type_lb_ () __attribute__ ((weak, alias ("MPI_Type_lb")));

int MPI_Type_match_size() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_match_size () __attribute__ ((weak, alias ("MPI_Type_match_size")));
int mpi_type_match_size_ () __attribute__ ((weak, alias ("MPI_Type_match_size")));

int MPI_Type_set_attr() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_set_attr () __attribute__ ((weak, alias ("MPI_Type_set_attr")));
int mpi_type_set_attr_ () __attribute__ ((weak, alias ("MPI_Type_set_attr")));

int MPI_Type_set_name() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_set_name () __attribute__ ((weak, alias ("MPI_Type_set_name")));
int mpi_type_set_name_ () __attribute__ ((weak, alias ("MPI_Type_set_name")));

int MPI_Type_size_x() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_size_x () __attribute__ ((weak, alias ("MPI_Type_size_x")));
int mpi_type_size_x_ () __attribute__ ((weak, alias ("MPI_Type_size_x")));

int MPI_Type_struct() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_struct () __attribute__ ((weak, alias ("MPI_Type_struct")));
int mpi_type_struct_ () __attribute__ ((weak, alias ("MPI_Type_struct")));

int MPI_Type_ub() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_ub () __attribute__ ((weak, alias ("MPI_Type_ub")));
int mpi_type_ub_ () __attribute__ ((weak, alias ("MPI_Type_ub")));

int MPI_Type_vector() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_vector () __attribute__ ((weak, alias ("MPI_Type_vector")));
int mpi_type_vector_ () __attribute__ ((weak, alias ("MPI_Type_vector")));

int MPI_Unpack() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Unpack () __attribute__ ((weak, alias ("MPI_Unpack")));
int mpi_unpack_ () __attribute__ ((weak, alias ("MPI_Unpack")));

int MPI_Unpublish_name() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Unpublish_name () __attribute__ ((weak, alias ("MPI_Unpublish_name")));
int mpi_unpublish_name_ () __attribute__ ((weak, alias ("MPI_Unpublish_name")));

int MPI_Unpack_external () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Unpack_external () __attribute__ ((weak, alias ("MPI_Unpack_external")));
int mpi_unpack_external_ () __attribute__ ((weak, alias ("MPI_Unpack_external")));

int MPI_Waitany() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Waitany () __attribute__ ((weak, alias ("MPI_Waitany")));
int mpi_waitany_ () __attribute__ ((weak, alias ("MPI_Waitany")));

int MPI_Waitsome() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Waitsome () __attribute__ ((weak, alias ("MPI_Waitsome")));
int mpi_waitsome_ () __attribute__ ((weak, alias ("MPI_Waitsome")));

int MPI_Win_allocate() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_allocate () __attribute__ ((weak, alias ("MPI_Win_allocate")));
int mpi_win_allocate_ () __attribute__ ((weak, alias ("MPI_Win_allocate")));

int MPI_Win_allocate_shared() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_allocate_shared () __attribute__ ((weak, alias ("MPI_Win_allocate_shared")));
int mpi_win_allocate_shared_ () __attribute__ ((weak, alias ("MPI_Win_allocate_shared")));

int MPI_Win_attach() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_attach () __attribute__ ((weak, alias ("MPI_Win_attach")));
int mpi_win_attach_ () __attribute__ ((weak, alias ("MPI_Win_attach")));

int MPI_Win_call_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_call_errhandler () __attribute__ ((weak, alias ("MPI_Win_call_errhandler")));
int mpi_win_call_errhandler_ () __attribute__ ((weak, alias ("MPI_Win_call_errhandler")));

int MPI_Win_complete() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_complete () __attribute__ ((weak, alias ("MPI_Win_complete")));
int mpi_win_complete_ () __attribute__ ((weak, alias ("MPI_Win_complete")));

int MPI_Win_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_create () __attribute__ ((weak, alias ("MPI_Win_create")));
int mpi_win_create_ () __attribute__ ((weak, alias ("MPI_Win_create")));

int MPI_Win_create_dynamic() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_create_dynamic () __attribute__ ((weak, alias ("MPI_Win_create_dynamic")));
int mpi_win_create_dynamic_ () __attribute__ ((weak, alias ("MPI_Win_create_dynamic")));

int MPI_Win_create_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_create_errhandler () __attribute__ ((weak, alias ("MPI_Win_create_errhandler")));
int mpi_win_create_errhandler_ () __attribute__ ((weak, alias ("MPI_Win_create_errhandler")));

int MPI_Win_create_keyval() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_create_keyval () __attribute__ ((weak, alias ("MPI_Win_create_keyval")));
int mpi_win_create_keyval_ () __attribute__ ((weak, alias ("MPI_Win_create_keyval")));

int MPI_Win_delete_attr() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_delete_attr () __attribute__ ((weak, alias ("MPI_Win_delete_attr")));
int mpi_win_delete_attr_ () __attribute__ ((weak, alias ("MPI_Win_delete_attr")));

int MPI_Win_detach() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_detach () __attribute__ ((weak, alias ("MPI_Win_detach")));
int mpi_win_detach_ () __attribute__ ((weak, alias ("MPI_Win_detach")));

int MPI_Win_fence() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_fence () __attribute__ ((weak, alias ("MPI_Win_fence")));
int mpi_win_fence_ () __attribute__ ((weak, alias ("MPI_Win_fence")));

int MPI_Win_flush() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_flush () __attribute__ ((weak, alias ("MPI_Win_flush")));
int mpi_win_flush_ () __attribute__ ((weak, alias ("MPI_Win_flush")));

int MPI_Win_flush_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_flush_all () __attribute__ ((weak, alias ("MPI_Win_flush_all")));
int mpi_win_flush_all_ () __attribute__ ((weak, alias ("MPI_Win_flush_all")));

int MPI_Win_flush_local() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_flush_local () __attribute__ ((weak, alias ("MPI_Win_flush_local")));
int mpi_win_flush_local_ () __attribute__ ((weak, alias ("MPI_Win_flush_local")));

int MPI_Win_flush_local_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_flush_local_all () __attribute__ ((weak, alias ("MPI_Win_flush_local_all")));
int mpi_win_flush_local_all_ () __attribute__ ((weak, alias ("MPI_Win_flush_local_all")));

int MPI_Win_free() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_free () __attribute__ ((weak, alias ("MPI_Win_free")));
int mpi_win_free_ () __attribute__ ((weak, alias ("MPI_Win_free")));

int MPI_Win_free_keyval() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_free_keyval () __attribute__ ((weak, alias ("MPI_Win_free_keyval")));
int mpi_win_free_keyval_ () __attribute__ ((weak, alias ("MPI_Win_free_keyval")));

int MPI_Win_get_attr() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_get_attr () __attribute__ ((weak, alias ("MPI_Win_get_attr")));
int mpi_win_get_attr_ () __attribute__ ((weak, alias ("MPI_Win_get_attr")));

int MPI_Win_get_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_get_errhandler () __attribute__ ((weak, alias ("MPI_Win_get_errhandler")));
int mpi_win_get_errhandler_ () __attribute__ ((weak, alias ("MPI_Win_get_errhandler")));

int MPI_Win_get_group() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_get_group () __attribute__ ((weak, alias ("MPI_Win_get_group")));
int mpi_win_get_group_ () __attribute__ ((weak, alias ("MPI_Win_get_group")));

int MPI_Win_get_info() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_get_info () __attribute__ ((weak, alias ("MPI_Win_get_info")));
int mpi_win_get_info_ () __attribute__ ((weak, alias ("MPI_Win_get_info")));

int MPI_Win_get_name() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_get_name () __attribute__ ((weak, alias ("MPI_Win_get_name")));
int mpi_win_get_name_ () __attribute__ ((weak, alias ("MPI_Win_get_name")));

int MPI_Win_lock() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_lock () __attribute__ ((weak, alias ("MPI_Win_lock")));
int mpi_win_lock_ () __attribute__ ((weak, alias ("MPI_Win_lock")));

int MPI_Win_lock_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_lock_all () __attribute__ ((weak, alias ("MPI_Win_lock_all")));
int mpi_win_lock_all_ () __attribute__ ((weak, alias ("MPI_Win_lock_all")));

int MPI_Win_post() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_post () __attribute__ ((weak, alias ("MPI_Win_post")));
int mpi_win_post_ () __attribute__ ((weak, alias ("MPI_Win_post")));

int MPI_Win_set_attr() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_set_attr () __attribute__ ((weak, alias ("MPI_Win_set_attr")));
int mpi_win_set_attr_ () __attribute__ ((weak, alias ("MPI_Win_set_attr")));

int MPI_Win_set_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_set_errhandler () __attribute__ ((weak, alias ("MPI_Win_set_errhandler")));
int mpi_win_set_errhandler_ () __attribute__ ((weak, alias ("MPI_Win_set_errhandler")));

int MPI_Win_set_info() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_set_info () __attribute__ ((weak, alias ("MPI_Win_set_info")));
int mpi_win_set_info_ () __attribute__ ((weak, alias ("MPI_Win_set_info")));

int MPI_Win_set_name() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_set_name () __attribute__ ((weak, alias ("MPI_Win_set_name")));
int mpi_win_set_name_ () __attribute__ ((weak, alias ("MPI_Win_set_name")));

int MPI_Win_shared_query() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_shared_query () __attribute__ ((weak, alias ("MPI_Win_shared_query")));
int mpi_win_shared_query_ () __attribute__ ((weak, alias ("MPI_Win_shared_query")));

int MPI_Win_start() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_start () __attribute__ ((weak, alias ("MPI_Win_start")));
int mpi_win_start_ () __attribute__ ((weak, alias ("MPI_Win_start")));

int MPI_Win_sync() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_sync () __attribute__ ((weak, alias ("MPI_Win_sync")));
int mpi_win_sync_ () __attribute__ ((weak, alias ("MPI_Win_sync")));

int MPI_Win_test() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_test () __attribute__ ((weak, alias ("MPI_Win_test")));
int mpi_win_test_ () __attribute__ ((weak, alias ("MPI_Win_test")));

int MPI_Win_unlock() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_unlock () __attribute__ ((weak, alias ("MPI_Win_unlock")));
int mpi_win_unlock_ () __attribute__ ((weak, alias ("MPI_Win_unlock")));

int MPI_Win_unlock_all() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_unlock_all () __attribute__ ((weak, alias ("MPI_Win_unlock_all")));
int mpi_win_unlock_all_ () __attribute__ ((weak, alias ("MPI_Win_unlock_all")));

int MPI_Win_wait() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Win_wait () __attribute__ ((weak, alias ("MPI_Win_wait")));
int mpi_win_wait_ () __attribute__ ((weak, alias ("MPI_Win_wait")));

double MPI_Wtick() {
  assert(0); 
  return -1; // To satisfy the compiler
}
double PMPI_Wtick () __attribute__ ((weak, alias ("MPI_Wtick")));
double mpi_wtick_ () __attribute__ ((weak, alias ("MPI_Wtick")));

int MPI_Cart_rank () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Cart_rank () __attribute__ ((weak, alias ("MPI_Cart_rank")));
int mpi_cart_rank_ () __attribute__ ((weak, alias ("MPI_Cart_rank")));

int MPI_Cartdim_get () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Cartdim_get () __attribute__ ((weak, alias ("MPI_Cartdim_get")));
int mpi_cartdim_get_ () __attribute__ ((weak, alias ("MPI_Cartdim_get")));

int MPI_Barrier () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Barrier () __attribute__ ((weak, alias ("MPI_Barrier")));
int mpi_barrier_ () __attribute__ ((weak, alias ("MPI_Barrier")));

int MPI_Comm_size () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_size () __attribute__ ((weak, alias ("MPI_Comm_size")));
int mpi_comm_size_ () __attribute__ ((weak, alias ("MPI_Comm_size")));

int MPI_Comm_rank () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_rank () __attribute__ ((weak, alias ("MPI_Comm_rank")));
int mpi_comm_rank_ () __attribute__ ((weak, alias ("MPI_Comm_rank")));

int MPI_Abort () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Abort () __attribute__ ((weak, alias ("MPI_Abort")));
int mpi_abort_ () __attribute__ ((weak, alias ("MPI_Abort")));

int MPI_Comm_dup () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_dup () __attribute__ ((weak, alias ("MPI_Comm_dup")));
int mpi_comm_dup_ () __attribute__ ((weak, alias ("MPI_Comm_dup")));

int MPI_Comm_compare () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_compare () __attribute__ ((weak, alias ("MPI_Comm_compare")));
int mpi_comm_compare_ () __attribute__ ((weak, alias ("MPI_Comm_compare")));

int MPI_Comm_free () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_free () __attribute__ ((weak, alias ("MPI_Comm_free")));
int mpi_comm_free_ () __attribute__ ((weak, alias ("MPI_Comm_free")));

int MPI_Comm_group () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_group () __attribute__ ((weak, alias ("MPI_Comm_group")));
int mpi_comm_group_ () __attribute__ ((weak, alias ("MPI_Comm_group")));

int MPI_Group_size () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_size () __attribute__ ((weak, alias ("MPI_Group_size")));
int mpi_group_size_ () __attribute__ ((weak, alias ("MPI_Group_size")));

int MPI_Group_free () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_free () __attribute__ ((weak, alias ("MPI_Group_free")));
int mpi_group_free_ () __attribute__ ((weak, alias ("MPI_Group_free")));

int MPI_Group_rank () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_rank () __attribute__ ((weak, alias ("MPI_Group_rank")));
int mpi_group_rank_ () __attribute__ ((weak, alias ("MPI_Group_rank")));

int MPI_Test () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Test () __attribute__ ((weak, alias ("MPI_Test")));
int mpi_test_ () __attribute__ ((weak, alias ("MPI_Test")));

int MPI_Wait () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Wait () __attribute__ ((weak, alias ("MPI_Wait")));
int mpi_wait_ () __attribute__ ((weak, alias ("MPI_Wait")));

int MPI_Type_size () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_size () __attribute__ ((weak, alias ("MPI_Type_size")));
int mpi_type_size_ () __attribute__ ((weak, alias ("MPI_Type_size")));

int MPI_Type_commit () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_commit () __attribute__ ((weak, alias ("MPI_Type_commit")));
int mpi_type_commit_ () __attribute__ ((weak, alias ("MPI_Type_commit")));

int MPI_Type_free () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_free () __attribute__ ((weak, alias ("MPI_Type_free")));
int mpi_type_free_ () __attribute__ ((weak, alias ("MPI_Type_free")));

int MPI_Init () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Init () __attribute__ ((weak, alias ("MPI_Init")));
int mpi_init_ () __attribute__ ((weak, alias ("MPI_Init")));

int MPI_Finalize () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Finalize () __attribute__ ((weak, alias ("MPI_Finalize")));
int mpi_finalize_ () __attribute__ ((weak, alias ("MPI_Finalize")));

int MPI_Finalized () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Finalized () __attribute__ ((weak, alias ("MPI_Finalized")));
int mpi_finalized_ () __attribute__ ((weak, alias ("MPI_Finalized")));

int MPI_Get_processor_name () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Get_processor_name () __attribute__ ((weak, alias ("MPI_Get_processor_name")));
int mpi_get_processor_name_ () __attribute__ ((weak, alias ("MPI_Get_processor_name")));

double MPI_Wtime () {
  assert(0); 
  return -1; // To satisfy the compiler
}
double PMPI_Wtime () __attribute__ ((weak, alias ("MPI_Wtime")));
double mpi_wtime_ () __attribute__ ((weak, alias ("MPI_Wtime")));

int MPI_Initialized () {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Initialized () __attribute__ ((weak, alias ("MPI_Initialized")));
int mpi_initialized_ () __attribute__ ((weak, alias ("MPI_Initialized")));

int MPI_Cart_coords() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Cart_coords () __attribute__ ((weak, alias ("MPI_Cart_coords")));
int mpi_cart_coords_ () __attribute__ ((weak, alias ("MPI_Cart_coords")));

int MPI_Cart_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Cart_create () __attribute__ ((weak, alias ("MPI_Cart_create")));
int mpi_cart_create_ () __attribute__ ((weak, alias ("MPI_Cart_create")));

int MPI_Cart_get() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Cart_get () __attribute__ ((weak, alias ("MPI_Cart_get")));
int mpi_cart_get_ () __attribute__ ((weak, alias ("MPI_Cart_get")));

int MPI_Cart_map() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Cart_map () __attribute__ ((weak, alias ("MPI_Cart_map")));
int mpi_cart_map_ () __attribute__ ((weak, alias ("MPI_Cart_map")));

int MPI_Cart_shift() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Cart_shift () __attribute__ ((weak, alias ("MPI_Cart_shift")));
int mpi_cart_shift_ () __attribute__ ((weak, alias ("MPI_Cart_shift")));

int MPI_Cart_sub() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Cart_sub () __attribute__ ((weak, alias ("MPI_Cart_sub")));
int mpi_cart_sub_ () __attribute__ ((weak, alias ("MPI_Cart_sub")));

int MPI_Bcast() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Bcast () __attribute__ ((weak, alias ("MPI_Bcast")));
int mpi_bcast_ () __attribute__ ((weak, alias ("MPI_Bcast")));

int MPI_Allreduce() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Allreduce () __attribute__ ((weak, alias ("MPI_Allreduce")));
int mpi_allreduce_ () __attribute__ ((weak, alias ("MPI_Allreduce")));

int MPI_Reduce() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Reduce () __attribute__ ((weak, alias ("MPI_Reduce")));
int mpi_reduce_ () __attribute__ ((weak, alias ("MPI_Reduce")));

int MPI_Alltoall() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Alltoall () __attribute__ ((weak, alias ("MPI_Alltoall")));
int mpi_alltoall_ () __attribute__ ((weak, alias ("MPI_Alltoall")));

int MPI_Alltoallv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Alltoallv () __attribute__ ((weak, alias ("MPI_Alltoallv")));
int mpi_alltoallv_ () __attribute__ ((weak, alias ("MPI_Alltoallv")));

int MPI_Allgather() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Allgather () __attribute__ ((weak, alias ("MPI_Allgather")));
int mpi_allgather_ () __attribute__ ((weak, alias ("MPI_Allgather")));

int MPI_Allgatherv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Allgatherv () __attribute__ ((weak, alias ("MPI_Allgatherv")));
int mpi_allgatherv_ () __attribute__ ((weak, alias ("MPI_Allgatherv")));

int MPI_Gather() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Gather () __attribute__ ((weak, alias ("MPI_Gather")));
int mpi_gather_ () __attribute__ ((weak, alias ("MPI_Gather")));

int MPI_Gatherv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Gatherv () __attribute__ ((weak, alias ("MPI_Gatherv")));
int mpi_gatherv_ () __attribute__ ((weak, alias ("MPI_Gatherv")));

int MPI_Scatter() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Scatter () __attribute__ ((weak, alias ("MPI_Scatter")));
int mpi_scatter_ () __attribute__ ((weak, alias ("MPI_Scatter")));

int MPI_Scatterv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Scatterv () __attribute__ ((weak, alias ("MPI_Scatterv")));
int mpi_scatterv_ () __attribute__ ((weak, alias ("MPI_Scatterv")));

int MPI_Scan() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Scan () __attribute__ ((weak, alias ("MPI_Scan")));
int mpi_scan_ () __attribute__ ((weak, alias ("MPI_Scan")));

int MPI_Comm_split() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_split () __attribute__ ((weak, alias ("MPI_Comm_split")));
int mpi_comm_split_ () __attribute__ ((weak, alias ("MPI_Comm_split")));

int MPI_Comm_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_create () __attribute__ ((weak, alias ("MPI_Comm_create")));
int mpi_comm_create_ () __attribute__ ((weak, alias ("MPI_Comm_create")));

int MPI_Comm_set_errhandler() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Comm_set_errhandler () __attribute__ ((weak, alias ("MPI_Comm_set_errhandler")));
int mpi_comm_set_errhandler_ () __attribute__ ((weak, alias ("MPI_Comm_set_errhandler")));

int MPI_Group_compare() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_compare () __attribute__ ((weak, alias ("MPI_Group_compare")));
int mpi_group_compare_ () __attribute__ ((weak, alias ("MPI_Group_compare")));

int MPI_Op_create() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Op_create () __attribute__ ((weak, alias ("MPI_Op_create")));
int mpi_op_create_ () __attribute__ ((weak, alias ("MPI_Op_create")));

int MPI_Send() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Send () __attribute__ ((weak, alias ("MPI_Send")));
int mpi_send_ () __attribute__ ((weak, alias ("MPI_Send")));

int MPI_Isend() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Isend () __attribute__ ((weak, alias ("MPI_Isend")));
int mpi_isend_ () __attribute__ ((weak, alias ("MPI_Isend")));

int MPI_Irecv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Irecv () __attribute__ ((weak, alias ("MPI_Irecv")));
int mpi_irecv_ () __attribute__ ((weak, alias ("MPI_Irecv")));

int MPI_Recv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Recv () __attribute__ ((weak, alias ("MPI_Recv")));
int mpi_recv_ () __attribute__ ((weak, alias ("MPI_Recv")));

int MPI_Sendrecv() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Sendrecv () __attribute__ ((weak, alias ("MPI_Sendrecv")));
int mpi_sendrecv_ () __attribute__ ((weak, alias ("MPI_Sendrecv")));

int MPI_Sendrecv_replace() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Sendrecv_replace () __attribute__ ((weak, alias ("MPI_Sendrecv_replace")));
int mpi_sendrecv_replace_ () __attribute__ ((weak, alias ("MPI_Sendrecv_replace")));

int MPI_Iprobe() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Iprobe () __attribute__ ((weak, alias ("MPI_Iprobe")));
int mpi_iprobe_ () __attribute__ ((weak, alias ("MPI_Iprobe")));

int MPI_Probe() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Probe () __attribute__ ((weak, alias ("MPI_Probe")));
int mpi_probe_ () __attribute__ ((weak, alias ("MPI_Probe")));

int MPI_Waitall() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Waitall () __attribute__ ((weak, alias ("MPI_Waitall")));
int mpi_waitall_ () __attribute__ ((weak, alias ("MPI_Waitall")));

int MPI_Type_contiguous() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Type_contiguous () __attribute__ ((weak, alias ("MPI_Type_contiguous")));
int mpi_type_contiguous_ () __attribute__ ((weak, alias ("MPI_Type_contiguous")));

int MPI_Init_thread() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Init_thread () __attribute__ ((weak, alias ("MPI_Init_thread")));
int mpi_init_thread_ () __attribute__ ((weak, alias ("MPI_Init_thread")));

int MPI_Get_count() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Get_count () __attribute__ ((weak, alias ("MPI_Get_count")));
int mpi_get_count_ () __attribute__ ((weak, alias ("MPI_Get_count")));

int MPI_Group_incl() {
  assert(0); 
  return -1; // To satisfy the compiler
}
int PMPI_Group_incl () __attribute__ ((weak, alias ("MPI_Group_incl")));
int mpi_group_incl_ () __attribute__ ((weak, alias ("MPI_Group_incl")));

