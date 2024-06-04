#ifndef NCCL_AUTOGEN_WRAPPERS_H
#define NCCL_AUTOGEN_WRAPPERS_H

#include <nccl.h>

extern "C" ncclResult_t  ncclGetVersion(int *version) __attribute__((weak));
#define ncclGetVersion(version) (ncclGetVersion ? ncclGetVersion(version) : 0)

extern "C" ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId) __attribute__((weak));
#define ncclGetUniqueId(uniqueId) (ncclGetUniqueId ? ncclGetUniqueId(uniqueId) : 0)

extern "C" ncclResult_t  ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) __attribute__((weak));
#define ncclCommInitRankConfig(comm, nranks, commId, rank, config) (ncclCommInitRankConfig ? ncclCommInitRankConfig(comm, nranks, commId, rank, config) : 0)

extern "C" ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) __attribute__((weak));
#define ncclCommInitRank(comm, nranks, commId, rank) (ncclCommInitRank ? ncclCommInitRank(comm, nranks, commId, rank) : 0)

extern "C" ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) __attribute__((weak));
#define ncclCommInitAll(comm, ndev, devlist) (ncclCommInitAll ? ncclCommInitAll(comm, ndev, devlist) : 0)

extern "C" ncclResult_t  ncclCommFinalize(ncclComm_t comm) __attribute__((weak));
#define ncclCommFinalize(comm) (ncclCommFinalize ? ncclCommFinalize(comm) : 0)

extern "C" ncclResult_t  ncclCommDestroy(ncclComm_t comm) __attribute__((weak));
#define ncclCommDestroy(comm) (ncclCommDestroy ? ncclCommDestroy(comm) : 0)

extern "C" ncclResult_t  ncclCommAbort(ncclComm_t comm) __attribute__((weak));
#define ncclCommAbort(comm) (ncclCommAbort ? ncclCommAbort(comm) : 0)

extern "C" const char*  ncclGetErrorString(ncclResult_t result) __attribute__((weak));
#define ncclGetErrorString(result) (ncclGetErrorString ? ncclGetErrorString(result) : 0)

extern "C" const char*  ncclGetLastError(ncclComm_t comm) __attribute__((weak));
#define ncclGetLastError(comm) (ncclGetLastError ? ncclGetLastError(comm) : 0)

extern "C" ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) __attribute__((weak));
#define ncclCommGetAsyncError(comm, asyncError) (ncclCommGetAsyncError ? ncclCommGetAsyncError(comm, asyncError) : 0)

extern "C" ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count) __attribute__((weak));
#define ncclCommCount(comm, count) (ncclCommCount ? ncclCommCount(comm, count) : 0)

extern "C" ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device) __attribute__((weak));
#define ncclCommCuDevice(comm, device) (ncclCommCuDevice ? ncclCommCuDevice(comm, device) : 0)

extern "C" ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank) __attribute__((weak));
#define ncclCommUserRank(comm, rank) (ncclCommUserRank ? ncclCommUserRank(comm, rank) : 0)

extern "C" ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) __attribute__((weak));
#define ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm) (ncclRedOpCreatePreMulSum ? ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm) : 0)

extern "C" ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) __attribute__((weak));
#define ncclRedOpDestroy(op, comm) (ncclRedOpDestroy ? ncclRedOpDestroy(op, comm) : 0)

extern "C" ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
#define ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream) (ncclReduce ? ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream) : 0)

extern "C" ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
#define ncclBcast(buff, count, datatype, root, comm, stream) (ncclBcast ? ncclBcast(buff, count, datatype, root, comm, stream) : 0)

extern "C" ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
#define ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream) (ncclBroadcast ? ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream) : 0)

extern "C" ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));      
#define ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream) (ncclAllReduce ? ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream) : 0)

extern "C" ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
#define ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream) (ncclReduceScatter ? ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream) : 0)

extern "C" ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
#define ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream) (ncclAllGather ? ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream) : 0)

extern "C" ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
#define ncclSend(sendbuff, count, datatype, peer, comm, stream) (ncclSend ? ncclSend(sendbuff, count, datatype, peer, comm, stream) : 0)

extern "C" ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) __attribute__((weak));
#define ncclRecv(recvbuff, count, datatype, peer, comm, stream) (ncclRecv ? ncclRecv(recvbuff, count, datatype, peer, comm, stream) : 0)

extern "C" ncclResult_t  ncclGroupStart() __attribute__((weak));
#define ncclGroupStart() (ncclGroupStart ? ncclGroupStart() : 0)

extern "C" ncclResult_t  ncclGroupEnd() __attribute__((weak));
#define ncclGroupEnd() (ncclGroupEnd ? ncclGroupEnd() : 0)

#endif // NCCL_AUTOGEN_WRAPPERS_H