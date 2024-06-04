#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <nccl.h>

#include "common.h"
#include "dmtcp.h"
#include "switch_context.h"
#include "upper-half-wrappers.h"
// #include "upper-half-cuda-wrappers.h"
#include "nccl_autogen_wrappers.h"

#define REAL_FNC_1(fnc) \
  ({ fnc##_t fnc##Fnc = (fnc##_t) -1; \
  if (!initialized) { \
    initialize_wrappers(); \
  } \
  if (fnc##Fnc == (fnc##_t) -1) { \
    LhDlsym_t dlsymFptr = (LhDlsym_t)lhInfo.lhDlsym; \
    fnc##Fnc = (fnc##_t)dlsymFptr(Cuda_Fnc_##fnc); \
  } \
  fnc##Fnc; })

#undef ncclGetVersion
extern "C" ncclResult_t  ncclGetVersion(int *version)
{
    typedef ncclResult_t (*ncclGetVersion_t)(int *version);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclGetVersion)(version);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclGetUniqueId
extern "C" ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId)
{
    typedef ncclResult_t (*ncclGetUniqueId_t)(ncclUniqueId* uniqueId);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclGetUniqueId)(uniqueId);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommInitRankConfig
extern "C" ncclResult_t  ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config)
{
    typedef ncclResult_t (*ncclCommInitRankConfig_t)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommInitRankConfig)(comm, nranks, commId, rank, config);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommInitRank
extern "C" ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)
{
    typedef ncclResult_t (*ncclCommInitRank_t)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommInitRank)(comm, nranks, commId, rank);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommInitAll
extern "C" ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist)
{
    typedef ncclResult_t (*ncclCommInitAll_t)(ncclComm_t* comm, int ndev, const int* devlist);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommInitAll)(comm, ndev, devlist);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommFinalize
extern "C" ncclResult_t  ncclCommFinalize(ncclComm_t comm)
{
    typedef ncclResult_t (*ncclCommFinalize_t)(ncclComm_t comm);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommFinalize)(comm);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommDestroy
extern "C" ncclResult_t  ncclCommDestroy(ncclComm_t comm)
{
    typedef ncclResult_t (*ncclCommDestroy_t)(ncclComm_t comm);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommDestroy)(comm);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommAbort
extern "C" ncclResult_t  ncclCommAbort(ncclComm_t comm)
{
    typedef ncclResult_t (*ncclCommAbort_t)(ncclComm_t comm);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommAbort)(comm);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclGetErrorString
const char*  ncclGetErrorString(ncclResult_t result) 
{
    typedef const char* (*ncclGetErrorString_t)(ncclResult_t result);
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    char* ret_val = nullptr;
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = const_cast<char*>(REAL_FNC_1(ncclGetErrorString)(result));
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclGetLastError
extern "C" const char*  ncclGetLastError(ncclComm_t comm)
{
    typedef const char* (*ncclGetLastError_t)(ncclComm_t comm);
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    char* ret_val = nullptr;
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = const_cast<char*>(REAL_FNC_1(ncclGetLastError)(comm));
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommGetAsyncError
extern "C" ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError)
{
    typedef ncclResult_t (*ncclCommGetAsyncError_t)(ncclComm_t comm, ncclResult_t *asyncError);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommGetAsyncError)(comm, asyncError);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommCount
extern "C" ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count)
{
    typedef ncclResult_t (*ncclCommCount_t)(const ncclComm_t comm, int* count);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommCount)(comm, count);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommCuDevice
extern "C" ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device)
{
    typedef ncclResult_t (*ncclCommCuDevice_t)(const ncclComm_t comm, int* device);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommCuDevice)(comm, device);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclCommUserRank
extern "C" ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank)
{
    typedef ncclResult_t (*ncclCommUserRank_t)(const ncclComm_t comm, int* rank);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclCommUserRank)(comm, rank);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclRedOpCreatePreMulSum
extern "C" ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm)
{
    typedef ncclResult_t (*ncclRedOpCreatePreMulSum_t)(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclRedOpCreatePreMulSum)(op, scalar, datatype, residence, comm);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclRedOpDestroy
extern "C" ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) 
{
    typedef ncclResult_t (*ncclRedOpDestroy_t)(ncclRedOp_t op, ncclComm_t comm);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclRedOpDestroy)(op, comm);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclReduce
extern "C" ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
    typedef ncclResult_t (*ncclReduce_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclReduce)(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclBcast
extern "C" ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    typedef ncclResult_t (*ncclBcast_t)(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclBcast)(buff, count, datatype, root, comm, stream);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclBroadcast
extern "C" ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    typedef ncclResult_t (*ncclBroadcast_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclBroadcast)(sendbuff, recvbuff, count, datatype, root, comm, stream);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclAllReduce
extern "C" ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    typedef ncclResult_t (*ncclAllReduce_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclAllReduce)(sendbuff, recvbuff, count, datatype, op, comm, stream);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclReduceScatter
extern "C" ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    typedef ncclResult_t (*ncclReduceScatter_t)(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclReduceScatter)(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclAllGather
extern "C" ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) 
{
    typedef ncclResult_t (*ncclAllGather_t)(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclAllGather)(sendbuff, recvbuff, sendcount, datatype, comm, stream);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}
 
#undef ncclSend
extern "C" ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) 
{
    typedef ncclResult_t (*ncclSend_t)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclSend)(sendbuff, count, datatype, peer, comm, stream);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclRecv
extern "C" ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) 
{
    typedef ncclResult_t (*ncclRecv_t)(const void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclRecv)(recvbuff, count, datatype, peer, comm, stream);
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclGroupStart
extern "C" ncclResult_t  ncclGroupStart() 
{
    typedef ncclResult_t (*ncclGroupStart_t)();
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclGroupStart)();
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

#undef ncclGroupEnd
extern "C" ncclResult_t  ncclGroupEnd()
{
    typedef ncclResult_t (*ncclGroupEnd_t)();
    ncclResult_t ret_val = ncclSuccess;
    printf("Called at func '%s' in line %i.\n", __FUNCTION__, __LINE__ );
    DMTCP_PLUGIN_DISABLE_CKPT();
    JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr); //replay_gpu_status();
    ret_val = REAL_FNC_1(ncclGroupEnd)();
    RETURN_TO_UPPER_HALF();
    /* Insert logging code here */
    DMTCP_PLUGIN_ENABLE_CKPT();
    return ret_val;
}

