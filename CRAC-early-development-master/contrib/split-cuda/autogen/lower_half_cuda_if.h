#ifndef LOWER_HALF_CUDA_IF_H
#define LOWER_HALF_CUDA_IF_H


#define FOREACH_FNC(MACRO) \
MACRO(cudaCreateTextureObject) ,\
MACRO(cudaDestroyTextureObject) ,\
MACRO(cudaBindTexture) ,\
MACRO(cudaBindTexture2D) ,\
MACRO(cudaBindTextureToArray) ,\
MACRO(cudaUnbindTexture) ,\
MACRO(cudaCreateChannelDesc) ,\
MACRO(cudaEventCreate) ,\
MACRO(cudaEventCreateWithFlags) ,\
MACRO(cudaEventDestroy) ,\
MACRO(cudaEventElapsedTime) ,\
MACRO(cudaEventQuery) ,\
MACRO(cudaEventRecord) ,\
MACRO(cudaEventSynchronize) ,\
MACRO(cudaMalloc) ,\
MACRO(cudaFree) ,\
MACRO(cudaMallocArray) ,\
MACRO(cudaFreeArray) ,\
MACRO(cudaHostRegister) ,\
MACRO(cudaDeviceGetAttribute) ,\
MACRO(cudaMallocHost) ,\
MACRO(cudaFreeHost) ,\
MACRO(cudaHostAlloc) ,\
MACRO(cudaMallocPitch) ,\
MACRO(cudaGetDevice) ,\
MACRO(cudaSetDevice) ,\
MACRO(cudaDeviceGetLimit) ,\
MACRO(cudaDeviceSetLimit) ,\
MACRO(cudaGetDeviceCount) ,\
MACRO(cudaDeviceSetCacheConfig) ,\
MACRO(cudaGetDeviceProperties) ,\
MACRO(cudaDeviceCanAccessPeer) ,\
MACRO(cudaDeviceGetPCIBusId) ,\
MACRO(cudaDeviceReset) ,\
MACRO(cudaDeviceSynchronize) ,\
MACRO(cudaLaunchKernel) ,\
MACRO(cudaMallocManaged) ,\
MACRO(cudaMemcpy) ,\
MACRO(cudaMemcpy2D) ,\
MACRO(cudaMemcpyToArray) ,\
MACRO(cudaMemcpyToSymbol) ,\
MACRO(cudaMemcpyToSymbolAsync) ,\
MACRO(cudaMemcpyAsync) ,\
MACRO(cudaMemset) ,\
MACRO(cudaMemset2D) ,\
MACRO(cudaMemsetAsync) ,\
MACRO(cudaMemGetInfo) ,\
MACRO(cudaMemAdvise) ,\
MACRO(cudaMemPrefetchAsync) ,\
MACRO(cudaStreamCreate) ,\
MACRO(cudaStreamCreateWithPriority) ,\
MACRO(cudaStreamCreateWithFlags) ,\
MACRO(cudaStreamIsCapturing) ,\
MACRO(cudaStreamGetCaptureInfo) ,\
MACRO(cudaStreamDestroy) ,\
MACRO(cudaStreamSynchronize) ,\
MACRO(cudaStreamWaitEvent) ,\
MACRO(cudaThreadSynchronize) ,\
MACRO(cudaThreadExit) ,\
MACRO(cudaPointerGetAttributes) ,\
MACRO(cudaGetErrorString) ,\
MACRO(cudaGetErrorName) ,\
MACRO(cudaGetLastError) ,\
MACRO(cudaPeekAtLastError) ,\
MACRO(cudaFuncSetCacheConfig) ,\
MACRO(__cudaInitModule) ,\
MACRO(__cudaPopCallConfiguration) ,\
MACRO(__cudaPushCallConfiguration) ,\
MACRO(__cudaRegisterFatBinary) ,\
MACRO(__cudaUnregisterFatBinary) ,\
MACRO(__cudaRegisterFunction) ,\
MACRO(__cudaRegisterManagedVar) ,\
MACRO(__cudaRegisterTexture) ,\
MACRO(__cudaRegisterSurface) ,\
MACRO(__cudaRegisterVar) ,\
MACRO(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) ,\
MACRO(cudaFuncGetAttributes) ,\
MACRO(cublasCreate_v2) ,\
MACRO(cublasSgemm_v2) ,\
MACRO(cublasSgemmStridedBatched) ,\
MACRO(cublasLtCreate) ,\
MACRO(cublasLtDestroy) ,\
MACRO(cublasLtMatmul) ,\
MACRO(cublasLtMatmulAlgoGetHeuristic) ,\
MACRO(cublasSetStream_v2) ,\
MACRO(cublasSetMathMode) ,\
MACRO(cublasGetMathMode) ,\
MACRO(cublasDdot_v2) ,\
MACRO(cublasDestroy_v2) ,\
MACRO(cublasDaxpy_v2) ,\
MACRO(cublasDasum_v2) ,\
MACRO(cublasDgemm_v2) ,\
MACRO(cublasDgemv_v2) ,\
MACRO(cublasDnrm2_v2) ,\
MACRO(cublasDscal_v2) ,\
MACRO(cublasDswap_v2) ,\
MACRO(cublasIdamax_v2) ,\
MACRO(cublasSscal_v2),\
MACRO(cublasZdotc_v2),\
MACRO(cublasInit) ,\
MACRO(cublasShutdown) ,\
MACRO(cublasGetError) ,\
MACRO(cublasAlloc) ,\
MACRO(cublasFree) ,\
MACRO(cublasSetKernelStream) ,\
MACRO(cublasSnrm2) ,\
MACRO(cublasDnrm2) ,\
MACRO(cublasScnrm2) ,\
MACRO(cublasDznrm2) ,\
MACRO(cublasSdot) ,\
MACRO(cublasDdot) ,\
MACRO(cublasCdotu) ,\
MACRO(cublasCdotc) ,\
MACRO(cublasZdotu) ,\
MACRO(cublasZdotc) ,\
MACRO(cublasSscal) ,\
MACRO(cublasDscal) ,\
MACRO(cublasCscal) ,\
MACRO(cublasZscal) ,\
MACRO(cublasCsscal) ,\
MACRO(cublasZdscal) ,\
MACRO(cublasSaxpy) ,\
MACRO(cublasDaxpy) ,\
MACRO(cublasCaxpy) ,\
MACRO(cublasZaxpy) ,\
MACRO(cublasScopy) ,\
MACRO(cublasDcopy) ,\
MACRO(cublasCcopy) ,\
MACRO(cublasZcopy) ,\
MACRO(cublasSswap) ,\
MACRO(cublasDswap) ,\
MACRO(cublasCswap) ,\
MACRO(cublasZswap) ,\
MACRO(cublasIsamax) ,\
MACRO(cublasIdamax) ,\
MACRO(cublasIcamax) ,\
MACRO(cublasIzamax) ,\
MACRO(cublasIsamin) ,\
MACRO(cublasIdamin) ,\
MACRO(cublasIcamin) ,\
MACRO(cublasIzamin) ,\
MACRO(cublasSasum) ,\
MACRO(cublasDasum) ,\
MACRO(cublasScasum) ,\
MACRO(cublasDzasum) ,\
MACRO(cublasSrot) ,\
MACRO(cublasDrot) ,\
MACRO(cublasCrot) ,\
MACRO(cublasZrot) ,\
MACRO(cublasCsrot) ,\
MACRO(cublasZdrot) ,\
MACRO(cublasSrotg) ,\
MACRO(cublasDrotg) ,\
MACRO(cublasCrotg) ,\
MACRO(cublasZrotg) ,\
MACRO(cublasSrotm) ,\
MACRO(cublasDrotm) ,\
MACRO(cublasSrotmg) ,\
MACRO(cublasDrotmg) ,\
MACRO(cublasSgemv) ,\
MACRO(cublasDgemv) ,\
MACRO(cublasCgemv) ,\
MACRO(cublasZgemv) ,\
MACRO(cublasSgbmv) ,\
MACRO(cublasDgbmv) ,\
MACRO(cublasCgbmv) ,\
MACRO(cublasZgbmv) ,\
MACRO(cublasStrmv) ,\
MACRO(cublasDtrmv) ,\
MACRO(cublasCtrmv) ,\
MACRO(cublasZtrmv) ,\
MACRO(cublasStbmv) ,\
MACRO(cublasDtbmv) ,\
MACRO(cublasCtbmv) ,\
MACRO(cublasZtbmv) ,\
MACRO(cublasStpmv) ,\
MACRO(cublasDtpmv) ,\
MACRO(cublasCtpmv) ,\
MACRO(cublasZtpmv) ,\
MACRO(cublasStrsv) ,\
MACRO(cublasDtrsv) ,\
MACRO(cublasCtrsv) ,\
MACRO(cublasZtrsv) ,\
MACRO(cublasStpsv) ,\
MACRO(cublasDtpsv) ,\
MACRO(cublasCtpsv) ,\
MACRO(cublasZtpsv) ,\
MACRO(cublasStbsv) ,\
MACRO(cublasDtbsv) ,\
MACRO(cublasCtbsv) ,\
MACRO(cublasZtbsv) ,\
MACRO(cublasSsymv) ,\
MACRO(cublasDsymv) ,\
MACRO(cublasChemv) ,\
MACRO(cublasZhemv) ,\
MACRO(cublasSsbmv) ,\
MACRO(cublasDsbmv) ,\
MACRO(cublasChbmv) ,\
MACRO(cublasZhbmv) ,\
MACRO(cublasSspmv) ,\
MACRO(cublasDspmv) ,\
MACRO(cublasChpmv) ,\
MACRO(cublasZhpmv) ,\
MACRO(cublasSger) ,\
MACRO(cublasDger) ,\
MACRO(cublasCgeru) ,\
MACRO(cublasCgerc) ,\
MACRO(cublasZgeru) ,\
MACRO(cublasZgerc) ,\
MACRO(cublasSsyr) ,\
MACRO(cublasDsyr) ,\
MACRO(cublasCher) ,\
MACRO(cublasZher) ,\
MACRO(cublasSspr) ,\
MACRO(cublasDspr) ,\
MACRO(cublasChpr) ,\
MACRO(cublasZhpr) ,\
MACRO(cublasSsyr2) ,\
MACRO(cublasDsyr2) ,\
MACRO(cublasCher2) ,\
MACRO(cublasZher2) ,\
MACRO(cublasSspr2) ,\
MACRO(cublasDspr2) ,\
MACRO(cublasChpr2) ,\
MACRO(cublasZhpr2) ,\
MACRO(cublasSgemm) ,\
MACRO(cublasDgemm) ,\
MACRO(cublasCgemm) ,\
MACRO(cublasZgemm) ,\
MACRO(cublasSsyrk) ,\
MACRO(cublasDsyrk) ,\
MACRO(cublasCsyrk) ,\
MACRO(cublasZsyrk) ,\
MACRO(cublasCherk) ,\
MACRO(cublasZherk) ,\
MACRO(cublasSsyr2k) ,\
MACRO(cublasDsyr2k) ,\
MACRO(cublasCsyr2k) ,\
MACRO(cublasZsyr2k) ,\
MACRO(cublasCher2k) ,\
MACRO(cublasZher2k) ,\
MACRO(cublasSsymm) ,\
MACRO(cublasDsymm) ,\
MACRO(cublasCsymm) ,\
MACRO(cublasZsymm) ,\
MACRO(cublasChemm) ,\
MACRO(cublasZhemm) ,\
MACRO(cublasStrsm) ,\
MACRO(cublasDtrsm) ,\
MACRO(cublasCtrsm) ,\
MACRO(cublasZtrsm) ,\
MACRO(cublasStrmm) ,\
MACRO(cublasDtrmm) ,\
MACRO(cublasCtrmm) ,\
MACRO(cublasZtrmm) ,\
MACRO(cublasSetMatrix) ,\
MACRO(cublasGetMatrix) ,\
MACRO(cublasSetMatrixAsync) ,\
MACRO(cublasGetMatrixAsync) ,\
MACRO(cublasSetVector) ,\
MACRO(cublasGetVector) ,\
MACRO(cublasSetVectorAsync) ,\
MACRO(cublasGetVectorAsync) ,\
MACRO(cusparseCreate) ,\
MACRO(cusparseSetStream) ,\
MACRO(cusparseCreateMatDescr) ,\
MACRO(cusparseSetMatType) ,\
MACRO(cusparseSetMatIndexBase) ,\
MACRO(cusparseDestroy) ,\
MACRO(cusparseDestroyMatDescr) ,\
MACRO(cusparseGetMatType) ,\
MACRO(cusparseSetMatFillMode) ,\
MACRO(cusparseGetMatFillMode) ,\
MACRO(cusparseSetMatDiagType) ,\
MACRO(cusparseGetMatDiagType) ,\
MACRO(cusparseGetMatIndexBase) ,\
MACRO(cusparseSetPointerMode) ,\
MACRO(cusolverDnCreate) ,\
MACRO(cusolverDnDestroy) ,\
MACRO(cusolverDnSetStream) ,\
MACRO(cusolverDnGetStream) ,\
MACRO(cusolverDnDgetrf_bufferSize) ,\
MACRO(cusolverDnDgetrf) ,\
MACRO(cusolverDnDgetrs) ,\
MACRO(cusolverDnDpotrf_bufferSize) ,\
MACRO(cusolverDnDpotrf) ,\
MACRO(cusolverDnDpotrs) ,\
MACRO(cuInit) ,\
MACRO(cuDriverGetVersion) ,\
MACRO(cuDeviceGet) ,\
MACRO(cuGetProcAddress) ,\
MACRO(cuDeviceGetAttribute) ,\
MACRO(cuDeviceGetCount) ,\
MACRO(cuDeviceGetName) ,\
MACRO(cuDeviceGetUuid) ,\
MACRO(cuDeviceTotalMem_v2) ,\
MACRO(cuDeviceComputeCapability) ,\
MACRO(cuDeviceGetProperties) ,\
MACRO(cuDevicePrimaryCtxGetState) ,\
MACRO(cuDevicePrimaryCtxRelease_v2) ,\
MACRO(cuDevicePrimaryCtxReset_v2) ,\
MACRO(cuDevicePrimaryCtxRetain) ,\
MACRO(cuCtxCreate_v2) ,\
MACRO(cuCtxDestroy_v2) ,\
MACRO(cuCtxGetApiVersion) ,\
MACRO(cuCtxGetCacheConfig) ,\
MACRO(cuCtxGetCurrent) ,\
MACRO(cuCtxGetDevice) ,\
MACRO(cuCtxGetFlags) ,\
MACRO(cuCtxGetLimit) ,\
MACRO(cuCtxGetSharedMemConfig) ,\
MACRO(cuCtxGetStreamPriorityRange) ,\
MACRO(cuCtxPopCurrent_v2) ,\
MACRO(cuCtxPushCurrent_v2) ,\
MACRO(cuCtxSetCacheConfig) ,\
MACRO(cuCtxSetCurrent) ,\
MACRO(cuCtxSetLimit) ,\
MACRO(cuCtxSetSharedMemConfig) ,\
MACRO(cuCtxSynchronize) ,\
MACRO(cuCtxAttach) ,\
MACRO(cuCtxDetach) ,\
MACRO(cuLinkAddData_v2) ,\
MACRO(cuLinkAddFile_v2) ,\
MACRO(cuLinkComplete) ,\
MACRO(cuLinkCreate_v2) ,\
MACRO(cuLinkDestroy) ,\
MACRO(cuModuleGetFunction) ,\
MACRO(cuModuleGetGlobal_v2) ,\
MACRO(cuModuleGetSurfRef) ,\
MACRO(cuModuleGetTexRef) ,\
MACRO(cuModuleLoad) ,\
MACRO(cuModuleLoadData) ,\
MACRO(cuModuleLoadDataEx) ,\
MACRO(cuModuleLoadFatBinary) ,\
MACRO(cuModuleUnload) ,\
MACRO(cuArray3DCreate_v2) ,\
MACRO(cuArray3DGetDescriptor_v2) ,\
MACRO(cuArrayCreate_v2) ,\
MACRO(cuArrayDestroy) ,\
MACRO(cuArrayGetDescriptor_v2) ,\
MACRO(cuDeviceGetByPCIBusId) ,\
MACRO(cuDeviceGetPCIBusId) ,\
MACRO(cuIpcCloseMemHandle) ,\
MACRO(cuIpcGetEventHandle) ,\
MACRO(cuIpcGetMemHandle) ,\
MACRO(cuIpcOpenEventHandle) ,\
MACRO(cuMemAlloc_v2) ,\
MACRO(cuMemAllocHost_v2) ,\
MACRO(cuMemAllocManaged) ,\
MACRO(cuMemAllocPitch_v2) ,\
MACRO(cuMemFree_v2) ,\
MACRO(cuMemFreeHost) ,\
MACRO(cuMemGetAddressRange_v2) ,\
MACRO(cuMemGetInfo_v2) ,\
MACRO(cuMemHostAlloc) ,\
MACRO(cuMemHostGetDevicePointer_v2) ,\
MACRO(cuMemHostGetFlags) ,\
MACRO(cuMemHostRegister_v2) ,\
MACRO(cuMemHostUnregister) ,\
MACRO(cuMemcpy) ,\
MACRO(cuMemcpy2D_v2) ,\
MACRO(cuMemcpy2DAsync_v2) ,\
MACRO(cuMemcpy2DUnaligned_v2) ,\
MACRO(cuMemcpy3D_v2) ,\
MACRO(cuMemcpy3DAsync_v2) ,\
MACRO(cuMemcpy3DPeer) ,\
MACRO(cuMemcpy3DPeerAsync) ,\
MACRO(cuMemcpyAsync) ,\
MACRO(cuMemcpyAtoA_v2) ,\
MACRO(cuMemcpyAtoD_v2) ,\
MACRO(cuMemcpyAtoH_v2) ,\
MACRO(cuMemcpyAtoHAsync_v2) ,\
MACRO(cuMemcpyDtoA_v2) ,\
MACRO(cuMemcpyDtoD_v2) ,\
MACRO(cuMemcpyDtoDAsync_v2) ,\
MACRO(cuMemcpyDtoH_v2) ,\
MACRO(cuMemcpyDtoHAsync_v2) ,\
MACRO(cuMemcpyHtoA_v2) ,\
MACRO(cuMemcpyHtoAAsync_v2) ,\
MACRO(cuMemcpyHtoD_v2) ,\
MACRO(cuMemcpyHtoDAsync_v2) ,\
MACRO(cuMemcpyPeer) ,\
MACRO(cuMemcpyPeerAsync) ,\
MACRO(cuMemsetD16_v2) ,\
MACRO(cuMemsetD16Async) ,\
MACRO(cuMemsetD2D16_v2) ,\
MACRO(cuMemsetD2D16Async) ,\
MACRO(cuMemsetD2D32_v2) ,\
MACRO(cuMemsetD2D32Async) ,\
MACRO(cuMemsetD2D8_v2) ,\
MACRO(cuMemsetD2D8Async) ,\
MACRO(cuMemsetD32_v2) ,\
MACRO(cuMemsetD32Async) ,\
MACRO(cuMemsetD8_v2) ,\
MACRO(cuMemsetD8Async) ,\
MACRO(cuMipmappedArrayCreate) ,\
MACRO(cuMipmappedArrayDestroy) ,\
MACRO(cuMipmappedArrayGetLevel) ,\
MACRO(cuMemAdvise) ,\
MACRO(cuMemPrefetchAsync) ,\
MACRO(cuMemRangeGetAttribute) ,\
MACRO(cuMemRangeGetAttributes) ,\
MACRO(cuPointerGetAttribute) ,\
MACRO(cuPointerGetAttributes) ,\
MACRO(cuPointerSetAttribute) ,\
MACRO(cuStreamAddCallback) ,\
MACRO(cuStreamAttachMemAsync) ,\
MACRO(cuStreamCreate) ,\
MACRO(cuStreamCreateWithPriority) ,\
MACRO(cuStreamDestroy_v2) ,\
MACRO(cuStreamEndCapture) ,\
MACRO(cuStreamGetCtx) ,\
MACRO(cuStreamGetFlags) ,\
MACRO(cuStreamGetPriority) ,\
MACRO(cuStreamIsCapturing) ,\
MACRO(cuStreamQuery) ,\
MACRO(cuStreamSynchronize) ,\
MACRO(cuStreamWaitEvent) ,\
MACRO(cuEventCreate) ,\
MACRO(cuEventDestroy_v2) ,\
MACRO(cuEventElapsedTime) ,\
MACRO(cuEventQuery) ,\
MACRO(cuEventRecord) ,\
MACRO(cuEventSynchronize) ,\
MACRO(cuDestroyExternalMemory) ,\
MACRO(cuDestroyExternalSemaphore) ,\
MACRO(cuExternalMemoryGetMappedBuffer) ,\
MACRO(cuExternalMemoryGetMappedMipmappedArray) ,\
MACRO(cuImportExternalMemory) ,\
MACRO(cuImportExternalSemaphore) ,\
MACRO(cuSignalExternalSemaphoresAsync) ,\
MACRO(cuWaitExternalSemaphoresAsync) ,\
MACRO(cuStreamBatchMemOp) ,\
MACRO(cuStreamWaitValue32) ,\
MACRO(cuStreamWaitValue64) ,\
MACRO(cuStreamWriteValue32) ,\
MACRO(cuStreamWriteValue64) ,\
MACRO(cuFuncGetAttribute) ,\
MACRO(cuFuncSetAttribute) ,\
MACRO(cuFuncSetCacheConfig) ,\
MACRO(cuFuncSetSharedMemConfig) ,\
MACRO(cuLaunchCooperativeKernel) ,\
MACRO(cuLaunchCooperativeKernelMultiDevice) ,\
MACRO(cuLaunchHostFunc) ,\
MACRO(cuLaunchKernel) ,\
MACRO(cuFuncSetBlockShape) ,\
MACRO(cuFuncSetSharedSize) ,\
MACRO(cuLaunch) ,\
MACRO(cuLaunchGrid) ,\
MACRO(cuLaunchGridAsync) ,\
MACRO(cuParamSetSize) ,\
MACRO(cuParamSetTexRef) ,\
MACRO(cuParamSetf) ,\
MACRO(cuParamSeti) ,\
MACRO(cuParamSetv) ,\
MACRO(cuGraphCreate) ,\
MACRO(cuGraphDestroy) ,\
MACRO(cuGraphDestroyNode) ,\
MACRO(cuGraphExecDestroy) ,\
MACRO(cuGraphGetEdges) ,\
MACRO(cuGraphGetNodes) ,\
MACRO(cuGraphGetRootNodes) ,\
MACRO(cuGraphHostNodeGetParams) ,\
MACRO(cuGraphHostNodeSetParams) ,\
MACRO(cuGraphKernelNodeGetParams) ,\
MACRO(cuGraphKernelNodeSetParams) ,\
MACRO(cuGraphLaunch) ,\
MACRO(cuGraphMemcpyNodeGetParams) ,\
MACRO(cuGraphMemcpyNodeSetParams) ,\
MACRO(cuGraphMemsetNodeGetParams) ,\
MACRO(cuGraphMemsetNodeSetParams) ,\
MACRO(cuGraphNodeFindInClone) ,\
MACRO(cuGraphNodeGetDependencies) ,\
MACRO(cuGraphNodeGetDependentNodes) ,\
MACRO(cuGraphNodeGetType) ,\
MACRO(cuOccupancyMaxActiveBlocksPerMultiprocessor) ,\
MACRO(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) ,\
MACRO(cuOccupancyMaxPotentialBlockSize) ,\
MACRO(cuOccupancyMaxPotentialBlockSizeWithFlags) ,\
MACRO(cuTexRefCreate) ,\
MACRO(cuTexRefDestroy) ,\
MACRO(cuTexRefGetAddress_v2) ,\
MACRO(cuTexRefGetAddressMode) ,\
MACRO(cuTexRefGetArray) ,\
MACRO(cuTexRefGetBorderColor) ,\
MACRO(cuTexRefGetFilterMode) ,\
MACRO(cuTexRefGetFlags) ,\
MACRO(cuTexRefGetFormat) ,\
MACRO(cuTexRefGetMaxAnisotropy) ,\
MACRO(cuTexRefGetMipmapFilterMode) ,\
MACRO(cuTexRefGetMipmapLevelBias) ,\
MACRO(cuTexRefGetMipmapLevelClamp) ,\
MACRO(cuTexRefGetMipmappedArray) ,\
MACRO(cuTexRefSetAddress_v2) ,\
MACRO(cuTexRefSetAddress2D_v3) ,\
MACRO(cuTexRefSetAddressMode) ,\
MACRO(cuTexRefSetArray) ,\
MACRO(cuTexRefSetBorderColor) ,\
MACRO(cuTexRefSetFilterMode) ,\
MACRO(cuTexRefSetFlags) ,\
MACRO(cuTexRefSetFormat) ,\
MACRO(cuTexRefSetMaxAnisotropy) ,\
MACRO(cuTexRefSetMipmapFilterMode) ,\
MACRO(cuTexRefSetMipmapLevelBias) ,\
MACRO(cuTexRefSetMipmapLevelClamp) ,\
MACRO(cuTexRefSetMipmappedArray) ,\
MACRO(cuSurfRefGetArray) ,\
MACRO(cuSurfRefSetArray) ,\
MACRO(cuTexObjectCreate) ,\
MACRO(cuTexObjectDestroy) ,\
MACRO(cuTexObjectGetResourceDesc) ,\
MACRO(cuTexObjectGetResourceViewDesc) ,\
MACRO(cuTexObjectGetTextureDesc) ,\
MACRO(cuSurfObjectCreate) ,\
MACRO(cuSurfObjectDestroy) ,\
MACRO(cuSurfObjectGetResourceDesc) ,\
MACRO(cuCtxDisablePeerAccess) ,\
MACRO(cuCtxEnablePeerAccess) ,\
MACRO(cuDeviceCanAccessPeer) ,\
MACRO(cuDeviceGetP2PAttribute) ,\
MACRO(cuGraphicsMapResources) ,\
MACRO(cuGraphicsResourceGetMappedMipmappedArray) ,\
MACRO(cuGraphicsResourceGetMappedPointer_v2) ,\
MACRO(cuGraphicsResourceSetMapFlags_v2) ,\
MACRO(cuGraphicsSubResourceGetMappedArray) ,\
MACRO(cuGraphicsUnmapResources) ,\
MACRO(cuGraphicsUnregisterResource) ,\
MACRO(cuIpcOpenMemHandle_v2), \
MACRO(cufftPlan1d) ,\
MACRO(cufftPlan2d) ,\
MACRO(cufftPlan3d) ,\
MACRO(cufftPlanMany) ,\
MACRO(cufftMakePlan1d) ,\
MACRO(cufftMakePlan2d) ,\
MACRO(cufftMakePlan3d) ,\
MACRO(cufftMakePlanMany) ,\
MACRO(cufftMakePlanMany64) ,\
MACRO(cufftGetSizeMany64) ,\
MACRO(cufftEstimate1d) ,\
MACRO(cufftEstimate2d) ,\
MACRO(cufftEstimate3d) ,\
MACRO(cufftEstimateMany) ,\
MACRO(cufftCreate) ,\
MACRO(cufftGetSize1d) ,\
MACRO(cufftGetSize2d) ,\
MACRO(cufftGetSize3d) ,\
MACRO(cufftGetSizeMany) ,\
MACRO(cufftGetSize) ,\
MACRO(cufftSetWorkArea) ,\
MACRO(cufftSetAutoAllocation) ,\
MACRO(cufftExecC2C) ,\
MACRO(cufftExecR2C) ,\
MACRO(cufftExecC2R) ,\
MACRO(cufftExecZ2Z) ,\
MACRO(cufftExecD2Z) ,\
MACRO(cufftExecZ2D) ,\
MACRO(cufftSetStream) ,\
MACRO(cufftDestroy) ,\
MACRO(cufftGetVersion) ,\
MACRO(cufftGetProperty) ,\
MACRO(__cudaRegisterFatBinaryEnd) ,\
MACRO(cudaStreamGetCaptureInfo_v2) ,\
MACRO(cudaUserObjectCreate) ,\
MACRO(cudaGraphRetainUserObject) ,\
MACRO(cudaStreamUpdateCaptureDependencies) ,\
MACRO(cudaGetDriverEntryPoint) ,\
MACRO(cudaFuncSetAttribute) ,\
MACRO(cudaIpcGetMemHandle) ,\
MACRO(cudaIpcOpenMemHandle) ,\
MACRO(cudaIpcCloseMemHandle) ,\
MACRO(cudaDriverGetVersion) ,\
MACRO(cudaDeviceGetByPCIBusId) ,\
MACRO(cudaThreadExchangeStreamCaptureMode) ,\
MACRO(cudaHostGetDevicePointer) ,\
MACRO(cudaHostUnregister) ,\
MACRO(cudaGraphAddEventWaitNode) ,\
MACRO(cudaGraphAddEventRecordNode) ,\
MACRO(cudaLaunchHostFunc) ,\
MACRO(cudaGraphAddHostNode) ,\
MACRO(cudaDeviceEnablePeerAccess) ,\
MACRO(cudaGraphAddKernelNode) ,\
MACRO(cuGetErrorName) ,\
MACRO(ncclCommInitRank), \
MACRO(ncclCommInitAll), \
MACRO(ncclCommDestroy), \
MACRO(ncclCommAbort), \
MACRO(ncclGetErrorString), \
MACRO(ncclCommGetAsyncError), \
MACRO(ncclCommCount), \
MACRO(ncclCommCuDevice), \
MACRO(ncclCommUserRank), \
MACRO(ncclReduce), \
MACRO(ncclBcast), \
MACRO(ncclBroadcast), \
MACRO(ncclAllReduce), \
MACRO(ncclReduceScatter), \
MACRO(ncclAllGather), \
MACRO(ncclSend), \
MACRO(ncclRecv), \
MACRO(ncclGroupStart), \
MACRO(ncclGroupEnd), \
MACRO(ncclGetUniqueId), \
MACRO(ncclGetVersion), \
MACRO(cublasGemmEx), \
MACRO(cublasGemmStridedBatchedEx), 


#define GENERATE_ENUM(ENUM) Cuda_Fnc_##ENUM

#define GENERATE_FNC_PTR(FNC) ((void*)&FNC)

typedef enum __Cuda_Fncs {
  Cuda_Fnc_NULL,
  FOREACH_FNC(GENERATE_ENUM)
  Cuda_Fnc_Invalid,
} Cuda_Fncs_t;

static const char *cuda_Fnc_to_str[]  __attribute__((used)) =
{
  "Cuda_Fnc_NULL", 
  "cudaCreateTextureObject",
  "cudaDestroyTextureObject",
  "cudaBindTexture",
  "cudaBindTexture2D",
  "cudaBindTextureToArray",
  "cudaUnbindTexture",
  "cudaCreateChannelDesc",
  "cudaEventCreate",
  "cudaEventCreateWithFlags",
  "cudaEventDestroy",
  "cudaEventElapsedTime",
  "cudaEventQuery",
  "cudaEventRecord",
  "cudaEventSynchronize",
  "cudaMalloc",
  "cudaFree",
  "cudaMallocArray",
  "cudaFreeArray",
  "cudaHostRegister",
  "cudaDeviceGetAttribute",
  "cudaMallocHost",
  "cudaFreeHost",
  "cudaHostAlloc",
  "cudaMallocPitch",
  "cudaGetDevice",
  "cudaSetDevice",
  "cudaDeviceGetLimit",
  "cudaDeviceSetLimit",
  "cudaGetDeviceCount",
  "cudaDeviceSetCacheConfig",
  "cudaGetDeviceProperties",
  "cudaDeviceCanAccessPeer",
  "cudaDeviceGetPCIBusId",
  "cudaDeviceReset",
  "cudaDeviceSynchronize",
  "cudaLaunchKernel",
  "cudaMallocManaged",
  "cudaMemcpy",
  "cudaMemcpy2D",
  "cudaMemcpyToArray",
  "cudaMemcpyToSymbol",
  "cudaMemcpyToSymbolAsync",
  "cudaMemcpyAsync",
  "cudaMemset",
  "cudaMemset2D",
  "cudaMemsetAsync",
  "cudaMemGetInfo",
  "cudaMemAdvise",
  "cudaMemPrefetchAsync",
  "cudaStreamCreate",
  "cudaStreamCreateWithPriority",
  "cudaStreamCreateWithFlags",
  "cudaStreamIsCapturing",
  "cudaStreamGetCaptureInfo",
  "cudaStreamDestroy",
  "cudaStreamSynchronize",
  "cudaStreamWaitEvent",
  "cudaThreadSynchronize",
  "cudaThreadExit",
  "cudaPointerGetAttributes",
  "cudaGetErrorString",
  "cudaGetErrorName",
  "cudaGetLastError",
  "cudaPeekAtLastError",
  "cudaFuncSetCacheConfig",
  "__cudaInitModule",
  "__cudaPopCallConfiguration",
  "__cudaPushCallConfiguration",
  "__cudaRegisterFatBinary",
  "__cudaUnregisterFatBinary",
  "__cudaRegisterFunction",
  "__cudaRegisterManagedVar",
  "__cudaRegisterTexture",
  "__cudaRegisterSurface",
  "__cudaRegisterVar",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "cudaFuncGetAttributes",
  "cublasCreate_v2",
  "cublasSgemm_v2",
  "cublasSgemmStridedBatched",
  "cublasLtCreate",
  "cublasLtDestroy",
  "cublasLtMatmul",
  "cublasLtMatmulAlgoGetHeuristic",
  "cublasSetStream_v2",
  "cublasSetMathMode",
  "cublasGetMathMode",
  "cublasDdot_v2",
  "cublasDestroy_v2",
  "cublasDaxpy_v2",
  "cublasDasum_v2",
  "cublasDgemm_v2",
  "cublasDgemv_v2",
  "cublasDnrm2_v2",
  "cublasDscal_v2",
  "cublasDswap_v2",
  "cublasIdamax_v2",
  "cublasSscal_v2",
  "cublasZdotc_v2",
  "cublasInit",
  "cublasShutdown",
  "cublasGetError",
  "cublasAlloc",
  "cublasFree",
  "cublasSetKernelStream",
  "cublasSnrm2",
  "cublasDnrm2",
  "cublasScnrm2",
  "cublasDznrm2",
  "cublasSdot",
  "cublasDdot",
  "cublasCdotu",
  "cublasCdotc",
  "cublasZdotu",
  "cublasZdotc",
  "cublasSscal",
  "cublasDscal",
  "cublasCscal",
  "cublasZscal",
  "cublasCsscal",
  "cublasZdscal",
  "cublasSaxpy",
  "cublasDaxpy",
  "cublasCaxpy",
  "cublasZaxpy",
  "cublasScopy",
  "cublasDcopy",
  "cublasCcopy",
  "cublasZcopy",
  "cublasSswap",
  "cublasDswap",
  "cublasCswap",
  "cublasZswap",
  "cublasIsamax",
  "cublasIdamax",
  "cublasIcamax",
  "cublasIzamax",
  "cublasIsamin",
  "cublasIdamin",
  "cublasIcamin",
  "cublasIzamin",
  "cublasSasum",
  "cublasDasum",
  "cublasScasum",
  "cublasDzasum",
  "cublasSrot",
  "cublasDrot",
  "cublasCrot",
  "cublasZrot",
  "cublasCsrot",
  "cublasZdrot",
  "cublasSrotg",
  "cublasDrotg",
  "cublasCrotg",
  "cublasZrotg",
  "cublasSrotm",
  "cublasDrotm",
  "cublasSrotmg",
  "cublasDrotmg",
  "cublasSgemv",
  "cublasDgemv",
  "cublasCgemv",
  "cublasZgemv",
  "cublasSgbmv",
  "cublasDgbmv",
  "cublasCgbmv",
  "cublasZgbmv",
  "cublasStrmv",
  "cublasDtrmv",
  "cublasCtrmv",
  "cublasZtrmv",
  "cublasStbmv",
  "cublasDtbmv",
  "cublasCtbmv",
  "cublasZtbmv",
  "cublasStpmv",
  "cublasDtpmv",
  "cublasCtpmv",
  "cublasZtpmv",
  "cublasStrsv",
  "cublasDtrsv",
  "cublasCtrsv",
  "cublasZtrsv",
  "cublasStpsv",
  "cublasDtpsv",
  "cublasCtpsv",
  "cublasZtpsv",
  "cublasStbsv",
  "cublasDtbsv",
  "cublasCtbsv",
  "cublasZtbsv",
  "cublasSsymv",
  "cublasDsymv",
  "cublasChemv",
  "cublasZhemv",
  "cublasSsbmv",
  "cublasDsbmv",
  "cublasChbmv",
  "cublasZhbmv",
  "cublasSspmv",
  "cublasDspmv",
  "cublasChpmv",
  "cublasZhpmv",
  "cublasSger",
  "cublasDger",
  "cublasCgeru",
  "cublasCgerc",
  "cublasZgeru",
  "cublasZgerc",
  "cublasSsyr",
  "cublasDsyr",
  "cublasCher",
  "cublasZher",
  "cublasSspr",
  "cublasDspr",
  "cublasChpr",
  "cublasZhpr",
  "cublasSsyr2",
  "cublasDsyr2",
  "cublasCher2",
  "cublasZher2",
  "cublasSspr2",
  "cublasDspr2",
  "cublasChpr2",
  "cublasZhpr2",
  "cublasSgemm",
  "cublasDgemm",
  "cublasCgemm",
  "cublasZgemm",
  "cublasSsyrk",
  "cublasDsyrk",
  "cublasCsyrk",
  "cublasZsyrk",
  "cublasCherk",
  "cublasZherk",
  "cublasSsyr2k",
  "cublasDsyr2k",
  "cublasCsyr2k",
  "cublasZsyr2k",
  "cublasCher2k",
  "cublasZher2k",
  "cublasSsymm",
  "cublasDsymm",
  "cublasCsymm",
  "cublasZsymm",
  "cublasChemm",
  "cublasZhemm",
  "cublasStrsm",
  "cublasDtrsm",
  "cublasCtrsm",
  "cublasZtrsm",
  "cublasStrmm",
  "cublasDtrmm",
  "cublasCtrmm",
  "cublasZtrmm",
  "cublasSetMatrix",
  "cublasGetMatrix",
  "cublasSetMatrixAsync",
  "cublasGetMatrixAsync",
  "cublasSetVector",
  "cublasGetVector",
  "cublasSetVectorAsync",
  "cublasGetVectorAsync",
  "cusparseCreate",
  "cusparseSetStream",
  "cusparseCreateMatDescr",
  "cusparseSetMatType",
  "cusparseSetMatIndexBase",
  "cusparseDestroy",
  "cusparseDestroyMatDescr",
  "cusparseGetMatType",
  "cusparseSetMatFillMode",
  "cusparseGetMatFillMode",
  "cusparseSetMatDiagType",
  "cusparseGetMatDiagType",
  "cusparseGetMatIndexBase",
  "cusparseSetPointerMode",
  "cusolverDnCreate",
  "cusolverDnDestroy",
  "cusolverDnSetStream",
  "cusolverDnGetStream",
  "cusolverDnDgetrf_bufferSize",
  "cusolverDnDgetrf",
  "cusolverDnDgetrs",
  "cusolverDnDpotrf_bufferSize",
  "cusolverDnDpotrf",
  "cusolverDnDpotrs",
  "cuInit",
  "cuDriverGetVersion",
  "cuDeviceGet",
  "cuGetProcAddress",
  "cuDeviceGetAttribute",
  "cuDeviceGetCount",
  "cuDeviceGetName",
  "cuDeviceGetUuid",
  "cuDeviceTotalMem_v2",
  "cuDeviceComputeCapability",
  "cuDeviceGetProperties",
  "cuDevicePrimaryCtxGetState",
  "cuDevicePrimaryCtxRelease_v2",
  "cuDevicePrimaryCtxReset_v2",
  "cuDevicePrimaryCtxRetain",
  "cuCtxCreate_v2",
  "cuCtxDestroy_v2",
  "cuCtxGetApiVersion",
  "cuCtxGetCacheConfig",
  "cuCtxGetCurrent",
  "cuCtxGetDevice",
  "cuCtxGetFlags",
  "cuCtxGetLimit",
  "cuCtxGetSharedMemConfig",
  "cuCtxGetStreamPriorityRange",
  "cuCtxPopCurrent_v2",
  "cuCtxPushCurrent_v2",
  "cuCtxSetCacheConfig",
  "cuCtxSetCurrent",
  "cuCtxSetLimit",
  "cuCtxSetSharedMemConfig",
  "cuCtxSynchronize",
  "cuCtxAttach",
  "cuCtxDetach",
  "cuLinkAddData_v2",
  "cuLinkAddFile_v2",
  "cuLinkComplete",
  "cuLinkCreate_v2",
  "cuLinkDestroy",
  "cuModuleGetFunction",
  "cuModuleGetGlobal_v2",
  "cuModuleGetSurfRef",
  "cuModuleGetTexRef",
  "cuModuleLoad",
  "cuModuleLoadData",
  "cuModuleLoadDataEx",
  "cuModuleLoadFatBinary",
  "cuModuleUnload",
  "cuArray3DCreate_v2",
  "cuArray3DGetDescriptor_v2",
  "cuArrayCreate_v2",
  "cuArrayDestroy",
  "cuArrayGetDescriptor_v2",
  "cuDeviceGetByPCIBusId",
  "cuDeviceGetPCIBusId",
  "cuIpcCloseMemHandle",
  "cuIpcGetEventHandle",
  "cuIpcGetMemHandle",
  "cuIpcOpenEventHandle",
  "cuMemAlloc_v2",
  "cuMemAllocHost_v2",
  "cuMemAllocManaged",
  "cuMemAllocPitch_v2",
  "cuMemFree_v2",
  "cuMemFreeHost",
  "cuMemGetAddressRange_v2",
  "cuMemGetInfo_v2",
  "cuMemHostAlloc",
  "cuMemHostGetDevicePointer_v2",
  "cuMemHostGetFlags",
  "cuMemHostRegister_v2",
  "cuMemHostUnregister",
  "cuMemcpy",
  "cuMemcpy2D_v2",
  "cuMemcpy2DAsync_v2",
  "cuMemcpy2DUnaligned_v2",
  "cuMemcpy3D_v2",
  "cuMemcpy3DAsync_v2",
  "cuMemcpy3DPeer",
  "cuMemcpy3DPeerAsync",
  "cuMemcpyAsync",
  "cuMemcpyAtoA_v2",
  "cuMemcpyAtoD_v2",
  "cuMemcpyAtoH_v2",
  "cuMemcpyAtoHAsync_v2",
  "cuMemcpyDtoA_v2",
  "cuMemcpyDtoD_v2",
  "cuMemcpyDtoDAsync_v2",
  "cuMemcpyDtoH_v2",
  "cuMemcpyDtoHAsync_v2",
  "cuMemcpyHtoA_v2",
  "cuMemcpyHtoAAsync_v2",
  "cuMemcpyHtoD_v2",
  "cuMemcpyHtoDAsync_v2",
  "cuMemcpyPeer",
  "cuMemcpyPeerAsync",
  "cuMemsetD16_v2",
  "cuMemsetD16Async",
  "cuMemsetD2D16_v2",
  "cuMemsetD2D16Async",
  "cuMemsetD2D32_v2",
  "cuMemsetD2D32Async",
  "cuMemsetD2D8_v2",
  "cuMemsetD2D8Async",
  "cuMemsetD32_v2",
  "cuMemsetD32Async",
  "cuMemsetD8_v2",
  "cuMemsetD8Async",
  "cuMipmappedArrayCreate",
  "cuMipmappedArrayDestroy",
  "cuMipmappedArrayGetLevel",
  "cuMemAdvise",
  "cuMemPrefetchAsync",
  "cuMemRangeGetAttribute",
  "cuMemRangeGetAttributes",
  "cuPointerGetAttribute",
  "cuPointerGetAttributes",
  "cuPointerSetAttribute",
  "cuStreamAddCallback",
  "cuStreamAttachMemAsync",
  "cuStreamCreate",
  "cuStreamCreateWithPriority",
  "cuStreamDestroy_v2",
  "cuStreamEndCapture",
  "cuStreamGetCtx",
  "cuStreamGetFlags",
  "cuStreamGetPriority",
  "cuStreamIsCapturing",
  "cuStreamQuery",
  "cuStreamSynchronize",
  "cuStreamWaitEvent",
  "cuEventCreate",
  "cuEventDestroy_v2",
  "cuEventElapsedTime",
  "cuEventQuery",
  "cuEventRecord",
  "cuEventSynchronize",
  "cuDestroyExternalMemory",
  "cuDestroyExternalSemaphore",
  "cuExternalMemoryGetMappedBuffer",
  "cuExternalMemoryGetMappedMipmappedArray",
  "cuImportExternalMemory",
  "cuImportExternalSemaphore",
  "cuSignalExternalSemaphoresAsync",
  "cuWaitExternalSemaphoresAsync",
  "cuStreamBatchMemOp",
  "cuStreamWaitValue32",
  "cuStreamWaitValue64",
  "cuStreamWriteValue32",
  "cuStreamWriteValue64",
  "cuFuncGetAttribute",
  "cuFuncSetAttribute",
  "cuFuncSetCacheConfig",
  "cuFuncSetSharedMemConfig",
  "cuLaunchCooperativeKernel",
  "cuLaunchCooperativeKernelMultiDevice",
  "cuLaunchHostFunc",
  "cuLaunchKernel",
  "cuFuncSetBlockShape",
  "cuFuncSetSharedSize",
  "cuLaunch",
  "cuLaunchGrid",
  "cuLaunchGridAsync",
  "cuParamSetSize",
  "cuParamSetTexRef",
  "cuParamSetf",
  "cuParamSeti",
  "cuParamSetv",
  "cuGraphCreate",
  "cuGraphDestroy",
  "cuGraphDestroyNode",
  "cuGraphExecDestroy",
  "cuGraphGetEdges",
  "cuGraphGetNodes",
  "cuGraphGetRootNodes",
  "cuGraphHostNodeGetParams",
  "cuGraphHostNodeSetParams",
  "cuGraphKernelNodeGetParams",
  "cuGraphKernelNodeSetParams",
  "cuGraphLaunch",
  "cuGraphMemcpyNodeGetParams",
  "cuGraphMemcpyNodeSetParams",
  "cuGraphMemsetNodeGetParams",
  "cuGraphMemsetNodeSetParams",
  "cuGraphNodeFindInClone",
  "cuGraphNodeGetDependencies",
  "cuGraphNodeGetDependentNodes",
  "cuGraphNodeGetType",
  "cuOccupancyMaxActiveBlocksPerMultiprocessor",
  "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "cuOccupancyMaxPotentialBlockSize",
  "cuOccupancyMaxPotentialBlockSizeWithFlags",
  "cuTexRefCreate",
  "cuTexRefDestroy",
  "cuTexRefGetAddress_v2",
  "cuTexRefGetAddressMode",
  "cuTexRefGetArray",
  "cuTexRefGetBorderColor",
  "cuTexRefGetFilterMode",
  "cuTexRefGetFlags",
  "cuTexRefGetFormat",
  "cuTexRefGetMaxAnisotropy",
  "cuTexRefGetMipmapFilterMode",
  "cuTexRefGetMipmapLevelBias",
  "cuTexRefGetMipmapLevelClamp",
  "cuTexRefGetMipmappedArray",
  "cuTexRefSetAddress_v2",
  "cuTexRefSetAddress2D_v3",
  "cuTexRefSetAddressMode",
  "cuTexRefSetArray",
  "cuTexRefSetBorderColor",
  "cuTexRefSetFilterMode",
  "cuTexRefSetFlags",
  "cuTexRefSetFormat",
  "cuTexRefSetMaxAnisotropy",
  "cuTexRefSetMipmapFilterMode",
  "cuTexRefSetMipmapLevelBias",
  "cuTexRefSetMipmapLevelClamp",
  "cuTexRefSetMipmappedArray",
  "cuSurfRefGetArray",
  "cuSurfRefSetArray",
  "cuTexObjectCreate",
  "cuTexObjectDestroy",
  "cuTexObjectGetResourceDesc",
  "cuTexObjectGetResourceViewDesc",
  "cuTexObjectGetTextureDesc",
  "cuSurfObjectCreate",
  "cuSurfObjectDestroy",
  "cuSurfObjectGetResourceDesc",
  "cuCtxDisablePeerAccess",
  "cuCtxEnablePeerAccess",
  "cuDeviceCanAccessPeer",
  "cuDeviceGetP2PAttribute",
  "cuGraphicsMapResources",
  "cuGraphicsResourceGetMappedMipmappedArray",
  "cuGraphicsResourceGetMappedPointer_v2",
  "cuGraphicsResourceSetMapFlags_v2",
  "cuGraphicsSubResourceGetMappedArray",
  "cuGraphicsUnmapResources",
  "cuGraphicsUnregisterResource",
  "cuIpcOpenMemHandle_v2",
  "cufftPlan1d",
  "cufftPlan2d",
  "cufftPlan3d",
  "cufftPlanMany",
  "cufftMakePlan1d",
  "cufftMakePlan2d",
  "cufftMakePlan3d",
  "cufftMakePlanMany",
  "cufftMakePlanMany64",
  "cufftGetSizeMany64",
  "cufftEstimate1d",
  "cufftEstimate2d",
  "cufftEstimate3d",
  "cufftEstimateMany",
  "cufftCreate",
  "cufftGetSize1d",
  "cufftGetSize2d",
  "cufftGetSize3d",
  "cufftGetSizeMany",
  "cufftGetSize",
  "cufftSetWorkArea",
  "cufftSetAutoAllocation",
  "cufftExecC2C",
  "cufftExecR2C",
  "cufftExecC2R",
  "cufftExecZ2Z",
  "cufftExecD2Z",
  "cufftExecZ2D",
  "cufftSetStream",
  "cufftDestroy",
  "cufftGetVersion",
  "cufftGetProperty",
  "__cudaRegisterFatBinaryEnd",
  "cudaStreamGetCaptureInfo_v2",
  "cudaUserObjectCreate",
  "cudaGraphRetainUserObject",
  "cudaStreamUpdateCaptureDependencies",
  "cudaGetDriverEntryPoint",
  "cudaFuncSetAttribute",
  "cudaIpcGetMemHandle",
  "cudaIpcOpenMemHandle",
  "cudaIpcCloseMemHandle",
  "cudaDriverGetVersion",
  "cudaDeviceGetByPCIBusId",
  "cudaThreadExchangeStreamCaptureMode",
  "cudaHostGetDevicePointer",
  "cudaHostUnregister",
  "cudaGraphAddEventWaitNode",
  "cudaGraphAddEventRecordNode",
  "cudaLaunchHostFunc",
  "cudaGraphAddHostNode",
  "cudaDeviceEnablePeerAccess",
  "cudaGraphAddKernelNode",
  "cuGetErrorName",
  "ncclCommInitRank",
  "ncclCommInitAll",
  "ncclCommDestroy",
  "ncclCommAbort",
  "ncclGetErrorString",
  "ncclCommGetAsyncError",
  "ncclCommCount",
  "ncclCommCuDevice",
  "ncclCommUserRank",
  "ncclReduce",
  "ncclBcast",
  "ncclBroadcast",
  "ncclAllReduce", 
  "ncclReduceScatter", 
  "ncclAllGather",
  "ncclSend",
  "ncclRecv",
  "ncclGroupStart",
  "ncclGroupEnd",
  "ncclGetUniqueId",
  "ncclGetVersion",
  "cublasGemmEx",
  "cublasGemmStridedBatchedEx", 
  "Cuda_Fnc_Invalid"
};
#endif // LOWER_HALF_CUDA_IF_H