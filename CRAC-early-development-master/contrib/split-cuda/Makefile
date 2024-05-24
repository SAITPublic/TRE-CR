# NOTE: Update the following variables for your system
CC=/usr/bin/gcc-8
CXX=/usr/bin/g++-8
NVCC=${CUDA_DIR}/bin/nvcc
LD=${CXX}
RTLD_PATH=/lib64/ld-linux-x86-64.so.2
CUDA_INCLUDE_PATH=${CUDA_DIR}/include/
NCCL_PATH=/home/tian01.liu/AI-workspace/nccl-master/build/lib

# Declare the directory of MPI restart plugin
MANA_HELPER_DIR=${PWD}/restart_plugin
# The name will be the same as the current directory name.
NAME=${shell basename $$PWD}

# By default, your resulting plugin library will have this name.
LIBNAME=libdmtcp_${NAME}

# Wrapper library against which the target application will be linked.
WRAPPER_LIB=cuda_wrappers

# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=../..
endif
DMTCP_INCLUDE=${DMTCP_ROOT}/include
JALIB_INCLUDE=${DMTCP_ROOT}/jalib
NVCOMP_ROOT=${DMTCP_ROOT}/contrib/split-cuda/lh/nvComp

# Flags for compile and link configurations

#biao.xing@samsung.com modify 0x800000 to 0xae00000 to support checkpointing miniDFT
#NVCC_FLAGS=-Xlinker -Ttext-segment -Xlinker 0x800000 --cudart shared
NVCC_FLAGS=-Xlinker -Ttext-segment -Xlinker 0xae00000 --cudart shared
NVCC_OPTFLAGS=-Xcompiler -g3 -O0

SPLIT_CUDA_INCLUDE_FLAGS=-I${DMTCP_ROOT}/contrib/split-cuda/common \
		-I${DMTCP_ROOT}/contrib/split-cuda/utils \
		-I${DMTCP_ROOT}/contrib/split-cuda/lh \
		-I${DMTCP_ROOT}/contrib/mpi-proxy-split/lower-half \
		-I${DMTCP_ROOT}/src \
		-I${DMTCP_ROOT}/contrib/split-cuda/lh/lh_wrappers \
		-I${DMTCP_ROOT}/contrib/split-cuda/lh/restore_mem \
		-I${DMTCP_ROOT}/contrib/split-cuda/lh/ld_loader \
		-I${DMTCP_ROOT}/contrib/split-cuda/lh/replay_cuda_api \
		-I${DMTCP_ROOT}/contrib/split-cuda/uh \
		-I${DMTCP_ROOT}/contrib/split-cuda/uh/log_cuda_api \
		-I${DMTCP_ROOT}/contrib/split-cuda/uh/switch_context \
		-I${DMTCP_ROOT}/contrib/split-cuda/uh/uh_wrappers \
		-I${DMTCP_ROOT}/contrib/split-cuda/autogen

NVCOMP_INCLUDE_FLAGS=-I${NVCOMP_ROOT}/include \
		     -I${NVCOMP_ROOT}/include/nvcomp \
		     -I${NVCOMP_ROOT}/include/gdeflate

INCLUDE_FLAGS=-I. -I${CUDA_INCLUDE_PATH} -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} \
              ${SPLIT_CUDA_INCLUDE_FLAGS} ${NVCOMP_INCLUDE_FLAGS}
WARNING_FLAGS=-Wall -Wno-deprecated-declarations -Werror
# variables related to kernel loader
KERNEL_LOADER_OBJS=lh/kernel-loader.o utils/procmapsutils.o lh/ld_loader/custom-loader.o \
			lh/lh_wrappers/mmap-wrapper.o lh/lh_wrappers/sbrk-wrapper.o lh/lh_wrappers/cuda-lh-if.o \
			utils/utils.o  lh/replay_cuda_api/replay_cuda_calls.o \
			utils/trampoline_setup.o lh/restore_mem/mem-restore.o lh/mmu/MMU.o \
			lh/increamental/increamental.o lh/increamental/parsha256_src/parsha256.o\
			uh/switch_context/switch_context.o utils/mtcp_util.o

##################
#biao.xing@samsung.com added on 2023.3.30 for restoring MPI
ifneq ($(MANA_HELPER_DIR),)
INCLUDE_FLAGS+=-I$(MANA_HELPER_DIR)
KERNEL_LOADER_OBJS+=restart_plugin/mtcp_restart_plugin.o \
                    restart_plugin/mtcp_split_process.o  lh/split_process.o lh/procmapsutils.o

CFLAGS+=-DMTCP_PLUGIN_H="<mtcp_restart_plugin.h>"
endif
##################
override CFLAGS += -g3 -O0 -fPIC ${INCLUDE_FLAGS} -c -std=gnu11 \
                ${WARNING_FLAGS} -fno-stack-protector
override CXXFLAGS += -g3 -O0 -fPIC ${INCLUDE_FLAGS} -c -std=c++11 \
                  ${WARNING_FLAGS} -fno-stack-protector -DTIMING

KERNEL_LOADER_CFLAGS=-DSTANDALONE
KERNEL_LOADER_BIN=kernel-loader.exe

# variables related to small cuda application (target.c)
TARGET_OBJS=target.o
SHA_OBJ=lh/increamental/parsha256_src/parsha256.o
TARGET_BIN=target.dummy.exe

# variables related to autogen code
AUTOGEN_OBJS=uh/uh_wrappers/cuda_autogen_wrappers.o \
             uh/uh_wrappers/cublas_v1_wrappers.o

# variables related to stub library which should be linked with user's cuda-
# application; This library has dummy wrappers for cuda APIs
WRAPPER_LIB_OBJS=cuda_stub_lib/cuda_stub.o

#fengtao.xie added 7/8
# variables related to dmtcp's cuda-plugin (upper-half)
UTILITY_OBJS=utils/procmapsutils.o utils/utils.o \
        utils/trampoline_setup.o
PLUGIN_LIB_OBJS=${AUTOGEN_OBJS} ${UTILITY_OBJS}  \
                uh/uh_wrappers/upper-half-wrappers.o uh/cuda-plugin.o  lh/increamental/increamental.o\
                uh/log_cuda_api/log_cuda_api.o uh/switch_context/switch_context.o uh/log_cuda_api/replay_api_utils.o

#variables for make check
DMTCP_LAUNCH=${DMTCP_ROOT}/bin/dmtcp_launch
DMTCP_LAUNCH_FLAGS=--new-coordinator --cuda --kernel-loader \
		   $$PWD/${KERNEL_LOADER_BIN} --target-ld ${RTLD_PATH} \
		   --with-plugin $$PWD/${LIBNAME}.so


default: ${AUTOGEN_OBJS} ${WRAPPER_LIB_OBJS} ${LIBNAME}.so ${KERNEL_LOADER_BIN}\
         lib${WRAPPER_LIB}.so ${TARGET_BIN}


autogen: uh/uh_wrappers/cuda_autogen_wrappers.o
uh/uh_wrappers/cuda_autogen_wrappers.o: uh/uh_wrappers/cuda_autogen_wrappers.cpp
	${CXX} ${CXXFLAGS} -Wno-unused-variable -c $< -o $@
#autogen/cuda_autogen_wrappers.cpp: autogen/generate-split-proc-wrappers.py \
#                           autogen/cuda_wrapper_signatures.txt
#	python autogen/generate-split-proc-wrappers.py \
#  autogen/cuda_wrapper_signatures.txt --default \
#  > autogen/cuda_autogen_wrappers.cpp ;

uh/uh_wrappers/cublas_v1_wrappers.o: uh/uh_wrappers/cublas_v1_wrappers.cpp
	${CXX} ${CXXFLAGS} -Wno-unused-variable -c $< -o $@
cuda_stub_lib/cuda-stub.o: cuda_stub_lib/cuda_stub.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@


${LIBNAME}.so: ${PLUGIN_LIB_OBJS}
	${CXX} -shared -fPIC -g3 -O0 -o $@ $^

lib${WRAPPER_LIB}.so: ${WRAPPER_LIB_OBJS}
	${CC} -shared -fPIC -g3 -O0 -o $@ $^

disableASLR:
	@- [ `cat /proc/sys/kernel/randomize_va_space` = 0 ] \
	|| sudo sh -c 'echo 0 > /proc/sys/kernel/randomize_va_space'

enableASLR:
	@- [ `cat /proc/sys/kernel/randomize_va_space` != 2 ] \
	&& sudo sh -c 'echo 2 > /proc/sys/kernel/randomize_va_space'

check: default
	${DMTCP_LAUNCH} ${DMTCP_LAUNCH_FLAGS} $$PWD/test/${TARGET_BIN}

.c.o:
	${CC} ${CFLAGS} $< -o $@

.cpp.o:
	${CXX} ${CXXFLAGS} $< -o $@ -DTIMING

${SHA_OBJ}: lh/increamental/parsha256_src/parsha256.cu
	${NVCC} --compiler-options '-fPIC' -std=c++11 ${NVCC_OPTFLAGS} -c $< -o $@

${TARGET_OBJS}: test/target.c
	${NVCC} ${INCLUDE_FLAGS} ${NVCC_OPTFLAGS} -c $< -o test/$@

${TARGET_BIN}: ${TARGET_OBJS}
	${LD} test/$< -o test/$@ -L. -l${WRAPPER_LIB};
	${NVCC} -g test/$< -lcudart -o test/target.exe


# Apparently, Nvidia libraries don't like -pie; so, we are forced
# to link the kernel loader (which is really just emulating the lower
# half) to a fixed address (0x800000)
${KERNEL_LOADER_BIN}: ${KERNEL_LOADER_OBJS}
	#${NVCC} ${NVCC_FLAGS} -L/home/alienware/Documents/CRAC-monitored/CRAC-early-development/contrib/split-cuda/mylib/ $^ -o $@ -lcuda -ldl -lcublas -lcusparse -lcusolver -lwrapper
	${NVCC} ${NVCC_FLAGS}  $^ -o $@ -lcuda -ldl  -lpthread -lcusparse -lcusolver -lcublas -lcublasLt -lcufft -L${NVCOMP_ROOT}/lib/ -lnvcomp -lnvcomp_gdeflate -lnvcomp_bitcomp -L${DMTCP_ROOT}/src/ -ljalib -ldmtcpinternal -lnohijack -lsyscallsreal -L${NCCL_PATH} -lnccl

vi vim:
	vim ${FILE}.cpp

tags:
	gtags .

dist: clean
	(dir=`basename $$PWD` && cd .. && tar zcvf $$dir.tgz $$dir)
	(dir=`basename $$PWD` && ls -l ../$$dir.tgz)

tidy:
	rm -f ./ckpt.img *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp \
	ckpt_*.dmtcp.temp lhInfo* uhInfo* *.exe

clean: tidy
	rm -f ${KERNEL_LOADER_OBJS} test/${TARGET_OBJS} ${KERNEL_LOADER_BIN} \
	      test/${TARGET_BIN} ${PLUGIN_LIB_OBJS} ${UTILITY_OBJS} \
		  ${LIBNAME}.so GTAGS GRTAGS GPATH uh/uh_wrappers/cuda_autogen_wrappers.o \
      lib${WRAPPER_LIB}.so ${WRAPPER_LIB_OBJS} cuda_stub.*  \
	  lib${CUBLAS_V1_WRAPPER_LIB}.so ${CUBLAS_V1_WRAPPER_LIB_OBJS} cuda_stub.*

.PHONY: dist vi vim clean gdb tags tidy restart run enableASLR disableASLR check
