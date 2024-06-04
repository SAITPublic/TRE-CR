# MPI compiler according to the platform
PLATFORM=${shell echo $$HOST}
include Makefile_config

# These are needed for libmana.so, since the wrapper functions there will
# call the lower half.  That requires temporarily changing the FS register,
# and apparently, glibc detects this as an attempt to do stack smashing.
override CFLAGS += -fno-stack-protector
override CXXFLAGS += -fno-stack-protector

#biao.xing@samsung.com added
#Please add the path of openmpi in your .bashrc
#In my .bashrc, my configuration is "export MPI_INSTALL_DIR=/home/srcx/vasp/tools/install/openmpi-4.0.6"
MPI_INSTALL_PATH = ${MPI_INSTALL_DIR}
MPICC = $(MPI_INSTALL_PATH)/bin/mpicc
MPICXX = $(MPI_INSTALL_PATH)/bin/mpicxx

ifeq (${MPICXX},)
  MPICXX = ${MPICXX}
endif

ifeq (${MPICC},)
  MPICC = ${MPICC}
endif

# Targets to install:
#   libmana.so (using mpi-wrappers/libmpiwrappers.a), lh_proxy, libmpistub.so,
#   mana_coordinator.o (needed to compile dmtcp_coordinator)

# Subdirectories needed to install
LOWER_HALF_SRCDIR=lower-half
WRAPPERS_SRCDIR=mpi-wrappers

# Test subdirectory:
#   tests

# As you add new files to your plugin library, add the object file names here.

LIBOBJS = mpi_plugin.o p2p_drain_send_recv.o p2p_log_replay.o \
          record-replay.o seq_num.o switch_context.o  convert.o\
          ${LOWER_HALF_SRCDIR}/procmapsutils.o

#MANA_COORD_OBJS = mana_coordinator.o

LIBPROXY = libproxy.a

# Modify if your CRAC_ROOT is located elsewhere.
ifndef CRAC_ROOT
  CRAC_ROOT=../..
endif
# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=${CRAC_ROOT}
endif
DMTCP_INCLUDE=${DMTCP_ROOT}/include
JALIB_INCLUDE=${DMTCP_ROOT}/jalib

override CFLAGS += -fPIC -I${DMTCP_INCLUDE} -I${LOWER_HALF_SRCDIR}
override CXXFLAGS += -fPIC -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} \
                     -I${WRAPPERS_SRCDIR} -I. \
                     -I${DMTCP_ROOT}/src -I${LOWER_HALF_SRCDIR}

ifneq (${MANA_USE_OPENMPI},)
  override CXXFLAGS += -DMANA_USE_OPENMPI
  override CFLAGS += -DMANA_USE_OPENMPI
endif

# ${WRAPPERS_SRCDIR}/libmpiwrappers.a ia a prerequisite for libmana.so
# Always do 'make default' or 'make install' -- not 'make libmana.so'
default: ${MANA_COORD_OBJS}
	make -C ${WRAPPERS_SRCDIR} libmpiwrappers.a
	@make libmpi-inteceptor.so
	@make -C ${LOWER_HALF_SRCDIR} lh_proxy
	@make -C ${LOWER_HALF_SRCDIR} lh_proxy_da
	@make -C ${WRAPPERS_SRCDIR} libmpistub.so

# ${WRAPPERS_SRCDIR}/libmpiwrappers.a made in install/default before libmana.so
libmpi-inteceptor.so: ${LIBOBJS} ${WRAPPERS_SRCDIR}/libmpiwrappers.a
	${MPICXX} -shared -fPIC -g3 -O0 -o $@ ${LIBOBJS} -Wl,--whole-archive ${WRAPPERS_SRCDIR}/libmpiwrappers.a -Wl,--no-whole-archive

mpi_unimplemented_wrappers.cpp: generate-mpi-unimplemented-wrappers.py \
                                mpi_unimplemented_wrappers.txt
	python $^ > $@

.c.o:
	${MPICC} ${CFLAGS} -c -o $@ $<

.cpp.o: virtual-ids.h
	${MPICXX} ${CXXFLAGS} -c -o $@ $<

${LOWER_HALF_SRCDIR}/procmapsutils.o: ${LOWER_HALF_SRCDIR}/procmapsutils.c
	@make -C ${LOWER_HALF_SRCDIR} procmapsutils.o

vi vim: ${FILE}
	vim $<

touch: ${FILE}
	$@ $<

gdb: ${basename lh_proxy.c ${FILE}}
	$@ $<

tests:
	@make -C test/ default

check: libmpi-inteceptor.so ${LOWER_HALF_SRCDIR}/lh_proxy ./autotest.py
	@make -C test/ default
	@python ./autotest.py

check-unit: libmpi-inteceptor.so
	@make -C unit-test/ check

${CRAC_ROOT}/lib/plugin/libmpi-inteceptor.so: libmpi-inteceptor.so
	cp -f $< $@
${CRAC_ROOT}/lib/dmtcp/libmpistub.so:
	make -C ${WRAPPERS_SRCDIR} install
${CRAC_ROOT}/bin/lh_proxy:
	make -C ${LOWER_HALF_SRCDIR} install
${CRAC_ROOT}/bin/gethostbyname_proxy:
	make -C ${LOWER_HALF_SRCDIR}/gethostbyname-static install
${CRAC_ROOT}/bin/mana_p2p_update_logs: ${WRAPPERS_SRCDIR}/mana_p2p_update_logs.c
	make -C ${WRAPPERS_SRCDIR} install

# ${WRAPPERS_SRCDIR}/libmpiwrappers.a ia a prerequisite for libmana.so,
#   which is a prerequisite for ${CRAC_ROOT}/lib/dmtcp/libmana.so
# Always do 'make default' or 'make install' -- not 'make libmana.so'
# MANA_COORD_OBJS needed for 'make mana' at top level.
install: ${MANA_COORD_OBJS}
	make -C ${WRAPPERS_SRCDIR} libmpiwrappers.a
	make ${CRAC_ROOT}/lib/plugin/libmpi-inteceptor.so
	make ${CRAC_ROOT}/lib/dmtcp/libmpistub.so
	make ${CRAC_ROOT}/bin/lh_proxy
	make ${CRAC_ROOT}/bin/gethostbyname_proxy
	make ${CRAC_ROOT}/bin/mana_p2p_update_logs

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp
	rm -rf ckpt_rank_*

clean: tidy
	rm -f ${LIBOBJS} ${MANA_COORD_OBJS}
	rm -f libmpi-inteceptor.so
	rm -f ${CRAC_ROOT}/lib/plugin/libmpi-inteceptor.so
	@cd ${LOWER_HALF_SRCDIR} && make clean
	@cd ${WRAPPERS_SRCDIR} && make clean

distclean: clean
	#cd test && make clean
	cd unit-test && make clean

dist: distclean
	dir=`basename $$PWD` && cd .. && tar czvf $$dir.tgz ./$$dir
	dir=`basename $$PWD` && ls -l ../$$dir.tgz

.PHONY: default clean dist distclean vi vim touch gdb tidy check \
        tests check-unit install
