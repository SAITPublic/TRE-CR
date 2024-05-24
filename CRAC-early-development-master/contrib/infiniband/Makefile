PLUGIN=infiniband
CC = gcc
M32=0
CFLAGS = -g -O2
CPPFLAGS = 
CXX = g++
CXXFLAGS = -g -O2
LDFLAGS = 

# Allow the user to specify the install program.
INSTALL = /bin/install -c
INSTALL_PROGRAM = ${INSTALL}
INSTALL_DATA = ${INSTALL} -m 644
INSTALL_SCRIPT = ${INSTALL}

MKDIR_P = /bin/mkdir -p
PACKAGE = dmtcp
PACKAGE_BUGREPORT = dmtcp-forum@lists.sourceforge.net
PACKAGE_NAME = DMTCP
PACKAGE_STRING = DMTCP 3.0.0
PACKAGE_TARNAME = dmtcp
PACKAGE_URL = http://dmtcp.sourceforge.net
PACKAGE_VERSION = 3.0.0
prefix=/usr/local
exec_prefix=${prefix}
datarootdir=${prefix}/share
bindir = ${exec_prefix}/bin
docdir = ${datarootdir}/doc/${PACKAGE_TARNAME}
includedir = ${prefix}/include
infodir = ${datarootdir}/info
libdir = ${exec_prefix}/lib
pkglibdir = $(libdir)/dmtcp
top_builddir = ../..
top_srcdir = ../..
srcdir = .

targetdir = $(top_builddir)

ifeq ($(M32),1)
  targetdir = $(top_builddir)/lib/$(PACKAGE)/32
endif

LIBNAME=$(targetdir)/lib/${PACKAGE}/libdmtcp_${PLUGIN}.so

DMTCP_BIN = ${targetdir}/bin
DMTCP_INCLUDE_PATH = $(top_srcdir)/include
JALIB_PATH = $(top_srcdir)/jalib

INCLUDES = -I$(JALIB_PATH) -I$(DMTCP_INCLUDE_PATH)
LIB_LDFLAGS = -libverbs -shared -Wl,--unresolved-symbols=ignore-all

override CFLAGS += -fPIC
override CXXFLAGS += -fPIC

ifeq ($(M32),1)
CFLAGS += -m32 -Wa,--32
CXXFLAGS += -m32 -Wa,--32
LDFLAGS += -m32 -Wl,-m32 -Wl,-melf_i386 -Wa,--32
endif

COMPILE = $(CC) $(INCLUDES) $(CPPFLAGS) $(CFLAGS) -c -o $@
LINK = $(CC) $(CFLAGS) $(LDFLAGS) -o $@
CXXCOMPILE = $(CXX) $(INCLUDES) $(CPPFLAGS) $(CXXFLAGS) -c -o $@
CXXLINK = $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@

HEADERS = $(DMTCP_INCLUDE_PATH)/dmtcp.h lib/list.h \
	  ibvctx.h ibv_internal.h ibvidentifier.h debug.h

LIBOBJS = infinibandwrappers.o ibvctx.o lib/list.o
TRACEOBJS =  infinibandtrace.o
DMTCP_PORT=7783

ifndef HOST0
	HOST0 = compute-0-5
endif
ifndef HOST1
	HOST1 = compute-0-0
endif
ifndef HOST2
	HOST2 = compute-0-1
endif

all: default

default: check-compat ${LIBNAME}

check-compat:
	@ grep "INFINIBAND_SUPPORT='yes'" ${top_srcdir}/config.log > /dev/null \
	  || \
	  (echo && \
	  echo DMTCP not configured with --enable-infiniband-support. && \
	  echo Infiniband plugin cannot work. Execute the following && \
	  echo to reconfigure DMTCP before proceeding: && \
	  echo './configure --enable-infiniband-support; make clean; make' && \
	  false)

libibtrace.so: ${TRACEOBJS}
	${CC} -shared -fPIC -o $@ $^

${LIBNAME}: ${LIBOBJS}
	#${CXX} -shared -fPIC -o $@ $^
	@rm -f ${LIBNAME} libinfiniband.so
	$(CXXLINK) $(LIBOBJS) $(LIB_LDFLAGS)
	ln -sf ${LIBNAME} libinfiniband.so

get_qp_from_pointer.ic: get_XX_from_pointer.ic
	sed -e 's%XX%qp%g' get_XX_from_pointer.ic > get_qp_from_pointer.ic
get_cq_from_pointer.ic: get_XX_from_pointer.ic
	sed -e 's%XX%cq%g' get_XX_from_pointer.ic > get_cq_from_pointer.ic
get_srq_from_pointer.ic: get_XX_from_pointer.ic
	sed -e 's%XX%srq%g' get_XX_from_pointer.ic > get_srq_from_pointer.ic
ibv_wr_ops_send.ic: ibv_wr_ops.ic
	sed -e 's%SENDRECV%send%g' ibv_wr_ops.ic > ibv_wr_ops_send.ic
ibv_wr_ops_recv.ic: ibv_wr_ops.ic
	sed -e 's%SENDRECV%recv%g' ibv_wr_ops.ic > ibv_wr_ops_recv.ic
ibvctx.c: ibv_wr_ops_send.ic ibv_wr_ops_recv.ic get_qp_from_pointer.ic get_cq_from_pointer.ic get_srq_from_pointer.ic

%.o: %.c ${HEADERS}
	$(COMPILE)  $<
	#${CC} ${CFLAGS} -o $@ $<

%.o: %.cpp ${HEADERS}
	$(CXXCOMPILE)  $<
	#${CXX} ${CPPFLAGS} -o $@ $<

examples/orig:
	cd examples && ${MAKE} default

rdma:
	cd examples/rdma && ${MAKE}

check: examples/orig
	- pkill -9 dmtcp_coord
	${DMTCP_BIN}/dmtcp_coordinator -p ${DMTCP_PORT} --daemon
	export dir=$$PWD && ssh ${HOST1} "cd $$dir && \
	  ${DMTCP_BIN}/dmtcp_launch -h ${HOST0} -p ${DMTCP_PORT} --ib ./examples/orig" &
	export dir=$$PWD && ssh ${HOST2} "cd $$dir && \
	  ${DMTCP_BIN}/dmtcp_launch -h ${HOST0} -p ${DMTCP_PORT} --ib ./examples/orig ${HOST1}" &
	@ sleep 15 && ${DMTCP_BIN}/dmtcp_command -p ${DMTCP_PORT} --checkpoint
	sleep 20 ; ${DMTCP_BIN}/dmtcp_command -p ${DMTCP_PORT} --quit

tidy:
	rm -rf ckpt_*.dmtcp core.* dmtcp_restart_script*

clean: tidy
	rm -f ${LIBOBJS} ${LIBNAME} libinfiniband.so
	rm -f ${TRACEOBJS} libibtrace.so
	rm -f ibv_wr_ops_send.ic ibv_wr_ops_recv.ic get_srq_from_pointer.ic get_qp_from_pointer.ic get_cq_from_pointer.ic

distclean: clean
	rm -f Makefile

dist: distclean
	dir=`basename $$PWD`; cd ..; tar czvf $$dir.tar.gz ./$$dir
	dir=`basename $$PWD`; ls -l ../$$dir.tar.gz

install-libs: all
	${INSTALL_PROGRAM} ${LIBNAME} $(DESTDIR)$(pkglibdir)

uninstall-libs:
	rm -f $(DESTDIR)$(pkglibdir)/$(LIBNAME)

install: install-libs

uninstall: uninstall-libs

.PHONY: default all build tidy clean distclean install uninstall
