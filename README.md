# TRE-CR

## Contents

  [1. Introduction](#1-introduction)<br>
  [2. Overview](#1-overview)  
  [3. Contribution process](#2-contribution-process)  
  [4. Compile and Run](#3-compile-and-run)  
  [5. Contact](#4-contact)

## 1.Introduction
 
This is an enhanced DMTCP(https://github.com/dmtcp/dmtcp.git) plugin to checkpoint- restart CUDA and MPI application with noval split-process architecture, it is developed upon CRAC(https://github.com/DMTCP-CRAC/CRAC-early-development) and MANA(https://github.com/mpickpt/mana). The Plugin code contains two part:1)CUDA plugin, the code is in the contrib/trecr-cuda_interceptor directory, 2)MPI plugin, the code is in the contrib/trecr-mpi_interceptor. <br>

The main contribution of TRE-CR:<br>
GMMU - GPU Memory Management Unit(contrib/trecr-cuda_interceptor/lh/mmu)<br>
Incremental and Compression checkpoint - Increatal(contrib/trecr-cuda_interceptor/lh/increamental) and Compression(contrib/trecr-cuda_interceptor/uh)<br>
NCCL application support<br>

## 2. Overview

TRE-CR is a robust and efficient checkpointing and restoration(C&R) solution for heterogeneous platform. It is a system level C&R solution, currently TRE-CR support CPU, GPU, MPI, NCCL applications. It is helpful to improve the RAS of heterogeneous platform, such as fault recovery, task migration and Infrastructure maintenance.

TRE-CR introduces a virtualization layer which complements the original proxy solution with virtual memory management, call reinterpretation, virtual space isolation, and data transfer optimizations. First, the delegator starts up and loads the user program into its own process space for efficient context switching. The user program runs with a set of wrapper libraries (e.g., CUDA, MPI, and NCCL wrapper). While the delegator is launched with the real library to proxy operations on the GPU and network. The wrapper library aims to intercept CUDA, MPI and NCCL calls for logging and then trap to the delegator. In the delegator, we call the real function based on the mapping of the function name to the function pointer of the dynamic library. In addition, we divide the virtual address space into device reserved space, CPU required space and kernel managed space. Based on that, we can restore CPU and GPU process space via snapshots and runtime via log replay. 


## 3. Contribution process

Researchers and Individuals who want to contribute to the project is recommanded refer to the following steps : 

```
git clone https://github.com/SAITPublic/TRE-CR.git
cd TRE-CR
git remote add origin https://github.com/SAITPublic/TRE-CR.git
git branch -M main
git add *.source file
git commit *.source file
git push -uf origin main
```

## 4. Compile and Run

For compile and simple test the TRE-CR for checkpointing and restore GPU applications, we recommand the following steps : 

### 4.1 Compile

```
cd TRE-CR/
./configure
make
cd TRE-CR/contrib/trecr-mpi_interceptor
make
make install
cd TRE-CR/contrib/trecr-cuda_interceptor
make
```

After compiled finished, users can generate a bin/ diretory under TRE-CR/ which contains binary execution file for launch, restart et al and kernel-loader, cuda-plugin file under TRE-CR/contrib/trecr-cuda_interceptor/ directory.

### 4.2 Run

For running the TRE-CR with simple GPU applications, we provide a sample for reference : 

```
Step 1:
Start coordinator:
$ <trecr_dir>/bin/trecr_coordinator –port 7790
Step 2:
Open a new terminal, and start demo application:
$<trecr_dir>/contrib/trecr-cuda_interceptor/trecrproxy.exe –target-ld <libc2.31 dir>/lib/ld-linuxx86-64.so.2 –with-plugin <trecr_dir>/contrib/trecr-cuda_interceptor/libcuda-interceptor.so -j <sample_dir>/simpleIPC
```

## Contact

* Tian Liu (tian01.liu@samsung.com)
* Biao Xing (biao.xing@samsung.com)
* Fengtao Xie (fengtao.xie@samsung.com)
* Huiru Deng (huiru.deng@samsung.com)
