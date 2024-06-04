# Timing Algorithm User Guide

# Table of Contents

* [Introduction](#introduction)
* [Usage Guide](#Usage Guide)

## Introduction
Timing algorithm is a new function we supported in CRAC which mainly has following features:
 1) Automatically profile CUDA application original log based on given interval and specific algorithm;
 2) Automatically generate profiled log file which contains optimal checkpoint location;
 3) Do checkpoint based on optimal checkpoint location and reduce overall checkpoint workload;

## Usage Guide
To enable timing algorithm in CRAC, one should refer to following steps : 


Step 1 : 
 1. Generate original log file using Nvprof tools (profile CUDA application only)
 2. reference link :  [https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof)
 3. Original log reference format : CRAC_SRCX/CRAC-early-development-master/timing/test.txt


Step 2 :
 1. Put the original log file(test.txt) in specify path, default path is : CRAC_SRCX/CRAC-early-development-master/timing
 2. Modify dmtcp_coordinator.cpp:line179-line182 and threadlist.cpp:line376-line379 to absolute path of original log file like below, then, profiled log will automatically generated in same directory : 
     char* ori_log_path = "/Absolute_path_of_CRAC/timing/test.txt"; // original log file


Step 3 :
 1. Open CRAC_SRCX/CRAC-early-development-master/include/config.h file and uncomment line 38 like below:
     //#define TIMING 1 -> #define TIMING 1

Step 4 :
 1. Based on API kind in profiled_log.txt, add checkpoint control signal for each api in cuda_autogen_wrappers.cpp,re    fer to line4949-line4979 for cuMemAlloc_v2 API  


Step 5 :
 1. Execute following command and test the timing algorithm using lammps(balance):

    /Absolute_path_of_CRAC/bin/dmtcp_launch --new-coordinator --cuda --interval 1 --kernel-loader /Absolute_path_of_C    RAC/contrib/split-cuda/kernel-loader.exe --target-ld /usr/local/glibc-2.31/lib/ld-linux-x86-64.so.2 --with-plugin    /Absolute_path_of_CRAC/contrib/split-cuda/libdmtcp_split-cuda.so /Absolute_path_of_lammps/lmp -sf gpu -pk gpu 1 -    in /Absolute_path_of_lammps_example/in.balance




