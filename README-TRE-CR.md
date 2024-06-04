# TRE-CR

## Contents

  [1. Overview](#1-overview)  
  [2. Contribution process](#2-contribution-process)  
  [3. Compile and Run](#3-compile-and-run)  
  [4. Contact](#4-contact)

## 1. Overview

MANA transparently checkpoints MPI application, CRAC transparently checkpoints CUDA application. In order to transparently checkpoints MPI based CUDA application, a new solution is designed called TRE-CR, this 
is an efficient and time-saving checkpointing and restore mechanism. In compared with CRAC or DMTCP, we additionly support some new features and fix some bugs, detailed change point can refer to changelog-TRE-CR.txt.

## 2. Contribution process

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

## 3. Compile and Run

For compile and simple test the TRE-CR for checkpointing and restore GPU applications, we recommand the following steps : 

### 3.1 Compile

```
cd TRE-CR/CRAC-early-development-master/
./configure
make
cd TRE-CR/CRAC-early-development-master/contrib/split-cuda
make
```

After compiled finished, users can generate a bin/ diretory under CRAC-early-development-master/ which contains binary execution file for launch, restart et al and kernel-loader, cuda-plugin file under TRE-CR/CRAC-early-development-master/contrib/split-cuda directory.

### 3.2 Run

For running the TRE-CR with simple GPU applications, we provide a Lammps sample for reference : 

```
/home/TRE-CR/CRAC-early-development-master/bin/dmtcp_launch --cuda --coord-host host_addr --coord-port port_number --kernel-loader /home/TRE-CR/CRAC-early-development-master/contrib/split-cuda/kernel-loader.exe --target-ld /usr/local/glibc-2.31/lib/ld-linux-x86-64.so.2 --with-plugin /home/TRE-CR/CRAC-early-development-master/contrib/split-cuda/libdmtcp_split-cuda.so -j /home/Lammps/bin/lmp -sf gpu -pk gpu 1 -in in.1k
```

## Contact

* Tian Liu (tian01.liu@samsung.com)
* Biao Xing (biao.xing@samsung.com)
* Yueyang Li (yueyang.li@samsung.com)
* Fengtao Xie (fengtao.xie@samsung.com)
* Huiru Deng (huiru.deng@samsung.com)
* Mingming Liu (mming.liu@samsung.com)
* Siying Cao (siying.cao@samsung.com)
