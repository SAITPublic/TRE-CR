#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <assert.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>

#include <map>

typedef struct ptxLog{
void* addr;
long len;
void* ptx;
}ptxlog;
ptxlog ptxlg[10];
