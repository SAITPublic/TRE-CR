#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

static void processArgs(int, const char** );

int
main(int argc, char **argv)
{
  int i = 0;
  void *cuda_ptr1 = NULL;
  void *cuda_ptr2 = NULL;
  void *cuda_ptr3 = NULL;
  void *cuda_ptr4 = NULL;

  processArgs(argc, (const char**)argv);
  cudaSetDevice(0);

  char* test_align=NULL;
  printf("before align, addr:%p\n", test_align);
  //test_align = (char*)malloc(200*1024);
  posix_memalign((void**)&test_align, 512, 1024LL*300);
  printf("after align, ret=%i, addr:%p\n", 0, test_align);

  test_align = realloc(test_align, 345LL*1024*1024);
  printf("after realloc, addr:%p\n", test_align);

  //test_align = realloc(test_align, 25952256);
  //printf("after realloc, addr:%p\n", test_align);
  
  return 0;

  char* testHostMem = (char*)malloc(1025);
  memset(testHostMem, 0, 1025);
  cudaError_t rc = cudaMalloc(&cuda_ptr1, 436*sizeof(char));
  printf("cudaMalloc returned: %d, cuda_ptr1: %p\n", (int)rc, cuda_ptr1);
  cudaMemset(cuda_ptr1, 'A', 1025);

  //sleep(10);  // give enough time to checkpoint

  rc = cudaMalloc(&cuda_ptr2, 43*sizeof(char));
  printf("cudaMalloc returned: %d, cuda_ptr2: %p\n", (int)rc, cuda_ptr2);
  sleep(10);

  rc = cudaMemcpy(testHostMem, cuda_ptr1, 436, cudaMemcpyDeviceToHost);
  printf("cudaMemcpy ret:%i, str:%s\n", rc, testHostMem);

  rc = cudaMalloc(&cuda_ptr3, 1025*sizeof(char));
  cudaMemset(cuda_ptr3, 'C', 1024);
  printf("cudaMalloc returned: %d, cuda_ptr3: %p\n", (int)rc, cuda_ptr3);

  rc = cudaMemcpy(testHostMem, cuda_ptr3, 1024, cudaMemcpyDeviceToHost);
  printf("cudaMemcpy ret:%i, str:%s\n", rc, testHostMem);

  cudaFree(cuda_ptr1);
  cudaFree(cuda_ptr2);
  cudaFree(cuda_ptr3);
  return 0;
}

static void
processArgs(int argc, const char** argv)
{
  if (argc > 1) {
    printf("Application was called with the following args: ");
    for (int j = 1; j < argc; j++) {
      printf("%s ", argv[j]);
    }
    printf("\n");
  }
}
