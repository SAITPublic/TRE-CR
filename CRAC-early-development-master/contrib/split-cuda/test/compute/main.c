#include<stdio.h>
#include<stdlib.h>
#include <unistd.h>
#include<mpi.h>

void cudaFun (int devid, int nx, int nz);
int main(int argc, char *argv[])
{
    int myid, numprocs, count, nx, nz;
    float * vp ;
    MPI_Status status;
    int buf = 1;

    nx = 1000 ; nz = 1000 ;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Hello, world, I am %d of %d\n", myid, numprocs);
    if (myid == 0)
    {
        //MPI_Send(&buf, 1, MPI_INT, (myid + 1) % numprocs, 0, MPI_COMM_WORLD);
        //sleep(3);
        printf("Hello, world, I am id 0 \n");
        cudaFun(myid, nx, nz);
    }

    //MPI_Recv(&buf, 1, MPI_INT, (myid - 1 + numprocs) % numprocs, 0, MPI_COMM_WORLD, &status);
    //printf("%d process receives %d message, start to invoke CUDA functions. \n", myid, (myid - 1 + numprocs) % numprocs);

    if (myid != 0) 
    {
        /* Send to neighbor on right */
        //MPI_Send(&buf, 1, MPI_INT, (myid + 1) % numprocs, 0, MPI_COMM_WORLD);
        //sleep(3);
        printf("Hello, world, I am id 1 \n");
        cudaFun(myid, nx, nz);
    }

    MPI_Finalize();
    printf("Finalize MPI ... \n");
    return 0;
}