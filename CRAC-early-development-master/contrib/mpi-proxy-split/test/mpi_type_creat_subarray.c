#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

void printarr(int** data, int n, char* str);
int** allocarray(int n);

int main(int argc, char** argv) {

    /* array sizes */
    const int bigsize = 10;
    const int subsize = 5;

    /* communications parameters */
    const int sender = 0;
    const int receiver = 1;
    const int ourtag = 2;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < receiver + 1) {
        if (rank == 0)
            fprintf(stderr, "%s: Needs at least %d  processors.\n", argv[0], receiver + 1);
        MPI_Finalize();
        return 1;
    }

    if (rank == sender) {
        int** bigarray = allocarray(bigsize);
        for (int i = 0; i < bigsize; i++)
            for (int j = 0; j < bigsize; j++)
                bigarray[i][j] = i * bigsize + j;


        printarr(bigarray, bigsize, " Sender: Big array ");

        MPI_Datatype mysubarray;

        int starts[2] = { 5,3 };
        int subsizes[2] = { subsize,subsize };
        int bigsizes[2] = { bigsize, bigsize };
        MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
            MPI_ORDER_C, MPI_INT, &mysubarray);
        MPI_Type_commit(&mysubarray);
        MPI_Send(&(bigarray[0][0]), 1, mysubarray, receiver, ourtag, MPI_COMM_WORLD);
        MPI_Type_free(&mysubarray);
        free(bigarray[0]);
        free(bigarray);

    }
    else if (rank == receiver) {
        int** subarray = allocarray(subsize);
        MPI_Recv(&(subarray[0][0]), subsize * subsize, MPI_INT, sender, ourtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printarr(subarray, subsize, " Receiver: Subarray -- after receive");
        free(subarray[0]);
        free(subarray);
    }
    usleep(1000000);
    MPI_Finalize();
    return 0;
}

void printarr(int** data, int n, char* str) {
    fprintf(stderr, "-- %s --\n", str);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            usleep(10000);
            fprintf(stderr, "%3d ", data[i][j]);
        }
        fprintf(stderr, "\n");
    }
}

int** allocarray(int n) {
    int* data = malloc(n * n * sizeof(int));
    int** arr = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++)
        arr[i] = &(data[i * n]);

    return arr;
}
