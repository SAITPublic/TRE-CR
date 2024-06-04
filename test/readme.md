Compile the .c into an executable

mpicc -o creat_subarray creat_subarray.c

[option]
mpirun -np 4 ./creat_subarray
