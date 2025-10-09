#include <mpi.h>
#include <cstdio>

int main() {
  MPI_Init(nullptr, nullptr);

  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("Hello world, I am proc %d of total %d\n", rank, size);

  MPI_Finalize();

  return 0;
}
