#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main() {
  MPI_Init(nullptr, nullptr);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == 0) {
      fprintf(stderr, "moet exact 2 processes hebben\n");
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  const size_t N = 1048576;
  auto *sendbuf = (float *)malloc(N * sizeof(float));
  auto *recvbuf = (float *)malloc(N * sizeof(float));

  if (!sendbuf || !recvbuf) {
    fprintf(stderr, "Rank %d: malloc failed\n", rank);
    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < N; ++i) {
    sendbuf[i] = (float)rank;
  }

  int other = (rank == 0) ? 1 : 0;
  MPI_Status status;

  if (rank == 0) {
    MPI_Ssend(sendbuf, (int)N, MPI_FLOAT, other, 0, MPI_COMM_WORLD);
    MPI_Recv(recvbuf, (int)N, MPI_FLOAT, other, 0, MPI_COMM_WORLD, &status);
  } else {
    MPI_Recv(recvbuf, (int)N, MPI_FLOAT, other, 0, MPI_COMM_WORLD, &status);
    MPI_Ssend(sendbuf, (int)N, MPI_FLOAT, other, 0, MPI_COMM_WORLD);
  }

  printf("I am process %d and I have received b(0) = %.2f\n", rank, recvbuf[0]);

  free(sendbuf);
  free(recvbuf);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
