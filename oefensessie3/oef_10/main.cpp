#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "Dit programma vereist precies 2 processen.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const size_t N = 1048576; // 1,048,576 elementen
    float *sendbuf = (float *)malloc(N * sizeof(float));
    float *recvbuf = (float *)malloc(N * sizeof(float));

    if (!sendbuf || !recvbuf) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        free(sendbuf);
        free(recvbuf);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Initialiseer array met rank
    for (size_t i = 0; i < N; ++i) {
        sendbuf[i] = (float)rank;
    }

    int other = (rank == 0) ? 1 : 0;
    MPI_Status status;

    // Gebruik MPI_Sendrecv om tegelijk te sturen en te ontvangen
    MPI_Sendrecv(
        sendbuf, (int)N, MPI_FLOAT, other, 0,   // send
        recvbuf, (int)N, MPI_FLOAT, other, 0,   // recv
        MPI_COMM_WORLD, &status
    );

    printf("I am process %d and I have received b(0) = %.2f\n", rank, recvbuf[0]);

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
