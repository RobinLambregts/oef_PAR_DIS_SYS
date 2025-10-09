#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            cerr << "Dit programma vereist precies 2 processen." << endl;
        MPI_Finalize();
        return 1;
    }

    const int repetitions = 1'000'000; // 10^6 herhalingen

    // n = 0..10, bufferlengte = 2^n bytes
    for (int n = 0; n <= 10; ++n) {
        int buf_size_bytes = 1 << n;   // 2^n bytes
        vector<char> buffer(buf_size_bytes, 'x'); // dummy data

        MPI_Barrier(MPI_COMM_WORLD); // synchronisatie

        double start_time = 0.0;
        if (rank == 0) {
            start_time = MPI_Wtime();
            for (int i = 0; i < repetitions; ++i) {
                // stuur naar proces 1
                MPI_Send(buffer.data(), buf_size_bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                // ontvang van proces 1
                MPI_Recv(buffer.data(), buf_size_bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            double end_time = MPI_Wtime();
            double avg_round_trip = (end_time - start_time) / repetitions;
            cout << "Buffer size: " << buf_size_bytes << " bytes, "
                 << "Average round-trip time: " << avg_round_trip * 1e6 << " microseconds" << endl;
        } else if (rank == 1) {
            for (int i = 0; i < repetitions; ++i) {
                MPI_Recv(buffer.data(), buf_size_bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.data(), buf_size_bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}

/*
* OUTPUT
* 1 node - 2 processen:
*
SLURM_JOB_ID: 65251316
SLURM_JOB_USER: vsc37933
SLURM_JOB_ACCOUNT: lp_h_pds_iiw
SLURM_JOB_NAME: main
SLURM_CLUSTER_NAME: wice
SLURM_JOB_PARTITION: batch
SLURM_NNODES: 1
SLURM_NODELIST: n28c20n2
SLURM_JOB_CPUS_PER_NODE: 4
Date: Sun Oct  5 13:27:29 CEST 2025
Walltime: 00-00:10:00
========================================================================
Buffer size: 1 bytes, Average round-trip time: 0.328656 microseconds
Buffer size: 2 bytes, Average round-trip time: 0.329504 microseconds
Buffer size: 4 bytes, Average round-trip time: 0.329652 microseconds
Buffer size: 8 bytes, Average round-trip time: 0.329036 microseconds
Buffer size: 16 bytes, Average round-trip time: 0.328537 microseconds
Buffer size: 32 bytes, Average round-trip time: 0.474945 microseconds
Buffer size: 64 bytes, Average round-trip time: 0.402845 microseconds
Buffer size: 128 bytes, Average round-trip time: 0.72937 microseconds
Buffer size: 256 bytes, Average round-trip time: 0.788953 microseconds
Buffer size: 512 bytes, Average round-trip time: 0.93574 microseconds
Buffer size: 1024 bytes, Average round-trip time: 1.02886 microseconds
*
* ---------------------------------------------------------------------------
*
* 2 nodes - elk 1 proces
*SLURM_JOB_ID: 65251322
SLURM_JOB_USER: vsc37933
SLURM_JOB_ACCOUNT: lp_h_pds_iiw
SLURM_JOB_NAME: main
SLURM_CLUSTER_NAME: wice
SLURM_JOB_PARTITION: batch
SLURM_NNODES: 2
SLURM_NODELIST: n28c20n2,n28c22n4
SLURM_JOB_CPUS_PER_NODE: 1(x2)
Date: Sun Oct  5 13:37:22 CEST 2025
Walltime: 00-00:10:00
========================================================================
Buffer size: 1 bytes, Average round-trip time: 2.24171 microseconds
Buffer size: 2 bytes, Average round-trip time: 2.24028 microseconds
Buffer size: 4 bytes, Average round-trip time: 2.25114 microseconds
Buffer size: 8 bytes, Average round-trip time: 2.25111 microseconds
Buffer size: 16 bytes, Average round-trip time: 2.25525 microseconds
Buffer size: 32 bytes, Average round-trip time: 2.31869 microseconds
Buffer size: 64 bytes, Average round-trip time: 2.44482 microseconds
Buffer size: 128 bytes, Average round-trip time: 2.5059 microseconds
Buffer size: 256 bytes, Average round-trip time: 2.95379 microseconds
Buffer size: 512 bytes, Average round-trip time: 3.12836 microseconds
Buffer size: 1024 bytes, Average round-trip time: 3.40799 microseconds
 *
* ---------------------------------------------------------------------------
* CONCLUSIE:
* De resultaten laten zien dat het versturen van gegevens tussen processen
* op dezelfde computer (1 node, 2 processen) heel snel gaat,
* meestal rond 0,3 microseconden, en iets trager wordt bij grotere
* hoeveelheden data.
* Als de processen op verschillende computers zitten
* (2 nodes, 1 proces per node), duurt het een stuk langer,
* van ongeveer 2,2 tot 3,4 microseconden.
* Dit komt omdat de communicatie nu via het netwerk gaat in plaats
* van direct in het geheugen.
* Kortom: processen die veel met elkaar moeten praten,
* draaien het best op dezelfde machine.
* */