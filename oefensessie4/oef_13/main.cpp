#include <mpi.h>
#include <cmath>
#include <iostream>
#include "timer.h"

double calculateArea(double bin, double binsize) {
    return sqrt(1 - bin*bin) * binsize;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double min = -1.0, max = 1.0;
    long N = 1000000;
    double binsize = (max - min) / N;

    long chunk = N / size;
    long start = rank * chunk;
    long end = (rank == size - 1) ? N : start + chunk;

    double localSum = 0.0;
    for (long i = start; i < end; i++) {
        double bin = min + i * binsize;
        localSum += calculateArea(bin, binsize);
    }

    AutoAverageTimer tReduce("MPI_Reduce");
    tReduce.start();
    double total_reduce = 0.0;
    MPI_Reduce(&localSum, &total_reduce, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    tReduce.stop();

    if(rank == 0){
        std::cout << "MPI_Reduce calculated area: " << total_reduce << std::endl;
        std::cout << "Actual area: " << M_PI / 2 << std::endl;
        tReduce.report(std::cout);
    }

    MPI_Finalize();
    return 0;
}

/*
* OUTPUT:
*
 * SLURM_JOB_ID: 65293174
SLURM_JOB_USER: vsc37933
SLURM_JOB_ACCOUNT: lp_h_pds_iiw
SLURM_JOB_NAME: main
SLURM_CLUSTER_NAME: wice
SLURM_JOB_PARTITION: batch
SLURM_NNODES: 1
SLURM_NODELIST: s28c11n1
SLURM_JOB_CPUS_PER_NODE: 5
Date: Sun Oct 12 16:40:42 CEST 2025
Walltime: 00-00:10:00
========================================================================
MPI_Reduce calculated area: 1.5708
Actual area: 1.5708
#MPI_Reduce 0.010098 +/- 0 sec (1 measurements)
 * */