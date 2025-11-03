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

    double *total_acc = nullptr;
    if(rank == 0) total_acc = new double(0.0);

    MPI_Win win;
    MPI_Win_create(total_acc, (rank == 0) ? sizeof(double) : 0, sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    AutoAverageTimer tAcc("MPI_Accumulate");
    tAcc.start();
    MPI_Win_fence(0, win);
    MPI_Accumulate(&localSum, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, win);
    MPI_Win_fence(0, win);
    tAcc.stop();

    if(rank == 0){
        std::cout << "MPI_Accumulate calculated area: " << *total_acc << std::endl;
        std::cout << "Actual area: " << M_PI / 2 << std::endl;
        tAcc.report(std::cout);
        delete total_acc;
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}

/*
OUTPUT:

* SLURM_JOB_ID: 65293173
SLURM_JOB_USER: vsc37933
SLURM_JOB_ACCOUNT: lp_h_pds_iiw
SLURM_JOB_NAME: main
SLURM_CLUSTER_NAME: wice
SLURM_JOB_PARTITION: batch
SLURM_NNODES: 1
SLURM_NODELIST: s28c11n1
SLURM_JOB_CPUS_PER_NODE: 5
Date: Sun Oct 12 16:37:37 CEST 2025
Walltime: 00-00:10:00
========================================================================
MPI_Accumulate calculated area: 1.5708
Actual area: 1.5708
#MPI_Accumulate 0.00554443 +/- 0 sec (1 measurements)

BESPREKING:
In de oorspronkelijke versie van het programma worden alle resultaten eerst lokaal
berekend door elk proces en daarna verzameld en opgeteld door het hoofproces met MPI_Reduce.
Dit betekent dat alle processen moeten wachten totdat iedereen klaar is en dat er veel
communicatie tussen de processen en het hoofproces plaatsvindt.

In de nieuwe versie schrijven de processen hun resultaten direct naar het geheugen van het
hoofproces met MPI_Accumulate. Hierdoor hoeft het hoofproces niet zelf alle resultaten samen
te voegen en kunnen de andere processen vrijwel direct bijdragen aan het eindresultaat.
Dit vermindert de hoeveelheid communicatie en de tijd dat processen op elkaar moeten wachten.

Door deze directe aanpak is het programma over het algemeen sneller,
vooral wanneer er meerdere processen zijn of grote hoeveelheden data verwerkt moeten worden.
Het resultaat blijft precies hetzelfde, maar de berekening verloopt efficiënter en soepeler.
 * */