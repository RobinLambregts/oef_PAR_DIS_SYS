#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=lp_h_pds_iiw
#SBATCH --job-name=main
#SBATCH --output=mainTwoCores.out
#SBATCH --error=mainTwoCores.err
#SBATCH --ntasks=2               # aantal MPI-processen
#SBATCH --nodes=2                # aantal nodes
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00          # max runtime

module purge
module load foss/2022a

mpirun main
