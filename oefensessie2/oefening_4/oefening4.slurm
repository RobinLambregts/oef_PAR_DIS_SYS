#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=lp_h_pds_iiw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:02:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

# Own code, remove if it causes problems
source /data/leuven/303/vsc30380/slurmhooks

module purge

module load CMake/3.31.3

cmake --version

sleep 43
