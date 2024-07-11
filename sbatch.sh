#!/bin/bash
#SBATCH --job-name=pymatcal-mpi
#SBATCH --output=pymatcal-mpi.out
#SBATCH --error=pymatcal-mpi.err
#SBATCH --time=4:00:00
#SBATCH --nodes=4
#SBATCH --partition=general-compute
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=general-compute
#SBATCH --tasks-per-node=12


# load modules
module load gcc/11.20 openmpi/4.1.1 scipy-bundle/2021.10

# install mpi4py module 

# setup for MPI
export I_MPI_FABRICS_LIST=tcp
export I_MPI_DEBUG=4
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
# run the script
echo SLURM_NPROCS'='$SLURM_NPROCS
srun -n $SLURM_NPROCS python /projects/academic/rutaoyao/share/fanghan/github/pymatcal-github/get_all_ppdfs_mpi.py /projects/academic/rutaoyao/share/fanghan/github/pydetgen-github/test_20240711_160852.yml