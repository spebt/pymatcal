#!/bin/bash -l
# ===============================================
# UB HPC general-compute partition
# ===============================================

#  Request number of tasks and CPUs per task
#SBATCH -n216
#SBATCH --cpus-per-task=4

#  Nodes constraint, nodes with Intel Infiniband
#SBATCH --constraint="[SAPPHIRE-RAPIDS-IB|ICE-LAKE-IB|CASCADE-LAKE-IB]"

#  Request memory per CPU
#SBATCH --mem-per-cpu=8G

#   Give your job a name, so you can recognize it in the queue
#SBATCH --job-name="torch-sysmat"

#   Tell slurm where to send emails about this job
#SBATCH --mail-user=myemailaddress@institution.edu

#   Tell slurm the types of emails to send.
#   Options: NONE, BEGIN, END, FAIL, ALL

#SBATCH --mail-type=NONE

#   Tell Slurm which cluster, partition and qos to use to schedule this job.
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --cluster=ub-hpc

# load modules
module load gcc/11.2.0 openmpi/4.1.1 pytorch/1.13.1-CUDA-11.8.0 h5py/3.6.0

# Echo number of nodes
echo "Number of nodes allocated:" "$SLURM_JOB_NUM_NODES"

# Make the directory for the output
mkdir -p system_matrix_data

# Run the code
for i in $(seq 0 599); do
  echo "Generating matirx $i"
  srun --mpi=pmi2 --exclusive --verbose python pytorch_ppdf.py "$i"
done
