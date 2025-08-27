#!/bin/bash

#SBATCH --job-name=ppdf_gen        # Job name for identification
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --time=02:00:00             # Walltime limit (HH:MM:SS), e.g., 2 hours
#SBATCH --nodes=1                   # Run all tasks on a single node
#SBATCH --ntasks=1                  # Request 1 task (our python script)
#SBATCH --cpus-per-task=1           # Request 1 CPU core per task
#SBATCH --mem=4G                    # Request 4 GB of memory per task
#SBATCH --array=0-19                 # Creates a job array with 10 tasks, indexed 0-9
#SBATCH --mail-user=<update_email>@buffalo.edu
#SBATCH --mail-type=FAIL,END

# --- Output/Error Logging ---
# Create a directory for logs if it doesn't exist
mkdir -p slurm_logs
# %A is the job ID, %a is the array task ID
#SBATCH --output=slurm_logs/out/ppdf_%A_%a.out
#SBATCH --error=slurm_logs/err/ppdf_%A_%a.err

# --- Environment Setup ---
echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================================="

# Loading virtual environment
source ../venv/bin/activate

# --- Execute the Python Script ---
# The script is called with the Slurm array task ID as its argument.
# Slurm will run this command 10 times, with $SLURM_ARRAY_TASK_ID being 0, 1, 2, ..., 9
python arg_ppdf_calculation_no_mpi_layouts.py $SLURM_ARRAY_TASK_ID