#!/bin/bash

#SBATCH --job-name=ppdf_layout_gen      # Job name for identification
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --time=02:00:00                 # Walltime limit (HH:MM:SS)
#SBATCH --nodes=1                       # Run all tasks on a single node
#SBATCH --ntasks=1                      # Request 1 task (our python script)
#SBATCH --cpus-per-task=10               # Request 2 CPU cores for PyTorch
#SBATCH --mem=4G                        # Request 4 GB of memory per task
#SBATCH --array=0-39                    # Creates a job array for 20 layouts, indexed 0-19
#SBATCH --mail-user=smehta28@buffalo.edu
#SBATCH --mail-type=FAIL,END

# --- Output/Error Logging ---
# Create a directory for logs if it doesn't exist
mkdir -p slurm_logs/out slurm_logs/err
# %A is the main job ID, %a is the array task ID
#SBATCH --output=slurm_logs/out/ppdf_%A_%a.out
#SBATCH --error=slurm_logs/err/ppdf_%A_%a.err

# --- Environment Setup ---
echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start Time: $(date)"
echo "=========================================================="

# Load necessary modules (if any, e.g., module load anaconda3)

# Activate your Python environment
source /vscratch/grp-rutaoyao/sid/venv/bin/activate

# --- Execute the Python Script ---
# The script is called with the Slurm array task ID as its argument.
# Slurm runs this command for each task ID in the array (0, 1, 2, ..., 19)
echo "Executing Python script for layout index $SLURM_ARRAY_TASK_ID..."

python arg_ppdf_calculation.py $SLURM_ARRAY_TASK_ID

echo "=========================================================="
echo "End Time: $(date)"
echo "Job finished with exit code $?"
echo "=========================================================="
