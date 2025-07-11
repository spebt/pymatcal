#!/bin/bash
#SBATCH --job-name=ppdf_torch
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --cluster=ub-hpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/ppdf_%j.out
#SBATCH --error=logs/ppdf_%j.err

# Load modules if needed (skip if your venv has everything)
# module load pytorch/2.0.1

# Activate your virtual environment
source ~/myenv/bin/activate

# Run your script
python /vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern_rots_trans/ppdf_calculation_no_mpi_layouts.py
#python /vscratch/grp-rutaoyao/Harsh/new_config_mic/ppdf-analysis/beam-analysis/extract_beams_properties.py
#python /vscratch/grp-rutaoyao/Harsh/new_config_mic/ppdf-analysis/beam-analysis/extract_beams_masks.py
#python /vscratch/grp-rutaoyao/Harsh/new_config_mic/ppdf-analysis/beam-analysis/analyze_extracted_properties.py
#python /vscratch/grp-rutaoyao/Harsh/new_config_mic/ppdf-analysis/beam-analysis/calculate_and_plot_ppds.py
