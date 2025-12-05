#!/bin/bash
#SBATCH --job-name=ppdf_chunk_array
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=12G
#SBATCH --array=0-54
#SBATCH --mail-user=smehta28@buffalo.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --output=slurm_logs/out/chunk_%A_%a.out
#SBATCH --error=slurm_logs/err/chunk_%A_%a.err

set -euo pipefail
mkdir -p slurm_logs/out slurm_logs/err

# ----- Static config -----
LAYOUT=0
SCANNER_FILE="../data/scanner_layouts/mph_hourglass_single_position_base_3d_v2.tensor"
OUT_SHARDS_DIR="../data/mph_hourglass_single_position_base_3d_v2/outputs_3d_shards"
N_CRYSTALS=1364        # total detectors

# How many detectors per array task
DETS_PER_CHUNK=25

# FOV/grid
NX=256; NY=256; NZ=8
SX=64.0; SY=64.0; SZ=2.0
TILES="4,4,4"

# Ensure we’re running from the submit directory
cd "$SLURM_SUBMIT_DIR"

# Python env
source ../venv/bin/activate
PYTHON="$(command -v python)"

# Let BLAS / Torch use all cpus-per-task in THIS single process
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$OUT_SHARDS_DIR"

# Sanity checks
if [[ ! -r "$SCANNER_FILE" ]]; then
  echo "ERROR: Cannot read scanner file: $SCANNER_FILE"
  exit 13
fi
if [[ ! -f "$SLURM_SUBMIT_DIR/arg_ppdf_calculation_3d_batch.py" ]]; then
  echo "ERROR: Missing arg_ppdf_calculation_3d_batch.py in submit dir: $SLURM_SUBMIT_DIR"
  exit 14
fi

echo "=========================================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Array ID    : $SLURM_ARRAY_TASK_ID"
echo "Host        : $(hostname)"
echo "CPUs/task   : $SLURM_CPUS_PER_TASK"
echo "Start Time  : $(date)"
echo "=========================================================="

ARRAY_MIN=${SLURM_ARRAY_TASK_MIN:-0}
ARRAY_MAX=${SLURM_ARRAY_TASK_MAX:-24}
ARRAY_SIZE=$(( ARRAY_MAX - ARRAY_MIN + 1 ))
CHUNK_ID=$(( SLURM_ARRAY_TASK_ID - ARRAY_MIN ))

# Compute which detector range this array task will handle
START=$(( CHUNK_ID * DETS_PER_CHUNK ))
if (( START >= N_CRYSTALS )); then
  echo "Array task $SLURM_ARRAY_TASK_ID has no work (START=$START >= N_CRYSTALS=$N_CRYSTALS)."
  exit 0
fi
REM=$(( N_CRYSTALS - START ))
COUNT=$(( REM < DETS_PER_CHUNK ? REM : DETS_PER_CHUNK ))

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: START_DET=$START  COUNT=$COUNT"

"$PYTHON" -u "$SLURM_SUBMIT_DIR/arg_ppdf_calculation_3d_batch.py" \
    --layout "$LAYOUT" \
    --start "$START" \
    --count "$COUNT" \
    --out_shards "$OUT_SHARDS_DIR" \
    --nx "$NX" --ny "$NY" --nz "$NZ" \
    --sx "$SX" --sy "$SY" --sz "$SZ" \
    --tiles "$TILES"  \
    --chunk-size 50000 \
    --layout_file "$SCANNER_FILE"

echo "=========================================================="
echo "End Time: $(date)"
echo "=========================================================="
