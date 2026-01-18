#!/bin/bash
#SBATCH --job-name=spectral_array
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --array=0-14%8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# SPECTRAL BAND ARRAY JOB
# =============================================================================
#
# Runs 15 training jobs (one per band combination) using SLURM job arrays.
# Max 8 jobs run concurrently (controlled by %8 in --array).
#
# USAGE:
#   sbatch run_spectral_array.sh
#
# TO RERUN FAILED JOBS:
#   Check the status log at:
#     /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/spectral_array_status.log
#
#   Then resubmit specific indices:
#     sbatch --array=3,7,12 run_spectral_array.sh
#
# AFTER JOBS COMPLETE:
#   Sync wandb runs from login node:
#     wandb sync /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/wandb/offline-run-*
#
# =============================================================================

# Define the 15 band combinations (index matches SLURM_ARRAY_TASK_ID)
BAND_COMBOS=(
    # --- WITHOUT NIR (indices 0-6) ---
    "red,green,blue"                      # 0: RGB (3 ch)
    "red,green,blue,swir16"               # 1: RGB+SWIR16 (4 ch)
    "red,green,blue,swir22"               # 2: RGB+SWIR22 (4 ch)
    "red,blue,swir16"                     # 3: RB+SWIR16 (3 ch)
    "red,blue,swir22"                     # 4: RB+SWIR22 (3 ch)
    "blue,swir16"                         # 5: B+SWIR16 (2 ch)
    "blue,swir22"                         # 6: B+SWIR22 (2 ch)

    # --- WITH NIR (indices 7-13) ---
    "red,green,blue,nir"                  # 7: RGB+NIR (4 ch)
    "red,green,blue,nir,swir16"           # 8: RGB+NIR+SWIR16 (5 ch)
    "red,green,blue,nir,swir22"           # 9: RGB+NIR+SWIR22 (5 ch)
    "red,blue,nir,swir16"                 # 10: RB+NIR+SWIR16 (4 ch)
    "red,blue,nir,swir22"                 # 11: RB+NIR+SWIR22 (4 ch)
    "blue,nir,swir16"                     # 12: B+NIR+SWIR16 (3 ch)
    "blue,nir,swir22"                     # 13: B+NIR+SWIR22 (3 ch)

    # --- ALL BANDS (index 14) ---
    "red,green,blue,nir,swir16,swir22"    # 14: Full 6-channel
)

# Get this task's band combination
BANDS="${BAND_COMBOS[$SLURM_ARRAY_TASK_ID]}"

# Convert comma-separated to Python list format for the argument
# e.g., "red,green,blue" -> "['red','green','blue']"
NC_CHANNELS="[$(echo $BANDS | sed "s/,/','/g" | sed "s/^/'/" | sed "s/$/'/" )]"

echo "=============================================="
echo "Spectral Band Array Job"
echo "=============================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Band combination: $BANDS"
echo "NC channels arg: $NC_CHANNELS"
echo "=============================================="

# Create logs directory if it doesn't exist
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile"
mkdir -p "$SHERLOCK_DIR/logs"

# Status log file (shared across all array tasks)
STATUS_LOG="$SHERLOCK_DIR/logs/spectral_array_${SLURM_ARRAY_JOB_ID}_status.log"

# Load modules
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.2.1_py312
ml py-torchvision/0.17.1_py312
ml py-scikit-learn/1.5.1_py312

# Install xarray for NC loading
pip install --user xarray netcdf4

# Set paths
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
LABELS_CSV="$REPO_DIR/labels.csv"
IMAGE_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/jpg_tiles"
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc"
BAND_STATS="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats.json"

# Add repo to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Wandb offline mode
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"
export WANDB_PROJECT="cloudy-tile-spectral"
export WANDB_RUN_GROUP="spectral_array_${SLURM_ARRAY_JOB_ID}"
export WANDB_TAGS="spectral,array,$BANDS"

cd $SHERLOCK_DIR

# Fixed hyperparameters (matching sweep_spectral.yaml)
LR=1e-3
BATCH_SIZE=32
WEIGHT_DECAY=1e-4
EPOCHS=100
IMG_SIZE=512
CHANNELS="[16,32,64]"
FC_LAYERS="[128]"

echo ""
echo "Starting training..."
echo "Start time: $(date)"
echo ""

# Record start time for duration calculation
START_TIME=$(date +%s)

# Run training and capture exit code
python3 "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_CSV" \
    --image_dir "$IMAGE_DIR" \
    --nc_dir "$NC_DIR" \
    --band_stats "$BAND_STATS" \
    --use_nc \
    --use_scheduler true \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --weight_decay $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --img_size $IMG_SIZE \
    --channels "$CHANNELS" \
    --fc_layers "$FC_LAYERS" \
    --nc_channels "$NC_CHANNELS" \
    --optimize_metric precision

EXIT_CODE=$?

# Calculate duration
END_TIME=$(date +%s)
DURATION_SEC=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION_SEC / 60))
DURATION_HR=$((DURATION_MIN / 60))
DURATION_MIN_REM=$((DURATION_MIN % 60))

echo ""
echo "End time: $(date)"
echo "Duration: ${DURATION_HR}h ${DURATION_MIN_REM}m (${DURATION_SEC}s total)"
echo "Exit code: $EXIT_CODE"

# Write status to shared log file (with file locking to avoid race conditions)
{
    flock -x 200
    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') | Task $SLURM_ARRAY_TASK_ID | SUCCESS | ${DURATION_HR}h${DURATION_MIN_REM}m | $BANDS" >> "$STATUS_LOG"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') | Task $SLURM_ARRAY_TASK_ID | FAILED (exit $EXIT_CODE) | ${DURATION_HR}h${DURATION_MIN_REM}m | $BANDS" >> "$STATUS_LOG"
    fi
} 200>"$STATUS_LOG.lock"

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training FAILED with exit code $EXIT_CODE"
fi
echo "Status logged to: $STATUS_LOG"
echo "=============================================="
echo ""
echo "To sync wandb runs after all jobs complete:"
echo "  wandb sync $SHERLOCK_DIR/wandb/offline-run-*"
echo ""
echo "To sync only runs from a specific date (e.g., Jan 17, 2026):"
echo "  wandb sync $SHERLOCK_DIR/wandb/offline-run-20260117-*"

exit $EXIT_CODE
