#!/bin/bash
#SBATCH --job-name=train_top3
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.err
#SBATCH --time=:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# TRAIN TOP 3 SPECTRAL MODELS AND SAVE WEIGHTS
# =============================================================================
#
# Trains the 3 best-performing spectral band combinations and saves model weights
# for use in inference on lake NC files.
#
# Top performers:
#   0: RGB (red, green, blue)
#   1: RGB+NIR (red, green, blue, nir)
#   2: B+NIR+SWIR16 (blue, nir, swir16)
#
# USAGE:
#   sbatch train_top3_models.sh
#
# OUTPUT:
#   Model weights saved to:
#     /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/models/
#       - cloudytile_rgb.pth
#       - cloudytile_rgbn.pth
#       - cloudytile_bns16.pth
#
# =============================================================================

# Define the 3 model configurations
declare -a BAND_COMBOS=(
    "red,green,blue"           # 0: RGB
    "red,green,blue,nir"       # 1: RGB+NIR
    "blue,nir,swir16"          # 2: B+NIR+SWIR16
)

declare -a MODEL_NAMES=(
    "cloudytile_rgb"
    "cloudytile_rgbn"
    "cloudytile_bns16"
)

# Get this task's configuration
BANDS="${BAND_COMBOS[$SLURM_ARRAY_TASK_ID]}"
MODEL_NAME="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"

# Convert comma-separated to Python list format
NC_CHANNELS="[$(echo $BANDS | sed "s/,/','/g" | sed "s/^/'/" | sed "s/$/'/" )]"

echo "=============================================="
echo "Training Top 3 Models - Task $SLURM_ARRAY_TASK_ID"
echo "=============================================="
echo "Model name: $MODEL_NAME"
echo "Band combination: $BANDS"
echo "NC channels arg: $NC_CHANNELS"
echo "=============================================="

# Create directories
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile"
MODELS_DIR="$SHERLOCK_DIR/models"
mkdir -p "$SHERLOCK_DIR/logs"
mkdir -p "$MODELS_DIR"

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
SAVE_PATH="$MODELS_DIR/${MODEL_NAME}.pth"

# Add repo to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Wandb offline mode
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"
export WANDB_PROJECT="cloudy-tile-top3"
export WANDB_RUN_GROUP="top3_models"
export WANDB_TAGS="top3,$MODEL_NAME,$BANDS"

cd $SHERLOCK_DIR

# Fixed hyperparameters (matching best from sweep)
LR=1e-3
BATCH_SIZE=32
WEIGHT_DECAY=1e-4
EPOCHS=100
IMG_SIZE=512
CHANNELS="[16,32,64]"
FC_LAYERS="[128]"

echo ""
echo "Starting training..."
echo "Model will be saved to: $SAVE_PATH"
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

# Run training with model saving
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
    --optimize_metric loss \
    --save_path "$SAVE_PATH"

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION_SEC=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION_SEC / 60))
DURATION_HR=$((DURATION_MIN / 60))
DURATION_MIN_REM=$((DURATION_MIN % 60))

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Duration: ${DURATION_HR}h ${DURATION_MIN_REM}m (${DURATION_SEC}s total)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    if [ -f "$SAVE_PATH" ]; then
        echo "Model saved to: $SAVE_PATH"
        ls -lh "$SAVE_PATH"
    else
        echo "WARNING: Model file not found at $SAVE_PATH"
    fi
else
    echo "Training FAILED with exit code $EXIT_CODE"
fi
echo "=============================================="

exit $EXIT_CODE
