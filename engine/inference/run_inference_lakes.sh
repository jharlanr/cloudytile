#!/bin/bash
#SBATCH --job-name=inference_lakes
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# RUN CLOUDY-TILE INFERENCE ON LAKE NC FILES
# =============================================================================
#
# Runs inference using the top 3 spectral models on lake time-stack NC files,
# adding cloudy_seq_rgb, cloudy_seq_rgbn, and cloudy_seq_bns16 variables.
#
# PREREQUISITES:
#   - Run train_top3_models.sh first to generate model weights
#   - Model weights should be in:
#       /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/models/
#
# USAGE:
#   sbatch run_inference_lakes.sh
#
# OUTPUT:
#   Each lake NC file will have 3 new variables added:
#     - cloudy_seq_rgb:   predictions from RGB model
#     - cloudy_seq_rgbn:  predictions from RGB+NIR model
#     - cloudy_seq_bns16: predictions from B+NIR+SWIR16 model
#
# =============================================================================

# Define the 3 model configurations (must match train_top3_models.sh)
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

declare -a VAR_NAMES=(
    "cloudy_seq_rgb"
    "cloudy_seq_rgbn"
    "cloudy_seq_bns16"
)

# Get this task's configuration
BANDS="${BAND_COMBOS[$SLURM_ARRAY_TASK_ID]}"
MODEL_NAME="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"
VAR_NAME="${VAR_NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================="
echo "Lake Inference - Task $SLURM_ARRAY_TASK_ID"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Bands: $BANDS"
echo "Output variable: $VAR_NAME"
echo "=============================================="

# Set paths
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
MODELS_DIR="$SHERLOCK_DIR/models"
MODEL_PATH="$MODELS_DIR/${MODEL_NAME}.pth"
BAND_STATS="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats.json"
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_processed"

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please run train_top3_models.sh first to train and save models."
    exit 1
fi

# Create logs directory
mkdir -p "$SHERLOCK_DIR/logs"

# Load modules
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.2.1_py312
ml py-torchvision/0.17.1_py312

# Install xarray for NC loading
pip install --user xarray netcdf4

# Add repo to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

cd $SHERLOCK_DIR

echo ""
echo "Starting inference..."
echo "Model path: $MODEL_PATH"
echo "NC directory: $NC_DIR"
echo "Band stats: $BAND_STATS"
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

# Run inference using Python
python3 << EOF
import sys
sys.path.insert(0, "$REPO_DIR")

from cloudytile.inference import process_directory

# Configuration
nc_dir = "$NC_DIR"
model_path = "$MODEL_PATH"
band_stats_path = "$BAND_STATS"
nc_channels = "$BANDS".split(",")
var_name = "$VAR_NAME"

# Model architecture (must match training)
channels = [16, 32, 64]
fc_layers = [128]
img_size = (512, 512)

print(f"Running inference with {len(nc_channels)} channels: {nc_channels}")
print(f"Output variable: {var_name}")

n_processed = process_directory(
    nc_dir=nc_dir,
    model_path=model_path,
    img_size=img_size,
    channels=channels,
    fc_layers=fc_layers,
    nc_channels=nc_channels,
    band_stats_path=band_stats_path,
    threshold=0.5,
    batch_size=64,
    pattern="*.nc",
    var_name=var_name,
)

print(f"\nCompleted: {n_processed} files processed")
EOF

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION_SEC=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION_SEC / 60))

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Duration: ${DURATION_MIN}m (${DURATION_SEC}s total)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Inference completed successfully!"
else
    echo "Inference FAILED with exit code $EXIT_CODE"
fi
echo "=============================================="

exit $EXIT_CODE
