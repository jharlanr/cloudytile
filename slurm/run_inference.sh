#!/bin/bash
#SBATCH --job-name=cloudytile_inference
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# CLOUDYTILE INFERENCE SCRIPT
# =============================================================================
#
# USAGE:
#   sbatch run_inference.sh /path/to/nc_dir
#
# This script applies the trained cloudytile model to all .nc files in the
# specified directory, adding/overwriting the 'cloudy_seq' variable.
#
# =============================================================================

# Check for input directory argument
if [ -z "$1" ]; then
    echo "ERROR: Input directory required!"
    echo ""
    echo "Usage: sbatch run_inference.sh /path/to/nc_dir"
    exit 1
fi

INPUT_DIR=$1
echo "Processing NetCDF files in: $INPUT_DIR"

# Create logs directory if it doesn't exist
mkdir -p /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs

# Load modules
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.2.1_py312
ml py-torchvision/0.17.1_py312

# Set paths
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile"
MODEL_PATH="$SHERLOCK_DIR/models/best_model.pth"

# Add repo to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# ============================================================================
# MODEL ARCHITECTURE - MUST MATCH TRAINING
# ============================================================================
# Update these if you trained with different architecture:
CHANNELS="16 32 64"
FC_LAYERS="128"
IMG_SIZE=512
# ============================================================================

# Run inference
python3 $REPO_DIR/engine/run_inference.py \
    --model $MODEL_PATH \
    --input $INPUT_DIR \
    --img_size $IMG_SIZE \
    --channels $CHANNELS \
    --fc_layers $FC_LAYERS \
    --threshold 0.5 \
    --batch_size 32

echo ""
echo "Inference complete!"
