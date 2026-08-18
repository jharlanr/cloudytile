#!/bin/bash
#SBATCH --job-name=band_stats
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# Compute per-band normalization statistics from training NC files
#
# Run this AFTER extract_training_nc.sh and BEFORE training
# =============================================================================

# Load modules
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312

# Install xarray if needed
pip install --user xarray netcdf4

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# =============================================================================
# CONFIGURE THESE PATHS
# =============================================================================

# Directory containing single-timestep NC files for training
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc_10k"

# Output path for band statistics JSON
OUTPUT_PATH="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats_10k.json"

# Number of files to sample (use all if you have time, or sample for speed)
SAMPLE_SIZE=9999999

# Repository directory
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"

# =============================================================================
# RUN COMPUTATION
# =============================================================================

echo "=============================================="
echo "Computing band statistics for normalization"
echo "=============================================="
echo "NC directory: $NC_DIR"
echo "Output path:  $OUTPUT_PATH"
echo "Sample size:  $SAMPLE_SIZE"
echo ""

python3 $REPO_DIR/engine/compute_band_stats.py \
    --nc_dir $NC_DIR \
    --output_path $OUTPUT_PATH \
    --sample_size $SAMPLE_SIZE

echo ""
echo "=============================================="
echo "Done! Band statistics saved to: $OUTPUT_PATH"
echo "=============================================="
echo ""
echo "You can now run training with:"
echo "  --band_stats $OUTPUT_PATH"
