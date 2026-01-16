#!/bin/bash
#SBATCH --job-name=extract_nc
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# Extract single-timestep NC files for training from existing labeled JPGs
#
# This script:
# 1. Reads your existing labels.csv
# 2. Finds the corresponding timesteps in the processed lake .nc files
# 3. Extracts just those timesteps as individual .nc files with spectral bands
#
# Prerequisites:
# - Run lake-vision preprocessing first (preprocess_tstacks.sh) to create
#   processed .nc files with spectral bands (RGB + NIR + SWIR1 + SWIR2)
# =============================================================================

# Load modules
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312

# Install xarray if needed
pip install --user xarray netcdf4

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# =============================================================================
# CONFIGURE THESE PATHS
# =============================================================================

# Path to your existing labels CSV (with 'filename' and 'label' columns)
LABELS_CSV="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile/labels.csv"

# Directory containing processed lake .nc files (from lake-vision preprocessing)
# These should have spectral bands: red, green, blue, nir, swir16, swir22, mask
INPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_processed"

# Directory to save single-timestep .nc files for training
OUTPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc"

# Repository directory
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"

# Channels to include in training NC files (mask is excluded by default)
# Note: Sentinel-2 uses 'swir16' (band 11, 1.6μm) and 'swir22' (band 12, 2.2μm)
CHANNELS="red green blue nir swir16 swir22"

# =============================================================================
# RUN EXTRACTION
# =============================================================================

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Extracting NC files for training"
echo "=============================================="
echo "Labels CSV: $LABELS_CSV"
echo "Input dir:  $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Channels:   $CHANNELS"
echo ""

python3 $REPO_DIR/engine/preprocessing/extract_nc_from_labels.py \
    --labels_csv $LABELS_CSV \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --channels $CHANNELS

echo ""
echo "=============================================="
echo "Done! NC files saved to $OUTPUT_DIR"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Update run_training.sh to use --use_nc --nc_dir $OUTPUT_DIR"
echo "2. Run training with: sbatch run_training.sh"
