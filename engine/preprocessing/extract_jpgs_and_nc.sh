#!/bin/bash
#SBATCH --job-name=extract_all
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# Extract JPG + NC files from scratch for a new labeling + training dataset
#
# This script generates BOTH:
# - JPG files (for uploading to Labelbox and labeling)
# - NC files (for training with spectral bands)
#
# Prerequisites:
# - Run lake-vision preprocessing first (preprocess_tstacks.sh) to create
#   processed .nc files with spectral bands (RGB + NIR + SWIR1 + SWIR2)
#
# Workflow:
# 1. Run this script to generate JPGs + NCs
# 2. Upload JPGs to Labelbox for labeling
# 3. Export labels as labels.csv
# 4. Train with: --use_nc --nc_dir <NC_OUTPUT_DIR>
# =============================================================================

# Load modules
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml viz
ml py-matplotlib/3.8.3_py312
ml py-scipy/1.12.0_py312

# Install dependencies if needed
pip install --user xarray netcdf4

# =============================================================================
# CONFIGURE THESE PATHS
# =============================================================================

# Directory containing processed lake .nc files (from lake-vision preprocessing)
# These should have spectral bands: red, green, blue, nir, swir1, swir2, mask
INPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_processed"

# Directory to save JPG files (for Labelbox labeling)
JPG_OUTPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/labeling_jpgs"

# Directory to save single-timestep NC files (for training)
NC_OUTPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc"

# Repository directory
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"

# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================

# Fraction of timesteps to sample from each lake (0.0-1.0)
# 0.15 = 15% of timesteps per lake
SAMPLE_FRACTION=0.15

# Maximum number of lake files to process (leave empty for all)
MAX_FILES=

# Random seed for reproducibility
SEED=42

# Channels to include in NC files for training
CHANNELS="red green blue nir swir1 swir2"

# =============================================================================
# RUN EXTRACTION
# =============================================================================

# Create output directories
mkdir -p $JPG_OUTPUT_DIR
mkdir -p $NC_OUTPUT_DIR

echo "=============================================="
echo "Extracting JPG + NC files for labeling/training"
echo "=============================================="
echo "Input dir:        $INPUT_DIR"
echo "JPG output dir:   $JPG_OUTPUT_DIR"
echo "NC output dir:    $NC_OUTPUT_DIR"
echo "Sample fraction:  $SAMPLE_FRACTION"
echo "Max files:        ${MAX_FILES:-all}"
echo "Channels:         $CHANNELS"
echo ""

# Build command
CMD="python3 $REPO_DIR/engine/preprocessing/extract_jpgs_and_nc.py \
    --input_dir $INPUT_DIR \
    --jpg_output_dir $JPG_OUTPUT_DIR \
    --nc_output_dir $NC_OUTPUT_DIR \
    --sample_fraction $SAMPLE_FRACTION \
    --seed $SEED \
    --channels $CHANNELS"

if [ -n "$MAX_FILES" ]; then
    CMD="$CMD --max_files $MAX_FILES"
fi

# Run extraction
$CMD

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo ""
echo "JPGs saved to: $JPG_OUTPUT_DIR"
echo "NCs saved to:  $NC_OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Upload JPGs to Labelbox: $JPG_OUTPUT_DIR"
echo "  2. Label images and export as labels.csv"
echo "  3. Train with spectral bands:"
echo "     python run_training.py --labels_csv labels.csv \\"
echo "       --image_dir $JPG_OUTPUT_DIR --use_nc --nc_dir $NC_OUTPUT_DIR"
