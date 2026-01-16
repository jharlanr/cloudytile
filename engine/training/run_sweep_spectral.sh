#!/bin/bash
#SBATCH --job-name=spectral_sweep
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C "GPU_SKU:A100"
#SBATCH --array=0-359%8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# SPECTRAL BAND SWEEP RUNNER
# =============================================================================
#
# This sweep tests different band combinations (RGB, RGB+NIR, RGB+SWIR, etc.)
# along with architecture hyperparameters.
#
# USAGE (three-step process):
#
# 1. Create the sweep from LOGIN NODE (has internet):
#    $ cd /oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile/engine/training
#    $ wandb sweep sweep_spectral.yaml
#    # This prints: "Created sweep with ID: <USERNAME>/<PROJECT>/<SWEEP_ID>"
#
# 2. Submit this job with the FULL sweep path:
#    $ sbatch run_sweep_spectral.sh <USERNAME>/<PROJECT>/<SWEEP_ID>
#    # Example: sbatch run_sweep_spectral.sh jrines/cloudy-tile/abc123xyz
#
#    To submit multiple parallel agents (recommended for 360 runs):
#    $ for i in {1..4}; do sbatch run_sweep_spectral.sh jrines/cloudy-tile/abc123xyz; done
#    NOTE: limit to 4 jobs on serc partition (can go up to 8 if on a deadline, but should touch base with admin first)
#
# 3. After jobs complete, sync from LOGIN NODE:
#    $ cd /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile
#    $ wandb sync wandb/offline-run-*
#
# 4. Check sweep status and resume if needed:
#    $ wandb sweep --status jrines/cloudy-tile/XXXXXX
#    If runs are still pending, submit more agents with the same command from step 2.
#
# =============================================================================

# Check for sweep ID argument
if [ -z "$1" ]; then
    echo "ERROR: Sweep ID required!"
    echo ""
    echo "Usage: sbatch run_sweep_spectral.sh <SWEEP_PATH>"
    echo ""
    echo "First create the sweep from a login node:"
    echo "  cd /oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile/engine/training"
    echo "  wandb sweep sweep_spectral.yaml"
    echo ""
    echo "Then submit with the returned sweep path (e.g., jrines/cloudy-tile/abc123):"
    echo "  sbatch run_sweep_spectral.sh jrines/cloudy-tile/abc123"
    exit 1
fi

SWEEP_PATH=$1
echo "Running spectral band sweep agent for: $SWEEP_PATH"

# Create logs directory if it doesn't exist
mkdir -p /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs

# Load modules and activate environment
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
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile"
LABELS_CSV="$REPO_DIR/labels.csv"
IMAGE_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/jpg_tiles"
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc"

# Add repo to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Wandb offline mode (no internet on compute nodes)
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"

cd $SHERLOCK_DIR

echo ""
echo "=============================================="
echo "Spectral Band Sweep Configuration"
echo "=============================================="
echo "Labels CSV: $LABELS_CSV"
echo "Image dir:  $IMAGE_DIR"
echo "NC dir:     $NC_DIR"
echo "Sweep path: $SWEEP_PATH"
echo ""
echo "Band combinations being tested:"
echo "  - RGB (3 ch)"
echo "  - RGB+SWIR1, RGB+SWIR2 (4 ch)"
echo "  - RB+SWIR1, RB+SWIR2 (3 ch)"
echo "  - B+SWIR1, B+SWIR2 (2 ch)"
echo "  - All above + NIR variants"
echo "  - Full 6-channel (RGB+NIR+SWIR1+SWIR2)"
echo "=============================================="
echo ""

# Run the sweep agent
# Grid sweep: 15 bands x 3 lr x 2 batch x 2 channels x 2 fc = 360 runs
# Remove --count to run entire grid, or set to limit
wandb agent $SWEEP_PATH

echo ""
echo "========================================"
echo "Spectral sweep agent complete!"
echo "========================================"
echo ""
echo "To sync results to wandb.ai, run from LOGIN NODE:"
echo "  wandb sync $SHERLOCK_DIR/wandb/offline-run-*"
echo ""
echo "View results at: https://wandb.ai/$SWEEP_PATH"
echo ""
echo "Filter runs by band combination in wandb:"
echo "  - config.band_combo (e.g., 'red+green+blue+nir')"
echo "  - config.has_nir (true/false)"
echo "  - config.has_swir1 (true/false)"
echo "  - config.n_channels (2-6)"
