#!/bin/bash
#SBATCH --job-name=cloudytile_sweep
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# WANDB SWEEP RUNNER FOR SHERLOCK
# =============================================================================
#
# USAGE (three-step process):
#
# 1. Create the sweep from LOGIN NODE (has internet):
#    $ cd /oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile/engine/training
#    $ wandb sweep sweep.yaml
#    # This prints: "Created sweep with ID: <USERNAME>/<PROJECT>/<SWEEP_ID>"
#
# 2. Submit this job with the FULL sweep path:
#    $ sbatch run_sweep.sh <USERNAME>/<PROJECT>/<SWEEP_ID>
#    # Example: sbatch run_sweep.sh jrines/cloudy-tile/abc123xyz
#
# 3. After job completes, sync from LOGIN NODE:
#    $ cd /oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile/engine/training
#    $ wandb sync wandb/offline-run-*
#
# =============================================================================

# Check for sweep ID argument
if [ -z "$1" ]; then
    echo "ERROR: Sweep ID required!"
    echo ""
    echo "Usage: sbatch run_sweep.sh <SWEEP_PATH>"
    echo ""
    echo "First create the sweep from a login node:"
    echo "  cd /oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile/engine/training"
    echo "  wandb sweep sweep.yaml"
    echo ""
    echo "Then submit with the returned sweep path (e.g., jrines/cloudy-tile/abc123):"
    echo "  sbatch run_sweep.sh jrines/cloudy-tile/abc123"
    exit 1
fi

SWEEP_PATH=$1
echo "Running wandb sweep agent for: $SWEEP_PATH"

# Create logs directory if it doesn't exist
mkdir -p /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs

# Load modules and activate environment
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml cuda/12.2.0
ml cudnn/8.9.5.29

# Set paths
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
LABELS_CSV="$REPO_DIR/labels.csv"
IMAGE_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/jpg_tiles"

# Add repo to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Wandb offline mode (no internet on compute nodes)
export WANDB_MODE=offline

# Change to training directory
cd $REPO_DIR/engine/training

# Run the sweep agent
# --count limits number of runs (remove for unlimited until sweep completes)
wandb agent --count 20 $SWEEP_PATH

echo ""
echo "========================================"
echo "Sweep agent complete!"
echo "========================================"
echo ""
echo "To sync results to wandb.ai, run from LOGIN NODE:"
echo "  cd $REPO_DIR/engine/training"
echo "  wandb sync wandb/offline-run-*"
echo ""
echo "View results at: https://wandb.ai/$SWEEP_PATH"
