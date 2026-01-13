#!/bin/bash
#SBATCH --job-name=cloudytile_train
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

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
SAVE_PATH="/oak/stanford/groups/cyaolai/JoshRines/models/cloudytile/best_model.pth"

# Create model output directory
mkdir -p "$(dirname $SAVE_PATH)"

# Add repo to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Wandb offline mode (no internet on compute nodes)
export WANDB_MODE=offline

# Run training
python3 $REPO_DIR/engine/training/run_training.py \
    --labels_csv $LABELS_CSV \
    --image_dir $IMAGE_DIR \
    --save_path $SAVE_PATH \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3 \
    --img_size 512 \
    --num_workers 8 \
    --optimize_metric precision \
    --wandb_project cloudy-tile \
    --wandb_name "sherlock-run"

echo "Training complete!"
echo "Model saved to: $SAVE_PATH"
echo ""
echo "To sync wandb logs, run from login node:"
echo "  wandb sync $REPO_DIR/engine/training/wandb/offline-run-*"
