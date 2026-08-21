#!/bin/bash
#SBATCH --job-name=ct_final
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# THE FINAL MODEL. This is the only run that touches the 80 frozen test lakes.
#
# Config: rgb+nir_small_lr0.001_adamw, the finalist winner. It tied #1 on AUC
# (0.9847 +/- 0.0019 over 5 folds) with rgb+nir_wide_lr0.0003_adamw at 3.4x
# fewer parameters (32,401 vs 110,625), so the tie-break rule -- inside one
# fold spread, take the simpler model -- selects it.
#
# EVERY hyperparameter below is passed explicitly rather than left to a
# default, because run_training.py's defaults do NOT match the regime the
# config grid selected under: it defaults to img_size 512 (grid used 256),
# weight_decay 0.0 (grid used 1e-4), and optimizer adam (winner uses adamw).
# A model evaluated in a different regime than it was chosen in is not the
# model that was chosen.
#
# ON RUNNING THIS MORE THAN ONCE: re-running after a crash costs nothing -- a
# failed run teaches you nothing about the test lakes. What does spend the
# holdout is running it, disliking the number, and switching configs. The test
# estimate is only unbiased for a model chosen without reference to it. If the
# model genuinely has to change after seeing test results, freeze a
# splits/cloudytile_v2 and start the holdout over.
#
# Cost: ~14 s/epoch x 100 epochs on 7,000 training tiles ~ 30 min.
# =============================================================================

ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.2.1_py312
ml py-torchvision/0.17.1_py312
ml py-scikit-learn/1.5.1_py312

pip install --user xarray netcdf4
export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile"

REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc_10k"
BAND_STATS="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats_10k.json"
OUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/final_model"
SPLIT_DIR="$REPO_DIR/splits/cloudytile_v1"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
mkdir -p "$OUT_DIR"

python3 "$REPO_DIR/engine/run_training.py" \
    --labels_csv "$REPO_DIR/labels/labels.csv" \
    --split_dir "$SPLIT_DIR" \
    --nc_dir "$NC_DIR" \
    --band_stats "$BAND_STATS" \
    --nc_channels red green blue nir \
    --channels 16 32 64 \
    --fc_layers 128 \
    --head gap \
    --dropout 0.3 \
    --img_size 256 \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --optimizer adamw \
    --lr_schedule cosine \
    --weight_decay 1e-4 \
    --num_workers 8 \
    --seed 42 \
    --optimize_metric loss \
    --threshold_objective f1 \
    --save_path "$OUT_DIR/cloudytile_v1_best.pth" \
    --wandb_project cloudy-tile-final \
    --wandb_name rgb+nir_small_lr0.001_adamw_FINAL
