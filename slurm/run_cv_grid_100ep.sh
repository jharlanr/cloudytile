#!/bin/bash
#SBATCH --job-name=cv_grid100
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --array=0-31%4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# Lake-grouped CV grid, 100-epoch rerun.
#
# WHY THIS EXISTS: the 40-epoch grid was budget-limited. Across 51 complete
# curves, epochs 31-40 still bought +0.0112 val loss and +0.0074 val accuracy,
# and the curves had not turned. Worse, truncation was uneven -- `wide` hit the
# ceiling in 71% of runs vs 42% for `small`, and lr 3e-4 in 67% vs 46% for
# 1e-3. Those are exactly the arms that scored as "no difference", so the arch
# and lr conclusions were confounded with the epoch budget. (The band result
# was not: rgb+nir and rgb truncated at similar rates.)
#
# OUT_DIR MUST DIFFER from the 40-epoch grid's. Finished (config, fold) JSONs
# are skipped on rerun, so pointing this at the old directory would either do
# nothing or, worse, blend 40- and 100-epoch results into a single ranking that
# --summarize cannot detect. Same reason for the distinct wandb project.
#
# Cost: 24.0 s/epoch measured across all configs on the 40-epoch grid
#   => 32 configs x 3 folds x 100 epochs ~ 64 GPU-h ~ 16 h wall at 4 concurrent
#   => 2.0 h per array task, well inside the 8 h walltime.
#
# Aggregate afterwards from a login node:
#   python3 engine/run_cv_grid.py --out_dir <OUT_DIR> --summarize
# Sync wandb from a LOGIN node (compute nodes have no internet):
#   wandb sync /oak/.../sherlock_cloudytile/wandb/offline-run-*
#
# The winner is then retrained by engine/run_training.py against
# splits/cloudytile_v1 -- that run is the only one that touches the 80 test
# lakes, and it is where the reported number comes from.
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

# Compute nodes have no internet; wandb buffers to disk and is synced later
export WANDB_MODE=offline
export WANDB_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile"

REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc_10k"
BAND_STATS="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats_10k.json"
OUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/cv_results_100ep"
SPLIT_DIR="$REPO_DIR/splits/cloudytile_v1"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
mkdir -p "$OUT_DIR"

python3 "$REPO_DIR/engine/run_cv_grid.py" \
    --labels_csv "$REPO_DIR/labels/labels.csv" \
    --split_dir "$SPLIT_DIR" \
    --nc_dir "$NC_DIR" \
    --band_stats "$BAND_STATS" \
    --out_dir "$OUT_DIR" \
    --config_index "$SLURM_ARRAY_TASK_ID" \
    --folds 3 \
    --epochs 100 \
    --img_size 256 \
    --num_workers 8 \
    --wandb_project cloudy-tile-cv-100ep
