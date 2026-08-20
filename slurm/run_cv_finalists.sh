#!/bin/bash
#SBATCH --job-name=cv_final
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --array=0-5%4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# Finalist rerun: a few promising configs, longer and with more folds.
#
# WHY NOT A FINER GRID: the 40-epoch grid answered the axis questions (does
# NIR help, does capacity bind). What it could not answer is the ordering
# among the top few configs, whose gaps (~0.0007 AUC) sit far inside the
# fold-to-fold spread (~0.0041 AUC). More grid points do not fix that; more
# folds and more epochs do.
#
#   3 folds -> SE on the mean = 0.0041/sqrt(3) = 0.0024
#   5 folds -> SE on the mean = 0.0041/sqrt(5) = 0.0018
#
# WHY LONGER: the 40-epoch grid was budget-limited. Across 51 complete curves,
# epochs 31-40 still bought +0.0112 val loss and +0.0074 val accuracy and the
# curves had not turned. Truncation was also uneven -- `wide` hit the ceiling
# in 71% of runs vs 42% for `small`, lr 3e-4 in 67% vs 46% for 1e-3 -- so those
# two axes were confounded with the budget. 100 epochs decouples them.
#
# OUT_DIR MUST DIFFER from any previous grid's. Finished (config, fold) JSONs
# are skipped on rerun, so reusing a directory would either do nothing or blend
# two epoch budgets into one ranking that --summarize cannot detect.
#
# Cost at 24.0 s/epoch (measured on the 40-epoch grid):
#   6 configs x 5 folds x 100 epochs ~ 20 GPU-h ~ 5 h wall at 4 concurrent
#   ~3.3 h per array task, inside the 12 h walltime.
# =============================================================================

# --- WHICH CONFIGS ------------------------------------------------------------
# Indices into GRID in engine/run_cv_grid.py, which is
#   itertools.product(BAND_SETS, CHANNEL_SETS, LRS, OPTIMIZERS)
# i.e. index = 8*bands + 4*arch + 2*lr + optimizer, with
#   bands  0=rgb 1=rgb+nir 2=rgb+swir16 3=all6
#   arch   0=small 1=wide
#   lr     0=1e-3  1=3e-4
#   opt    0=adam  1=adamw
#
# List them by name first, then resolve to indices:
#   python3 -c "import sys; sys.path.insert(0,'engine'); \
#     from run_cv_grid import GRID, config_name; \
#     [print(i, config_name(c)) for i,c in enumerate(GRID)]"
#
# Chosen from the completed 40-epoch grid (96/96 runs, 32/32 configs).
#
# adamw only: the optimizer axis was null (+0.0012 AUC, inside the 0.0037 fold
# spread) and adamw led, so it is fixed rather than searched. rgb is dropped
# outright -- all 8 rgb configs occupied the bottom 8 of 32 ranks, a separation
# with odds near 1 in 10.5 million by chance. Both `wide` and lr 3e-4 are kept
# on purpose: those arms hit the 40-epoch ceiling most often (71% and 67% of
# runs), so their null result there is confounded with the budget and deserves
# a fair retest rather than being written off.
#
#   17  rgb+swir16_small_lr0.001_adamw    grid #1,  0.9780
#    9  rgb+nir_small_lr0.001_adamw       grid #3,  0.9768, tightest spread (0.0010)
#   15  rgb+nir_wide_lr0.0003_adamw       grid #2,  0.9775, wide + slow lr
#   23  rgb+swir16_wide_lr0.0003_adamw    grid #7,  0.9759, wide + slow lr
#   19  rgb+swir16_small_lr0.0003_adamw   grid #9,  0.9757, slow lr
#   25  all6_small_lr0.001_adamw          grid #11, 0.9757, do 6 bands win at convergence?
#
# NOTE: the --array range above must match the length of CONFIGS. Six configs
# means --array=0-5%4.
CONFIGS=(17 9 15 23 19 25)

if [ "${CONFIGS[0]}" = "TBD" ]; then
    echo "ERROR: CONFIGS is still the placeholder. Fill it in from --summarize first." >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#CONFIGS[@]}" ]; then
    echo "ERROR: array task $SLURM_ARRAY_TASK_ID exceeds ${#CONFIGS[@]} configs; fix --array." >&2
    exit 1
fi

CONFIG_INDEX="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "array task $SLURM_ARRAY_TASK_ID -> GRID config index $CONFIG_INDEX"

# --- ENVIRONMENT --------------------------------------------------------------
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
OUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/cv_results_finalists"
SPLIT_DIR="$REPO_DIR/splits/cloudytile_v1"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
mkdir -p "$OUT_DIR"

python3 "$REPO_DIR/engine/run_cv_grid.py" \
    --labels_csv "$REPO_DIR/labels/labels.csv" \
    --split_dir "$SPLIT_DIR" \
    --nc_dir "$NC_DIR" \
    --band_stats "$BAND_STATS" \
    --out_dir "$OUT_DIR" \
    --config_index "$CONFIG_INDEX" \
    --folds 5 \
    --epochs 100 \
    --img_size 256 \
    --num_workers 8 \
    --wandb_project cloudy-tile-cv-finalists
