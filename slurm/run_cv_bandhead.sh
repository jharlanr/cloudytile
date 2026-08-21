#!/bin/bash
#SBATCH --job-name=cv_bandhead
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --array=0-15%4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# BAND SWEEP at deployment resolution. One axis only: which bands to feed the
# network. Everything else is held constant at the values the earlier grids
# selected, so any difference between these eight runs is attributable to the
# band set and nothing else.
#
# WHY A FRESH GRID. The 32-config v1 grid ran at 256 px with a constant
# learning rate. Both of those have since changed: cosine annealing gained
# ~0.008 AUC and cut the fold spread 2.6x (0.0037 -> 0.0014), and 512 px is
# what inference actually uses. Conclusions drawn in the old regime are not
# guaranteed to hold in the deployed one, and the band effect is the only
# result the grid produced that is worth being sure about.
#
# WHY BANDS ONLY. Architecture was measured twice and came back null both
# times (wide vs small: +0.0001 at 40 epochs constant-lr, +0.0010 at 100 epochs
# annealed -- both inside the fold spread), and `small` is 3.4x cheaper.
# Learning rate was exactly null (0.9734 vs 0.9734), and annealing to ~0 makes
# the starting value matter even less. AdamW led optimizer consistently. Those
# three axes are settled; re-testing them would double or quadruple the cost to
# re-answer answered questions.
#
# THE EIGHT SETS are the complete lattice: RGB is always present, since it is
# the modality the task is defined on, and every subset of the three
# non-visible bands is tried on top of it. 2^3 = 8.
#
# Indices into GRID in engine/run_cv_grid.py (= 8*band + 4*arch + 2*lr + opt,
# so 8b+1 fixes small/1e-3/adamw and varies only the band set):
#    1  rgb                  3ch   control
#    9  rgb+nir              4ch
#   17  rgb+swir16           4ch
#   33  rgb+swir22           4ch
#   41  rgb+nir+swir16       5ch
#   49  rgb+nir+swir22       5ch
#   57  rgb+swir16+swir22    5ch
#   25  all6                 6ch   (= rgb+nir+swir16+swir22)
CONFIGS=(1 9 17 25 33 41 49 57)

# Second axis: the classifier head. Rather than making it a GRID axis -- which
# would renumber every existing index and silently invalidate the recorded
# finalist indices -- it is a CLI flag, and the array id selects it:
#   tasks  0-7  -> gap     : AdaptiveAvgPool2d(1), 1 number per channel.
#                            Fully translation-invariant; discards WHERE.
#   tasks 8-15  -> pool16  : AdaptiveAvgPool2d(16), a 16x16 grid per channel,
#                            flattened into an 8-wide hidden layer. Coarse
#                            spatial layout survives to the classifier, which
#                            plausibly matters here: half a tile under cloud and
#                            a uniformly hazy tile can average to the same thing
#                            under GAP but are different calls.
# Both are resolution-independent (adaptive pooling fixes the output grid), so
# neither reintroduces the flatten head's dependence on image area.
# fc width is tied to the head: 64x16x16 = 16,384 inputs, so 8 hidden units
# costs 155k params total while 128 would cost 2.1M.
N_BANDS=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge $((2 * N_BANDS)) ]; then
    echo "ERROR: array task $SLURM_ARRAY_TASK_ID exceeds $((2 * N_BANDS)) configs; fix --array." >&2
    exit 1
fi
CONFIG_INDEX="${CONFIGS[$((SLURM_ARRAY_TASK_ID % N_BANDS))]}"
if [ "$SLURM_ARRAY_TASK_ID" -lt "$N_BANDS" ]; then
    HEAD=gap;    FC_LAYERS=128
else
    HEAD=pool16; FC_LAYERS=8
fi
echo "array task $SLURM_ARRAY_TASK_ID -> GRID config index $CONFIG_INDEX, head $HEAD, fc $FC_LAYERS"

# OUT_DIR MUST DIFFER from previous grids. Finished (config, fold) JSONs are
# skipped on rerun, so pointing this at cv_results/ or cv_results_finalists/
# would either do nothing or blend three different regimes into one ranking
# that --summarize cannot detect.
#
# RESUME: each (config, fold) writes its own JSON, so a task killed partway
# loses at most the fold in flight. Resubmit the same sbatch to continue.
#
# AUGMENTATION IS OFF (--no_augment). Flips and rot90 are exact symmetries of
# this task, so this gives up free regularization, but the model is small and
# the finalist runs showed only a ~0.02-0.035 train/val loss gap, so there is
# little overfitting for it to suppress. Note this makes these numbers not
# directly comparable to the augmented grid and finalist runs.
#
# Cost: 512 px is 4x the pixels of the 256 px grids, whose measured median was
# 13.4 s/epoch, so ~54 s/epoch. 5 folds x 150 epochs ~ 11.3 h per task; 16
# tasks 4-concurrent ~ 45 h wall, ~180 A100-hours. The 18 h walltime leaves
# ~1.6x headroom per task for a slow node.
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
OUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/cv_results_bandhead512"
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
    --epochs 150 \
    --img_size 512 \
    --head "$HEAD" \
    --fc_layers "$FC_LAYERS" \
    --no_augment \
    --lr_schedule cosine \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --num_workers 8 \
    --seed 42 \
    --threshold_objective f1 \
    --wandb_project cloudy-tile-cv-bandhead512
