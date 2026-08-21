#!/bin/bash
#SBATCH --job-name=cv_bandhead
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.err
#SBATCH --time=30:00:00
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
# BANDS x HEADS sweep at deployment resolution. 4 band sets x 4 heads = 16
# configs, one per array task (index maps 1:1, no lookup table).
#
#   bands  rgb | rgb+nir | rgb+swir16 | rgb+swir22
#          RGB is the control; each other set adds exactly one band, so the
#          comparison isolates that band.
#
#   head   gap     | 128 ch pooled to 1x1  -> 128 values |   228,609 params
#          mixed   | 8 ch pooled to 4x4    -> 128 values |   229,657
#          spatial | 2 ch pooled to 8x8    -> 128 values |   228,871
#          full    | 128 ch at 8x8, 128 hid-> 8,192      | 1,276,401
#          The first three emit the same 128 values into the same 8-unit MLP
#          and differ in size by 0.46%, so any difference between them is
#          inductive bias -- what vs where -- and not capacity. "full" is a
#          deliberate 5.6x capacity probe; it widens the flatten AND the hidden
#          layer, so a win there needs a follow-up to attribute.
#
# ARCHITECTURE deep6 = [16,32,64,64,96,128], six conv blocks. Depth is what buys
# receptive field: 22px at 3 blocks vs 190px at 6, on a 512px tile. It is nearly
# free (1.3x the MACs of the 3-block stack) because blocks 4-6 run on 32x32 and
# smaller grids. Doubling the first block instead would cost 4.6x.
#
# THREE AXES ARE DELIBERATELY ABSENT. Conv width, learning rate and optimizer
# were each measured twice and landed inside the fold spread (+0.0001 then
# +0.0010; 0.9734 vs 0.9734; +0.0012 with AdamW ahead). They are fixed at
# deep6 / 1e-3 / adamw rather than re-tested.
#
# 512 px because that is what inference uses everywhere -- a model selected at
# one resolution and deployed at another is not the model that was selected.
# Cosine annealing because at constant lr the epoch-to-epoch swing in val loss
# exceeded the trend it was meant to reveal, making best-checkpoint selection
# partly a lottery; annealing cut that swing ~8x.
#
# OUT_DIR MUST DIFFER from every previous grid. Finished (config, fold) JSONs
# are skipped on rerun, so reusing a directory would blend regimes into one
# ranking that --summarize cannot detect. Resume is per (config, fold): a task
# killed partway loses at most the fold in flight.
#
# Cost: ~71 s/epoch at 512 px (head FLOPs are negligible next to the stack's
# 248M MACs, so all four heads cost the same). 5 folds x 200 epochs ~ 19.7 h per
# task; 16 tasks 4-concurrent ~ 79 h wall, ~315 A100-hours. The 30 h walltime
# leaves ~1.5x headroom per task.
#
# Aggregate from a LOGIN node when it finishes:
#   python3 engine/run_cv_grid.py --out_dir <OUT_DIR> --summarize
#   wandb sync /oak/.../sherlock_cloudytile/wandb/offline-run-<DATE>_*
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
OUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/cv_results_bandhead"
SPLIT_DIR="$REPO_DIR/splits/cloudytile_v1"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
mkdir -p "$OUT_DIR"

python3 "$REPO_DIR/engine/run_cv_grid.py" \
    --grid bandhead \
    --config_index "$SLURM_ARRAY_TASK_ID" \
    --labels_csv "$REPO_DIR/labels/labels.csv" \
    --split_dir "$SPLIT_DIR" \
    --nc_dir "$NC_DIR" \
    --band_stats "$BAND_STATS" \
    --out_dir "$OUT_DIR" \
    --folds 5 \
    --epochs 200 \
    --img_size 512 \
    --lr_schedule cosine \
    --no_augment \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --num_workers 8 \
    --seed 42 \
    --threshold_objective f1 \
    --wandb_project cloudy-tile-cv-bandhead
