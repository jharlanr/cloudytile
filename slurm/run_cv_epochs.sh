#!/bin/bash
#SBATCH --job-name=cv_epochs
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
#SBATCH --array=0-14%4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# ANNEALING HORIZON x HEAD. 3 heads x 5 horizons = 15 configs, one per array
# task (heads outer, horizons inner, index = 5*head + horizon):
#
#    0-4   gap      e40 e60 e80 e120 e200
#    5-9   mixed    e40 e60 e80 e120 e200
#   10-14  spatial  e40 e60 e80 e120 e200
#
# WHY. The bands x heads sweep found median best_epoch = 30, max 67 over all 80
# folds -- nothing near the 200-epoch budget. CosineAnnealingLR takes T_max from
# --epochs, so every checkpoint was selected at essentially full learning rate
# and the annealing finished long after the model stopped improving. --epochs is
# the SHAPE OF THE SCHEDULE here, not a budget: shortening it moves the low-lr
# window onto the epochs that actually matter. That is the hypothesis under test.
#
# WHY THE HEAD IS RE-TESTED. gap/mixed/spatial landed within 0.0010 at 200
# epochs, which --summarize correctly called a tie. A tie broken under one
# schedule does not transfer to another, so selecting the head at 200 and
# training the final model at 60 would deploy a config under conditions it was
# never compared in -- the same mistake as selecting at 256 px and inferring at
# 512. Head and horizon are settled together, on dev lakes, here.
#
# WHAT IS FIXED, and on what evidence (all paired per-fold, 20 pairs each):
#   bands rgb+swir16   beat rgb by +0.0058, winning 20/20 folds. rgb+swir22 is
#                      interchangeable (+0.0006, 16/20) -- either would do.
#   head  "full" out   lost to all three matched heads (-0.0024 to -0.0038,
#                      16-17/20) with 2.6x their fold spread. 5.6x the
#                      parameters made it worse, so the 128-value bottleneck
#                      costs nothing.
#   arch/lr/optimizer  measured null twice already; not re-tested.
#
# e200 IS INCLUDED AS THE CONTROL even though the bands x heads sweep already
# ran those three configs. It costs 3 tasks and buys a self-contained table
# whose rows share one provenance, plus a reproducibility check: same seed, same
# folds, same code, so e200 here should land on the bandhead numbers
# (rgb+swir16 gap .9869 / mixed .9879 / spatial .9874). If it does not, that is
# a finding in itself and worth chasing before trusting the rest.
#
# COST, from the bands x heads sweep's own elapsed_sec: ~18 s/epoch at 512 px,
# flat in horizon. Per task = 5 folds x horizon x 18 s -> e40 ~1.0 h, e200 ~5.0 h.
# Whole sweep ~37 GPU-hours, ~12 h wall at 4 concurrent. The 12 h walltime
# covers the longest single task (e200, ~5 h) with 2.4x headroom.
#
# The bands x heads sweep was I/O-bound on the .nc reads, which is why 512 px
# cost about what 256 px did. Expect the same here; the "projected N h/fold"
# line each task prints after epoch 1 is the check.
#
# OUT_DIR is separate from cv_results_bandhead. Names differ anyway (the _e<N>
# suffix), so this is belt-and-braces: one directory, one regime.
#
# Aggregate and upload from a LOGIN node when it finishes:
#   python3 engine/run_cv_grid.py --out_dir <OUT_DIR> --summarize
#   python3 engine/upload_wandb.py --out_dir <OUT_DIR> --sync --table \
#       --project cloudy-tile-cv-epochs
#
# --sync reads wandb_run_dir out of each result JSON and uploads exactly those
# offline runs, so there is no date glob to get wrong. --table posts the
# ranking as one summary run. Add --dry-run to --sync to look first.
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
OUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/cv_results_epochs"
SPLIT_DIR="$REPO_DIR/splits/cloudytile_v1"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
mkdir -p "$OUT_DIR"

# NOTE: --epochs is deliberately NOT passed. Every config in this grid carries
# its own horizon, which overrides the CLI; passing one here would only mislead
# a reader into thinking it applied.
python3 "$REPO_DIR/engine/run_cv_grid.py" \
    --grid epochs \
    --config_index "$SLURM_ARRAY_TASK_ID" \
    --labels_csv "$REPO_DIR/labels/labels.csv" \
    --split_dir "$SPLIT_DIR" \
    --nc_dir "$NC_DIR" \
    --band_stats "$BAND_STATS" \
    --out_dir "$OUT_DIR" \
    --folds 5 \
    --img_size 512 \
    --lr_schedule cosine \
    --no_augment \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --num_workers 8 \
    --seed 42 \
    --threshold_objective f1 \
    --wandb_project cloudy-tile-cv-epochs
