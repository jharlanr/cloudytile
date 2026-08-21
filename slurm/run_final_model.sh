#!/bin/bash
#SBATCH --job-name=ct_final
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=12:00:00
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
# FILL IN THE TWO VARIABLES BELOW from the bands x heads sweep before running:
#   python3 engine/run_cv_grid.py --out_dir <cv_results_bandhead> --summarize
# The script refuses to start while they say TBD, because a final run against
# the holdout is not something to launch with a stale default.
#
# Tie-break rule, applied to that table: if the gap to #2 is smaller than the
# fold-to-fold spread, it is a tie -- take the smaller model. --summarize
# prints this comparison itself.
# =============================================================================

WINNER_BANDS="TBD"      # one of: "red green blue"
                        #         "red green blue nir"
                        #         "red green blue swir16"
                        #         "red green blue swir22"
WINNER_HEAD="TBD"       # one of: gap | mixed | spatial | full

# -----------------------------------------------------------------------------
# Head presets, mirroring HEADS in engine/run_cv_grid.py. Kept as a lookup
# rather than three variables to transcribe by hand: "spatial" IS pool8 +
# head_reduce 2, and getting head_reduce wrong does not fail -- it silently
# trains a much larger model under the winning config's name.
# -----------------------------------------------------------------------------
# HEAD_REDUCE is an ARRAY, expanded as "${HEAD_REDUCE[@]}", so that the empty
# case contributes no argument at all. An unquoted string would work here only
# because bash word-splits it; the same line under zsh passes "--head_reduce 8"
# as a single token and argparse rejects it.
case "$WINNER_HEAD" in
    gap)     HEAD=gap;   HEAD_REDUCE=();                 FC_LAYERS=8   ;;
    mixed)   HEAD=pool4; HEAD_REDUCE=(--head_reduce 8);  FC_LAYERS=8   ;;
    spatial) HEAD=pool8; HEAD_REDUCE=(--head_reduce 2);  FC_LAYERS=8   ;;
    full)    HEAD=pool8; HEAD_REDUCE=();                 FC_LAYERS=128 ;;
    *) echo "ERROR: set WINNER_HEAD to gap|mixed|spatial|full (got '$WINNER_HEAD')" >&2
       exit 1 ;;
esac
if [ "$WINNER_BANDS" = "TBD" ]; then
    echo "ERROR: set WINNER_BANDS from the sweep summary before running" >&2
    exit 1
fi

# =============================================================================
# EVERY hyperparameter is passed explicitly rather than left to a default,
# because run_training.py's defaults do NOT match the selection regime:
# weight_decay defaults to 0.0 (the sweep used 1e-4), optimizer to adam (the
# sweep used adamw), lr_schedule to none (the sweep annealed).
#
# EPOCHS MUST STAY 200. Cosine annealing sets T_max from --epochs, so training
# the winner for a different number of epochs puts it on a different learning
# -rate trajectory -- a different regime from the one it was selected under,
# even with identical architecture.
#
# 512 px matches both the sweep and inference (cloudytile/inference.py,
# run_inference.py, run_inference_lakes.sh). Train at what you infer at.
#
# ON RUNNING THIS MORE THAN ONCE: re-running after a crash costs nothing -- a
# failed run teaches you nothing about the test lakes. What does spend the
# holdout is running it, disliking the number, and switching configs. The test
# estimate is only unbiased for a model chosen without reference to it. If the
# model genuinely has to change after seeing test results, freeze a
# splits/cloudytile_v2 and start the holdout over.
#
# Cost: one 200-epoch fit on 7,000 tiles, ~3-7 h by the sweep's own timings.
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

TAG="$(echo "$WINNER_BANDS" | tr ' ' '+')_${WINNER_HEAD}"
echo "Final model: bands=[$WINNER_BANDS] head=$WINNER_HEAD ($HEAD ${HEAD_REDUCE[*]} fc=$FC_LAYERS)"

python3 "$REPO_DIR/engine/run_training.py" \
    --labels_csv "$REPO_DIR/labels/labels.csv" \
    --split_dir "$SPLIT_DIR" \
    --nc_dir "$NC_DIR" \
    --band_stats "$BAND_STATS" \
    --nc_channels $WINNER_BANDS \
    --channels 16 32 64 64 96 128 \
    --fc_layers $FC_LAYERS \
    --head "$HEAD" "${HEAD_REDUCE[@]}" \
    --dropout 0.3 \
    --img_size 512 \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.001 \
    --optimizer adamw \
    --lr_schedule cosine \
    --weight_decay 1e-4 \
    --no_augment \
    --num_workers 8 \
    --seed 42 \
    --optimize_metric loss \
    --threshold_objective f1 \
    --save_path "$OUT_DIR/cloudytile_${TAG}_best.pth" \
    --wandb_project cloudy-tile-final \
    --wandb_name "${TAG}_FINAL"
