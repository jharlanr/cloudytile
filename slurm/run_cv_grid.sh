#!/bin/bash
#SBATCH --job-name=cv_grid
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
#SBATCH --array=0-31%8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# Lake-grouped CV grid: 32 configs (bands x arch x lr x optimizer) x 3 folds
#
# One config per array task. Resumable: rerun the same sbatch and finished
# (config, fold) JSONs are skipped. Aggregate afterwards from a login node:
#   python3 engine/run_cv_grid.py --out_dir <OUT_DIR> --summarize
# Then rerun the winning config at --folds 5 for the reported number.
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

REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc_10k"
BAND_STATS="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats_10k.json"
OUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/cv_results"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
mkdir -p "$OUT_DIR"

python3 "$REPO_DIR/engine/run_cv_grid.py" \
    --labels_csv "$REPO_DIR/labels/labels.csv" \
    --nc_dir "$NC_DIR" \
    --band_stats "$BAND_STATS" \
    --out_dir "$OUT_DIR" \
    --config_index "$SLURM_ARRAY_TASK_ID" \
    --folds 3 \
    --epochs 40 \
    --img_size 256 \
    --num_workers 8
