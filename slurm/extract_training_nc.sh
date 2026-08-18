#!/bin/bash
#SBATCH --job-name=training_nc
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# Extract per-tile 6-band training .nc for every labeled frame
#
# labels/labels.csv -> one {lake_id}_tNNN.nc per row, all six SDR bands.
# ~10,000 tiles x ~6.3 MB float32 = ~63 GB. Resumable (skips existing tiles).
# Run BEFORE compute_band_stats.sh and training.
# =============================================================================

ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

pip install --user xarray netcdf4
export PYTHONUNBUFFERED=1

REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
SDR_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/essd_sdr/data"
OUTPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc_10k"

mkdir -p "$OUTPUT_DIR"

python3 "$REPO_DIR/engine/extract_training_nc.py" \
    --labels_csv "$REPO_DIR/labels/labels.csv" \
    --nc_dirs "$SDR_DIR/CW2019" "$SDR_DIR/CW2018" \
    --output_dir "$OUTPUT_DIR"

echo "Done. Next: sbatch slurm/compute_band_stats.sh"
