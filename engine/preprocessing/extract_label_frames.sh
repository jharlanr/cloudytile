#!/bin/bash
#SBATCH --job-name=label_frames
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --array=0-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# SAMPLE LABELING FRAMES: 500 lakes x 10 frames, per year
# =============================================================================
#
# Array task 0 -> 2019, task 1 -> 2018. 5,000 frames each, ~0.5 GB total.
#
# USAGE:
#   sbatch extract_label_frames.sh
#
# Sanity check the counts first without reading imagery:
#   python3 extract_label_frames.py --nc_dir <dir> --output_dir <dir> --dry_run
#
# Imagery comes from the ESSD SDR deposit (679 CW2018 + 1000 CW2019 per-lake
# files, one schema for both years). Its per-timestep pct_nans lets the script
# skip empty frames without reading pixels, and boa_add_offset is applied when
# converting digital numbers to reflectance.
#
# Only the imagery is shared with the ESSD deposit; cloudy-tile keeps its own
# splits. Group them by lake_id, never by tile.
#
# AFTER THIS COMPLETES, copy the frames down and label locally — the jpgs are
# small but the OAK mount reads at well under 1 MB/s, which the GUI will feel:
#   rsync -a <sherlock>:$OUT_BASE/label_frames_2019 ~/data/
# =============================================================================

ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312

pip install --user xarray netcdf4

REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
SDR_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/essd_sdr/data"
OUT_BASE="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile"

YEARS=(2019 2018)
NC_DIRS=("$SDR_DIR/CW2019" "$SDR_DIR/CW2018")

YEAR="${YEARS[$SLURM_ARRAY_TASK_ID]}"
NC_DIR="${NC_DIRS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$OUT_BASE/label_frames_${YEAR}"

N_LAKES=500
FRAMES_PER_LAKE=10
MAX_NAN_FRAC=0.5
# Sentinel-2 L2A true-color convention; see the .py docstring on bright-scene saturation
IMAGERY_SCALE=10000

mkdir -p "$OUTPUT_DIR"
mkdir -p /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs

echo "=============================================="
echo "Extracting labeling frames for $YEAR"
echo "  NC dir:     $NC_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Sampling:   $N_LAKES lakes x $FRAMES_PER_LAKE frames"
echo "  NaN cutoff: $MAX_NAN_FRAC"
echo "  Start:      $(date)"
echo "=============================================="

python3 "$REPO_DIR/engine/preprocessing/extract_label_frames.py" \
    --nc_dir "$NC_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_lakes $N_LAKES \
    --frames_per_lake $FRAMES_PER_LAKE \
    --max_nan_frac $MAX_NAN_FRAC \
    --imagery_scale $IMAGERY_SCALE \
    --seed 42

EXIT_CODE=$?
echo "End: $(date) | Exit code: $EXIT_CODE"
exit $EXIT_CODE
