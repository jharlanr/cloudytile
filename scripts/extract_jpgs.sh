#!/bin/bash
#SBATCH --job-name=mk_jpgs
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules and activate environment
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml viz
ml py-matplotlib/3.8.3_py312
ml py-scipy/1.12.0_py312

# Set paths
INPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks"
OUTPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/jpg_tiles"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile"
SAMPLE_FRACTION=0.15
MAX_FILES=20

# Create output and logs directories if needed
mkdir -p $OUTPUT_DIR
mkdir -p $REPO_DIR/logs

# Run extraction
python3 $REPO_DIR/scripts/extract_jpgs.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --sample_fraction $SAMPLE_FRACTION \
    --max_files $MAX_FILES \
    --seed 42


echo "Done! JPGs saved to $OUTPUT_DIR"