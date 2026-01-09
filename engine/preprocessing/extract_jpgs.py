#!/usr/bin/env python
"""
Extract .jpg frames from NetCDF timestacks.

Usage:
    python extract_jpgs.py \
        --input_dir /path/to/nc/files \
        --output_dir /path/to/jpgs \
        --sample_fraction 0.3 \
        --max_files 20 \
        --seed 42

    SLURM submission (if on HPC):
        sbatch extract_jpgs.sh
"""
import argparse
import sys
from pathlib import Path

# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cloudytile.preprocessing import extract_frames_from_directory


def main():
    parser = argparse.ArgumentParser(
        description="Extract JPG frames from NetCDF timestacks."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .nc timestack files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save JPG files",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=0.15,
        help="Fraction of timesteps to sample from each file (default: 0.15)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of .nc files to process (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPG quality 1-100 (default: 95)",
    )

    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample fraction: {args.sample_fraction}")
    print(f"Max files: {args.max_files or 'all'}")
    print(f"Seed: {args.seed}")
    print()

    paths = extract_frames_from_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_fraction=args.sample_fraction,
        max_files=args.max_files,
        seed=args.seed,
        quality=args.quality,
    )

    print(f"\nTotal JPGs created: {len(paths)}")


if __name__ == "__main__":
    main()