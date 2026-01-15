#!/usr/bin/env python
"""
Extract both .jpg and .nc files from NetCDF timestacks for labeling and training.

This script generates:
- JPG files for labeling (upload to Labelbox)
- NC files for training (with spectral bands: RGB + NIR + SWIR1 + SWIR2)

Both files share the same filename base (e.g., CW2019_1579_t003.jpg and CW2019_1579_t003.nc)
so labels created from JPGs can be directly used with NC files for training.

Usage:
    python extract_jpgs_and_nc.py \
        --input_dir /path/to/processed/nc \
        --jpg_output_dir /path/to/jpgs \
        --nc_output_dir /path/to/training_nc \
        --sample_fraction 0.15 \
        --max_files 20

    SLURM submission (if on HPC):
        sbatch extract_jpgs_and_nc.sh
"""
import argparse
import sys
from pathlib import Path

# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloudytile.preprocessing import extract_frames_with_nc_from_directory


def main():
    parser = argparse.ArgumentParser(
        description="Extract JPG and NC files from NetCDF timestacks."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing processed .nc timestack files (with spectral bands)",
    )
    parser.add_argument(
        "--jpg_output_dir",
        type=str,
        required=True,
        help="Directory to save JPG files (for labeling)",
    )
    parser.add_argument(
        "--nc_output_dir",
        type=str,
        required=True,
        help="Directory to save single-timestep NC files (for training)",
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
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=["red", "green", "blue", "nir", "swir1", "swir2"],
        help="Channels to include in NC files (default: red green blue nir swir1 swir2)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-extract files even if they already exist",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Extracting JPG + NC files for labeling and training")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"JPG output dir:   {args.jpg_output_dir}")
    print(f"NC output dir:    {args.nc_output_dir}")
    print(f"Sample fraction:  {args.sample_fraction}")
    print(f"Max files:        {args.max_files or 'all'}")
    print(f"Seed:             {args.seed}")
    print(f"NC channels:      {args.channels}")
    print(f"Skip existing:    {not args.no_skip_existing}")
    print()

    jpg_paths, nc_paths = extract_frames_with_nc_from_directory(
        input_dir=args.input_dir,
        jpg_output_dir=args.jpg_output_dir,
        nc_output_dir=args.nc_output_dir,
        sample_fraction=args.sample_fraction,
        max_files=args.max_files,
        seed=args.seed,
        quality=args.quality,
        skip_existing=not args.no_skip_existing,
        training_channels=args.channels,
    )

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total JPG files: {len(jpg_paths)}")
    print(f"Total NC files:  {len(nc_paths)}")
    print()
    print("Next steps:")
    print(f"  1. Upload JPGs from {args.jpg_output_dir} to Labelbox for labeling")
    print(f"  2. Export labels as labels.csv")
    print(f"  3. Train with: python run_training.py --labels_csv labels.csv \\")
    print(f"       --image_dir {args.jpg_output_dir} --use_nc --nc_dir {args.nc_output_dir}")


if __name__ == "__main__":
    main()
