#!/usr/bin/env python
"""
Extract single-timestep .nc files for training, matching existing labeled JPGs.

This script reads your existing labels CSV and extracts the corresponding
timesteps as .nc files with all spectral bands (RGB + NIR + SWIR1 + SWIR2).

Usage:
    python extract_nc_from_labels.py \
        --labels_csv /path/to/labels.csv \
        --input_dir /path/to/processed/nc \
        --output_dir /path/to/training/nc

    SLURM submission (if on HPC):
        sbatch extract_nc_from_labels.sh
"""
import argparse
import sys
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import xarray as xr
import numpy as np

# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloudytile.preprocessing import extract_single_timestep_nc, nc_to_rgb_array, save_frame_as_jpg


def parse_filename(filename: str) -> tuple[str, int]:
    """
    Parse lake_id and timestep from filename.

    Args:
        filename: e.g., 'CW2019_1579_t003.jpg'

    Returns:
        tuple: (lake_id, timestep) e.g., ('CW2019_1579', 3)
    """
    # Match pattern: {lake_id}_t{timestep}.jpg
    match = re.match(r"(.+)_t(\d+)\.jpg", filename)
    if match:
        lake_id = match.group(1)
        timestep = int(match.group(2))
        return lake_id, timestep
    raise ValueError(f"Could not parse filename: {filename}")


def extract_nc_for_labels(
    labels_csv: str,
    input_dir: str,
    output_dir: str,
    channels: list[str] = None,
    also_extract_jpgs: bool = False,
    jpg_output_dir: str = None,
    skip_existing: bool = True,
) -> tuple[list[Path], list[Path]]:
    """
    Extract single-timestep NC files for all labeled samples.

    Args:
        labels_csv: Path to labels CSV with 'filename' column
        input_dir: Directory containing processed lake .nc files (full timestacks)
        output_dir: Directory to save single-timestep .nc files
        channels: Channels to include (default: RGB + NIR + SWIR)
        also_extract_jpgs: Also extract JPG files (for verification)
        jpg_output_dir: Directory for JPGs (defaults to output_dir/../jpgs)
        skip_existing: Skip files that already exist

    Returns:
        tuple: (nc_paths, jpg_paths)
    """
    if channels is None:
        channels = ["red", "green", "blue", "nir", "swir1", "swir2"]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if also_extract_jpgs:
        if jpg_output_dir is None:
            jpg_output_dir = output_dir.parent / "jpgs"
        jpg_output_dir = Path(jpg_output_dir)
        jpg_output_dir.mkdir(parents=True, exist_ok=True)

    # Read labels and group by lake_id
    df = pd.read_csv(labels_csv)
    print(f"Found {len(df)} labeled samples in {labels_csv}")

    # Group filenames by lake_id
    lake_timesteps = defaultdict(list)
    for filename in df["filename"]:
        try:
            lake_id, timestep = parse_filename(filename)
            lake_timesteps[lake_id].append((timestep, filename))
        except ValueError as e:
            print(f"  Warning: {e}")

    print(f"Samples span {len(lake_timesteps)} unique lakes")

    nc_paths = []
    jpg_paths = []
    skipped = 0
    errors = 0

    for i, (lake_id, timestep_list) in enumerate(sorted(lake_timesteps.items())):
        # Find the source NC file
        nc_file = input_dir / f"{lake_id}.nc"
        if not nc_file.exists():
            print(f"  [{i+1}/{len(lake_timesteps)}] {lake_id}: source file not found, skipping {len(timestep_list)} samples")
            errors += len(timestep_list)
            continue

        print(f"  [{i+1}/{len(lake_timesteps)}] {lake_id}: extracting {len(timestep_list)} timesteps...")

        # Open the source dataset once
        try:
            ds = xr.open_dataset(nc_file)
        except Exception as e:
            print(f"    Error opening {nc_file}: {e}")
            errors += len(timestep_list)
            continue

        for timestep, orig_filename in timestep_list:
            # Output paths
            base_name = orig_filename.replace(".jpg", "")
            nc_out_path = output_dir / f"{base_name}.nc"

            # Check if already exists
            if skip_existing and nc_out_path.exists():
                nc_paths.append(nc_out_path)
                if also_extract_jpgs:
                    jpg_out_path = jpg_output_dir / orig_filename
                    jpg_paths.append(jpg_out_path)
                skipped += 1
                continue

            try:
                # Extract NC
                extract_single_timestep_nc(ds, timestep, nc_out_path, channels=channels)
                nc_paths.append(nc_out_path)

                # Extract JPG if requested
                if also_extract_jpgs:
                    jpg_out_path = jpg_output_dir / orig_filename
                    if not (skip_existing and jpg_out_path.exists()):
                        rgb = nc_to_rgb_array(ds, timestep)
                        save_frame_as_jpg(rgb, jpg_out_path)
                    jpg_paths.append(jpg_out_path)

            except Exception as e:
                print(f"    Error extracting timestep {timestep}: {e}")
                errors += 1

        ds.close()

    print(f"\nDone!")
    print(f"  Extracted: {len(nc_paths) - skipped} NC files")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {errors}")
    if also_extract_jpgs:
        print(f"  JPGs: {len(jpg_paths)}")

    return nc_paths, jpg_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract single-timestep NC files matching existing labels."
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="Path to labels CSV with 'filename' column (e.g., labels.csv)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing processed lake .nc files (full timestacks with spectral bands)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save single-timestep .nc files for training",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=["red", "green", "blue", "nir", "swir1", "swir2"],
        help="Channels to include in output NC files (default: red green blue nir swir1 swir2)",
    )
    parser.add_argument(
        "--also_extract_jpgs",
        action="store_true",
        help="Also extract JPG files (for verification/comparison)",
    )
    parser.add_argument(
        "--jpg_output_dir",
        type=str,
        default=None,
        help="Directory for JPG files (default: output_dir/../jpgs)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-extract files even if they already exist",
    )

    args = parser.parse_args()

    print(f"Labels CSV: {args.labels_csv}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Channels: {args.channels}")
    print(f"Skip existing: {not args.no_skip_existing}")
    print()

    nc_paths, jpg_paths = extract_nc_for_labels(
        labels_csv=args.labels_csv,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        channels=args.channels,
        also_extract_jpgs=args.also_extract_jpgs,
        jpg_output_dir=args.jpg_output_dir,
        skip_existing=not args.no_skip_existing,
    )

    print(f"\nTotal NC files: {len(nc_paths)}")


if __name__ == "__main__":
    main()
