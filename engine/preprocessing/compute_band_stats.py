#!/usr/bin/env python
"""
Compute per-band mean and std statistics from training NC files.

This script samples NC files and computes statistics for normalization.
Output is saved as a JSON file that can be loaded by the dataloader.

Usage:
    python compute_band_stats.py \
        --nc_dir /path/to/training_nc \
        --output_path band_stats.json \
        --sample_size 1000
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import xarray as xr


def compute_band_stats(
    nc_dir: str,
    output_path: str = None,
    sample_size: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute per-band mean and std from NC files.

    Args:
        nc_dir: Directory containing single-timestep .nc files
        output_path: Path to save JSON stats file (optional)
        sample_size: Number of files to sample (default: 1000)
        seed: Random seed for sampling

    Returns:
        dict: Band statistics with format:
            {"red": {"mean": X, "std": Y}, ...}
    """
    nc_dir = Path(nc_dir)
    nc_files = list(nc_dir.glob("*.nc"))

    if len(nc_files) == 0:
        raise ValueError(f"No NC files found in {nc_dir}")

    # Sample files if we have more than sample_size
    random.seed(seed)
    if len(nc_files) > sample_size:
        nc_files = random.sample(nc_files, sample_size)

    print(f"Computing band statistics from {len(nc_files)} files...")

    # Get channel names from first file
    with xr.open_dataset(nc_files[0]) as ds:
        channels = list(ds.coords["channel"].values)

    print(f"Channels: {channels}")

    # Accumulate statistics using Welford's online algorithm
    # This avoids loading all data into memory
    n_pixels = {ch: 0 for ch in channels}
    mean = {ch: 0.0 for ch in channels}
    M2 = {ch: 0.0 for ch in channels}  # Sum of squared differences

    for i, nc_path in enumerate(nc_files):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(nc_files)}...")

        try:
            with xr.open_dataset(nc_path) as ds:
                for ch in channels:
                    data = ds["imagery"].sel(channel=ch).values.flatten()
                    # Remove NaNs
                    data = data[~np.isnan(data)]

                    if len(data) == 0:
                        continue

                    # Welford's online algorithm for mean and variance
                    for x in data:
                        n_pixels[ch] += 1
                        delta = x - mean[ch]
                        mean[ch] += delta / n_pixels[ch]
                        delta2 = x - mean[ch]
                        M2[ch] += delta * delta2

        except Exception as e:
            print(f"  Warning: Error processing {nc_path.name}: {e}")
            continue

    # Compute final statistics
    stats = {}
    for ch in channels:
        if n_pixels[ch] > 1:
            variance = M2[ch] / (n_pixels[ch] - 1)
            std = np.sqrt(variance)
            stats[ch] = {
                "mean": float(mean[ch]),
                "std": float(std),
                "n_pixels": int(n_pixels[ch]),
            }
            print(f"  {ch:12s}: mean={mean[ch]:10.2f}, std={std:10.2f}")
        else:
            print(f"  {ch:12s}: insufficient data")

    # Save to JSON if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nSaved band statistics to {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-band normalization statistics from NC files."
    )
    parser.add_argument(
        "--nc_dir",
        type=str,
        required=True,
        help="Directory containing training NC files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="band_stats.json",
        help="Output path for JSON stats file (default: band_stats.json)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of files to sample (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    compute_band_stats(
        nc_dir=args.nc_dir,
        output_path=args.output_path,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
