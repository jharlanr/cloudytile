"""
Preprocessing utilities for generating training data from the NetCDF timestacks.
"""
import numpy as np
import xarray as xr
from pathlib import Path
from PIL import Image
import random
from typing import Union

def nc_to_rgb_array(
    ds: xr.Dataset,
    timestep: int,
    imagery_scale: float = 10000.0,
) -> np.ndarray:
    """
    Extract a single RGB frame from a NetCDF dataset.

    Args:
        ds: xarray Dataset with 'imagery' variable [time, channel, y, x]
            where channel coordinate includes 'red', 'green', 'blue'
        timestep: Index of timestep to extract
        imagery_scale: Scale factor for normalization (default: 10000.0)

    Returns:
        RGB array of shape [H, W, 3] with values in [0, 255] as uint8
    """
    # select RGB channels by name (i.e., 'red', 'green', 'blue')
    imagery = ds["imagery"].isel(time=timestep)
    red = imagery.sel(channel="red").values
    green = imagery.sel(channel="green").values
    blue = imagery.sel(channel="blue").values

    # stack to [H, W, 3]
    rgb = np.stack([red, green, blue], axis=-1)

    # normalize to [0, 1] range and clip (consistent with lake-vision normalization based on Sentinel-2 pixel value range)
    rgb = np.clip(rgb / imagery_scale, 0.0, 1.0)

    # scale to [0, 255] for .jpg (replace NaNs with 0/black)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb

def save_frame_as_jpg(
    rgb_array: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95,
) -> None:
    """
    Save an RGB array as a .jpg file.

    Args:
        rgb_array: RGB array of shape [H, W, 3] as uint8 type
        output_path: path to save JPG file
        quality: JPG quality (1-100, default: 95)
    """
    img = Image.fromarray(rgb_array, mode="RGB")
    img.save(output_path, "JPEG", quality=quality)

def extract_frames_from_nc(
    nc_path: Union[str, Path],
    output_dir: Union[str, Path],
    sample_fraction: float = 0.3,
    imagery_scale: float = 10000.0,
    quality: int = 95,
    seed: int = None,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Extract frames from a single NetCDF timestack and save as JPGs.

    Args:
        nc_path: Path to NetCDF file (combined datasets with imagery)
        output_dir: Directory to save JPG files
        sample_fraction: fraction of timesteps to sample (0-1, default: 0.3)
        imagery_scale: Scale factor for normalization
        quality: JPG quality
        seed: random seed for reproducibility
        skip_existing: if True, skip files that already exist (default: True)

    Returns:
        List of paths to saved JPG files

    Output filename format: {lake_id}_t{timestep:03d}.jpg
    """
    nc_path = Path(nc_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    ds = xr.open_dataset(nc_path)
    n_timesteps = ds.sizes["time"]
    lake_id = ds.attrs.get("lake_id", nc_path.stem)

    # determine which timesteps to sample
    if seed is not None:
        random.seed(seed)

    n_samples = max(1, int(n_timesteps * sample_fraction))
    timesteps = sorted(random.sample(range(n_timesteps), n_samples))

    saved_paths = []

    for t in timesteps:
        filename = f"{lake_id}_t{t:03d}.jpg"
        output_path = output_dir / filename

        # skip if file already exists
        if skip_existing and output_path.exists():
            saved_paths.append(output_path)
            continue

        # extract RGB frame
        rgb = nc_to_rgb_array(ds, t, imagery_scale)

        # save as JPG
        save_frame_as_jpg(rgb, output_path, quality)
        saved_paths.append(output_path)

    ds.close()

    return saved_paths

def extract_frames_from_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    sample_fraction: float = 0.1,
    max_files: int = None,
    imagery_scale: float = 10000.0,
    quality: int = 95,
    seed: int = 42,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Extract frames from multiple NetCDF files in a directory.

    Args:
        input_dir: Directory containing .nc files
        output_dir: Directory to save JPG files
        sample_fraction: Fraction of timesteps to sample from each file (default: 0.1 = 10%)
        max_files: Maximum number of .nc files to process (None = all)
        imagery_scale: Scale factor for normalization
        quality: JPG quality
        seed: Random seed for reproducibility
        skip_existing: if True, skip files that already exist (default: True)

    Returns:
        List of paths to all saved JPG files

    Example:
        >>> paths = extract_frames_from_directory(
        ...     input_dir="data/processed_lakes/",
        ...     output_dir="data/training_jpgs/",
        ...     sample_fraction=0.15,
        ...     seed=42,
        ... )
        >>> print(f"Generated {len(paths)} training images as .jpg files")
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # find all .nc files in input_dir
    nc_files = sorted(input_dir.glob("*.nc"))

    if max_files is not None:
        if seed is not None:
            random.seed(seed)
        nc_files = random.sample(nc_files, min(max_files, len(nc_files)))
        nc_files = sorted(nc_files)

    print(f"Processing {len(nc_files)} NetCDF files...")

    all_saved_paths = []

    for i, nc_path in enumerate(nc_files):
        print(f"  [{i+1}/{len(nc_files)}] {nc_path.name}")

        paths = extract_frames_from_nc(
            nc_path=nc_path,
            output_dir=output_dir,
            sample_fraction=sample_fraction,
            imagery_scale=imagery_scale,
            quality=quality,
            seed=seed + i if seed is not None else None,
            skip_existing=skip_existing,
        )
        all_saved_paths.extend(paths)

    print(f"Done. Generated {len(all_saved_paths)} JPG files and saved in {output_dir}")

    return all_saved_paths