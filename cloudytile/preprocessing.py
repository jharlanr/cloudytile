"""
Preprocessing utilities for generating training data from the NetCDF timestacks.
"""
import numpy as np
import xarray as xr
from pathlib import Path
from PIL import Image
import random
from typing import Union, Optional, List

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


def extract_single_timestep_nc(
    ds: xr.Dataset,
    timestep: int,
    output_path: Union[str, Path],
    channels: Optional[List[str]] = None,
) -> None:
    """
    Extract a single timestep from a NetCDF dataset and save as a new .nc file.

    Args:
        ds: xarray Dataset with 'imagery' variable [time, channel, y, x]
        timestep: Index of timestep to extract
        output_path: Path to save the single-timestep .nc file
        channels: List of channel names to include. If None, includes all except 'mask'.
            Default channels for training: ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get available channels
    available_channels = list(ds.coords["channel"].values)

    # Default: include all channels except mask for training
    if channels is None:
        channels = [c for c in available_channels if c != "mask"]

    # Filter to only available channels
    channels_to_use = [c for c in channels if c in available_channels]

    if not channels_to_use:
        raise ValueError(f"No requested channels found. Available: {available_channels}")

    # Extract single timestep and select channels
    imagery = ds["imagery"].isel(time=timestep).sel(channel=channels_to_use)

    # Create new dataset
    ds_single = xr.Dataset(
        data_vars={
            "imagery": (["channel", "y", "x"], imagery.values),
        },
        coords={
            "channel": channels_to_use,
        },
        attrs={
            "lake_id": ds.attrs.get("lake_id", "unknown"),
            "timestep": timestep,
            "source_time": str(ds.coords["time"].values[timestep]),
        },
    )

    ds_single.to_netcdf(output_path)


def extract_frames_with_nc(
    nc_path: Union[str, Path],
    jpg_output_dir: Union[str, Path],
    nc_output_dir: Union[str, Path],
    sample_fraction: float = 0.3,
    imagery_scale: float = 10000.0,
    quality: int = 95,
    seed: Optional[int] = None,
    skip_existing: bool = True,
    training_channels: Optional[List[str]] = None,
) -> tuple[List[Path], List[Path]]:
    """
    Extract frames from a NetCDF timestack, saving both JPGs (for labeling) and
    single-timestep .nc files (for training with extra spectral bands).

    Args:
        nc_path: Path to NetCDF file (combined datasets with imagery)
        jpg_output_dir: Directory to save JPG files (for labeling)
        nc_output_dir: Directory to save single-timestep .nc files (for training)
        sample_fraction: Fraction of timesteps to sample (0-1, default: 0.3)
        imagery_scale: Scale factor for normalization
        quality: JPG quality
        seed: Random seed for reproducibility
        skip_existing: If True, skip files that already exist
        training_channels: Channels to include in .nc files for training.
            Default: ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']

    Returns:
        Tuple of (jpg_paths, nc_paths)

    Output filename format:
        - JPG: {lake_id}_t{timestep:03d}.jpg
        - NC:  {lake_id}_t{timestep:03d}.nc
    """
    nc_path = Path(nc_path)
    jpg_output_dir = Path(jpg_output_dir)
    nc_output_dir = Path(nc_output_dir)
    jpg_output_dir.mkdir(parents=True, exist_ok=True)
    nc_output_dir.mkdir(parents=True, exist_ok=True)

    if training_channels is None:
        training_channels = ["red", "green", "blue", "nir", "swir1", "swir2"]

    # Load dataset
    ds = xr.open_dataset(nc_path)
    n_timesteps = ds.sizes["time"]
    lake_id = ds.attrs.get("lake_id", nc_path.stem)

    # Determine which timesteps to sample
    if seed is not None:
        random.seed(seed)

    n_samples = max(1, int(n_timesteps * sample_fraction))
    timesteps = sorted(random.sample(range(n_timesteps), n_samples))

    jpg_paths = []
    nc_paths = []

    for t in timesteps:
        base_filename = f"{lake_id}_t{t:03d}"
        jpg_path = jpg_output_dir / f"{base_filename}.jpg"
        nc_single_path = nc_output_dir / f"{base_filename}.nc"

        # Check if both already exist
        jpg_exists = jpg_path.exists()
        nc_exists = nc_single_path.exists()

        if skip_existing and jpg_exists and nc_exists:
            jpg_paths.append(jpg_path)
            nc_paths.append(nc_single_path)
            continue

        # Extract and save JPG (for labeling)
        if not (skip_existing and jpg_exists):
            rgb = nc_to_rgb_array(ds, t, imagery_scale)
            save_frame_as_jpg(rgb, jpg_path, quality)
        jpg_paths.append(jpg_path)

        # Extract and save single-timestep NC (for training)
        if not (skip_existing and nc_exists):
            extract_single_timestep_nc(ds, t, nc_single_path, channels=training_channels)
        nc_paths.append(nc_single_path)

    ds.close()

    return jpg_paths, nc_paths


def extract_frames_with_nc_from_directory(
    input_dir: Union[str, Path],
    jpg_output_dir: Union[str, Path],
    nc_output_dir: Union[str, Path],
    sample_fraction: float = 0.1,
    max_files: Optional[int] = None,
    imagery_scale: float = 10000.0,
    quality: int = 95,
    seed: int = 42,
    skip_existing: bool = True,
    training_channels: Optional[List[str]] = None,
) -> tuple[List[Path], List[Path]]:
    """
    Extract frames from multiple NetCDF files, saving both JPGs and single-timestep .nc files.

    Args:
        input_dir: Directory containing processed .nc files (with spectral bands)
        jpg_output_dir: Directory to save JPG files (for labeling)
        nc_output_dir: Directory to save single-timestep .nc files (for training)
        sample_fraction: Fraction of timesteps to sample from each file (default: 0.1)
        max_files: Maximum number of .nc files to process (None = all)
        imagery_scale: Scale factor for normalization
        quality: JPG quality
        seed: Random seed for reproducibility
        skip_existing: If True, skip files that already exist
        training_channels: Channels to include in .nc files.
            Default: ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']

    Returns:
        Tuple of (all_jpg_paths, all_nc_paths)

    Example:
        >>> jpg_paths, nc_paths = extract_frames_with_nc_from_directory(
        ...     input_dir="data/processed_lakes/",
        ...     jpg_output_dir="data/labeling_jpgs/",
        ...     nc_output_dir="data/training_nc/",
        ...     sample_fraction=0.15,
        ... )
        >>> print(f"Generated {len(jpg_paths)} JPGs for labeling")
        >>> print(f"Generated {len(nc_paths)} NC files for training")
    """
    input_dir = Path(input_dir)
    jpg_output_dir = Path(jpg_output_dir)
    nc_output_dir = Path(nc_output_dir)

    # Find all .nc files
    nc_files = sorted(input_dir.glob("*.nc"))

    if max_files is not None:
        if seed is not None:
            random.seed(seed)
        nc_files = random.sample(nc_files, min(max_files, len(nc_files)))
        nc_files = sorted(nc_files)

    print(f"Processing {len(nc_files)} NetCDF files...")
    print(f"  JPGs will be saved to: {jpg_output_dir}")
    print(f"  Training NCs will be saved to: {nc_output_dir}")

    all_jpg_paths = []
    all_nc_paths = []

    for i, nc_path in enumerate(nc_files):
        print(f"  [{i+1}/{len(nc_files)}] {nc_path.name}")

        jpg_paths, nc_paths = extract_frames_with_nc(
            nc_path=nc_path,
            jpg_output_dir=jpg_output_dir,
            nc_output_dir=nc_output_dir,
            sample_fraction=sample_fraction,
            imagery_scale=imagery_scale,
            quality=quality,
            seed=seed + i if seed is not None else None,
            skip_existing=skip_existing,
            training_channels=training_channels,
        )
        all_jpg_paths.extend(jpg_paths)
        all_nc_paths.extend(nc_paths)

    print(f"Done!")
    print(f"  Generated {len(all_jpg_paths)} JPG files for labeling")
    print(f"  Generated {len(all_nc_paths)} NC files for training")

    return all_jpg_paths, all_nc_paths