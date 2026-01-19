"""
Inference utilities for CloudytileCNN model.
"""
import json
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from typing import Optional, Union

from .model import CloudyTileCNN


# Available spectral bands in NC files
AVAILABLE_BANDS = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22']


def load_band_stats(stats_path: Union[str, Path]) -> dict:
    """Load band statistics from JSON file."""
    with open(stats_path, "r") as f:
        return json.load(f)

def load_model(
    weights_path: str | Path,
    img_size: tuple[int, int] = (512, 512),
    channels: list[int] = None,
    fc_layers: list[int] = None,
    in_channels: int = 3,
    device: str = None,
) -> CloudyTileCNN:
    """
    Load a trained CloudyTileCNN model from weights.

    Args:
        weights_path: Path to .pth weights file
        img_size: Input image size model was trained with
        channels: Convolutional channel sizes (must match training)
        fc_layers: fully-connected layer sizes (must match training)
        in_channels: Number of input channels (must match training, default: 3 for RGB)
        device: device to load model on ('cuda', 'cpu' or None for auto)

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CloudyTileCNN(
        img_size=img_size,
        channels=channels,
        fc_layers=fc_layers,
        in_channels=in_channels,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def predict_from_nc(
    model: CloudyTileCNN,
    ds: xr.Dataset,
    img_size: tuple[int, int] = (512, 512),
    threshold: float = 0.5,
    nc_channels: list[str] = None,
    band_stats: Optional[dict] = None,
    imagery_scale: float = 10000.0,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Predict usefulness for all timesteps in an xarray Dataset.

    Args:
        model: Loaded CloudyTileCNN model
        ds: xarray Dataset with 'imagery' variable of shape [time, channel, y, x]
        img_size: Size to resize tiles to for inference
        threshold: Classification threshold
        nc_channels: List of channel names to use (e.g., ['red', 'green', 'blue'] or
            ['blue', 'nir', 'swir16']). Default: ['red', 'green', 'blue']
        band_stats: Dict with band statistics for normalization. If None, uses simple
            /imagery_scale normalization.
        imagery_scale: Scale factor for legacy normalization (default: 10000.0)
        batch_size: Number of frames to process at once

    Returns:
        Array of shape [time] with binary predictions (0=not useful, 1=useful)
    """
    device = next(model.parameters()).device

    # Default to RGB
    if nc_channels is None:
        nc_channels = ['red', 'green', 'blue']

    # Get channel names from NC file
    nc_channel_names = list(ds.coords['channel'].values)

    # Find indices for requested channels
    channel_indices = []
    for ch in nc_channels:
        if ch in nc_channel_names:
            channel_indices.append(nc_channel_names.index(ch))
        else:
            raise ValueError(f"Channel '{ch}' not found in NC file. Available: {nc_channel_names}")

    # Extract imagery: [time, channel, y, x]
    imagery = ds["imagery"].values

    # Select requested channels: [time, n_channels, y, x]
    selected = imagery[:, channel_indices, :, :].copy()
    n_frames = selected.shape[0]

    # Handle NaNs
    selected = np.nan_to_num(selected, nan=0.0)

    # Normalize per-band
    if band_stats is not None:
        for i, ch in enumerate(nc_channels):
            if ch in band_stats:
                mean = band_stats[ch]['mean']
                std = band_stats[ch]['std']
                selected[:, i, :, :] = (selected[:, i, :, :] - mean) / std
            else:
                # Fallback for unknown channels
                selected[:, i, :, :] = selected[:, i, :, :] / imagery_scale
    else:
        # Legacy normalization
        selected = np.clip(selected / imagery_scale, 0.0, 1.0)

    predictions = []

    # process in batches
    for i in range(0, n_frames, batch_size):
        batch = selected[i : i+batch_size]
        batch_tensor = torch.from_numpy(batch).float()

        # resize if needed
        if batch_tensor.shape[-2:] != img_size:
            batch_tensor = torch.nn.functional.interpolate(
                batch_tensor, size=img_size, mode="bilinear", align_corners=False
            )

        batch_tensor = batch_tensor.to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int().cpu().numpy()

        predictions.append(preds)

    return np.concatenate(predictions)

def add_cloudy_seq_to_nc(
    nc_path: str | Path,
    model: CloudyTileCNN,
    img_size: tuple[int, int] = (512, 512),
    threshold: float = 0.5,
    nc_channels: list[str] = None,
    band_stats: Optional[dict] = None,
    imagery_scale: float = 10000.0,
    batch_size: int = 32,
    output_path: str | Path = None,
    var_name: str = 'cloudy_seq',
) -> xr.Dataset:
    """
    Load a NetCDF file, run classifier on each timestep, add/overwrite cloudy_seq variable.

    Args:
        nc_path: Path to input NetCDF file
        model: Loaded CloudyTileCNN model (use load_model to create)
        img_size: Size to resize tiles to for inference
        threshold: Classification threshold (default: 0.5)
        nc_channels: List of channel names to use (e.g., ['red', 'green', 'blue'])
        band_stats: Dict with band statistics for normalization
        imagery_scale: Scale factor for normalization (default: 10000.0)
        batch_size: Number of frames to process at once
        output_path: Path to save output (default: overwrite input)
        var_name: Name for the output variable (default: 'cloudy_seq')

    Returns:
        xarray Dataset with cloudy_seq variable added
    """
    nc_path = Path(nc_path)
    if output_path is None:
        output_path = nc_path
    else:
        output_path = Path(output_path)

    # If output file already exists, load from there to preserve existing variables
    # (allows multiple models to add their cloudy_seq_* variables sequentially)
    if output_path.exists() and output_path != nc_path:
        ds = xr.open_dataset(output_path)
    else:
        ds = xr.open_dataset(nc_path)

    # Get predictions (always from input file for imagery)
    input_ds = xr.open_dataset(nc_path) if output_path.exists() and output_path != nc_path else ds
    cloudy_seq = predict_from_nc(
        model, input_ds, img_size=img_size, threshold=threshold,
        nc_channels=nc_channels, band_stats=band_stats,
        imagery_scale=imagery_scale, batch_size=batch_size
    )
    if input_ds is not ds:
        input_ds.close()

    # Drop existing variable if it exists
    if var_name in ds:
        ds = ds.drop_vars(var_name)

    # Build description of channels used
    channels_str = ','.join(nc_channels) if nc_channels else 'red,green,blue'

    # Add new variable
    ds[var_name] = xr.DataArray(
        cloudy_seq.flatten(),
        dims=['time'],
        attrs={
            'long_name': 'tile usefulness classification',
            'description': '1 = useful, 0 = not useful (cloudy/nodata)',
            'threshold': threshold,
            'channels': channels_str,
        }
    )

    # Save to temp file then rename (atomic write)
    temp_path = output_path.parent / f".{output_path.name}.tmp"
    ds.to_netcdf(temp_path)
    ds.close()
    temp_path.rename(output_path)

    return xr.open_dataset(output_path)


def process_directory(
    nc_dir: str | Path,
    model_path: str | Path,
    img_size: tuple[int, int] = (512, 512),
    channels: list[int] = None,
    fc_layers: list[int] = None,
    nc_channels: list[str] = None,
    band_stats_path: Optional[str | Path] = None,
    threshold: float = 0.5,
    imagery_scale: float = 10000.0,
    batch_size: int = 32,
    pattern: str = "*.nc",
    var_name: str = 'cloudy_seq',
    output_dir: Optional[str | Path] = None,
) -> int:
    """
    Add cloudy_seq to all NetCDF files in a directory.

    Args:
        nc_dir: Directory containing NetCDF files
        model_path: Path to trained model weights
        img_size: Image size model was trained with
        channels: Conv layer channels (must match training)
        fc_layers: FC layer sizes (must match training)
        nc_channels: List of channel names to use (e.g., ['red', 'green', 'blue'])
        band_stats_path: Path to band statistics JSON file
        threshold: Classification threshold
        imagery_scale: Scale factor for normalization
        batch_size: Batch size for inference
        pattern: Glob pattern for finding files (default: "*.nc")
        var_name: Name for the output variable (default: 'cloudy_seq')
        output_dir: Directory to save output files (default: overwrite input files in-place)

    Returns:
        Number of files processed
    """
    nc_dir = Path(nc_dir)
    nc_files = sorted(nc_dir.glob(pattern))

    # Set up output directory if specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    if not nc_files:
        print(f"No files matching '{pattern}' found in {nc_dir}")
        return 0

    print(f"Found {len(nc_files)} files to process")

    # Determine number of input channels
    in_channels = len(nc_channels) if nc_channels else 3

    # Load model once
    model = load_model(
        model_path,
        img_size=img_size,
        channels=channels,
        fc_layers=fc_layers,
        in_channels=in_channels,
    )
    print(f"Loaded model from {model_path}")
    print(f"Using channels: {nc_channels or ['red', 'green', 'blue']}")

    # Load band stats if provided
    band_stats = None
    if band_stats_path:
        band_stats = load_band_stats(band_stats_path)
        print(f"Loaded band statistics from {band_stats_path}")

    processed = 0
    for nc_path in nc_files:
        # Determine output path
        if output_dir is not None:
            out_path = output_dir / nc_path.name
        else:
            out_path = None  # Will overwrite input file

        try:
            add_cloudy_seq_to_nc(
                nc_path,
                model,
                img_size=img_size,
                threshold=threshold,
                nc_channels=nc_channels,
                band_stats=band_stats,
                imagery_scale=imagery_scale,
                batch_size=batch_size,
                output_path=out_path,
                var_name=var_name,
            )
            processed += 1
            print(f"  [{processed}/{len(nc_files)}] {nc_path.name}")
        except Exception as e:
            print(f"  ERROR processing {nc_path.name}: {e}")

    print(f"\nProcessed {processed}/{len(nc_files)} files")
    return processed