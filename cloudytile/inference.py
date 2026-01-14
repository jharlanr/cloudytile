"""
Inference utilities for CloudytileCNN model.
"""
import numpy as np
import torch
import xarray as xr
from pathlib import Path

from .model import CloudyTileCNN

def load_model(
    weights_path: str | Path,
    img_size: tuple[int, int] = (512, 512),
    channels: list[int] = None,
    fc_layers: list[int] = None,
    device: str = None,
) -> CloudyTileCNN:
    """
    Load a trained CloudyTileCNN model from weights.

    Args:
        weights_path: Path to .pth weights file
        img_size: Input image size model was trained with
        channels: Convolutional channel sizes (must match training)
        fc_layers: fully-connected layer sizes (must match training)
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
    rgb_indices: tuple[int, int, int] = (0, 1, 2),
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
        rgb_indices: Indices of R, G, B channels in the channel dimension
        imagery_scale: Scale factor for normalization (default: 10000.0 for Sentinel-2)
        batch_size: Number of frames to process at once

    Returns:
        Array of shape [time] with binary predictions (0=not useful, 1=useful)
    """
    device = next(model.parameters()).device

    # Extract imagery: [time, channel, y, x]
    imagery = ds["imagery"].values

    # Select RGB channels: [time, 3, y, x]
    rgb = imagery[:, rgb_indices, :, :]
    n_frames = rgb.shape[0]

    # Normalize to [0, 1] (matches lake-vision datasets.py)
    rgb = np.clip(rgb / imagery_scale, 0.0, 1.0)

    predictions = []

    # process in batches
    for i in range(0, n_frames, batch_size):
        batch = rgb[i : i+batch_size]
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
    imagery_scale: float = 10000.0,
    batch_size: int = 32,
    output_path: str | Path = None,
) -> xr.Dataset:
    """
    Load a NetCDF file, run classifier on each timestep, add/overwrite cloudy_seq variable.

    Args:
        nc_path: Path to input NetCDF file
        model: Loaded CloudyTileCNN model (use load_model to create)
        img_size: Size to resize tiles to for inference
        threshold: Classification threshold (default: 0.5)
        imagery_scale: Scale factor for normalization (default: 10000.0)
        batch_size: Number of frames to process at once
        output_path: Path to save output (default: overwrite input)

    Returns:
        xarray Dataset with cloudy_seq variable added
    """
    nc_path = Path(nc_path)
    if output_path is None:
        output_path = nc_path

    # Load dataset and get predictions
    ds = xr.open_dataset(nc_path)

    cloudy_seq = predict_from_nc(
        model, ds, img_size=img_size, threshold=threshold,
        imagery_scale=imagery_scale, batch_size=batch_size
    )

    # Drop existing cloudy_seq if it exists
    if 'cloudy_seq' in ds:
        ds = ds.drop_vars('cloudy_seq')

    # Add new cloudy_seq variable
    ds['cloudy_seq'] = xr.DataArray(
        cloudy_seq.flatten(),
        dims=['time'],
        attrs={
            'long_name': 'tile usefulness classification',
            'description': '1 = useful, 0 = not useful (cloudy/nodata)',
            'threshold': threshold,
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
    threshold: float = 0.5,
    imagery_scale: float = 10000.0,
    batch_size: int = 32,
    pattern: str = "*.nc",
) -> int:
    """
    Add cloudy_seq to all NetCDF files in a directory.

    Args:
        nc_dir: Directory containing NetCDF files
        model_path: Path to trained model weights
        img_size: Image size model was trained with
        channels: Conv layer channels (must match training)
        fc_layers: FC layer sizes (must match training)
        threshold: Classification threshold
        imagery_scale: Scale factor for normalization
        batch_size: Batch size for inference
        pattern: Glob pattern for finding files (default: "*.nc")

    Returns:
        Number of files processed
    """
    nc_dir = Path(nc_dir)
    nc_files = sorted(nc_dir.glob(pattern))

    if not nc_files:
        print(f"No files matching '{pattern}' found in {nc_dir}")
        return 0

    print(f"Found {len(nc_files)} files to process")

    # Load model once
    model = load_model(
        model_path,
        img_size=img_size,
        channels=channels,
        fc_layers=fc_layers,
    )
    print(f"Loaded model from {model_path}")

    processed = 0
    for nc_path in nc_files:
        try:
            add_cloudy_seq_to_nc(
                nc_path,
                model,
                img_size=img_size,
                threshold=threshold,
                imagery_scale=imagery_scale,
                batch_size=batch_size,
            )
            processed += 1
            print(f"  [{processed}/{len(nc_files)}] {nc_path.name}")
        except Exception as e:
            print(f"  ERROR processing {nc_path.name}: {e}")

    print(f"\nProcessed {processed}/{len(nc_files)} files")
    return processed