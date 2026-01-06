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
    img_size: tuple[int,int] = (512,512),
    threshold: float = 0.5,
    rgb_indices: tuple[int, int, int] = (0,1,2),
    batch_size: int = 32,
) -> np.ndarray:
    """
    Predict usefulness for all timesteps in an xarrya Dataset.

    Args:
        model: Loaded CloudyTileCNN model
        ds: xarray Dataset with 'imagery' variable of shape [time, channel, y, x]
        img_size: Size to resize tiles to for inference (e.g., 512x512)
        threshold: classification threshold
        rgb_indices: indices of R, G, B channels in the channel dimension
        batch_size: number of frames to process in one batch

    Returns:
        Array of shape [time] with binary predictions (0=not useful, 1=useful)
    """
    device = next(model.parameters()).device