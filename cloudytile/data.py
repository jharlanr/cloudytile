"""
PyTorch Dataset for CloudyTile training.
"""
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Union, Optional, List

class CloudyTileDataset(Dataset):
    """
    Dataset for loading satellite tile images with binary labels.

    Args:
        labels_csv: Path to CSV with columns 'filename' and 'label'
        image_dir: Directory containing the image files
        transform: Optional torchvision transforms
        img_size: Target image size (height, width). Default: (512, 512)

    Labels:
        0 = not useful (cloudy/no data)
        1 = useful (clear)
    """
    def __init__(
        self,
        labels_csv: Union[str, Path],
        image_dir: Union[str, Path],
        transform: transforms.Compose = None,
        img_size: tuple[int, int] = (512, 512),
    ):
        self.labels_df = pd.read_csv(labels_csv)
        self.image_dir = Path(image_dir)
        self.img_size = img_size

        # Default transform: resize and normalize to [0, 1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),  # Converts to [0, 1] and [C, H, W]
            ])
        else:
            self.transform = transform

        # Filter to only include images that exist
        self._filter_existing_images()

    def _filter_existing_images(self):
        """Remove entries for images that don't exist on disk."""
        existing_mask = self.labels_df["filename"].apply(
            lambda f: (self.image_dir / f).exists()
        )
        n_missing = (~existing_mask).sum()
        if n_missing > 0:
            print(f"Warning: {n_missing} images not found, skipping them")
        self.labels_df = self.labels_df[existing_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.labels_df.iloc[idx]
        img_path = self.image_dir / row["filename"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])

        return image, label

    @property
    def filenames(self) -> list[str]:
        """Return list of all filenames in the dataset."""
        return self.labels_df["filename"].tolist()


def create_splits(
    labels_csv: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Union[str, Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split labels CSV into stratified train/val/test sets.

    Args:
        labels_csv: Path to the full labels CSV
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.1)
        test_ratio: Fraction for testing (default: 0.1)
        seed: Random seed for reproducibility
        output_dir: If provided, save split CSVs to this directory

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    df = pd.read_csv(labels_csv)

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    # Second split: val vs test (from the temp set)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        stratify=temp_df["label"],
        random_state=seed,
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    n = len(df)
    print(f"Split {n} samples (stratified by label):")
    print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%) - "
          f"class 1: {(train_df['label']==1).mean()*100:.1f}%")
    print(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%) - "
          f"class 1: {(val_df['label']==1).mean()*100:.1f}%")
    print(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%) - "
          f"class 1: {(test_df['label']==1).mean()*100:.1f}%")

    # Save splits if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        print(f"  Saved to {output_dir}/")

    return train_df, val_df, test_df


class CloudyTileDatasetNC(Dataset):
    """
    Dataset for loading satellite tile images from NetCDF files with binary labels.

    Supports multi-channel imagery (RGB + NIR + SWIR) for training while using
    filenames that match JPG-based labels.

    Args:
        labels_csv: Path to CSV with columns 'filename' and 'label'
            (filename should be like 'CW2019_1579_t003.jpg')
        nc_dir: Directory containing the single-timestep .nc files
            (files should be like 'CW2019_1579_t003.nc')
        channels: List of channel names to load. Default: all available.
        img_size: Target image size (height, width). Default: (512, 512)
        imagery_scale: Scale factor for normalization (default: 10000.0)

    Labels:
        0 = not useful (cloudy/no data)
        1 = useful (clear)
    """
    def __init__(
        self,
        labels_csv: Union[str, Path],
        nc_dir: Union[str, Path],
        channels: Optional[List[str]] = None,
        img_size: tuple[int, int] = (512, 512),
        imagery_scale: float = 10000.0,
    ):
        self.labels_df = pd.read_csv(labels_csv)
        self.nc_dir = Path(nc_dir)
        self.channels = channels
        self.img_size = img_size
        self.imagery_scale = imagery_scale

        # Convert jpg filenames to nc filenames
        self.labels_df["nc_filename"] = self.labels_df["filename"].str.replace(
            ".jpg", ".nc", regex=False
        )

        # Filter to only include files that exist
        self._filter_existing_files()

        # Determine channel count from first file
        if len(self.labels_df) > 0:
            self._setup_channels()

    def _filter_existing_files(self):
        """Remove entries for files that don't exist on disk."""
        existing_mask = self.labels_df["nc_filename"].apply(
            lambda f: (self.nc_dir / f).exists()
        )
        n_missing = (~existing_mask).sum()
        if n_missing > 0:
            print(f"Warning: {n_missing} NC files not found, skipping them")
        self.labels_df = self.labels_df[existing_mask].reset_index(drop=True)

    def _setup_channels(self):
        """Determine channels from first NC file."""
        first_file = self.nc_dir / self.labels_df.iloc[0]["nc_filename"]
        with xr.open_dataset(first_file) as ds:
            available_channels = list(ds.coords["channel"].values)

        if self.channels is None:
            self.channels = available_channels
        else:
            # Filter to only available channels
            self.channels = [c for c in self.channels if c in available_channels]

        self.n_channels = len(self.channels)
        print(f"Loading {self.n_channels} channels: {self.channels}")

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.labels_df.iloc[idx]
        nc_path = self.nc_dir / row["nc_filename"]

        # Load NC file
        with xr.open_dataset(nc_path) as ds:
            # Select channels
            imagery = ds["imagery"].sel(channel=self.channels).values  # [C, H, W]

        # Normalize to [0, 1]
        imagery = np.clip(imagery / self.imagery_scale, 0.0, 1.0)
        imagery = np.nan_to_num(imagery, nan=0.0)

        # Resize if needed
        if imagery.shape[1:] != self.img_size:
            # Use torch interpolate for resizing
            imagery_tensor = torch.from_numpy(imagery).float().unsqueeze(0)
            imagery_tensor = torch.nn.functional.interpolate(
                imagery_tensor,
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            imagery = imagery_tensor.squeeze(0).numpy()

        # Convert to tensor
        image = torch.from_numpy(imagery).float()
        label = int(row["label"])

        return image, label

    @property
    def filenames(self) -> list[str]:
        """Return list of all original filenames in the dataset."""
        return self.labels_df["filename"].tolist()

    @property
    def nc_filenames(self) -> list[str]:
        """Return list of all NC filenames in the dataset."""
        return self.labels_df["nc_filename"].tolist()
