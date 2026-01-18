# CLAUDE.md

This file provides guidance for Claude Code when working with the cloudy-tile repository.

## Project Overview

`cloudytile` is a PyTorch-based binary classifier that determines whether satellite imagery tiles are "useful" or "not useful" based on cloud coverage and no-data pixels. It's designed as a preprocessing step for downstream tasks like lake drainage classification (see YaoGroup/lake-vision).

## Repository Structure

```
cloudy-tile/
├── cloudytile/           # Main Python package (importable)
│   ├── __init__.py
│   ├── model.py          # CloudyTileCNN architecture
│   ├── data.py           # PyTorch Dataset classes (RGB and multi-spectral)
│   ├── inference.py      # Model loading and prediction utilities
│   └── preprocessing.py  # NetCDF to JPG extraction for training data
├── engine/               # Runnable scripts (entry points)
│   ├── preprocessing/
│   │   └── extract_jpgs.py
│   ├── training/
│   │   ├── run_training.py           # Main training script
│   │   ├── run_spectral_array.sh     # SLURM array job for spectral band sweep
│   │   ├── train_top3_models.sh      # Train top 3 spectral models and save weights
│   │   ├── sweep_spectral.yaml       # Wandb sweep config for spectral bands
│   │   └── run_sweep*.sh             # Various sweep runner scripts
│   ├── inference/
│   │   └── run_inference_lakes.sh    # Run inference on lake NC files
│   └── labeling/
│       └── export_labelbox.py        # Fetch labels from Labelbox API
├── labels.csv            # Exported labels from Labelbox
└── assets/               # Documentation images
```

## Key Technical Details

### Model Architecture
- `CloudyTileCNN`: Simple CNN with configurable conv layers (default: [16, 32, 64]) and FC layers (default: [128])
- Input: Multi-spectral images (configurable channels), shape [B, C, H, W]
- Output: Logits (use sigmoid for probabilities)
- Default image size: 512x512
- Supports variable input channels via `in_channels` parameter

### Multi-Spectral Support

The model and data pipeline support these Sentinel-2 bands:
- `red`, `green`, `blue` (visible)
- `nir` (near-infrared)
- `swir16` (SWIR Band 11, 1.6μm)
- `swir22` (SWIR Band 12, 2.2μm)

**Top-performing band combinations** (from spectral sweep):
1. RGB (`red,green,blue`) - 3 channels
2. RGB+NIR (`red,green,blue,nir`) - 4 channels
3. B+NIR+SWIR16 (`blue,nir,swir16`) - 3 channels

### Data Classes

**`CloudyTileDataset`** (RGB/JPG-based):
- Loads JPG images from disk
- Simple `/10000` normalization to [0, 1]

**`CloudyTileDatasetNC`** (Multi-spectral/NetCDF-based):
- Loads directly from NetCDF files with `nc_channels` parameter
- Supports per-band normalization via `band_stats` JSON file
- Band stats location: `/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats.json`

### Normalization

Per-band mean/std normalization (recommended for multi-spectral):
```python
# band_stats.json format
{
    "red": {"mean": 1234.5, "std": 567.8},
    "green": {"mean": 1100.2, "std": 498.3},
    ...
}
```

### Inference Pipeline

**Key functions in `inference.py`:**

```python
# Load a trained model
model = load_model(
    weights_path="cloudytile_rgb.pth",
    img_size=(512, 512),
    channels=[16, 32, 64],
    fc_layers=[128],
    in_channels=3,  # Must match training
)

# Predict on a single NC file
predictions = predict_from_nc(
    model, ds,
    nc_channels=['red', 'green', 'blue'],
    band_stats=band_stats_dict,
)

# Add cloudy_seq to NC file (in-place)
add_cloudy_seq_to_nc(
    nc_path="lake_001.nc",
    model=model,
    nc_channels=['red', 'green', 'blue'],
    band_stats=band_stats_dict,
    var_name='cloudy_seq_rgb',  # Custom output variable name
)

# Process entire directory
process_directory(
    nc_dir="/path/to/nc/files",
    model_path="cloudytile_rgb.pth",
    nc_channels=['red', 'green', 'blue'],
    band_stats_path="band_stats.json",
    var_name='cloudy_seq_rgb',
)
```

## Sherlock HPC Workflows

### Training Spectral Sweep (SLURM Array Job)

The spectral band sweep tests 15 different band combinations using SLURM job arrays instead of wandb sweeps (avoids duplicate runs in offline mode).

```bash
cd /oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile/engine/training
sbatch run_spectral_array.sh
```

- Uses `#SBATCH --array=0-14%8` (15 configs, max 8 concurrent)
- Requires A100 GPUs: `#SBATCH -C GPU_SKU:A100_SXM4`
- Status logged to: `sherlock_cloudytile/logs/spectral_array_<JOB_ID>_status.log`

### Training Top 3 Models (Save Weights)

```bash
cd /oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile/engine/training
sbatch train_top3_models.sh
```

Trains and saves:
- `cloudytile_rgb.pth` - RGB model
- `cloudytile_rgbn.pth` - RGB+NIR model
- `cloudytile_bns16.pth` - B+NIR+SWIR16 model

Weights saved to: `/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/models/`

### Running Inference on Lake NC Files

```bash
cd /oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile/engine/inference
sbatch run_inference_lakes.sh
```

Adds three variables to each lake NC file:
- `cloudy_seq_rgb` - predictions from RGB model
- `cloudy_seq_rgbn` - predictions from RGB+NIR model
- `cloudy_seq_bns16` - predictions from B+NIR+SWIR16 model

Input directory: `/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_processed`

### Syncing Wandb Offline Runs

After training jobs complete, sync from login node:
```bash
cd /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile
wandb sync wandb/offline-run-*

# Or sync specific date
wandb sync wandb/offline-run-20260118_*
```

## Key File Paths on Sherlock

| Resource | Path |
|----------|------|
| Repository | `/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile` |
| Training NC data | `/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/training_nc` |
| Band statistics | `/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats.json` |
| Lake NC files | `/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_processed` |
| Sherlock workdir | `/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile` |
| Saved models | `/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/models` |
| Logs | `/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/logs` |

## Code Style

- Use type hints for function signatures
- Docstrings follow Google style with Args/Returns sections
- Imports organized: stdlib, third-party, local

## Dependencies

- PyTorch (model, training)
- xarray (NetCDF handling)
- PIL/Pillow (image I/O)
- labelbox (annotation fetching)
- pandas, numpy

## Related Projects

- [YaoGroup/lake-vision](https://github.com/YaoGroup/lake-vision) - Downstream lake drainage classification that consumes `cloudy_seq` outputs from this model

## Changelog

### January 2026
- Added multi-spectral support to `CloudyTileCNN` via `in_channels` parameter
- Added `CloudyTileDatasetNC` for loading multi-spectral NetCDF data
- Updated `inference.py` to support multi-spectral models with `nc_channels` and `band_stats`
- Added `var_name` parameter to inference functions for custom output variable names
- Created `run_spectral_array.sh` - SLURM array job for spectral band hyperparameter sweep
- Created `train_top3_models.sh` - Train and save top 3 spectral models
- Created `run_inference_lakes.sh` - Run inference on lake NC files with all 3 models
- GPU constraint for H100 incompatibility: use `-C GPU_SKU:A100_SXM4` on Sherlock
