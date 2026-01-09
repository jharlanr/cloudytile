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
│   ├── data.py           # PyTorch Dataset class for training
│   ├── inference.py      # Model loading and prediction utilities
│   └── preprocessing.py  # NetCDF to JPG extraction for training data
├── engine/               # Runnable scripts (entry points)
│   ├── preprocessing/
│   │   └── extract_jpgs.py
│   ├── training/
│   │   └── run_training.py
│   └── labeling/
│       └── export_labelbox.py   # Fetch labels from Labelbox API
├── labels.csv            # Exported labels from Labelbox
└── assets/               # Documentation images
```

## Key Technical Details

### Model Architecture
- `CloudyTileCNN`: Simple CNN with configurable conv layers (default: [16, 32, 64]) and FC layers (default: [128])
- Input: RGB images normalized to [0, 1], shape [B, 3, H, W]
- Output: Logits (use sigmoid for probabilities)
- Default image size: 512x512

### Data Format
- Training data: JPG/PNG images extracted from NetCDF timestacks
- Labels: Binary (0 = not useful/cloudy, 1 = useful)
- Imagery scale: 10000.0 (Sentinel-2 normalization)

### Preprocessing Pipeline
- `preprocessing.py:extract_frames_from_nc()` - Extract JPGs from a single NetCDF file
- `preprocessing.py:extract_frames_from_directory()` - Batch extract from multiple NetCDF files
- Both support `skip_existing=True` to avoid re-extracting existing files

### Labeling Workflow
- Labels are exported from Labelbox using `export_labelbox.py`
- Labelbox project ID: `cmk5q6tey0hru07y8024t1lsb`
- Use `--task_id` to reuse an existing export (faster than creating new)
- Output: CSV with columns `filename`, `label` (0=not useful, 1=useful)

### Inference Pipeline
- `inference.py:predict_from_nc()` - Run predictions on all timesteps in a NetCDF file
- `inference.py:add_cloudy_seq_to_nc()` - Add `cloudy_seq` variable to NetCDF files for use in lake-vision

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

- [YaoGroup/lake-vision](https://github.com/YaoGroup/lake-vision) - Downstream lake detection that consumes `cloudy_seq` outputs
