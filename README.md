# cloudytile

`cloudytile` is a PyTorch binary classifier that decides whether a satellite imagery tile is
**useful** (clear enough to read the lake, even through thin cloud) or **not useful**
(cloud-obscured / mostly no-data). It exists because Sentinel-2's own cloud masks are unreliable
over bright ice. Its per-timestep predictions (`cloudy_seq_rgb`) feed the downstream
[YaoGroup/lake-vision](https://github.com/YaoGroup/lake-vision) drainage classifier.

New here? Read **[docs/lecture-cloudytile.html](docs/lecture-cloudytile.html)** — a pedagogical
walkthrough of the pipeline, the dataset, and the architecture.

## Tile Examples
<table>
  <tr>
    <th align="right">useful</th>
    <td><img src="assets/eg_useful1.png" alt="useful 1" width="240"/></td>
    <td><img src="assets/eg_useful3.png" alt="useful 2" width="240"/></td>
  </tr>
  <tr>
    <th align="right">not useful</th>
    <td><img src="assets/eg_useless1.png" alt="not useful 1" width="240"/></td>
    <td><img src="assets/eg_useless2.png" alt="not useful 2" width="240"/></td>
  </tr>
</table>

Note the second "useful" example is visibly cloudy — the lake still reads through the thin layer,
so the tile carries information. Usability, not cloud presence, is the label.

## Repository Structure

```
cloudy-tile/
├── cloudytile/                  # Importable package
│   ├── model.py                 # CloudyTileCNN (GAP head; legacy flatten kept)
│   ├── data.py                  # Datasets (NC primary, JPG legacy)
│   ├── splits.py                # Lake-grouped splitting + baselines
│   ├── labels.py                # Label CSV store (atomic writes)
│   ├── training.py              # Train/eval loops and metrics
│   ├── inference.py             # cloudy_seq prediction on lake NC files
│   ├── preprocessing.py         # NetCDF -> JPG rendering helpers
│   └── tests/                   # pytest suite
├── engine/                      # Runnable scripts (flat, pure Python)
│   ├── extract_label_frames.py  # SDR .nc -> JPGs for labeling
│   ├── label_gui.py (+index.html)  # Browser labeling GUI
│   ├── extract_training_nc.py   # labels.csv -> per-tile 6-band .nc
│   ├── compute_band_stats.py    # Per-band mean/std for normalization
│   ├── run_training.py          # Single training run
│   ├── run_cv_grid.py           # Lake-grouped CV over a config grid
│   └── run_inference.py         # Apply model to lake NC files
├── slurm/                       # Sherlock submission wrappers (paths, modules)
├── labels/
│   ├── labels.csv               # Current campaign: 10,000 frames, 400 lakes
│   └── labels_deprecated.csv    # Retired Labelbox export (do not merge)
└── docs/                        # Lecture + (gitignored) claudiary
```

## The Pipeline

1. **Extract frames for labeling** — samples lakes × random timesteps from the ESSD SDR deposit,
   drops frames ≥50% no-data, renders true-color JPGs:
   ```bash
   sbatch slurm/extract_label_frames.sh
   ```
2. **Label** — one image, two keys, atomic CSV writes, automatic resume:
   ```bash
   python3 engine/label_gui.py --image_dir data/label_frames_2019
   # ← = 0 not useful | → = 1 useful | backspace = back | space = skip | h = help
   ```
3. **Extract training tiles** — one 6-band `.nc` per labeled frame
   (JPGs are for human eyes only; training reads NetCDF):
   ```bash
   sbatch slurm/extract_training_nc.sh
   sbatch slurm/compute_band_stats.sh
   ```
4. **Train**:
   ```bash
   python3 engine/run_training.py \
       --labels_csv labels/labels.csv \
       --nc_dir /path/to/training_nc_10k \
       --band_stats /path/to/band_stats_10k.json \
       --optimize_metric loss --save_path best_model.pth
   ```
   Splits are **lake-grouped by default** (`--split_by tile` exists only to reproduce old runs
   and warns). Test metrics are computed on the restored best-validation checkpoint — the same
   weights written to `--save_path`.
5. **Model selection** — small grid (bands × width × lr × optimizer = 32 configs), each scored
   with identical lake-grouped folds:
   ```bash
   sbatch slurm/run_cv_grid.sh
   python3 engine/run_cv_grid.py --out_dir <results> --summarize
   ```
6. **Inference** over the lake archive:
   ```bash
   python3 engine/run_inference.py --model best_model.pth --input /path/to/nc_dir
   ```

## Data Conventions

- Frame key: `{lake_id}_t{timestep:03d}` — `.jpg` for labeling, `.nc` for training.
- `labels/labels.csv`: columns `filename,label`; **0 = not useful, 1 = useful** (polarity is
  load-bearing downstream).
- Training tiles carry all six SDR bands (`red, green, blue, nir, swir16, swir22`) as float32
  surface-reflectance DN with NaN for no-data; band subsetting is a training-time choice
  (`--nc_channels`).
- Normalization: per-band mean/std from `band_stats.json`, then NaN → exactly 0.0
  (in normalized space, in that order — see `CloudyTileDatasetNC`).

## Model

`CloudyTileCNN`: three conv blocks (conv 3×3 → BatchNorm → ReLU → maxpool) and a
global-average-pooling head — ~32k parameters, input-size agnostic. The legacy flatten head
(33.5M parameters at 512px, 99.9% of them in one dense layer) is kept only for loading
pre-August-2026 checkpoints: `head="flatten", batch_norm=False, dropout=0.0`.

## Baselines

Report accuracy against these, not against chance (40 held-out lakes, current dataset):

| baseline | accuracy |
|---|---|
| majority class | 68.4% |
| JPG file-size threshold | 82.5% |

`cloudytile/splits.py::filesize_baseline` computes the second; anything a model earns above it is
work on the actual clear-vs-cloud problem.

## Tests

```bash
python3 -m pytest cloudytile/tests/
```

Covers label-store atomicity/merge semantics, lake-split disjointness (including a regression
test that the old tile-level split leaks), GAP/flatten parameter counts, and the
NaN-after-normalization contract.

## Key Paths (OAK)

| Resource | Path |
|---|---|
| SDR imagery (source of everything) | `data/essd_sdr/data/CW{2018,2019}/` |
| Labeling frames (JPG) | `data/cloudytile/label_frames_{2018,2019}/` |
| Training tiles (.nc) | `data/cloudytile/training_nc_10k/` |
| Band statistics | `data/cloudytile/band_stats_10k.json` |
| Models / logs / CV results | `sherlock/sherlock_cloudytile/` |

All under `/oak/stanford/groups/cyaolai/JoshRines` (locally `/Volumes/groups/cyaolai/JoshRines`).

## How to Contribute

PRs and issues welcome on [the GitHub repo](https://github.com/jharlanr/cloudy-tile).
