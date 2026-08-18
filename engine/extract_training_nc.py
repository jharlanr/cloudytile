#!/usr/bin/env python
"""
Extract per-tile 6-band training .nc files for every labeled frame.

Reads labels/labels.csv, opens each lake's SDR timestack once, and writes one
single-timestep NetCDF per labeled frame with all six bands
(red, green, blue, nir, swir16, swir22) as float32 surface-reflectance digital
numbers, NaN where no data. Band selection and normalization are training-time
choices (CloudyTileDatasetNC), so every tile carries the full band set.

The SDR deposit stores raw L2A DN with a per-timestep boa_add_offset
(surface_reflectance = (DN + offset) / 10000). The offset is applied here so
the written tiles are directly comparable across processing baselines; it is 0
for all 2018/2019 scenes checked.

Output schema matches what CloudyTileDatasetNC and compute_band_stats.py read:
    imagery(channel, y, x) float32, coord channel = band common names
    filename: {lake_id}_t{timestep:03d}.nc  (labels.csv key with .jpg -> .nc)

Usage:
    python extract_training_nc.py \
        --labels_csv labels/labels.csv \
        --nc_dirs /oak/.../essd_sdr/data/CW2019 /oak/.../essd_sdr/data/CW2018 \
        --output_dir /oak/.../data/cloudytile/training_nc_10k

    SLURM: sbatch slurm/extract_training_nc.sh
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# add repo root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BANDS = ["red", "green", "blue", "nir", "swir16", "swir22"]


def parse_filename(filename: str) -> tuple[str, int]:
    """'CW2019_1579_t003.jpg' -> ('CW2019_1579', 3)"""
    m = re.match(r"(.+)_t(\d+)\.(jpg|nc)$", filename)
    if not m:
        raise ValueError(f"cannot parse {filename!r}")
    return m.group(1), int(m.group(2))


def band_indices(ds: xr.Dataset) -> list[int]:
    """Indices of BANDS within the file's band dimension, by common name."""
    names = [str(c) for c in ds["common_name"].values]
    missing = [b for b in BANDS if b not in names]
    if missing:
        raise ValueError(f"bands {missing} not found in {names}")
    return [names.index(b) for b in BANDS]


def extract_lake(
    nc_path: Path,
    frames: list[tuple[int, str]],
    output_dir: Path,
    skip_existing: bool,
) -> tuple[int, int]:
    """Write one per-tile .nc per (timestep, out_name). Returns (written, skipped)."""
    written = skipped = 0
    with xr.open_dataset(nc_path) as ds:
        idx = band_indices(ds)
        lake_id = str(ds.attrs.get("lake_id", nc_path.stem))
        offsets = (
            np.asarray(ds["boa_add_offset"].values, dtype=float)
            if "boa_add_offset" in ds.variables else None
        )
        times = ds["time"].values if "time" in ds.variables else None

        for t, out_name in frames:
            out_path = output_dir / out_name
            if skip_existing and out_path.exists():
                skipped += 1
                continue

            arr = ds["reflectance"].isel(time=t, band=idx).values.astype(np.float32)
            if offsets is not None and np.isfinite(offsets[t]) and offsets[t] != 0:
                arr = arr + np.float32(offsets[t])

            tile = xr.Dataset(
                data_vars={"imagery": (["channel", "y", "x"], arr)},
                coords={"channel": BANDS},
                attrs={
                    "lake_id": lake_id,
                    "timestep": t,
                    "source_time": str(times[t])[:10] if times is not None else "",
                    "source_nc": nc_path.name,
                    "units": "L2A surface-reflectance DN (offset applied); "
                             "reflectance = DN / 10000",
                },
            )
            tmp = out_path.parent / f".{out_path.name}.tmp"
            tile.to_netcdf(tmp)
            tmp.rename(out_path)
            written += 1
    return written, skipped


def main():
    p = argparse.ArgumentParser(
        description="Extract per-tile 6-band training .nc for labeled frames."
    )
    p.add_argument("--labels_csv", type=str, required=True,
                   help="Labels CSV with 'filename' column")
    p.add_argument("--nc_dirs", type=str, nargs="+", required=True,
                   help="SDR per-lake .nc directories (searched in order for "
                        "{lake_id}.nc)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory for per-tile .nc files")
    p.add_argument("--no_skip_existing", action="store_true",
                   help="Rewrite tiles that already exist")
    args = p.parse_args()

    nc_dirs = [Path(d) for d in args.nc_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    by_lake: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for fn in df["filename"]:
        lake_id, t = parse_filename(fn)
        by_lake[lake_id].append((t, fn.rsplit(".", 1)[0] + ".nc"))

    print(f"{len(df)} labeled frames across {len(by_lake)} lakes")

    total_written = total_skipped = n_missing = 0
    for i, (lake_id, frames) in enumerate(sorted(by_lake.items())):
        nc_path = next(
            (d / f"{lake_id}.nc" for d in nc_dirs if (d / f"{lake_id}.nc").exists()),
            None,
        )
        if nc_path is None:
            print(f"  [{i+1}/{len(by_lake)}] {lake_id}: SOURCE NOT FOUND "
                  f"({len(frames)} frames dropped)")
            n_missing += len(frames)
            continue
        try:
            w, s = extract_lake(nc_path, frames, output_dir,
                                skip_existing=not args.no_skip_existing)
        except Exception as e:
            print(f"  [{i+1}/{len(by_lake)}] {lake_id}: ERROR {e}")
            n_missing += len(frames)
            continue
        total_written += w
        total_skipped += s
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(by_lake)}] {lake_id}: "
                  f"{total_written} written, {total_skipped} skipped")
        sys.stdout.flush()

    print(f"\nDone: {total_written} written, {total_skipped} already existed, "
          f"{n_missing} missing")
    print(f"Next: python engine/compute_band_stats.py --nc_dir {output_dir} "
          f"--output_path band_stats.json")


if __name__ == "__main__":
    main()
