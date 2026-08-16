#!/usr/bin/env python
"""
Sample frames for labeling: N random lakes x K random timesteps, skipping no-data frames.

Produces a lake-diverse labeling set. The previous training set covered only 60
lakes, too few to hold any out of a lake-grouped split; spreading the same
labeling budget over many more lakes fixes that. Timesteps are drawn at random
across the melt season rather than consecutively, since adjacent frames of one
lake are near-duplicates.

Frames whose RGB is >= --max_nan_frac no-data are dropped before a human ever
sees them. Those are decidable by rule, so labeling them spends effort on a
solved subproblem.

Source: the ESSD SDR deposit, /oak/.../data/essd_sdr/data/CW{2018,2019}/, which
holds 679 + 1000 per-lake files. Only the imagery is shared with that deposit —
cloudy-tile keeps its own splits, since tile usefulness is a different task from
drainage class and should not inherit that stratification.

Split by lake_id, never by tile: the previous 80/10/10 tile split put all 52
test lakes in train as well, so nothing was actually held out.

Usage:
    python extract_label_frames.py \
        --nc_dir /oak/.../data/essd_sdr/data/CW2019 \
        --output_dir /oak/.../data/cloudytile/label_frames_2019 \
        --n_lakes 500 --frames_per_lake 10

    SLURM submission:
        sbatch extract_label_frames.sh

Output:
    {output_dir}/{lake_id}_t{timestep:03d}.jpg    matches labels.csv's filename key
    {output_dir}/manifest.csv                     provenance + per-frame metadata

Note on --imagery_scale: 10000 is the Sentinel-2 quantification value, so the
default render is true color. Confirmed against the SDR data: on a clear CW2019
frame the median converts to 0.965 reflectance and the max to 1.09, i.e.
Greenland ice genuinely sits just below 1.0 and only a thin tail exceeds it.

Bright early-season scenes still push against the top of the 8-bit range. Raising
the scale would recover that headroom but renders a non-physical reflectance
range, so it is worth doing only if blown-out frames turn out to make thin cloud
over ice hard to call by eye.

This only affects what the human labeler sees. Training through
CloudyTileDatasetNC reads float reflectance from the .nc directly and is
unaffected by the JPG render.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# add repo root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloudytile.preprocessing import save_frame_as_jpg

RGB = ["red", "green", "blue"]


def resolve_rgb(ds: xr.Dataset) -> tuple[str, list[int]]:
    """
    Locate the RGB bands, which differ between the two timestack schemas.

    Raw tstacks store 'reflectance' on a 'band' dim; the processed 2019 files
    store 'imagery' on a named 'channel' dim. Raw files list swir16 twice in
    common_name (index 6 is really the cloudmask), so RGB is resolved by
    position from the band attribute rather than by name lookup.

    Returns:
        (variable_name, [red_idx, green_idx, blue_idx])
    """
    if "imagery" in ds.data_vars:
        channels = [str(c) for c in ds.coords["channel"].values]
        return "imagery", [channels.index(c) for c in RGB]

    if "reflectance" in ds.data_vars:
        names = None
        if "common_name" in ds.coords or "common_name" in ds.variables:
            names = [str(c) for c in ds["common_name"].values]
        elif "band" in ds.attrs:
            # attrs['band'] is Sentinel-2 IDs: B04=red, B03=green, B02=blue
            s2 = {"B04": "red", "B03": "green", "B02": "blue"}
            names = [s2.get(str(b), str(b)) for b in ds.attrs["band"]]
        if names is None:
            raise ValueError("cannot identify bands in 'reflectance' variable")
        return "reflectance", [names.index(c) for c in RGB]

    raise ValueError(f"no recognized imagery variable in {list(ds.data_vars)}")


def lake_id_from(ds: xr.Dataset, path: Path) -> str:
    """Normalize to '{BASIN}{YEAR}_{NNNN}', dropping any 'tstack_' filename prefix."""
    for source in (ds.attrs.get("lake_id"), ds.coords.get("lake_id")):
        if source is not None:
            return str(np.asarray(source).item() if hasattr(source, "values") else source)
    return path.stem.replace("tstack_", "")


def candidate_timesteps(ds: xr.Dataset, max_nan_frac: float) -> list[int]:
    """
    Cheap pre-filter using the precomputed per-timestep pct_nans coordinate.

    Reading pct_nans costs 153 floats; loading a timestep's RGB costs ~3 MB, so
    this avoids most of the I/O. The cut is deliberately loose (>=99% only) —
    pct_nans' exact semantics vary between files, so it is used solely to drop
    certainly-empty frames, and every survivor is verified against its real RGB.
    """
    n_time = ds.sizes["time"]
    if "pct_nans" in ds.variables and ds["pct_nans"].dims == ("time",):
        pct = np.asarray(ds["pct_nans"].values, dtype=float)
        # NaN pct_nans marks a timestep with no acquisition at all.
        return [t for t in range(n_time)
                if np.isfinite(pct[t]) and pct[t] < 100.0 * max_nan_frac]
    return list(range(n_time))


def frame_rgb_and_nan(ds: xr.Dataset, var: str, idx: list[int], t: int, scale: float):
    """
    Load one timestep's RGB. Returns (uint8 [H,W,3], nan_fraction).

    The SDR product stores raw L2A digital numbers, not reflectance:
    surface_reflectance = (DN + boa_add_offset) / 10000, with boa_add_offset
    supplied per timestep. It is 0 across the 2018/2019 scenes checked (they
    predate ESA processing baseline 04.00, which introduced a -1000 offset), but
    it is applied here rather than assumed so rescenes do not silently shift the
    render by 0.1 reflectance.
    """
    arr = ds[var].isel(time=t).isel({ds[var].dims[1]: idx}).values.astype(np.float32)

    # a pixel is unusable if any RGB band is missing there
    nan_frac = float(np.isnan(arr).any(axis=0).mean())

    offset = 0.0
    if "boa_add_offset" in ds.variables and ds["boa_add_offset"].dims == ("time",):
        raw = float(ds["boa_add_offset"].isel(time=t).values)
        offset = raw if np.isfinite(raw) else 0.0

    rgb = np.clip((np.nan_to_num(arr, nan=0.0) + offset) / scale, 0.0, 1.0)
    rgb = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
    return rgb, nan_frac


def extract(
    nc_dir: Path,
    output_dir: Path,
    n_lakes: int,
    frames_per_lake: int,
    max_nan_frac: float,
    seed: int,
    imagery_scale: float,
    quality: int,
    skip_existing: bool,
    dry_run: bool,
) -> list[dict]:
    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        raise SystemExit(f"ERROR: no .nc files in {nc_dir}")

    rng = random.Random(seed)
    if n_lakes < len(nc_files):
        nc_files = sorted(rng.sample(nc_files, n_lakes))
    print(f"Sampling {frames_per_lake} frames from each of {len(nc_files)} lakes "
          f"(max_nan_frac={max_nan_frac})")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_rejected = 0
    for i, nc_path in enumerate(nc_files):
        try:
            ds = xr.open_dataset(nc_path)
        except Exception as e:
            print(f"  [{i+1}/{len(nc_files)}] {nc_path.name}: OPEN FAILED ({e})")
            continue

        try:
            var, idx = resolve_rgb(ds)
            lake_id = lake_id_from(ds, nc_path)
            dates = ds.coords["time"].values if "time" in ds.coords else None
            cloud = (np.asarray(ds.coords["eo_cloud_cover"].values, dtype=float)
                     if "eo_cloud_cover" in ds.coords else None)

            # Per-lake seed keeps a rerun reproducible even if --n_lakes changes.
            cands = candidate_timesteps(ds, max_nan_frac)
            random.Random(f"{seed}:{lake_id}").shuffle(cands)

            kept = 0
            for t in cands:
                if kept >= frames_per_lake:
                    break
                filename = f"{lake_id}_t{t:03d}.jpg"
                out_path = output_dir / filename

                row = {
                    "filename": filename,
                    "lake_id": lake_id,
                    "timestep": t,
                    "date": str(dates[t])[:10] if dates is not None else "",
                    "nan_frac": "",
                    "eo_cloud_cover": (f"{cloud[t]:.2f}"
                                       if cloud is not None and np.isfinite(cloud[t]) else ""),
                    "source_nc": nc_path.name,
                }

                # An existing frame already passed the NaN check when it was written.
                if (skip_existing and out_path.exists()) or dry_run:
                    kept += 1
                    rows.append(row)
                    continue

                try:
                    rgb, nan_frac = frame_rgb_and_nan(ds, var, idx, t, imagery_scale)
                except Exception as e:
                    print(f"      t{t:03d}: read failed ({e})")
                    continue

                if nan_frac >= max_nan_frac:
                    n_rejected += 1
                    continue

                save_frame_as_jpg(rgb, out_path, quality)
                kept += 1
                row["nan_frac"] = f"{nan_frac:.4f}"
                rows.append(row)

            if kept < frames_per_lake:
                print(f"  [{i+1}/{len(nc_files)}] {lake_id}: only {kept}/{frames_per_lake} "
                      f"usable frames")
            elif (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(nc_files)}] {lake_id}: {len(rows)} frames so far")
        finally:
            ds.close()

    print(f"\nKept {len(rows)} frames from {len(nc_files)} lakes "
          f"({n_rejected} rejected as >= {max_nan_frac:.0%} no-data)")
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    import csv
    cols = ["filename", "lake_id", "timestep", "date", "nan_frac", "eo_cloud_cover", "source_nc"]
    tmp = path.parent / f".{path.name}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    import os
    os.replace(tmp, path)
    print(f"Manifest: {path}")


def main():
    p = argparse.ArgumentParser(
        description="Sample lake-diverse frames for labeling, skipping no-data frames."
    )
    p.add_argument("--nc_dir", type=str, required=True,
                   help="Directory of lake timestack .nc files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write JPG frames into")
    p.add_argument("--n_lakes", type=int, default=200,
                   help="Number of lakes to sample (default: 200)")
    p.add_argument("--frames_per_lake", type=int, default=25,
                   help="Frames to draw per lake (default: 25)")
    p.add_argument("--max_nan_frac", type=float, default=0.5,
                   help="Reject frames with >= this fraction of no-data RGB (default: 0.5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imagery_scale", type=float, default=10000.0,
                   help="Reflectance scale for the 8-bit render (default: 10000.0, the "
                        "Sentinel-2 L2A true-color convention)")
    p.add_argument("--quality", type=int, default=95, help="JPG quality (default: 95)")
    p.add_argument("--manifest", type=str, default=None,
                   help="Manifest CSV path (default: <output_dir>/manifest.csv)")
    p.add_argument("--no_skip_existing", action="store_true",
                   help="Re-render frames that already exist")
    p.add_argument("--dry_run", action="store_true",
                   help="Report counts without reading imagery or writing files")
    args = p.parse_args()

    nc_dir = Path(args.nc_dir)
    output_dir = Path(args.output_dir)
    if not nc_dir.is_dir():
        raise SystemExit(f"ERROR: --nc_dir does not exist: {nc_dir}")

    print(f"NC dir:     {nc_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target:     {args.n_lakes} lakes x {args.frames_per_lake} frames "
          f"= {args.n_lakes * args.frames_per_lake} max\n")

    rows = extract(
        nc_dir=nc_dir,
        output_dir=output_dir,
        n_lakes=args.n_lakes,
        frames_per_lake=args.frames_per_lake,
        max_nan_frac=args.max_nan_frac,
        seed=args.seed,
        imagery_scale=args.imagery_scale,
        quality=args.quality,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )

    if rows and not args.dry_run:
        manifest = Path(args.manifest) if args.manifest else output_dir / "manifest.csv"
        write_manifest(rows, manifest)
        print(f"\nLabel with:\n  python engine/labeling/label_gui.py "
              f"--image_dir {output_dir} --labels_csv labels_v2.csv")


if __name__ == "__main__":
    main()
