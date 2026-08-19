#!/usr/bin/env python3
"""
Compute the reference baselines on the frozen test lakes.

Any reported model accuracy is meaningless without a floor to compare it to,
and the floor has to be measured on the SAME lakes as the model. Earlier
baseline figures (68.4% / 82.5%) were computed ad hoc, before
splits/cloudytile_v1 was frozen, on a held-out set that is not the current 80
test lakes -- so they could not be compared like-for-like against anything, and
nothing in the repo reproduced them. This script fixes that.

Two baselines:

  majority class   -- always predict the majority label of the DEV set. Knows
                      nothing; the floor below which a model is worse than a
                      constant.
  JPG file size    -- threshold on the byte count of the rendered JPG, nothing
                      else. JPEG is an entropy meter: clear ice is detailed and
                      compresses poorly, cloud is smooth and compresses well.
                      This is a stat() call, and a model that fails to beat it
                      has learned nothing JPEG did not already know.

Both are FIT on the development lakes (train+val) and SCORED on the frozen test
lakes, exactly like the model -- fitting the threshold on test would flatter the
baseline in the same way selection on test flatters a model.

Usage:
    python3 engine/run_baselines.py \
        --labels_csv labels/labels.csv \
        --split_dir splits/cloudytile_v1 \
        --image_dir /oak/.../data/cloudytile/label_frames_2018 \
        --image_dir /oak/.../data/cloudytile/label_frames_2019 \
        --out baselines_cloudytile_v1.json

Only file SIZES are read, never pixels, so this is fast even over a network
mount.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from cloudytile.splits import dev_labels, filesize_baseline, load_frozen_split


def resolve_images(df: pd.DataFrame, image_dirs: list[Path]) -> pd.DataFrame:
    """Keep rows whose JPG exists in one of image_dirs, recording that dir."""
    index = {}
    for d in image_dirs:
        for p in d.glob("*.jpg"):
            index.setdefault(p.name, p)
    hit = df["filename"].map(index)
    missing = int(hit.isna().sum())
    if missing:
        print(f"  warning: {missing}/{len(df)} labeled frames have no JPG on disk")
    out = df.loc[hit.notna()].copy()
    out["path"] = hit.loc[hit.notna()].astype(str)
    return out


def majority_baseline(dev: pd.DataFrame, test: pd.DataFrame) -> dict:
    majority = int(dev["label"].mode().iloc[0])
    acc = float((test["label"] == majority).mean())
    return {"majority_label": majority, "accuracy": acc, "n": int(len(test))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_csv", default="labels/labels.csv")
    p.add_argument("--split_dir", default="splits/cloudytile_v1")
    p.add_argument("--image_dir", action="append", required=True,
                   help="directory of labeling JPGs; repeat for multiple years")
    p.add_argument("--out", default=None, help="write results as JSON here")
    args = p.parse_args()

    image_dirs = [Path(d) for d in args.image_dir]
    for d in image_dirs:
        if not d.is_dir():
            raise SystemExit(f"not a directory: {d}")

    _, _, test = load_frozen_split(args.split_dir, args.labels_csv)
    dev = dev_labels(args.split_dir, args.labels_csv)
    print(f"dev  {len(dev):6d} tiles / {dev['lake_id'].nunique():3d} lakes")
    print(f"test {len(test):6d} tiles / {test['lake_id'].nunique():3d} lakes "
          f"({100 * test['label'].mean():.1f}% useful)")

    overlap = set(dev["lake_id"]) & set(test["lake_id"])
    assert not overlap, f"dev/test lake overlap: {sorted(overlap)[:5]}"

    maj = majority_baseline(dev, test)
    print(f"\nmajority class (always {maj['majority_label']}): "
          f"{100 * maj['accuracy']:.1f}%  (n={maj['n']})")

    dev_i, test_i = resolve_images(dev, image_dirs), resolve_images(test, image_dirs)
    common = image_dirs[0].parent if len(image_dirs) > 1 else image_dirs[0]
    for frame in (dev_i, test_i):
        frame["filename"] = frame["path"].map(lambda s: str(Path(s).relative_to(common)))

    fs = filesize_baseline(args.labels_csv, common, train_df=dev_i, test_df=test_i)
    print(f"JPG file-size threshold ({fs['threshold_bytes'] / 1024:.0f} KB, fit on dev): "
          f"{100 * fs['accuracy']:.1f}%  (n={fs['n']})")

    results = {
        "split_dir": str(args.split_dir),
        "labels_csv": str(args.labels_csv),
        "n_test_tiles": int(len(test)),
        "n_test_lakes": int(test["lake_id"].nunique()),
        "test_useful_rate": float(test["label"].mean()),
        "majority": maj,
        "filesize": fs,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
