#!/usr/bin/env python
"""
Freeze the lake-grouped train/val/test split to disk.

Run this ONCE, before any model selection, and commit the result. Everything
downstream reads these ID lists instead of re-deriving a split from a seed:

    engine/run_cv_grid.py   folds over train+val only (test never touched)
    engine/run_training.py  --split_dir splits/cloudytile_v1

Freezing matters for two reasons. A seed-derived split silently changes the
moment labels.csv changes, so numbers from different weeks stop being
comparable. And keeping the test lakes out of the grid is what stops
hyperparameter selection from quietly optimizing against the test set.

Usage:
    python engine/make_splits.py --labels_csv labels/labels.csv \
        --out_dir splits/cloudytile_v1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudytile.splits import freeze_split


def main():
    p = argparse.ArgumentParser(description="Freeze a lake-grouped split to disk")
    p.add_argument("--labels_csv", type=str, default="labels/labels.csv")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory for {train,val,test}_ids.json + split_meta.json")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.2,
                   help="Held-out test share. 0.2 of 400 lakes = 80 lakes / "
                        "~2000 tiles, enough that the estimate is not dominated "
                        "by which lakes were drawn.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing frozen split")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if (out_dir / "test_ids.json").exists() and not args.force:
        raise SystemExit(
            f"ERROR: {out_dir} already holds a frozen split. Overwriting it "
            f"invalidates every number measured against it — pass --force only "
            f"if that is what you intend, or write to a new directory."
        )

    meta = freeze_split(
        args.labels_csv,
        out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print("\nsplit_meta.json:")
    for k in ("n_lakes", "n_tiles", "positive_rate"):
        print(f"  {k}: {meta[k]}")
    print(f"\nCommit {out_dir}/ so the test set is pinned for the project.")


if __name__ == "__main__":
    main()
