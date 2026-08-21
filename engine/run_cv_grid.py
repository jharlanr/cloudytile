#!/usr/bin/env python
"""
Lake-grouped cross-validation over a small config grid.

Grid axes (32 configs):
    bands:     rgb | rgb+nir | rgb+swir16 | all6
    channels:  [16,32,64] | [32,64,128]
    lr:        1e-3 | 3e-4
    optimizer: adam | adamw

Every config is scored with the same lake-grouped folds, so numbers are
comparable across configs and none of them ever sees a test lake in training.
Within each fold, 15% of the *training* lakes are held out as validation for
checkpoint selection — the fold's test lakes touch nothing but the final eval.

Each (config, fold) run writes one JSON to --out_dir; a SLURM array maps one
config per task (--config_index), and --summarize aggregates the JSONs into a
ranked table with mean +/- std across folds. Designed to be resumable: existing
result files are skipped.

Usage:
    # one config (SLURM array task)
    python run_cv_grid.py --labels_csv labels/labels.csv --nc_dir <tiles> \
        --band_stats band_stats.json --out_dir cv_results --config_index 7

    # everything sequentially (local/debug)
    python run_cv_grid.py ... --config_index -1

    # aggregate
    python run_cv_grid.py --out_dir cv_results --summarize
"""
import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# add repo root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# RGB is always present -- it is the modality the task is defined on -- so the
# band question is which subset of the three non-visible bands to add. These
# eight entries are that complete lattice.
#
# ORDER IS LOAD-BEARING. GRID is itertools.product(BAND_SETS, ...) with bands
# outermost, so a config's index is 8*band + 4*arch + 2*lr + optimizer. New
# band sets must therefore be APPENDED, never inserted or reordered: appending
# leaves every existing index untouched (verified for the 32-config v1 grid and
# the finalist index list) while inserting would silently re-point them.
# "all6" keeps its original name rather than becoming "rgb+nir+swir16+swir22"
# for the same reason -- results are matched by config_name.
BAND_SETS = {
    "rgb": ["red", "green", "blue"],
    "rgb+nir": ["red", "green", "blue", "nir"],
    "rgb+swir16": ["red", "green", "blue", "swir16"],
    "all6": ["red", "green", "blue", "nir", "swir16", "swir22"],
    "rgb+swir22": ["red", "green", "blue", "swir22"],
    "rgb+nir+swir16": ["red", "green", "blue", "nir", "swir16"],
    "rgb+nir+swir22": ["red", "green", "blue", "nir", "swir22"],
    "rgb+swir16+swir22": ["red", "green", "blue", "swir16", "swir22"],
}
CHANNEL_SETS = {"small": [16, 32, 64], "wide": [32, 64, 128]}
LRS = [1e-3, 3e-4]
OPTIMIZERS = ["adam", "adamw"]

GRID = [
    {"bands": b, "arch": a, "lr": lr, "optimizer": opt}
    for b, a, lr, opt in itertools.product(BAND_SETS, CHANNEL_SETS, LRS, OPTIMIZERS)
]


def config_name(cfg: dict) -> str:
    return f"{cfg['bands']}_{cfg['arch']}_lr{cfg['lr']:g}_{cfg['optimizer']}"


def split_off_val_lakes(train_df: pd.DataFrame, seed: int, val_frac: float = 0.15):
    """Hold out whole lakes from a fold's training set for checkpoint selection."""
    lakes = np.sort(train_df["lake_id"].unique())
    if len(lakes) < 2:
        raise ValueError(
            f"fold has {len(lakes)} training lake(s); need >=2 to hold one out "
            f"for checkpoint selection. Use fewer folds or more lakes."
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(lakes)
    # never consume the whole training set
    n_val = min(max(1, int(round(len(shuffled) * val_frac))), len(shuffled) - 1)
    val_lakes = set(shuffled[:n_val])
    val_df = train_df[train_df["lake_id"].isin(val_lakes)].reset_index(drop=True)
    tr_df = train_df[~train_df["lake_id"].isin(val_lakes)].reset_index(drop=True)
    return tr_df, val_df


def run_one(cfg: dict, fold: int, train_df, test_df, args) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from cloudytile.data import CloudyTileDatasetNC
    from cloudytile.model import CloudyTileCNN
    from cloudytile.splits import assert_lake_disjoint
    from cloudytile.training import (evaluate, pick_threshold, predict_probs,
                                     train_one_epoch)

    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)

    # One wandb run per (config, fold), grouped by config so the UI averages
    # folds natively. Compute nodes have no internet: WANDB_MODE=offline is set
    # by the SLURM wrapper, and runs are synced from a login node afterwards.
    run = None
    if WANDB_AVAILABLE and args.wandb_project:
        run = wandb.init(
            project=args.wandb_project,
            name=f"{config_name(cfg)}_fold{fold}",
            group=config_name(cfg),
            job_type="cv",
            tags=[cfg["bands"], cfg["arch"], cfg["optimizer"], f"fold{fold}"],
            config={**cfg, "fold": fold, "img_size": args.img_size,
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "weight_decay": args.weight_decay, "seed": args.seed,
                    "lr_schedule": args.lr_schedule,
                    "n_bands": len(BAND_SETS[cfg["bands"]])},
            reinit=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = BAND_SETS[cfg["bands"]]
    img_size = (args.img_size, args.img_size)

    tr_df, val_df = split_off_val_lakes(train_df, seed=args.seed + fold)
    assert_lake_disjoint(tr_df, val_df, test_df)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        splits = {}
        for name, df in (("train", tr_df), ("val", val_df), ("test", test_df)):
            df.to_csv(tmp / f"{name}.csv", index=False)
            splits[name] = tmp / f"{name}.csv"

        common = dict(nc_dir=args.nc_dir, channels=channels,
                      img_size=img_size, band_stats=args.band_stats)
        train_ds = CloudyTileDatasetNC(splits["train"], augment=True, **common)
        val_ds = CloudyTileDatasetNC(splits["val"], **common)
        test_ds = CloudyTileDatasetNC(splits["test"], **common)

        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True),
            "val": DataLoader(val_ds, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True),
            "test": DataLoader(test_ds, batch_size=args.batch_size,
                               num_workers=args.num_workers, pin_memory=True),
        }

        model = CloudyTileCNN(
            img_size=img_size,
            channels=CHANNEL_SETS[cfg["arch"]],
            in_channels=len(channels),
        ).to(device)

        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg["optimizer"]]
        optimizer = opt_cls(model.parameters(), lr=cfg["lr"],
                            weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # At constant lr the tail of the val curve is noisier than the trend it
        # is meant to reveal: on the 40-epoch grid the mean epoch-to-epoch val
        # swing was 0.0136 against a total improvement of ~0.0112 over the last
        # ten epochs. Since the checkpoint is chosen by argmin over that series,
        # noise that large makes selection partly a lottery. Annealing to ~0
        # settles the tail so the minimum means something. Default stays "none"
        # so the completed 40-epoch grid remains reproducible.
        scheduler = None
        if args.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs)

        best_val, best_state, best_epoch = float("inf"), None, None
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, loaders["train"], optimizer,
                                         criterion, device)
            val_loss, val_metrics = evaluate(model, loaders["val"], criterion, device)
            if scheduler is not None:
                scheduler.step()
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch + 1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"  [{config_name(cfg)} fold {fold}] epoch {epoch+1}/{args.epochs} "
                  f"train {train_loss:.4f} val {val_loss:.4f} "
                  f"acc {val_metrics['accuracy']:.3f}")
            sys.stdout.flush()
            if run is not None:
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss,
                           "val_loss": val_loss,
                           "lr": optimizer.param_groups[0]["lr"],
                           **{f"val_{k}": v for k, v in val_metrics.items()}})

        # Score the checkpoint that would actually be shipped, at an operating
        # point chosen on this fold's validation lakes (never on its test lakes).
        model.load_state_dict(best_state)
        _, val_labels, val_probs = predict_probs(model, loaders["val"],
                                                 criterion, device)
        threshold, _ = pick_threshold(val_labels, val_probs,
                                      objective=args.threshold_objective,
                                      target_precision=args.target_precision)
        test_loss, test_metrics = evaluate(model, loaders["test"], criterion,
                                           device, threshold=threshold)

    result = {
        "config": cfg,
        "config_name": config_name(cfg),
        "fold": fold,
        "epochs": args.epochs,
        "lr_schedule": args.lr_schedule,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "threshold": threshold,
        "n_parameters": model.n_parameters(),
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "n_test_tiles": len(test_df),
        "n_test_lakes": int(test_df["lake_id"].nunique()),
    }

    if run is not None:
        for k, v in result.items():
            if isinstance(v, (int, float)):
                wandb.summary[k] = v
        wandb.finish()

    return result


def summarize(out_dir: Path):
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not rows:
        print(f"no result JSONs in {out_dir}")
        return
    df = pd.DataFrame(rows)
    # Ranked by AUC: it is threshold-free, so it compares configs without also
    # comparing whatever operating point each fold happened to choose. Accuracy
    # is reported alongside but sits close to the 68% majority rate.
    agg = (df.groupby("config_name")
             .agg(folds=("fold", "count"),
                  auc_mean=("test_auc", "mean"),
                  auc_std=("test_auc", "std"),
                  acc_mean=("test_accuracy", "mean"),
                  acc_std=("test_accuracy", "std"),
                  prec_mean=("test_precision", "mean"),
                  rec_mean=("test_recall", "mean"),
                  f1_mean=("test_f1", "mean"),
                  params=("n_parameters", "first"))
             .sort_values("auc_mean", ascending=False))
    pd.set_option("display.width", 200)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))
    out = out_dir / "summary.csv"
    agg.to_csv(out)
    print(f"\nWrote {out}")

    if len(agg) > 1:
        top, second = agg.iloc[0], agg.iloc[1]
        gap = top["auc_mean"] - second["auc_mean"]
        spread = float(top["auc_std"]) if pd.notna(top["auc_std"]) else 0.0
        print(f"\nTop: {agg.index[0]} (AUC {top['auc_mean']:.4f} "
              f"+/- {spread:.4f} across folds)")
        if gap < spread:
            print(f"  NOTE: the gap to #2 ({gap:.4f}) is smaller than the "
                  f"fold-to-fold spread — treat the ranking as a tie and "
                  f"prefer the simpler config.")


def main():
    p = argparse.ArgumentParser(description="Lake-grouped CV over a config grid")
    p.add_argument("--labels_csv", type=str, default="labels/labels.csv")
    p.add_argument("--split_dir", type=str, default=None,
                   help="Frozen split directory (engine/make_splits.py). Folds "
                        "are drawn from train+val ONLY; the frozen test lakes "
                        "are never seen during selection. Strongly recommended.")
    p.add_argument("--nc_dir", type=str, default=None,
                   help="Directory of per-tile .nc files")
    p.add_argument("--band_stats", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--config_index", type=int, default=-1,
                   help=f"0..{len(GRID)-1} for one config (SLURM array), "
                        f"-1 for all sequentially")
    p.add_argument("--folds", type=int, default=3,
                   help="Lake-grouped folds per config (default 3 for the "
                        "grid; rerun the winner at 5)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=256,
                   help="GAP head is size-agnostic; 256 makes the grid ~4x "
                        "cheaper than 512")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_schedule", type=str, default="none",
                   choices=["none", "cosine"],
                   help="'cosine' anneals lr to ~0 over --epochs, which settles "
                        "the val curve so argmin checkpoint selection is not "
                        "dominated by epoch-to-epoch noise. Default 'none' "
                        "reproduces the original 40-epoch grid.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--wandb_project", type=str, default="cloudy-tile-cv",
                   help="wandb project; one run per (config, fold), grouped by "
                        "config. Set --no_wandb to disable.")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--threshold_objective", type=str, default="f1",
                   choices=["f1", "target_precision"])
    p.add_argument("--target_precision", type=float, default=0.95)
    args = p.parse_args()
    if args.no_wandb:
        args.wandb_project = None
    elif not WANDB_AVAILABLE:
        print("wandb not installed; results still written as JSON")
        args.wandb_project = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize(out_dir)
        return

    if args.nc_dir is None:
        p.error("--nc_dir is required unless --summarize")

    import tempfile as _tempfile

    from cloudytile.splits import dev_labels, lake_group_kfold

    configs = GRID if args.config_index < 0 else [GRID[args.config_index]]
    print(f"{len(GRID)} configs in grid; running {len(configs)} "
          f"x {args.folds} folds at {args.img_size}px")

    # Restrict selection to the development lakes. Without this the grid folds
    # over every lake, and picking the best of 32 configs by fold score is
    # selection on the test set.
    labels_for_cv = args.labels_csv
    _tmp_ctx = None
    if args.split_dir:
        dev = dev_labels(args.split_dir, args.labels_csv)
        _tmp_ctx = _tempfile.TemporaryDirectory()
        labels_for_cv = Path(_tmp_ctx.name) / "dev_labels.csv"
        dev[["filename", "label"]].to_csv(labels_for_cv, index=False)
        print(f"Selection pool: {len(dev)} tiles / {dev['lake_id'].nunique()} "
              f"dev lakes (frozen test lakes excluded)")
    else:
        print("WARNING: no --split_dir; folding over ALL lakes. The winning "
              "config's score will be optimistically biased.")

    # Folds are a function of (labels, folds, seed) only — identical for every
    # config, which is what makes config scores comparable.
    folds = list(lake_group_kfold(labels_for_cv, n_splits=args.folds,
                                  seed=args.seed))
    if _tmp_ctx is not None:
        _tmp_ctx.cleanup()

    for cfg in configs:
        for fold, train_df, test_df in folds:
            out_path = out_dir / f"{config_name(cfg)}_fold{fold}.json"
            if out_path.exists():
                print(f"skip existing {out_path.name}")
                continue
            result = run_one(cfg, fold, train_df, test_df, args)
            tmp = out_path.parent / f".{out_path.name}.tmp"
            tmp.write_text(json.dumps(result, indent=2))
            tmp.rename(out_path)
            print(f"wrote {out_path.name}: acc={result['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
