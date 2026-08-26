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
import time
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
CHANNEL_SETS = {
    "small": [16, 32, 64],
    "wide": [32, 64, 128],
    # Six blocks. Depth is what buys receptive field -- 22px at 3 blocks vs
    # 190px at 6, on a 512px tile -- and it is nearly free because blocks 4-6
    # run on 32x32 and smaller grids: 1.3x the MACs of `small`. Doubling the
    # FIRST block instead ([32,64,128,...]) costs 4.6x, because that conv runs
    # at 256x256. Grow channels late.
    "deep6": [16, 32, 64, 64, 96, 128],
}

# Named heads. Every one is "reduce to K channels with a 1x1 conv, then average
# -pool to NxN", so the flattened vector is K*N^2 long. The first three fix that
# product at 128, which makes them the same size to within 0.5% -- any
# difference between them is inductive bias, not capacity. "full" deliberately
# breaks that to probe whether the 128-value bottleneck costs anything; note it
# widens both the flatten AND the hidden layer, so a win there would need a
# follow-up to attribute.
#     head       K     N     values   model (deep6, 4-band)
#     gap        128   1x1   128        228,609
#     mixed      8     4x4   128        229,657
#     spatial    2     8x8   128        228,871
#     full       128   8x8   8,192    1,276,401
HEADS = {
    "gap":     {"head": "gap",   "head_reduce": None, "fc_layers": [8]},
    "mixed":   {"head": "pool4", "head_reduce": 8,    "fc_layers": [8]},
    "spatial": {"head": "pool8", "head_reduce": 2,    "fc_layers": [8]},
    "full":    {"head": "pool8", "head_reduce": None, "fc_layers": [128]},
}
LRS = [1e-3, 3e-4]
OPTIMIZERS = ["adam", "adamw"]

# FROZEN. The v1 grid's index arithmetic (12 configs per band set would be
# wrong: it is 8 = 2 archs x 2 lrs x 2 optimizers) is recorded in slurm scripts
# and in the finalist config list, so GRID must be built from an explicit arch
# list rather than from CHANNEL_SETS -- otherwise adding an architecture such as
# deep6 silently renumbers every existing index.
V1_ARCHS = ["small", "wide"]

GRID = [
    {"bands": b, "arch": a, "lr": lr, "optimizer": opt}
    for b, a, lr, opt in itertools.product(BAND_SETS, V1_ARCHS, LRS, OPTIMIZERS)
]


# The bands x heads sweep. Architecture, lr and optimizer are fixed at values
# already measured twice and found null, so the only axes left are the two open
# questions: which bands, and how the head trades channel context against
# spatial context. Kept separate from GRID so that grid's indices -- and the
# finalist config list recorded against them -- stay frozen.
BANDHEAD_BANDS = ["rgb", "rgb+nir", "rgb+swir16", "rgb+swir22"]
BANDHEAD_GRID = [
    {"bands": b, "head": h, "arch": "deep6", "lr": 1e-3, "optimizer": "adamw"}
    for b, h in itertools.product(BANDHEAD_BANDS, HEADS)
]

# The annealing-horizon sweep. The bands x heads sweep answered its own
# question and raised a new one: median best_epoch was 30 and the maximum over
# all 80 folds was 67, so nothing came close to the 200-epoch budget. Because
# CosineAnnealingLR takes T_max from --epochs, every one of those checkpoints
# was found at essentially full learning rate, and the annealing that was
# supposed to settle the tail did its work long after the model had stopped
# improving. --epochs is therefore not just a budget here, it is the shape of
# the learning-rate schedule, and it belongs on an axis.
#
# THE HEAD RIDES ALONG DELIBERATELY. gap/mixed/spatial finished within 0.0010
# of each other at 200 epochs -- a tie -- and a tie decided under one schedule
# does not transfer to another. Selecting the head at 200 and then deploying it
# at 60 would repeat the resolution mistake: a config chosen under conditions
# it is not run under. Both are settled here, on dev lakes, together.
#
# "full" is dropped: it lost to all three matched heads (-0.0024 to -0.0038,
# 16-17 folds of 20) with 2.6x their fold spread. Bands are fixed at
# rgb+swir16, which beat rgb by +0.0058 winning 20/20 folds; rgb+swir22 is
# statistically interchangeable with it (+0.0006, 16/20).
EPOCH_HORIZONS = [40, 60, 80, 120, 200]
EPOCHS_HEADS = ["gap", "mixed", "spatial"]
EPOCHS_GRID = [
    {"bands": "rgb+swir16", "head": h, "arch": "deep6", "lr": 1e-3,
     "optimizer": "adamw", "epochs": e}
    for h, e in itertools.product(EPOCHS_HEADS, EPOCH_HORIZONS)
]

GRIDS = {"v1": GRID, "bandhead": BANDHEAD_GRID, "epochs": EPOCHS_GRID}


def config_name(cfg: dict, head: str = "gap") -> str:
    """Result-file and wandb name. The head suffix is omitted for "gap" so that
    names from earlier grids, which predate the head axis, still match."""
    if "head" in cfg:                      # bands x heads sweep
        base = f"{cfg['bands']}_{cfg['head']}"
        # Only the epochs sweep carries an "epochs" key, so bandhead names are
        # unchanged -- its 80 finished result files keep matching.
        return base if "epochs" not in cfg else f"{base}_e{cfg['epochs']}"
    base = f"{cfg['bands']}_{cfg['arch']}_lr{cfg['lr']:g}_{cfg['optimizer']}"
    return base if head == "gap" else f"{base}_{head}"


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

    # A config from BANDHEAD_GRID names its head; one from GRID predates the
    # head axis and takes it from the CLI. Resolved before wandb.init, which
    # logs it.
    hcfg = HEADS[cfg["head"]] if "head" in cfg else {
        "head": args.head, "head_reduce": args.head_reduce,
        "fc_layers": args.fc_layers,
    }
    # Only the epochs sweep puts this in the config; every other grid takes the
    # CLI value. It sets BOTH the loop bound and the cosine T_max -- they must
    # stay the same number, since the horizon is what is being compared.
    epochs = int(cfg.get("epochs", args.epochs))

    # One wandb run per (config, fold), grouped by config so the UI averages
    # folds natively. Compute nodes have no internet: WANDB_MODE=offline is set
    # by the SLURM wrapper, and runs are synced from a login node afterwards.
    run = None
    if WANDB_AVAILABLE and args.wandb_project:
        run = wandb.init(
            project=args.wandb_project,
            name=f"{config_name(cfg, args.head)}_fold{fold}",
            group=config_name(cfg, args.head),
            job_type="cv",
            tags=[cfg["bands"], cfg["arch"], cfg["optimizer"],
                  f"head:{cfg.get('head', args.head)}", f"fold{fold}"],
            config={**cfg, "fold": fold, "img_size": args.img_size,
                    "epochs": epochs, "batch_size": args.batch_size,
                    "weight_decay": args.weight_decay, "seed": args.seed,
                    "lr_schedule": args.lr_schedule,
                    **{f"head_{k}" if k != "head" else "head_spec": v
                       for k, v in hcfg.items()},
                    "augment": not args.no_augment,
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
        train_ds = CloudyTileDatasetNC(splits["train"],
                                      augment=not args.no_augment, **common)
        val_ds = CloudyTileDatasetNC(splits["val"], **common)
        test_ds = CloudyTileDatasetNC(splits["test"], **common)

        # The dataset drops requested bands that a .nc file does not carry, so
        # a typo or a re-extracted tile set would quietly train a config on
        # fewer bands than its name claims -- and the band axis is the whole
        # point of this sweep. in_channels below is taken from BAND_SETS, so a
        # mismatch would also surface as a shape error mid-forward; fail here
        # instead, where the message says which band went missing.
        for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            if list(ds.channels) != list(channels):
                raise ValueError(
                    f"{name} split resolved to {ds.channels}, but config "
                    f"{cfg['bands']!r} asks for {channels}. Missing: "
                    f"{sorted(set(channels) - set(ds.channels))}"
                )

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
            head=hcfg["head"],
            head_reduce=hcfg["head_reduce"],
            fc_layers=hcfg["fc_layers"],
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
                optimizer, T_max=epochs)

        best_val, best_state, best_epoch = float("inf"), None, None
        fold_start = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = train_one_epoch(model, loaders["train"], optimizer,
                                         criterion, device)
            val_loss, val_metrics = evaluate(model, loaders["val"], criterion, device)
            if scheduler is not None:
                scheduler.step()
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch + 1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            secs = time.time() - epoch_start
            print(f"  [{config_name(cfg, args.head)} fold {fold}] epoch {epoch+1}/{epochs} "
                  f"train {train_loss:.4f} val {val_loss:.4f} "
                  f"acc {val_metrics['accuracy']:.3f} {secs:.1f}s")
            # A task that will blow its walltime is knowable from the first
            # epoch, not from the log 30 hours later. Project the whole task
            # (remaining folds included) while there is still time to requeue.
            if epoch == 0:
                per_fold = secs * epochs / 3600
                print(f"  [{config_name(cfg, args.head)} fold {fold}] "
                      f"projected {per_fold:.1f} h/fold, "
                      f"{per_fold * args.folds:.1f} h for all {args.folds} folds")
            sys.stdout.flush()
            if run is not None:
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss,
                           "val_loss": val_loss,
                           "lr": optimizer.param_groups[0]["lr"],
                           **{f"val_{k}": v for k, v in val_metrics.items()}})

        # No finite val loss in any epoch means training diverged; say so,
        # rather than failing inside load_state_dict(None) several lines later
        # with an error that names neither the config nor the cause.
        if best_state is None:
            raise RuntimeError(
                f"{config_name(cfg, args.head)} fold {fold}: no finite "
                f"validation loss in {epochs} epochs — training diverged"
            )

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
        "config_name": config_name(cfg, args.head),
        "head": cfg.get("head", args.head),
        "head_spec": hcfg,
        "augment": not args.no_augment,
        "fold": fold,
        "epochs": epochs,
        "lr_schedule": args.lr_schedule,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "elapsed_sec": round(time.time() - fold_start, 1),
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
    p.add_argument("--grid", type=str, default="v1", choices=sorted(GRIDS),
                   help="'v1' is the original bands x width x lr x optimizer "
                        "product (indices frozen); 'bandhead' is the 4 bands x "
                        "4 heads sweep at fixed architecture.")
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
    p.add_argument("--head", type=str, default="gap",
                   help="Classifier head: 'gap' (one number per channel, no "
                        "spatial layout), 'pool<N>' (NxN grid per channel, so "
                        "coarse spatial structure reaches the classifier), or "
                        "'flatten' (legacy, resolution-dependent). Pair "
                        "pool<N> with a narrow --fc_layers.")
    p.add_argument("--head_reduce", type=int, default=None,
                   help="With pool<N>: collapse the conv stack's channels to "
                        "this many with a 1x1 conv BEFORE pooling. 1 gives a "
                        "single NxN usability map (256 values at N=16) instead "
                        "of N*N per channel (16,384), which is what keeps the "
                        "flattened vector — and the dense layer — small.")
    p.add_argument("--fc_layers", type=int, nargs="+", default=[128],
                   help="Hidden widths of the classifier MLP (default 128)")
    p.add_argument("--no_augment", action="store_true",
                   help="Disable train-time flips/rot90")
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

    grid = GRIDS[args.grid]
    # A SLURM --array wider than the grid would otherwise die on a bare
    # IndexError in every surplus task, with nothing in the message naming the
    # mismatch between the array range and the grid size.
    if args.config_index >= len(grid):
        p.error(f"--config_index {args.config_index} is out of range for grid "
                f"'{args.grid}' ({len(grid)} configs, valid 0..{len(grid)-1}). "
                f"Check the SLURM --array range.")
    configs = grid if args.config_index < 0 else [grid[args.config_index]]
    print(f"grid '{args.grid}': {len(grid)} configs; running {len(configs)} "
          f"x {args.folds} folds at {args.img_size}px")
    for c in configs:
        print(f"  -> {config_name(c, args.head)}  {c}")

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
            out_path = out_dir / f"{config_name(cfg, args.head)}_fold{fold}.json"
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
