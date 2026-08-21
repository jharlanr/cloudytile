"""
Training script for CloudyTileCNN with wandb integration.

Usage:
    python run_training.py --labels_csv ../../labels.csv --image_dir /path/to/images

For wandb sweeps:
    wandb sweep sweep.yaml
    wandb agent <sweep_id>
"""
import argparse
import ast
import sys
import tempfile
from pathlib import Path


def _tokens(value):
    """
    Normalize the three shapes a list argument arrives in to one token string.

    These flags are written three different ways and all three have to work:
      --channels 16 32 64      argparse nargs -> ['16','32','64']  (SLURM
                               scripts, and how run_cv_grid.py spells it)
      --channels '16 32 64'    one shell word -> '16 32 64'
      channels: '[16,32,64]'   a wandb sweep  -> '[16,32,64]'
    Until now only the last two parsed; the first raised "unrecognized
    arguments: 32 64", which is exactly how run_final_model.sh was written.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def parse_list(value):
    """Ints from '[1,2,3]', '1 2 3', ['1','2','3'], or [1,2,3]."""
    value = _tokens(value)
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [int(x) for x in value.split()]


def parse_string_list(value):
    """Strings from "['a','b']", 'a b', or ['a','b']."""
    value = _tokens(value)
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return value.split()

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudytile.data import CloudyTileDataset, CloudyTileDatasetNC, create_splits
from cloudytile.model import CloudyTileCNN
from cloudytile.splits import (assert_lake_disjoint, create_lake_splits,
                               load_frozen_split)
from cloudytile.training import train_one_epoch, evaluate, predict_probs, pick_threshold


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, logging disabled")


VALID_METRICS = ["accuracy", "precision", "recall", "f1", "auc", "loss"]


def train(config: dict):
    """Main training function."""
    # Set seed for reproducible weight initialization
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate optimization metric
    opt_metric = config.get("optimize_metric", "precision")
    if opt_metric not in VALID_METRICS:
        raise ValueError(f"optimize_metric must be one of {VALID_METRICS}, got {opt_metric}")

    print(f"Optimizing for: {opt_metric}")

    # Create data splits. 'lake' groups by lake_id so a lake cannot appear in
    # two splits; 'tile' is the original behavior and leaks near-duplicate
    # frames of the same lake across the split — kept only to reproduce old runs.
    split_by = config.get("split_by", "lake")
    split_kwargs = dict(
        train_ratio=config.get("train_ratio", 0.8),
        val_ratio=config.get("val_ratio", 0.1),
        test_ratio=config.get("test_ratio", 0.1),
        seed=config.get("seed", 42),
        output_dir=config.get("splits_dir"),
    )
    if config.get("split_dir"):
        # Frozen lake-ID lists: the reproducible path, and the only one that
        # keeps the test lakes fixed as labels grow.
        train_df, val_df, test_df = load_frozen_split(
            config["split_dir"], config["labels_csv"]
        )
        assert_lake_disjoint(train_df, val_df, test_df)
    elif split_by == "lake":
        train_df, val_df, test_df = create_lake_splits(
            config["labels_csv"], **split_kwargs
        )
        assert_lake_disjoint(train_df, val_df, test_df)
    elif split_by == "tile":
        print("WARNING: --split_by tile shares lakes across splits; "
              "test metrics will be optimistic.")
        train_df, val_df, test_df = create_splits(
            config["labels_csv"], **split_kwargs
        )
    else:
        raise ValueError(f"--split_by must be 'lake' or 'tile', got {split_by!r}")

    # Save splits temporarily for Dataset to read
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        train_df.to_csv(tmpdir / "train.csv", index=False)
        val_df.to_csv(tmpdir / "val.csv", index=False)

        # Create datasets. NC is the primary path: JPGs exist only for
        # labeling, training reads the per-tile 6-band .nc files.
        img_size = (config.get("img_size", 512), config.get("img_size", 512))
        use_nc = config.get("nc_dir") is not None

        if use_nc:
            # Multi-spectral mode: load from NC files
            nc_dir = config["nc_dir"]
            channels = config.get("nc_channels") or \
                ["red", "green", "blue", "nir", "swir16", "swir22"]
            # Auto-detect in_channels from channel list if not explicitly set
            in_channels = config.get("in_channels") or len(channels)
            # Create a readable band combo string for wandb grouping
            band_combo = "+".join(channels)
            print(f"Using {in_channels} channels: {channels}")

            # Log band info to wandb for easy filtering
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.config.update({
                    "band_combo": band_combo,
                    "n_channels": in_channels,
                    "has_nir": "nir" in channels,
                    "has_swir16": "swir16" in channels,
                    "has_swir22": "swir22" in channels,
                }, allow_val_change=True)

            # Get band statistics path for normalization
            band_stats = config.get("band_stats")
            if band_stats:
                print(f"Using band statistics from: {band_stats}")

            train_dataset = CloudyTileDatasetNC(
                tmpdir / "train.csv",
                nc_dir,
                channels=channels,
                img_size=img_size,
                band_stats=band_stats,
                augment=config.get("augment", True),
            )
            val_dataset = CloudyTileDatasetNC(
                tmpdir / "val.csv",
                nc_dir,
                channels=channels,
                img_size=img_size,
                band_stats=band_stats,
            )
        else:
            # Legacy RGB mode: load from JPG files
            in_channels = config.get("in_channels") or 3
            train_dataset = CloudyTileDataset(
                tmpdir / "train.csv",
                config["image_dir"],
                img_size=img_size,
            )
            val_dataset = CloudyTileDataset(
                tmpdir / "val.csv",
                config["image_dir"],
                img_size=img_size,
            )

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
        )

        # Create model
        model = CloudyTileCNN(
            img_size=img_size,
            channels=config.get("channels", [16, 32, 64]),
            fc_layers=config.get("fc_layers", [128]),
            in_channels=in_channels,
            head=config.get("head", "gap"),
            head_reduce=config.get("head_reduce"),
            batch_norm=not config.get("no_batchnorm", False),
            dropout=config.get("dropout", 0.3),
        ).to(device)
        print(f"Model: head={model.head}, {model.n_parameters():,} parameters")

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        epochs = config.get("epochs", 100)

        # Must be able to reproduce the optimizer the config grid selected
        # under; AdamW was the grid's marginal winner and is what the finalist
        # configs use.
        opt_name = config.get("optimizer", "adam")
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[opt_name]
        optimizer = opt_cls(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )

        # Learning-rate schedule. Model selection happens under cosine annealing
        # to ~0 across the full run, so the final model must train in the same
        # regime it was chosen under. Annealing does two things: it improves the
        # score outright (~0.008 AUC on the finalist reruns, comparable to the
        # entire band effect), and it settles the tail of the validation curve —
        # epoch-to-epoch swing dropped from 0.0136 to 0.0018 — so that picking
        # the best checkpoint by argmin is a real choice rather than a lottery
        # over noise. "plateau" preserves the older --use_scheduler behaviour.
        lr_schedule = config.get("lr_schedule", "none")
        if lr_schedule == "none" and config.get("use_scheduler", False):
            lr_schedule = "plateau"
        scheduler = None
        if lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        elif lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

        # Training loop
        # For loss, lower is better; for other metrics, higher is better
        best_metric_value = float("inf") if opt_metric == "loss" else 0.0
        best_state = None
        best_epoch = None

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

            if scheduler is not None:
                # CosineAnnealingLR.step() takes no argument; ReduceLROnPlateau
                # needs the metric it is monitoring.
                if lr_schedule == "cosine":
                    scheduler.step()
                else:
                    scheduler.step(val_loss)

            # Logging
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "lr": optimizer.param_groups[0]["lr"],
            }

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc: {val_metrics['accuracy']:.3f} | "
                f"Prec: {val_metrics['precision']:.3f} | "
                f"F1: {val_metrics['f1']:.3f}"
            )

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(log_dict)

            # Save best model based on chosen metric
            if opt_metric == "loss":
                current_metric = val_loss
                is_better = current_metric < best_metric_value
            else:
                current_metric = val_metrics[opt_metric]
                is_better = current_metric > best_metric_value

            if is_better:
                best_metric_value = current_metric
                best_epoch = epoch + 1
                # Held on CPU so the test evaluation below scores these weights
                # rather than whatever the last epoch happened to land on.
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                if config.get("save_path"):
                    torch.save(best_state, config["save_path"])
                    print(f"  Saved best model ({opt_metric}={current_metric:.4f})")

        # Restore the best checkpoint before testing. Without this the reported
        # test metrics describe the final epoch while the saved .pth holds the
        # best-validation epoch, so the numbers never described the weights.
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"\nRestored best checkpoint (epoch {best_epoch}, "
                  f"{opt_metric}={best_metric_value:.4f}) for test evaluation")
        else:
            print("\nWARNING: no epoch improved on the initial value; "
                  "testing final-epoch weights")

        # Final test evaluation
        test_df.to_csv(tmpdir / "test.csv", index=False)
        if use_nc:
            test_dataset = CloudyTileDatasetNC(
                tmpdir / "test.csv",
                nc_dir,
                channels=channels,
                img_size=img_size,
                band_stats=band_stats,
            )
        else:
            test_dataset = CloudyTileDataset(
                tmpdir / "test.csv",
                config["image_dir"],
                img_size=img_size,
            )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=config.get("num_workers", 4),
        )

        # Choose the operating point on VALIDATION, then apply it to test.
        # 0.5 is only optimal when the two error types cost the same; letting a
        # cloudy frame through corrupts a downstream drainage call, while
        # dropping a clear frame costs one observation out of ~90 per lake.
        threshold = 0.5
        if config.get("tune_threshold", True):
            _, val_labels, val_probs = predict_probs(
                model, val_loader, criterion, device
            )
            threshold, val_at_t = pick_threshold(
                val_labels, val_probs,
                objective=config.get("threshold_objective", "f1"),
                target_precision=config.get("target_precision", 0.95),
            )
            print(f"\nOperating point from validation: threshold={threshold:.3f} "
                  f"(val precision={val_at_t['precision']:.3f} "
                  f"recall={val_at_t['recall']:.3f} f1={val_at_t['f1']:.3f})")

        test_loss, test_metrics = evaluate(model, test_loader, criterion, device,
                                           threshold=threshold)
        print(f"\nTest Results (threshold={threshold:.3f}):")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1:        {test_metrics['f1']:.4f}")
        print(f"  AUC:       {test_metrics['auc']:.4f}")

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "test_loss": test_loss,
                "test_acc": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_auc": test_metrics["auc"],
            })
            wandb.summary[f"best_val_{opt_metric}"] = best_metric_value
            wandb.summary["best_epoch"] = best_epoch
            wandb.summary["test_acc"] = test_metrics["accuracy"]
            wandb.summary["test_precision"] = test_metrics["precision"]
            wandb.summary["test_f1"] = test_metrics["f1"]
            wandb.summary["test_auc"] = test_metrics["auc"]

            # Log misclassified test samples
            try:
                _log_misclassified(model, test_dataset, device)
            except Exception as e:
                print(f"  Warning: Failed to log misclassified samples: {e}")

    return best_metric_value, test_metrics


def _log_misclassified(model, dataset, device):
    """Log misclassified samples to wandb as a table (filenames only).

    Args:
        model: Trained model
        dataset: Dataset to evaluate
        device: torch device
    """
    model.eval()
    columns = ["filename", "prediction", "label", "probability"]

    misclassified = []
    filenames = dataset.filenames

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            image_batch = image.unsqueeze(0).to(device)

            output = model(image_batch)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

            if pred != label:
                misclassified.append((filenames[idx], pred, label, prob))

    print(f"  Found {len(misclassified)} misclassified samples")

    if len(misclassified) == 0:
        return

    # Sort by confidence (most confident mistakes first)
    misclassified.sort(key=lambda x: x[3] if x[1] == 1 else (1 - x[3]), reverse=True)

    # Log to wandb table
    table = wandb.Table(columns=columns)
    for filename, pred, label, prob in misclassified:
        table.add_data(filename, pred, label, prob)

    wandb.log({"misclassified_samples": table})

    # Also print to stdout for the .out file
    print(f"\n  Misclassified samples:")
    print(f"  {'Filename':<50} {'Pred':>5} {'Label':>6} {'Prob':>8}")
    print(f"  {'-'*50} {'-'*5} {'-'*6} {'-'*8}")
    for filename, pred, label, prob in misclassified:
        print(f"  {filename:<50} {pred:>5} {label:>6} {prob:>8.4f}")

    print(f"\n  Logged {len(misclassified)} misclassified samples to wandb")


def build_parser() -> argparse.ArgumentParser:
    """
    The CLI, separated from main() so tests can assert the contract that binds
    this script to run_cv_grid.py: every config the grid can SELECT, this
    script must be able to BUILD. That contract has now been broken twice --
    once by --optimizer (the grid chose adamw while this script hardcoded Adam)
    and once by --head_reduce (the grid's 'mixed'/'spatial' heads were
    unbuildable here) -- both times silently, because nothing compared the two
    surfaces. test_selection_regime_is_reproducible does now.
    """
    parser = argparse.ArgumentParser(description="Train CloudyTileCNN")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="Path to labels CSV")
    parser.add_argument("--nc_dir", type=str, default=None,
                        help="Directory of per-tile .nc files (the standard "
                             "path; see extract_training_nc.py)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of JPG images (legacy RGB path; used "
                             "only if --nc_dir is not given)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--channels", type=str, nargs="+", default=[16, 32, 64],
                        help="Conv channel sizes (e.g., '16 32 64' or '[16,32,64]')")
    parser.add_argument("--fc_layers", type=str, nargs="+", default=[128],
                        help="FC layer sizes (e.g., '128' or '[128,64]')")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw"],
                        help="AdamW's decoupled weight decay pairs better with "
                             "BatchNorm and is what the config grid selected")
    parser.add_argument("--lr_schedule", type=str, default="none",
                        choices=["none", "cosine", "plateau"],
                        help="'cosine' anneals lr to ~0 over --epochs; use it to "
                             "match the regime the config grid selected under")
    parser.add_argument("--use_scheduler", type=lambda x: x.lower() == 'true',
                        default=False,
                        help="Deprecated alias for --lr_schedule plateau")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_by", type=str, default="lake",
                        choices=["lake", "tile"],
                        help="Group splits by lake_id (default) or split "
                             "individual tiles (legacy; leaks lakes)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save best model weights")
    parser.add_argument("--optimize_metric", type=str, default="precision",
                        choices=VALID_METRICS,
                        help="Metric to optimize for model selection (default: precision)")
    parser.add_argument("--wandb_project", type=str, default="cloudy-tile")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--nc_channels", type=str, nargs="+", default=None,
                        help="Channels to load from NC files (default: all six "
                             "SDR bands red green blue nir swir16 swir22)")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Number of input channels (auto-detected if not set)")
    parser.add_argument("--band_stats", type=str, default=None,
                        help="Path to JSON file with per-band normalization statistics")
    parser.add_argument("--head", type=str, default="gap",
                        help="Classifier head: 'gap', 'pool<N>' (NxN grid per "
                             "channel), or 'flatten' (legacy checkpoints only). "
                             "Not a fixed choice list: the CV grid selects over "
                             "pool<N> heads, and this script has to be able to "
                             "train whichever one wins.")
    parser.add_argument("--head_reduce", type=int, default=None,
                        help="With pool<N>: collapse the conv stack's channels "
                             "to this many with a 1x1 conv BEFORE pooling. This "
                             "is not a tweak — it defines the head. The grid's "
                             "'mixed' head is pool4 + head_reduce 8 and "
                             "'spatial' is pool8 + head_reduce 2; omitting it "
                             "builds a far larger model under the same name.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_batchnorm", action="store_true")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable train-time flips/rotations")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Frozen split directory from engine/make_splits.py "
                             "(overrides --split_by; the reproducible path)")
    parser.add_argument("--no_tune_threshold", action="store_true",
                        help="Report test metrics at a fixed 0.5 instead of at "
                             "an operating point chosen on validation")
    parser.add_argument("--threshold_objective", type=str, default="f1",
                        choices=["f1", "target_precision"],
                        help="'target_precision' maximizes recall subject to a "
                             "precision floor — use when false positives are "
                             "costlier than false negatives")
    parser.add_argument("--target_precision", type=float, default=0.95)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    # nargs="+" gives a list of raw tokens; normalize to typed lists here so
    # every spelling in _tokens() converges on the same value.
    args.channels = parse_list(args.channels)
    args.fc_layers = parse_list(args.fc_layers)
    args.nc_channels = parse_string_list(args.nc_channels)
    args.augment = not args.no_augment
    args.tune_threshold = not args.no_tune_threshold

    if args.nc_dir is None and args.image_dir is None:
        parser.error("provide --nc_dir (standard) or --image_dir (legacy JPG)")

    config = vars(args)

    # Initialize wandb
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
        )
        # Allow sweep to override config. A sweep supplies these as strings
        # ('[16,32,64]'), so re-normalize -- otherwise a swept architecture
        # reaches the model as text.
        config = dict(wandb.config)
        config["channels"] = parse_list(config.get("channels"))
        config["fc_layers"] = parse_list(config.get("fc_layers"))
        config["nc_channels"] = parse_string_list(config.get("nc_channels"))
        # Ensure required paths are set
        config["labels_csv"] = args.labels_csv
        config["image_dir"] = args.image_dir
        config["nc_dir"] = args.nc_dir

    train(config)

    if WANDB_AVAILABLE and wandb.run is not None:
        # Print run directory for easy syncing (shows in .out file)
        # wandb.run.dir ends in /files, but sync needs the parent directory
        run_dir = Path(wandb.run.dir).parent
        print(f"\n{'='*60}")
        print(f"WANDB RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Run ID: {wandb.run.id}")
        print(f"Run directory: {run_dir}")
        print(f"\nTo sync this run to wandb.ai, use:")
        print(f"  wandb sync {run_dir}")
        print(f"{'='*60}\n")
        wandb.finish()


if __name__ == "__main__":
    main()
