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


def parse_list(value):
    """Parse a list from string (handles both '[1,2,3]' and '1 2 3' formats)."""
    if isinstance(value, list):
        return value
    # Try to parse as Python literal (handles [1, 2, 3] format from wandb)
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    # Fall back to space-separated integers
    return [int(x) for x in value.split()]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloudytile.data import CloudyTileDataset, CloudyTileDatasetNC, create_splits
from cloudytile.model import CloudyTileCNN
from cloudytile.training import train_one_epoch, evaluate

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, logging disabled")


VALID_METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def train(config: dict):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate optimization metric
    opt_metric = config.get("optimize_metric", "precision")
    if opt_metric not in VALID_METRICS:
        raise ValueError(f"optimize_metric must be one of {VALID_METRICS}, got {opt_metric}")

    print(f"Optimizing for: {opt_metric}")

    # Create data splits
    train_df, val_df, test_df = create_splits(
        config["labels_csv"],
        train_ratio=config.get("train_ratio", 0.8),
        val_ratio=config.get("val_ratio", 0.1),
        test_ratio=config.get("test_ratio", 0.1),
        seed=config.get("seed", 42),
        output_dir=config.get("splits_dir"),
    )

    # Save splits temporarily for Dataset to read
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        train_df.to_csv(tmpdir / "train.csv", index=False)
        val_df.to_csv(tmpdir / "val.csv", index=False)

        # Create datasets
        img_size = (config.get("img_size", 512), config.get("img_size", 512))
        use_nc = config.get("use_nc", False)
        in_channels = config.get("in_channels", 6 if use_nc else 3)

        if use_nc:
            # Multi-spectral mode: load from NC files
            nc_dir = config.get("nc_dir", config["image_dir"])
            channels = config.get("nc_channels", ["red", "green", "blue", "nir", "swir1", "swir2"])
            train_dataset = CloudyTileDatasetNC(
                tmpdir / "train.csv",
                nc_dir,
                channels=channels,
                img_size=img_size,
            )
            val_dataset = CloudyTileDatasetNC(
                tmpdir / "val.csv",
                nc_dir,
                channels=channels,
                img_size=img_size,
            )
        else:
            # Legacy RGB mode: load from JPG files
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
        ).to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )

        # Learning rate scheduler
        scheduler = None
        if config.get("use_scheduler", False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

        # Training loop
        best_metric_value = 0.0
        epochs = config.get("epochs", 20)

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

            if scheduler:
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
            current_metric = val_metrics[opt_metric]
            if current_metric > best_metric_value:
                best_metric_value = current_metric
                if config.get("save_path"):
                    torch.save(model.state_dict(), config["save_path"])
                    print(f"  Saved best model ({opt_metric}={current_metric:.4f})")

        # Final test evaluation
        test_df.to_csv(tmpdir / "test.csv", index=False)
        if use_nc:
            test_dataset = CloudyTileDatasetNC(
                tmpdir / "test.csv",
                nc_dir,
                channels=channels,
                img_size=img_size,
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

        test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"\nTest Results:")
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


def main():
    parser = argparse.ArgumentParser(description="Train CloudyTileCNN")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="Path to labels CSV")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--channels", type=parse_list, default=[16, 32, 64],
                        help="Conv channel sizes (e.g., '16 32 64' or '[16,32,64]')")
    parser.add_argument("--fc_layers", type=parse_list, default=[128],
                        help="FC layer sizes (e.g., '128' or '[128,64]')")
    parser.add_argument("--use_scheduler", type=lambda x: x.lower() == 'true',
                        default=False, help="Use learning rate scheduler")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save best model weights")
    parser.add_argument("--optimize_metric", type=str, default="precision",
                        choices=VALID_METRICS,
                        help="Metric to optimize for model selection (default: precision)")
    parser.add_argument("--wandb_project", type=str, default="cloudy-tile")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--use_nc", action="store_true",
                        help="Use NetCDF files instead of JPGs (multi-spectral mode)")
    parser.add_argument("--nc_dir", type=str, default=None,
                        help="Directory containing NC files (defaults to image_dir)")
    parser.add_argument("--nc_channels", type=str, nargs="+",
                        default=["red", "green", "blue", "nir", "swir1", "swir2"],
                        help="Channels to load from NC files")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Number of input channels (auto-detected if not set)")
    args = parser.parse_args()

    config = vars(args)

    # Initialize wandb
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
        )
        # Allow sweep to override config
        config = dict(wandb.config)
        # Ensure required paths are set
        config["labels_csv"] = args.labels_csv
        config["image_dir"] = args.image_dir

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
