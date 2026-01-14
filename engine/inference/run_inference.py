#!/usr/bin/env python3
"""
Apply trained CloudyTileCNN model to NetCDF files.

Adds/overwrites the 'cloudy_seq' variable indicating tile usefulness.

Usage:
    # Single file
    python run_inference.py --model best_model.pth --input tile.nc

    # Directory of files
    python run_inference.py --model best_model.pth --input /path/to/nc_dir

    # With custom architecture (must match training)
    python run_inference.py --model best_model.pth --input tile.nc \
        --channels 32 64 128 --fc_layers 128
"""
import argparse
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloudytile.inference import load_model, add_cloudy_seq_to_nc, process_directory


def main():
    parser = argparse.ArgumentParser(
        description="Add cloudy_seq predictions to NetCDF files"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model weights (.pth file)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input NetCDF file or directory of NetCDF files"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: overwrite input). Only for single file."
    )
    parser.add_argument(
        "--img_size", type=int, default=512,
        help="Image size model was trained with (default: 512)"
    )
    parser.add_argument(
        "--channels", type=int, nargs="+", default=[16, 32, 64],
        help="Conv layer channels, must match training (default: 16 32 64)"
    )
    parser.add_argument(
        "--fc_layers", type=int, nargs="+", default=[128],
        help="FC layer sizes, must match training (default: 128)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--pattern", type=str, default="*.nc",
        help="Glob pattern for directory mode (default: *.nc)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    img_size = (args.img_size, args.img_size)

    if input_path.is_dir():
        # Directory mode
        if args.output is not None:
            print("Warning: --output ignored in directory mode (files overwritten in place)")

        process_directory(
            nc_dir=input_path,
            model_path=args.model,
            img_size=img_size,
            channels=args.channels,
            fc_layers=args.fc_layers,
            threshold=args.threshold,
            batch_size=args.batch_size,
            pattern=args.pattern,
        )

    elif input_path.is_file():
        # Single file mode
        print(f"Loading model from {args.model}")
        model = load_model(
            args.model,
            img_size=img_size,
            channels=args.channels,
            fc_layers=args.fc_layers,
        )

        output_path = Path(args.output) if args.output else input_path
        print(f"Processing {input_path}")

        add_cloudy_seq_to_nc(
            nc_path=input_path,
            model=model,
            img_size=img_size,
            threshold=args.threshold,
            batch_size=args.batch_size,
            output_path=output_path,
        )

        print(f"Saved to {output_path}")

    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
