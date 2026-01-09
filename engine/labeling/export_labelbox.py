"""
Export labels from Labelbox and save as .csv for training.

Usage:
    export LABELBOX_API_KEY="your_key"

    # Create new export:
    python export_labelbox.py --output /path/to/labels.csv

    # Reuse existing export task:
    python export_labelbox.py --task_id <task_id> --output /path/to/labels.csv
"""
import argparse
import os
import labelbox
import pandas as pd
from pathlib import Path


def export_labels(api_key: str, project_id: str, task_id: str = None) -> pd.DataFrame:
    """
    Fetch labels from Labelbox and return as a DataFrame.

    Args:
        api_key: Labelbox API key
        project_id: Labelbox project ID
        task_id: Optional existing export task ID (skip creating new export)

    Returns:
        DataFrame with columns: filename, label
        where label is 1 (useful) or 0 (not useful)
    """
    client = labelbox.Client(api_key=api_key)

    if task_id:
        # Use existing export task
        print(f"  Fetching existing export task: {task_id}")
        export_task = labelbox.ExportTask.get_task(client, task_id)
    else:
        # Create a new export
        project = client.get_project(project_id)
        print("  Creating export task...")
        export_task = project.export(params={
            "data_row_details": True,
            "project_details": True,
            "label_details": True,
        })
        print(f"  Export task ID: {export_task.uid}")
        print("  Waiting for export to complete...")
        export_task.wait_till_done(timeout_seconds=1000)

        if export_task.has_errors():
            raise RuntimeError(f"Export failed: {export_task.errors}")

    print("  Fetching results...")

    # Use get_buffered_stream() to get the export results
    stream = export_task.get_buffered_stream()

    rows = []
    for item in stream:
        # Each item has a .json attribute containing the parsed dict
        rec = item.json

        filename = rec["data_row"]["external_id"]

        # Get label from the nested structure
        labels = rec["projects"][project_id]["labels"]
        if not labels:
            continue

        classifications = labels[0]["annotations"]["classifications"]
        if not classifications:
            continue

        answer = classifications[0]["radio_answer"]["name"]
        rows.append({
            "filename": filename,
            "label": 1 if answer == "yes" else 0,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Export labels from Labelbox")
    parser.add_argument(
        "--project_id",
        type=str,
        default="cmk5q6tey0hru07y8024t1lsb",
        help="Labelbox project ID",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help="Existing export task ID (skip creating new export)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="labels.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.environ.get("LABELBOX_API_KEY")
    if not api_key:
        raise ValueError("LABELBOX_API_KEY environment variable not set")

    print(f"Fetching labels from Labelbox project: {args.project_id}")

    df = export_labels(api_key, args.project_id, args.task_id)

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nExported {len(df)} labels to {output_path}")
    print(f"  Useful (1): {(df['label'] == 1).sum()}")
    print(f"  Not useful (0): {(df['label'] == 0).sum()}")


if __name__ == "__main__":
    main()
