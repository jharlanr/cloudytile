#!/usr/bin/env python
"""
Push a CV sweep's results to wandb from a login node.

Two things live in two places after a sweep, and both are worth having online:

  training curves   one offline wandb run per (config, fold), buffered to disk
                    because compute nodes have no internet. These carry the
                    per-epoch loss/lr traces.
  the ranking       the result JSONs, which carry the final scores, the regime,
                    and the fold structure -- everything --summarize prints.

`--sync` uploads the first. It reads wandb_run_dir out of each result JSON, so
it uploads exactly the runs this sweep produced. The alternative, a date glob
over the wandb directory, once matched 636 directories -- 540 of them from a
January campaign already deleted server-side -- and answered HTTP 410 for each
one, burying the wanted runs. Never guess by glob when the path was recorded.

`--table` uploads the second as a single run holding a wandb.Table of every
(config, fold) row plus the aggregated ranking, so the comparison is readable
in the UI without re-running --summarize. Safe to repeat: it overwrites the run
of the same name rather than accumulating duplicates.

Usage (login node, after `module load` and with wandb logged in):
    python3 engine/upload_wandb.py --out_dir <cv_results_dir> --sync --table
    python3 engine/upload_wandb.py --out_dir <dir> --sync --dry-run   # look first
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_rows(out_dir: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not rows:
        sys.exit(f"no result JSONs in {out_dir}")
    return pd.DataFrame(rows)


def sync(df: pd.DataFrame, dry_run: bool) -> None:
    if "wandb_run_dir" not in df.columns:
        print("These results predate wandb_run_dir, so the exact paths were "
              "not recorded. Fall back to a DATE-SCOPED glob, never a bare "
              "offline-run-*:\n"
              "    wandb sync <wandb_dir>/offline-run-<YYYYMMDD>_*")
        return
    dirs = sorted({d for d in df["wandb_run_dir"].dropna().unique() if d})
    missing = [d for d in dirs if not Path(d).exists()]
    dirs = [d for d in dirs if Path(d).exists()]
    print(f"{len(dirs)} offline runs to sync"
          + (f"; {len(missing)} recorded but no longer on disk" if missing else ""))
    if not dirs:
        return
    if dry_run:
        print("  (dry run) would run:")
        for d in dirs[:3]:
            print(f"    wandb sync {d}")
        if len(dirs) > 3:
            print(f"    ... and {len(dirs) - 3} more")
        return
    # One call, all paths: wandb sync accepts many, and a single invocation
    # avoids re-authenticating per run.
    r = subprocess.run(["wandb", "sync", *dirs])
    if r.returncode != 0:
        sys.exit(f"wandb sync exited {r.returncode}")
    print(f"synced {len(dirs)} runs")


def table(df: pd.DataFrame, project: str, name: str) -> None:
    try:
        import wandb
    except ImportError:
        sys.exit("wandb is not installed in this environment")
    from run_cv_grid import REGIME_KEYS

    regime = {k: df[k].iloc[0] for k in REGIME_KEYS if k in df.columns}
    if any(df[k].nunique() > 1 for k in regime):
        sys.exit("this directory mixes regimes; split it before uploading "
                 "(run --summarize to see which)")

    agg = (df.groupby("config_name")
             .agg(folds=("fold", "count"), auc_mean=("test_auc", "mean"),
                  auc_std=("test_auc", "std"), acc_mean=("test_accuracy", "mean"),
                  f1_mean=("test_f1", "mean"),
                  best_epoch_median=("best_epoch", "median"),
                  params=("n_parameters", "first"))
             .sort_values("auc_mean", ascending=False).reset_index())

    run = wandb.init(project=project, name=name, job_type="summary",
                     config={**regime, "n_configs": len(agg), "n_runs": len(df)},
                     id=name, resume="allow")     # same name -> overwrite, not duplicate
    keep = [c for c in ("config_name", "fold", "epochs", "best_epoch",
                        "best_val_loss", "test_auc", "test_accuracy", "test_f1",
                        "test_precision", "test_recall", "threshold",
                        "n_parameters", "elapsed_sec") if c in df.columns]
    wandb.log({"folds": wandb.Table(dataframe=df[keep]),
               "ranking": wandb.Table(dataframe=agg)})
    top = agg.iloc[0]
    for k in ("auc_mean", "auc_std", "acc_mean", "f1_mean"):
        wandb.summary[f"top_{k}"] = float(top[k])
    wandb.summary["top_config"] = top["config_name"]
    wandb.finish()
    print(f"uploaded ranking ({len(agg)} configs, {len(df)} runs) to "
          f"{project}/{name}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, help="a cv_results_* directory")
    p.add_argument("--project", default="cloudy-tile-cv")
    p.add_argument("--name", default=None,
                   help="run name for --table (default: the directory name)")
    p.add_argument("--sync", action="store_true",
                   help="upload the offline per-fold runs (training curves)")
    p.add_argument("--table", action="store_true",
                   help="upload the ranking as one summary run")
    p.add_argument("--dry-run", action="store_true",
                   help="with --sync, list what would be uploaded")
    args = p.parse_args()
    if not (args.sync or args.table):
        p.error("choose --sync, --table, or both")

    out_dir = Path(args.out_dir)
    df = load_rows(out_dir)
    print(f"{len(df)} result JSONs in {out_dir}")
    if args.sync:
        sync(df, args.dry_run)
    if args.table:
        table(df, args.project, args.name or out_dir.name)


if __name__ == "__main__":
    main()
