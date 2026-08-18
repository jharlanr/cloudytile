"""
Lake-grouped train/val/test splitting.

Tiles from one lake at nearby timesteps are near-duplicate scenes, so splitting
at the tile level leaks a lake across the split: with the original 80/10/10
tile split on the 2,090-row labels_deprecated.csv, all 52 test lakes appeared in
train and nothing was genuinely held out. Every function here splits on
lake_id, so a lake lands wholly inside one split.

This supersedes data.py::create_splits for any reported number. That function
is left in place unchanged so older runs stay reproducible.
"""
import json
import re
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd

# '{lake_id}_t{timestep:03d}.jpg' — the filename contract from preprocessing.py
_TILE_RE = re.compile(r"^(?P<lake_id>.+)_t(?P<timestep>\d+)\.(jpg|nc)$")


def lake_id_from_filename(filename: str) -> str:
    """
    Extract lake_id from a tile filename.

    >>> lake_id_from_filename("CW2019_1530_t006.jpg")
    'CW2019_1530'
    """
    m = _TILE_RE.match(str(filename))
    if not m:
        raise ValueError(f"cannot parse lake_id from {filename!r}")
    return m.group("lake_id")


def add_lake_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a 'lake_id' column derived from 'filename'."""
    out = df.copy()
    out["lake_id"] = out["filename"].map(lake_id_from_filename)
    return out


def _summarize(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    frac = df["label"].mean() if n else float("nan")
    print(f"  {name:5s} {n:>5} tiles  {df['lake_id'].nunique():>4} lakes  "
          f"class 1: {100 * frac:.1f}%")


def create_lake_splits(
    labels_csv: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a labels CSV into lake-disjoint train/val/test sets.

    Lakes are shuffled and partitioned by lake count, so the resulting tile
    proportions only approximate the requested ratios — exact tile ratios and
    intact lakes are not simultaneously achievable. Actual counts are printed.

    Args:
        labels_csv: CSV with 'filename' and 'label' columns
        train_ratio/val_ratio/test_ratio: must sum to 1.0
        seed: RNG seed for the lake shuffle
        output_dir: if given, write train.csv/val.csv/test.csv here

    Returns:
        (train_df, val_df, test_df), each with a 'lake_id' column added
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    df = add_lake_id(pd.read_csv(labels_csv))
    lakes = np.sort(df["lake_id"].unique())
    if len(lakes) < 3:
        raise ValueError(
            f"need at least 3 lakes to form three splits, found {len(lakes)}"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(lakes)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Guarantee every split gets at least one lake even at small n.
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    groups = {
        "train": set(shuffled[:n_train]),
        "val": set(shuffled[n_train:n_train + n_val]),
        "test": set(shuffled[n_train + n_val:]),
    }

    parts = tuple(
        df[df["lake_id"].isin(groups[k])].reset_index(drop=True)
        for k in ("train", "val", "test")
    )

    print(f"Lake-grouped split of {len(df)} tiles from {n} lakes (seed {seed}):")
    for name, part in zip(("train", "val", "test"), parts):
        _summarize(name, part)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, part in zip(("train", "val", "test"), parts):
            part.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"  Saved to {output_dir}/")

    return parts


def lake_group_kfold(
    labels_csv: Union[str, Path],
    n_splits: int = 5,
    seed: int = 42,
    stratify: bool = True,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """
    Yield (fold_index, train_df, test_df) for lake-grouped cross-validation.

    A single held-out split of ~40 lakes gives a noisy estimate — the original
    spectral sweep's 15 band combinations all landed within three test tiles of
    each other. Cross-validation is what turns that into a number with a spread
    attached.

    Args:
        labels_csv: CSV with 'filename' and 'label' columns
        n_splits: number of folds
        seed: RNG seed (shuffling only applies when stratify=False)
        stratify: balance label proportions across folds while keeping lakes
            intact (sklearn StratifiedGroupKFold); if False, plain GroupKFold
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    df = add_lake_id(pd.read_csv(labels_csv))
    groups = df["lake_id"].values
    y = df["label"].values

    if stratify:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    else:
        splitter = GroupKFold(n_splits=n_splits)

    for i, (tr, te) in enumerate(splitter.split(df, y, groups)):
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        print(f"Fold {i}: ", end="")
        print(f"train {len(train_df)} tiles / {train_df['lake_id'].nunique()} lakes, "
              f"test {len(test_df)} tiles / {test_df['lake_id'].nunique()} lakes, "
              f"test class 1: {100 * test_df['label'].mean():.1f}%")
        yield i, train_df, test_df


def freeze_split(
    labels_csv: Union[str, Path],
    out_dir: Union[str, Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Write a split to disk as lake-ID lists, in the ESSD deposit's layout.

    A seed-derived split is only reproducible while labels.csv is byte-identical
    — add labels later and every partition silently moves. Freezing the lake IDs
    pins the test set for the life of the project, which is what lets numbers
    from different weeks be compared, and what a reader needs to reproduce them.

    Writes {train,val,test}_ids.json and split_meta.json into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te = create_lake_splits(
        labels_csv, train_ratio, val_ratio, test_ratio, seed=seed
    )
    assert_lake_disjoint(tr, va, te)

    ids = {
        "train": sorted(tr["lake_id"].unique().tolist()),
        "val": sorted(va["lake_id"].unique().tolist()),
        "test": sorted(te["lake_id"].unique().tolist()),
    }
    for name, lake_ids in ids.items():
        (out_dir / f"{name}_ids.json").write_text(json.dumps(lake_ids, indent=2))

    meta = {
        "labels_csv": str(labels_csv),
        "n_labels": int(sum(len(p) for p in (tr, va, te))),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "grouped_by": "lake_id",
        "n_lakes": {k: len(v) for k, v in ids.items()},
        "n_tiles": {"train": len(tr), "val": len(va), "test": len(te)},
        "positive_rate": {
            "train": float(tr["label"].mean()),
            "val": float(va["label"].mean()),
            "test": float(te["label"].mean()),
        },
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Froze split to {out_dir}/")
    return meta


def load_frozen_split(
    split_dir: Union[str, Path],
    labels_csv: Union[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a frozen split and slice a labels CSV by it.

    Labels for lakes absent from the CSV are simply missing; labels for lakes
    absent from the split raise, since that means the split predates the labels
    and silently dropping rows would be worse than stopping.
    """
    split_dir = Path(split_dir)
    df = add_lake_id(pd.read_csv(labels_csv))

    ids = {
        name: set(json.loads((split_dir / f"{name}_ids.json").read_text()))
        for name in ("train", "val", "test")
    }
    known = ids["train"] | ids["val"] | ids["test"]
    unassigned = set(df["lake_id"]) - known
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} lakes in {labels_csv} are not in {split_dir} "
            f"(e.g. {sorted(unassigned)[:3]}). Re-freeze the split, or the new "
            f"labels will be silently dropped."
        )

    parts = tuple(
        df[df["lake_id"].isin(ids[name])].reset_index(drop=True)
        for name in ("train", "val", "test")
    )
    assert_lake_disjoint(*parts)
    print(f"Frozen split from {split_dir}:")
    for name, part in zip(("train", "val", "test"), parts):
        _summarize(name, part)
    return parts


def dev_labels(
    split_dir: Union[str, Path],
    labels_csv: Union[str, Path],
) -> pd.DataFrame:
    """
    Train+val rows only — the development set for model selection.

    Hyperparameter search must never fold over the frozen test lakes: picking
    the best of N configs by test score is selection on the test set, and the
    winner's score is biased upward by roughly the spread across configs.
    """
    tr, va, _ = load_frozen_split(split_dir, labels_csv)
    return pd.concat([tr, va], ignore_index=True)


def assert_lake_disjoint(*parts: pd.DataFrame) -> None:
    """Raise if any lake appears in more than one of the given splits."""
    sets = [set(p["lake_id"]) for p in parts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise AssertionError(
                    f"splits {i} and {j} share {len(overlap)} lakes, "
                    f"e.g. {sorted(overlap)[:3]}"
                )


def filesize_baseline(
    labels_csv: Union[str, Path],
    image_dir: Union[str, Path],
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Accuracy of thresholding on JPG file size alone — the baseline to beat.

    JPEG size tracks image detail, and 'useful' largely means 'has visible
    texture', so this scores far above chance without reading a pixel: 88.3% on
    the original 2,090-tile labels.csv. Any reported CNN accuracy should be
    quoted against this, not against the 50% majority class.

    The threshold is fit on train_df and applied to test_df when both are
    given; otherwise it is fit and scored on the whole CSV (optimistic, but a
    single threshold over thousands of rows barely overfits).

    Returns:
        dict with 'threshold_bytes', 'accuracy', and 'n'
    """
    image_dir = Path(image_dir)

    def sized(df):
        rows = []
        for fn, lab in zip(df["filename"], df["label"]):
            p = image_dir / fn
            if p.exists():
                rows.append((p.stat().st_size, int(lab)))
        return rows

    if train_df is None or test_df is None:
        df = pd.read_csv(labels_csv)
        fit = score = sized(df)
    else:
        fit, score = sized(train_df), sized(test_df)

    if not fit or not score:
        raise ValueError("no images found on disk for the given labels")

    candidates = sorted({s for s, _ in fit})
    best_t, best_acc = candidates[0], 0.0
    for t in candidates:
        acc = sum(1 for s, y in fit if (s >= t) == (y == 1)) / len(fit)
        if acc > best_acc:
            best_t, best_acc = t, acc

    acc = sum(1 for s, y in score if (s >= best_t) == (y == 1)) / len(score)
    return {"threshold_bytes": best_t, "accuracy": acc, "n": len(score)}
