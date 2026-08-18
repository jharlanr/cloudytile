"""
Tests for frozen splits and threshold selection.

Frozen splits exist so the test lakes survive changes to labels.csv; the guard
that matters is that adding labels for an unknown lake raises instead of
silently dropping rows.
"""
import json

import pandas as pd
import pytest

from cloudytile.splits import (
    add_lake_id,
    assert_lake_disjoint,
    dev_labels,
    freeze_split,
    load_frozen_split,
)
from cloudytile.training import compute_metrics, pick_threshold


def make_labels(tmp_path, n_lakes=40, tiles_per_lake=10, seed=0, name="labels.csv"):
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n_lakes):
        lake = f"CW2019_{1500 + i}"
        p = rng.uniform(0.3, 0.9)
        for t in range(tiles_per_lake):
            rows.append({"filename": f"{lake}_t{t:03d}.jpg",
                         "label": int(rng.random() < p)})
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestFreezeAndLoad:
    def test_writes_expected_files(self, tmp_path):
        csv = make_labels(tmp_path)
        out = tmp_path / "split"
        freeze_split(csv, out, seed=1)
        for f in ("train_ids.json", "val_ids.json", "test_ids.json",
                  "split_meta.json"):
            assert (out / f).exists(), f

    def test_roundtrip_is_disjoint_and_complete(self, tmp_path):
        csv = make_labels(tmp_path)
        out = tmp_path / "split"
        freeze_split(csv, out, seed=1)
        tr, va, te = load_frozen_split(out, csv)
        assert_lake_disjoint(tr, va, te)
        assert len(tr) + len(va) + len(te) == len(pd.read_csv(csv))

    def test_split_is_stable_across_reloads(self, tmp_path):
        csv = make_labels(tmp_path)
        out = tmp_path / "split"
        freeze_split(csv, out, seed=1)
        a = load_frozen_split(out, csv)[2]["filename"].tolist()
        b = load_frozen_split(out, csv)[2]["filename"].tolist()
        assert a == b

    def test_survives_labels_growing_within_known_lakes(self, tmp_path):
        """More tiles for existing lakes must not move any lake."""
        csv = make_labels(tmp_path, n_lakes=40, tiles_per_lake=10)
        out = tmp_path / "split"
        freeze_split(csv, out, seed=1)
        test_lakes_before = set(load_frozen_split(out, csv)[2]["lake_id"])

        grown = make_labels(tmp_path, n_lakes=40, tiles_per_lake=20,
                            name="grown.csv")
        test_lakes_after = set(load_frozen_split(out, grown)[2]["lake_id"])
        assert test_lakes_before == test_lakes_after

    def test_raises_on_unknown_lake(self, tmp_path):
        """A split predating new lakes must fail loudly, not drop them."""
        csv = make_labels(tmp_path, n_lakes=40)
        out = tmp_path / "split"
        freeze_split(csv, out, seed=1)
        bigger = make_labels(tmp_path, n_lakes=50, name="bigger.csv")
        with pytest.raises(ValueError, match="not in"):
            load_frozen_split(out, bigger)

    def test_meta_records_shape(self, tmp_path):
        csv = make_labels(tmp_path, n_lakes=40)
        out = tmp_path / "split"
        meta = freeze_split(csv, out, train_ratio=0.7, val_ratio=0.1,
                            test_ratio=0.2, seed=1)
        assert sum(meta["n_lakes"].values()) == 40
        assert meta["n_lakes"]["test"] == 8
        assert json.loads((out / "split_meta.json").read_text()) == meta


class TestDevLabels:
    def test_excludes_test_lakes(self, tmp_path):
        csv = make_labels(tmp_path)
        out = tmp_path / "split"
        freeze_split(csv, out, seed=1)
        dev = dev_labels(out, csv)
        test_lakes = set(load_frozen_split(out, csv)[2]["lake_id"])
        assert not (set(dev["lake_id"]) & test_lakes)

    def test_covers_train_and_val(self, tmp_path):
        csv = make_labels(tmp_path)
        out = tmp_path / "split"
        freeze_split(csv, out, seed=1)
        tr, va, te = load_frozen_split(out, csv)
        assert len(dev_labels(out, csv)) == len(tr) + len(va)


class TestPickThreshold:
    def test_perfect_separation_finds_a_split_point(self):
        labels = [0] * 50 + [1] * 50
        probs = [0.1] * 50 + [0.9] * 50
        t, m = pick_threshold(labels, probs, objective="f1")
        assert 0.1 < t <= 0.9
        assert m["f1"] == 1.0

    def test_target_precision_is_respected_when_reachable(self):
        # top scores are pure positives, so high precision is reachable
        labels = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
        probs = ([0.95] * 40 + [0.6] * 10 + [0.55] * 10 + [0.1] * 40)
        t, m = pick_threshold(labels, probs, objective="target_precision",
                              target_precision=0.95)
        assert m["precision"] >= 0.95

    def test_target_precision_beats_f1_on_precision(self):
        labels = [1] * 60 + [0] * 40
        probs = [0.9] * 40 + [0.6] * 20 + [0.55] * 20 + [0.2] * 20
        _, m_f1 = pick_threshold(labels, probs, objective="f1")
        _, m_p = pick_threshold(labels, probs, objective="target_precision",
                                target_precision=0.99)
        assert m_p["precision"] >= m_f1["precision"]

    def test_returns_threshold_in_metrics(self):
        labels = [0, 1, 0, 1]
        probs = [0.2, 0.8, 0.3, 0.7]
        t, m = pick_threshold(labels, probs)
        assert m["threshold"] == t

    def test_rejects_unknown_objective(self):
        with pytest.raises(ValueError):
            pick_threshold([0, 1], [0.2, 0.8], objective="nonsense")

    def test_threshold_changes_predictions(self):
        labels = [0, 0, 1, 1]
        probs = [0.4, 0.45, 0.55, 0.6]
        lo = compute_metrics(labels, [float(p >= 0.3) for p in probs], probs)
        hi = compute_metrics(labels, [float(p >= 0.7) for p in probs], probs)
        assert lo["recall"] > hi["recall"]
