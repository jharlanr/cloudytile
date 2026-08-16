"""
Tests for lake-grouped splitting.

The property that matters is disjointness: no lake may appear in two splits.
That is what the original tile-level split violated, and it is silent when
wrong — the run still trains and reports an inflated number.
"""
import pandas as pd
import pytest

from cloudytile.splits import (
    add_lake_id,
    assert_lake_disjoint,
    create_lake_splits,
    filesize_baseline,
    lake_group_kfold,
    lake_id_from_filename,
)


def make_labels(tmp_path, n_lakes=40, tiles_per_lake=25, seed=0):
    """A labels CSV shaped like the new extraction: many lakes, tiles each."""
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n_lakes):
        lake = f"CW2019_{1500 + i}"
        # per-lake cloudiness so labels correlate within a lake, as in reality
        p = rng.uniform(0.2, 0.8)
        for t in range(tiles_per_lake):
            rows.append({
                "filename": f"{lake}_t{t:03d}.jpg",
                "label": int(rng.random() < p),
            })
    path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestLakeIdParsing:
    def test_basic(self):
        assert lake_id_from_filename("CW2019_1530_t006.jpg") == "CW2019_1530"

    def test_nc_extension(self):
        assert lake_id_from_filename("CW2018_1077_t142.nc") == "CW2018_1077"

    def test_three_digit_timestep_not_greedy(self):
        # the lake id itself contains underscores and digits
        assert lake_id_from_filename("CW2019_2504_t151.jpg") == "CW2019_2504"

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            lake_id_from_filename("not_a_tile.png")

    def test_add_lake_id_column(self):
        df = pd.DataFrame({"filename": ["CW2019_1_t000.jpg"], "label": [1]})
        assert add_lake_id(df)["lake_id"].tolist() == ["CW2019_1"]


class TestLakeSplits:
    def test_no_lake_crosses_splits(self, tmp_path):
        tr, va, te = create_lake_splits(make_labels(tmp_path), seed=1)
        assert_lake_disjoint(tr, va, te)

    def test_every_tile_is_kept(self, tmp_path):
        path = make_labels(tmp_path, n_lakes=40, tiles_per_lake=25)
        tr, va, te = create_lake_splits(path, seed=1)
        assert len(tr) + len(va) + len(te) == 40 * 25

    def test_lakes_stay_whole(self, tmp_path):
        path = make_labels(tmp_path)
        parts = create_lake_splits(path, seed=1)
        full = add_lake_id(pd.read_csv(path))
        sizes = full.groupby("lake_id").size()
        for part in parts:
            for lake, n in part.groupby("lake_id").size().items():
                assert n == sizes[lake], f"{lake} was split across sets"

    def test_deterministic(self, tmp_path):
        path = make_labels(tmp_path)
        a = create_lake_splits(path, seed=7)[2]["filename"].tolist()
        b = create_lake_splits(path, seed=7)[2]["filename"].tolist()
        assert a == b

    def test_seed_changes_partition(self, tmp_path):
        path = make_labels(tmp_path)
        a = set(create_lake_splits(path, seed=1)[2]["lake_id"])
        b = set(create_lake_splits(path, seed=2)[2]["lake_id"])
        assert a != b

    def test_ratios_approximately_respected(self, tmp_path):
        path = make_labels(tmp_path, n_lakes=100, tiles_per_lake=25)
        tr, va, te = create_lake_splits(path, 0.8, 0.1, 0.1, seed=3)
        assert tr["lake_id"].nunique() == 80
        assert va["lake_id"].nunique() == 10
        assert te["lake_id"].nunique() == 10

    def test_all_splits_nonempty_with_few_lakes(self, tmp_path):
        path = make_labels(tmp_path, n_lakes=3, tiles_per_lake=4)
        for part in create_lake_splits(path, seed=1):
            assert len(part) > 0

    def test_rejects_too_few_lakes(self, tmp_path):
        path = make_labels(tmp_path, n_lakes=2, tiles_per_lake=4)
        with pytest.raises(ValueError):
            create_lake_splits(path, seed=1)

    def test_ratios_must_sum_to_one(self, tmp_path):
        with pytest.raises(AssertionError):
            create_lake_splits(make_labels(tmp_path), 0.8, 0.8, 0.8)


class TestRegressionAgainstTileSplit:
    """The bug this module exists to prevent."""

    def test_tile_level_split_would_leak(self, tmp_path):
        from sklearn.model_selection import train_test_split
        path = make_labels(tmp_path, n_lakes=40, tiles_per_lake=25)
        df = add_lake_id(pd.read_csv(path))
        tr, te = train_test_split(df, train_size=0.9, stratify=df["label"],
                                  random_state=42)
        leaked = set(te["lake_id"]) & set(tr["lake_id"])
        assert leaked, "expected the tile split to leak — it is the bug"

        # the grouped split must not
        gtr, gva, gte = create_lake_splits(path, seed=42)
        assert not (set(gte["lake_id"]) & set(gtr["lake_id"]))

    def test_assert_lake_disjoint_catches_overlap(self, tmp_path):
        df = add_lake_id(pd.read_csv(make_labels(tmp_path, n_lakes=4)))
        with pytest.raises(AssertionError):
            assert_lake_disjoint(df, df)


class TestKFold:
    def test_folds_are_disjoint_and_cover(self, tmp_path):
        path = make_labels(tmp_path, n_lakes=40, tiles_per_lake=25)
        seen, n_folds = set(), 0
        for i, tr, te in lake_group_kfold(path, n_splits=5, seed=1):
            assert_lake_disjoint(tr, te)
            seen |= set(te["lake_id"])
            n_folds += 1
        assert n_folds == 5
        assert seen == set(add_lake_id(pd.read_csv(path))["lake_id"])

    def test_unstratified_also_disjoint(self, tmp_path):
        path = make_labels(tmp_path, n_lakes=20, tiles_per_lake=10)
        for _, tr, te in lake_group_kfold(path, n_splits=4, stratify=False):
            assert_lake_disjoint(tr, te)


class TestFilesizeBaseline:
    def test_beats_chance_on_separable_sizes(self, tmp_path):
        from PIL import Image
        import numpy as np

        image_dir = tmp_path / "jpgs"
        image_dir.mkdir()
        rows = []
        rng = np.random.default_rng(0)
        for i in range(30):
            lake = f"CW2019_{1500 + i // 3}"
            fn = f"{lake}_t{i % 3:03d}.jpg"
            label = i % 2
            # label 1 -> noisy (large file); label 0 -> flat (tiny file)
            arr = (rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) if label
                   else np.zeros((64, 64, 3), dtype=np.uint8))
            Image.fromarray(arr).save(image_dir / fn, quality=95)
            rows.append({"filename": fn, "label": label})
        path = tmp_path / "labels.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        res = filesize_baseline(path, image_dir)
        assert res["n"] == 30
        assert res["accuracy"] > 0.9

    def test_raises_when_images_missing(self, tmp_path):
        path = make_labels(tmp_path, n_lakes=4)
        with pytest.raises(ValueError):
            filesize_baseline(path, tmp_path / "nonexistent")
