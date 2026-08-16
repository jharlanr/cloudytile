"""
Tests for LabelStore label CSV I/O.

This is the layer where silent data loss would happen: labels.csv holds
thousands of hand-labeled rows, so merge/resume/atomic-write behavior is
tested here without any GUI involvement.
"""
import csv
import os
from pathlib import Path

import pytest

from cloudytile.labels import LabelStore

EXISTING_ROWS = [
    ("CW2019_1530_t006.jpg", 0),
    ("CW2019_1530_t007.jpg", 0),
    ("CW2019_1530_t008.jpg", 1),
    ("CW2019_1579_t003.jpg", 1),
    ("CW2019_2504_t151.jpg", 0),
]


@pytest.fixture
def existing_csv(tmp_path):
    """A labels.csv with real-looking prior rows (pandas-style: LF, trailing newline)."""
    path = tmp_path / "labels.csv"
    text = "filename,label\n" + "".join(f"{f},{l}\n" for f, l in EXISTING_ROWS)
    path.write_text(text)
    return path


class TestFreshStart:
    """Starting with no labels.csv."""

    def test_missing_file_starts_empty(self, tmp_path):
        store = LabelStore(tmp_path / "labels.csv")
        assert len(store) == 0

    def test_writes_correct_rows(self, tmp_path):
        path = tmp_path / "labels.csv"
        store = LabelStore(path)
        store.set("A_t000.jpg", 0)
        store.set("A_t001.jpg", 1)
        store.save()

        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows == [["filename", "label"], ["A_t000.jpg", "0"], ["A_t001.jpg", "1"]]

    def test_rejects_bad_label(self, tmp_path):
        store = LabelStore(tmp_path / "labels.csv")
        with pytest.raises(ValueError):
            store.set("A_t000.jpg", 2)


class TestResume:
    """Loading an existing CSV so a session continues where it stopped."""

    def test_loads_existing_labels(self, existing_csv):
        store = LabelStore(existing_csv)
        assert len(store) == len(EXISTING_ROWS)
        for filename, label in EXISTING_ROWS:
            assert filename in store
            assert store.get(filename) == label

    def test_membership_drives_skip(self, existing_csv):
        store = LabelStore(existing_csv)
        assert "CW2019_1530_t006.jpg" in store
        assert "CW2019_9999_t000.jpg" not in store

    def test_counts(self, existing_csv):
        store = LabelStore(existing_csv)
        assert store.counts() == (3, 2)
        # restricted to a subset (the GUI's image_dir scope)
        subset = ["CW2019_1530_t008.jpg", "CW2019_1579_t003.jpg", "CW2019_9999_t000.jpg"]
        assert store.counts(subset) == (0, 2)

    def test_rejects_wrong_schema(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("lake_id,label\nCW2019_1530,0\n")
        with pytest.raises(ValueError):
            LabelStore(path)


class TestMerge:
    """New labels must never drop, reorder, or rewrite pre-existing rows."""

    def test_existing_rows_preserved_byte_for_byte(self, existing_csv):
        original_text = existing_csv.read_text()
        store = LabelStore(existing_csv)
        store.set("NW2020_0001_t000.jpg", 1)
        store.set("NW2020_0001_t001.jpg", 0)
        store.save()

        new_text = existing_csv.read_text()
        assert new_text.startswith(original_text)
        assert new_text == original_text + "NW2020_0001_t000.jpg,1\nNW2020_0001_t001.jpg,0\n"

    def test_save_without_changes_is_identity(self, existing_csv):
        original_text = existing_csv.read_text()
        store = LabelStore(existing_csv)
        store.save()
        assert existing_csv.read_text() == original_text


class TestUpdateInPlace:
    """Re-labeling an existing filename updates its row, never appends."""

    def test_relabel_updates_row_in_place(self, existing_csv):
        store = LabelStore(existing_csv)
        store.set("CW2019_1530_t007.jpg", 1)
        store.save()

        with open(existing_csv, newline="") as f:
            rows = list(csv.reader(f))[1:]
        # same position, same total count, new value
        assert rows[1] == ["CW2019_1530_t007.jpg", "1"]
        assert len(rows) == len(EXISTING_ROWS)

    def test_filenames_stay_unique(self, existing_csv):
        store = LabelStore(existing_csv)
        for _ in range(3):
            store.set("CW2019_1530_t006.jpg", 1)
            store.set("NW2020_0001_t000.jpg", 0)
        store.save()

        reloaded = LabelStore(existing_csv)
        filenames = reloaded.filenames
        assert len(filenames) == len(set(filenames)) == len(EXISTING_ROWS) + 1


class TestAtomicWrite:
    """A crash mid-save must leave the previous complete file on disk."""

    def test_failed_replace_leaves_original_intact(self, existing_csv, monkeypatch):
        original_text = existing_csv.read_text()
        store = LabelStore(existing_csv)
        store.set("NW2020_0001_t000.jpg", 1)

        def boom(src, dst):
            raise OSError("simulated crash")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            store.save()

        assert existing_csv.read_text() == original_text
        assert not (existing_csv.parent / f".{existing_csv.name}.tmp").exists()

    def test_no_temp_file_left_after_save(self, existing_csv):
        store = LabelStore(existing_csv)
        store.set("NW2020_0001_t000.jpg", 1)
        store.save()
        assert not (existing_csv.parent / f".{existing_csv.name}.tmp").exists()


class TestDatasetCompat:
    """The written CSV must be readable by CloudyTileDataset unmodified."""

    def test_cloudytile_dataset_reads_output(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        from PIL import Image
        from cloudytile.data import CloudyTileDataset

        image_dir = tmp_path / "jpgs"
        image_dir.mkdir()
        store = LabelStore(tmp_path / "labels.csv")
        for i in range(4):
            filename = f"CW2019_0001_t{i:03d}.jpg"
            Image.new("RGB", (64, 64), (i * 40, 100, 150)).save(image_dir / filename)
            store.set(filename, i % 2)
        store.save()

        dataset = CloudyTileDataset(tmp_path / "labels.csv", image_dir, img_size=(64, 64))
        assert len(dataset) == 4
        image, label = dataset[1]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert label == 1
