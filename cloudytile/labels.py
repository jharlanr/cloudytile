"""
Label CSV I/O for the tile-labeling GUI.

The on-disk format is the canonical labels.csv consumed by
CloudyTileDataset: columns `filename,label`, one row per JPG frame,
where 0 = not useful (cloudy/no data) and 1 = useful (clear).
"""
import csv
import os
from pathlib import Path
from typing import Optional, Union


class LabelStore:
    """
    Ordered filename -> label mapping backed by a labels CSV.

    Pre-existing rows are preserved in their original order and
    re-labeling a filename updates its row in place, so `filename`
    stays unique and a session can always resume where it stopped.
    save() rewrites the file atomically (temp file + os.replace, same
    pattern as add_cloudy_seq_to_nc): a crash mid-write leaves the
    previous complete file on disk, never a truncated one.
    """

    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        self._labels: dict[str, int] = {}  # insertion-ordered
        if self.csv_path.exists():
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is not None and not {"filename", "label"} <= set(reader.fieldnames):
                    raise ValueError(
                        f"{self.csv_path} must have 'filename' and 'label' columns "
                        f"(found: {reader.fieldnames})"
                    )
                for row in reader:
                    self._labels[row["filename"]] = int(row["label"])

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, filename: str) -> bool:
        return filename in self._labels

    @property
    def filenames(self) -> list[str]:
        return list(self._labels)

    def get(self, filename: str) -> Optional[int]:
        return self._labels.get(filename)

    def set(self, filename: str, label: int) -> None:
        label = int(label)
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label}")
        self._labels[filename] = label

    def counts(self, filenames=None) -> tuple[int, int]:
        """(n_zeros, n_ones), optionally restricted to `filenames`."""
        labels = (
            self._labels.values()
            if filenames is None
            else [self._labels[f] for f in filenames if f in self._labels]
        )
        n1 = sum(labels)
        return len(labels) - n1, n1

    def save(self) -> None:
        # lineterminator="\n" keeps the file byte-identical to the
        # pandas-written original for untouched rows.
        tmp_path = self.csv_path.parent / f".{self.csv_path.name}.tmp"
        try:
            with open(tmp_path, "w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["filename", "label"])
                writer.writerows(self._labels.items())
            os.replace(tmp_path, self.csv_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
