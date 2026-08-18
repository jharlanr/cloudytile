"""
Tests for the GAP architecture and the NC dataset normalization contract.

The two invariants that history says need guarding:
  - parameter count: the legacy flatten head silently scaled with image area
    (33.5M params at 512px); the GAP head must stay resolution-independent.
  - normalization order: NaNs must be zeroed AFTER per-band normalization,
    or missing data lands at ~-4 sigma instead of 0.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from cloudytile.data import CloudyTileDatasetNC
from cloudytile.model import CloudyTileCNN

BANDS = ["red", "green", "blue", "nir", "swir16", "swir22"]
STATS = {b: {"mean": 5000.0, "std": 1000.0} for b in BANDS}


def make_tiles(tmp_path, n=6, size=32, nan_frac=0.25, seed=0):
    """Per-tile .nc files in the extract_training_nc.py output schema."""
    rng = np.random.default_rng(seed)
    nc_dir = tmp_path / "tiles"
    nc_dir.mkdir()
    rows = []
    for i in range(n):
        arr = rng.normal(5000, 1000, (len(BANDS), size, size)).astype(np.float32)
        arr[:, : int(size * nan_frac), :] = np.nan  # top band of no-data
        name = f"CW2019_{1500 + i // 3}_t{i % 3:03d}"
        xr.Dataset(
            {"imagery": (["channel", "y", "x"], arr)},
            coords={"channel": BANDS},
        ).to_netcdf(nc_dir / f"{name}.nc")
        rows.append({"filename": f"{name}.jpg", "label": i % 2})
    csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, nc_dir


class TestModel:
    def test_gap_param_count_independent_of_img_size(self):
        small = CloudyTileCNN(img_size=(256, 256), in_channels=3)
        large = CloudyTileCNN(img_size=(512, 512), in_channels=3)
        assert small.n_parameters() == large.n_parameters()
        assert small.n_parameters() < 50_000

    def test_gap_forward_any_size(self):
        m = CloudyTileCNN(in_channels=6)
        m.eval()
        for s in (64, 96):
            out = m(torch.zeros(2, 6, s, s))
            assert out.shape == (2,)

    def test_flatten_legacy_shape_and_count(self):
        # exact legacy structure: this is what pre-Aug-2026 checkpoints expect
        m = CloudyTileCNN(img_size=(512, 512), in_channels=3,
                          head="flatten", batch_norm=False, dropout=0.0)
        assert m.n_parameters() == 33_578_273
        sd = m.state_dict()
        assert sd["classifier.1.weight"].shape == (128, 262144)

    def test_gap_vs_flatten_ratio(self):
        gap = CloudyTileCNN(img_size=(512, 512), in_channels=3)
        flat = CloudyTileCNN(img_size=(512, 512), in_channels=3,
                             head="flatten", batch_norm=False, dropout=0.0)
        assert flat.n_parameters() / gap.n_parameters() > 500

    def test_rejects_unknown_head(self):
        with pytest.raises(ValueError):
            CloudyTileCNN(head="attention")


class TestNCDataset:
    def test_nan_is_exactly_zero_after_normalization(self, tmp_path):
        csv, nc_dir = make_tiles(tmp_path)
        ds = CloudyTileDatasetNC(csv, nc_dir, img_size=(32, 32), band_stats=STATS)
        image, _ = ds[0]
        nan_rows = image[:, :8, :]          # the region written as NaN
        valid_rows = image[:, 8:, :]
        assert torch.all(nan_rows == 0.0), "no-data must be exactly 0.0"
        # valid pixels are standardized: mean ~0, std ~1, and NOT ~-5
        assert abs(valid_rows.mean().item()) < 0.2
        assert 0.5 < valid_rows.std().item() < 1.5

    def test_old_order_would_have_failed_this(self, tmp_path):
        # regression guard: nan_to_num-then-normalize maps no-data to -5 sigma
        csv, nc_dir = make_tiles(tmp_path)
        ds = CloudyTileDatasetNC(csv, nc_dir, img_size=(32, 32), band_stats=STATS)
        image, _ = ds[0]
        wrong_value = (0.0 - 5000.0) / 1000.0  # what the old code produced
        assert not torch.any(torch.isclose(image[:, :8, :],
                                           torch.tensor(wrong_value)))

    def test_legacy_scale_path_also_zeroes_nans(self, tmp_path):
        csv, nc_dir = make_tiles(tmp_path)
        ds = CloudyTileDatasetNC(csv, nc_dir, img_size=(32, 32), band_stats=None)
        image, _ = ds[0]
        assert torch.all(image[:, :8, :] == 0.0)

    def test_channel_subset(self, tmp_path):
        csv, nc_dir = make_tiles(tmp_path)
        ds = CloudyTileDatasetNC(csv, nc_dir, channels=["red", "green", "blue"],
                                 img_size=(32, 32), band_stats=STATS)
        image, _ = ds[0]
        assert image.shape == (3, 32, 32)

    def test_augment_preserves_shape_and_content_set(self, tmp_path):
        csv, nc_dir = make_tiles(tmp_path, nan_frac=0.0)
        plain = CloudyTileDatasetNC(csv, nc_dir, img_size=(32, 32),
                                    band_stats=STATS)
        aug = CloudyTileDatasetNC(csv, nc_dir, img_size=(32, 32),
                                  band_stats=STATS, augment=True)
        a, la = plain[1]
        torch.manual_seed(0)
        b, lb = aug[1]
        assert a.shape == b.shape and la == lb
        # flips/rot90 permute pixels but never change their values
        assert torch.allclose(a.flatten().sort().values,
                              b.flatten().sort().values)

    def test_no_nan_ever_reaches_the_model(self, tmp_path):
        csv, nc_dir = make_tiles(tmp_path, nan_frac=0.6)
        ds = CloudyTileDatasetNC(csv, nc_dir, img_size=(32, 32), band_stats=STATS)
        for i in range(len(ds)):
            image, _ = ds[i]
            assert not torch.any(torch.isnan(image))
