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

    def test_pool_head_is_resolution_independent(self):
        # the whole point of adaptive pooling: an NxN grid regardless of input,
        # so a spatial head does not reintroduce the flatten head's dependence
        # on image area
        a = CloudyTileCNN(img_size=(256, 256), in_channels=4,
                          head="pool16", fc_layers=[8])
        b = CloudyTileCNN(img_size=(512, 512), in_channels=4,
                          head="pool16", fc_layers=[8])
        assert a.n_parameters() == b.n_parameters()

    def test_pool_head_forward_and_size(self):
        m = CloudyTileCNN(in_channels=4, head="pool16", fc_layers=[8])
        m.eval()
        # 64 channels x 16 x 16 = 16,384 into an 8-wide hidden layer
        assert m.state_dict()["classifier.2.weight"].shape == (8, 16384)
        for s in (128, 512):
            assert m(torch.zeros(2, 4, s, s)).shape == (2,)

    def test_pool_head_narrow_fc_keeps_it_small(self):
        # 16x16 spatial x a wide fc is how you accidentally rebuild the 33M
        # model; the narrow hidden layer is what makes this head affordable
        narrow = CloudyTileCNN(in_channels=4, head="pool16", fc_layers=[8])
        wide = CloudyTileCNN(in_channels=4, head="pool16", fc_layers=[128])
        assert narrow.n_parameters() < 200_000
        assert wide.n_parameters() > 2_000_000

    def test_pool1_matches_gap(self):
        p1 = CloudyTileCNN(in_channels=4, head="pool1")
        gap = CloudyTileCNN(in_channels=4, head="gap")
        assert p1.n_parameters() == gap.n_parameters()

    def test_head_reduce_collapses_channels_before_flatten(self):
        # the difference between "16x16 over 64 channels" (16,384 values) and
        # "16x16 over 1 channel" (256 values) is the whole parameter story
        full = CloudyTileCNN(in_channels=4, head="pool16", fc_layers=[8])
        red = CloudyTileCNN(in_channels=4, head="pool16", head_reduce=1,
                            fc_layers=[8])
        assert full.state_dict()["classifier.2.weight"].shape == (8, 16384)
        assert red.state_dict()["classifier.2.weight"].shape == (8, 256)
        # and the reduced head is cheaper than GAP, whose 64->128 layer alone
        # is larger than this entire head
        gap = CloudyTileCNN(in_channels=4, head="gap")
        assert red.n_parameters() < gap.n_parameters() < full.n_parameters()

    def test_head_reduce_still_resolution_independent(self):
        a = CloudyTileCNN(img_size=(256, 256), in_channels=4, head="pool16",
                          head_reduce=1, fc_layers=[8])
        b = CloudyTileCNN(img_size=(512, 512), in_channels=4, head="pool16",
                          head_reduce=1, fc_layers=[8])
        assert a.n_parameters() == b.n_parameters()
        b.eval()
        for s in (128, 512):
            assert b(torch.zeros(2, 4, s, s)).shape == (2,)

    def test_head_reduce_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            CloudyTileCNN(head="pool16", head_reduce=0)

    def test_rejects_malformed_pool_head(self):
        for bad in ("poolx", "pool", "pool0", "pool-4"):
            with pytest.raises(ValueError):
                CloudyTileCNN(head=bad)


class TestBandHeadGrid:
    """The sweep's configs must build, and the v1 indices must stay frozen."""

    def _grid(self):
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "engine"))
        import run_cv_grid as R
        return R

    def test_v1_indices_are_frozen(self):
        R = self._grid()
        # recorded in slurm/run_cv_finalists.sh; adding band sets or
        # architectures must never renumber these
        assert R.config_name(R.GRID[9]) == "rgb+nir_small_lr0.001_adamw"
        assert R.config_name(R.GRID[17]) == "rgb+swir16_small_lr0.001_adamw"
        assert R.config_name(R.GRID[25]) == "all6_small_lr0.001_adamw"

    def test_bandhead_grid_shape(self):
        R = self._grid()
        assert len(R.BANDHEAD_GRID) == 16
        assert {c["head"] for c in R.BANDHEAD_GRID} == set(R.HEADS)
        assert {c["bands"] for c in R.BANDHEAD_GRID} == set(R.BANDHEAD_BANDS)

    def test_every_config_builds_with_expected_size(self):
        R = self._grid()
        expected = {"gap": 228_609, "mixed": 229_657,
                    "spatial": 228_871, "full": 1_276_401}
        for cfg in R.BANDHEAD_GRID:
            if cfg["bands"] != "rgb+nir":      # 4-band reference counts
                continue
            h = R.HEADS[cfg["head"]]
            m = CloudyTileCNN(
                img_size=(512, 512), channels=R.CHANNEL_SETS[cfg["arch"]],
                in_channels=len(R.BAND_SETS[cfg["bands"]]), head=h["head"],
                head_reduce=h["head_reduce"], fc_layers=h["fc_layers"])
            assert m.n_parameters() == expected[cfg["head"]], cfg
            m.eval()
            assert m(torch.zeros(2, 4, 512, 512)).shape == (2,)

    def test_matched_heads_are_within_one_percent(self):
        R = self._grid()
        sizes = []
        for name in ("gap", "mixed", "spatial"):
            h = R.HEADS[name]
            sizes.append(CloudyTileCNN(
                channels=R.CHANNEL_SETS["deep6"], in_channels=4,
                head=h["head"], head_reduce=h["head_reduce"],
                fc_layers=h["fc_layers"]).n_parameters())
        assert (max(sizes) - min(sizes)) / min(sizes) < 0.01


class TestRunOneSmoke:
    """
    Actually execute one (config, fold) end to end.

    The bands x heads sweep died on all 16 array tasks in under a minute
    because run_one read a variable in the wandb.init config block that was
    not assigned until after it -- an UnboundLocalError that only fires when
    wandb is present, which no test exercised. Every unit test above builds
    models and datasets in isolation; none had ever run the function that
    wires them together. This one does, with wandb stubbed so the branch that
    broke is the branch under test.
    """

    def _args(self, nc_dir, **over):
        import argparse
        a = dict(seed=0, head="gap", head_reduce=None, fc_layers=[8],
                 wandb_project="smoke", img_size=64, epochs=1, batch_size=4,
                 weight_decay=1e-4, lr_schedule="cosine", no_augment=True,
                 nc_dir=nc_dir, band_stats=STATS, num_workers=0,
                 threshold_objective="f1", target_precision=0.9)
        a.update(over)
        return argparse.Namespace(**a)

    @pytest.mark.parametrize("head", ["gap", "mixed", "spatial", "full"])
    def test_one_fold_runs_end_to_end(self, tmp_path, head, monkeypatch):
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "engine"))
        import run_cv_grid as R
        from cloudytile.splits import add_lake_id

        # force the wandb path on even where wandb is not installed: the bug
        # lived inside `if WANDB_AVAILABLE and args.wandb_project`
        logged = []

        final = {}

        class FakeWandb:
            summary = final  # run_one writes final metrics via wandb.summary

            @staticmethod
            def init(**kw):
                logged.append(kw["config"])
                return object()

            @staticmethod
            def log(d):
                pass

            @staticmethod
            def finish():
                logged.append("finish")

        monkeypatch.setattr(R, "wandb", FakeWandb, raising=False)
        monkeypatch.setattr(R, "WANDB_AVAILABLE", True, raising=False)

        csv, nc_dir = make_tiles(tmp_path, n=24, size=64, nan_frac=0.1)
        df = add_lake_id(pd.read_csv(csv))
        lakes = np.sort(df["lake_id"].unique())
        train_df = df[df["lake_id"].isin(lakes[:-2])].reset_index(drop=True)
        test_df = df[df["lake_id"].isin(lakes[-2:])].reset_index(drop=True)

        cfg = {"bands": "rgb+nir", "arch": "deep6", "lr": 1e-3,
               "optimizer": "adamw", "head": head}
        result = R.run_one(cfg, 0, train_df, test_df, self._args(nc_dir))

        assert result["head"] == head
        assert result["head_spec"] == R.HEADS[head]
        assert result["best_epoch"] == 1
        assert 0.0 <= result["test_accuracy"] <= 1.0
        assert result["n_parameters"] > 0
        # the wandb config must carry the resolved head, not a placeholder
        assert logged[0]["head_spec"] == R.HEADS[head]["head"]
        assert logged[-1] == "finish"
        assert final["test_auc"] == result["test_auc"]

    def test_v1_config_without_head_takes_it_from_cli(self, tmp_path, monkeypatch):
        # GRID configs predate the head axis; run_one must fall back to flags
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "engine"))
        import run_cv_grid as R
        from cloudytile.splits import add_lake_id

        monkeypatch.setattr(R, "WANDB_AVAILABLE", False, raising=False)
        csv, nc_dir = make_tiles(tmp_path, n=24, size=64, nan_frac=0.1)
        df = add_lake_id(pd.read_csv(csv))
        lakes = np.sort(df["lake_id"].unique())
        train_df = df[df["lake_id"].isin(lakes[:-2])].reset_index(drop=True)
        test_df = df[df["lake_id"].isin(lakes[-2:])].reset_index(drop=True)

        cfg = R.GRID[9]  # rgb+nir_small_lr0.001_adamw, no "head" key
        assert "head" not in cfg
        result = R.run_one(cfg, 0, train_df, test_df,
                           self._args(nc_dir, head="gap", fc_layers=[128]))
        assert result["head"] == "gap"
        assert result["head_spec"]["fc_layers"] == [128]


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
