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

    def test_bandhead_index_order_matches_the_slurm_comment(self):
        # slurm/run_cv_bandhead.sh documents array index -> config as a 1:1 map
        # with no lookup table, so the product order (bands outer, heads inner,
        # both in declaration order) is load-bearing. Reordering HEADS or
        # BANDHEAD_BANDS would silently re-point every array index.
        R = self._grid()
        names = [R.config_name(c) for c in R.BANDHEAD_GRID]
        assert names[:4] == ["rgb_gap", "rgb_mixed", "rgb_spatial", "rgb_full"]
        assert names[4] == "rgb+nir_gap"
        assert names[8] == "rgb+swir16_gap"
        assert names[12] == "rgb+swir22_gap"
        assert names[15] == "rgb+swir22_full"

    def test_epochs_grid_shape_and_names(self):
        R = self._grid()
        assert len(R.EPOCHS_GRID) == 15
        # heads outer, horizons inner -- the SLURM script maps array index
        # this way with no lookup table
        names = [R.config_name(c) for c in R.EPOCHS_GRID]
        assert names[0] == "rgb+swir16_gap_e40"
        assert names[4] == "rgb+swir16_gap_e200"
        assert names[5] == "rgb+swir16_mixed_e40"
        assert names[14] == "rgb+swir16_spatial_e200"
        assert len(set(names)) == 15
        assert all(c["bands"] == "rgb+swir16" for c in R.EPOCHS_GRID)
        assert "full" not in {c["head"] for c in R.EPOCHS_GRID}

    def test_epochs_suffix_does_not_touch_bandhead_names(self):
        # the bandhead sweep has 80 finished result JSONs on disk; adding the
        # epochs axis must not rename any of them, or --summarize silently
        # splits one config's folds across two rows
        R = self._grid()
        assert R.config_name(R.BANDHEAD_GRID[0]) == "rgb_gap"
        assert R.config_name(R.BANDHEAD_GRID[9]) == "rgb+swir16_mixed"
        assert all("_e" not in R.config_name(c) for c in R.BANDHEAD_GRID)

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


class TestSelectionRegimeIsReproducible:
    """
    run_cv_grid.py SELECTS a config; run_training.py has to BUILD it.

    Nothing enforced that, and the gap opened twice: the grid chose AdamW while
    run_training hardcoded Adam, and the grid's 'mixed'/'spatial' heads are
    defined by head_reduce, which run_training had no flag for and did not pass
    to the model. The second one is the dangerous shape -- head='pool4' is
    valid without head_reduce, so it would have trained a 16x-wider head under
    the winner's name rather than failing.
    """

    def _mods(self):
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "engine"))
        import run_cv_grid as R
        import run_training as T
        return R, T

    @pytest.mark.parametrize("head", ["gap", "mixed", "spatial", "full"])
    def test_every_grid_head_survives_the_cli(self, head):
        R, T = self._mods()
        spec = R.HEADS[head]
        argv = ["--labels_csv", "l", "--nc_dir", "d",
                "--nc_channels", "red", "green", "blue", "nir",
                "--channels", *[str(c) for c in R.CHANNEL_SETS["deep6"]],
                "--fc_layers", *[str(f) for f in spec["fc_layers"]],
                "--head", spec["head"], "--img_size", "512"]
        if spec["head_reduce"] is not None:
            argv += ["--head_reduce", str(spec["head_reduce"])]
        args = T.build_parser().parse_args(argv)
        args.channels = T.parse_list(args.channels)
        args.fc_layers = T.parse_list(args.fc_layers)
        args.nc_channels = T.parse_string_list(args.nc_channels)

        # built exactly as run_training.train() builds it
        cfg = vars(args)
        from_cli = CloudyTileCNN(
            img_size=(cfg["img_size"], cfg["img_size"]),
            channels=cfg["channels"], fc_layers=cfg["fc_layers"],
            in_channels=len(cfg["nc_channels"]), head=cfg["head"],
            head_reduce=cfg.get("head_reduce"))
        from_grid = CloudyTileCNN(
            img_size=(512, 512), channels=R.CHANNEL_SETS["deep6"],
            in_channels=4, head=spec["head"],
            head_reduce=spec["head_reduce"], fc_layers=spec["fc_layers"])

        assert from_cli.n_parameters() == from_grid.n_parameters()
        assert [tuple(v.shape) for v in from_cli.state_dict().values()] == \
               [tuple(v.shape) for v in from_grid.state_dict().values()]

    def test_optimizer_and_schedule_cover_the_grid(self):
        R, T = self._mods()
        p = T.build_parser()
        opts = {a.option_strings[0]: a for a in p._actions}
        assert set(R.OPTIMIZERS) <= set(opts["--optimizer"].choices)
        assert "cosine" in opts["--lr_schedule"].choices

    @pytest.mark.parametrize("spelling", [
        ["--channels", "16", "32", "64"],       # nargs, as SLURM scripts write it
        ["--channels", "16 32 64"],             # one shell word
        ["--channels", "[16,32,64]"],           # a wandb sweep
    ])
    def test_list_flags_accept_every_spelling(self, spelling):
        _, T = self._mods()
        args = T.build_parser().parse_args(
            ["--labels_csv", "l", "--nc_dir", "d"] + spelling)
        assert T.parse_list(args.channels) == [16, 32, 64]

    def test_head_reduce_without_pool_head_is_rejected(self):
        # silently ignoring it is how "spatial" becomes a different model
        with pytest.raises(ValueError):
            CloudyTileCNN(head="gap", head_reduce=2)
        with pytest.raises(ValueError):
            CloudyTileCNN(head="flatten", head_reduce=2)


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
                 nc_dir=nc_dir, band_stats=STATS, num_workers=0, folds=5,
                 split_dir=None,
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

        class FakeRun:
            # real wandb.run.dir is <offline-run-dir>/files; run_one records
            # the parent so the upload can target it exactly
            dir = str(tmp_path / "wandb" / "offline-run-XYZ" / "files")

        class FakeWandb:
            summary = final  # run_one writes final metrics via wandb.summary

            @staticmethod
            def init(**kw):
                logged.append(kw["config"])
                return FakeRun()

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
        # provenance: the offline run dir must be recorded, not left to a glob
        assert result["wandb_run_dir"].endswith("offline-run-XYZ")
        # regime: enough to tell this row from a k=3 256px one
        for k in ("img_size", "folds", "seed", "lr_schedule", "augment",
                  "batch_size", "weight_decay", "split_dir"):
            assert k in result, k
        assert result["img_size"] == 64 and result["folds"] == 5
        assert result["augment"] is False

    def test_config_epochs_override_cli_and_set_T_max(self, tmp_path, monkeypatch):
        # the horizon IS the schedule: cfg["epochs"] must drive the loop bound
        # and CosineAnnealingLR's T_max together, not just the loop
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "engine"))
        import run_cv_grid as R
        from cloudytile.splits import add_lake_id

        monkeypatch.setattr(R, "WANDB_AVAILABLE", False, raising=False)
        seen = {}
        real = __import__("torch").optim.lr_scheduler.CosineAnnealingLR

        def spy(optimizer, T_max, **kw):
            seen["T_max"] = T_max
            return real(optimizer, T_max=T_max, **kw)

        monkeypatch.setattr(
            __import__("torch").optim.lr_scheduler, "CosineAnnealingLR", spy)

        csv, nc_dir = make_tiles(tmp_path, n=24, size=64, nan_frac=0.1)
        df = add_lake_id(pd.read_csv(csv))
        lakes = np.sort(df["lake_id"].unique())
        train_df = df[df["lake_id"].isin(lakes[:-2])].reset_index(drop=True)
        test_df = df[df["lake_id"].isin(lakes[-2:])].reset_index(drop=True)

        cfg = dict(R.EPOCHS_GRID[0])       # gap, 40 epochs
        cfg["epochs"] = 3                  # keep the test fast
        # args says 1; the config must win
        result = R.run_one(cfg, 0, train_df, test_df,
                           self._args(nc_dir, epochs=1))
        assert result["epochs"] == 3
        assert seen["T_max"] == 3
        assert result["best_epoch"] <= 3

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


class TestGridCLI:
    """
    Invoke run_cv_grid.py as a subprocess, the way SLURM does.

    TestRunOneSmoke covers run_one; this covers everything around it that a
    unit test cannot reach -- argument parsing, frozen-split loading, fold
    generation, the result JSON, resume, and --summarize. Between them the
    whole path a SLURM task walks is executed before it is queued.
    """

    def _fixture(self, tmp_path):
        import json
        n_lakes, n_per, size = 8, 4, 64
        rng = np.random.default_rng(0)
        nc = tmp_path / "tiles"
        nc.mkdir()
        rows = []
        for li in range(n_lakes):
            lake = f"CW2019_{1500 + li}"
            for t in range(n_per):
                label = (li + t) % 2
                arr = rng.normal(5000 + 400 * label, 900,
                                 (6, size, size)).astype(np.float32)
                arr[:, :6, :] = np.nan
                name = f"{lake}_t{t:03d}"
                xr.Dataset({"imagery": (["channel", "y", "x"], arr)},
                           coords={"channel": BANDS}).to_netcdf(nc / f"{name}.nc")
                rows.append({"filename": f"{name}.jpg", "label": label})
        pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)

        lakes = [f"CW2019_{1500 + i}" for i in range(n_lakes)]
        split = tmp_path / "split"
        split.mkdir()
        for name, ids in (("train", lakes[:5]), ("val", lakes[5:6]),
                          ("test", lakes[6:])):
            (split / f"{name}_ids.json").write_text(json.dumps(ids))
        (tmp_path / "band_stats.json").write_text(json.dumps(
            {b: {"mean": 5000.0, "std": 1000.0} for b in BANDS}))
        return size

    def _run(self, tmp_path, *extra):
        import subprocess
        import sys
        from pathlib import Path as _P
        script = _P(__file__).resolve().parents[2] / "engine" / "run_cv_grid.py"
        return subprocess.run(
            [sys.executable, str(script),
             "--labels_csv", str(tmp_path / "labels.csv"),
             "--split_dir", str(tmp_path / "split"),
             "--nc_dir", str(tmp_path / "tiles"),
             "--band_stats", str(tmp_path / "band_stats.json"),
             "--out_dir", str(tmp_path / "out"), "--no_wandb", *extra],
            capture_output=True, text=True)

    def test_slurm_task_runs_and_summarizes(self, tmp_path):
        import json
        size = self._fixture(tmp_path)
        common = ("--grid", "bandhead", "--folds", "2", "--epochs", "1",
                  "--img_size", str(size), "--batch_size", "8",
                  "--num_workers", "0", "--lr_schedule", "cosine",
                  "--no_augment", "--seed", "42")
        r = self._run(tmp_path, "--config_index", "1", *common)  # rgb_mixed
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

        outs = sorted((tmp_path / "out").glob("*.json"))
        assert [p.name for p in outs] == ["rgb_mixed_fold0.json",
                                          "rgb_mixed_fold1.json"]
        d = json.loads(outs[0].read_text())
        assert d["head"] == "mixed"
        assert d["head_spec"]["head_reduce"] == 8
        assert d["config_name"] == "rgb_mixed"
        assert d["epochs"] == 1 and d["lr_schedule"] == "cosine"
        assert d["augment"] is False
        assert d["elapsed_sec"] > 0

        # resume: finished folds are skipped, not silently recomputed
        r = self._run(tmp_path, "--config_index", "1", *common)
        assert r.stdout.count("skip existing") == 2

        r = self._run(tmp_path, "--summarize")
        assert r.returncode == 0, r.stderr[-1000:]
        assert "rgb_mixed" in r.stdout
        assert (tmp_path / "out" / "summary.csv").exists()

    def test_summarize_refuses_to_blend_two_regimes(self, tmp_path):
        # the exact accident the SLURM scripts warn about in prose: two sweeps
        # into one directory produce a ranking whose rows are not comparable
        import json
        size = self._fixture(tmp_path)
        base = ("--grid", "bandhead", "--folds", "2", "--epochs", "1",
                "--num_workers", "0", "--no_augment",
                "--lr_schedule", "cosine", "--seed", "42",
                "--img_size", str(size))
        # identical except batch_size, which is a regime key: same folds, same
        # shapes, but not the same training setup
        r = self._run(tmp_path, "--config_index", "1", "--batch_size", "8", *base)
        assert r.returncode == 0, r.stderr[-1500:]
        r = self._run(tmp_path, "--config_index", "2", "--batch_size", "16", *base)
        assert r.returncode == 0, r.stderr[-1500:]

        r = self._run(tmp_path, "--summarize")
        assert r.returncode == 0
        assert "2 DIFFERENT REGIMES" in r.stdout
        assert "NOT comparable" in r.stdout
        assert "separate --out_dir" in r.stdout

    def test_summarize_states_the_regime_when_consistent(self, tmp_path):
        size = self._fixture(tmp_path)
        r = self._run(tmp_path, "--config_index", "1", "--grid", "bandhead",
                      "--folds", "2", "--epochs", "1", "--img_size", str(size),
                      "--batch_size", "8", "--num_workers", "0", "--no_augment",
                      "--lr_schedule", "cosine", "--seed", "42")
        assert r.returncode == 0, r.stderr[-1500:]
        r = self._run(tmp_path, "--summarize")
        assert "Regime:" in r.stdout
        assert f"img_size={size}" in r.stdout and "folds=2" in r.stdout
        assert "DIFFERENT REGIMES" not in r.stdout

    def test_config_index_past_the_grid_is_rejected(self, tmp_path):
        self._fixture(tmp_path)
        r = self._run(tmp_path, "--grid", "bandhead", "--config_index", "16")
        assert r.returncode != 0
        assert "out of range" in r.stderr and "0..15" in r.stderr


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
