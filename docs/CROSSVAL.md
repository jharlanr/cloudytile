# Cross-validation and model selection

How `cloudy-tile` picks a model, why the protocol is shaped this way, and how to run it.

---

## 1. The protocol in one picture

```
400 labeled lakes
│
├── splits/cloudytile_v1/   (frozen, committed, ESSD-style lake-ID lists)
│   │
│   ├── train 280 lakes ─┐
│   │                    ├── 320 DEV lakes ──> model selection lives here
│   ├── val    40 lakes ─┘        │
│   │                             └── 3-fold lake-grouped CV over 32 configs
│   │                                   each fold: inner val split for
│   │                                   checkpoint + threshold selection
│   │
│   └── test   80 lakes ─────────> touched ONCE, at the very end
│
└── winner retrained on all 320 dev lakes, evaluated once on the 80
```

Three rules hold everywhere, and each exists because it was violated once:

1. **Group by lake, never by tile.** Frames of one lake days apart are near-duplicates. The
   original 80/10/10 tile split put all 52 test lakes into training as well, so the reported
   ~95% measured recall of memorized lakes.
2. **Freeze the split.** A seed-derived split silently moves whenever `labels.csv` changes.
   `splits/cloudytile_v1/` pins the test lakes for the life of the project.
3. **Selection never sees test.** Picking the best of 32 configs by their fold scores is
   selection on whatever data those folds cover. `run_cv_grid.py --split_dir` restricts folds
   to the dev lakes so the 80 test lakes stay clean for one final number.

---

## 2. The grid

32 configs = 4 band sets × 2 widths × 2 learning rates × 2 optimizers.

| axis | values | why |
|---|---|---|
| `bands` | `rgb`, `rgb+nir`, `rgb+swir16`, `all6` | Does spectral information beyond RGB help? The January sweep said no, but it was measured on a leaked split, so the question is genuinely open. |
| `arch` | `small` = [16,32,64], `wide` = [32,64,128] | With the GAP head both are ~32k and ~130k params — cheap to test whether capacity binds. |
| `lr` | `1e-3`, `3e-4` | The dominant hyperparameter for Adam-family optimizers at this scale. |
| `optimizer` | `adam`, `adamw` | AdamW's decoupled weight decay pairs better with BatchNorm; expect a small or null difference. |

Every config is scored on **identical folds** — folds are a function of `(labels, n_folds, seed)`
alone — so differences between configs are not confounded by different data.

Fixed across the grid: GAP head, BatchNorm, dropout 0.3, flip/rot90 augmentation, 40 epochs,
batch 32, weight decay 1e-4, **256 px**. The GAP head is resolution-independent, so the grid runs
at 256 for ~4× the throughput; rerun the winner at 512 if resolution turns out to matter.

---

## 3. What happens inside one (config, fold)

```
fold train lakes ──split_off_val_lakes(15%)──> inner train  ──> fit
                                          └──> inner val    ──> best epoch + threshold
fold test lakes ─────────────────────────────────────────────> scored once
```

- **Checkpoint selection**: the epoch with the lowest inner-val loss is kept. The test pass runs
  on those restored weights, so reported metrics describe the model you would ship — the January
  code evaluated the *final* epoch while saving the *best* one, and the two were different.
- **Threshold selection**: `pick_threshold` chooses an operating point on inner-val probabilities
  and applies it to test. Never chosen on test.
- **Disjointness** is asserted, not assumed (`assert_lake_disjoint`).

---

## 4. Class balance and the operating point

The data is **68.6% useful / 31.4% not useful — about 2.2:1**, which is mild. This does *not*
call for focal loss or class weights; plain `BCEWithLogitsLoss` is right, and
`StratifiedGroupKFold` already balances label proportions across folds.

What is asymmetric is **cost**, not frequency:

| error | consequence |
|---|---|
| false positive (cloudy called useful) | a bad frame enters lake-vision's timeseries and can corrupt a drainage call |
| false negative (clear called cloudy) | one observation lost from a lake with ~90 usable ones |

So precision on the useful class is worth more than recall — and the lever for that is the
**decision threshold**, not the loss function. One training run then yields the entire
precision/recall curve, the operating point stays a single reportable number, and it can be
changed later without retraining.

```bash
--threshold_objective f1                                        # default, balanced
--threshold_objective target_precision --target_precision 0.97  # cost-asymmetric
```

---

## 5. Reading the results

Configs are ranked by **AUC**, deliberately:

- AUC is threshold-free, so it compares models without also comparing operating points.
- Accuracy sits near the 68.4% majority rate, which compresses differences.
- An 80-lake holdout's own positive rate varies ±2.6 points (2σ) across draws, so accuracy
  differences of a couple of points are noise.

Always quote against the baselines, not against chance:

| baseline | accuracy |
|---|---|
| majority class ("always useful") | 68.4% |
| JPG file-size threshold | 82.5% |

The file-size baseline is a `stat()` call with one threshold — it exploits the fact that clear ice
compresses poorly and cloud compresses well. **A model beating 68% but not 82.5% has learned
nothing that JPEG did not already know.**

`--summarize` prints the ranked table and warns when the gap to second place is smaller than the
fold-to-fold spread — in that case treat it as a tie and take the simpler config.

---

## 6. Running it

```bash
# 0. once: freeze the split (already committed as splits/cloudytile_v1)
python3 engine/make_splits.py --labels_csv labels/labels.csv \
    --out_dir splits/cloudytile_v1

# 1. per-tile training data + normalization stats
sbatch slurm/extract_training_nc.sh
sbatch slurm/compute_band_stats.sh

# 2. the grid: 32 array tasks, 8 concurrent, ~16-25 A100-hours total
sbatch slurm/run_cv_grid.sh

# 3. aggregate (login node)
python3 engine/run_cv_grid.py --out_dir <OUT_DIR> --summarize

# 4. final model: winner retrained, test lakes touched once
python3 engine/run_training.py \
    --labels_csv labels/labels.csv \
    --split_dir splits/cloudytile_v1 \
    --nc_dir <tiles> --band_stats <stats.json> \
    --nc_channels red green blue nir \
    --optimize_metric loss --save_path best_model.pth
```

**Resumability**: each `(config, fold)` writes one JSON and existing ones are skipped, so a
timed-out or partially failed array can be resubmitted with the same command.

---

## 7. wandb

Each `(config, fold)` is one wandb run, `group=<config_name>`, so the UI averages folds natively.
Tags carry the band set, width, optimizer, and fold index for filtering. Per-epoch train/val
metrics are logged; test metrics and the chosen threshold land in `wandb.summary`.

Compute nodes have no internet, so `slurm/run_cv_grid.sh` sets `WANDB_MODE=offline` and buffers
runs to `$WANDB_DIR`. Sync from a **login** node after the array finishes:

```bash
wandb sync /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile/wandb/offline-run-*
```

Useful views once synced: group by `group` for per-config fold means; filter `bands=all6` to see
whether the extra channels pay; scatter `n_bands` against `test_auc`. Disable with `--no_wandb`;
the JSON results are written regardless, and `--summarize` never needs wandb.

---

## 8. Known limits

- **Selection bias is reduced, not eliminated.** The winner is picked on dev folds, so its *dev*
  score is optimistic; the 80-lake test number is the honest one. Report that.
- **One frozen split, not nested CV.** Full nested CV (outer folds × inner selection) would give
  an unbiased estimate with a spread, at ~5× the cost. Given a single-number headline is what the
  paper needs, the frozen holdout is the right trade — but the test estimate carries the variance
  of one 80-lake draw (roughly ±2–3 points).
- **Years are pooled.** Test is 33 lakes from 2019 and 47 from 2018, and 2018 is much cloudier
  (39.0% vs 23.8% not-useful). A cross-year split — train 2019, test 2018 — would answer a
  different and harder question, and is not what this protocol measures.
- **The grid is small on purpose.** 32 configs over 3 folds is enough to answer "do extra bands
  help, and is capacity binding"; it is not a hyperparameter search. If nothing clears the
  file-size baseline, the answer is more labels or a different architecture, not a finer grid.
