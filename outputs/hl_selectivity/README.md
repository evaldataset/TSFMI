# Hewitt & Liang control-task selectivity (`outputs/hl_selectivity/`)

Produced by `scripts/run_hl_selectivity.py`; the full command is stored in `results.json`
(`meta.last_command` and on each row as `command`). The console log is in `run.log`.

## What this replaces

The earlier "selectivity" in this repo (`src/metrics`, `compute_selectivity`) was
*linear-probe accuracy − MLP-probe accuracy*. That is **not** Hewitt & Liang selectivity. It
should be called the **linear–MLP accessibility gap**. This directory reports H&L selectivity.

## Definition (from Hewitt & Liang 2019, EMNLP-IJCNLP, pp. 2733–2743)

* **Sec. 1 / Fig. 2:** selectivity = linguistic-task accuracy − control-task accuracy. Both
  accuracies come from the same probe.
* **Sec. 2:** a control task has *structure* (the output for a token is a deterministic function
  of its word type) and *randomness* (the output for each type is sampled independently at
  random). It shares the real task's output space Y.
* **Sec. 2.1, eq. 1 and footnote 2:** the control behaviour C(v) is sampled once per type v.
  For POS it is drawn from the empirical label distribution "so the marginal probability of each
  label is similar".
* **Sec. 2.3:** the control-task ceiling is the fraction of evaluation tokens whose type occurs
  in training, plus chance on the rest. The probe can only succeed by memorising type identity.
* **Sec. 3 / Table 1:** the same probe family and hyperparameters are used for both tasks.

## Time-series adaptation

Time series have no recurring word types, so we use a model-independent stand-in for "type":

| Element | Choice |
|---|---|
| Type | k-means cluster id of the **raw input window**. The descriptor is the per-window z-normalised series, block-mean downsampled 512→64 points. KMeans(`n_init=10`, `random_state=0`) is fit on the full dataset. We report k ∈ {20, 50}. |
| Control label | One draw per cluster, `numpy.default_rng(0).choice(y, k)`, i.e. drawn from the empirical label distribution (classification) or the empirical target distribution (seasonality, scored with R²). Every window in a cluster gets its cluster's label. There is one control task per (property, dataset size), fixed across split seeds, as in H&L. |
| Probe | `run_canonical_benchmark.train_and_score`: StandardScaler + LogisticRegression(lbfgs, max_iter=1000, random_state=split seed), or Ridge(α=1). |
| Splits | `three_way_split`, 60/20/20, split seeds 0–4. Test-split scores are reported. |
| Layer | The canonical **per-seed** best layer (`best_layer_per_seed` in `outputs/canonical/<dir>_<prop>/canonical_results.json`). Seed *s* uses the layer that canonical seed *s* selected on validation. |
| CI | `run_canonical_benchmark.bootstrap_ci`: 1000 resamples of the 5 seed values, rng 42. |
| Baselines | `hand_crafted_features` and `random_projection` imported from `scripts/run_canonical_baselines.py`, plus the raw signal. Data is the same generators with seed 42. |

Selectivity is computed per seed as task − control, then summarised.

## Reductions actually used (deviation from `extract_all_canonical.sh`)

Before computing any control score, the task score of every cell had to reproduce the committed
canonical score. The committed numbers **do not** come from the recipe in
`scripts/extract_all_canonical.sh`. We identified the recipe that does reproduce them by checking
seed by seed on anomaly, the only property that is not at ceiling:

| Model | n samples | Reduction that reproduces canonical | `extract_all_canonical.sh` says |
|---|---|---|---|
| moment | 5000 | PCA-512 **fit on all 5000 samples** (exact match on all 5 seeds) | PCA-512 train-only, n=1000 |
| gpt4ts | 5000 | PCA-512 **fit on all samples** (exact match on all 5 seeds) | PCA-512 train-only, n=1000 |
| chronos | 5000 | **mean-pool** over the 33 tokens (match within 1–3 test samples per seed; the difference is GPU nondeterminism) | identity, n=1000 |
| timer / timesfm / moirai | 1000 | mean-pool | same |

Checks that ruled out the documented recipe: at n=1000, test accuracy moves in steps of 0.005,
but the canonical chronos, moment and gpt4ts anomaly scores are multiples of 0.001, which needs
n_test=1000 and therefore n=5000. With train-only PCA, moment anomaly gives 0.540 against a
canonical 0.549. Identity-flattened chronos gives 0.645 (n=5000) and 0.52 (n=1000) against a
canonical 0.727. We use the reproducing recipe (`MODEL_SPECS` in the script). Model rows
therefore use n=5000 for moment, gpt4ts and chronos and n=1000 for the rest. Baselines are
reported at n=1000, which reproduces `outputs/canonical_baselines`, and at n=5000 for a
like-for-like comparison with the n=5000 models.

Tensors are stored in `outputs/representations_selectivity/` (raw layers under
`raw/`, canonical views under `<canonical_dir>/`). Only the per-seed best layers were extracted.

## Sanity check

All 36 model cells and 18 baseline cells (n=1000) reproduce the canonical test mean inside its
95% CI (± 1e-3). 34 of the 36 model cells match to 4 decimals. The other two are within the CI:
chronos anomaly 0.7274 vs 0.727, and timesfm per-seed differences of one test sample. See
`task.sanity` on each row.

## Negative control (raw signal, k = n = 1000: every window is its own type)

By Sec. 2.3 the ceiling is chance, because no test type is seen in training
(`frac_test_types_seen_in_train` = 0.0). Observed control scores:

| Property | Control (95% CI) | Majority-class chance |
|---|---|---|
| trend | 0.335 [0.316, 0.354] | 0.312 |
| frequency | 0.122 [0.099, 0.143] | 0.154 |
| stationarity | 0.500 [0.478, 0.519] | 0.511 |
| anomaly | 0.484 [0.475, 0.493] | 0.511 |
| change_point | 0.498 [0.458, 0.536] | 0.511 |
| seasonality (R²) | −0.968 [−1.058, −0.858] | ≈0 or below |

Control accuracy falls to chance, so the construction does not leak labels.

## Reading the numbers

* `control_majority_chance` is the test accuracy of always predicting the train-majority control
  label. With only k=20 randomly-labelled clusters of unequal size, this chance level is often
  high (0.70–0.82 for trend, stationarity, anomaly and change_point). Much of the control
  accuracy at k=20 is therefore chance, not memorisation. Compare control against this column,
  not against 1/|Y|.
* For seasonality, control R² can be negative (the probe is worse than predicting the mean on
  test), so selectivity can exceed 1. It is reported unclipped.
* The linear–MLP accessibility gap was not recomputed here (optional in the spec; not cheap to
  replicate faithfully).
