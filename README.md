# TSFMI: Baseline-Controlled Probing of Time-Series Foundation Models

Code and result artefact for **TSFMI**, a diagnostic protocol that asks when a probe score on a frozen
time-series foundation model (TSFM) representation says something about the model rather than about
the task. Every model and every non-model control goes through the same estimator, splits,
statistics, and leak-free extraction pipeline.

- 📂 **Code (anonymous mirror)**: https://anonymous.4open.science/r/TSFMI/
- 🤗 **Dataset**: https://huggingface.co/datasets/EvalData/TSFMI (Croissant 1.0 + RAI)

## Headline results

Canonical protocol (v2): n = 1000 windows of length 512, 60/20/20 split, split seeds 0–4,
validation-selected layer, test score reported. All values come from `outputs/canonical_v2/`.

| Property | Raw signal | Hand-crafted (8-D) | Best model |
|---|---:|---:|---:|
| Trend | 1.000 | 1.000 | 1.000 |
| Seasonality (R²) | 0.789 | 0.959 | 1.000 |
| Frequency | 1.000 | 0.944 | 1.000 |
| Stationarity | 0.496 | 1.000 | 1.000 |
| **Anomaly** | 0.485 | **0.858** | **0.756** (TimesFM 2.0) |
| Change point | 1.000 | 1.000 | 1.000 |

- **Strongest control.** Across 20 task settings (6 canonical properties, 5 hard variants, a
  mixed-type anomaly generator under 3 data seeds, 6 UCR datasets) the best model exceeds the strongest
  non-model control in 6, by at most 0.019, below what 200 test windows can resolve (`sc/*` keys).
- **Resolution.** With 50,000 freshly generated windows (10,000 paired test windows) the ties resolve
  in favour of the controls: on the realistic anomaly task `tsfresh` leads the best model by 0.050, on
  the canonical task `max|x|` by 0.057 (McNemar p < 0.001; `outputs/resolution_study/`).
- **Conditional probing.** Mean-pooled model representations add nothing beyond ±0.01 accuracy
  (TOST) to the union of hand-crafted, `tsfresh`, and ROCKET-style features. Through a cross-fitted
  attention pooling of its tokens, TimesFM adds +0.013 [0.008, 0.019] (Holm p < 0.001); no other model
  does.
- **Pretraining, not architecture.** Re-initialised models match the pretrained ones on 34 of 35
  saturated (model, property) cells and fall towards chance on anomaly tasks
  (`outputs/canonical_v2_randinit/`).
- **Embedding store.** On 113 UCR datasets, 1-NN over the embeddings of six of the seven models beats
  Euclidean distance and is level with DTW and ROCKET-style features (`outputs/embedding_store/`);
  TimesFM's one-step forecast surprise (AUC 0.889) is slightly below an AR(16) residual detector
  (0.895) that is about 930 times cheaper (`outputs/forecast_surprise/`).
- **Erasure.** A label proxy as predictive as the erased statistics removes as much signal; an exact
  bound, corr ≤ sqrt(1 − R²), explains why (`outputs/erasure_bound/`).

Intervals are corrected resampled-t intervals (Nadeau & Bengio, 2003) for the overlapping split seeds.

## Quickstart (CPU, <10 minutes)

```bash
curl -L -o TSFMI.zip "https://anonymous.4open.science/api/repo/TSFMI/zip"
unzip TSFMI.zip && cd TSFMI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
make smoke
```

`make smoke` reproduces one cell of the canonical baseline grid, the per-feature attribution, and the
test suite. No GPU is required. `make reproduce-one-cell` extracts one model from scratch
(default Chronos-Bolt on canonical anomaly; `ONE_MODEL=...` to change), runs the canonical v2 probe,
and compares the result with the committed `outputs/canonical_v2` value (a few minutes on one GPU).

## Full reproduction

```bash
make reproduce-icde    # every result the manuscript reads (GPU; many hours)
```

`reproduce-icde` runs the targets below in order. Representations go to `REPR_ROOT`
(default `outputs/representations_v2`, gitignored); result JSONs go to `outputs/`.

| Target | Output | Content |
|---|---|---|
| `make reproduce-canonical-v2` | `outputs/canonical_v2/` | 7 TSFMs + raw / hand-crafted / random-projection controls, one protocol and environment |
| `make realistic-anomaly-v2` | `outputs/realistic_anomaly_v2/` | Mixed-type anomaly generator through the same pipeline |
| `make hard-variants-v2` | `outputs/hard_variants_v2/` | Five hard variants, models and controls |
| `make dataseed-v2` | `outputs/dataseed_v2/` | Canonical and realistic anomaly for generator seeds 43 and 44 |
| `make hl-selectivity-v2` | `outputs/hl_selectivity_v2/` | Hewitt & Liang control-task selectivity (k = 20, 50; k = n negative control) |
| `make statistic-recoverability` | `outputs/statistic_recoverability/` | Linear recoverability of the anomaly statistics |
| `make statistic-erasure` | `outputs/statistic_erasure/` | LEACE erasure of statistics / hand-crafted features with random and label controls |
| `make anomaly-diagnostics` | `outputs/anomaly_diagnostics/` | Per-anomaly-kind recall and label-efficiency curves |
| `make view-sensitivity` | `outputs/view_sensitivity/` | MOMENT and GPT-2 under mean, max, and last-token views |
| `make ucr-probe` | `outputs/ucr_tsfm_probe_v2/`, `outputs/ucr_tsfm_probe_ext/` | Minority-class probes on six UCR datasets |
| `make wilinski-ldr` | `outputs/wilinski_moment_ldr/` | Wiliński et al.'s separability pipeline with non-model controls |
| `make capacity-probe-v2` | `outputs/capacity_probe_v2/` | Linear, MLP (64–1024), and random-forest probes; per-probe validation-selected layer, all models |
| `make strong-controls-ext` | `outputs/strong_controls_ext/` | ROCKET-style and `tsfresh` controls on realistic, hard, UCR, and canonical tasks |
| `make canonical-v2-randinit` | `outputs/canonical_v2_randinit/` | Every model with re-initialised weights, canonical protocol |
| `make resolution-study` | `outputs/resolution_study/` | n = 50,000 study: paired tests, TOST, conditional probing, views, surrogates, magnitude sweep (arrays in `RESOLUTION_STORE`) |
| `make forecast-surprise` | `outputs/forecast_surprise/` | Label-free scores: TimesFM / Chronos-Bolt forecast surprise vs AR(16), first differences, matrix profile |
| `make embedding-store` | `outputs/embedding_store/` | 1-NN on every fixed-length UCR 2018 dataset (`UCR2018=<archive root>`): ED, DTW, ROCKET-style, 7 TSFMs, random-init TSFMs |
| `make ucr-conditional` | `outputs/ucr_conditional/` | Conditional probes on the six binary UCR datasets |
| `make erasure-bound` | `outputs/erasure_bound/` | R² of the label on each erased concept and the implied correlation cap |
| `make icde-numbers` | `outputs/paper_numbers/` | Every manuscript number (`numbers.tex`) and figure, generated from the result files |
| `make per-feature-all-properties`, `make per-feature-anomaly` | `outputs/per_feature_*` | Single-feature ablations |
| `make bayes-ceiling`, `make rocket-baselines`, `make strong-baselines` | `outputs/bayes_ceiling/`, `outputs/rocket_baselines/`, `outputs/strong_baselines/` | Bayes rule, ROCKET-style and `tsfresh` controls |
| `make representation-storage`, `make extraction-throughput` | `outputs/representation_storage/`, `outputs/extraction_throughput/` | Storage footprint and extraction cost |

Earlier targets (`make reproduce-all`, `make paired-wilcoxon`, `make mdl-probe`, …) are kept for the
previous version of the study; `make help` lists them.

## Repository layout

```
TSFMI/
├── src/                 11 model wrappers, 11 synthetic generators, probe/LEACE/CKA library
├── scripts/             extraction, canonical benchmark, controls, and every analysis
├── tests/               automated tests (CPU subset runs in CI)
├── configs/             reference YAML configurations (scripts take CLI flags)
├── outputs/             committed result JSONs (tensors are not committed)
├── data/                README only; real-world datasets come from their original sources
├── LICENSE              MIT (code) + pointers to public-dataset original licenses
├── CROISSANT.json       Croissant 1.0 + RAI metadata describing the synthetic data
├── Makefile             one entry point per reproducible result
└── pyproject.toml       Python 3.10+, pinned dependencies
```

## HuggingFace dataset

The synthetic data (11 task configurations, 60/20/20 splits) is mirrored on the Hub:

```python
from datasets import load_dataset

ds = load_dataset("EvalData/TSFMI", "anomaly")
print(ds)            # train: 600, validation: 200, test: 200
print(ds["train"][0]["sequence"][:8], ds["train"][0]["label"])
```

Configurations: `trend`, `seasonality`, `frequency`, `stationarity`, `anomaly`, `change_point` and
the `*_hard` variants (no `seasonality_hard`). Each row has `sequence` (list[float] × 512), `label`,
`seed=42`, `split_seed=0`. The dataset card on the Hub has the full schema and Croissant + RAI metadata.

## Evaluated checkpoints

Checkpoints are downloaded from their official Hugging Face pages on first use; TSFMI does not
redistribute weights. Layer counts and sizes are those of the checkpoints the code loads.

| Model | Hugging Face ID | Layers | Params | Probed view |
|---|---|---|---|---|
| MOMENT | `AutonLab/MOMENT-1-large` | 24 | 385 M | PCA-512 (train-only) |
| Chronos-Bolt | `amazon/chronos-bolt-small` | 6 encoder blocks probed | 48 M | mean-pool |
| Timer | `thuml/timer-base-84m` | 8 | 84 M | mean-pool |
| TimesFM 2.0 | `google/timesfm-2.0-500m-pytorch` | 50 | 494 M | mean-pool |
| Moirai 2.0 | `Salesforce/moirai-2.0-R-small` | 6 | 11.4 M | mean-pool |
| GPT-2 (random patch embedding) | `gpt2` (frozen; untrained linear patch embedding, not a GPT4TS checkpoint) | 12 | 124 M | PCA-512 (train-only) |
| PatchTST-FM | `ibm-granite/granite-timeseries-patchtst-fm-r1` | 20 | 258 M | mean-pool |

PatchTST-FM replaces an earlier PatchTST checkpoint that was a supervised ETTh1 forecaster
(`outputs/patchtst_replacement/`).

## Validation

```bash
make test     # pytest tests/ -q
make lint     # ruff check
```

## Citation

```bibtex
@misc{tsfmi2026,
  title  = {{TSFMI}: A Baseline-Controlled Diagnostic Protocol for Time-Series Foundation Model Representations},
  author = {Anonymous Authors},
  year   = {2026},
  url    = {https://anonymous.4open.science/r/TSFMI}
}
```
