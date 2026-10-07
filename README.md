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

| Property | Raw signal | Hand-crafted (8-D) | Best TSFM |
|---|---:|---:|---:|
| Trend | 1.000 | 1.000 | 1.000 |
| Seasonality (R²) | 0.789 | 0.959 | 1.000 |
| Frequency | 1.000 | 0.944 | 1.000 |
| Stationarity | 0.496 | 1.000 | 1.000 |
| **Anomaly** | 0.485 | **0.858** | **0.753** (TimesFM 2.0) |
| Change point | 1.000 | 1.000 | 1.000 |

- **Five of six properties are recovered without any model** at or near the ceiling, so probe
  accuracy on them says little about the model.
- **Anomaly is nearly solved by one input statistic.** `max(|x|)` alone reaches 0.907, the
  Bayes-optimal rule 0.906 (`outputs/per_feature_anomaly/`, `outputs/bayes_ceiling/`). Erasing linear
  information about the sufficient statistics with LEACE removes most of the models' anomaly signal
  (`outputs/statistic_erasure/`).
- **On mixed-type anomalies the ordering reverses for the strongest models.** TimesFM 2.0 (0.730) and
  PatchTST-FM (0.732) exceed the hand-crafted vector (0.701), and keep accuracy well above chance after
  every hand-crafted feature is erased (`outputs/realistic_anomaly_v2/`,
  `outputs/statistic_erasure/realistic_handcrafted/`).

## Quickstart (CPU, <10 minutes)

```bash
curl -L -o TSFMI.zip "https://anonymous.4open.science/api/repo/TSFMI/zip"
unzip TSFMI.zip && cd TSFMI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
make smoke
```

`make smoke` reproduces one cell of the canonical baseline grid, the per-feature attribution, and the
test suite. No GPU is required.

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
| `make view-sensitivity` | `outputs/view_sensitivity/` | MOMENT and GPT4TS under mean, max, and last-token views |
| `make ucr-probe` | `outputs/ucr_tsfm_probe_v2/`, `outputs/ucr_tsfm_probe_ext/` | Minority-class probes on six UCR datasets |
| `make wilinski-ldr` | `outputs/wilinski_moment_ldr/` | Wiliński et al.'s separability pipeline with non-model controls |
| `make multicapacity-probe` | `outputs/multicapacity_probe/` | Linear, MLP (64–1024), and random-forest probes |
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
| Chronos-Bolt | `amazon/chronos-bolt-small` | 6 enc. + 6 dec. | 48 M | mean-pool (encoder) |
| Timer | `thuml/timer-base-84m` | 8 | 84 M | mean-pool |
| TimesFM 2.0 | `google/timesfm-2.0-500m-pytorch` | 50 | 494 M | mean-pool |
| Moirai 2.0 | `Salesforce/moirai-2.0-R-small` | 6 | 11.4 M | mean-pool |
| GPT4TS | `gpt2` (frozen backbone) | 12 | 124 M | PCA-512 (train-only) |
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
