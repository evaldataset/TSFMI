# TSFMI: Baseline-Controlled Probing of Time-Series Foundation Models

Anonymous code+data artefact for the paper **"TSFMI: A Baseline-Controlled
Evaluation Protocol for Time-Series Foundation Model Representations"**
(NeurIPS 2026 Evaluations & Datasets Track, double-blind submission).

- 📂 **Code (anonymous mirror)**: https://anonymous.4open.science/r/TSFMI/
- 🤗 **Dataset**: https://huggingface.co/datasets/EvalData/TSFMI (Croissant 1.0 + RAI)
- 📄 **Paper**: see OpenReview supplementary (`outputs/paper/latex/main.pdf` locally; intentionally excluded from this public mirror until camera-ready)

## Headline result

Under a canonical 60/20/20 train/val/test protocol with matched sklearn
estimators, 5 seeds, and bootstrap 95% CIs:

| Property | HC | Best TSFM | Gap |
|---|---:|---:|---:|
| Trend | 1.000 | 1.000 | 0.000 |
| Seasonality (R²) | 0.959 | 1.000 | +0.041 |
| Frequency | 0.944 | 1.000 | +0.056 |
| Stationarity | 1.000 | 1.000 | 0.000 |
| **Anomaly** | **0.858** | **0.753** | **−0.105** |
| Change Point | 1.000 | 1.000 | 0.000 |

**Anomaly inversion** is the only meaningful gap, and it has a direct
mechanistic explanation (kurtosis sufficiency; Appendix A.8 of the paper).
A single kurtosis feature alone reaches 0.859, and `max(|x|)` alone reaches
0.907 — six of seven TSFMs cannot regress kurtosis from their representations.

## Quickstart (CPU, <10 minutes)

```bash
curl -L -o TSFMI.zip "https://anonymous.4open.science/api/repo/TSFMI/zip"
unzip TSFMI.zip && cd TSFMI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
make smoke
```

`make smoke` reproduces one cell of the canonical baseline grid + per-feature
kurtosis attribution + the test suite. No GPU required.

## Full reproduction (~48 A100-hours)

```bash
make extract-representations    # ~40 GPU-hours, 7 models × 6 datasets
make reproduce-all              # baselines + canonical + figures + paper
```

## Confirmatory additional experiments (CHECK.md G1–G7)

Each is a one-line Make target that writes results to `outputs/<exp>/`:

| Target | Output | Purpose |
|---|---|---|
| `make per-feature-anomaly` | `outputs/per_feature_anomaly/` | Per-HC-feature attribution (kurtosis sufficiency) |
| `make paired-wilcoxon` | `outputs/paired_tests/` | Paired Wilcoxon HC vs each TSFM, Holm-corrected |
| `make rocket-baselines` | `outputs/rocket_baselines/` | 1024-kernel ROCKET baseline |
| `make dataset-seed-bootstrap` | `outputs/dataset_seed_bootstrap/` | 5×5 (data × split) seed CI |
| `make realistic-anomaly` | `outputs/realistic_anomaly/` | Mixed-type anomaly on structured background |
| `make mdl-probe` | `outputs/mdl_probe/` | Voita-Titov MDL probe |
| `make cross-leace-bootstrap` | `outputs/cross_leace_bootstrap/` | Bootstrap entanglement diagnostic |
| `make benchmark-time` | `outputs/paper/timing.json` | Wall-clock instrumentation |

## Repository layout

```
TSFMI/
├── src/                 7 model wrappers, 12 synthetic generators, probe/LEACE/CKA library
├── scripts/             extraction, canonical benchmark, baselines, additional experiments
├── tests/               170 automated tests
├── configs/             reference YAML configurations
├── outputs/             paper-cited confirmatory artefacts (canonical/, leace/, cka/, ...)
├── data/                README only — users obtain real-world datasets from original sources
├── LICENSE              MIT (code) + pointers to public-dataset original licenses
├── CROISSANT.json       Croissant 1.0 + RAI metadata describing the synthetic data
├── Makefile             one-line entry points for every reproducible target
└── pyproject.toml       Python 3.10+, torch 2.10.0, transformers 4.57.6 (pinned)
```

## HuggingFace dataset

The synthetic data (11 task configurations, 60/20/20 splits) is mirrored on
the Hub:

```python
from datasets import load_dataset

ds = load_dataset("EvalData/TSFMI", "anomaly")
print(ds)            # train: 600, validation: 200, test: 200
print(ds["train"][0]["sequence"][:8], ds["train"][0]["label"])
```

Configurations: `trend`, `seasonality`, `frequency`, `stationarity`, `anomaly`,
`change_point` + `*_hard` variants. Each row has `sequence` (list[float]×512),
`label`, `seed=42`, `split_seed=0`. See the dataset card on the Hub for full
schema and Croissant + RAI metadata.

## Pretrained model checkpoints

All 7 TSFM checkpoints are loaded via HuggingFace `transformers.AutoModel`
(or `momentfm` for MOMENT) — auto-downloaded from the official model card on
first run; no TSFMI-side redistribution.

| Model | HuggingFace ID | Layers | Params |
|---|---|---|---|
| MOMENT | `AutonLab/MOMENT-1-large` | 24 | 385 M |
| Chronos-Bolt | `amazon/chronos-bolt-small` | 6 | 710 M |
| PatchTST | `ibm-granite/granite-timeseries-patchtst` | 3 | 1.5 M |
| GPT4TS | `gpt2` (frozen backbone) | 12 | 124 M |
| Timer | `thuml/timer-base-84m` | 8 | 84 M |
| TimesFM | `google/timesfm-2.0-500m-pytorch` | 50 | 494 M |
| Moirai | `Salesforce/moirai-2.0-R-small` | 6 | 11.4 M |

## Validation

```bash
make test     # 170/170 passing
make lint     # ruff clean
```

## Citation

```bibtex
@misc{tsfmi2026,
  title  = {{TSFMI}: A Baseline-Controlled Evaluation Protocol for Time-Series Foundation Model Representations},
  author = {Anonymous Authors},
  howpublished = {Anonymous submission to the NeurIPS 2026 Evaluations \& Datasets Track},
  year   = {2026},
  url    = {https://anonymous.4open.science/r/TSFMI}
}
```
