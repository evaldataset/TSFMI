# Supplementary Materials — TSFMI

> NeurIPS 2026 Evaluations & Datasets (E&D) Track submission.
> Anonymous double-blind release.

This folder contains the reviewer-facing supplementary bundle for the paper
**"TSFMI: A Baseline-Controlled Evaluation Protocol for Time-Series
Foundation Model Representations"**.

## Contents

| File | Purpose |
|---|---|
| `tsfmi_supplementary.zip` | Anonymized code/config artefact for OpenReview supplementary upload (~320 KB, 155 files) |
| `README.md` | This file — reproduction map for reviewers |

The supplementary zip contains:

- `src/` — 7 model wrappers, 12 synthetic generators, probe / LEACE / CKA / steering library
- `scripts/` — extraction, canonical benchmark, baselines, figures, additional CHECK.md experiments
- `tests/` — 170 automated tests (`make test`)
- `configs/` — per-experiment configuration references
- `LICENSE` — MIT (code) + pointers to original licenses for the public real-world datasets
- `CROISSANT.json` — Croissant 1.0 + RAI metadata describing the synthetic data
- `Makefile`, `pyproject.toml`, `requirements.txt`, `README.md`

The full paper PDF is `outputs/paper/latex/main.pdf` (48 pages: 9 main + 2 references + 37 appendix).

## Reviewer quick start (CPU, <10 min)

```bash
unzip tsfmi_supplementary.zip && cd tsfmi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
make smoke
```

`make smoke` reproduces one cell of the canonical baseline grid (HC anomaly =
0.858) plus per-feature attribution (kurtosis alone = 0.859, max-abs alone =
0.907) and runs the test suite. **No GPU required.** Pretrained TSFM
extraction is deferred to `make extract-representations` (~40 GPU-hours on a
single A100).

## Full reproduction (~48 A100-hours)

| Step | Make target | Approximate cost |
|---|---|---|
| 1. Extract frozen representations (7 models × 6 datasets) | `make extract-representations` | 40 A100-hours |
| 2. Canonical 60/20/20 + 5-seed bootstrap | `make reproduce-canonical` | 6 hours (CPU+GPU I/O) |
| 3. Canonical baselines (HC, raw, random projection) | `make reproduce-baselines` | <1 minute (CPU) |
| 4. Manifest count | `make manifest` | <1 minute |
| 5. Figures + paper compile | `make reproduce-figures && make paper` | 10 minutes |
| 6. CHECK.md G1–G7 (additional confirmatory experiments) | see below | ~6 hours total |

The end-to-end target is `make reproduce-all-from-scratch`.

## Confirmatory additional experiments (CHECK.md G1–G7)

Each is a one-line Make target that regenerates a single output bundle:

| Target | Output | Purpose |
|---|---|---|
| `make per-feature-anomaly` | `outputs/per_feature_anomaly/` | Per-HC-feature attribution (kurtosis sufficiency) |
| `make paired-wilcoxon` | `outputs/paired_tests/` | Paired Wilcoxon HC vs each TSFM, Holm-corrected |
| `make rocket-baselines` | `outputs/rocket_baselines/` | 1024-kernel ROCKET baseline |
| `make dataset-seed-bootstrap` | `outputs/dataset_seed_bootstrap/` | 5×5 (data-seed × split-seed) cell CI |
| `make realistic-anomaly` | `outputs/realistic_anomaly/` | Mixed-type anomaly on structured background |
| `make mdl-probe` | `outputs/mdl_probe/` | Voita-Titov MDL probe, 4 models × 3 properties |
| `make cross-leace-bootstrap` | `outputs/cross_leace_bootstrap/` | Bootstrap entanglement diagnostic |
| `make benchmark-time` | `outputs/paper/timing.json` | Logged wall-clock instrumentation |

## Synthetic data generation

All synthetic datasets are produced on demand by deterministic generators
(`src/datasets/synthetic.py`) seeded with `seed=42` for the canonical
artefacts. The data is therefore code, not a pre-existing dataset; the
`CROISSANT.json` file captures the dataset-level metadata for the
seed=42 instantiation:

- **Trend** — 3-class (up / down / flat), 1000 × 512
- **Seasonality** — period regression, periods ∈ {8, 16, 32, 64}, 1000 × 512
- **Frequency** — 8-class, 1000 × 512
- **Stationarity** — binary (noise vs random walk), 1000 × 512
- **Anomaly** — binary (single ±5σ spike), 1000 × 512 — **kurtosis is a sufficient statistic**
- **Change Point** — binary (mean+variance shift at midpoint), 1000 × 512

Plus five hard variants (trend-hard, frequency-hard, stationarity-hard,
anomaly-hard, change-point-hard) that add structured background and reduce
saturation. See paper §3.3 and Appendix A.7 for full task definitions.

## Real-world datasets (used, not redistributed)

| Dataset | License | Source |
|---|---|---|
| ETTh1, ETTh2 | CC-BY 4.0 | https://github.com/zhouhaoyi/ETDataset |
| Weather (Jena Climate) | CC-BY 4.0 | Max Planck Institute for Biogeochemistry |
| Electricity (LD2011_2014) | CC-BY 4.0 | UCI ML Repository |
| Traffic (PEMS) | public domain | California DoT |
| Exchange Rate | per distributors | research use |
| UCR Archive | per archive terms | https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ |

The `data/` directory in the supplementary zip contains the loader code only;
users obtain their own copy of each dataset under its original license.

## Pretrained model checkpoints

All 7 TSFM checkpoints are loaded via HuggingFace `transformers.AutoModel`
(or `momentfm` for MOMENT) — auto-downloaded from the official model card on
first run, no TSFMI-side redistribution.

| Model | HuggingFace ID | Layers | Params |
|---|---|---|---|
| MOMENT | `AutonLab/MOMENT-1-large` | 24 | 385 M |
| Chronos-Bolt | `amazon/chronos-bolt-small` | 6 | 710 M (T5 enc-dec) |
| PatchTST | `ibm-granite/granite-timeseries-patchtst` | 3 | 1.5 M |
| GPT4TS | `gpt2` (frozen backbone) | 12 | 124 M |
| Timer | `thuml/timer-base-84m` | 8 | 84 M |
| TimesFM | `google/timesfm-2.0-500m-pytorch` | 50 | 494 M |
| Moirai | `Salesforce/moirai-2.0-R-small` | 6 | 11.4 M |

## Compute requirements

Single NVIDIA A100 80GB. Logged times in `outputs/paper/timing.json`:

- Extraction (7 × 6): ~40 hours
- Canonical probing + baselines: <1 hour wall-clock (CPU-bound sklearn fits)
- LEACE / CKA / steering: ~5 hours
- Figures + paper compile: ~10 minutes
- **Total: ~48 A100-hours** including multi-seed and enhancement runs.

CPU-only smoke: <10 minutes (`make smoke`).

## Anonymity confirmation

The supplementary zip is generated by
`scripts/prepare_anonymous_release.sh`, which excludes:

- internal review docs (AUDIT.md, CHECK.md, REVIEW*.md, REVISION.md, PLAN.md, AGENTS.md, PROVENANCE.md, SUBMISSION_CHECKLIST.md)
- frozen representation tensors (837 GB), per-extraction logs, byte-code, virtual envs
- absolute paths, usernames, API tokens, git history

A triple-redundant safety sweep at the end of the script fails the build if
any sentinel file leaks. See zip output for the audit trail.
