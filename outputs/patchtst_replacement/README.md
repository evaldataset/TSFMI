# PatchTST checkpoint replacement

**Decision:** replace the PatchTST checkpoint with
**`ibm-granite/granite-timeseries-patchtst-fm-r1`** (PatchTST-FM-r1, Apache-2.0, ungated), as
canonical cell `patchtst_fm_meanpool`. The legacy checkpoint is **not** fixed by switching to
`ibm-granite/granite-timeseries-patchtst`: that repo holds the *same* supervised ETTh1 model.

All evidence below was fetched on 2026-09-25 from `https://huggingface.co/<id>/raw/main/README.md`,
`.../config.json` and `https://huggingface.co/api/models/<id>`.

## 1. What the legacy checkpoint actually is

`namctin/patchtst_etth1_forecast` now redirects to a gated repo:

> Access to model ibm-research/testing-patchtst_etth1_forecast is restricted. You must have access
> to it and be authenticated to access it.

`ibm-granite/granite-timeseries-patchtst` is the same model. Its `config.json` has
`"_name_or_path": "namctin/patchtst_etth1_forecast"` and model-index name `patchtst_etth1_forecast`.
The card says:

> # PatchTST model pre-trained on ETTh1 dataset
>
> This repository contains a pre-trained `PatchTST` model encompassing all seven channels of the
> `ETTh1` dataset. [...] produces a Mean Squared Error (MSE) of 0.3881 on the `test` split of the
> `ETTh1` dataset when forecasting 96 hours into the future

> ### Training Data
> [`ETTh1`/train split](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv).

So "PatchTST-Pre" is a supervised 96-step ETTh1 forecaster (3 layers, d_model 128, patch 12,
`num_input_channels: 7`), not a foundation model. Every local artifact labelled
`patchtst_pretrained` was already produced from `ibm-granite/granite-timeseries-patchtst`
(`scripts/extract_representations.py` `DEFAULT_CHECKPOINTS`, and
`outputs/representations/patchtst_pretrained/synthetic_anomaly/metadata.json`). Paper B's sentence
that it "substitute[s] a public pretrained PatchTST (granite) [...] consistent with the canonical
model" is therefore describing identical weights, not a substitute.

## 2. Candidates evaluated

Search used: `https://huggingface.co/api/models?search=patchtst&sort=downloads` (plus `PatchTST`,
`patch_tst`, `patch-tst`) and `?filter=patchtst`.

Criteria, in order: (1) training data disclosed and disjoint from ETT / Weather (Jena) / Electricity
/ Traffic / Exchange-Rate; (2) genuinely PatchTST; (3) ungated, license allows research use;
(4) loadable with installed transformers 4.57.6 / torch 2.10.0.

| Checkpoint | (1) Training data (quote) | (2) Arch | (3) Access / license | (4) Loads | Verdict |
|---|---|---|---|---|---|
| `ibm-granite/granite-timeseries-patchtst` | "`ETTh1`/train split" | HF PatchTST, 3L, d128 | ungated, apache-2.0 | yes | **Reject.** Same weights as legacy; trained on ETTh1. |
| `namctin/patchtst_etth1_forecast` | (gated; same model) | same | **gated** | no | **Reject.** |
| `ibm-research/test-patchtst` | "For the official pretrained PatchTST, please see [...]granite-timeseries-patchtst"; config `_name_or_path: namctin/patchtst_etth1_forecast` | same | ungated, apache-2.0 | yes | **Reject.** Copy of ETTh1 model. |
| `onnx-community/granite-timeseries-patchtst` | "[granite-timeseries-patchtst] with ONNX weights to be compatible with Transformers.js" | same (ONNX) | ungated | no (ONNX) | **Reject.** |
| `ibm-research/patchtst-etth1-pretrain` | "PatchTST model pre-trained on ETTh1 dataset" | HF PatchTST (masked pretraining) | ungated, apache-2.0 | yes | **Reject.** ETTh1. |
| `ibm-research/testing-patchtst_etth1_pretrain` | "fine-tuned version of [](https://huggingface.co/) on an unknown dataset" (name: etth1) | HF PatchTST | ungated | yes | **Reject.** |
| `Coaster41/patchtst-tsmixup`, `Coaster41/patchtst_tsmixup_final` | "This model is a fine-tuned version of [](https://huggingface.co/) on an unknown dataset." | HF PatchTST (3L, d1024 / d256) | ungated, no license | yes | **Reject.** Data undisclosed. |
| `lvizcaya/patchtst-exchange-rate` | "on an unknown dataset" (name: exchange-rate) | HF PatchTST | ungated | yes | **Reject.** Undisclosed, likely Exchange-Rate. |
| `SamuelM0422/PatchTST-Hourly-Electricity-Demand-Brazil` | dataset `Hourly-Electricity-Demand-Brazil-Dataset` | HF PatchTST | ungated | yes | **Reject.** Single-dataset supervised forecaster, not a pretrained model. |
| `GilpinLab/panda-72M` (also `panda`, `panda_mlm`) | "We train Panda on a novel synthetic, extensible dataset of 2 × 10^4 chaotic dynamical systems [...] Trained purely on simulated data" | **Modified** PatchTST: config has `channel_attention: true`, `use_dynamics_embedding: true`, `num_poly_feats: 188`, `num_rff: 376`, `norm_type: rmsnorm`, which stock `PatchTSTForPrediction` does not implement | ungated, cc-by-nc-4.0 | **no** (needs the `panda` repo; stock HF would drop weights) | **Runner-up.** Fully disjoint data, but custom architecture that won't load in stock transformers, and a chaotic-ODE domain model rather than a general TSFM. |
| **`ibm-granite/granite-timeseries-patchtst-fm-r1`** | see §3 | PatchTST-style FM: plain pre-norm Transformer, non-overlapping 16-step patches | **ungated, apache-2.0** | **yes**, via `tsfm_public` (granite-tsfm 0.3.9) on transformers 4.57.6 | **Chosen.** |
| `ibm-research/patchtst-fm-r1` | "Datasets from [GiftEvalPretrain]" + KernelSynth + TSMixup | same as FM-r1 | ungated, **cc-by-nc-sa-4.0** ("non-commercial, research version") | yes | Equivalent alternative. The Granite build has the more permissive license. |
| `ibm-granite/granite-timeseries-patchtst-fm-r2` | FM-r1 corpus + "CauKer dataset. ~0.5M sequences" | "Plain-transformer blocks are replaced by Conformer blocks", 30 blocks, overlapping patches (16/8), ~385M | ungated, openmdw-1.0 / apache-2.0 | yes (tsfm_public) | **Reject.** Conformer blocks (convolution sublayers) move it further from PatchTST. |

TinyTimeMixer/TTM and PatchTSMixer were not needed, because a PatchTST-family candidate passed.

## 3. Chosen model: PatchTST-FM-r1 and its leakage, stated precisely

Card (`ibm-granite/granite-timeseries-patchtst-fm-r1`):

> PatchTST-FM (patched time-series transformer-based foundation model) essentially has the
> architectural simplicity of PatchTST, but differs in some crucial ways. [...] residual blocks in the
> input and output projections; a quantile head to support probabilistic forecasting; [...] trained
> with reconstruction-loss objective

> The training data composes three separate sources:
> - Selected subset of datasets from [GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain),
> - Custom synthesized data: Based on KernelSynth [...]
> - A TSMixup dataset, based on the same process described in the Chronos paper, but using only
>   datasets which are not in the GIFT-Eval evaluation set.

Paper (Wen et al., arXiv:2602.06909, §4.1): "Real Data: We utilize the GIFT-Eval-Pretrain
collection, consisting of 132 datasets [...] Zero-shot Model: Trained on Real Data + 10M
KernelSynth + 4M Clean TSMixup [...] a 'Clean' set of 4 million series with all test-set overlaps
removed."

The GIFT-Eval evaluation set (`Salesforce/GiftEval`) contains `ett1`, `ett2`, `electricity`,
`jena_weather` (among others), so the TSMixup component excludes them by construction. Below, the
`Salesforce/GiftEvalPretrain` directory listing (152 dataset directories) is compared with the five real-world
TSFMI datasets:

| TSFMI dataset | In FM-r1 pretraining? | Evidence |
|---|---|---|
| **ETTh1 / ETT family** | **No** | no `ett*` in GiftEvalPretrain; `ett1`/`ett2` are GIFT-Eval *eval* sets, excluded from clean TSMixup |
| **Electricity** (UCI 321 clients) | **No** | `electricity` is a GIFT-Eval eval set; GiftEvalPretrain has other load data (`australian_electricity_demand`, `elecdemand`, `lcl`, `buildings_900k`, `gfc*_load`) but not UCI ElectricityLoadDiagrams |
| **Weather** (Jena) | **No** (direct) | `jena_weather` is a GIFT-Eval eval set. GiftEvalPretrain `weather` is Monash/BOM Australian, `oikolab_weather` and `era5_*` are gridded reanalysis. Indirect only: ERA5 covers the Jena grid cell |
| **Traffic** (862 PeMS sensors) | **Likely yes** | GiftEvalPretrain contains `traffic_hourly` and `traffic_weekly`: "862 [...] 1H [...] Caltrans", the same 862-sensor Caltrans PeMS source as `traffic.txt`. The card says "selected subset", so inclusion can't be confirmed, but the paper uses the full collection |
| **Exchange-Rate** (8 currencies, daily) | **No** (direct) | no `exchange_rate` in GiftEvalPretrain. `fred_md` includes monthly USD exchange-rate series for some of the same currencies (indirect, different resolution) |

**Consequence:** the new model can be used on ETTh1, Weather, Electricity and Exchange-Rate
real-world probes. Any PatchTST-FM **Traffic** result must be marked as possibly in-pretraining (or
dropped). The legacy model overlaps with ETTh1 only (see §6).

## 4. Integration and architecture facts (from the loaded model)

- `src/models/patchtst_wrapper.py`: `DEFAULT_PATCHTST_CHECKPOINT =
  "ibm-granite/granite-timeseries-patchtst-fm-r1"`. The legacy ids stay reachable as
  `LEGACY_ETTH1_CHECKPOINT` (namctin, gated) and `LEGACY_GRANITE_ETTH1_CHECKPOINT`, via
  `load(checkpoint=...)`. The loader branches on `config.json` `model_type` (`patchtst_fm` goes to
  `tsfm_public`, anything else to HF `PatchTSTForPrediction`; `""` still means random init).
- `scripts/extract_representations.py`: new model key `patchtst_fm`. `patchtst_pretrained` is
  unchanged (legacy).
- `scripts/run_realistic_anomaly_tsfm.py`: its `patchtst_pretrained` entry used the wrapper
  default, so it now pins `LEGACY_GRANITE_ETTH1_CHECKPOINT` explicitly and never reports FM-r1
  under the legacy label.
- Encoding: the univariate input (length T ≤ 8192) is left-padded with its mean to the 8192
  context, the pad is flagged in `pad_mask`, and `pred_mask` is all False (no forecast slots). The
  backbone prunes the all-pad patches, so each hooked block sees only the ceil(T/16) observed patches.
  Normalization (RevIN + asinh) uses observed values only. Sanity check: the median reconstruction
  of a noisy sine has correlation 0.991 with the input.
- Dependency: `granite-tsfm==0.3.9` installed with `--no-deps`, plus `datasets==5.0.1` (which
  `tsfm_public/__init__` imports). A plain install would upgrade torch to 2.11 / CUDA 13, and its
  `scikit-learn<1.8` pin conflicts with `requirements.txt` (`scikit-learn==1.8.0`). For that reason
  it is not added to `requirements.txt`, following the precedent of `chronos-forecasting` and
  `uni2ts`. Install command: `pip install --no-deps granite-tsfm==0.3.9 && pip install datasets==5.0.1`.

| Fact (loaded `PatchTSTFMForPrediction`) | Value |
|---|---|
| Hookable layers | `backbone.blocks.0` … `backbone.blocks.19` (20 `TransformerBlock`, pre-norm LayerNorm, MLP ratio 4) |
| Hidden size (`d_model`) | 1024 |
| Heads | 16 |
| Patch length / stride | 16 / non-overlapping (`patch_stride: null`) |
| Max context | 8192 (`n_patch` 512) |
| Parameters | **257,895,552** total; 251,924,480 in the 20 blocks |
| Activation per block, T=512 | `(N, 32, 1024)` |
| Canonical view | mean over the 32 patches, `(N, 1024)` (same reduction as Timer/TimesFM/Moirai) |

Legacy for comparison: 3 × `model.encoder.layers.{i}`, d_model 128, patch 12 + CLS, so 43 tokens,
canonical identity view 5504-D.

Tests (`tests/test_model_wrappers.py`): default-checkpoint constant; random-init HF path; a
tiny *local* PatchTST-FM checkpoint round-trip checking hooked shapes (skipped when `tsfm_public` is
absent, e.g. in CI); and `@pytest.mark.slow` `test_patchtst_fm_r1_pretrained_shapes`, which downloads
FM-r1 and asserts 20 layers, 257,895,552 params, and `(2, 32, 1024)` activations.

## 5. Canonical re-run: old vs new (synthetic, 6 properties)

Protocol: `scripts/run_canonical_benchmark.py`, the same as `run_canonical_all.sh::run_one` (60/20/20,
val-only best layer, seeds 0-4, 1000-resample bootstrap 95% CI, LogisticRegression / Ridge).
Extraction: 1000 samples, seq_len 512, seed 42, GPU 1. Full numbers are in `results.json`
(`scripts/summarize_patchtst_replacement.py`).

| Property (metric) | Old committed `patchtst_pretrained` | **New `patchtst_fm_meanpool`** | Paired Δ new − legacy re-run [95% CI] | Best no-model baseline |
|---|---|---|---|---|
| trend (acc) | 1.000 [1.000, 1.000] | **1.000** [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 (raw) |
| seasonality (R²) | 1.000 [1.000, 1.000] | **0.9999** [0.9999, 0.9999] | 0.0000 [0.0000, 0.0001] | 0.959 (hand-crafted) |
| frequency (acc) | 1.000 [1.000, 1.000] | **0.998** [0.994, 1.000] | −0.002 [−0.006, 0.000] | 1.000 (raw) |
| stationarity (acc) | 0.9998 [0.9994, 1.000] | **1.000** [1.000, 1.000] | +0.001 [0.000, 0.003] | 1.000 (hand-crafted) |
| **anomaly (acc)** | **0.500** [0.490, 0.511] | **0.704** [0.675, 0.733] | **+0.200 [+0.121, +0.267]** | 0.858 (hand-crafted) |
| change_point (acc) | 0.9988 [0.9978, 0.9998] | **1.000** [1.000, 1.000] | +0.001 [0.000, 0.003] | 1.000 (raw) |

Controls for the two protocol differences between the old and new cells:

1. **Sample count.** Committed legacy per-seed scores lie on a 1/1000 grid (a 5000-sample
   extraction, n_test = 1000), while the current pipeline and the new cells use 1000 samples
   (n_test = 200). `legacy_rerun/` re-derives the legacy cells on the identical 1000-sample
   data and splits. It reproduces the committed values (anomaly 0.504 vs 0.500; others within 0.001).
2. **Reduction (model × view 2x2, anomaly):**

   | | flatten (identity) | mean-pool |
   |---|---|---|
   | legacy ETTh1 PatchTST | 0.504 [0.465, 0.558] | 0.457 [0.438, 0.474] |
   | PatchTST-FM-r1 | 0.528 [0.517, 0.538] | **0.704 [0.675, 0.733]** |

   Same-view paired Δ: mean-pool +0.247 [+0.206, +0.282]; flatten +0.024 [−0.035, +0.072]. The
   anomaly gain is a model effect that appears under mean-pool, the canonical view for large patch
   TSFMs. It does not appear in the 32,768-D flattened view, where 600 training samples leave the
   probe badly under-determined.

**Reading.** The triviality story is unchanged. On the four raw-recoverable properties, and on
seasonality and change-point, the new PatchTST saturates just like the old one and like the no-model
baselines. Anomaly moves PatchTST from chance (0.500, last in the Paper A ordering) to **0.704**,
between Moirai (0.694) and Chronos (0.727), and still below the hand-crafted baseline (0.858). The
"hand-crafted beats every TSFM on anomaly" claim therefore still holds, but the sentence listing
"PatchTST 0.500" needs the new number and the new checkpoint name.

## 6. Leakage audit: PatchTST-Pre (ETTh1-trained) used on ETTh1

The legacy model's training data is the ETTh1 train split, so every ETTh1 real-world result with it
is contaminated. Its Weather/Electricity/Traffic/Exchange results are not leaked (disjoint data),
but they still come from a supervised single-dataset forecaster presented as "pre-trained".

**Must be dropped or re-run with PatchTST-FM-r1:**

| # | Artifact | Where it surfaces | Action |
|---|---|---|---|
| 1 | `scripts/run_finetune_compare.py` (defaults `--model patchtst_pretrained --dataset etth1`; fine-tunes on `etth1_trend` windows, then probes `etth1_trend`) | `revision/latex/main.tex` `tab:finetune` (75.5→79.3 / 77.4→79.3 / 77.4→83.0), `fig:finetune_fig` (`fig8_finetune.pdf`), and the text "Fine-tuning PatchTST-Pre on ETTh1 trend yields +5.7%" (l.278, l.1431); same in `neurips2026/latex/main.tex` | **Drop.** The model was already trained on ETTh1, so this measures re-fitting its own training data. Re-run with `--model patchtst_fm` (not in-corpus) if the experiment is kept. Output not in this checkout; archived at `/mnt/nas/Users/suan/disk_cleanup_2026-05-08/LLM4TS/TSFMI/outputs/finetune_compare/patchtst_pretrained_etth1/finetune_comparison.json`. |
| 2 | `outputs/eval/patchtst_pretrained_etth1_{trend,stationarity,change_point,seasonality,seasonality_binary}/` | `tab:real_world_trend_main` "PatchTST-Pre & 76.9%" (ETTh1 column); seasonality-binary table "PatchTST-Pre & 84.6%" (ETTh1 column) | **Drop ETTh1 cells / re-run** with `patchtst_fm`. |
| 3 | `outputs/leace_realworld/patchtst_pretrained_etth1_{trend,stationarity}/leace_results.json` | `tab:realworld_leace` PatchTST ETTh1 columns (trend 98.1→90.6, Δ7.5, L1; stationarity 100.0→94.3, Δ5.7, L2) | **Drop / re-run.** |
| 4 | `scripts/run_realworld_temporal_split.py` (`"PatchTST": "patchtst_pretrained"`; output archived at `/mnt/nas/Users/suan/disk_cleanup_2026-05-08/LLM4TS/TSFMI/outputs/realworld_temporal/results.json`: PatchTST ETTh1 trend 0.750, stationarity 0.7115, n_test 52) | `tab:realworld_temporal` rows "ETTh1 trend 0.750", "ETTh1 stationarity 0.712" (PatchTST column) | **Drop / re-run.** |
| 5 | `scripts/run_neurips_enhancements.py::run_cross_dataset_transfer` (PatchTST synthetic-trend probe applied to `patchtst_pretrained/etth1_trend`; output archived at `/mnt/nas/Users/suan/disk_cleanup_2026-05-08/LLM4TS/TSFMI/outputs/neurips_enhancements/cross_dataset_transfer_results.json`: PatchTST synthetic→etth1 0.302) | `fig:cross_dataset_transfer` (`fig27_cross_dataset_transfer.pdf`), PatchTST × ETTh1 point | **Drop the PatchTST-ETTh1 point / re-run.** |
| 6 | Real-world pipelines that include `patchtst_pretrained` on ETTh1: `scripts/run_new_realworld.sh`, `run_real_world_experiments.sh`, `run_realworld_leace.sh`, `run_realworld_probes.sh`, `run_realworld_extraction.sh` | produce items 2-3 | Swap the model key to `patchtst_fm` before re-running. |
| 7 | `neurips2026/paper_structure.md` Finding 5 "ETTh1: PatchTST-Pre stationarity 80.8%" (older draft) | NeurIPS draft only | Drop. |

**Not leaked, but should be updated to the new checkpoint** (label/provenance, not contamination):

- Paper A (`paper/main_a.tex` l.62) anomaly ordering "PatchTST $0.500$" becomes **0.704**
  (`outputs/canonical/patchtst_fm_meanpool_anomaly`). Update the model table (`README.md` l.109:
  granite / 3 layers / 1.5M) to FM-r1 / 20 blocks / 257.9M.
- Paper B (`paper/main_b.tex` l.44-49) mechanistic PatchTST substitute is the *same* ETTh1 model.
  Rewrite the sentence, and re-run the kurtosis/SAE analyses on FM-r1 if PatchTST stays in Paper B.
- All synthetic-only `patchtst_pretrained` artifacts (`outputs/canonical`, `cka`, `leace`,
  `interventions`, `eval/*synthetic*`, `run_neurips_enhancements.py::run_finetuning_prediction`,
  `run_anomaly_noise_analysis.py`) have no data leakage (the synthetic generators share nothing with
  ETTh1), but they describe the supervised ETTh1 model.
- `outputs/realistic_anomaly_tsfm/results.json` records `patchtst_pretrained` as
  `"error": "ValueError: Cannot load PatchTST checkpoint: namctin/patchtst_etth1_forecast"` (the
  gated default). Fixed in the script by pinning the legacy id. To include the replacement, add a
  `patchtst_fm` entry.
- `outputs/eval/patchtst_pretrained_{weather,electricity,traffic,exchange_rate}_*` and
  `outputs/leace_realworld/patchtst_pretrained_weather_*`: no leakage for the legacy model. If they
  are re-run with FM-r1, the **Traffic** cells must be flagged as possibly in-pretraining (§3).
- `configs/probe_patchtst.yaml` still names `namctin/patchtst_etth1_forecast` (reference only;
  never loaded).

## 7. Reproduce

```bash
cd TSFMI
.venv/bin/python -m pip install --no-deps granite-tsfm==0.3.9 && .venv/bin/python -m pip install datasets==5.0.1
for ds in synthetic_trend synthetic_seasonality synthetic_frequency synthetic_stationarity synthetic_anomaly synthetic_change_point; do
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/extract_representations.py \
    --model patchtst_fm --dataset $ds --layers all --output_dir outputs/representations
  PYTHONPATH=. .venv/bin/python scripts/meanpool_representations.py \
    --input_dir outputs/representations/patchtst_fm/$ds --output_dir outputs/representations/patchtst_fm_meanpool/$ds
done
# then per property (trend/frequency/stationarity/anomaly/change_point: classification; seasonality: regression)
PYTHONPATH=. .venv/bin/python scripts/run_canonical_benchmark.py \
  --representations_dir outputs/representations/patchtst_fm_meanpool/synthetic_anomaly \
  --property anomaly --task_type classification --output_dir outputs/canonical/patchtst_fm_meanpool_anomaly
PYTHONPATH=. .venv/bin/python scripts/summarize_patchtst_replacement.py
```

The controls (`legacy_rerun/`, `legacy_rerun_meanpool/`, `sensitivity_flat/`) use the same
benchmark script. Their inputs are the legacy extraction in
`outputs/representations_patchtst_legacy_rerun/` and the raw
`outputs/representations/patchtst_fm/<ds>`.

`patchtst_fm_meanpool` is **not** yet in `scripts/extract_all_canonical.sh` /
`scripts/run_canonical_all.sh` `MODELS`. Adding it (and removing or renaming `patchtst_pretrained`)
changes `make reproduce-all` and the paper tables, so that is left as an explicit paper decision.
