# MOMENT-Large Fisher LDR vs. raw / hand-crafted controls (Wiliński et al., ICML 2025)

Produced by `scripts/run_wilinski_moment_ldr.py`. The exact command, the upstream commit hash, library
versions, series hashes and all per-layer and per-(layer, patch) numbers are in `results.json`.

- Upstream: https://github.com/moment-timeseries-foundation-model/representations-in-tsfms @ `72337c6e4072d4c213588c116633be71a1b87760`
- Paper: PMLR v267 (`wilinski25a.pdf`), Fig. 7 (p. 7) and Fig. 14 (p. 19)

## What was reproduced (their code, called via `sys.path`, nothing vendored)

| Step | Their code | Notes |
|---|---|---|
| Data | `steertool.dataset_generator.generate_datasets(config_dir="configs", random_seed=42)` | Same as their `cli generate`. It seeds once and then works through all 16 YAMLs in `os.listdir` order, so the series depend on that order, which the file system sets. The order used here is recorded. Every YAML has `n_series=512`, `length=512` and **no noise**. |
| Activations | `steertool.moment.get_activations_MOMENT` | `AutonLab/MOMENT-1-large`, reconstruction head, a forward hook on `encoder.block[i].layer[-1]` for all 24 blocks, output `(24, n, 64 patches, 1024)`. The `input_mask` is all ones. The only input normalisation is MOMENT's internal RevIN. The only change is a cache so `from_pretrained` loads the model once. |
| LDR | `steertool.separability.compute_linear_separability` | sklearn `LinearDiscriminantAnalysis()` is fitted **in-sample** on 2n points. The score is `(μ1-μ2)² / (σ1²+σ2²)` of the 1-D LDA projection. It is computed for every (layer, patch) cell and for each layer's mean over patches (the red curve in their figures). |
| Scaling | their formula `(x-min)/(max-min)` | Applied over the whole 24×64 heatmap and, separately, over the 24-point curve. Plots come from their `plot_linear_separability`. |
| Sample size | `--samples 20` in `run_separability_analysis.sh`, which means `dataset[:20]` per class | Also run with n=512 (all series; this is the setting of TSFMI's `wilinski_reproduction`). |

**Important discovery about fidelity: their activations were taken with dropout on.** `MOMENTPipeline.from_pretrained` goes
through `PyTorchModelHubMixin`. Its safetensors branch never calls `model.eval()`; only the pickle branch does.
This holds both in their pinned `huggingface-hub==0.32.0` (checked in the wheel source) and in our 0.36.2.
`get_activations_MOMENT` never calls `eval()` either, and `torch.no_grad()` does not turn dropout off. So their
pipeline runs MOMENT with T5 dropout p=0.1 active. The faithful runs (`train_dropout_seed{0,1,2}`) keep that
behaviour, with three torch seeds because their runs are unseeded. The `eval` run is the same pipeline with
`model.eval()` and serves as a sensitivity check.

Controls are computed on **exactly the series that were fed to MOMENT** (the first n rows of the same parquet
files; `series_sha256_16` in the JSON), using the same LDR function:
`raw_signal` (512-D), `raw_revin` (MOMENT's own RevIN applied to the input, i.e. what the encoder
actually sees), `hand_crafted` (the 8-D features from `scripts/run_wilinski_reproduction.py`, imported),
`raw_patch` and `raw_revin_patch` (8-D per-patch analogues of the patch heatmap).

## Does the scaled heatmap match their figures?

**Faithful pipeline, n=20: yes, qualitatively.** Compare `heatmap_constant_vs_sine_n20_train_dropout_seed0.png`
with Fig. 7 / Fig. 14(i). In both, the scaled red curve is about 0 at layer 0, rises steeply over the first
few layers, peaks in the middle of the network (theirs near layer 8, ours at layers 11–14 depending on the
seed), stays between 0.6 and 0.9 afterwards, and falls off towards layer 23. The heatmap is mostly dark blue
with horizontal patch stripes, and the brightest cells sit in layers 19–23. The extra pairs also line up with
Fig. 14. `sine_vs_decreasing`: bright patches near 12 and 50 at layers 19–23 with a plateaued curve, like
panel (iv). `sine_vs_increasing`: bright top patches and patches near 60 in the last layers, like panel (iii).
`constant_vs_increasing`: early jump, then 0.6–0.9, like panel (ii). Fig. 14's panel titles ("Trend /
Periodicity / Amplitude") do not agree with its caption. We followed the caption's pairs, and they match.
No panel in the paper shows `increasing_vs_decreasing` or `high_vs_low_freq`.

**The pattern does not reproduce in two other settings:**
- eval mode: LDR values reach 1e10–1e15, the curve peaks at layer 0–1 and is noisy after that.
- n=512: the in-sample LDA has 1024 points in 1024 dimensions and is ill-conditioned. The heatmap is a few
  isolated hot cells and the curve is spiky.

The published picture therefore depends on n=20 **and** dropout noise.

## Headline numbers (best layer; faithful runs give the range over 3 dropout seeds)

`pooled` = patch-mean per layer (their red curve), unscaled. `cell` = maximum over the (layer, patch) heatmap.
`PCA-8` = the same LDR after PCA to 8 dimensions, with PCA fit on the pooled two-class data of that space.
`held-out` = 5-fold LDA, with the Fisher ratio computed on the test-fold projections.

| Pair | n | MOMENT pooled | MOMENT cell | MOMENT PCA-8 | MOMENT held-out | raw | raw PCA-8 | raw held-out | HC | HC held-out | MOMENT eval-mode pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| constant vs sine | 20 | 255–296 | 45–47 | 2.8e3–3.6e3 | 1.4e3–2.0e3 | 0.001* | 161 | 0.07* | 5e31† | 2e32† | 1.9e14† |
| constant vs sine | 512 | 2.1e7–4.2e7‡ | 0.7e8–1.9e8‡ | 2.2e3–2.5e3 | 2.1e3–2.3e3 | 0.001* | 2.4 | 0.02* | 3e31† | 1e32† | 2.5e15† |
| increasing vs decreasing | 20 | 66–69 | 149–161 | 608–814 | 288–751 | 70.5 | 74.8 | 75.2 | 85.1 | 71.8 | 1.3e14† |
| increasing vs decreasing | 512 | 2.9e6–9.8e6‡ | 0.4e8–2.5e8‡ | 389–435 | 301–321 | 56.4 | 56.7 | 56.9 | 56.6 | 56.6 | 2.1e14† |
| high vs low freq | 20 | 183–216 | 33–43 | 2.0e3–3.2e3 | 464–682 | 3.1e8† | 49.6 | 281 | 152 | 161 | 5.0e9† |
| high vs low freq | 512 | 1.1e7–1.5e8‡ | 0.3e8–1.2e8‡ | 1.7e3–1.9e3 | 1.5e3–1.6e3 | 1.1e8† | 32.1 | 8.8e7† | 85.8 | 86.9 | 2.5e10† |

\* sklearn's rank truncation (`tol=1e-4`) throws away the only discriminative direction because its within-class
variance is exactly 0. Held-out LDA accuracy then drops to chance (0.48), yet a plain difference-of-means
direction separates the classes perfectly (held-out accuracy 1.0). The existing TSFMI number
`fisher_ldr_raw_signal = 0.0004` for this pair is this same artifact. It does **not** mean the raw signal is
hard to separate.
† Near-zero within-class variance along the discriminant, caused by noise-free data and/or RevIN, which
removes intercept and slope magnitude. The ratio is effectively ∞ and the digits are numerical noise.
‡ In-sample LDA with 1024 points in 1024 dimensions over-fits. The value grows with n and is not a stable
property of the representation.

Held-out classification accuracy is 1.0 for MOMENT at its best layer in every pair and setting. It is 1.0
for HC with the LDA classifier; the difference-of-means classifier on HC reaches 0.70 (n=20) and 0.96 (n=512)
on the frequency pair only. It is 1.0 for raw with difference-of-means in every pair. Every concept is
therefore perfectly linearly separable without a model.

## Interpretation: is "HC ≥ MOMENT" meaningful under this metric?

1. **The LDR is a 1-D ratio.** Their function projects onto the LDA axis first, so the number has no units and
   in principle can be compared across spaces. In practice it is fitted **in-sample**, and the maximum
   achievable in-sample Fisher ratio grows with dimension d relative to n. With n=20 per class, 1024-D MOMENT
   (and 512-D raw) have far more room to over-fit than 8-D HC. Native magnitudes are therefore confounded by
   dimensionality, and the PCA-8 and held-out columns are the fairer comparison.
2. **With noise-free series the metric is degenerate.** Any space in which a class has zero within-class
   variance along the discriminant gives ∞, or ≈0 once sklearn discards that direction. That covers raw,
   RevIN input, HC, and MOMENT in eval mode, because RevIN maps all `none_constant`, all `sine_constant`, all
   `none_increasing` and all `none_decreasing` series to one identical input each. MOMENT's published LDRs are
   finite and moderate **only because dropout adds noise to the activations**. Dropout noise, not concept
   structure, sets the scale of MOMENT's LDR.
3. **The ordering depends on the pair and the variant.**
   - Increasing vs decreasing:
     - n=20: raw (70.5) and HC (85.1) exceed MOMENT's best pooled layer (66–69), but MOMENT's best cell is
       higher (149–161).
     - After PCA-8 or in held-out scoring, MOMENT (dropout) is higher (300–800 vs 56–91).
   - Frequency: HC (152 / 86) is below MOMENT in every dropout variant, while raw ranges from 32 (PCA-8)
     to 3e8 (native).
   - Constant vs sine: HC is ∞. Raw is ≈0 (artifact) natively and 2.4–161 after PCA-8.

   **A general claim that "HC ≥ MOMENT in Fisher LDR" is therefore not supported.** What the data do support
   is weaker and more robust. Every concept pair is perfectly linearly separable from 8 hand-crafted features
   and (with a difference-of-means readout) from the raw signal: held-out accuracy is 1.0, with the one HC
   exception on the frequency pair noted above. Most control LDRs are already infinite on this noise-free data.
   MOMENT's finite LDR values are not evidence that MOMENT adds separability beyond what the input already has.
   Their magnitudes reflect dropout noise, sample size (they grow by about 1e5 from n=20 to n=512) and
   dimensionality, not concept encoding.

## Deviations from their pipeline and why

- Library versions: torch 2.10 / transformers 4.57 / huggingface-hub 0.36 / sklearn 1.7.2, against their pinned
  torch 2.4 / transformers 4.52 / hub 0.32 / sklearn 1.6.1. The train-mode behaviour and the LDA solver are the
  same in both sets.
- `steertool/__init__.py` imports the Chronos module (`nnsight`, `chronos`), which the MOMENT path never uses.
  Missing modules are stubbed. Their LDR function is shipped to joblib workers by value.
- For n=512, activations are extracted in chunks of 128 instead of one batch of 512. In eval mode, chunking
  changes activations by at most 1.3e-3 (float32 batch non-determinism; recorded in the JSON). For n=20 it is
  their exact single-batch call.
- Dropout is seeded (seeds 0, 1, 2) so results are reproducible; their runs are unseeded. Their figures
  correspond to one unknown dropout draw.
- The generated series are not bit-identical to theirs, because they depend on `os.listdir` order. They also
  differ from the series in `outputs/wilinski_reproduction/` (that script used `default_rng(42)` per dataset),
  so the control numbers here are the ones to compare with MOMENT.
- Added outputs that their code does not produce: unscaled values, PCA-8 and held-out LDR, held-out LDA and
  difference-of-means accuracies, within/between variance diagnostics, and the eval-mode sensitivity run.
