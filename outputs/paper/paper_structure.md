# Paper Structure: Dissecting Time Series Foundation Models

## Title

**Dissecting Time Series Foundation Models: Probing, Erasing, and Steering Temporal Concepts Across Model Families**

Alternative titles:
- "What Do Time Series Foundation Models Learn? A Comprehensive Probing Study with Concept Erasure and Steering"
- "BERTology for Time Series: Layer-wise Analysis of Temporal Concept Encoding in Foundation Models"

---

## Abstract (Draft)

Time series foundation models (TSFMs) have achieved strong performance across forecasting, classification, and anomaly detection, yet little is known about *what* temporal properties their internal representations encode and *how* this encoding differs across model families. We present a comprehensive "BERTology for time series" study that applies **linear probing**, **LEACE concept erasure**, **LDA-based steering interventions**, and **cross-model CKA analysis** to dissect the representations of four pre-trained model families: native TSFMs (MOMENT, Chronos-Bolt), a Transformer-TS model (PatchTST), and an LLM-adapted model (GPT4TS), alongside two random-initialization baselines. We probe six temporal properties — trend, seasonality, frequency, stationarity, anomaly presence, and change points — across synthetic datasets with exact ground truth and five real-world benchmarks.

Our key findings are: (1) all pre-trained models achieve near-perfect linear probing accuracy on trend, frequency, stationarity, and change points from their earliest layers, while anomaly detection remains challenging (42–58%); (2) LEACE erasure confirms that stationarity and change-point information is causally encoded in linearly accessible subspaces, while frequency information survives erasure, suggesting redundant or nonlinear encoding; (3) GPT4TS learns qualitatively different representations from native TS models (CKA = 0.34 vs. PatchTST) and is most vulnerable to LDA steering interventions; (4) MOMENT and Chronos-Bolt converge to nearly identical representation spaces (CKA = 0.989) despite fundamentally different architectures; and (5) real-world probing performance degrades substantially, with Chronos-Bolt and PatchTST consistently outperforming PCA-reduced models.

---

## 1. Introduction (~1.5 pages)

### Key Elements:
- **Opening**: Time series foundation models are increasingly deployed, but their internal mechanisms remain opaque
- **Motivation**: NLP has "BERTology" — systematic probing of what BERT learns. Time series lacks an equivalent framework.
- **Gap**: Existing work (Wiliński et al., 2025) probes individual models; no study systematically compares across model families with causal interventions
- **Research Questions**:
  - **RQ1**: What temporal properties (trend, seasonality, frequency, stationarity, anomaly, change points) are linearly encoded at which layers?
  - **RQ2**: How do Foundation Models (MOMENT, Chronos) differ from Transformer-TS (PatchTST) and LLM-adapted (GPT4TS) models in representation quality?
  - **RQ3**: How much temporal information is linearly vs. nonlinearly accessible? (Selectivity analysis)
  - **RQ4**: What is the cross-model representational similarity structure? (CKA analysis)
- **Contributions** (5 points):
  1. First comprehensive cross-family probing study: 4 pre-trained models + 2 baselines × 6 temporal properties × 6 datasets
  2. First application of LEACE concept erasure to temporal properties in TSFMs
  3. LDA-based precision steering — mathematically rigorous interventions beyond heuristic steering
  4. Cross-model CKA revealing convergence in native TS models vs. divergence in LLM-adapted models
  5. Systematic synthetic-to-real evaluation exposing the gap between controlled and real-world probing

---

## 2. Related Work (~1.5 pages)

### 2.1 Time Series Foundation Models
- MOMENT (Goswami et al., ICML 2024), Chronos (Ansari et al., 2024), TimesFM (Das et al., 2024), Moirai (Woo et al., 2024)
- Architecture taxonomy: encoder-only (MOMENT), encoder-decoder (Chronos-Bolt/T5), decoder-only (TimesFM)
- PatchTST (Nie et al., ICLR 2023) as Transformer-TS baseline

### 2.2 LLM Repurposing for Time Series
- GPT4TS/One Fits All (Zhou et al., NeurIPS 2023): frozen GPT-2 + patch embedding
- Time-LLM (Jin et al., ICLR 2024): input reprogramming with text prototypes
- CALF (Liu et al., AAAI 2025): cross-modal alignment
- The debate: "Are LLMs Actually Useful for TS Forecasting?" (Tan et al., 2024) vs. "Prompting Underestimates LLM Capability" (Schumacher et al., 2026)

### 2.3 Probing Classifiers and BERTology
- NLP origins: Belinkov (2022), Hewitt & Manning (2019), Hewitt & Liang (2019)
- Control tasks and selectivity: avoiding overestimation of what representations encode
- Extension to vision: Raghu et al. (NeurIPS 2021) CKA analysis of ViTs
- Extension to audio: Chowdhury et al. (2023) probing Wav2Vec 2.0

### 2.4 Representation Analysis of TSFMs
- Wiliński et al. (ICML 2025): layer-wise probing + concept steering on MOMENT — our most direct predecessor
- Pandey et al. (2025): linear recoverability of temporal concepts
- Han et al. (2025): intermediate representations for anomaly detection
- Kalnāre et al. (IJCCI 2025): mechanistic interpretability (activation patching, sparse autoencoders)

### 2.5 Concept Erasure and Representation Engineering
- INLP (Ravfogel et al., ACL 2020): iterative null-space projection
- LEACE (Belrose et al., NeurIPS 2023): closed-form linear concept erasure
- Representation Engineering (Zou et al., 2023): reading/controlling LLM representations
- Steering vectors in LLMs: Turner et al. (2023), Templeton et al. (2024)
- **Our novelty**: First to apply LEACE and LDA steering to temporal concepts

---

## 3. Methodology (~2.5 pages)

### 3.1 Problem Formulation
- Given a pre-trained model $M$ with $L$ layers, extract frozen representations $\mathbf{h}^{(l)} \in \mathbb{R}^{N \times d_l}$ for each layer $l$
- Probe each $\mathbf{h}^{(l)}$ for temporal property $p$ using a linear classifier/regressor
- **Goal**: Quantify what temporal information is linearly accessible at each depth

### 3.2 Target Models
- Table listing all 6 models with architecture details (layers, d_model, total features, pre-training data)
- **Foundation Models**: MOMENT-Large (24L, 65536D→PCA512), Chronos-Bolt (6L, 16896D)
- **Transformer-TS**: PatchTST-Pre (ibm-granite, 3L, 5504D)
- **LLM-Adapted**: GPT4TS (frozen GPT-2, 12L, 24576D→PCA512)
- **Random Baselines**: PatchTST-Rnd (3L, 10368D), iTransformer-Rnd (6L, 512D)

### 3.3 Temporal Properties and Synthetic Data
- 6 base properties with controlled synthetic generation:
  - **Trend** (classification): rising/falling/stationary — deterministic + noise
  - **Seasonality** (regression): period length in [10, 100]
  - **Frequency** (classification): dominant FFT frequency band (5 classes)
  - **Stationarity** (classification): stationary vs. non-stationary (variance drift)
  - **Anomaly** (classification): presence/absence of point anomalies
  - **Change Point** (classification): presence/absence of distributional shift
- 3 hard variants: trend_hard (subtle slopes), frequency_hard (multi-component), anomaly_hard (realistic amplitude)
- Advantages of synthetic data: exact ground-truth labels, controlled difficulty

### 3.4 Real-World Datasets
- ETTh1 (17,420 rows, OT column), Weather/Jena Climate (420,551 rows, T(degC))
- Electricity (370 clients, LD2011_2014), Traffic (862 sensors), Exchange-Rate (8 currencies)
- Auto-labeling pipeline: FFT for seasonality, ADF for stationarity, linear regression for trend, CUSUM for change points

### 3.5 Linear Probing
- Architecture: `nn.Linear(d, num_classes)` for classification; `nn.Linear(d, 1)` for regression
- Training: Adam optimizer, lr=0.001, 100 epochs, batch_size=256
- 80/20 train/test split with fixed seed
- **Selectivity**: Subtract MLP control probe accuracy to measure linear accessibility beyond memorization

### 3.6 LEACE Concept Erasure
- Given representations $\mathbf{X}$ and concept labels $\mathbf{Z}$, compute the closed-form linear projection that removes all linear information about $\mathbf{Z}$ from $\mathbf{X}$
- Apply erasure, then re-train linear probe on erased representations
- Performance drop = causal necessity of linear concept encoding
- Key: LEACE is optimal (least-squares) — removes ALL linear information, not just some directions

### 3.7 LDA Steering Interventions
- Compute LDA direction $\mathbf{w}$ that maximally separates concept classes
- Add $\alpha \cdot \mathbf{w}$ to representations, measure downstream probe accuracy
- Sweep $\alpha \in [-5, 5]$ to find maximum disruption
- Also test mean-shift baseline: shift representations toward opposite class mean

### 3.8 CKA (Centered Kernel Alignment)
- Measure representational similarity between layers within and across models
- Linear CKA: efficient, interpretable similarity metric (Kornblith et al., 2019)
- Intra-model: $L \times L$ heatmap showing layer redundancy structure
- Cross-model: $L_A \times L_B$ rectangular heatmap showing alignment

---

## 4. Experimental Setup (~0.5 pages)

- Hardware: NVIDIA GPU (CUDA), single-GPU experiments
- Software: PyTorch 2.1, HuggingFace Transformers, scikit-learn
- Synthetic data: 5,000 samples per property, seq_len=512
- Real-world: stride=64 for ETTh1 (265 windows), stride=256 for larger datasets
- PCA reduction: 512 dimensions for MOMENT (65536D) and GPT4TS (24576D); auto-clamped to min(n_samples, n_features)
- Reproducibility: all RNGs seeded (seed=42)

---

## 5. Results (~4 pages)

### 5.1 RQ1: Layer-wise Temporal Property Encoding

**Finding 1**: Trend, frequency, stationarity, and change points are linearly decodable from Layer 0 in all pre-trained models (100% accuracy).
- → These properties are trivially encoded even in patch embeddings
- Random-init baselines also achieve 100% on trend and frequency → architecture-inherent, not learned

**Finding 2**: Anomaly detection is the only genuinely hard task (42–58% best layer).
- Chronos-Bolt leads at 58%, suggesting encoder-decoder architecture captures subtle distributional anomalies better
- Layer progression shows gradual improvement across depth (not concentrated)

**Finding 3**: Seasonality (regression) separates models sharply.
- Chronos (R²=0.9999), PatchTST-Pre (R²=0.9998) — excellent
- MOMENT-PCA512 (R²=-1.12), GPT4TS-PCA512 (R²=-1.04) — **PCA destroys seasonality information**
- Critical implication: dimensionality reduction is not neutral; it selectively damages certain temporal features

**Figure**: Layer progression plots for 4 pre-trained models × 6 properties (existing fig1_*.png)

### 5.2 RQ2: Foundation vs. Transformer-TS vs. LLM-Adapted

**Finding 4**: On synthetic data, all pre-trained models are nearly equivalent (except PCA-damaged seasonality).
- Pre-training advantage over random baselines is most visible in stationarity (100% vs. 46% for PatchTST-Rnd)

**Finding 5**: On real-world data, Chronos-Bolt and PatchTST-Pre consistently outperform MOMENT-PCA512 and GPT4TS-PCA512.
- Weather: Chronos trend 96%, GPT4TS 86%, MOMENT 85%
- ETTh1: PatchTST-Pre stationarity 80.8%, GPT4TS 30.8%
- Electricity: Chronos/PatchTST change_point 100%, GPT4TS 1.9%

**Finding 6**: GPT4TS shows the most inconsistent real-world performance across datasets.
- Good on Exchange-Rate trend (77.3%), terrible on Traffic trend (11.5%)
- Suggests LLM-adapted representations are less robust to distribution shifts

**Table**: Cross-dataset comparison matrix

### 5.3 RQ3: Selectivity Analysis

**Finding 7**: For most properties, selectivity ≈ 0 — MLP control probes match linear probes.
- Trend, frequency, change_point: near-zero selectivity across all models
- → Information is either trivially extractable or not accessible at all
- Anomaly shows slight positive selectivity for MOMENT (+0.054), suggesting marginal linear advantage

**Finding 8**: Seasonality has large negative selectivity for MOMENT (-2.117).
- PCA-damaged representations: MLP recovers some nonlinear residual that linear probe cannot
- → PCA reduction creates a setting where nonlinear probing is strictly necessary

### 5.4 RQ4: Cross-Model CKA

**Finding 9**: MOMENT ↔ Chronos CKA = 0.989 — nearly identical representation spaces.
- Despite different architectures (encoder-only vs. T5 encoder) and training data
- Suggests convergent evolution: independent pre-training on time series produces similar internal representations

**Finding 10**: GPT4TS is the outlier — CKA = 0.34 with PatchTST, 0.69 with MOMENT/Chronos.
- LLM-adapted model lives in a fundamentally different representational space
- Consistent with CALF (Liu et al., 2025) observation of modality misalignment

**Figure**: Cross-model CKA heatmaps (6 pairs)

### 5.5 Intervention Results

#### 5.5.1 LEACE Concept Erasure

**Finding 11**: LEACE confirms causal linear encoding of stationarity and change_point.
- Stationarity drops to near-chance after erasure across all models (46–67% drop)
- Change_point similarly drops (25–64% drop)
- → These concepts are causally necessary in the linear subspace

**Finding 12**: Frequency survives LEACE (0% drop for 3/4 models, 10.7% for GPT4TS).
- Information is either redundantly encoded across many linear directions, or partially nonlinear
- GPT4TS's partial drop suggests less redundant encoding than native TS models

**Finding 13**: Anomaly is not linearly erasable (0–10% drop).
- Consistent with low probing accuracy: anomaly information is weakly and diffusely encoded

#### 5.5.2 LDA Steering

**Finding 14**: GPT4TS is most vulnerable to steering interventions.
- trend_hard: 46.3% accuracy drop, stationarity: 49.3%, frequency_hard: 59.6%
- MOMENT also susceptible on hard variants (trend_hard: 53.3%, frequency_hard: 63.7%)
- Chronos and PatchTST are remarkably robust (< 3% drop on most properties)

**Finding 15**: Steering effect concentrates in later layers for MOMENT (blocks 12-23).
- Consistent with deeper layers encoding higher-level features

---

## 6. Discussion (~1.5 pages)

### 6.1 The "Trivial Encoding" Problem
- Most temporal properties are 100% decodable from Layer 0 — even random models achieve this for trend and frequency
- This echoes findings in NLP where surface-level features are trivially encoded
- **Implication**: Linear probing alone overestimates what models "learn" — need hard variants + selectivity + interventions

### 6.2 PCA as Information Bottleneck
- MOMENT (65536D→512D) and GPT4TS (24576D→512D) lose seasonality information
- PCA captures 69-99% total variance but selectively damages periodic components
- **Recommendation**: Use task-specific dimensionality reduction, or probe full representations

### 6.3 Convergent Representations in Native TS Models
- MOMENT ↔ Chronos CKA = 0.989 suggests a universal "time series representation" emerges from large-scale pre-training
- Analogous to NLP finding that different LLMs converge to similar representation spaces (Bansal et al., 2021)
- GPT4TS divergence (CKA = 0.34 with PatchTST) indicates LLM-adapted models occupy a separate manifold

### 6.4 Causal vs. Correlational Encoding
- LEACE erasure distinguishes between "information is present" (probing) and "information is causally necessary" (erasure)
- Frequency's LEACE resilience is the most striking finding — suggesting highly redundant encoding
- Stationarity/change_point's vulnerability confirms these are linearly localized concepts

### 6.5 The Real-World Gap
- Synthetic probing: controlled, 100% accuracy on many tasks
- Real-world: 50-97% accuracy, highly variable across datasets
- Auto-labeling quality is a major confounder — FFT-based seasonality labels fail on real data (R² << 0)
- **Recommendation**: Develop better auto-labeling or use human-annotated real-world probing benchmarks

### 6.6 Limitations
- PCA reduction affects MOMENT and GPT4TS comparisons (unfair to reduced models)
- Univariate analysis only — multivariate interactions not captured
- iTransformer with num_variates=1 is degenerate
- Auto-labeling of real-world data introduces noise
- Only linear/MLP probes tested — no structural probes (Hewitt & Manning, 2019)

---

## 7. Conclusion (~0.5 pages)

- Summary of 5 key contributions and 15 findings
- "BERTology for Time Series" is viable and reveals both universal patterns and family-specific characteristics
- Call for better probing benchmarks (beyond synthetic) and standardized interpretability toolkits for TSFMs
- Future work: multivariate analysis, structural probes, more model families, causal tracing

---

## Appendices

### A. Synthetic Data Generation Details
- Mathematical formulations for each generator
- Example visualizations

### B. Real-World Auto-Labeling Pipeline
- Step-by-step procedure for each property
- Failure cases and mitigations

### C. Full Layer-wise Results
- Complete tables for all layers × all models × all datasets

### D. Hyperparameter Sensitivity
- Learning rate, epochs, PCA dimensions

---

## Target Venue

**Primary**: ICML 2026, NeurIPS 2026, or ICLR 2027
**Secondary**: AAAI 2027, AISTATS 2026
**Workshop**: ICML 2026 Workshop on Interpretable ML, NeurIPS 2026 TS Workshop

**Page budget**: 9 pages main + unlimited appendix (ICML/NeurIPS format)

---

## Figure/Table Plan

| ID | Type | Content | Status |
|----|------|---------|--------|
| Fig 1 | Layer progression | 4 models × key properties | ✅ Exists |
| Fig 2 | Cross-model CKA | 6 pairwise heatmaps | ✅ Exists |
| Fig 3 | LEACE before/after | Bar chart grouped by model | 🔲 Need |
| Fig 4 | LDA steering curves | Alpha sweep for GPT4TS + MOMENT | 🔲 Need |
| Fig 5 | Intra-model CKA | Block redundancy structure | ✅ Exists |
| Table 1 | Model specs | Architecture, layers, features | 🔲 Need |
| Table 2 | Synthetic probing | Best layer accuracy per model | ✅ Exists |
| Table 3 | Real-world probing | By dataset × model | ✅ Exists |
| Table 4 | Selectivity | Linear − MLP control | ✅ Exists |
| Table 5 | Cross-model CKA | Pairwise similarity | ✅ Exists |
| Table 6 | LEACE results | Before/after/drop | ✅ Exists |
| Table 7 | LDA steering | Max drop per model | ✅ Exists |
| Table 8 | Easy vs Hard | Difficulty scaling | ✅ Exists |
