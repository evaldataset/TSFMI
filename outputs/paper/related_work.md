# Related Work

## 2.1 Time Series Foundation Models

The emergence of large-scale pre-trained models for time series has paralleled the success of foundation models in NLP and computer vision. **MOMENT** (Goswami et al., 2024) introduced a family of encoder-only masked time series models pre-trained on the Time Series Pile, demonstrating strong transfer across forecasting, classification, and anomaly detection. **Chronos** (Ansari et al., 2024) takes a language modeling approach, tokenizing time series values into discrete bins and training T5-based encoder-decoder architectures for probabilistic forecasting. **TimesFM** (Das et al., 2024) adopts a decoder-only architecture with input patching, pre-trained on a large corpus of Google Trends and synthetic data. **Moirai** (Woo et al., 2024) introduces a universal forecasting transformer with any-variate attention.

On the task-specific side, **PatchTST** (Nie et al., 2023) established channel-independent patching with self-supervised pre-training as a strong baseline for time series Transformers, while **iTransformer** (Liu et al., 2024) inverts the attention axis to attend over variates rather than time steps. These models, alongside earlier architectures like Autoformer (Wu et al., 2021) and TimesNet (Wu et al., 2023), form the Transformer-TS family that we compare against foundation models in our study.

Despite rapid architectural progress, the question of *what* these models actually learn internally remains largely unanswered. Our work addresses this gap through systematic representation analysis across model families.

## 2.2 LLM Repurposing for Time Series

A parallel research thread investigates whether pre-trained language models can be directly repurposed for time series tasks. **GPT4TS** (Zhou et al., 2023) demonstrated that a frozen GPT-2 backbone with only input/output layers fine-tuned achieves competitive forecasting performance, theoretically arguing that self-attention behaves similarly to PCA across domains. **Time-LLM** (Jin et al., 2024) takes a reprogramming approach, aligning time series patches with text prototypes before feeding them into a frozen LLM. **CALF** (Liu et al., 2025) identifies significant distribution discrepancies between temporal and textual modalities and proposes cross-modal fine-tuning to bridge this gap.

This line of work has sparked considerable debate. Tan et al. (2024) challenge whether LLMs are "actually useful" for time series, showing that simple linear models can match LLM-based forecasters. Conversely, Schumacher et al. (2026) demonstrate that prompt-based evaluation severely underestimates LLM capability: while zero-shot prompting achieves F1 scores of only 0.15–0.26 on time series classification, linear probes over the *same* internal representations reach 0.61–0.67, matching specialized models. This finding directly motivates our probing-based analysis of GPT4TS, suggesting that the gap lies in the output interface rather than the internal representations.

Our cross-model CKA analysis (§5.4) provides new evidence in this debate: GPT4TS representations occupy a fundamentally different manifold from native TS models (CKA = 0.34 with PatchTST), yet encode meaningful temporal properties that are linearly decodable — supporting the "capable but misaligned" interpretation.

## 2.3 Probing Classifiers and BERTology

The methodology of probing pre-trained representations with diagnostic classifiers originates in NLP. Belinkov (2022) provides a comprehensive survey of probing approaches, their promises, and their limitations. A central concern is that high probing accuracy may reflect the probe's own learning capacity rather than the representation's encoding of a property. Hewitt & Liang (2019) address this through **control tasks**: by comparing probe performance against a baseline trained on random labels, they define **selectivity** as the difference, measuring how much information is genuinely accessible beyond what any representation would yield.

Hewitt & Manning (2019) introduce **structural probes** that recover syntactic tree distances from BERT representations, demonstrating that geometric structure — not just classification boundaries — is linearly encoded. These ideas have been extended beyond NLP: Raghu et al. (2021) use **CKA (Centered Kernel Alignment)** to compare ViT and CNN representations, finding that vision transformers develop more uniform representations across layers. Chowdhury et al. (2023) apply layer-wise probing to speech models (Wav2Vec 2.0), discovering that simple properties like speaker gender are concentrated in fewer neurons while complex properties like dialect are distributed across deeper layers.

Our work transplants this entire methodological toolkit — linear probing, control-task selectivity, CKA similarity — to the time series domain, extending it with causal interventions (LEACE, LDA steering) that go beyond correlational probing.

## 2.4 Representation Analysis of Time Series Models

The most direct predecessor to our work is Wiliński et al. (2025), who conduct the first systematic representation analysis of TSFMs. They analyze layer-wise self-similarity in MOMENT using CKA, discover block-like redundancy structures enabling model pruning, and demonstrate **concept-informed steering** — adding learned concept directions (e.g., periodicity) to latent representations to manipulate model outputs. Their work establishes that temporal concepts are localized in specific layer regions and can be used for controllable generation.

Pandey et al. (2025) extend this with a focus on **linear recoverability**, systematically testing which temporal concepts can be linearly decoded from TSFM representations and how this evolves across model depth. They show that early layers encode local patterns while deeper layers capture more abstract compositional features. Han et al. (2025) take an applied perspective, proposing **TimeRep**, which leverages distances between intermediate-layer representations for anomaly detection, confirming that middle layers often contain more discriminative information than final outputs.

From a mechanistic interpretability angle, Kalnāre et al. (2025) adapt NLP techniques — activation patching, attention saliency, and sparse autoencoders — to time series Transformers, constructing causal graphs of information flow. Zou et al. (2025) investigate TSFM "hallucinations" through signal subspace analysis, using latent-space interventions to amplify context signals and mitigate forecasting errors.

Our work differs from these studies in several key dimensions. First, we perform the first **cross-family** comparison, systematically contrasting Foundation Models, Transformer-TS, and LLM-adapted architectures rather than analyzing individual models. Second, we introduce **LEACE concept erasure** (§2.5) to test causal necessity — going beyond the correlational evidence of probing to determine whether removing linear concept information degrades downstream performance. Third, our **LDA-based steering** provides mathematically grounded interventions with a principled sweep protocol, compared to the heuristic steering directions used in prior work. Finally, we evaluate on both synthetic data with exact ground truth and five real-world benchmarks, exposing the significant gap between controlled and naturalistic settings.

## 2.5 Concept Erasure and Representation Engineering

**Concept erasure** methods remove specific information from neural representations to test its causal role. Ravfogel et al. (2020) introduce **INLP (Iterative Null-space Projection)**, which iteratively trains linear classifiers and projects representations onto their null spaces. Belrose et al. (2023) propose **LEACE (LEAst-squares Concept Erasure)**, which provides a closed-form, optimal solution that removes *all* linear information about a concept while minimizing representation distortion. LEACE guarantees that no linear classifier can recover the erased concept, making it strictly stronger than INLP's iterative approximation.

In a parallel development, **Representation Engineering** (Zou et al., 2023) and **steering vectors** (Turner et al., 2023; Templeton et al., 2024) demonstrate that adding specific directions to LLM hidden states can reliably control model behavior — honesty, toxicity, persona, and other high-level concepts. This framework has been applied extensively to language models but remains unexplored for time series.

To our knowledge, we are the first to apply LEACE concept erasure to temporal properties in time series models, and the first to use LDA-based steering vectors for controlled intervention in TSFM representations. Our LEACE results reveal that stationarity and change-point information are causally encoded in linearly accessible subspaces (46–67% accuracy drop after erasure), while frequency information survives erasure across all native TS models (0% drop), suggesting redundant or nonlinear encoding. These causal findings complement the correlational evidence from probing, providing a more complete picture of how temporal concepts are represented.

---

### References

- Ansari, A. F., Stella, L., Turkmen, C., et al. (2024). Chronos: Learning the Language of Time Series. *arXiv:2403.07815*.
- Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. *Computational Linguistics*, 48(1), 207–219.
- Belrose, N., Schneider-Joseph, D., Ravfogel, S., et al. (2023). LEACE: Perfect Linear Concept Erasure in Closed Form. *NeurIPS 2023*. arXiv:2306.03819.
- Chowdhury, S. A., Durrani, N., & Ali, A. (2023). What Do End-to-End Speech Models Learn? *Neurocomputing*. arXiv:2107.00439.
- Das, A., Kong, W., Leber, A., et al. (2024). A Decoder-Only Foundation Model for Time-Series Forecasting. *ICML 2024*. arXiv:2310.10688.
- Goswami, M., Szafer, K., Choudhry, A., et al. (2024). MOMENT: A Family of Open Time-Series Foundation Models. *ICML 2024*. arXiv:2402.03885.
- Han, C. S. & Lee, K. M. (2025). Leveraging Intermediate Representations of TSFMs for Anomaly Detection. *arXiv:2509.12650*.
- Hewitt, J. & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP 2019*.
- Hewitt, J. & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations. *NAACL 2019*.
- Jin, M., Wang, S., Ma, L., et al. (2024). Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. *ICLR 2024*. arXiv:2310.01728.
- Kalnāre, M., Kitharidis, S., Bäck, T., & van Stein, N. (2025). Mechanistic Interpretability for Transformer-based TS Classification. *IJCCI 2025*. arXiv:2511.21514.
- Liu, P., Guo, H., Dai, T., et al. (2025). CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning. *AAAI 2025*. arXiv:2403.07300.
- Liu, Y., Hu, T., Zhang, H., et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR 2024*. arXiv:2310.06625.
- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. arXiv:2211.14730.
- Pandey, A., Neog, A., & Jajoo, G. (2025). On the Internal Semantics of Time-Series Foundation Models. *arXiv:2511.15324*.
- Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., & Dosovitskiy, A. (2021). Do Vision Transformers See Like Convolutional Neural Networks? *NeurIPS 2021*. arXiv:2108.08810.
- Ravfogel, S., Elazar, Y., Gonen, H., Trost, M., & Goldberg, Y. (2020). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. *ACL 2020*.
- Schumacher, D., Nourbakhsh, E., Slavin, R., & Rios, A. (2026). Prompting Underestimates LLM Capability for Time Series Classification. *arXiv:2601.03464*.
- Tan, C., et al. (2024). Are Language Models Actually Useful for Time Series Forecasting? *NeurIPS 2024*. arXiv:2406.16964.
- Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Anthropic Technical Report*.
- Turner, A., et al. (2023). Activation Addition: Steering Language Models Without Optimization. *arXiv:2308.10248*.
- Wiliński, M., Goswami, M., Żukowska, N., Potosnak, W., & Dubrawski, A. (2025). Exploring Representations and Interventions in Time Series Foundation Models. *ICML 2025*. arXiv:2409.12915.
- Woo, G., Liu, C., Kumar, A., et al. (2024). Unified Training of Universal Time Series Forecasting Transformers. *ICML 2024*. arXiv:2402.02592.
- Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. *NeurIPS 2021*.
- Wu, H., Hu, T., Liu, Y., et al. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. *ICLR 2023*.
- Zhou, T., Niu, P., Wang, X., Sun, L., & Jin, R. (2023). One Fits All: Power General Time Series Analysis by Pretrained LM. *NeurIPS 2023*. arXiv:2302.11939.
- Zou, A., Phan, L., Chen, S., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.
- Zou, Y., Wang, Z., Klabjan, D., & Liu, H. (2025). Investigating Hallucinations of TSFMs through Signal Subspace Analysis. *NeurIPS 2025*.
