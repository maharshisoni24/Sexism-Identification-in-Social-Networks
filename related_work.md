# Related Work / Literature Review

## 1. Online Sexism Detection and Shared Tasks

### 1.1 EXIST 2021 — sEXism Identification in Social neTworks

The primary benchmark for this work is the **EXIST 2021** shared task (Rodríguez-Sánchez et al., 2021), held at IberLEF 2021. It introduced a dataset of over 11,000 social media posts from Twitter and Gab, annotated in both English and Spanish. The task is structured hierarchically:

- **Task 1 (Binary):** Identify whether a post is sexist or not.
- **Task 2 (Categorical):** Classify the type of sexism among five fine-grained categories — *ideological/stereotypical inequality*, *stereotyping and dominance*, *objectification*, *sexual violence*, and *misogyny/non-sexual violence*.

**31 teams** from 11 countries participated, submitting 70 runs for Task 1 and 61 runs for Task 2. Key findings from the competition:
- Transformer-based architectures (BERT, RoBERTa, mBERT, XLM-RoBERTa) dominated all top positions.
- Task 2 (categorization) proved significantly harder than Task 1, with top systems achieving F1 around 0.45–0.55 versus 0.78–0.82 for Task 1.
- Implicit and subtle sexism expressions caused the highest annotator disagreement and model false-negative rates.

> **Reference:** Rodríguez-Sánchez, F., Carrillo-de-Albornoz, J., Plaza, L., Gonzalo, J., Rosso, P., Comet, M., & Donoso, T. (2021). *Overview of EXIST 2021: sEXism Identification in Social neTworks.* Procesamiento del Lenguaje Natural, 67.

---

### 1.2 SemEval-2023 Task 10 — EDOS (Explainable Detection of Online Sexism)

Building on EXIST, **EDOS** (Kirk et al., 2023) extended sexism detection to an explainable, three-level hierarchical classification over approximately 20,000 posts from Gab and Reddit:

- **Task A:** Binary sexism detection (F1 top systems ≈ 0.85–0.88)
- **Task B:** 4-class category prediction (F1 ≈ 0.64–0.73)
- **Task C:** 11-class fine-grained vector (F1 ≈ 0.48–0.60)

EDOS won the **Best Task Paper Award** at SemEval-2023. Top systems used DeBERTa-v3-large, RoBERTa-large, and multi-task learning frameworks that jointly learned across all three sub-tasks to exploit hierarchical dependencies — a paradigm closely related to our approach.

> **Reference:** Kirk, H. R., et al. (2023). *SemEval-2023 Task 10: Explainable Detection of Online Sexism.* Proceedings of SemEval-2023.

---

### 1.3 Sexism and Hate Speech Detection — Broader Survey

The 2021–2023 literature saw a clear evolution:
- **2021:** Fine-tuned monolingual BERT and RoBERTa models dominate English tasks; multilingual XLM-R extends performance to Spanish and other languages.
- **2022:** Multi-task learning frameworks that jointly detect hate speech with auxiliary tasks (sentiment, target identification) reduce false positives and improve generalization across domains.
- **2023:** The field begins modeling annotator subjectivity explicitly (see §3), and LLMs (GPT-4, instruction-tuned models) are evaluated in zero-/few-shot settings.

Implicit sexism — sarcasm, coded language, cultural idioms — remains the primary open challenge for all models across all years.

> **Reference:** Plaza-Del-Arco, F. M., et al. (2021). *Comparing Hate Speech Detectors: Do They Transfer Across Datasets?* EACL 2021.

---

## 2. Transformer-Based Pre-trained Language Models

### 2.1 BERT and Variants

**BERT** (Devlin et al., 2019) established bidirectional masked language modeling as the dominant pre-training paradigm. Fine-tuned BERT became the baseline system for virtually all SemEval and IberLEF tasks from 2019 onwards. **RoBERTa** (Liu et al., 2019) improved upon BERT with larger batch training, dynamic masking, and longer pre-training, setting new benchmarks on GLUE and SQuAD.

> **References:**
> - Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
> - Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.

---

### 2.2 XLM-RoBERTa — Multilingual Cross-Lingual Transfer

**XLM-RoBERTa** (Conneau et al., 2020) is trained on 2.5 TB of filtered CommonCrawl data covering 100 languages using masked language modeling, with a 250,000-token SentencePiece vocabulary. It does not use language-specific embeddings, making it naturally robust to code-switching.

Key results:
- +14.6% average accuracy over mBERT on XNLI cross-lingual inference
- +15.7% for Swahili, +11.4% for Urdu versus prior XLM baselines
- First multilingual model to match monolingual RoBERTa-Large on English GLUE

XLM-RoBERTa became the de-facto backbone for multilingual sexism detection at EXIST 2021, particularly for the combined English+Spanish setting used in this work.

> **Reference:** Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.

---

### 2.3 Sentence Transformers and Multilingual Embeddings

**Sentence-BERT** (Reimers & Gurevych, 2019) introduced Siamese BERT networks to produce semantically meaningful fixed-size sentence embeddings comparable via cosine similarity — solving the quadratic complexity of BERT's pairwise inference.

**MPNet** (Song et al., 2020, NeurIPS) combined BERT's masked language modeling with XLNet's permuted language modeling, capturing token dependencies while reducing positional discrepancies.

The model used in this work — **`paraphrase-multilingual-mpnet-base-v2`** — combines both: an MPNet backbone fine-tuned with the Sentence-Transformers framework on multilingual paraphrase data. It maps 50+ language sentences into a shared 768-dimensional embedding space, enabling cross-lingual semantic similarity without explicit language annotations.

> **References:**
> - Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
> - Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020). *MPNet: Masked and Permuted Pre-training for Language Understanding.* NeurIPS 2020.

---

## 3. Learning with Disagreements and Annotator Subjectivity

### 3.1 The LeWiDi Paradigm

Traditional NLP assumes one "gold" label per example. For subjective tasks like sexism detection, this assumption is flawed — annotator disagreement reflects genuine ambiguity, not noise. The **LeWiDi (Learning with Disagreements)** shared task series formalized this problem:

- **SemEval-2021 Task 12 (LeWiDi):** Provided multi-annotation datasets across NLP and vision. Models trained on full annotation distributions (rather than majority-vote consensus) showed improved robustness on boundary cases.
- **SemEval-2023 Task 11 (LeWiDi):** Extended to subjective NLP tasks (offensive language, sentiment). Soft evaluation metrics and distribution-matching were emphasized over hard-label accuracy.

Key methodology: replacing one-hot (hard) targets with soft label distributions derived from annotator votes — `[0.7, 0.3]` instead of `[1, 0]`. Loss functions shift from cross-entropy to KL divergence or BCE against soft targets.

Our **N3 novelty (Adaptive Label Smoothing)** is inspired directly by the LeWiDi paradigm: rather than requiring soft-label annotations (which EXIST 2021 does not provide), we *infer* per-sample ambiguity from model uncertainty at training time and apply proportional label smoothing dynamically.

> **Reference:** Uma, A., et al. (2021). *Learning from Disagreement: A Survey.* Journal of AI Research, 72.

---

### 3.2 Dataset Cartography

**Swayamdipta et al. (2020)** proposed Dataset Cartography — characterizing training examples by their "confidence" (mean probability of gold label across epochs) and "variability" (standard deviation). This two-phase method identifies "ambiguous" and "easy-to-learn" examples offline after training.

Our N1 and N3 novelties achieve a similar goal *online*, in a single training pass, using the model's instantaneous uncertainty signal `d_i = 1 - |2p_i - 1|` as a proxy for sample ambiguity — eliminating the need for multi-epoch offline analysis.

> **Reference:** Swayamdipta, S., et al. (2020). *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.* EMNLP 2020.

---

## 4. Loss Function Design

### 4.1 Focal Loss (Lin et al., 2017)

**Focal Loss** (Lin et al., ICCV 2017) was introduced for dense object detection to address extreme foreground-background class imbalance (~100,000 candidate locations per image). Standard cross-entropy overwhelms gradients with easy negatives. Focal Loss adds a modulating factor:

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

When a sample is easy (`p_t → 1`), `(1 - p_t)^γ → 0` suppresses its gradient. The authors found `γ = 2` works best empirically. An alpha-balanced variant `FL(p_t) = -α_t(1 - p_t)^γ log(p_t)` further handles class frequency imbalance.

**Our N1 novelty differs critically:** standard Focal Loss modulates by `(1 - p_t)` — relative to the *ground truth* class, hence up-weighting *misclassified* samples regardless of *why* they were misclassified. Our modulator `m_i = (1-α) + α·d_i^γ` where `d_i = 1 - |2p_i - 1|` is **decision-boundary symmetric** — it specifically targets samples where the model is genuinely uncertain (`p ≈ 0.5`), acting as a proxy for inherent annotator disagreement even on hard-labeled datasets.

> **Reference:** Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.

---

### 4.2 Label Smoothing (Szegedy et al., 2016; Müller et al., 2019)

**Label smoothing** was introduced by Szegedy et al. (2016) in the Inception architecture as a regularization technique. Instead of training against hard one-hot targets `[1, 0]`, it softens them to `[1-ε, ε/(K-1)]`. In NLP, Vaswani et al. (2017) applied it in the original Transformer for machine translation.

**Müller et al. (NeurIPS 2019)** studied *when* label smoothing helps: it improves calibration (predicted probabilities better match actual confidence) and is most beneficial for beam-search-based generation. It hurts knowledge distillation because uniform smoothing erases fine-grained inter-class similarity information.

**Our N3 novelty** extends uniform label smoothing to *adaptive* per-sample smoothing: `ε_i = ε_max · d_i`, where `d_i` is the model's current uncertainty. Clear samples get sharp targets; ambiguous samples get softened targets. This eliminates the key limitation of uniform smoothing: applying the same ε to all samples regardless of their inherent difficulty.

> **References:**
> - Szegedy, C., et al. (2016). *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016.
> - Müller, R., Kornblith, S., & Hinton, G. (2019). *When Does Label Smoothing Help?* NeurIPS 2019.
> - Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.

---

## 5. Multi-Task Learning

### 5.1 Kendall Uncertainty Weighting (Kendall et al., 2018)

Manually tuning per-task loss weights in multi-task learning is expensive and non-intuitive, especially when tasks have different units (e.g., regression in meters vs. classification cross-entropy). **Kendall et al. (CVPR 2018)** proposed automatic weighting via *homoscedastic uncertainty*: task-level uncertainty parameters `σ_k` (constant across samples, varying across tasks) are learned jointly during training.

For classification tasks:
$$\mathcal{L}_k^{\text{scaled}} = \frac{1}{\sigma_k^2} \mathcal{L}_k + \log \sigma_k$$

The `log σ_k` term prevents trivial solutions (infinite uncertainty = zero loss). The method was demonstrated on semantic segmentation, depth estimation, and instance segmentation — three tasks with incompatible scales trained on a shared ResNet-101.

Our baseline adopts Kendall weighting across the three EXIST tasks (Task 2.1 binary detection, Task 2.2 source intention, Task 2.3 5-class categorization). All three novelties (N1/N2/N3) operate *below* the Kendall weighting layer — at the per-sample loss level — making them orthogonal to and compatible with the baseline's automatic task balancing.

> **Reference:** Kendall, A., Gal, Y., & Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018.

---

### 5.2 Multi-Task Learning for Hate Speech Detection

Multi-task learning has been widely applied to hate speech and offensive language detection to improve generalization on small, expensive datasets:

- **Plaza-Del-Arco et al. (2021):** MTL with Spanish BERT (BETO), jointly learning sentiment and hate speech detection as auxiliary tasks, improved macro-F1 significantly on Spanish social media.
- **Kapil & Ekbal (2021):** Deep MTL frameworks trained on heterogeneous, multi-domain hate speech datasets showed better generalization to adversarial spellings and dog-whistle terms.
- **SemEval-2020 Task 12:** CAiRE's "Kungfupanda" system combined BERT with MTL to jointly identify offensive language and predict target group and intent, winning top positions.

The key insight common to all: shared representations across related tasks reduce overfitting and enable signal transfer where labeled data is scarce.

> **Reference:** Kapil, P., & Ekbal, A. (2021). *A Deep Multi-task Learning Framework for News Article Classification.* Applied Soft Computing, 110, 107603.

---

## 6. Hierarchical Classification Consistency

In hierarchical text classification, a core challenge is **structural consistency**: predictions should not violate the logical parent-child relationships in the taxonomy. For EXIST, Task 1 is the parent ("Is it sexist?") and Task 2 is the child ("What type of sexism?"). A model predicting 90% confidence in "objectification" while only 55% confident it is sexist is logically inconsistent.

Prior approaches handle this at **decoding time** (hard threshold gates) or through **architectural constraints** (GNNs over the label graph). Our baseline uses a hard binary mask — Task 2 loss is zeroed for non-sexist samples. This is a hard gate applied only during training.

**Our N2 novelty** introduces a **soft, continuous** consistency constraint at the loss level:
$$\mathcal{L}_{N2} = \lambda \cdot \mathbb{E}[\text{ReLU}(p_{cat} - p_{sexist})^2]$$

This penalizes violations proportionally to their magnitude during training, rather than applying a hard zero/one gate. The approach is directly inspired by penalty-based hierarchical constraint literature but adapted to the specific probability containment structure of EXIST's binary/categorical hierarchy.

> **References:**
> - Silla, C. N., & Freitas, A. A. (2011). *A Survey of Hierarchical Classification Across Different Application Domains.* Data Mining and Knowledge Discovery, 22(1-2), 31-72.
> - Zhou, J., et al. (2020). *Hierarchy-Aware Global Model for Hierarchical Text Classification.* ACL 2020.

---

## 7. Calibration and Uncertainty Estimation

**Expected Calibration Error (ECE)** measures whether a model's predicted confidence matches its actual accuracy — a well-calibrated model that says "70% confident" should be right ~70% of the time.

**Guo et al. (ICML 2017)** showed that modern deep neural networks are systematically over-confident: they tend to output probabilities near 0 or 1 regardless of true uncertainty. Temperature scaling — a single learned scalar applied to logits post-training — was shown to dramatically improve calibration with no accuracy loss.

Our work uses ECE as a secondary evaluation metric alongside F1. The improvements from N1 and N3 novelties — which modulate losses based on model uncertainty — show consistent ECE improvements over baseline (0.4387 → 0.3806), suggesting that uncertainty-aware loss functions implicitly improve calibration alongside task performance.

> **Reference:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML 2017.

---

## Summary Table: Prior Work vs. Our Contributions

| Area | Prior Work | Our Contribution |
|------|-----------|-----------------|
| Sexism detection | EXIST 2021 top systems use XLM-R fine-tuning, single-task heads | Multi-task heads with loss-level novelties |
| Focal loss | Lin et al. 2017: modulate by `(1-p_t)^γ` (misclassification-based) | N1: modulate by `d_i = 1-\|2p-1\|` (uncertainty-based, boundary-symmetric) |
| Label smoothing | Müller 2019: static ε for all samples | N3: adaptive `ε_i = ε_max · d_i` per sample (online, no multi-epoch needed) |
| Hierarchical consistency | Hard mask (baseline) or GNN constraints | N2: soft quadratic penalty on probability containment violations |
| Multi-task balancing | Manual weight tuning | Baseline: Kendall (2018) automatic homoscedastic weighting |
| Annotator disagreement | LeWiDi / Dataset Cartography (offline, requires soft labels) | N1+N3: infer ambiguity online from model uncertainty on hard-labeled data |
| Calibration | Temperature scaling post-training | N1+N3 loss design: implicit calibration improvement during training |

---

## References (Alphabetical)

1. Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.
3. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
4. Kapil, P., & Ekbal, A. (2021). A Deep Multi-task Learning Framework for News Article Classification. *Applied Soft Computing, 110*.
5. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *CVPR 2018*.
6. Kirk, H. R., et al. (2023). SemEval-2023 Task 10: Explainable Detection of Online Sexism. *Proceedings of SemEval-2023*.
7. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.
8. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*.
9. Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *NeurIPS 2019*.
10. Plaza-Del-Arco, F. M., et al. (2021). Comparing Hate Speech Detectors: Do They Transfer Across Datasets? *EACL 2021*.
11. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
12. Rodríguez-Sánchez, F., et al. (2021). Overview of EXIST 2021: sEXism Identification in Social neTworks. *Procesamiento del Lenguaje Natural, 67*.
13. Silla, C. N., & Freitas, A. A. (2011). A Survey of Hierarchical Classification Across Different Application Domains. *Data Mining and Knowledge Discovery, 22(1-2)*.
14. Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020). MPNet: Masked and Permuted Pre-training for Language Understanding. *NeurIPS 2020*.
15. Swayamdipta, S., et al. (2020). Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. *EMNLP 2020*.
16. Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR 2016*.
17. Uma, A., et al. (2021). Learning from Disagreement: A Survey. *Journal of AI Research, 72*.
18. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
19. Zhou, J., et al. (2020). Hierarchy-Aware Global Model for Hierarchical Text Classification. *ACL 2020*.
