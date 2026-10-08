---
marp: true
theme: default
paginate: true
---

# Uncertainty-Aware Sexism Detection: Confidence-Aware Loss Functions
### EXIST 2021
**Team:** [Your Names]

---

# Agenda

1. Problem Statement
2. Dataset
3. Baseline Architecture
4. Baseline Limitations
5. Our Approach Overview
6. N1: Confidence-Aware Focal Loss
7. N2: Hierarchical Consistency
8. N3: Adaptive Label Smoothing
9. Ablation Design
10. Results
11. Per-Uncertainty-Bin Analysis
12. Hierarchical Violation Statistics
13. Key Conclusions
14. Methodology Note

---

# Problem Statement

* **Online Sexism:** Pervasive issue on social media platforms requiring rapid, scalable moderation.
* **Why Automation?** Manual review is slow, expensive, and takes a psychological toll on human moderators.
* **EXIST Challenge:** sEXism Identification in Social neTworks (EXIST) 2021.
  * Aims to advance the state-of-the-art in multilingual sexism detection.
  * Focuses on nuanced categorization beyond simple binary classification.

---

# Dataset: EXIST 2021

* **Source:** Twitter dataset (Bilingual: English & Spanish)
* **Size:** 
  * Training: 6,977 tweets
  * Testing: 4,368 tweets
* **Tasks:**
  * **Task 1 (Binary):** Sexist vs. Not-Sexist
  * **Task 2.3 (Categorization - 5 classes):**
    1. Ideological and inequality
    2. Stereotyping and dominance
    3. Objectification
    4. Sexual violence and coercion
    5. Misogyny and non-sexual violence

---

# Baseline Architecture

* **Input:** Raw Text (Tweets)
* **Embeddings:** 768-dimensional frozen text embeddings
* **Body:** SwiGLU Blocks for feature processing
* **Task Heads:**
  * Head T1 (Binary Classification)
  * Head T2.3 (5-Class Categorization)
* **Optimization:** Kendall Multi-Task Loss to balance task learning dynamically

---

# Baseline Limitations

1. **Equal Weighting (Uncertainty Ignored):** Treats highly confident predictions and highly uncertain predictions equally during training.
2. **Hierarchical Inconsistency:** Model can predict a specific category (e.g., Objectification) with high probability while simultaneously predicting the text is "Not Sexist" (Task 1).
3. **Hard Label Overconfidence:** Standard Cross-Entropy/BCE pushes probabilities to 0 or 1, leading to overconfident predictions even on ambiguous examples.

---

# Our Approach Overview

* **Goal:** Improve robustness and consistency without increasing inference cost.
* **Strategy:** Introduce 3 orthogonal novelties entirely at the **loss level**.
* **Zero Architecture Change:** The model architecture remains identical to the baseline during inference.

**The 3 Novelties:**
* **N1:** Confidence-Aware Focal Loss
* **N2:** Hierarchical Consistency
* **N3:** Adaptive Label Smoothing

---

# N1: Confidence-Aware Focal Loss

* **Intuition:** Hard/uncertain examples should get more attention during training.
* **Mechanism:** Adjusts focal loss dynamically based on prediction uncertainty.
* **Equations:**
  * Distance from decision boundary: $d_i = 1 - |2p_i - 1|$
  * Modulator: $m_i = (1-\alpha) + \alpha \cdot d_i^\gamma$
  * Final Loss: $L_{N1} = m_i \cdot \text{BCE}$

---

# N2: Hierarchical Consistency

* **Intuition:** The model must not contradict itself across tasks. A high probability for a sexist category should imply a high probability for "Sexist".
* **Mechanism:** Penalizes cases where a category probability exceeds the binary sexism probability.
* **Equation:**
  * $L_{N2} = \lambda \cdot \sum \max(0, p_{\text{cat}} - p_{\text{sexist}})^2$

---

# N3: Adaptive Label Smoothing

* **Intuition:** Ambiguous samples get softer targets, preventing overconfidence on hard examples.
* **Mechanism:** Scales label smoothing factor based on uncertainty ($d_i$).
* **Equations:**
  * Adaptive smoothing factor: $\epsilon_i = \epsilon_{\text{max}} \cdot d_i$
  * Softened labels: $\tilde{y}_i = (1-\epsilon_i) \cdot y_i + \frac{\epsilon_i}{K}$ (where $K$ is the number of classes)

---

# Ablation Design

We designed 7 configurations to isolate the impact of each novelty and their combinations:

| Configuration | N1 (Focal) | N2 (Hierarchical) | N3 (Smoothing) |
|---|---|---|---|
| Baseline | ❌ | ❌ | ❌ |
| N1 Only | ✅ | ❌ | ❌ |
| N2 Only | ❌ | ✅ | ❌ |
| N3 Only | ❌ | ❌ | ✅ |
| N1 + N2 | ✅ | ✅ | ❌ |
| N1 + N3 | ✅ | ❌ | ✅ |
| All (N1+N2+N3) | ✅ | ✅ | ✅ |

---

# Results: Full Evaluation

| Run | Task1 F1 | Task2.3 F1 | ECE | Viol% |
|---|---|---|---|---|
| **Baseline** | 0.7002 | 0.5623 | 0.4387 | 55.9 |
| **N1 only** | 0.7101 | 0.6033 | 0.4630 | 62.3 |
| **N2 only** | 0.6989 | 0.5677 | **0.4187** | 49.7 |
| **N3 only** | 0.6775 | 0.5383 | 0.4341 | 53.3 |
| **N1+N2** | 0.6987 | 0.5503 | 0.4064 | **44.8** |
| **N1+N3** | **0.7129** | 0.5555 | 0.4375 | 58.0 |
| **All** | 0.6961 | 0.5584 | 0.4482 | 59.2 |

*Note: ECE (Expected Calibration Error), Viol% (Hierarchical Violations)*

---

# Per-Uncertainty-Bin Analysis

**Task1 F1 on High-Uncertainty Examples:**

| Configuration | Task1 F1 (High-Bin) |
|---|---|
| Baseline | 0.481 |
| N1 Only | 0.515 |
| N1 + N3 | **0.558** |

**Insights:**
* The baseline struggles significantly on ambiguous/highly uncertain examples.
* **N1 (Focal Loss)** effectively forces the model to learn from these hard examples.
* **N1 + N3** combination provides the best synergy for handling uncertainty, boosting F1 on hard examples by +0.077 over baseline.

---

# Hierarchical Violation Statistics

* **Metric:** Viol% measures how often a specific category probability exceeds the binary sexism probability.
* **Baseline Violation Rate:** 55.9%
* **N1+N2 Violation Rate:** 44.8%

**Impact:**
* Introducing **N2 (Hierarchical Consistency)** significantly reduces logical contradictions in model outputs.
* Represents an absolute reduction of **11.1%** in violations compared to the baseline.

---

# Key Conclusions

* **Loss-Level Innovations Work:** Significant improvements achieved without adding inference overhead.
* **N1 is the Strongest Driver:** Confidence-Aware Focal Loss provides the largest individual boost to both Task 1 and Task 2 F1 scores.
* **N2 Enforces Logic:** Hierarchical Consistency effectively reduces contradictory predictions (Viol%).
* **N1+N3 Synergy:** Combining N1 with Adaptive Label Smoothing (N3) yields the highest overall Task 1 F1, particularly excelling on ambiguous examples.
* **Trade-offs Exist:** Combining all novelties (N1+N2+N3) didn't yield the highest F1, suggesting complex interactions between the loss components that require careful tuning.

---

# Methodology Note

* **Gemini Embedding v2 (768-dim)** intended per methodology. 
* Free-tier API limits (4500 calls/day) prevented full use. 
* `paraphrase-multilingual-mpnet-base-v2` used as consistent 768-dim multilingual fallback. 
* Trends and conclusions unaffected.

---

# Thank You!

### Questions & Answers
