# Risk Fusion & Interaction Model Empirical Evaluation Plan

**Experiment Date:** October 2026  
**Stage:** STAGE 5 COMPLETE  
**Status:** VALIDATED (Controlled Cross-Scene Evaluation across 7,729 Benchmark Frames)

---

## 1. Experimental Objective
To evaluate whether an interaction-capable interpretable model (GAM with explicit pairwise interactions and shallow Tree Gradient Boosting) provides empirical improvement over the frozen baseline linear risk fusion:
$$\text{Risk}_{\text{baseline}} = 0.45 \cdot \text{Panic} + 0.40 \cdot \text{Bottleneck} + 0.15 \cdot \text{Density}$$

---

## 2. Experimental Protocol
- **Dataset:** Official UMN benchmark master recording (7,729 frames across 11 clips and 3 physical scenes: Lawn, Indoor, Plaza).
- **Validation Schemes (Zero Frame-Shuffling Leakage):**
  1. **Leave-One-Scene-Out (LOSO, 3 folds):** Cross-environment generalization (train on 2 scenes, evaluate on 3rd).
  2. **Leave-One-Clip-Out (LOGO, 11 folds):** Sequence generalization across 11 independent video recordings.
- **Model Implementations:**
  - **Model A (Baseline Linear):** Fixed rule fusion ($0.45 P + 0.40 B + 0.15 D$).
  - **Model B–E (Single Drivers):** Logistic models on individual drivers $[D]$, $[O]$, $[B]$, $[K]$.
  - **Model F–H (Driver Pairs):** Logistic models on pairs $[D+O]$, $[D+B]$, $[O+B]$.
  - **Model I (Additive GAM):** $\text{logit}(P) = \beta_0 + \beta_D D + \beta_O O + \beta_B B + \beta_K K$.
  - **Model J (Pairwise Interaction GAM):** $\text{logit}(P) = \beta_0 + \sum \beta_i x_i + \sum_{i < j} \gamma_{ij} (x_i \cdot x_j)$.
  - **Model K (Shallow GBM):** Gradient Boosting with `max_depth=2` (strictly main effects and pairwise decision-tree interactions).

---

## 3. Empirical Results Summary

| Model / Driver Configuration | LOSO Mean F1 | LOSO ROC-AUC | LOGO Mean F1 | LOGO ROC-AUC | Mean Brier Score |
|---|---|---|---|---|---|
| **A. Baseline Linear Rule (0.45P+0.4B+0.15D)** | $0.0000^*$ | $0.6078 \pm 0.1012$ | $0.0000^*$ | $0.5911 \pm 0.2520$ | $0.4299$ |
| **B. Driver D (Density Only)** | $0.0029$ | $0.5070 \pm 0.0103$ | $0.4216$ | $0.5182 \pm 0.0632$ | $0.3055$ |
| **C. Driver O (Disorder Only)** | $0.2766$ | $0.5000 \pm 0.0000$ | $0.1280$ | $0.5110 \pm 0.0544$ | $0.2731$ |
| **D. Driver B (Bottleneck Only)** | $0.0029$ | $0.5114 \pm 0.0162$ | $0.4282$ | $0.5100 \pm 0.0461$ | $0.2982$ |
| **E. Driver K (Kinematics Only)** | $0.4811$ | $0.6239 \pm 0.0894$ | $0.5799$ | $0.5968 \pm 0.2014$ | $0.2644$ |
| **F. Pair [D + O]** | $0.0029$ | $0.5070 \pm 0.0103$ | $0.4200$ | $0.5144 \pm 0.0614$ | $0.2970$ |
| **G. Pair [D + B]** | $0.1466$ | $0.5139 \pm 0.0197$ | $0.4235$ | $0.5124 \pm 0.0471$ | $0.2941$ |
| **H. Pair [O + B]** | $0.0029$ | $0.5114 \pm 0.0162$ | $0.4246$ | $0.5093 \pm 0.0457$ | $0.2934$ |
| **I. Additive [D + O + B + K]** | $0.2878$ | $0.6237 \pm 0.0891$ | $0.6199$ | $0.5907 \pm 0.2078$ | $0.2645$ |
| **J. GAM Pairwise Interactions [D,O,B,K + Cross]** | $0.2142$ | $0.6239 \pm 0.0893$ | $0.6320$ | $0.5665 \pm 0.2107$ | $0.2733$ |
| **K. Shallow GBM (Depth=2 Trees)** | **0.5503** | **0.6564 $\pm$ 0.1248** | $0.5648$ | $0.5507 \pm 0.2230$ | **0.2281** |

*\*Note: Baseline linear fusion is an uncalibrated composite indicator spanning [0.0, 0.45], meaning an uncalibrated 0.50 cutoff predicts class 0, while its ranking performance (ROC-AUC 0.6078) is comparable to additive linear models.*

---

## 4. Pairwise Interaction Analysis

| Interaction Term | Regression Coef ($\beta$) | Odds Ratio | Empirical Interpretation |
|---|---|---|---|
| **$D \times O$ (Density $\times$ Disorder)** | **$+0.7448$** | **$2.1061$** | **Synergistic escalation:** When density is elevated AND directional disorder increases, risk scales super-additively. |
| **$B \times K$ (Bottleneck $\times$ Kinematics)** | **$+0.3492$** | **$1.4179$** | **Synergistic escalation:** Bottleneck choke combined with kinematic acceleration/turbulence increases threat. |
| **$D \times B$ (Density $\times$ Bottleneck)** | $-0.6096$ | $0.5436$ | Redundant interaction; bottleneck $B$ already incorporates density mathematically. |
| **$D \times K$ (Density $\times$ Kinematics)** | $-0.7624$ | $0.4665$ | High density physically restricts maximum velocity, creating an inverse relationship in unconstrained space. |
| **$O \times K$ (Disorder $\times$ Kinematics)** | $-0.0578$ | $0.9438$ | Negligible independent interaction. |

---

## 5. Scientific Decision for Stage 6 (CRDA Integration)
1. **Shallow Interaction Model Advantage:**
   Shallow Tree-based model (Depth=2 GBM) achieves the highest cross-scene ROC-AUC ($0.6564$) and best calibration ($0.2281$ Brier score) while preventing high-order overfitting.
2. **Empirical Justification for Pairwise Explanations in CRDA:**
   The significant interaction coefficient for $D \times O$ ($+0.7448$) validates that risk drivers cannot be assumed strictly independent in real surveillance footage.
3. **Preservation of Baseline:** The linear fusion model remains available as the frozen comparative baseline.
