# Literature Novelty Verification & Gap Audit

**Proposed Add-On:** CRDA — Spatially Localized Counterfactual Risk-Driver Attribution  
**Audit Date:** October 2026  
**Status:** COMPLETE  
**Novelty Verdict:** **CONDITIONAL**  

---

## 1. Literature Search Protocol

Searches performed across **Google Scholar, IEEE Xplore, ACM Digital Library, Springer, ScienceDirect, arXiv, and Google Patents** covering 2017–2026:
- `"counterfactual" "crowd" "risk" OR "evacuation"`
- `"video anomaly detection" "counterfactual explanation" "spatiotemporal"`
- `"explainable crowd risk" OR "counterfactual bottleneck"`
- `"actionable recourse" "pedestrian safety" OR "crowd management"`
- `"intervention" "crowd simulation" "evacuation"`

---

## 2. Classification of Related Works

### Work 1: Generative Counterfactual Video Anomaly Detection (EAD-CE / Causally Invariant VAD)
* **Citation:** Sun et al. / Li et al. (AAAI 2023, CVPR/ICCV 2024–2025)
* **Classification:** **YELLOW (Partial Overlap - Claim Must Be Narrowed)**
* **What it does:** Uses generative diffusion / spatiotemporal auto-encoders to alter video pixel regions and hallucinate "normal" appearance versions of anomalous video frames.
* **Overlap:** Uses counterfactual imagery to explain video anomalies in surveillance.
* **What remains different:** They focus on unsupervised visual reconstruction (pixel-level auto-encoders) for general object anomalies (e.g. bicyclist on sidewalk), NOT interpretable kinematic/density driver attribution ($D, O, B, K$), spatial choke-point decomposition, or minimal intervention sets for crowd evacuation decision-support.
* **Safe Claim:** We do not claim generative video counterfactuals. We evaluate explicit semantic crowd telemetry drivers ($D, O, B, K$) on spatial grids.

### Work 2: Counterfactual Explanations in Tabular & Safety-Critical Machine Learning (DiCE / Alibi)
* **Citation:** Wachter et al. (2017), Mothilal et al. (ACM FAccT 2020)
* **Classification:** **RED (Direct Concept Overlap - Cannot Claim General Novelty)**
* **What it does:** Formalizes feature perturbation $\mathbf{x}^* = \mathbf{x} + \mathbf{\delta}$ to change a classification outcome $f(\mathbf{x}) \to y^*$ with minimal distance.
* **Overlap:** The mathematical definition of counterfactual explanation $\Delta R = f(\mathbf{x}) - f(\mathbf{x}_{cf})$ is directly based on these foundational works.
* **What remains different:** General CFE libraries operate on static tabular rows (e.g., credit risk, loan approval). They do not account for spatial grid crowd physics, same-scene empirical calm references, or multi-agent physical bottlenecks.
* **Safe Claim:** We do NOT claim counterfactuals are novel. We apply domain-specific driver-group intervention analysis to spatial crowd telemetry.

### Work 3: Agent-Based Evacuation Simulation & 'What-If' Parameter Sensitivity
* **Citation:** Helbing et al. (Nature 2000), Moussaïd et al. (PNAS 2011), RESCUE (ICCV 2025)
* **Classification:** **YELLOW (Partial Overlap - Claim Must Be Narrowed)**
* **What it does:** Simulates pedestrian dynamics (Social Force Model / Cellular Automata) under modified structural parameters (e.g. door width 1m vs 2m, exit placement, flow barriers).
* **Overlap:** Investigates how physical changes impact crowd evacuation.
* **What remains different:** Evacuation physics simulators simulate forward trajectories given an environment; they do NOT perform real-time automated video-to-driver attribution, spatial cell ranking, or automated minimal intervention set search from live CCTV streams.
* **Safe Claim:** Simulation is used as an **independent evaluation testbed** to validate whether the vision model's identified primary bottleneck matches physical simulator throughput improvements.

### Work 4: Feature Attribution for Crowd Density & Flow (Grad-CAM, SHAP on Flow)
* **Citation:** Various IEEE Trans. Intelligent Transportation / Multimedia (2020–2024)
* **Classification:** **GREEN (Clearly Different)**
* **What it does:** Computes feature heatmaps / SHAP values showing correlational feature importance.
* **What remains different:** Standard attribution shows correlation ("Density contributed 35%"), whereas CRDA determines actionable counterfactual recourse ("Reducing bottleneck to reference lowers risk from HIGH to MODERATE; reducing density alone is insufficient").

---

## 3. Novelty Verdict: CONDITIONAL

### Why CONDITIONAL?
1. Counterfactual explanations are **NOT** a new AI theory (established since Wachter 2017).
2. Video anomaly detection and pedestrian simulation are well-established fields.
3. **The defensible novelty is strictly the integration & validation:**
   - Spatially localized crowd-driver attribution ($D, O, B, K$) on surveillance grid cells.
   - Same-scene calm-reference counterfactual perturbation ($\mathbf{x}_c \to \mathbf{\bar{x}}_g$).
   - Minimal intervention subset search ($S^* \subseteq \{D, O, B, K\}$) to drop below the alert threshold.
   - Validation against an **independent simulator-native outcome** (evacuation time / bottleneck throughput) to avoid circular mathematical proof.

---

## 4. Exact Claims Permitted vs. Strictly Prohibited

| Claim Category | Scientifically Permitted Claim | Strictly Prohibited / Unjustified Claim |
|---|---|---|
| **Novelty** | "We propose a spatially localized model-based counterfactual driver attribution framework for video crowd risk." | ❌ "We present the first-ever counterfactual explanation system in AI/crowd safety." |
| **Causality** | "Model-based intervention analysis / counterfactual sensitivity." | ❌ "Causal proof / proven real-world causal discovery." |
| **Prediction** | "Decision-support risk scoring with early trend detection." | ❌ "Guaranteed disaster prevention / perfect stampede prediction." |
| **Performance** | "Evaluated under GroupKFold / Leave-One-Scene-Out cross-validation." | ❌ "100% accuracy on synthetic demo frames." |
