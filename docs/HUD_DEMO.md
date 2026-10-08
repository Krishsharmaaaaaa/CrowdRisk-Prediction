# Stage 9 — Decision-Support HUD Demonstration

**Date**: October 2026  
**Status**: Complete & Verified  
**Architecture Version**: v1.8.0 (Decision-Support HUD Integrated Prototype)  
**Total Test Suite**: **43 / 43 passing** (`pytest -q`)

---

## 1. Executive Summary

Stage 9 establishes the final **Decision-Support HUD Visualization Layer** over the crowd risk prediction pipeline. The HUD visualizes existing outputs without modifying any underlying detection, tracking, feature extraction, risk fusion, or CRDA models.

The system translates complex computer vision telemetry into clear, research-grade, non-causal decision-support indicators for emergency operators and crowd safety researchers.

---

## 2. Threshold Consistency Verification (R_safe)

Before implementation, a scientific consistency audit was conducted regarding the threshold discrepancy:
* **Stage 6/7 Standard**: `R_safe = high_risk_threshold = 0.60` (the boundary below which risk is MODERATE or LOW).
* **Stage 8 Mention**: A text summary in `docs/ABLATION_ROBUSTNESS.md` mentioned `R < R_safe (0.35)`.

### Audit Finding
1. **Actual Code Execution**: In `scripts/run_ablation_robustness.py`, `crda_engine` was initialized with `high_risk_threshold=0.60` and evaluated `safe_reached = bool(cf_r < 0.60)`. In `results/ablation_robustness.json`, all evaluated subsets had `safe_reached: true` because their counterfactual risks were below 0.60.
2. **Documentation Typo**: The text `0.35` in `docs/ABLATION_ROBUSTNESS.md` was an accidental documentation typo conflating the moderate/low risk boundary (0.30) or baseline calm weights with `R_safe`.
3. **Strict Resolution**:
   - The project-wide standard remains `R_safe = 0.60` across `src/explainability/crda_engine.py`, `docs/CRDA_DESIGN.md`, and all experiment runners.
   - Clarification has been formally recorded in `docs/ABLATION_ROBUSTNESS.md`: even under a hypothetical strict experimental threshold of `R_safe = 0.35`, Density ({D}) remains the unique minimal intervention subset crossing below 0.35 (reaching R ~ 0.19).

---

## 3. HUD Architecture & Visual Components

The HUD is implemented in [`src/visualization/decision_hud.py`](file:///c:/Users/ns488/Downloads/CrowdRisk-Prediction/src/visualization/decision_hud.py) via `DecisionSupportHUD`, integrated into [`src/visualization/visualizer.py`](file:///c:/Users/ns488/Downloads/CrowdRisk-Prediction/src/visualization/visualizer.py) and [`src/pipeline.py`](file:///c:/Users/ns488/Downloads/CrowdRisk-Prediction/src/pipeline.py).

### Panel 1: Global Risk Panel (Top-Left)
* **Metric Display**: Labeled strictly as **`MODEL RISK: [score]`** (e.g. `0.72`).
* **Threat Level**: `LOW`, `MODERATE`, `HIGH`, `CRITICAL` with corresponding color tokens.
* **Early Warning State**: `NORMAL`, `MONITORING`, `WARNING`, `ACTIVE` plus temporal trend (`RISING`, `FALLING`, `STABLE`).
* **Timestamp**: Frame index and video timestamp in seconds.
* **Non-Probability Guardrail**: Explicit label: `[Non-calibrated model index in [0, 1]]`. It is never labeled as a calibrated probability.

### Panel 2: CRDA Driver Attribution Panel (Top-Right)
* **Four Driver Groups**: Density (D), Disorder (O), Bottleneck (B), Kinematics (K).
* **Exact Attributions**: Displays actual delta_R values from counterfactual evaluation (e.g. `D: dR = 0.312`, `O: dR = 0.082`). No invented normalized percentages.
* **Ordinal Rankings**: Ordered `#1` through `#4` by descending delta_R.
* **Driver Magnitude Bars**: Horizontal proportional indicator tracks for intuitive operator comprehension.
* **Primary Driver**: Highlights top model-based risk contributor.

### Panel 3: 4x4 Spatial Risk Grid Heatmap Overlay (Video Center)
* **Grid Overlay**: Renders 16 spatial cells (`C1` through `C16`) directly registered to surveillance video coordinates.
* **Cell Metadata**: Each active cell displays `C{id} R:{local_risk} [{top_driver_key}]`.
* **Elevated Highlighting**: Cells with `R >= 0.30` or active occupants receive translucent threat-colored tinting (15% - 30% alpha).
* **Focus Cell Selection**: The most critical elevated cell (or operator-selected cell) receives an electric cyan/white highlight border and is linked to the intervention recommender.
* **Label**: `4x4 Spatial Grid (Model Risk)`. Heatmap intensity is explicitly model-derived.

### Panel 4: Minimal Intervention Recommender Panel (Top-Right Subpanel)
* **Target Cell**: Displays active focus cell ID (e.g. `ZONE C10` or `ZONE C6`).
* **State Transition**: `Model Risk: {initial_risk} -> {counterfactual_risk}`.
* **Minimal Subset**: Minimum-cardinality subset S* (e.g. `{Density}` or `{Disorder + Kinematics}`).
* **Predicted delta_R**: Risk reduction magnitude.
* **Safe Threshold Status**:
  - If counterfactual risk is below R_safe:
    `Safe Threshold Reached: YES (< 0.60)`
  - If counterfactual risk is at or above R_safe:
    **`NO SAFE-THRESHOLD INTERVENTION FOUND`**  
    `Fallback (Max dR): {subset} (dR=0.09)`  
    `Safe Threshold Reached: NO (CF Risk 0.61 >= 0.60)`  
    *(Never falsely labeled as a minimal safe intervention).*

### Panel 5: Operator Decision Support Card (Bottom-Left)
* **Operator Guidance**: Concise, non-causal action recommendation:
  - *Example (Safe subset found)*:
    `MODEL-BASED DECISION SUPPORT`  
    `Primary risk driver: Density.`  
    `Perturbing [Density] toward calm reference reduces model risk by dR = 0.31.`  
    `(Sensitivity attribution. Evacuation simulation provides independent benchmark.)`
  - *Example (High risk sustained)*:
    `Primary risk driver: Disorder (High Risk Sustained).`  
    `Perturbing [Disorder + Kinematics] provides maximal model risk reduction dR = 0.09.`
* **Guardrails**: Avoids causal claims ("prevent a stampede", "guaranteed egress improvement").

### Panel 6: Scientific Disclaimer Banner (Bottom Footer)
* Permanent footer across all output frames:
  `RESEARCH PROTOTYPE | MODEL-DERIVED RISK | CRDA: COUNTERFACTUAL ATTRIBUTION (NON-CAUSAL) | SIMULATION: INDEPENDENT BENCHMARK`

---

## 4. Demo Outputs & Artifacts

| Artifact | File Path | Resolution / Spec | Description |
|---|---|---|---|
| **UMN Demo Video** | `results/hud_demo_umn_indoor.mp4` | 960x720 @ 30 FPS (250 frames) | Full HUD overlay on official UMN Indoor Clip 4 benchmark. |
| **Concert Demo Video** | `results/hud_demo_concert.mp4` | 1280x720 @ 30 FPS (120 frames) | Full HUD overlay on real crowded concert footage. |
| **Normal Screenshot** | `results/hud_demo_frames/umn_indoor_01_normal_monitoring.png` | 960x720 PNG | UMN calm window with baseline tracking and low model risk. |
| **Elevated Screenshot** | `results/hud_demo_frames/real_concert_02_elevated_risk_crda.png` | 1280x720 PNG | Concert footage showing elevated risk and active CRDA attribution. |
| **Crisis Screenshot** | `results/hud_demo_frames/umn_indoor_03_peak_crisis_intervention.png` | 960x720 PNG | Peak UMN panic evacuation with minimal intervention recommendation. |
| **Telemetry JSON** | `results/hud_demo_telemetry.json` | 25 KB JSON | Complete per-frame machine-readable HUD and CRDA audit log. |

---

## 5. Reproducibility & CLI Execution

### 1. Run Complete HUD Demo Generator
```bash
python scripts/generate_hud_demo.py
```

### 2. Run Standard Pipeline with HUD Overlay on Custom Video
```bash
python run.py --source data/videos/indoor_clip4.mp4 --demo
```

### 3. Run Dedicated HUD Unit Tests
```bash
python -m pytest tests/test_decision_hud.py -v
```

---

## 6. Unit Test Verification (Stage 9)

**6 dedicated HUD tests passing** (`tests/test_decision_hud.py`):
1. `test_hud_risk_display_and_labeling`: Verifies MODEL RISK score, threat level, and non-probabilistic index labeling.
2. `test_hud_driver_ranking_and_deltas`: Verifies exact D, O, B, K delta_R attributions and descending ranking.
3. `test_hud_safe_threshold_reached`: Verifies that minimal subset crossing R_safe sets `safe_threshold_reached = True`.
4. `test_hud_no_safe_intervention_found_state`: Verifies explicit `NO SAFE-THRESHOLD INTERVENTION FOUND` banner when safe threshold is unreachable.
5. `test_hud_cell_selection_and_override`: Verifies default most-critical cell selection and explicit cell ID override.
6. `test_hud_deterministic_output`: Verifies exact pixel-level and telemetry reproducibility across identical runs.

**Total Project Test Suite**: **43 / 43 passing** (`pytest -q`).

---

## 7. Remaining Scientific Limitations

1. **Non-Causal Nature**: CRDA measures model prediction sensitivity with respect to a calm reference perturbation. It does not model physical crowd contact forces or panic psychology directly.
2. **Microscopic Simulation Gap**: As documented in Stage 7 and 8, CRDA agrees with Social Force evacuation clearance in 3/5 scenarios (60% Top-1 agreement). In disorder-driven panic, the statistical model prioritizes density while physical egress is throttled by angular disorder.
3. **Calibration**: Model risk scores [0, 1] are ordinal indicators of situational threat, not calibrated probabilities of evacuation failure.
