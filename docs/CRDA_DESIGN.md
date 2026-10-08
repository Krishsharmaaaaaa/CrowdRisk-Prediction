# Spatially Localized Risk-Driver Attribution (CRDA) Design Specification

**Status:** STAGE 6 COMPLETE (CRDA Engine Implemented, Tested & Integrated)  
**Date:** October 2026

---

## 1. Mathematical Formulation of CRDA

For any elevated spatial grid cell $c \in \{C_1, \dots, C_{16}\}$:
- **Observed Feature Vector:** $\mathbf{x}_c = [D_c, O_c, B_c, K_c] \in [0, 1]^4$
- **Same-Scene Calm Baseline Reference:** $\mathbf{\bar{x}}_c^{\text{calm}} = [\bar{D}_c, \bar{O}_c, \bar{B}_c, \bar{K}_c] \in [0, 1]^4$ (empirical 25th-percentile / median of low-threat windows from the same scene)
- **Perturbation Strength Parameter:** $\alpha \in [0.0, 1.0]$ (default $\alpha = 1.0$)

### Single-Driver Counterfactual Attribution
For each driver group $g \in \{D, O, B, K\}$:
$$\mathbf{x}_c^{\text{CF}}(g, \alpha)_i = \begin{cases} (1 - \alpha) x_i + \alpha \bar{x}_i^{\text{calm}}, & i = g \\ x_i, & i \neq g \end{cases}$$
$$\Delta R_g = \max\left(0.0, R(\mathbf{x}_c) - R(\mathbf{x}_c^{\text{CF}}(g, \alpha))\right)$$

* **Driver Ranking:** Drivers are sorted by $\Delta R_g$ descending. The driver yielding the largest $\Delta R_g$ is designated the **Primary Model-Based Risk Driver** ($g^*$).

---

## 2. Exhaustive Minimal Intervention Subset Search ($S^*$)

Because there are only 4 candidate drivers ($|G| = 4$), the power set of non-empty driver subsets contains exactly $2^4 - 1 = 15$ subsets:
- **Cardinality 1 (4 subsets):** $\{D\}, \{O\}, \{B\}, \{K\}$
- **Cardinality 2 (6 subsets):** $\{D, O\}, \{D, B\}, \{D, K\}, \{O, B\}, \{O, K\}, \{B, K\}$
- **Cardinality 3 (4 subsets):** $\{D, O, B\}, \{D, O, K\}, \{D, B, K\}, \{O, B, K\}$
- **Cardinality 4 (1 subset):** $\{D, O, B, K\}$

For each subset $S \subseteq \{D, O, B, K\}$:
$$\mathbf{x}_c^{\text{CF}}(S, \alpha)_i = \begin{cases} (1 - \alpha) x_i + \alpha \bar{x}_i^{\text{calm}}, & i \in S \\ x_i, & i \notin S \end{cases}$$
$$R_{\text{CF}}(S) = R(\mathbf{x}_c^{\text{CF}}(S, \alpha))$$
$$\Delta R_S = R(\mathbf{x}_c) - R_{\text{CF}}(S)$$

### Optimization Objective for Minimal Intervention $S^*$:
$$S^* = \arg\min_{S \subseteq G} |S| \quad \text{subject to } R_{\text{CF}}(S) < R_{\text{safe}}$$
Ties with equal minimal cardinality $|S|$ are resolved by selecting the subset with the largest risk reduction $\Delta R_S$.
If no subset brings risk below $R_{\text{safe}}$, the subset maximizing $\Delta R_S$ is reported.

---

## 3. Data Structures (`src/explainability/crda_engine.py`)

1. **`DriverAttribution`:**
   - `driver_key`: `'D'`, `'O'`, `'B'`, `'K'`
   - `driver_name`: `'Density'`, `'Disorder'`, `'Bottleneck'`, `'Kinematics'`
   - `delta_R`: Risk reduction achieved by perturbing this driver alone
   - `initial_value`, `calm_reference`, `counterfactual_value`, `counterfactual_risk`, `rank`
2. **`SubsetIntervention`:**
   - `subset_keys`, `subset_names`, `cardinality`
   - `initial_risk`, `counterfactual_risk`, `delta_R`
   - `achieved_safe_threshold`, `resulting_threat_level`
3. **`CellCRDAExplanation`:**
   - `cell_id`, `row`, `col`, `initial_risk`, `initial_threat_level`
   - `top_driver`, `top_driver_key`, `top_driver_delta_R`
   - `single_driver_attributions` (4 entries, sorted by $\Delta R$)
   - `minimal_intervention_subset` (minimal cardinality solution)
   - `all_subset_interventions` (all 15 evaluated subsets)
   - `narrative_explanation`
4. **`GridCRDAReport`:**
   - Aggregates explanations across all elevated cells in the surveillance frame.

---

## 4. Example Output (Real Crowd Video / UMN Video)

```json
{
    "cell_id": "C5",
    "initial_risk": 0.4174,
    "initial_threat_level": "MODERATE",
    "top_driver": "Density",
    "top_driver_delta_R": 0.0823,
    "single_driver_attributions": [
        {
            "driver_name": "Density",
            "delta_R": 0.0823,
            "counterfactual_risk": 0.3351,
            "rank": 1
        },
        {
            "driver_name": "Kinematics",
            "delta_R": 0.0093,
            "counterfactual_risk": 0.4081,
            "rank": 2
        }
    ],
    "minimal_intervention_subset": {
        "subset_names": ["Density"],
        "cardinality": 1,
        "delta_R": 0.0823,
        "counterfactual_risk": 0.3351,
        "achieved_safe_threshold": true
    },
    "narrative_explanation": "ZONE C5 | Risk: MODERATE (0.42). Primary risk driver is Density (Delta R = 0.08). Minimal intervention: reducing [Density] toward calm baseline reduces predicted risk from 0.42 -> 0.34 (MODERATE -> MODERATE)."
}
```

---

## 5. Non-Causal Semantics Guardrail
* All CRDA outputs are described as:
  - **"Model-based counterfactual attribution"**
  - **"Predictive risk sensitivity to calm reference perturbation"**
  - **"Minimal model-based intervention subset"**
* **NEVER described as:** causal proof, causal inference, or ground-truth intervention guarantee.

---

## 6. Formal Mathematical Definitions & Simulation Validation

### 6.1 Thresholds and Minimal Subset Selection
1. **$R_{\text{safe}}$ Definition**: Target safe threshold ($R_{\text{safe}}$), defaulting to `high_risk_threshold = 0.60`. Risk levels below $0.60$ are classified as `MODERATE` ($[0.30, 0.60)$) or `LOW` ($< 0.30$).
2. **`achieved_safe_threshold` Condition**: Evaluated strictly as:
   $$R(\mathbf{x}_c^{\text{CF}}) < R_{\text{safe}}$$
3. **Minimal Intervention Subset Criterion**: Strictly selects the **minimum-cardinality subset** among all subsets satisfying $R(\mathbf{x}_c^{\text{CF}}) < R_{\text{safe}}$:
   $$S^* = \arg\min_{S \subseteq \{D, O, B, K\}, R(\mathbf{x}_c^{S}) < R_{\text{safe}}} |S|$$
   Ties in cardinality are broken by selecting the subset yielding maximal risk reduction $\Delta R$.
4. **Fallback When No Subset Achieves $R_{\text{safe}}$**: If all 15 subsets yield $R \ge R_{\text{safe}}$, the engine selects the subset producing the maximal global risk reduction ($\max \Delta R$), setting `achieved_safe_threshold = False`.
5. **Driver Invariance**: When subset $S$ is perturbed, all features $j \notin S$ remain strictly identical to their initial observed state $\mathbf{x}_c[j]$.

### 6.2 Independent Simulation Validation (Stage 7)
Validated against an independent 2D microscopic pedestrian evacuation simulation testbed (`src/simulation/pedestrian_sim.py`) across 175 runs:
- Recommended multi-driver subsets produce the strongest clearance time reductions (up to $+6.39$s).
- Detailed in [`docs/SIMULATION_VALIDATION.md`](file:///c:/Users/ns488/Downloads/CrowdRisk-Prediction/docs/SIMULATION_VALIDATION.md).

### 6.3 Decision-Support HUD Visualization Layer (Stage 9)
Integrated into real-time surveillance video display (`src/visualization/decision_hud.py`):
- **Global Risk Panel**: Labeled strictly as `MODEL RISK` (non-calibrated model index).
- **CRDA Driver Panel**: Renders exact $\Delta R_D, \Delta R_O, \Delta R_B, \Delta R_K$ attributions and descending rankings.
- **4x4 Spatial Heatmap**: Displays per-cell local risk $R_c$, threat level, and dominant driver.
- **Minimal Intervention Recommender**: Visualizes $S^*$, predicted $\Delta R$, and safe threshold verification. Explicitly displays `NO SAFE-THRESHOLD INTERVENTION FOUND` when $R \ge R_{\text{safe}}$.
- **Detailed in**: [`docs/HUD_DEMO.md`](file:///c:/Users/ns488/Downloads/CrowdRisk-Prediction/docs/HUD_DEMO.md).


