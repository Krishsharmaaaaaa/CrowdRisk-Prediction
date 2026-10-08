"""
Spatially Localized Counterfactual Risk-Driver Attribution (CRDA) Engine.
Computes model-based intervention analysis on elevated-risk surveillance grid cells:
- Perturbs candidate risk drivers (D, O, B, K) toward same-scene calm reference states.
- Quantifies risk reduction Delta_R_g = R(x_c) - R(x_c^CF).
- Identifies Top-1 driver attribution.
- Evaluates exhaustive non-empty subsets (15 subsets) to find minimal intervention set S*.
Note: This provides model-based intervention analysis, NOT causal proof.
"""

from dataclasses import asdict, dataclass, field
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from ..features.cell_grid import (
    SameSceneCalmReferenceManager,
    SpatialCellFeatureVector,
    SpatialGridFeatureMap
)


DRIVER_NAMES = ["Density", "Disorder", "Bottleneck", "Kinematics"]
DRIVER_KEYS = ["D", "O", "B", "K"]


@dataclass
class DriverAttribution:
    """Attribution metrics for a single risk driver."""
    driver_key: str              # 'D', 'O', 'B', 'K'
    driver_name: str             # 'Density', 'Disorder', 'Bottleneck', 'Kinematics'
    delta_R: float               # Risk reduction: R_orig - R_cf
    initial_value: float         # Current observed driver value
    calm_reference: float        # Calm reference baseline value
    counterfactual_value: float  # Value after perturbation toward calm
    counterfactual_risk: float   # Predicted risk if this driver alone is perturbed
    rank: int = 1                # 1 = highest risk reduction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "driver_key": self.driver_key,
            "driver_name": self.driver_name,
            "delta_R": round(self.delta_R, 4),
            "initial_value": round(self.initial_value, 4),
            "calm_reference": round(self.calm_reference, 4),
            "counterfactual_value": round(self.counterfactual_value, 4),
            "counterfactual_risk": round(self.counterfactual_risk, 4),
            "rank": self.rank
        }


@dataclass
class SubsetIntervention:
    """Intervention outcome for a subset of drivers perturbed together."""
    subset_keys: List[str]       # e.g., ['B'] or ['B', 'D']
    subset_names: List[str]      # e.g., ['Bottleneck', 'Density']
    cardinality: int             # 1, 2, 3, or 4
    initial_risk: float
    counterfactual_risk: float
    delta_R: float
    achieved_safe_threshold: bool
    resulting_threat_level: str  # 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subset_keys": self.subset_keys,
            "subset_names": self.subset_names,
            "cardinality": self.cardinality,
            "initial_risk": round(self.initial_risk, 4),
            "counterfactual_risk": round(self.counterfactual_risk, 4),
            "delta_R": round(self.delta_R, 4),
            "achieved_safe_threshold": self.achieved_safe_threshold,
            "resulting_threat_level": self.resulting_threat_level
        }


@dataclass
class CellCRDAExplanation:
    """Full CRDA explanation report for a single spatial grid cell."""
    cell_id: str                 # e.g., 'C3'
    row: int
    col: int
    initial_risk: float
    initial_threat_level: str
    current_features: Dict[str, float]
    calm_reference: Dict[str, float]
    alpha: float                 # Perturbation strength (e.g. 1.0)
    top_driver: str              # Name of driver with largest single-driver delta_R
    top_driver_key: str          # 'D', 'O', 'B', 'K'
    top_driver_delta_R: float
    single_driver_attributions: List[DriverAttribution]
    all_subset_interventions: List[SubsetIntervention]
    minimal_intervention_subset: SubsetIntervention
    narrative_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "col": self.col,
            "initial_risk": round(self.initial_risk, 4),
            "initial_threat_level": self.initial_threat_level,
            "current_features": {k: round(v, 4) for k, v in self.current_features.items()},
            "calm_reference": {k: round(v, 4) for k, v in self.calm_reference.items()},
            "alpha": self.alpha,
            "top_driver": self.top_driver,
            "top_driver_key": self.top_driver_key,
            "top_driver_delta_R": round(self.top_driver_delta_R, 4),
            "single_driver_attributions": [a.to_dict() for a in self.single_driver_attributions],
            "minimal_intervention_subset": self.minimal_intervention_subset.to_dict(),
            "all_subset_interventions": [s.to_dict() for s in self.all_subset_interventions],
            "narrative_explanation": self.narrative_explanation
        }


@dataclass
class GridCRDAReport:
    """CRDA report across all elevated-risk cells in a video frame."""
    frame_idx: int
    timestamp: float
    high_risk_threshold: float
    elevated_cells_count: int
    cell_explanations: List[CellCRDAExplanation]
    most_critical_cell: Optional[CellCRDAExplanation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "high_risk_threshold": self.high_risk_threshold,
            "elevated_cells_count": self.elevated_cells_count,
            "most_critical_cell_id": self.most_critical_cell.cell_id if self.most_critical_cell else None,
            "cell_explanations": [c.to_dict() for c in self.cell_explanations]
        }


class CRDAEngine:
    """
    Spatially Localized Counterfactual Risk-Driver Attribution Engine.
    Evaluates localized driver perturbations toward same-scene calm reference states.
    """

    def __init__(
        self,
        risk_model_fn: Optional[Callable[[np.ndarray], float]] = None,
        high_risk_threshold: float = 0.60,
        moderate_risk_threshold: float = 0.30,
        default_alpha: float = 1.0,
        enable_interaction_synergy: bool = True
    ):
        self.risk_model_fn = risk_model_fn or self._default_cell_risk_fusion
        self.high_risk_threshold = high_risk_threshold
        self.moderate_risk_threshold = moderate_risk_threshold
        self.default_alpha = float(np.clip(default_alpha, 0.0, 1.0))
        self.enable_interaction_synergy = enable_interaction_synergy

        # Driver mapping
        self.key_to_name = dict(zip(DRIVER_KEYS, DRIVER_NAMES))
        self.key_to_idx = {k: i for i, k in enumerate(DRIVER_KEYS)}

    def _default_cell_risk_fusion(self, x: np.ndarray) -> float:
        """
        Default model-based cell risk evaluator combining additive main effects
        with empirical pairwise interaction synergy (D x O and B x K).
        Vector x: [D, O, B, K] in [0, 1]^4.
        """
        D, O, B, K = float(x[0]), float(x[1]), float(x[2]), float(x[3])

        # Main additive drivers
        base = 0.35 * D + 0.20 * O + 0.30 * B + 0.15 * K

        if self.enable_interaction_synergy:
            # Pairwise interaction synergy validated in Stage 5:
            # Density x Disorder synergy (+0.25) and Bottleneck x Kinematics synergy (+0.15)
            interaction_boost = 0.25 * (D * O) + 0.15 * (B * K)
            score = base + interaction_boost
        else:
            score = base

        return float(np.clip(score, 0.0, 1.0))

    def _classify_threat_level(self, risk: float) -> str:
        if risk < self.moderate_risk_threshold:
            return "LOW"
        elif risk < self.high_risk_threshold:
            return "MODERATE"
        elif risk < 0.80:
            return "HIGH"
        else:
            return "CRITICAL"

    def compute_cell_risk(self, driver_vector: np.ndarray) -> float:
        """Evaluates model risk on a given 4-dimensional driver vector."""
        return self.risk_model_fn(driver_vector)

    def explain_cell(
        self,
        cell: SpatialCellFeatureVector,
        calm_reference: np.ndarray,
        alpha: Optional[float] = None,
        target_safe_threshold: Optional[float] = None
    ) -> CellCRDAExplanation:
        """
        Computes single-driver attribution and exhaustive minimal intervention subset
        for a single spatial grid cell.
        """
        a = self.default_alpha if alpha is None else float(np.clip(alpha, 0.0, 1.0))
        target_thresh = self.high_risk_threshold if target_safe_threshold is None else target_safe_threshold

        x_curr = cell.to_driver_vector().copy()
        x_calm = np.asarray(calm_reference, dtype=np.float32).copy()

        # Initial risk state
        initial_risk = self.compute_cell_risk(x_curr)
        initial_threat = self._classify_threat_level(initial_risk)

        # ----------------------------------------------------
        # 1. Single-Driver Counterfactual Attributions
        # ----------------------------------------------------
        single_attributions: List[DriverAttribution] = []

        for k in DRIVER_KEYS:
            idx = self.key_to_idx[k]
            x_cf = x_curr.copy()
            # Perturb only driver idx toward its calm reference
            x_cf[idx] = (1.0 - a) * x_curr[idx] + a * x_calm[idx]

            cf_risk = self.compute_cell_risk(x_cf)
            delta_R = max(0.0, initial_risk - cf_risk)

            attr = DriverAttribution(
                driver_key=k,
                driver_name=self.key_to_name[k],
                delta_R=delta_R,
                initial_value=float(x_curr[idx]),
                calm_reference=float(x_calm[idx]),
                counterfactual_value=float(x_cf[idx]),
                counterfactual_risk=cf_risk
            )
            single_attributions.append(attr)

        # Sort single attributions by delta_R descending
        single_attributions.sort(key=lambda x: x.delta_R, reverse=True)
        for i, attr in enumerate(single_attributions):
            attr.rank = i + 1

        top_attr = single_attributions[0]
        top_driver_name = top_attr.driver_name
        top_driver_key = top_attr.driver_key
        top_delta = top_attr.delta_R

        # ----------------------------------------------------
        # 2. Exhaustive Subset Interventions (2^4 - 1 = 15 subsets)
        # ----------------------------------------------------
        all_subsets_results: List[SubsetIntervention] = []

        # Generate subsets of size 1, 2, 3, 4
        for r in range(1, len(DRIVER_KEYS) + 1):
            for subset in itertools.combinations(DRIVER_KEYS, r):
                x_cf_sub = x_curr.copy()
                for k in subset:
                    idx = self.key_to_idx[k]
                    x_cf_sub[idx] = (1.0 - a) * x_curr[idx] + a * x_calm[idx]

                cf_sub_risk = self.compute_cell_risk(x_cf_sub)
                delta_R_sub = max(0.0, initial_risk - cf_sub_risk)
                achieved_safe = cf_sub_risk < target_thresh
                threat_lvl = self._classify_threat_level(cf_sub_risk)

                sub_interv = SubsetIntervention(
                    subset_keys=list(subset),
                    subset_names=[self.key_to_name[k] for k in subset],
                    cardinality=len(subset),
                    initial_risk=initial_risk,
                    counterfactual_risk=cf_sub_risk,
                    delta_R=delta_R_sub,
                    achieved_safe_threshold=achieved_safe,
                    resulting_threat_level=threat_lvl
                )
                all_subsets_results.append(sub_interv)

        # ----------------------------------------------------
        # 3. Minimal Intervention Subset Selection
        # ----------------------------------------------------
        # Prioritize subsets that achieve safe threshold (< target_thresh):
        # 1st criterion: smallest cardinality (simplest intervention)
        # 2nd criterion: largest delta_R (strongest risk reduction)
        valid_safe_subsets = [s for s in all_subsets_results if s.achieved_safe_threshold]

        if valid_safe_subsets:
            # Sort by cardinality ascending, then delta_R descending
            valid_safe_subsets.sort(key=lambda s: (s.cardinality, -s.delta_R))
            minimal_subset = valid_safe_subsets[0]
        else:
            # If no subset achieves safe threshold, select subset with maximum risk reduction
            all_subsets_results.sort(key=lambda s: (-s.delta_R, s.cardinality))
            minimal_subset = all_subsets_results[0]

        # ----------------------------------------------------
        # 4. Synthesize Readable Narrative
        # ----------------------------------------------------
        subset_str = " + ".join(minimal_subset.subset_names)
        narrative = (
            f"ZONE {cell.cell_id} | Risk: {initial_threat} ({initial_risk:.2f}). "
            f"Primary risk driver is {top_driver_name} (Delta R = {top_delta:.2f}). "
            f"Minimal intervention: reducing [{subset_str}] toward calm baseline "
            f"reduces predicted risk from {initial_risk:.2f} -> {minimal_subset.counterfactual_risk:.2f} "
            f"({initial_threat} -> {minimal_subset.resulting_threat_level})."
        )

        return CellCRDAExplanation(
            cell_id=cell.cell_id,
            row=cell.row,
            col=cell.col,
            initial_risk=initial_risk,
            initial_threat_level=initial_threat,
            current_features={"D": cell.D, "O": cell.O, "B": cell.B, "K": cell.K},
            calm_reference={"D": float(x_calm[0]), "O": float(x_calm[1]), "B": float(x_calm[2]), "K": float(x_calm[3])},
            alpha=a,
            top_driver=top_driver_name,
            top_driver_key=top_driver_key,
            top_driver_delta_R=top_delta,
            single_driver_attributions=single_attributions,
            all_subset_interventions=all_subsets_results,
            minimal_intervention_subset=minimal_subset,
            narrative_explanation=narrative
        )

    def explain_grid(
        self,
        grid_map: SpatialGridFeatureMap,
        calm_mgr: SameSceneCalmReferenceManager,
        alpha: Optional[float] = None,
        only_elevated: bool = True
    ) -> GridCRDAReport:
        """
        Generates CRDA explanations across all elevated or active cells in a frame.
        """
        explanations: List[CellCRDAExplanation] = []

        for cell in grid_map.cells:
            # Check cell risk
            calm_ref = calm_mgr.get_reference_vector(cell.cell_id)
            c_risk = self.compute_cell_risk(cell.to_driver_vector())

            # Only explain cells with elevated threat or non-zero occupants if configured
            if only_elevated:
                if c_risk < self.moderate_risk_threshold and cell.person_count == 0:
                    continue

            expl = self.explain_cell(cell, calm_ref, alpha=alpha)
            explanations.append(expl)

        # Identify most critical cell
        if explanations:
            most_crit = max(explanations, key=lambda e: e.initial_risk)
        else:
            most_crit = None

        return GridCRDAReport(
            frame_idx=grid_map.frame_idx,
            timestamp=grid_map.timestamp,
            high_risk_threshold=self.high_risk_threshold,
            elevated_cells_count=len(explanations),
            cell_explanations=explanations,
            most_critical_cell=most_crit
        )
