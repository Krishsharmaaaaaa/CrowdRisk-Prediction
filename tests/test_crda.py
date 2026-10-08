"""
Unit tests for Spatially Localized Counterfactual Risk-Driver Attribution (CRDA) Engine.
Tests:
- Single-driver attribution computation (D, O, B, K -> calm)
- Invariance of non-perturbed drivers
- Exhaustive generation of 15 non-empty driver subsets
- Minimal intervention subset selection (cardinality minimality)
- Perturbation strength alpha monotonicity
- Handling of calm/empty cells
- Narrative explanation synthesis
- Full grid CRDA report generation
"""

import numpy as np
import pytest

from src.explainability.crda_engine import (
    CRDAEngine,
    CellCRDAExplanation,
    DriverAttribution,
    GridCRDAReport,
    SubsetIntervention
)
from src.features.cell_grid import (
    SameSceneCalmReferenceManager,
    SpatialCellFeatureExtractor,
    SpatialCellFeatureVector,
    SpatialGridFeatureMap
)
from src.tracking.bytetrack_tracker import Track
from src.tracking.trajectory import TrajectoryManager


def test_crda_engine_initialization():
    engine = CRDAEngine(high_risk_threshold=0.60, moderate_risk_threshold=0.35)
    assert engine.high_risk_threshold == 0.60
    assert engine.moderate_risk_threshold == 0.35

    # Test baseline evaluation on zero vector
    risk_zero = engine.compute_cell_risk(np.zeros(4))
    assert risk_zero == 0.0

    # Test evaluation on elevated vector
    risk_elevated = engine.compute_cell_risk(np.array([0.8, 0.7, 0.9, 0.6]))
    assert risk_elevated > 0.60


def test_single_driver_attributions_and_invariance():
    engine = CRDAEngine(enable_interaction_synergy=False)  # Pure linear for exact math verification
    # Weights: D: 0.35, O: 0.20, B: 0.30, K: 0.15

    cell = SpatialCellFeatureVector(
        cell_id="C3",
        row=0,
        col=2,
        x1=320,
        y1=0,
        x2=480,
        y2=120,
        area_px=19200.0,
        D=0.80,
        O=0.20,
        B=0.70,
        K=0.10
    )
    calm_ref = np.array([0.10, 0.05, 0.05, 0.05], dtype=np.float32)

    expl = engine.explain_cell(cell, calm_ref, alpha=1.0)

    # Initial risk = 0.35*0.80 + 0.20*0.20 + 0.30*0.70 + 0.15*0.10 = 0.28 + 0.04 + 0.21 + 0.015 = 0.545
    assert abs(expl.initial_risk - 0.545) < 1e-4

    # Single attributions
    assert len(expl.single_driver_attributions) == 4

    # Delta for D: 0.35 * (0.80 - 0.10) = 0.35 * 0.70 = 0.245
    # Delta for B: 0.30 * (0.70 - 0.05) = 0.30 * 0.65 = 0.195
    # Delta for O: 0.20 * (0.20 - 0.05) = 0.20 * 0.15 = 0.030
    # Delta for K: 0.15 * (0.10 - 0.05) = 0.15 * 0.05 = 0.0075
    d_attr = next(a for a in expl.single_driver_attributions if a.driver_key == "D")
    b_attr = next(a for a in expl.single_driver_attributions if a.driver_key == "B")
    o_attr = next(a for a in expl.single_driver_attributions if a.driver_key == "O")
    k_attr = next(a for a in expl.single_driver_attributions if a.driver_key == "K")

    assert abs(d_attr.delta_R - 0.245) < 1e-4
    assert abs(b_attr.delta_R - 0.195) < 1e-4
    assert abs(o_attr.delta_R - 0.030) < 1e-4
    assert abs(k_attr.delta_R - 0.0075) < 1e-4

    # D has largest delta -> Top driver must be Density
    assert expl.top_driver == "Density"
    assert expl.top_driver_key == "D"
    assert expl.single_driver_attributions[0].driver_key == "D"
    assert expl.single_driver_attributions[1].driver_key == "B"


def test_exhaustive_subsets_and_minimal_intervention():
    engine = CRDAEngine(high_risk_threshold=0.60)

    # High bottleneck cell
    cell = SpatialCellFeatureVector(
        cell_id="C5",
        row=1,
        col=0,
        x1=0,
        y1=120,
        x2=160,
        y2=240,
        area_px=19200.0,
        D=0.65,
        O=0.10,
        B=0.85,
        K=0.10
    )
    calm_ref = np.array([0.10, 0.00, 0.00, 0.00], dtype=np.float32)

    expl = engine.explain_cell(cell, calm_ref, alpha=1.0, target_safe_threshold=0.50)

    # Must contain exactly 15 non-empty subsets (2^4 - 1)
    assert len(expl.all_subset_interventions) == 15

    # Check distribution of subset sizes
    sizes = [s.cardinality for s in expl.all_subset_interventions]
    assert sizes.count(1) == 4
    assert sizes.count(2) == 6
    assert sizes.count(3) == 4
    assert sizes.count(4) == 1

    # Check minimal intervention set
    min_set = expl.minimal_intervention_subset
    assert min_set is not None
    assert min_set.achieved_safe_threshold is True
    assert min_set.counterfactual_risk < 0.50
    # Must choose smallest possible number of drivers
    assert min_set.cardinality <= 2


def test_perturbation_alpha_monotonicity():
    engine = CRDAEngine()

    cell = SpatialCellFeatureVector(
        cell_id="C1",
        row=0,
        col=0,
        x1=0,
        y1=0,
        x2=160,
        y2=120,
        area_px=19200.0,
        D=0.80,
        O=0.60,
        B=0.75,
        K=0.40
    )
    calm_ref = np.array([0.10, 0.05, 0.05, 0.05], dtype=np.float32)

    expl_25 = engine.explain_cell(cell, calm_ref, alpha=0.25)
    expl_50 = engine.explain_cell(cell, calm_ref, alpha=0.50)
    expl_100 = engine.explain_cell(cell, calm_ref, alpha=1.00)

    # Higher alpha moves closer to calm reference -> risk reduction must be monotonically non-decreasing
    assert expl_25.top_driver_delta_R <= expl_50.top_driver_delta_R + 1e-6
    assert expl_50.top_driver_delta_R <= expl_100.top_driver_delta_R + 1e-6


def test_calm_and_empty_cell():
    engine = CRDAEngine()

    empty_cell = SpatialCellFeatureVector(
        cell_id="C16",
        row=3,
        col=3,
        x1=480,
        y1=360,
        x2=640,
        y2=480,
        area_px=19200.0,
        D=0.0,
        O=0.0,
        B=0.0,
        K=0.0
    )
    calm_ref = np.zeros(4, dtype=np.float32)

    expl = engine.explain_cell(empty_cell, calm_ref)
    assert expl.initial_risk == 0.0
    assert expl.initial_threat_level == "LOW"
    assert expl.top_driver_delta_R == 0.0
    for a in expl.single_driver_attributions:
        assert a.delta_R == 0.0


def test_narrative_explanation_content():
    engine = CRDAEngine()

    cell = SpatialCellFeatureVector(
        cell_id="C7",
        row=1,
        col=2,
        x1=320,
        y1=120,
        x2=480,
        y2=240,
        area_px=19200.0,
        D=0.75,
        O=0.80,
        B=0.70,
        K=0.30
    )
    calm_ref = np.array([0.10, 0.05, 0.05, 0.05], dtype=np.float32)

    expl = engine.explain_cell(cell, calm_ref)
    narrative = expl.narrative_explanation

    assert "ZONE C7" in narrative
    assert "Risk:" in narrative
    assert "Primary risk driver is" in narrative
    assert "Minimal intervention:" in narrative


def test_grid_crda_report_generation():
    engine = CRDAEngine(moderate_risk_threshold=0.30)
    calm_mgr = SameSceneCalmReferenceManager(rows=2, cols=2)
    extractor = SpatialCellFeatureExtractor(grid_rows=2, grid_cols=2)
    traj = TrajectoryManager()

    # Create synthetic grid with 1 high risk cell
    grid_map = extractor.extract_grid(
        frame_idx=10,
        timestamp=0.33,
        frame_width=400,
        frame_height=400,
        active_tracks=[],
        trajectory_mgr=traj
    )
    # Manually elevate cell C2
    c2 = grid_map.get_cell("C2")
    c2.D = 0.85
    c2.B = 0.75
    c2.person_count = 12

    report = engine.explain_grid(grid_map, calm_mgr, only_elevated=True)

    assert report.elevated_cells_count >= 1
    assert report.most_critical_cell is not None
    assert report.most_critical_cell.cell_id == "C2"
    assert report.most_critical_cell.initial_risk > 0.50
    dict_rep = report.to_dict()
    assert "cell_explanations" in dict_rep
    assert dict_rep["most_critical_cell_id"] == "C2"


def test_no_subset_reaches_safe_threshold():
    """
    Test scenario where even full 4-driver intervention fails to cross R_safe.
    Confirms that engine falls back to selecting the maximal risk-reduction subset
    with achieved_safe_threshold = False.
    """
    engine = CRDAEngine(high_risk_threshold=0.60, moderate_risk_threshold=0.30)

    # Extremely high risk cell
    cell = SpatialCellFeatureVector(
        cell_id="C4",
        row=0,
        col=3,
        x1=480,
        y1=0,
        x2=640,
        y2=120,
        area_px=19200.0,
        D=1.0,
        O=1.0,
        B=1.0,
        K=1.0
    )
    # Calm reference is also moderately high (cannot reduce risk below 0.20)
    calm_ref = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    # Set an unreachable target safe threshold of 0.20
    expl = engine.explain_cell(cell, calm_ref, alpha=1.0, target_safe_threshold=0.20)

    # No subset reaches < 0.20
    min_subset = expl.minimal_intervention_subset
    assert min_subset.achieved_safe_threshold is False
    assert min_subset.counterfactual_risk >= 0.20
    # Must choose the subset that produces the maximum delta_R
    max_delta = max(s.delta_R for s in expl.all_subset_interventions)
    assert min_subset.delta_R == max_delta


def test_moderate_to_moderate_transition_labeling():
    """
    Confirm that a risk reduction from 0.4174 -> 0.3351 remains correctly labeled MODERATE -> MODERATE.
    """
    engine = CRDAEngine(moderate_risk_threshold=0.30, high_risk_threshold=0.60)
    assert engine._classify_threat_level(0.4174) == "MODERATE"
    assert engine._classify_threat_level(0.3351) == "MODERATE"

