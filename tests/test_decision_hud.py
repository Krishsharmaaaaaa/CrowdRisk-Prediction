"""
Unit tests for Stage 9 Decision-Support HUD Visualization Engine.
Verifies:
1. Correct Model Risk display and threat level classification
2. Correct D/O/B/K driver delta values and ranking
3. Correct minimal intervention subset and safe-threshold flag
4. Correct 'NO SAFE-THRESHOLD INTERVENTION FOUND' state when safe threshold is unreachable
5. Correct cell selection (default most critical vs. explicit cell ID)
6. Deterministic pixel and telemetry output
"""

import numpy as np
import pytest

from src.explainability.crda_engine import (
    CRDAEngine,
    CellCRDAExplanation,
    DriverAttribution,
    GridCRDAReport,
    SubsetIntervention,
)
from src.features.cell_grid import (
    SameSceneCalmReferenceManager,
    SpatialCellFeatureExtractor,
    SpatialCellFeatureVector,
    SpatialGridFeatureMap,
)
from src.risk.risk_fusion import RiskAssessment
from src.visualization.decision_hud import DecisionSupportHUD, HUDTelemetryFrame
from src.warning.early_warning import EarlyWarningStatus


def create_mock_grid(w: int = 640, h: int = 480) -> SpatialGridFeatureMap:
    """Helper to generate a clean 4x4 spatial grid."""
    extractor = SpatialCellFeatureExtractor(grid_rows=4, grid_cols=4)
    return extractor.extract_grid(
        frame_idx=10,
        timestamp=0.33,
        frame_width=w,
        frame_height=h,
        active_tracks=[],
        trajectory_mgr=None
    )


def test_hud_risk_display_and_labeling():
    """Verifies global MODEL RISK score, threat level, and non-probabilistic labeling."""
    hud = DecisionSupportHUD(high_risk_threshold=0.60, moderate_risk_threshold=0.30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    risk_ass = RiskAssessment(
        overall_risk=0.724,
        risk_level="HIGH",
        panic_score=0.65,
        bottleneck_score=0.60,
        density_score=0.85,
        panic_weighted_contrib=0.30,
        bottleneck_weighted_contrib=0.25,
        density_weighted_contrib=0.174
    )
    warn_stat = EarlyWarningStatus(
        is_warning_active=True,
        status_text="WARNING",
        consecutive_high_risk_frames=5,
        risk_trend="RISING",
        trend_slope=0.03,
        first_trigger_frame=35,
        first_trigger_timestamp=1.16,
        lead_time_seconds=None
    )

    out, telemetry = hud.render(
        frame=frame,
        frame_idx=42,
        timestamp=1.40,
        risk_assessment=risk_ass,
        warning_status=warn_stat
    )

    assert isinstance(telemetry, HUDTelemetryFrame)
    assert telemetry.model_risk == 0.724
    assert telemetry.threat_level == "HIGH"
    assert telemetry.warning_status == "WARNING"
    assert telemetry.risk_trend == "RISING"
    assert telemetry.frame_idx == 42
    assert telemetry.timestamp == 1.40

    # Ensure output canvas is modified and non-empty
    assert out.shape == (480, 640, 3)
    assert not np.array_equal(out, frame)


def test_hud_driver_ranking_and_deltas():
    """Verifies exact D, O, B, K delta_R attributions and strict descending ranking."""
    hud = DecisionSupportHUD()
    engine = CRDAEngine(high_risk_threshold=0.60, moderate_risk_threshold=0.30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Cell with Density as dominant driver
    mock_cell = SpatialCellFeatureVector(
        cell_id="C5", row=1, col=0, x1=0, y1=120, x2=160, y2=240, area_px=19200,
        D=0.88, O=0.20, B=0.15, K=0.25
    )
    calm_ref = np.array([0.15, 0.10, 0.08, 0.12], dtype=np.float32)

    expl = engine.explain_cell(mock_cell, calm_ref, alpha=1.0)
    report = GridCRDAReport(
        frame_idx=50, timestamp=1.67, high_risk_threshold=0.60,
        elevated_cells_count=1, cell_explanations=[expl], most_critical_cell=expl
    )

    out, telemetry = hud.render(
        frame=frame,
        frame_idx=50,
        timestamp=1.67,
        crda_report=report
    )

    assert telemetry.selected_cell_id == "C5"
    assert telemetry.top_driver == "Density"
    assert "D" in telemetry.driver_deltas
    assert "O" in telemetry.driver_deltas
    assert "B" in telemetry.driver_deltas
    assert "K" in telemetry.driver_deltas

    # Verify descending delta ranking
    d_vals = [telemetry.driver_deltas[k] for k in telemetry.driver_rankings]
    assert d_vals == sorted(d_vals, reverse=True)
    assert telemetry.driver_rankings[0] == "D"


def test_hud_safe_threshold_reached():
    """Verifies that when a minimal subset crosses R_safe, achieved_safe is True."""
    hud = DecisionSupportHUD(high_risk_threshold=0.60)
    engine = CRDAEngine(high_risk_threshold=0.60)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Moderate risk cell where reducing Density reaches safe threshold (< 0.60)
    mock_cell = SpatialCellFeatureVector(
        cell_id="C6", row=1, col=1, x1=160, y1=120, x2=320, y2=240, area_px=19200,
        D=0.85, O=0.22, B=0.18, K=0.26
    )
    calm_ref = np.array([0.10, 0.08, 0.05, 0.10], dtype=np.float32)
    expl = engine.explain_cell(mock_cell, calm_ref, alpha=1.0, target_safe_threshold=0.60)

    report = GridCRDAReport(
        frame_idx=60, timestamp=2.0, high_risk_threshold=0.60,
        elevated_cells_count=1, cell_explanations=[expl], most_critical_cell=expl
    )

    out, telemetry = hud.render(
        frame=frame,
        frame_idx=60,
        timestamp=2.0,
        crda_report=report
    )

    assert telemetry.safe_threshold_reached is True
    assert telemetry.predicted_cf_risk < 0.60
    assert telemetry.predicted_delta_R > 0.0
    assert "Density" in telemetry.minimal_intervention_subset


def test_hud_no_safe_intervention_found_state():
    """
    Verifies that when NO subset reaches the target safe threshold:
    - safe_threshold_reached is False
    - HUD shows fallback maximum delta_R subset
    - Does NOT falsely claim safe threshold was reached.
    """
    # Strict target safe threshold that cannot be achieved even with all 4 drivers
    strict_thresh = 0.01
    hud = DecisionSupportHUD(high_risk_threshold=strict_thresh)
    engine = CRDAEngine(high_risk_threshold=strict_thresh)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_cell = SpatialCellFeatureVector(
        cell_id="C10", row=2, col=1, x1=160, y1=240, x2=320, y2=360, area_px=19200,
        D=0.95, O=0.80, B=0.85, K=0.75
    )
    calm_ref = np.array([0.30, 0.25, 0.25, 0.20], dtype=np.float32)

    expl = engine.explain_cell(mock_cell, calm_ref, alpha=1.0, target_safe_threshold=strict_thresh)
    assert expl.minimal_intervention_subset.achieved_safe_threshold is False

    report = GridCRDAReport(
        frame_idx=80, timestamp=2.67, high_risk_threshold=strict_thresh,
        elevated_cells_count=1, cell_explanations=[expl], most_critical_cell=expl
    )

    out, telemetry = hud.render(
        frame=frame,
        frame_idx=80,
        timestamp=2.67,
        crda_report=report
    )

    assert telemetry.safe_threshold_reached is False
    assert telemetry.predicted_cf_risk >= strict_thresh
    assert telemetry.safe_threshold_val == strict_thresh


def test_hud_cell_selection_and_override():
    """Verifies target cell selection defaults to most critical, and supports explicit override."""
    hud = DecisionSupportHUD()
    engine = CRDAEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    calm_ref = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    cell_a = SpatialCellFeatureVector("C1", 0, 0, 0, 0, 100, 100, 10000, D=0.4, O=0.2, B=0.2, K=0.2)
    cell_b = SpatialCellFeatureVector("C2", 0, 1, 100, 0, 200, 100, 10000, D=0.9, O=0.4, B=0.3, K=0.3)

    expl_a = engine.explain_cell(cell_a, calm_ref)
    expl_b = engine.explain_cell(cell_b, calm_ref)

    report = GridCRDAReport(
        frame_idx=10, timestamp=0.33, high_risk_threshold=0.60,
        elevated_cells_count=2, cell_explanations=[expl_a, expl_b], most_critical_cell=expl_b
    )

    # Default selection: should pick C2 (most critical)
    _, tel_default = hud.render(frame=frame, frame_idx=10, timestamp=0.33, crda_report=report)
    assert tel_default.selected_cell_id == "C2"

    # Explicit override: should pick C1
    _, tel_override = hud.render(frame=frame, frame_idx=10, timestamp=0.33, crda_report=report, selected_cell_id="C1")
    assert tel_override.selected_cell_id == "C1"


def test_hud_deterministic_output():
    """Verifies that running HUD twice on identical inputs produces identical pixels and telemetry."""
    hud = DecisionSupportHUD()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    risk_ass = RiskAssessment(
        overall_risk=0.55,
        risk_level="MODERATE",
        panic_score=0.45,
        bottleneck_score=0.50,
        density_score=0.60,
        panic_weighted_contrib=0.20,
        bottleneck_weighted_contrib=0.20,
        density_weighted_contrib=0.15
    )
    warn_stat = EarlyWarningStatus(
        is_warning_active=False,
        status_text="NORMAL",
        consecutive_high_risk_frames=0,
        risk_trend="STABLE",
        trend_slope=0.0,
        first_trigger_frame=None,
        first_trigger_timestamp=None,
        lead_time_seconds=None
    )

    out1, tel1 = hud.render(frame=frame, frame_idx=100, timestamp=3.33, risk_assessment=risk_ass, warning_status=warn_stat)
    out2, tel2 = hud.render(frame=frame, frame_idx=100, timestamp=3.33, risk_assessment=risk_ass, warning_status=warn_stat)

    assert np.array_equal(out1, out2)
    assert tel1.to_dict() == tel2.to_dict()
