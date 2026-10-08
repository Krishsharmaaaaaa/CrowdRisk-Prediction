"""
Unit tests for Risk Fusion and Threat Categorization.
"""

import pytest

from src.bottleneck.bottleneck_detector import BottleneckResult
from src.features.motion import CrowdMotionFeatures
from src.panic.panic_detector import PanicResult
from src.risk.risk_fusion import RiskFusionEngine


def test_risk_fusion_bounds_and_levels():
    engine = RiskFusionEngine(
        weights={"panic": 0.45, "bottleneck": 0.40, "density": 0.15},
        low_cutoff=0.30,
        moderate_cutoff=0.60,
        high_cutoff=0.80
    )

    mf = CrowdMotionFeatures(
        timestamp=1.0,
        frame=30,
        person_count=10,
        mean_speed=2.0,
        speed_variance=0.5,
        max_speed=3.0,
        mean_acceleration=0.1,
        direction_entropy=0.1,
        direction_variance=0.1,
        moving_ratio=0.8,
        density=0.20,
        density_change=0.0,
        trajectory_irregularity=1.0
    )

    # Low Risk Case
    p_low = PanicResult(
        score=0.1, level="LOW", mode_used="heuristic",
        speed_surge_contribution=0.0, speed_var_contribution=0.0,
        direction_disorder_contribution=0.0, flow_magnitude_contribution=0.0, flow_var_contribution=0.0
    )
    b_low = BottleneckResult(
        score=0.1, level="LOW",
        density_contribution=0.0, speed_reduction_contribution=0.0,
        flow_imbalance_contribution=0.0, irregularity_contribution=0.0
    )
    risk_low = engine.compute_risk(p_low, b_low, mf)
    assert risk_low.risk_level == "LOW"
    assert 0.0 <= risk_low.overall_risk < 0.30

    # Critical Risk Case (Panic = 0.9, Bottleneck = 0.85, Density = 0.90)
    mf.density = 0.90
    p_high = PanicResult(
        score=0.90, level="HIGH", mode_used="heuristic",
        speed_surge_contribution=0.0, speed_var_contribution=0.0,
        direction_disorder_contribution=0.0, flow_magnitude_contribution=0.0, flow_var_contribution=0.0
    )
    b_high = BottleneckResult(
        score=0.85, level="CRITICAL",
        density_contribution=0.0, speed_reduction_contribution=0.0,
        flow_imbalance_contribution=0.0, irregularity_contribution=0.0
    )
    risk_crit = engine.compute_risk(p_high, b_high, mf)
    assert risk_crit.risk_level == "CRITICAL"
    assert risk_crit.overall_risk >= 0.80
    assert risk_crit.overall_risk <= 1.0
