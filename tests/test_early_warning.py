"""
Unit tests for Early Warning Temporal Threshold Monitoring.
"""

import pytest

from src.warning.early_warning import EarlyWarningSystem


def test_early_warning_warmup_suppression():
    """
    During warmup period (frames < warmup_frames), no warnings should trigger
    even if risk is critically high.
    """
    sys = EarlyWarningSystem(risk_threshold=0.60, sustained_window=5, warmup_frames=20)

    for f in range(20):
        status = sys.update(frame_id=f, timestamp=f * 0.033, current_risk=0.95)
        assert status.is_warning_active is False
        assert status.status_text == "WARMUP"
        assert status.first_trigger_frame is None

    # After warmup finishes, sustained high risk should now evaluate and trigger
    for f in range(20, 25):
        status = sys.update(frame_id=f, timestamp=f * 0.033, current_risk=0.95)

    assert status.is_warning_active is True
    assert status.status_text == "ACTIVE"
    assert status.first_trigger_frame == 24


def test_early_warning_sustained_condition():
    sys = EarlyWarningSystem(risk_threshold=0.60, sustained_window=5, warmup_frames=0)

    # 1. Single spike below sustained window should NOT trigger active warning
    for f in range(4):
        status = sys.update(frame_id=f, timestamp=f * 0.033, current_risk=0.75)
        assert status.is_warning_active is False
        assert status.status_text in ["MONITORING", "NORMAL"]

    # 2. Reaching 5 consecutive frames above threshold MUST trigger active warning
    status = sys.update(frame_id=4, timestamp=4 * 0.033, current_risk=0.75)
    assert status.is_warning_active is True
    assert status.status_text == "ACTIVE"
    assert status.first_trigger_frame == 4


def test_early_warning_trend_detection():
    sys = EarlyWarningSystem(risk_threshold=0.60, sustained_window=10, trend_window=8, warmup_frames=0)

    # Progressively escalating risk
    rising_risks = [0.10, 0.18, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    for f, r in enumerate(rising_risks):
        status = sys.update(frame_id=f, timestamp=f * 0.033, current_risk=r)

    assert status.risk_trend == "RISING"
    assert status.trend_slope > 0.0
