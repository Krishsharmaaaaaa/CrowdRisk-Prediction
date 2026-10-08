"""
Early Warning System.
Monitors temporal risk trajectories over a sliding window to prevent transient false alarms
and trigger reliable evacuation early alerts.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import numpy as np


@dataclass
class EarlyWarningStatus:
    """State of early warning system for current frame."""
    is_warning_active: bool
    status_text: str                     # "NORMAL", "MONITORING", "ACTIVE"
    consecutive_high_risk_frames: int
    risk_trend: str                      # "RISING", "FALLING", "STABLE"
    trend_slope: float                   # Linear regression slope over window
    first_trigger_frame: Optional[int]
    first_trigger_timestamp: Optional[float]
    lead_time_seconds: Optional[float]   # Only if ground-truth onset is known


class EarlyWarningSystem:
    """
    Temporal risk monitor that triggers alerts only on sustained threshold breaches
    or rapid progressive escalation. Includes a warmup period to suppress tracker startup artifacts.
    """

    def __init__(
        self,
        risk_threshold: float = 0.60,
        sustained_window: int = 10,
        trend_window: int = 15,
        trend_slope_thresh: float = 0.015,
        lead_time_fps: float = 30.0,
        ground_truth_onset_frame: Optional[int] = None,
        warmup_frames: int = 20
    ):
        self.risk_threshold = risk_threshold
        self.sustained_window = sustained_window
        self.trend_window = trend_window
        self.trend_slope_thresh = trend_slope_thresh
        self.lead_time_fps = max(1.0, lead_time_fps)
        self.ground_truth_onset_frame = ground_truth_onset_frame
        self.warmup_frames = max(0, warmup_frames)

        self.consecutive_high_frames = 0
        self.risk_history: Deque[float] = deque(maxlen=self.trend_window)
        self.frame_history: Deque[int] = deque(maxlen=self.trend_window)

        self.first_trigger_frame: Optional[int] = None
        self.first_trigger_timestamp: Optional[float] = None
        self.lead_time_seconds: Optional[float] = None

    def reset(self):
        self.consecutive_high_frames = 0
        self.risk_history.clear()
        self.frame_history.clear()
        self.first_trigger_frame = None
        self.first_trigger_timestamp = None
        self.lead_time_seconds = None

    def update(
        self,
        frame_id: int,
        timestamp: float,
        current_risk: float
    ) -> EarlyWarningStatus:
        """
        Updates temporal risk history and evaluates warning criteria.
        Suppresses alert activations during initial warmup frames.
        """
        self.risk_history.append(current_risk)
        self.frame_history.append(frame_id)

        in_warmup = frame_id < self.warmup_frames

        # 1. Check sustained threshold breach (only accumulated after warmup)
        if not in_warmup:
            if current_risk >= self.risk_threshold:
                self.consecutive_high_frames += 1
            else:
                self.consecutive_high_frames = max(0, self.consecutive_high_frames - 1)
        else:
            self.consecutive_high_frames = 0

        # 2. Compute risk trend slope (linear regression over rolling window)
        if len(self.risk_history) >= 5:
            x = np.arange(len(self.risk_history), dtype=np.float32)
            y = np.array(self.risk_history, dtype=np.float32)
            # Linear fit: y = slope * x + intercept
            slope, _ = np.polyfit(x, y, 1)
            slope = float(slope)

            if slope > self.trend_slope_thresh:
                trend = "RISING"
            elif slope < -self.trend_slope_thresh:
                trend = "FALLING"
            else:
                trend = "STABLE"
        else:
            slope = 0.0
            trend = "STABLE"

        # 3. Determine Warning Activation Condition (Active only post-warmup)
        if in_warmup:
            is_active = False
            status_text = "WARMUP"
        else:
            # Condition A: Sustained high risk >= threshold for N consecutive frames
            is_sustained = self.consecutive_high_frames >= self.sustained_window
            # Condition B: High risk (> 0.50) with strong rising trend slope
            is_rapid_escalation = (current_risk >= 0.50) and (trend == "RISING") and (slope >= 0.025)

            is_active = is_sustained or is_rapid_escalation

            if is_active:
                status_text = "ACTIVE"
                if self.first_trigger_frame is None:
                    self.first_trigger_frame = frame_id
                    self.first_trigger_timestamp = timestamp
                    # Compute lead time if ground truth onset was configured
                    if self.ground_truth_onset_frame is not None:
                        diff_frames = self.ground_truth_onset_frame - frame_id
                        self.lead_time_seconds = float(diff_frames / self.lead_time_fps)
            elif self.consecutive_high_frames > 0 or current_risk >= 0.40:
                status_text = "MONITORING"
            else:
                status_text = "NORMAL"

        return EarlyWarningStatus(
            is_warning_active=is_active,
            status_text=status_text,
            consecutive_high_risk_frames=self.consecutive_high_frames,
            risk_trend=trend,
            trend_slope=slope,
            first_trigger_frame=self.first_trigger_frame,
            first_trigger_timestamp=self.first_trigger_timestamp,
            lead_time_seconds=self.lead_time_seconds
        )
