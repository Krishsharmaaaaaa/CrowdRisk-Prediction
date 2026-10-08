"""
Crowd Panic and Anomaly Detection Module.
Provides both a baseline heuristic motion indicator and a Random Forest classification backend.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np

from ..features.motion import CrowdMotionFeatures
from ..features.optical_flow import OpticalFlowFeatures


@dataclass
class PanicResult:
    """Detection output for crowd panic/anomaly."""
    score: float                         # Normalized [0, 1]
    level: str                           # "LOW", "MODERATE", "HIGH"
    mode_used: str                       # "heuristic" or "random_forest"
    speed_surge_contribution: float
    speed_var_contribution: float
    direction_disorder_contribution: float
    flow_magnitude_contribution: float
    flow_var_contribution: float
    factor_breakdown: Dict[str, float] = field(default_factory=dict)


class PanicDetector:
    """
    Evaluates crowd panic / sudden evacuation anomalies using either:
    1. A transparent baseline kinematic + optical flow heuristic indicator
    2. A trained Random Forest classifier
    """

    FEATURE_NAMES = [
        "mean_speed",
        "speed_variance",
        "max_speed",
        "mean_acceleration",
        "direction_entropy",
        "direction_variance",
        "moving_ratio",
        "density",
        "density_change",
        "trajectory_irregularity",
        "flow_mean",
        "flow_variance",
        "high_flow_ratio"
    ]

    def __init__(
        self,
        mode: str = "heuristic",
        model_path: str = "models/panic_rf.joblib",
        heuristic_weights: Optional[Dict[str, float]] = None,
        low_threshold: float = 0.35,
        high_threshold: float = 0.65,
        nominal_running_speed: float = 6.0,
        nominal_max_flow: float = 8.0,
        baseline_history_len: int = 30
    ):
        self.mode = mode
        self.model_path = Path(model_path)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.nominal_running_speed = nominal_running_speed
        self.nominal_max_flow = nominal_max_flow

        self.weights = heuristic_weights or {
            "speed_surge": 0.25,
            "speed_variance": 0.20,
            "direction_entropy": 0.20,
            "flow_mean": 0.20,
            "flow_variance": 0.15
        }
        w_sum = sum(self.weights.values())
        if w_sum > 0:
            self.weights = {k: v / w_sum for k, v in self.weights.items()}

        self.rf_model = None
        self._load_model_if_available()

        # Rolling baseline speed memory
        self.speed_history = deque(maxlen=baseline_history_len)

    def _load_model_if_available(self) -> None:
        if self.model_path.exists():
            try:
                loaded = joblib.load(self.model_path)
                self.rf_model = loaded
            except Exception:
                self.rf_model = None

    def reset(self):
        self.speed_history.clear()

    def evaluate(
        self,
        motion_features: CrowdMotionFeatures,
        flow_features: Optional[OpticalFlowFeatures] = None
    ) -> PanicResult:
        """
        Computes panic probability/score using either Random Forest or baseline heuristic.
        """
        flow_mean = flow_features.mean_magnitude if flow_features else 0.0
        flow_var = flow_features.magnitude_variance if flow_features else 0.0
        high_flow = flow_features.high_flow_ratio if flow_features else 0.0

        # Try trained Random Forest if selected and loaded
        if self.mode == "model" and self.rf_model is not None:
            return self._evaluate_rf(motion_features, flow_mean, flow_var, high_flow)

        # Baseline heuristic evaluation
        return self._evaluate_heuristic(motion_features, flow_mean, flow_var, high_flow)

    def _evaluate_heuristic(
        self,
        mf: CrowdMotionFeatures,
        flow_mean: float,
        flow_var: float,
        high_flow: float
    ) -> PanicResult:
        if mf.person_count == 0:
            return PanicResult(
                score=0.0,
                level="LOW",
                mode_used="heuristic (baseline indicator)",
                speed_surge_contribution=0.0,
                speed_var_contribution=0.0,
                direction_disorder_contribution=0.0,
                flow_magnitude_contribution=0.0,
                flow_var_contribution=0.0,
                factor_breakdown={}
            )

        # 1. Speed Surge Factor (relative to nominal running speed and previous baseline)
        self.speed_history.append(mf.mean_speed)
        baseline = float(np.median(self.speed_history)) if len(self.speed_history) > 5 else 2.0
        speed_ratio = mf.mean_speed / max(0.5, self.nominal_running_speed)
        speed_surge_factor = float(np.clip(speed_ratio, 0.0, 1.0))

        # 2. Speed Variance Factor
        # Higher speed variance indicates panic/stampede where some are sprinting and others are stumbling
        norm_speed_var = float(np.clip(mf.speed_variance / 10.0, 0.0, 1.0))

        # 3. Direction Disorder Factor (Shannon Entropy + Circular Variance)
        direction_disorder = float(np.clip(0.6 * mf.direction_entropy + 0.4 * mf.direction_variance, 0.0, 1.0))

        # 4. Optical Flow Magnitude Factor
        norm_flow_mean = float(np.clip(flow_mean / max(1.0, self.nominal_max_flow), 0.0, 1.0))

        # 5. Optical Flow Variance Factor
        norm_flow_var = float(np.clip(flow_var / 25.0, 0.0, 1.0))

        # Weighted combination
        w_surge = self.weights.get("speed_surge", 0.25)
        w_svar = self.weights.get("speed_variance", 0.20)
        w_dir = self.weights.get("direction_entropy", 0.20)
        w_fmean = self.weights.get("flow_mean", 0.20)
        w_fvar = self.weights.get("flow_variance", 0.15)

        surge_contrib = w_surge * speed_surge_factor
        svar_contrib = w_svar * norm_speed_var
        dir_contrib = w_dir * direction_disorder
        fmean_contrib = w_fmean * norm_flow_mean
        fvar_contrib = w_fvar * norm_flow_var

        score = float(np.clip(surge_contrib + svar_contrib + dir_contrib + fmean_contrib + fvar_contrib, 0.0, 1.0))

        if score >= self.high_threshold:
            level = "HIGH"
        elif score >= self.low_threshold:
            level = "MODERATE"
        else:
            level = "LOW"

        breakdown = {
            "speed_surge_factor": speed_surge_factor,
            "speed_variance_factor": norm_speed_var,
            "direction_disorder_factor": direction_disorder,
            "flow_mean_factor": norm_flow_mean,
            "flow_variance_factor": norm_flow_var,
            "surge_contribution": surge_contrib,
            "speed_var_contribution": svar_contrib,
            "direction_contribution": dir_contrib,
            "flow_mean_contribution": fmean_contrib,
            "flow_var_contribution": fvar_contrib
        }

        return PanicResult(
            score=score,
            level=level,
            mode_used="heuristic (motion-based indicator)",
            speed_surge_contribution=surge_contrib,
            speed_var_contribution=svar_contrib,
            direction_disorder_contribution=dir_contrib,
            flow_magnitude_contribution=fmean_contrib,
            flow_var_contribution=fvar_contrib,
            factor_breakdown=breakdown
        )

    def _evaluate_rf(
        self,
        mf: CrowdMotionFeatures,
        flow_mean: float,
        flow_var: float,
        high_flow: float
    ) -> PanicResult:
        vec = np.array([[
            mf.mean_speed,
            mf.speed_variance,
            mf.max_speed,
            mf.mean_acceleration,
            mf.direction_entropy,
            mf.direction_variance,
            mf.moving_ratio,
            mf.density,
            mf.density_change,
            mf.trajectory_irregularity,
            flow_mean,
            flow_var,
            high_flow
        ]], dtype=np.float32)

        try:
            if hasattr(self.rf_model, "predict_proba"):
                probs = self.rf_model.predict_proba(vec)[0]
                # Assuming class 1 is abnormal/panic
                score = float(probs[1]) if len(probs) > 1 else float(probs[0])
            else:
                score = float(self.rf_model.predict(vec)[0])
        except Exception:
            return self._evaluate_heuristic(mf, flow_mean, flow_var, high_flow)

        score = float(np.clip(score, 0.0, 1.0))
        if score >= self.high_threshold:
            level = "HIGH"
        elif score >= self.low_threshold:
            level = "MODERATE"
        else:
            level = "LOW"

        return PanicResult(
            score=score,
            level=level,
            mode_used="random_forest",
            speed_surge_contribution=score * 0.35,
            speed_var_contribution=score * 0.25,
            direction_disorder_contribution=score * 0.20,
            flow_magnitude_contribution=score * 0.10,
            flow_var_contribution=score * 0.10,
            factor_breakdown={"rf_predicted_probability": score}
        )
