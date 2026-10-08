"""
Risk Fusion Module.
Synthesizes heterogeneous risk indicators (panic, bottleneck, density) into an explainable unified threat score.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional
import numpy as np

from ..bottleneck.bottleneck_detector import BottleneckResult
from ..features.motion import CrowdMotionFeatures
from ..panic.panic_detector import PanicResult


@dataclass
class RiskAssessment:
    """Unified multi-dimensional risk evaluation."""
    overall_risk: float                  # Unified score [0, 1]
    risk_level: str                      # "LOW", "MODERATE", "HIGH", "CRITICAL"
    panic_score: float                   # [0, 1]
    bottleneck_score: float              # [0, 1]
    density_score: float                 # [0, 1]
    panic_weighted_contrib: float
    bottleneck_weighted_contrib: float
    density_weighted_contrib: float
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {
            "overall_risk": self.overall_risk,
            "risk_level": self.risk_level,
            "panic_score": self.panic_score,
            "bottleneck_score": self.bottleneck_score,
            "density_score": self.density_score,
            "panic_weighted_contrib": self.panic_weighted_contrib,
            "bottleneck_weighted_contrib": self.bottleneck_weighted_contrib,
            "density_weighted_contrib": self.density_weighted_contrib,
        }


class RiskFusionEngine:
    """
    Combines Panic, Bottleneck, and Crowd Density scores using transparent prototype weighting.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        low_cutoff: float = 0.30,
        moderate_cutoff: float = 0.60,
        high_cutoff: float = 0.80
    ):
        self.weights = weights or {
            "panic": 0.45,
            "bottleneck": 0.40,
            "density": 0.15
        }
        # Normalize weights so sum is 1.0
        w_sum = sum(self.weights.values())
        if w_sum > 0:
            self.weights = {k: v / w_sum for k, v in self.weights.items()}

        self.low_cutoff = low_cutoff
        self.moderate_cutoff = moderate_cutoff
        self.high_cutoff = high_cutoff

    def compute_risk(
        self,
        panic_result: PanicResult,
        bottleneck_result: BottleneckResult,
        motion_features: CrowdMotionFeatures
    ) -> RiskAssessment:
        """
        Calculates unified risk score and categorizes threat level.
        """
        p_score = float(np.clip(panic_result.score, 0.0, 1.0))
        b_score = float(np.clip(bottleneck_result.score, 0.0, 1.0))
        d_score = float(np.clip(motion_features.density, 0.0, 1.0))

        w_panic = self.weights.get("panic", 0.45)
        w_bottle = self.weights.get("bottleneck", 0.40)
        w_dens = self.weights.get("density", 0.15)

        p_contrib = w_panic * p_score
        b_contrib = w_bottle * b_score
        d_contrib = w_dens * d_score

        overall_risk = float(np.clip(p_contrib + b_contrib + d_contrib, 0.0, 1.0))

        # Categorize Risk Level
        if overall_risk >= self.high_cutoff:
            level = "CRITICAL"
        elif overall_risk >= self.moderate_cutoff:
            level = "HIGH"
        elif overall_risk >= self.low_cutoff:
            level = "MODERATE"
        else:
            level = "LOW"

        details = {
            "w_panic": w_panic,
            "w_bottleneck": w_bottle,
            "w_density": w_dens,
            "panic_level": panic_result.level,
            "bottleneck_level": bottleneck_result.level
        }

        return RiskAssessment(
            overall_risk=overall_risk,
            risk_level=level,
            panic_score=p_score,
            bottleneck_score=b_score,
            density_score=d_score,
            panic_weighted_contrib=p_contrib,
            bottleneck_weighted_contrib=b_contrib,
            density_weighted_contrib=d_contrib,
            details=details
        )
