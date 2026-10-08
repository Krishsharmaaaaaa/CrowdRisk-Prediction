"""
Bottleneck and Congestion Detection Module.
Detects physical choke points and localized compression zones using multi-factor fusion.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..features.density import DensityGrid, GridCellInfo
from ..features.motion import CrowdMotionFeatures
from ..features.optical_flow import OpticalFlowFeatures


@dataclass
class BottleneckZone:
    """Bounding coordinates and metrics for a localized bottleneck zone."""
    x1: int
    y1: int
    x2: int
    y2: int
    cell_row: int
    cell_col: int
    score: float
    density: float
    avg_speed: float


@dataclass
class BottleneckResult:
    """Comprehensive bottleneck assessment for the current frame."""
    score: float                         # [0, 1]
    level: str                           # "LOW", "WARNING", "CRITICAL"
    density_contribution: float
    speed_reduction_contribution: float
    flow_imbalance_contribution: float
    irregularity_contribution: float
    active_zones: List[BottleneckZone] = field(default_factory=list)
    factor_breakdown: Dict[str, float] = field(default_factory=dict)


class BottleneckDetector:
    """
    Evaluates crowd bottleneck risk by analyzing localized density accumulation,
    speed deceleration, directional conflict, and trajectory turbulence.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        speed_drop_threshold: float = 0.40,
        low_threshold: float = 0.40,
        critical_threshold: float = 0.70,
        history_len: int = 15,
        nominal_free_speed: float = 4.0
    ):
        self.weights = weights or {
            "density": 0.30,
            "speed_reduction": 0.30,
            "flow_imbalance": 0.20,
            "irregularity": 0.20
        }
        # Normalize weights to sum to 1.0
        w_sum = sum(self.weights.values())
        if w_sum > 0:
            self.weights = {k: v / w_sum for k, v in self.weights.items()}

        self.speed_drop_threshold = speed_drop_threshold
        self.low_threshold = low_threshold
        self.critical_threshold = critical_threshold
        self.nominal_free_speed = nominal_free_speed

        self.speed_history = deque(maxlen=history_len)

    def reset(self):
        self.speed_history.clear()

    def evaluate(
        self,
        motion_features: CrowdMotionFeatures,
        density_grid: DensityGrid,
        flow_features: Optional[OpticalFlowFeatures] = None
    ) -> BottleneckResult:
        """
        Calculates the bottleneck score and factor breakdown.
        """
        # 1. Local Peak Density Factor
        # Look at the highest density cell, as bottlenecks are local
        max_cell_dens = density_grid.max_cell_density
        mean_grid_dens = density_grid.mean_grid_density
        # Blend max cell density with global density
        density_factor = float(np.clip(0.7 * max_cell_dens + 0.3 * mean_grid_dens, 0.0, 1.0))

        # 2. Speed Reduction Factor
        curr_speed = motion_features.mean_speed
        self.speed_history.append(curr_speed)
        baseline_speed = float(np.max(self.speed_history)) if len(self.speed_history) > 0 else self.nominal_free_speed
        baseline_speed = max(self.nominal_free_speed, baseline_speed)

        # Relative speed loss (higher when people slow down despite crowd present)
        if motion_features.person_count > 0:
            speed_loss_ratio = max(0.0, 1.0 - (curr_speed / max(0.1, baseline_speed)))
            speed_reduction_factor = float(np.clip(speed_loss_ratio, 0.0, 1.0))
        else:
            speed_reduction_factor = 0.0

        # 3. Directional / Flow Imbalance Factor
        # High direction variance or high direction entropy indicates turbulence/clash
        flow_imbalance = float(
            np.clip(
                0.5 * motion_features.direction_entropy + 0.5 * motion_features.direction_variance,
                0.0,
                1.0
            )
        )

        # 4. Trajectory Irregularity Factor
        # Normalized irregularity (1.0 = straight line, >= 3.0 = extreme chaos)
        irreg = motion_features.trajectory_irregularity
        irregularity_factor = float(np.clip((irreg - 1.0) / 2.0, 0.0, 1.0))

        # Combine weighted components
        w_dens = self.weights.get("density", 0.30)
        w_speed = self.weights.get("speed_reduction", 0.30)
        w_flow = self.weights.get("flow_imbalance", 0.20)
        w_irreg = self.weights.get("irregularity", 0.20)

        dens_contrib = w_dens * density_factor
        speed_contrib = w_speed * speed_reduction_factor
        flow_contrib = w_flow * flow_imbalance
        irreg_contrib = w_irreg * irregularity_factor

        bottleneck_score = float(np.clip(dens_contrib + speed_contrib + flow_contrib + irreg_contrib, 0.0, 1.0))

        # Determine qualitative level
        if bottleneck_score >= self.critical_threshold:
            level = "CRITICAL"
        elif bottleneck_score >= self.low_threshold:
            level = "WARNING"
        else:
            level = "LOW"

        # Identify specific spatial bottleneck cells
        active_zones: List[BottleneckZone] = []
        for cell in density_grid.cells:
            if cell.person_count >= 2:
                # Cell-specific bottleneck formula: local density and low local speed
                cell_speed_drop = max(0.0, 1.0 - (cell.average_speed / max(0.1, self.nominal_free_speed)))
                cell_score = float(np.clip(0.6 * cell.normalized_density + 0.4 * cell_speed_drop, 0.0, 1.0))
                if cell_score >= self.low_threshold:
                    active_zones.append(
                        BottleneckZone(
                            x1=cell.x1,
                            y1=cell.y1,
                            x2=cell.x2,
                            y2=cell.y2,
                            cell_row=cell.row,
                            cell_col=cell.col,
                            score=cell_score,
                            density=cell.normalized_density,
                            avg_speed=cell.average_speed
                        )
                    )

        factor_breakdown = {
            "density_contribution": dens_contrib,
            "speed_reduction_contribution": speed_contrib,
            "flow_imbalance_contribution": flow_contrib,
            "irregularity_contribution": irreg_contrib,
            "raw_density_factor": density_factor,
            "raw_speed_reduction_factor": speed_reduction_factor,
            "raw_flow_imbalance": flow_imbalance,
            "raw_irregularity": irregularity_factor
        }

        return BottleneckResult(
            score=bottleneck_score,
            level=level,
            density_contribution=dens_contrib,
            speed_reduction_contribution=speed_contrib,
            flow_imbalance_contribution=flow_contrib,
            irregularity_contribution=irreg_contrib,
            active_zones=active_zones,
            factor_breakdown=factor_breakdown
        )
