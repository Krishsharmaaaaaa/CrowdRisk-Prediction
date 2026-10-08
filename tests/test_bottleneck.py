"""
Unit tests for Bottleneck and Congestion Detection.
"""

import pytest

from src.bottleneck.bottleneck_detector import BottleneckDetector
from src.features.density import DensityGrid, GridCellInfo
from src.features.motion import CrowdMotionFeatures


def test_bottleneck_detection_logic():
    detector = BottleneckDetector(
        weights={"density": 0.30, "speed_reduction": 0.30, "flow_imbalance": 0.20, "irregularity": 0.20},
        low_threshold=0.40,
        critical_threshold=0.70
    )

    # Simulate free-flowing crowd
    mf_normal = CrowdMotionFeatures(
        timestamp=1.0, frame=30, person_count=10,
        mean_speed=3.5, speed_variance=0.2, max_speed=4.0, mean_acceleration=0.05,
        direction_entropy=0.1, direction_variance=0.05, moving_ratio=0.9,
        density=0.2, density_change=0.0, trajectory_irregularity=1.05
    )
    cell_normal = GridCellInfo(
        row=0, col=0, x1=0, y1=0, x2=100, y2=100,
        person_count=3, normalized_density=0.2, average_speed=3.5,
        mean_dx=3.0, mean_dy=0.0, track_ids=[1, 2, 3]
    )
    grid_normal = DensityGrid(rows=1, cols=1, max_cell_density=0.2, mean_grid_density=0.2, cells=[cell_normal])

    res_normal = detector.evaluate(mf_normal, grid_normal)
    assert res_normal.level == "LOW"
    assert res_normal.score < 0.40

    # Simulate severe choke point: high local density, speed dropping to near 0, chaotic directions
    mf_bottleneck = CrowdMotionFeatures(
        timestamp=5.0, frame=150, person_count=35,
        mean_speed=0.2, speed_variance=0.1, max_speed=0.5, mean_acceleration=0.01,
        direction_entropy=0.85, direction_variance=0.80, moving_ratio=0.1,
        density=0.95, density_change=0.4, trajectory_irregularity=2.8
    )
    cell_bottleneck = GridCellInfo(
        row=0, col=0, x1=0, y1=0, x2=100, y2=100,
        person_count=15, normalized_density=1.0, average_speed=0.2,
        mean_dx=0.0, mean_dy=0.0, track_ids=[i for i in range(15)]
    )
    grid_bottleneck = DensityGrid(
        rows=1, cols=1, max_cell_density=1.0, mean_grid_density=0.95, cells=[cell_bottleneck]
    )

    res_bottleneck = detector.evaluate(mf_bottleneck, grid_bottleneck)
    assert res_bottleneck.level in ["WARNING", "CRITICAL"]
    assert res_bottleneck.score >= 0.50
    assert len(res_bottleneck.active_zones) >= 1
