"""
Unit tests for Spatial and Global Crowd Density Estimation.
"""

import numpy as np
import pytest

from src.features.density import SpatialDensityEstimator
from src.features.motion import MotionFeatureExtractor
from src.tracking.bytetrack_tracker import Track
from src.tracking.trajectory import TrajectoryManager


def test_spatial_density_grid():
    """Verifies local grid cell allocation and local capacity normalization."""
    estimator = SpatialDensityEstimator(grid_rows=2, grid_cols=2, saturation_capacity=5)
    traj_mgr = TrajectoryManager()

    # Frame 400x400: Cell (0,0) is [0..200, 0..200]
    tracks = []
    for i in range(5):
        # Place 5 people in top-left cell
        t = Track(np.array([50, 50, 90, 110]), 0.9, 0)
        t.is_activated = True
        tracks.append(t)
        traj_mgr.update(t.track_id, 0, (70, 80))

    grid = estimator.compute_grid(
        frame_width=400,
        frame_height=400,
        active_tracks=tracks,
        trajectory_mgr=traj_mgr
    )

    assert grid.rows == 2
    assert grid.cols == 2
    assert len(grid.cells) == 4

    top_left_cell = grid.cells[0]
    assert top_left_cell.row == 0
    assert top_left_cell.col == 0
    assert top_left_cell.person_count == 5
    assert top_left_cell.normalized_density == 1.0  # 5 / 5 capacity

    # Other cells should be 0
    for cell in grid.cells[1:]:
        assert cell.person_count == 0
        assert cell.normalized_density == 0.0


def test_global_area_aware_density():
    """
    Verifies image-area-aware global density scaling:
    1. Empty frame produces 0.0 density
    2. Low density (e.g., 10 people on 1080p) produces low density
    3. Medium crowd (e.g., 60 people on 1080p) does NOT saturate at 1.0 (avoids fixed 40-person cutoff)
    4. Heavy crowd (e.g., 500 people on 1080p) scales up cleanly to 1.0
    5. Bounds remain strictly in [0, 1]
    """
    extractor = MotionFeatureExtractor(reference_density_per_mpx=200.0)
    traj_mgr = TrajectoryManager()

    # 1. Empty Frame
    feat_empty = extractor.extract(
        frame_id=0,
        timestamp=0.0,
        active_tracks=[],
        trajectory_mgr=traj_mgr,
        frame_width=1920,
        frame_height=1080
    )
    assert feat_empty.density == 0.0
    assert feat_empty.density_change == 0.0

    # 2. Low Density (10 people on 1080p ~ 2.07 Mpx -> 10 / 2.07 ~= 4.8 people/Mpx -> 4.8 / 200 ~= 0.024)
    tracks_10 = []
    for i in range(10):
        t = Track(np.array([100 + i * 20, 100, 120 + i * 20, 140]), 0.9, 0)
        t.is_activated = True
        tracks_10.append(t)
        traj_mgr.update(t.track_id, 1, (110 + i * 20, 120))

    feat_low = extractor.extract(
        frame_id=1,
        timestamp=0.033,
        active_tracks=tracks_10,
        trajectory_mgr=traj_mgr,
        frame_width=1920,
        frame_height=1080
    )
    assert 0.0 < feat_low.density < 0.05
    assert 0.0 <= feat_low.density <= 1.0

    # 3. Medium-High Crowd (60 people on 1080p -> 60 / 2.07 ~= 28.9 people/Mpx -> 28.9 / 200 ~= 0.145)
    # Under old logic (nominal_max_crowd = 40), 60 people saturated to 1.0 immediately!
    tracks_60 = []
    for i in range(60):
        t = Track(np.array([50 + (i % 10) * 30, 50 + (i // 10) * 30, 70, 70]), 0.9, 0)
        t.is_activated = True
        tracks_60.append(t)
        traj_mgr.update(t.track_id, 2, (60 + (i % 10) * 30, 60 + (i // 10) * 30))

    feat_med = extractor.extract(
        frame_id=2,
        timestamp=0.066,
        active_tracks=tracks_60,
        trajectory_mgr=traj_mgr,
        frame_width=1920,
        frame_height=1080
    )
    assert 0.10 <= feat_med.density <= 0.20  # Does NOT saturate to 1.0
    assert 0.0 <= feat_med.density <= 1.0

    # 4. Extreme Crush (500 people on 1080p -> 500 / 2.07 ~= 241 people/Mpx -> 241 / 200 = 1.2 -> clips to 1.0)
    tracks_500 = []
    for i in range(500):
        t = Track(np.array([50, 50, 70, 70]), 0.9, 0)
        t.is_activated = True
        tracks_500.append(t)

    feat_extreme = extractor.extract(
        frame_id=3,
        timestamp=0.100,
        active_tracks=tracks_500,
        trajectory_mgr=traj_mgr,
        frame_width=1920,
        frame_height=1080
    )
    assert feat_extreme.density == 1.0
    assert 0.0 <= feat_extreme.density <= 1.0
