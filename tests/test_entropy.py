"""
Unit tests for Direction Entropy and Circular Statistics.
"""

import math
import numpy as np
import pytest

from src.features.motion import MotionFeatureExtractor
from src.tracking.bytetrack_tracker import Track
from src.tracking.trajectory import TrajectoryManager


def test_direction_entropy_uniform():
    """
    Uniform distribution of angles across all directions should produce maximum normalized entropy (near 1.0).
    """
    extractor = MotionFeatureExtractor(entropy_num_bins=8, moving_speed_thresh=0.1)
    traj_mgr = TrajectoryManager(window_size=10, min_track_length=2)

    # 8 people moving in 8 equidistant directions
    tracks = []
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for i, ang in enumerate(angles):
        t = Track(np.array([100 + i * 10, 100, 120 + i * 10, 140]), 0.9, 0)
        t.is_activated = True
        tracks.append(t)

        # First point
        traj_mgr.update(t.track_id, 0, (100 + i * 10, 100))
        # Second point moving in direction ang
        rad = math.radians(ang)
        dx = 5.0 * math.cos(rad)
        dy = 5.0 * math.sin(rad)
        traj_mgr.update(t.track_id, 1, (100 + i * 10 + dx, 100 + dy))

    feat = extractor.extract(frame_id=1, timestamp=0.033, active_tracks=tracks, trajectory_mgr=traj_mgr)

    # Entropy should be close to 1.0 (uniform spread across 8 bins)
    assert feat.direction_entropy >= 0.90
    assert 0.0 <= feat.direction_entropy <= 1.0
    # Direction variance should also be high
    assert feat.direction_variance >= 0.80


def test_direction_entropy_unidirectional():
    """
    All people moving in the same direction should produce 0.0 entropy and 0.0 circular variance.
    """
    extractor = MotionFeatureExtractor(entropy_num_bins=8, moving_speed_thresh=0.1)
    traj_mgr = TrajectoryManager(window_size=10, min_track_length=2)

    tracks = []
    for i in range(10):
        t = Track(np.array([100 + i * 10, 100, 120 + i * 10, 140]), 0.9, 0)
        t.is_activated = True
        tracks.append(t)

        traj_mgr.update(t.track_id, 0, (100 + i * 10, 100))
        # All moving strictly right (dx = 5, dy = 0)
        traj_mgr.update(t.track_id, 1, (100 + i * 10 + 5.0, 100))

    feat = extractor.extract(frame_id=1, timestamp=0.033, active_tracks=tracks, trajectory_mgr=traj_mgr)

    assert feat.direction_entropy == 0.0
    assert feat.direction_variance < 0.01
