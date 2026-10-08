"""
Unit tests for the Spatially Localized Cell-Grid Feature Architecture.
Tests:
- Grid dimension and cell ID assignment (C1..C16 for 4x4)
- Driver normalization bounds [0.0, 1.0] for D, O, B, K
- Empty cell behavior
- Single-person cell behavior (O=0)
- High directional disorder vs. laminar flow
- Bottleneck jamming condition
- Same-scene calm reference manager
"""

import math
import numpy as np
import pytest

from src.features.cell_grid import (
    SameSceneCalmReferenceManager,
    SpatialCellFeatureExtractor,
    SpatialCellFeatureVector,
    SpatialGridFeatureMap
)
from src.tracking.bytetrack_tracker import Track
from src.tracking.trajectory import TrajectoryManager


def test_grid_initialization_and_cell_count():
    extractor = SpatialCellFeatureExtractor(grid_rows=4, grid_cols=4)
    trajectory_mgr = TrajectoryManager()
    tracks = []

    grid_map = extractor.extract_grid(
        frame_idx=0,
        timestamp=0.0,
        frame_width=640,
        frame_height=480,
        active_tracks=tracks,
        trajectory_mgr=trajectory_mgr
    )

    assert grid_map.rows == 4
    assert grid_map.cols == 4
    assert len(grid_map.cells) == 16
    assert grid_map.cells[0].cell_id == "C1"
    assert grid_map.cells[15].cell_id == "C16"
    assert grid_map.total_tracked_persons == 0


def test_empty_cell_properties():
    extractor = SpatialCellFeatureExtractor(grid_rows=4, grid_cols=4)
    trajectory_mgr = TrajectoryManager()

    grid_map = extractor.extract_grid(
        frame_idx=1,
        timestamp=0.033,
        frame_width=640,
        frame_height=480,
        active_tracks=[],
        trajectory_mgr=trajectory_mgr
    )

    for cell in grid_map.cells:
        assert cell.person_count == 0
        assert cell.D == 0.0
        assert cell.O == 0.0
        assert cell.B == 0.0
        assert cell.K == 0.0
        vec = cell.to_driver_vector()
        assert vec.shape == (4,)
        assert np.all(vec == 0.0)


def test_single_person_cell():
    Track.reset_counter()
    extractor = SpatialCellFeatureExtractor(grid_rows=4, grid_cols=4)
    trajectory_mgr = TrajectoryManager()

    # Track in cell (0, 0) -> C1 (bounds [0..160, 0..120])
    track1 = Track(bbox=np.array([50, 40, 90, 100], dtype=np.float32), score=0.9, frame_id=0)
    for f in range(6):
        trajectory_mgr.update(track1.track_id, f, (50 + f * 3, 40 + f * 3), fps=30.0)

    grid_map = extractor.extract_grid(
        frame_idx=6,
        timestamp=0.2,
        frame_width=640,
        frame_height=480,
        active_tracks=[track1],
        trajectory_mgr=trajectory_mgr
    )

    c1 = grid_map.get_cell("C1")
    assert c1 is not None
    assert c1.person_count == 1
    assert c1.D > 0.0
    # O (directional disorder) must be 0 for single person
    assert c1.O == 0.0

    # Other empty cells must have 0 persons
    c2 = grid_map.get_cell("C2")
    assert c2.person_count == 0
    assert c2.D == 0.0


def test_directional_disorder_contrast():
    Track.reset_counter()
    extractor = SpatialCellFeatureExtractor(grid_rows=2, grid_cols=2)

    # 1. Laminar crowd moving in unison (all moving right +X)
    traj_laminar = TrajectoryManager()
    tracks_laminar = []
    for i in range(4):
        t = Track(bbox=np.array([20, 20 + i * 20, 40, 40 + i * 20], dtype=np.float32), score=0.9, frame_id=0)
        tracks_laminar.append(t)
        for f in range(5):
            traj_laminar.update(t.track_id, f, (20 + f * 4, 30 + i * 20), fps=30.0)

    grid_laminar = extractor.extract_grid(
        frame_idx=5,
        timestamp=0.16,
        frame_width=400,
        frame_height=400,
        active_tracks=tracks_laminar,
        trajectory_mgr=traj_laminar
    )
    c1_laminar = grid_laminar.get_cell("C1")

    # 2. Disordered crowd moving in 4 opposing directions
    traj_disordered = TrajectoryManager()
    tracks_disordered = []
    angles = [0.0, math.pi / 2, math.pi, -math.pi / 2]  # Right, Down, Left, Up
    for i, ang in enumerate(angles):
        t = Track(bbox=np.array([20, 20, 40, 40], dtype=np.float32), score=0.9, frame_id=0)
        tracks_disordered.append(t)
        dx = 4.0 * math.cos(ang)
        dy = 4.0 * math.sin(ang)
        for f in range(5):
            traj_disordered.update(t.track_id, f, (100 + f * dx, 100 + f * dy), fps=30.0)

    grid_disordered = extractor.extract_grid(
        frame_idx=5,
        timestamp=0.16,
        frame_width=400,
        frame_height=400,
        active_tracks=tracks_disordered,
        trajectory_mgr=traj_disordered
    )
    c1_disordered = grid_disordered.get_cell("C1")

    # Disordered cell must have significantly higher O than laminar cell
    assert c1_laminar.O < 0.15
    assert c1_disordered.O > 0.70
    assert c1_disordered.O > c1_laminar.O


def test_bottleneck_jamming_detection():
    Track.reset_counter()
    extractor = SpatialCellFeatureExtractor(grid_rows=2, grid_cols=2, free_flow_speed_px=5.0)
    traj_jam = TrajectoryManager()
    tracks_jam = []

    # 6 people stuck / moving very slowly (< 0.2 px/frame) in cell C1
    for i in range(6):
        t = Track(bbox=np.array([30 + i * 10, 30 + i * 10, 50 + i * 10, 50 + i * 10], dtype=np.float32), score=0.9, frame_id=0)
        tracks_jam.append(t)
        for f in range(5):
            traj_jam.update(t.track_id, f, (40 + i * 10 + np.random.uniform(-0.1, 0.1), 40 + i * 10), fps=30.0)

    grid_jam = extractor.extract_grid(
        frame_idx=5,
        timestamp=0.16,
        frame_width=400,
        frame_height=400,
        active_tracks=tracks_jam,
        trajectory_mgr=traj_jam
    )

    c1 = grid_jam.get_cell("C1")
    assert c1.D > 0.3
    assert c1.stopped_ratio > 0.8
    assert c1.B > 0.2


def test_driver_bounds_are_strictly_unit_interval():
    Track.reset_counter()
    extractor = SpatialCellFeatureExtractor(grid_rows=4, grid_cols=4)
    traj = TrajectoryManager()
    tracks = []

    # Add 25 dense, high speed, colliding tracks in cell C1
    for i in range(25):
        t = Track(bbox=np.array([10, 10, 80, 80], dtype=np.float32), score=0.9, frame_id=0)
        tracks.append(t)
        for f in range(6):
            traj.update(t.track_id, f, (50 + np.random.uniform(-20, 20), 50 + np.random.uniform(-20, 20)), fps=30.0)

    flow_field = np.full((480, 640, 2), 15.0, dtype=np.float32)

    grid_map = extractor.extract_grid(
        frame_idx=6,
        timestamp=0.2,
        frame_width=640,
        frame_height=480,
        active_tracks=tracks,
        trajectory_mgr=traj,
        flow_field=flow_field
    )

    for cell in grid_map.cells:
        assert 0.0 <= cell.D <= 1.0
        assert 0.0 <= cell.O <= 1.0
        assert 0.0 <= cell.B <= 1.0
        assert 0.0 <= cell.K <= 1.0


def test_same_scene_calm_reference_manager():
    ref_mgr = SameSceneCalmReferenceManager(rows=2, cols=2, warmup_frames=10)
    extractor = SpatialCellFeatureExtractor(grid_rows=2, grid_cols=2)
    traj = TrajectoryManager()

    for f in range(15):
        grid_map = extractor.extract_grid(
            frame_idx=f,
            timestamp=f * 0.033,
            frame_width=400,
            frame_height=400,
            active_tracks=[],
            trajectory_mgr=traj
        )
        ref_mgr.update(grid_map, is_calm_window=True)

    assert ref_mgr.frozen_reference is not None
    ref_c1 = ref_mgr.get_reference_vector("C1")
    assert ref_c1.shape == (4,)
    assert np.all(ref_c1 == 0.0)  # Empty baseline is 0.0
