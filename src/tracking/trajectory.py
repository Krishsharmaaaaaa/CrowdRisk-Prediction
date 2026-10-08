"""
Trajectory Management and Kinematic Feature Extraction.
Maintains smoothed trajectory history and rolling temporal statistics per tracked person.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


@dataclass
class TrajectoryPoint:
    """Represents a single smoothed point in an individual's trajectory."""
    frame_id: int
    timestamp: float
    x: float
    y: float
    raw_x: float
    raw_y: float
    dx: float = 0.0
    dy: float = 0.0
    distance: float = 0.0
    speed: float = 0.0           # pixels / frame
    direction_rad: float = 0.0   # [-pi, pi]
    direction_deg: float = 0.0   # [0, 360)
    acceleration: float = 0.0    # delta speed / frame


class TrajectoryManager:
    """
    Manages temporal trajectories for all tracked persons.
    Performs Exponential Moving Average (EMA) smoothing and windowed kinematics.
    """

    def __init__(
        self,
        window_size: int = 15,
        min_track_length: int = 5,
        smoothing_factor: float = 0.3
    ):
        self.window_size = window_size
        self.min_track_length = min_track_length
        self.smoothing_factor = smoothing_factor  # alpha for EMA: x_s = alpha * x_raw + (1 - alpha) * x_prev
        self.histories: Dict[int, deque] = {}

    def update(
        self,
        track_id: int,
        frame_id: int,
        raw_center: Tuple[float, float],
        fps: float = 30.0
    ) -> TrajectoryPoint:
        """
        Updates the trajectory history for a given track_id with raw center (cx, cy).
        Returns the computed TrajectoryPoint.
        """
        raw_x, raw_y = float(raw_center[0]), float(raw_center[1])
        timestamp = frame_id / max(1.0, fps)

        if track_id not in self.histories:
            self.histories[track_id] = deque(maxlen=self.window_size)

        history = self.histories[track_id]

        if len(history) == 0:
            smoothed_x = raw_x
            smoothed_y = raw_y
            pt = TrajectoryPoint(
                frame_id=frame_id,
                timestamp=timestamp,
                x=smoothed_x,
                y=smoothed_y,
                raw_x=raw_x,
                raw_y=raw_y,
                dx=0.0,
                dy=0.0,
                distance=0.0,
                speed=0.0,
                direction_rad=0.0,
                direction_deg=0.0,
                acceleration=0.0
            )
        else:
            prev = history[-1]
            # EMA Smoothing
            alpha = self.smoothing_factor
            smoothed_x = alpha * raw_x + (1.0 - alpha) * prev.x
            smoothed_y = alpha * raw_y + (1.0 - alpha) * prev.y

            dx = smoothed_x - prev.x
            dy = smoothed_y - prev.y
            dist = math.hypot(dx, dy)
            speed = dist  # px per frame

            direction_rad = math.atan2(dy, dx)
            direction_deg = (math.degrees(direction_rad) + 360.0) % 360.0

            accel = speed - prev.speed

            pt = TrajectoryPoint(
                frame_id=frame_id,
                timestamp=timestamp,
                x=smoothed_x,
                y=smoothed_y,
                raw_x=raw_x,
                raw_y=raw_y,
                dx=dx,
                dy=dy,
                distance=dist,
                speed=speed,
                direction_rad=direction_rad,
                direction_deg=direction_deg,
                acceleration=accel
            )

        history.append(pt)
        return pt

    def get_trail(self, track_id: int, max_points: Optional[int] = None) -> List[Tuple[int, int]]:
        """Returns list of (x, y) integer pixel coordinates for rendering trails."""
        if track_id not in self.histories:
            return []
        hist = list(self.histories[track_id])
        if max_points is not None and len(hist) > max_points:
            hist = hist[-max_points:]
        return [(int(round(p.x)), int(round(p.y))) for p in hist]

    def get_track_stats(self, track_id: int) -> Optional[Dict[str, float]]:
        """Computes windowed kinematics for a specific track."""
        if track_id not in self.histories:
            return None
        hist = list(self.histories[track_id])
        if len(hist) < self.min_track_length:
            return None

        speeds = [p.speed for p in hist]
        mean_speed = float(np.mean(speeds))
        max_speed = float(np.max(speeds))
        accels = [p.acceleration for p in hist]
        mean_accel = float(np.mean(np.abs(accels)))

        # Trajectory Irregularity: total path length / net Euclidean displacement
        # (Values close to 1 mean straight path; high values mean meandering/chaotic motion)
        total_path = sum(p.distance for p in hist[1:])
        net_dx = hist[-1].x - hist[0].x
        net_dy = hist[-1].y - hist[0].y
        net_displacement = math.hypot(net_dx, net_dy)
        irregularity = float(total_path / max(1.0, net_displacement)) if net_displacement > 0.1 else 1.0

        return {
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "mean_accel": mean_accel,
            "irregularity": irregularity,
            "latest_speed": hist[-1].speed,
            "latest_direction": hist[-1].direction_rad
        }

    def cleanup_old_tracks(self, active_track_ids: List[int], max_idle_frames: int = 60, current_frame: int = 0):
        """Removes track histories that are no longer active to free memory."""
        dead_ids = []
        for tid, hist in self.histories.items():
            if tid not in active_track_ids:
                if len(hist) > 0 and (current_frame - hist[-1].frame_id > max_idle_frames):
                    dead_ids.append(tid)
        for tid in dead_ids:
            del self.histories[tid]
