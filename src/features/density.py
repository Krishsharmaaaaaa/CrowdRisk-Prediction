"""
Spatial Crowd Density Estimator.
Discretizes the video frame into a multi-cell grid to detect localized crowd pressure.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from ..tracking.bytetrack_tracker import Track
from ..tracking.trajectory import TrajectoryManager


@dataclass
class GridCellInfo:
    """Statistics for an individual spatial grid cell."""
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int
    person_count: int
    normalized_density: float   # [0, 1]
    average_speed: float        # px / frame
    mean_dx: float
    mean_dy: float
    track_ids: List[int]


@dataclass
class DensityGrid:
    """Overall spatial grid state for a frame."""
    rows: int
    cols: int
    max_cell_density: float
    mean_grid_density: float
    cells: List[GridCellInfo]


class SpatialDensityEstimator:
    """Computes grid-based crowd density and cell-level kinematics."""

    def __init__(
        self,
        grid_rows: int = 4,
        grid_cols: int = 4,
        saturation_capacity: int = 15
    ):
        self.rows = max(1, grid_rows)
        self.cols = max(1, grid_cols)
        self.capacity = max(1, saturation_capacity)

    def compute_grid(
        self,
        frame_width: int,
        frame_height: int,
        active_tracks: List[Track],
        trajectory_mgr: TrajectoryManager
    ) -> DensityGrid:
        cell_w = frame_width / self.cols
        cell_h = frame_height / self.rows

        # Initialize cells
        cell_matrix: List[List[List[Track]]] = [
            [[] for _ in range(self.cols)] for _ in range(self.rows)
        ]

        # Bin tracks into grid cells based on their center coordinates
        for track in active_tracks:
            cx, cy = track.center
            col = int(np.clip(cx // cell_w, 0, self.cols - 1))
            row = int(np.clip(cy // cell_h, 0, self.rows - 1))
            cell_matrix[row][col].append(track)

        cell_infos: List[GridCellInfo] = []
        densities: List[float] = []

        for r in range(self.rows):
            for c in range(self.cols):
                tracks_in_cell = cell_matrix[r][c]
                count = len(tracks_in_cell)
                norm_dens = float(np.clip(count / self.capacity, 0.0, 1.0))
                densities.append(norm_dens)

                # Speeds and velocity vectors in cell
                cell_speeds = []
                dxs, dys = [], []
                tids = []

                for t in tracks_in_cell:
                    tids.append(t.track_id)
                    hist = list(trajectory_mgr.histories.get(t.track_id, []))
                    if hist:
                        latest = hist[-1]
                        cell_speeds.append(latest.speed)
                        dxs.append(latest.dx)
                        dys.append(latest.dy)

                avg_speed = float(np.mean(cell_speeds)) if cell_speeds else 0.0
                mean_dx = float(np.mean(dxs)) if dxs else 0.0
                mean_dy = float(np.mean(dys)) if dys else 0.0

                x1 = int(round(c * cell_w))
                y1 = int(round(r * cell_h))
                x2 = int(round((c + 1) * cell_w))
                y2 = int(round((r + 1) * cell_h))

                cell_infos.append(
                    GridCellInfo(
                        row=r,
                        col=c,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        person_count=count,
                        normalized_density=norm_dens,
                        average_speed=avg_speed,
                        mean_dx=mean_dx,
                        mean_dy=mean_dy,
                        track_ids=tids
                    )
                )

        max_dens = float(max(densities)) if densities else 0.0
        mean_dens = float(np.mean(densities)) if densities else 0.0

        return DensityGrid(
            rows=self.rows,
            cols=self.cols,
            max_cell_density=max_dens,
            mean_grid_density=mean_dens,
            cells=cell_infos
        )
