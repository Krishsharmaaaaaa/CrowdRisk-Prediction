"""
Spatially Localized Grid-Cell Feature Extraction Engine.
Extracts cell-wise risk-driver feature vectors:
- D: Density / Occupancy
- O: Directional Disorder / Angular Entropy
- B: Bottleneck / Congestion
- K: Kinematic Instability
along with same-scene calm reference baselines.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..tracking.bytetrack_tracker import Track
from ..tracking.trajectory import TrajectoryManager


@dataclass
class SpatialCellFeatureVector:
    """Feature representation for a single spatial grid cell c."""
    cell_id: str                   # e.g., 'C1', 'C2', ..., 'C16'
    row: int                       # 0-indexed row
    col: int                       # 0-indexed column
    x1: int                        # Top-left pixel x
    y1: int                        # Top-left pixel y
    x2: int                        # Bottom-right pixel x
    y2: int                        # Bottom-right pixel y
    area_px: float                 # Cell area in pixels

    # Core 4 Driver Groups (all normalized to [0.0, 1.0])
    D: float                       # Density / Occupancy
    O: float                       # Directional Disorder / Entropy
    B: float                       # Bottleneck / Congestion
    K: float                       # Kinematic Instability

    # Detailed sub-metrics for auditability
    person_count: int = 0
    persons_per_mpx: float = 0.0
    occupancy_ratio: float = 0.0
    circular_variance: float = 0.0
    direction_entropy: float = 0.0
    mean_speed: float = 0.0
    speed_variance: float = 0.0
    stopped_ratio: float = 0.0
    mean_acceleration: float = 0.0
    trajectory_irregularity: float = 0.0
    flow_magnitude_mean: float = 0.0
    track_ids: List[int] = field(default_factory=list)

    def to_driver_vector(self) -> np.ndarray:
        """Returns the 4-dimensional driver vector [D, O, B, K]."""
        return np.array([self.D, self.O, self.B, self.K], dtype=np.float32)

    def to_dict(self) -> Dict:
        """Returns serialized dictionary representation."""
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "col": self.col,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "D": round(self.D, 4),
            "O": round(self.O, 4),
            "B": round(self.B, 4),
            "K": round(self.K, 4),
            "person_count": self.person_count,
            "persons_per_mpx": round(self.persons_per_mpx, 2),
            "occupancy_ratio": round(self.occupancy_ratio, 4),
            "circular_variance": round(self.circular_variance, 4),
            "direction_entropy": round(self.direction_entropy, 4),
            "mean_speed": round(self.mean_speed, 2),
            "speed_variance": round(self.speed_variance, 2),
            "stopped_ratio": round(self.stopped_ratio, 4),
            "mean_acceleration": round(self.mean_acceleration, 2),
            "trajectory_irregularity": round(self.trajectory_irregularity, 4),
            "flow_magnitude_mean": round(self.flow_magnitude_mean, 4),
            "num_tracks": len(self.track_ids)
        }


@dataclass
class SpatialGridFeatureMap:
    """Full frame spatial grid container."""
    frame_idx: int
    timestamp: float
    rows: int
    cols: int
    frame_width: int
    frame_height: int
    cells: List[SpatialCellFeatureVector]
    cell_matrix: List[List[SpatialCellFeatureVector]]

    # Grid aggregate summaries
    mean_D: float = 0.0
    mean_O: float = 0.0
    mean_B: float = 0.0
    mean_K: float = 0.0
    max_D_cell: str = ""
    max_B_cell: str = ""
    total_tracked_persons: int = 0

    def get_cell(self, cell_id: str) -> Optional[SpatialCellFeatureVector]:
        for c in self.cells:
            if c.cell_id == cell_id:
                return c
        return None

    def get_driver_tensor(self) -> np.ndarray:
        """Returns shape (rows, cols, 4) tensor of [D, O, B, K]."""
        tensor = np.zeros((self.rows, self.cols, 4), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                tensor[r, c] = self.cell_matrix[r][c].to_driver_vector()
        return tensor

    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "grid_dimensions": f"{self.rows}x{self.cols}",
            "total_tracked_persons": self.total_tracked_persons,
            "mean_D": round(self.mean_D, 4),
            "mean_O": round(self.mean_O, 4),
            "mean_B": round(self.mean_B, 4),
            "mean_K": round(self.mean_K, 4),
            "max_D_cell": self.max_D_cell,
            "max_B_cell": self.max_B_cell,
            "cells": [c.to_dict() for c in self.cells]
        }


class SameSceneCalmReferenceManager:
    """
    Maintains empirical same-scene calm reference vectors \\bar{x}_c = [\\bar{D}, \\bar{O}, \\bar{B}, \\bar{K}]
    accumulated during low-risk/calm baseline windows.
    """

    def __init__(self, rows: int = 4, cols: int = 4, warmup_frames: int = 30):
        self.rows = rows
        self.cols = cols
        self.warmup_frames = warmup_frames
        self.reference_samples: Dict[str, List[np.ndarray]] = {
            f"C{r * cols + c + 1}": [] for r in range(rows) for c in range(cols)
        }
        self.frozen_reference: Optional[Dict[str, np.ndarray]] = None

    def reset(self):
        self.reference_samples = {
            f"C{r * self.cols + c + 1}": [] for r in range(self.rows) for c in range(self.cols)
        }
        self.frozen_reference = None

    def update(self, grid_map: SpatialGridFeatureMap, is_calm_window: bool = True):
        """Accumulates calm baseline observations."""
        if self.frozen_reference is not None:
            return

        for cell in grid_map.cells:
            vec = cell.to_driver_vector()
            self.reference_samples[cell.cell_id].append(vec)

        if grid_map.frame_idx >= self.warmup_frames and is_calm_window:
            self.freeze_reference()

    def freeze_reference(self):
        """Computes median / mean calm reference baseline for each cell."""
        ref: Dict[str, np.ndarray] = {}
        for cid, samples in self.reference_samples.items():
            if len(samples) > 0:
                arr = np.array(samples)
                # Use 25th percentile / median to represent robust calm baseline
                ref[cid] = np.percentile(arr, 25, axis=0)
            else:
                ref[cid] = np.zeros(4, dtype=np.float32)
        self.frozen_reference = ref

    def get_reference_vector(self, cell_id: str) -> np.ndarray:
        """Returns [\\bar{D}, \\bar{O}, \\bar{B}, \\bar{K}] for the specified cell."""
        if self.frozen_reference and cell_id in self.frozen_reference:
            return self.frozen_reference[cell_id]
        if cell_id in self.reference_samples and len(self.reference_samples[cell_id]) > 0:
            return np.mean(self.reference_samples[cell_id], axis=0)
        return np.zeros(4, dtype=np.float32)


class SpatialCellFeatureExtractor:
    """
    Computes spatially localized [D, O, B, K] feature representations
    across an M x N spatial grid.
    """

    def __init__(
        self,
        grid_rows: int = 4,
        grid_cols: int = 4,
        reference_density_per_mpx: float = 200.0,
        free_flow_speed_px: float = 4.0,
        ref_speed_variance: float = 16.0,
        ref_acceleration: float = 4.0,
        ref_flow_mag: float = 3.0,
        num_entropy_bins: int = 8
    ):
        self.rows = max(1, grid_rows)
        self.cols = max(1, grid_cols)
        self.ref_density_per_mpx = max(1.0, reference_density_per_mpx)
        self.free_flow_speed = max(0.1, free_flow_speed_px)
        self.ref_speed_var = max(0.1, ref_speed_variance)
        self.ref_accel = max(0.1, ref_acceleration)
        self.ref_flow_mag = max(0.1, ref_flow_mag)
        self.num_entropy_bins = num_entropy_bins

    def extract_grid(
        self,
        frame_idx: int,
        timestamp: float,
        frame_width: int,
        frame_height: int,
        active_tracks: List[Track],
        trajectory_mgr: TrajectoryManager,
        flow_field: Optional[np.ndarray] = None
    ) -> SpatialGridFeatureMap:
        """
        Extracts spatial cell feature vectors across all M x N cells.
        """
        cell_w = frame_width / self.cols
        cell_h = frame_height / self.rows
        total_frame_area = frame_width * frame_height

        # Bin tracks by their center positions
        track_matrix: List[List[List[Track]]] = [
            [[] for _ in range(self.cols)] for _ in range(self.rows)
        ]
        for track in active_tracks:
            cx, cy = track.center
            col_idx = int(np.clip(cx // cell_w, 0, self.cols - 1))
            row_idx = int(np.clip(cy // cell_h, 0, self.rows - 1))
            track_matrix[row_idx][col_idx].append(track)

        cells: List[SpatialCellFeatureVector] = []
        cell_matrix: List[List[SpatialCellFeatureVector]] = [
            [] for _ in range(self.rows)
        ]

        d_vals, o_vals, b_vals, k_vals = [], [], [], []

        for r in range(self.rows):
            for c in range(self.cols):
                cell_index = r * self.cols + c + 1
                cell_id = f"C{cell_index}"

                x1 = int(round(c * cell_w))
                y1 = int(round(r * cell_h))
                x2 = int(round((c + 1) * cell_w))
                y2 = int(round((r + 1) * cell_h))
                cell_area = float((x2 - x1) * (y2 - y1))

                tracks_in_cell = track_matrix[r][c]
                count = len(tracks_in_cell)
                tids = [t.track_id for t in tracks_in_cell]

                # ----------------------------------------------------
                # 1. GROUP D: DENSITY & OCCUPANCY
                # ----------------------------------------------------
                cell_area_mpx = cell_area / 1e6
                persons_per_mpx = (count / cell_area_mpx) if cell_area_mpx > 0 else 0.0
                norm_density = float(np.clip(persons_per_mpx / self.ref_density_per_mpx, 0.0, 1.0))

                # Compute bounding box pixel occupancy intersection
                box_area_sum = 0.0
                for t in tracks_in_cell:
                    bx1, by1, bx2, by2 = t.bbox
                    ix1 = max(x1, bx1)
                    iy1 = max(y1, by1)
                    ix2 = min(x2, bx2)
                    iy2 = min(y2, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        box_area_sum += (ix2 - ix1) * (iy2 - iy1)
                occupancy_ratio = float(np.clip(box_area_sum / max(1.0, cell_area), 0.0, 1.0))

                # Composite Density Driver D
                D = float(np.clip(0.7 * norm_density + 0.3 * occupancy_ratio, 0.0, 1.0))

                # ----------------------------------------------------
                # Gather Kinematics for tracks in this cell
                # ----------------------------------------------------
                speeds = []
                angles = []
                accels = []
                curvatures = []
                stopped_count = 0

                for t in tracks_in_cell:
                    hist = list(trajectory_mgr.histories.get(t.track_id, []))
                    if hist:
                        latest = hist[-1]
                        spd = float(latest.speed)
                        speeds.append(spd)
                        if spd < 0.5:
                            stopped_count += 1
                        if spd > 0.1:
                            # Heading angle in radians [-pi, pi]
                            ang = math.atan2(latest.dy, latest.dx)
                            angles.append(ang)
                        accels.append(abs(latest.acceleration))

                    stats = trajectory_mgr.get_track_stats(t.track_id)
                    if stats:
                        curvatures.append(stats.get("irregularity", 1.0))
                    else:
                        curvatures.append(1.0)

                mean_speed = float(np.mean(speeds)) if speeds else 0.0
                speed_var = float(np.var(speeds)) if len(speeds) > 1 else 0.0
                stopped_ratio = (stopped_count / count) if count > 0 else 0.0
                mean_accel = float(np.mean(accels)) if accels else 0.0
                mean_curv = float(np.mean(curvatures)) if curvatures else 0.0

                # ----------------------------------------------------
                # 2. GROUP O: DIRECTIONAL DISORDER / ENTROPY
                # ----------------------------------------------------
                if len(angles) >= 2:
                    # Circular variance: 1 - resultant vector length
                    cos_sum = sum(math.cos(a) for a in angles)
                    sin_sum = sum(math.sin(a) for a in angles)
                    R_bar = math.hypot(cos_sum, sin_sum) / len(angles)
                    circ_var = float(np.clip(1.0 - R_bar, 0.0, 1.0))

                    # Shannon angular entropy across K bins
                    bin_counts = [0] * self.num_entropy_bins
                    bin_width = (2.0 * math.pi) / self.num_entropy_bins
                    for a in angles:
                        # Map [-pi, pi] -> [0, 2pi) -> bin index
                        normalized_a = (a + math.pi) % (2.0 * math.pi)
                        b_idx = int(np.clip(normalized_a // bin_width, 0, self.num_entropy_bins - 1))
                        bin_counts[b_idx] += 1

                    entropy = 0.0
                    for bc in bin_counts:
                        if bc > 0:
                            p = bc / len(angles)
                            entropy -= p * math.log2(p)
                    max_entropy = math.log2(self.num_entropy_bins)
                    norm_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
                    O = float(np.clip(0.5 * circ_var + 0.5 * norm_entropy, 0.0, 1.0))
                else:
                    circ_var = 0.0
                    norm_entropy = 0.0
                    O = 0.0

                # ----------------------------------------------------
                # 3. GROUP B: BOTTLENECK & CONGESTION
                # ----------------------------------------------------
                if count >= 1:
                    speed_drop = float(np.clip(max(0.0, 1.0 - (mean_speed / self.free_flow_speed)), 0.0, 1.0))
                    jamming_factor = 0.6 * speed_drop + 0.4 * stopped_ratio
                    # Bottleneck manifests when elevated density coincides with velocity loss
                    B = float(np.clip(D * jamming_factor, 0.0, 1.0))
                else:
                    B = 0.0

                # ----------------------------------------------------
                # 4. GROUP K: KINEMATIC INSTABILITY
                # ----------------------------------------------------
                # Cell-cropped optical flow patch
                if flow_field is not None and flow_field.shape[0] > y1 and flow_field.shape[1] > x1:
                    patch = flow_field[y1:min(y2, flow_field.shape[0]), x1:min(x2, flow_field.shape[1])]
                    if patch.size > 0:
                        mag = np.sqrt(patch[..., 0]**2 + patch[..., 1]**2)
                        flow_mean = float(np.mean(mag))
                        flow_std = float(np.std(mag))
                    else:
                        flow_mean, flow_std = 0.0, 0.0
                else:
                    flow_mean, flow_std = 0.0, 0.0

                norm_speed_var = float(np.clip(math.sqrt(speed_var) / math.sqrt(self.ref_speed_var), 0.0, 1.0))
                norm_accel = float(np.clip(mean_accel / self.ref_accel, 0.0, 1.0))
                norm_curv = float(np.clip(mean_curv / math.pi, 0.0, 1.0))
                norm_flow = float(np.clip((flow_mean + flow_std) / self.ref_flow_mag, 0.0, 1.0))

                if count >= 1:
                    K = float(np.clip(
                        0.35 * norm_speed_var + 0.25 * norm_accel + 0.15 * norm_curv + 0.25 * norm_flow,
                        0.0,
                        1.0
                    ))
                else:
                    # For empty cells, kinematics reflect ambient background flow agitation (or 0.0)
                    K = float(np.clip(0.5 * norm_flow, 0.0, 1.0))

                cell_feat = SpatialCellFeatureVector(
                    cell_id=cell_id,
                    row=r,
                    col=c,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    area_px=cell_area,
                    D=D,
                    O=O,
                    B=B,
                    K=K,
                    person_count=count,
                    persons_per_mpx=persons_per_mpx,
                    occupancy_ratio=occupancy_ratio,
                    circular_variance=circ_var,
                    direction_entropy=norm_entropy,
                    mean_speed=mean_speed,
                    speed_variance=speed_var,
                    stopped_ratio=stopped_ratio,
                    mean_acceleration=mean_accel,
                    trajectory_irregularity=mean_curv,
                    flow_magnitude_mean=flow_mean,
                    track_ids=tids
                )

                cells.append(cell_feat)
                cell_matrix[r].append(cell_feat)

                d_vals.append(D)
                o_vals.append(O)
                b_vals.append(B)
                k_vals.append(K)

        # Compute aggregate grid summaries
        mean_d = float(np.mean(d_vals)) if d_vals else 0.0
        mean_o = float(np.mean(o_vals)) if o_vals else 0.0
        mean_b = float(np.mean(b_vals)) if b_vals else 0.0
        mean_k = float(np.mean(k_vals)) if k_vals else 0.0

        max_d_cell = cells[int(np.argmax(d_vals))].cell_id if d_vals else ""
        max_b_cell = cells[int(np.argmax(b_vals))].cell_id if b_vals else ""

        return SpatialGridFeatureMap(
            frame_idx=frame_idx,
            timestamp=timestamp,
            rows=self.rows,
            cols=self.cols,
            frame_width=frame_width,
            frame_height=frame_height,
            cells=cells,
            cell_matrix=cell_matrix,
            mean_D=mean_d,
            mean_O=mean_o,
            mean_B=mean_b,
            mean_K=mean_k,
            max_D_cell=max_d_cell,
            max_B_cell=max_b_cell,
            total_tracked_persons=len(active_tracks)
        )
