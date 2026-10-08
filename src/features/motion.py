"""
Crowd Motion Feature Extractor.
Computes frame-level kinematic aggregations, circular statistics, and direction entropy.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import math
import numpy as np

from ..tracking.trajectory import TrajectoryManager, TrajectoryPoint
from ..tracking.bytetrack_tracker import Track


@dataclass
class CrowdMotionFeatures:
    """Frame-level aggregated motion features."""
    timestamp: float
    frame: int
    person_count: int
    mean_speed: float
    speed_variance: float
    max_speed: float
    mean_acceleration: float
    direction_entropy: float        # Normalized Shannon entropy [0, 1]
    direction_variance: float       # Circular variance [0, 1]
    moving_ratio: float             # Proportion of moving people [0, 1]
    density: float                  # Normalized global density [0, 1]
    density_change: float           # Relative change from previous frame [-1, 1]
    trajectory_irregularity: float  # Mean trajectory path-to-displacement ratio

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class MotionFeatureExtractor:
    """
    Extracts aggregated crowd motion statistics across tracked individuals.
    Uses image-area-aware global density normalization to prevent count saturation.
    """

    def __init__(
        self,
        moving_speed_thresh: float = 0.5,
        entropy_num_bins: int = 8,
        reference_density_per_mpx: float = 200.0,
        default_frame_area: float = 1280 * 720
    ):
        self.moving_speed_thresh = moving_speed_thresh
        self.num_bins = entropy_num_bins
        self.reference_density_per_mpx = max(1.0, reference_density_per_mpx)
        self.default_frame_area = default_frame_area
        self.prev_person_count: Optional[int] = None
        self.prev_density: Optional[float] = None

    def extract(
        self,
        frame_id: int,
        timestamp: float,
        active_tracks: List[Track],
        trajectory_mgr: TrajectoryManager,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ) -> CrowdMotionFeatures:
        person_count = len(active_tracks)

        # Image area computation
        if frame_width is not None and frame_height is not None and frame_width > 0 and frame_height > 0:
            area_px = float(frame_width * frame_height)
        else:
            area_px = float(self.default_frame_area)
        area_mpx = area_px / 1_000_000.0

        if person_count == 0:
            density = 0.0
            density_change = 0.0 if self.prev_density is None else float(np.clip(0.0 - self.prev_density, -1.0, 1.0))
            self.prev_density = 0.0
            self.prev_person_count = 0
            return CrowdMotionFeatures(
                timestamp=timestamp,
                frame=frame_id,
                person_count=0,
                mean_speed=0.0,
                speed_variance=0.0,
                max_speed=0.0,
                mean_acceleration=0.0,
                direction_entropy=0.0,
                direction_variance=0.0,
                moving_ratio=0.0,
                density=density,
                density_change=density_change,
                trajectory_irregularity=1.0
            )

        speeds: List[float] = []
        accels: List[float] = []
        angles_rad: List[float] = []
        angles_deg: List[float] = []
        irregularities: List[float] = []
        moving_count = 0

        for track in active_tracks:
            stats = trajectory_mgr.get_track_stats(track.track_id)
            if stats is not None:
                speeds.append(stats["latest_speed"])
                accels.append(stats["mean_accel"])
                irregularities.append(stats["irregularity"])
                if stats["latest_speed"] >= self.moving_speed_thresh:
                    moving_count += 1
                    angles_rad.append(stats["latest_direction"])
                    deg = (math.degrees(stats["latest_direction"]) + 360.0) % 360.0
                    angles_deg.append(deg)
            else:
                # Track too short for full window stats
                hist = list(trajectory_mgr.histories.get(track.track_id, []))
                if hist:
                    latest = hist[-1]
                    speeds.append(latest.speed)
                    accels.append(abs(latest.acceleration))
                    irregularities.append(1.0)
                    if latest.speed >= self.moving_speed_thresh:
                        moving_count += 1
                        angles_rad.append(latest.direction_rad)
                        angles_deg.append(latest.direction_deg)

        # Basic Speed Statistics
        if speeds:
            mean_speed = float(np.mean(speeds))
            speed_var = float(np.var(speeds))
            max_speed = float(np.max(speeds))
            mean_accel = float(np.mean(accels)) if accels else 0.0
            mean_irreg = float(np.mean(irregularities)) if irregularities else 1.0
            moving_ratio = float(moving_count / max(1, person_count))
        else:
            mean_speed = 0.0
            speed_var = 0.0
            max_speed = 0.0
            mean_accel = 0.0
            mean_irreg = 1.0
            moving_ratio = 0.0

        # Directional Shannon Entropy & Circular Variance
        if len(angles_deg) >= 2:
            # Direction Entropy over discrete angle bins [0, 360)
            bin_width = 360.0 / self.num_bins
            bin_indices = [int(math.floor(round(deg % 360.0, 4) / bin_width)) % self.num_bins for deg in angles_deg]
            counts = np.bincount(bin_indices, minlength=self.num_bins)
            probs = counts / np.sum(counts)

            # Filter non-zero probabilities for Shannon calculation
            nonzero_p = probs[probs > 0]
            raw_entropy = -np.sum(nonzero_p * np.log2(nonzero_p))
            max_entropy = np.log2(self.num_bins)
            direction_entropy = float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0

            # Circular Variance: 1 - R, where R is length of mean resultant vector
            sin_sum = np.sum(np.sin(angles_rad))
            cos_sum = np.sum(np.cos(angles_rad))
            r_bar = np.hypot(sin_sum, cos_sum) / len(angles_rad)
            direction_variance = float(np.clip(1.0 - r_bar, 0.0, 1.0))
        else:
            direction_entropy = 0.0
            direction_variance = 0.0

        # Global Image-Area-Aware Density: persons per megapixel normalized by reference capacity
        persons_per_mpx = person_count / max(0.01, area_mpx)
        density = float(np.clip(persons_per_mpx / self.reference_density_per_mpx, 0.0, 1.0))

        if self.prev_density is None:
            density_change = 0.0
        else:
            density_change = float(np.clip(density - self.prev_density, -1.0, 1.0))

        self.prev_person_count = person_count
        self.prev_density = density

        return CrowdMotionFeatures(
            timestamp=timestamp,
            frame=frame_id,
            person_count=person_count,
            mean_speed=mean_speed,
            speed_variance=speed_var,
            max_speed=max_speed,
            mean_acceleration=mean_accel,
            direction_entropy=direction_entropy,
            direction_variance=direction_variance,
            moving_ratio=moving_ratio,
            density=density,
            density_change=density_change,
            trajectory_irregularity=mean_irreg
        )
