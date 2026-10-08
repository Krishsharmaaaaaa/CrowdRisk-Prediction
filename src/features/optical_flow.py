"""
Dense Optical Flow feature extractor using OpenCV Farneback algorithm.
Computes pixel-level motion statistics independently from bounding boxes.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple
import cv2
import math
import numpy as np


@dataclass
class OpticalFlowFeatures:
    """Aggregated dense optical flow metrics for a frame."""
    mean_magnitude: float
    magnitude_variance: float
    high_flow_ratio: float       # Ratio of pixels with magnitude > threshold
    dominant_direction_deg: float # [0, 360)
    dominant_direction_rad: float # [-pi, pi]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class OpticalFlowExtractor:
    """
    Computes dense Farneback optical flow on consecutive grayscale frames.
    Supports downscaling for real-time FPS efficiency.
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        fast_downscale: int = 2,
        flow_mag_threshold: float = 2.0
    ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.fast_downscale = max(1, fast_downscale)
        self.flow_mag_threshold = flow_mag_threshold

        self.prev_gray: Optional[np.ndarray] = None
        self.last_flow: Optional[np.ndarray] = None

    def reset(self):
        self.prev_gray = None
        self.last_flow = None

    def compute(self, frame_bgr: np.ndarray) -> OpticalFlowFeatures:
        """
        Processes frame and computes flow against previous frame.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Downscale for performance if configured
        if self.fast_downscale > 1:
            h, w = gray.shape
            gray_small = cv2.resize(
                gray,
                (w // self.fast_downscale, h // self.fast_downscale),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            gray_small = gray

        if self.prev_gray is None:
            self.prev_gray = gray_small
            return OpticalFlowFeatures(
                mean_magnitude=0.0,
                magnitude_variance=0.0,
                high_flow_ratio=0.0,
                dominant_direction_deg=0.0,
                dominant_direction_rad=0.0
            )

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray_small,
            None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0
        )
        self.prev_gray = gray_small
        self.last_flow = flow

        fx, fy = flow[..., 0], flow[..., 1]
        magnitude = np.hypot(fx, fy)

        # Rescale magnitude back if downscaled
        if self.fast_downscale > 1:
            magnitude *= self.fast_downscale

        mean_mag = float(np.mean(magnitude))
        var_mag = float(np.var(magnitude))

        # Fraction of pixels exceeding high flow threshold
        high_flow_pixels = np.sum(magnitude > self.flow_mag_threshold)
        high_flow_ratio = float(high_flow_pixels / max(1, magnitude.size))

        # Dominant flow direction
        mean_fx = float(np.mean(fx))
        mean_fy = float(np.mean(fy))
        dom_rad = math.atan2(mean_fy, mean_fx)
        dom_deg = (math.degrees(dom_rad) + 360.0) % 360.0

        return OpticalFlowFeatures(
            mean_magnitude=mean_mag,
            magnitude_variance=var_mag,
            high_flow_ratio=high_flow_ratio,
            dominant_direction_deg=dom_deg,
            dominant_direction_rad=dom_rad
        )
