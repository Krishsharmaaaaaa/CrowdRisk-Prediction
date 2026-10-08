"""Feature extraction module for motion, optical flow, and spatial density."""
from .motion import CrowdMotionFeatures, MotionFeatureExtractor
from .optical_flow import OpticalFlowExtractor, OpticalFlowFeatures
from .density import DensityGrid, SpatialDensityEstimator
from .cell_grid import (
    SpatialCellFeatureVector,
    SpatialGridFeatureMap,
    SpatialCellFeatureExtractor,
    SameSceneCalmReferenceManager
)

__all__ = [
    "CrowdMotionFeatures",
    "MotionFeatureExtractor",
    "OpticalFlowExtractor",
    "OpticalFlowFeatures",
    "DensityGrid",
    "SpatialDensityEstimator",
    "SpatialCellFeatureVector",
    "SpatialGridFeatureMap",
    "SpatialCellFeatureExtractor",
    "SameSceneCalmReferenceManager",
]
