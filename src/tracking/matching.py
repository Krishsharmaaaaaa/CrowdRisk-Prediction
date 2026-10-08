"""
Bipartite matching and IoU computation utilities for Multi-Object Tracking.
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_ious(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two sets of bounding boxes.
    boxes1: (N, 4) in [x1, y1, x2, y2]
    boxes2: (M, 4) in [x1, y1, x2, y2]
    Returns: (N, M) matrix of IoUs
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    boxes1 = np.ascontiguousarray(boxes1, dtype=np.float32)
    boxes2 = np.ascontiguousarray(boxes2, dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / np.clip(union, a_min=1e-6, a_max=None)
    return iou


def iou_distance(tracks: List[np.ndarray], detections: List[np.ndarray]) -> np.ndarray:
    """
    Computes cost matrix based on 1 - IoU.
    tracks: list of track bounding boxes [x1, y1, x2, y2]
    detections: list of detection bounding boxes [x1, y1, x2, y2]
    """
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float32)

    boxes1 = np.array(tracks, dtype=np.float32)
    boxes2 = np.array(detections, dtype=np.float32)
    ious = bbox_ious(boxes1, boxes2)
    return 1.0 - ious


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solves linear sum assignment problem with cost threshold.
    Returns:
        matches: list of (track_idx, det_idx)
        unmatched_tracks: list of track_idx
        unmatched_detections: list of det_idx
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_detections = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in unmatched_tracks:
                unmatched_tracks.remove(r)
            if c in unmatched_detections:
                unmatched_detections.remove(c)

    return matches, unmatched_tracks, unmatched_detections
