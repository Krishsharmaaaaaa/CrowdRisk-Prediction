"""
ByteTrack Multi-Object Tracker implementation.
Assigns persistent IDs to detected persons across video frames.
"""

from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from ..detection.yolo_detector import Detection
from .kalman_filter import KalmanFilterBox
from .matching import iou_distance, linear_assignment


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class Track:
    """Represents an individual person track across time."""

    _count = 0

    def __init__(self, bbox: np.ndarray, score: float, frame_id: int):
        Track._count += 1
        self.track_id = Track._count
        self.state = TrackState.NEW
        self.is_activated = False

        self.score = float(score)
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.time_since_update = 0

        self.kalman_filter = KalmanFilterBox()
        # bbox format: [x1, y1, x2, y2] -> convert to [cx, cy, aspect_ratio, height]
        measurement = self._tlbr_to_xyah(bbox)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)

        self._bbox = bbox.copy()

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    @staticmethod
    def _tlbr_to_xyah(tlbr: np.ndarray) -> np.ndarray:
        w = float(tlbr[2] - tlbr[0])
        h = float(tlbr[3] - tlbr[1])
        cx = float(tlbr[0] + w / 2.0)
        cy = float(tlbr[1] + h / 2.0)
        a = w / (h + 1e-6)
        return np.array([cx, cy, a, h], dtype=np.float32)

    @staticmethod
    def _xyah_to_tlbr(xyah: np.ndarray) -> np.ndarray:
        cx, cy, a, h = xyah[:4]
        w = a * h
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @property
    def tlbr(self) -> np.ndarray:
        return self._bbox

    @property
    def bbox(self) -> np.ndarray:
        return self._bbox

    @property
    def center(self) -> Tuple[float, float]:
        tlbr = self._bbox
        cx = float((tlbr[0] + tlbr[2]) / 2.0)
        cy = float((tlbr[1] + tlbr[3]) / 2.0)
        return (cx, cy)

    def predict(self) -> None:
        if self.state != TrackState.TRACKED:
            self.mean[7] = 0.0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self._bbox = self._xyah_to_tlbr(self.mean[:4])

    def update(self, detection: Detection, frame_id: int) -> None:
        self.frame_id = frame_id
        self.time_since_update = 0
        self.score = detection.confidence

        measurement = self._tlbr_to_xyah(detection.bbox)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, measurement)
        self._bbox = detection.bbox.copy()

        self.state = TrackState.TRACKED
        self.is_activated = True

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


class ByteTrackTracker:
    """ByteTrack Multi-Object Tracker."""

    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.80,
        track_buffer: int = 30,
        min_box_area: float = 100.0
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area

        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        self.frame_id = 0

    def reset(self):
        Track.reset_counter()
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_id = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Updates tracks with current frame detections.
        Returns list of actively confirmed tracks.
        """
        self.frame_id += 1

        # Filter detections by area
        valid_dets = [d for d in detections if d.area >= self.min_box_area]

        # Separate detections into high and low confidence sets
        dets_high = [d for d in valid_dets if d.confidence >= self.track_thresh]
        dets_low = [d for d in valid_dets if d.confidence < self.track_thresh]

        # Predict current locations for all existing tracks
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()

        # Step 1: Associate high confidence detections with confirmed tracks
        pool_tracks = [t for t in self.tracked_tracks if t.is_activated] + self.lost_tracks
        boxes_tracks = [t.tlbr for t in pool_tracks]
        boxes_dets_high = [d.bbox for d in dets_high]

        cost_matrix = iou_distance(boxes_tracks, boxes_dets_high)
        matches_a, u_track_a, u_det_high = linear_assignment(cost_matrix, thresh=self.match_thresh)

        matched_tracks = []
        for t_idx, d_idx in matches_a:
            track = pool_tracks[t_idx]
            det = dets_high[d_idx]
            track.update(det, self.frame_id)
            matched_tracks.append(track)
            if track in self.lost_tracks:
                self.lost_tracks.remove(track)
            if track not in self.tracked_tracks:
                self.tracked_tracks.append(track)

        # Step 2: Associate low confidence detections with remaining unmatched tracks
        remain_tracks = [pool_tracks[i] for i in u_track_a if pool_tracks[i].state == TrackState.TRACKED]
        boxes_remain_tracks = [t.tlbr for t in remain_tracks]
        boxes_dets_low = [d.bbox for d in dets_low]

        cost_matrix_low = iou_distance(boxes_remain_tracks, boxes_dets_low)
        matches_b, u_track_b, _ = linear_assignment(cost_matrix_low, thresh=0.5)

        for t_idx, d_idx in matches_b:
            track = remain_tracks[t_idx]
            det = dets_low[d_idx]
            track.update(det, self.frame_id)
            matched_tracks.append(track)
            if track in self.lost_tracks:
                self.lost_tracks.remove(track)
            if track not in self.tracked_tracks:
                self.tracked_tracks.append(track)

        # Mark unassociated tracks as lost
        for i in u_track_b:
            track = remain_tracks[i]
            track.mark_lost()
            if track in self.tracked_tracks:
                self.tracked_tracks.remove(track)
            if track not in self.lost_tracks:
                self.lost_tracks.append(track)

        # Step 3: Handle unconfirmed tracks with remaining high score detections
        unconfirmed = [t for t in self.tracked_tracks if not t.is_activated]
        boxes_unconfirmed = [t.tlbr for t in unconfirmed]
        remain_dets_high = [dets_high[i] for i in u_det_high]
        boxes_remain_dets = [d.bbox for d in remain_dets_high]

        cost_matrix_u = iou_distance(boxes_unconfirmed, boxes_remain_dets)
        matches_c, u_unconf, u_final_det = linear_assignment(cost_matrix_u, thresh=0.7)

        for t_idx, d_idx in matches_c:
            track = unconfirmed[t_idx]
            det = remain_dets_high[d_idx]
            track.update(det, self.frame_id)
            matched_tracks.append(track)

        for i in u_unconf:
            track = unconfirmed[i]
            track.mark_removed()
            if track in self.tracked_tracks:
                self.tracked_tracks.remove(track)

        # Initialize new tracks for unmatched high confidence detections
        for i in u_final_det:
            det = remain_dets_high[i]
            new_track = Track(det.bbox, det.confidence, self.frame_id)
            # Activate if confidence is sufficiently high
            new_track.is_activated = True
            new_track.state = TrackState.TRACKED
            self.tracked_tracks.append(new_track)

        # Manage lost track buffer
        for track in list(self.lost_tracks):
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                self.lost_tracks.remove(track)
                self.removed_tracks.append(track)

        # Output active tracks
        active_tracks = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
        return active_tracks
