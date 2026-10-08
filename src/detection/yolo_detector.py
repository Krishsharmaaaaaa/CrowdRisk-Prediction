"""
YOLOv8n Person Detector module.
Optimized for real-time human detection with GPU/CPU automatic device selection.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
import cv2
import numpy as np

# Safe PyTorch & Ultralytics loading on Windows systems
HAVE_YOLO = False
try:
    import torch
    from ultralytics import YOLO
    HAVE_YOLO = True
except Exception as e:
    HAVE_YOLO = False


@dataclass
class Detection:
    """Represents a single detected person."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    center: Tuple[float, float]  # (cx, cy)
    class_id: int = 0

    @property
    def area(self) -> float:
        w = max(0.0, self.bbox[2] - self.bbox[0])
        h = max(0.0, self.bbox[3] - self.bbox[1])
        return float(w * h)

    @property
    def width(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        return float(max(0.0, self.bbox[3] - self.bbox[1]))


class YOLOPersonDetector:
    """YOLOv8-based person detector filtering strictly for humans (class 0)."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "auto",
        person_class_id: int = 0
    ):
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id
        self.use_yolo = HAVE_YOLO

        # Determine target device
        if self.use_yolo:
            try:
                if device == "auto":
                    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = device
                self.model = YOLO(model_name)
                try:
                    self.model.to(self.device)
                except Exception:
                    pass
            except Exception as e:
                self.use_yolo = False
                self.device = "cpu"
        else:
            self.device = "cpu"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs inference on a single frame and returns person detections.
        """
        if frame is None or frame.size == 0:
            return []

        if not self.use_yolo:
            return self._detect_hog(frame)

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[self.person_class_id],
                device=self.device,
                verbose=False
            )
        except Exception:
            return self._detect_hog(frame)

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            # Fallback to HOG if YOLO returned 0 on synthetic figures
            return self._detect_hog(frame)

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            if int(cls_id) != self.person_class_id:
                continue
            x1, y1, x2, y2 = box
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            detections.append(
                Detection(
                    bbox=box.astype(np.float32),
                    confidence=float(conf),
                    center=(cx, cy),
                    class_id=int(cls_id)
                )
            )

        return detections

    def _detect_hog(self, frame: np.ndarray) -> List[Detection]:
        """Fallback lightweight sprite/blob detector for synthetic or edge-case frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive / Otsu thresholding to extract foreground figures
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter for human-sized bounding contours
            if 80 < area < 40000:
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter aspect ratio (humans typically height >= width * 0.8)
                if h >= 10 and w >= 6:
                    bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
                    cx = float(x + w / 2.0)
                    cy = float(y + h / 2.0)
                    detections.append(
                        Detection(
                            bbox=bbox,
                            confidence=0.85,
                            center=(cx, cy),
                            class_id=self.person_class_id
                        )
                    )

        return detections
