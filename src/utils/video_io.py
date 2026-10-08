"""
Robust Video Reader and Writer utilities with codec fallback and stream handling.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np


class VideoReader:
    """Reads video files or camera streams with metadata and frame iteration."""

    def __init__(self, source: str, max_dimension: Optional[int] = None):
        self.source_str = str(source)
        self.max_dimension = max_dimension
        self.is_webcam = False

        # Parse webcam index if digits provided
        if self.source_str.isdigit():
            self.cap = cv2.VideoCapture(int(self.source_str))
            self.is_webcam = True
        else:
            path = Path(self.source_str)
            if not path.exists():
                raise FileNotFoundError(f"Video source file not found: {self.source_str}")
            self.cap = cv2.VideoCapture(str(path))

        if not self.cap.isOpened():
            raise IOError(f"Could not open video stream or file: {self.source_str}")

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps_val) if fps_val and fps_val > 0 else 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_webcam else -1

        # Calculate scaled dimensions if max_dimension is given
        self.width = self.original_width
        self.height = self.original_height
        if self.max_dimension and max(self.original_width, self.original_height) > self.max_dimension:
            scale = self.max_dimension / max(self.original_width, self.original_height)
            self.width = int(round(self.original_width * scale))
            self.height = int(round(self.original_height * scale))
            # Ensure dimensions are even for video codecs
            self.width = self.width if self.width % 2 == 0 else self.width - 1
            self.height = self.height if self.height % 2 == 0 else self.height - 1

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            if (self.width, self.height) != (self.original_width, self.original_height):
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            yield frame_idx, frame
            frame_idx += 1

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """Writes processed frames to a video file with robust codec fallbacks."""

    CODEC_FALLBACKS = ["mp4v", "avc1", "XVID", "MJPG"]

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.writer = None
        self._init_writer()

    def _init_writer(self) -> None:
        for codec in self.CODEC_FALLBACKS:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            if writer.isOpened():
                self.writer = writer
                return
        raise IOError(f"Failed to open video writer for {self.output_path} with supported codecs.")

    def write(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height))
            self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
