"""Real-time vision detection pipeline using YOLO and Haar cascades.

YOLO is used for general object/person context; Haar is used for face presence/count.
This module does not claim weapon detection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class VisionDetectorConfig:
    YOLO_MODEL_PATH: str = "yolo26n.pt"
    YOLO_DEVICE: str = "auto"
    YOLO_IMGSZ: int = 320
    YOLO_CONF_THRESHOLD: float = 0.45

    HAAR_CASCADE_RESOURCE: str = "haarcascade_frontalface_default.xml"
    HAAR_SCALE_WIDTH: int = 480
    HAAR_SCALE_FACTOR: float = 1.05
    HAAR_MIN_NEIGHBORS: int = 3
    HAAR_MIN_SIZE: int = 40

    CAMERA_BUFFER_SIZE: int = 1
    CAMERA_FPS: int = 30
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    FRAME_RECOVERY_MAX_FAILURES: int = 10


class VisionDetector:
    """YOLO + Haar processor with explicit timing and health telemetry."""

    def __init__(
        self,
        yolo_device: str = "auto",
        yolo_imgsz: int = 320,
        haar_scale_width: int = 480,
        config: Optional[VisionDetectorConfig] = None,
    ) -> None:
        self.config = config or VisionDetectorConfig()
        self.yolo_imgsz = yolo_imgsz
        self.haar_scale_width = haar_scale_width

        self.gray_buffer: Optional[np.ndarray] = None
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self.last_timings: Dict[str, float] = {
            "yolo_ms": 0.0,
            "haar_ms": 0.0,
            "vision_total_ms": 0.0,
        }
        self.health_status = "OK"
        self.last_error: Optional[str] = None

        try:
            if not Path(self.config.YOLO_MODEL_PATH).exists():
                raise FileNotFoundError(
                    f"YOLO model not found: {self.config.YOLO_MODEL_PATH}"
                )
            self.yolo = YOLO(self.config.YOLO_MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc, exc_info=True)
            raise RuntimeError(f"Could not load YOLO model: {exc}") from exc

        try:
            self.device = yolo_device
            if yolo_device == "auto":
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as exc:
            logger.warning("Device detection failed; using CPU: %s", exc)
            self.device = "cpu"

        try:
            cascade_path = cv2.data.haarcascades + self.config.HAAR_CASCADE_RESOURCE
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError(f"Haar cascade empty at: {cascade_path}")
        except Exception as exc:
            logger.error("Failed to load Haar cascade: %s", exc, exc_info=True)
            raise RuntimeError(f"Could not load Haar cascade: {exc}") from exc

    def _record_error(self, message: str) -> None:
        self.health_status = "DEGRADED"
        self.last_error = message
        logger.warning(message)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR numpy array with shape (H, W, 3)")

        start = time.perf_counter()
        self.health_status = "OK"
        self.last_error = None

        yolo_start = time.perf_counter()
        person_count, object_names, annotated = self._detect_yolo(frame)
        self.last_timings["yolo_ms"] = (time.perf_counter() - yolo_start) * 1000.0

        haar_start = time.perf_counter()
        face_count = self._detect_haar_faces(annotated)
        self.last_timings["haar_ms"] = (time.perf_counter() - haar_start) * 1000.0

        self._update_fps()
        self.last_timings["vision_total_ms"] = (time.perf_counter() - start) * 1000.0

        return annotated, {
            "person_count": person_count,
            "face_count": face_count,
            "objects": object_names,
            "fps": self.fps,
            "timings": dict(self.last_timings),
            "health_status": self.health_status,
            "last_error": self.last_error,
        }

    def _detect_yolo(
        self,
        frame: np.ndarray,
    ) -> Tuple[int, list, np.ndarray]:
        try:
            results = self.yolo(
                frame,
                verbose=False,
                device=self.device,
                imgsz=self.yolo_imgsz,
                conf=self.config.YOLO_CONF_THRESHOLD,
            )
            annotated = results[0].plot()
        except Exception as exc:
            self._record_error(f"YOLO inference failed: {exc}")
            return 0, [], frame.copy()

        person_count = 0
        object_names = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return 0, [], annotated

        try:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo.names[class_id]
                confidence = float(box.conf[0])
                object_names.append((class_name, confidence))
                if class_name == "person":
                    person_count += 1
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            self._record_error(f"YOLO result parsing failed: {exc}")

        return person_count, object_names, annotated

    def _detect_haar_faces(self, frame: np.ndarray) -> int:
        orig_h, orig_w = frame.shape[:2]
        scale = self.haar_scale_width / orig_w if orig_w > self.haar_scale_width else 1.0

        if scale != 1.0:
            small_frame = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small_frame = frame

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.HAAR_SCALE_FACTOR,
                minNeighbors=self.config.HAAR_MIN_NEIGHBORS,
                minSize=(self.config.HAAR_MIN_SIZE, self.config.HAAR_MIN_SIZE),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except Exception as exc:
            self._record_error(f"Haar detection failed: {exc}")
            return 0

        for x, y, w, h in faces:
            if scale != 1.0:
                x, y, w, h = tuple(int(v / scale) for v in (x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return len(faces)

    def _update_fps(self) -> None:
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_frame_time = now

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": self.health_status,
            "last_error": self.last_error,
        }
