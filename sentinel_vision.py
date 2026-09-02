"""SentinelVision Phase-1 multimodal orchestration.

Responsibilities:
- Acquire/coordinate vision and audio evidence.
- Validate sensor availability and freshness.
- Delegate all threat scoring/classification to FusionEngine.
- Track health, timing, alerts and synchronized per-frame results.

This module intentionally does not contain a second vision-threat scoring algorithm.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from audio import AudioDetector, AudioDetectorConfig
from fusion_engine import AudioFeatures, FusionConfig, FusionEngine, FusionResult, VisionFeatures
from vision import VisionDetector, VisionDetectorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    timestamp: float
    frame_id: int
    risk_label: str
    threat_score: float
    reasoning: str


@dataclass
class SentinelUnifiedConfig:
    """Orchestrator configuration with FusionConfig as the scoring authority."""

    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Compatibility overrides. They are folded into `fusion` during initialization.
    audio_weight: Optional[float] = None
    vision_weight: Optional[float] = None

    alert_queue_size: int = 100
    alert_cooldown_seconds: float = 2.0
    enable_logging: bool = True
    graceful_degradation: bool = True
    sync_timeout_ms: int = 100
    audio_max_age_sec: float = 3.0

    def __post_init__(self) -> None:
        fusion = self.fusion
        if self.audio_weight is not None or self.vision_weight is not None:
            fusion = replace(
                fusion,
                audio_weight=fusion.audio_weight if self.audio_weight is None else self.audio_weight,
                vision_weight=fusion.vision_weight if self.vision_weight is None else self.vision_weight,
            )
        self.fusion = fusion
        self.audio_weight = fusion.audio_weight
        self.vision_weight = fusion.vision_weight

        if self.alert_queue_size <= 0:
            raise ValueError("alert_queue_size must be positive")
        if self.alert_cooldown_seconds < 0:
            raise ValueError("alert_cooldown_seconds must be non-negative")
        if self.audio_max_age_sec < 0:
            raise ValueError("audio_max_age_sec must be non-negative")
        if self.sync_timeout_ms < 0:
            raise ValueError("sync_timeout_ms must be non-negative")


class SentinelVisionUnified:
    """Thread-safe multimodal surveillance prototype."""

    def __init__(
        self,
        audio_weight: Optional[float] = None,
        vision_weight: Optional[float] = None,
        config: Optional[SentinelUnifiedConfig] = None,
        vision_config: Optional[VisionDetectorConfig] = None,
        audio_config: Optional[AudioDetectorConfig] = None,
    ) -> None:
        self.config = config or SentinelUnifiedConfig()

        if audio_weight is not None or vision_weight is not None:
            self.config = replace(
                self.config,
                audio_weight=audio_weight,
                vision_weight=vision_weight,
            )

        self.fusion_engine = FusionEngine(self.config.fusion)
        self.lock = threading.Lock()

        self.vision_available = False
        self.audio_available = False
        self.last_vision_error: Optional[str] = None
        self.last_audio_error: Optional[str] = None
        self.vision_health = "ERROR"
        self.audio_health = "ERROR"

        self.vision: Optional[VisionDetector] = None
        self.audio: Optional[AudioDetector] = None

        try:
            self.vision = VisionDetector(
                yolo_device=vision_config.YOLO_DEVICE if vision_config else "auto",
                yolo_imgsz=vision_config.YOLO_IMGSZ if vision_config else 320,
                haar_scale_width=vision_config.HAAR_SCALE_WIDTH if vision_config else 480,
                config=vision_config,
            )
            self.vision_available = True
            self.vision_health = "OK"
        except Exception as exc:
            self.last_vision_error = str(exc)
            self.vision_health = "ERROR"
            logger.error("Vision initialization failed: %s", exc, exc_info=True)
            if not self.config.graceful_degradation:
                self.stop()
                raise

        try:
            self.audio = AudioDetector(config=audio_config)
            self.audio_available = True
            self.audio_health = "OK"
        except Exception as exc:
            self.last_audio_error = str(exc)
            self.audio_health = "ERROR"
            logger.error("Audio initialization failed: %s", exc, exc_info=True)
            if not self.config.graceful_degradation:
                self.stop()
                raise

        if not self.vision_available and not self.audio_available:
            self.stop()
            raise RuntimeError("Both vision and audio sensors failed to initialize")

        self.last_audio_result: Dict[str, Any] = self._create_empty_audio_result()
        self.last_audio_timestamp: Optional[float] = None
        self.last_vision_features: Dict[str, Any] = self._create_empty_vision_result()

        self.alerts: Deque[Alert] = deque(maxlen=self.config.alert_queue_size)
        self.last_alert_timestamp: Optional[float] = None
        self.frame_counter = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.running = True

    @staticmethod
    def _create_empty_audio_result() -> Dict[str, Any]:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "threat_score": 0.0,
            "risk_label": "BENIGN",
            "top_predictions": [],
            "threat_predictions": [],
            "peak": 0.0,
            "rms": 0.0,
            "timestamp": None,
            "inference_latency_ms": 0.0,
        }

    @staticmethod
    def _create_empty_vision_result() -> Dict[str, Any]:
        return {
            "persons": 0,
            "faces": 0,
            "objects": [],
            "fps": 0.0,
            "timings": {"yolo_ms": 0.0, "haar_ms": 0.0, "vision_total_ms": 0.0},
            "health_status": "ERROR",
            "last_error": None,
        }

    def _update_fps(self, timestamp: float) -> None:
        elapsed = timestamp - self.last_frame_time
        self.frame_counter += 1
        if elapsed >= 1.0:
            self.fps = self.frame_counter / elapsed
            self.frame_counter = 0
            self.last_frame_time = timestamp

    def _process_vision(self, frame: np.ndarray, timestamp: float) -> tuple[np.ndarray, Dict[str, Any]]:
        if not self.vision_available or self.vision is None:
            fallback = frame.copy()
            result = self._create_empty_vision_result()
            result["health_status"] = "ERROR"
            result["last_error"] = self.last_vision_error
            return fallback, result

        try:
            annotated, features = self.vision.process(frame)
            health = self.vision.get_health()
            self.vision_health = health.get("status", "OK")
            self.last_vision_error = health.get("last_error")

            result = {
                "persons": int(features.get("person_count", 0)),
                "faces": int(features.get("face_count", 0)),
                "objects": features.get("objects", []),
                "fps": float(features.get("fps", 0.0)),
                "timings": features.get(
                    "timings",
                    {"yolo_ms": 0.0, "haar_ms": 0.0, "vision_total_ms": 0.0},
                ),
                "health_status": self.vision_health,
                "last_error": self.last_vision_error,
                "timestamp": timestamp,
            }
            self.last_vision_features = result.copy()
            return annotated, result
        except Exception as exc:
            message = str(exc)
            self.vision_health = "ERROR"
            self.last_vision_error = message
            self.vision_available = False
            logger.error("Vision processing failed: %s", exc, exc_info=True)
            if not self.config.graceful_degradation:
                raise
            result = self._create_empty_vision_result()
            result["last_error"] = message
            return frame.copy(), result

    def _get_current_audio(self, now: float) -> tuple[Dict[str, Any], float]:
        if not self.audio_available or self.audio is None:
            return self.last_audio_result.copy(), float("inf")

        try:
            result = self.audio.get_result()
            with self.lock:
                self.last_audio_result = result.copy()

            timestamp = result.get("timestamp")
            self.last_audio_timestamp = timestamp
            age_ms = float("inf") if timestamp is None else max(0.0, (now - timestamp) * 1000.0)

            health = self.audio.get_health()
            self.audio_health = health.get("status", "OK")
            self.last_audio_error = health.get("last_error")
            if self.audio_health == "ERROR":
                # Failed inference should not provide stale evidence.
                age_ms = float("inf")
            return result.copy(), age_ms
        except Exception as exc:
            self.audio_health = "ERROR"
            self.last_audio_error = str(exc)
            logger.error("Audio result retrieval failed: %s", exc, exc_info=True)
            if not self.config.graceful_degradation:
                raise
            return self._create_empty_audio_result(), float("inf")

    def _calculate_composite_threat(
        self,
        vision_result: Dict[str, Any],
        audio_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route raw sensor features through the centralized FusionEngine only."""
        now = time.time()
        audio_timestamp = audio_result.get("timestamp")
        audio_is_fresh = (
            self.audio_available
            and self.audio_health != "ERROR"
            and audio_timestamp is not None
            and (now - float(audio_timestamp)) <= self.config.audio_max_age_sec
        )

        audio_features: Optional[AudioFeatures]
        if audio_is_fresh:
            audio_features = AudioFeatures(
                dominant_class=audio_result.get("label"),
                dominant_conf=float(audio_result.get("confidence", 0.0)),
                threat_score=float(audio_result.get("threat_score", 0.0)),
                top_k=list(audio_result.get("top_predictions", [])),
            )
        else:
            audio_features = None

        vision_is_available = self.vision_available and vision_result.get("health_status") != "ERROR"
        vision_features: Optional[VisionFeatures]
        if vision_is_available:
            vision_features = VisionFeatures(
                person_count=int(vision_result.get("persons", 0)),
                face_count=int(vision_result.get("faces", 0)),
                other_objects=list(vision_result.get("objects", [])),
            )
        else:
            vision_features = None

        fusion_result = self.fusion_engine.fuse(
            audio_features,
            vision_features,
            audio_available=audio_is_fresh,
            vision_available=vision_is_available,
        )

        coverage = 1.0 if fusion_result.audio_available and fusion_result.vision_available else 0.5 if (fusion_result.audio_available or fusion_result.vision_available) else 0.0

        return {
            "risk_label": fusion_result.level.name,
            "threat_score": fusion_result.fused_score,
            "sensor_coverage": coverage,
            "audio_available": fusion_result.audio_available,
            "vision_available": fusion_result.vision_available,
            "audio_contribution": fusion_result.audio_contrib,
            "vision_contribution": fusion_result.vision_contrib,
            "raw_audio_score": fusion_result.raw_audio_score,
            "raw_vision_score": fusion_result.raw_vision_score,
            "effective_audio_weight": fusion_result.effective_audio_weight,
            "effective_vision_weight": fusion_result.effective_vision_weight,
            "fusion_explanation": fusion_result.explanation,
        }

    def _generate_reasoning(
        self,
        vision_result: Dict[str, Any],
        audio_result: Dict[str, Any],
        composite: Dict[str, Any],
    ) -> str:
        parts: List[str] = []
        persons = int(vision_result.get("persons", 0))
        faces = int(vision_result.get("faces", 0))

        if persons:
            parts.append(f"{persons} person" + ("s" if persons != 1 else "") + " detected")
        if faces:
            parts.append(f"{faces} face" + ("s" if faces != 1 else "") + " detected")

        audio_available = bool(composite.get("audio_available"))
        if audio_available:
            audio_label = str(audio_result.get("label", "Unknown"))
            audio_conf = float(audio_result.get("confidence", 0.0))
            audio_threat = float(audio_result.get("threat_score", 0.0))
            parts.append(
                f"audio '{audio_label}' (classifier confidence={audio_conf:.1%}, threat score={audio_threat:.1%})"
            )

        if composite.get("risk_label") == "BENIGN" and not parts:
            return "No active threat evidence detected"

        score = float(composite.get("threat_score", 0.0))
        risk = composite.get("risk_label", "BENIGN")
        prefix = {
            "THREATENING": "High-risk evidence detected",
            "SUSPICIOUS": "Anomalous evidence detected",
            "BENIGN": "No strong threat evidence",
        }.get(risk, "Assessment")

        detail = "; ".join(parts) if parts else "no supporting sensor details"
        return f"{prefix}: {detail}; fused score={score:.3f}"

    def _check_and_queue_alert(
        self,
        frame_id: int,
        timestamp: float,
        composite: Dict[str, Any],
        reasoning: str,
    ) -> None:
        risk = composite.get("risk_label", "BENIGN")
        if risk == "BENIGN":
            return

        with self.lock:
            if (
                self.last_alert_timestamp is not None
                and timestamp - self.last_alert_timestamp < self.config.alert_cooldown_seconds
            ):
                return
            self.alerts.append(
                Alert(
                    timestamp=timestamp,
                    frame_id=frame_id,
                    risk_label=risk,
                    threat_score=float(composite.get("threat_score", 0.0)),
                    reasoning=reasoning,
                )
            )
            self.last_alert_timestamp = timestamp

    def process_frame(self, frame: np.ndarray, frame_id: Optional[int] = None) -> Dict[str, Any]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR numpy array with shape (H, W, 3)")

        timestamp = time.time()
        if frame_id is None:
            frame_id = self.frame_counter

        vision_start = time.perf_counter()
        annotated, vision_result = self._process_vision(frame, timestamp)
        vision_wall_ms = (time.perf_counter() - vision_start) * 1000.0

        audio_result, audio_age_ms = self._get_current_audio(timestamp)
        composite = self._calculate_composite_threat(vision_result, audio_result)
        reasoning = self._generate_reasoning(vision_result, audio_result, composite)
        composite["reasoning"] = reasoning

        self._check_and_queue_alert(frame_id, timestamp, composite, reasoning)
        self._update_fps(timestamp)

        timings = vision_result.get("timings", {})
        performance = {
            "vision_latency_ms": float(vision_wall_ms),
            "audio_age_ms": float(audio_age_ms),
            "yolo_ms": float(timings.get("yolo_ms", 0.0)),
            "haar_ms": float(timings.get("haar_ms", 0.0)),
            "ast_ms": float(audio_result.get("inference_latency_ms", 0.0)),
            "fps": float(self.fps or vision_result.get("fps", 0.0)),
        }

        health = {
            "vision_status": self.vision_health,
            "audio_status": self.audio_health,
            "last_vision_error": self.last_vision_error,
            "last_audio_error": self.last_audio_error,
        }

        return {
            "timestamp": timestamp,
            "frame_id": frame_id,
            "annotated_frame": annotated,
            "vision": vision_result,
            "audio": audio_result,
            "composite": composite,
            "performance": performance,
            "health": health,
        }

    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        if count <= 0:
            return []
        with self.lock:
            alerts_list = list(self.alerts)
        return sorted(alerts_list[-count:], key=lambda alert: alert.timestamp, reverse=True)

    def get_status(self) -> Dict[str, Any]:
        audio_age_ms = float("inf")
        if self.last_audio_timestamp is not None:
            audio_age_ms = max(0.0, (time.time() - self.last_audio_timestamp) * 1000.0)

        with self.lock:
            alerts_total = len(self.alerts)

        return {
            "vision_available": self.vision_available,
            "audio_available": self.audio_available,
            "vision_status": self.vision_health,
            "audio_status": self.audio_health,
            "fps": self.fps,
            "vision_latency_ms": self.last_vision_features.get("timings", {}).get("vision_total_ms", 0.0),
            "audio_age_ms": audio_age_ms,
            "alerts_total": alerts_total,
            "audio_weight": self.fusion_engine.audio_weight,
            "vision_weight": self.fusion_engine.vision_weight,
            "graceful_degradation": self.config.graceful_degradation,
            "last_vision_error": self.last_vision_error,
            "last_audio_error": self.last_audio_error,
        }

    def stop(self) -> None:
        self.running = False
        if self.audio is not None:
            try:
                self.audio.stop()
            except Exception as exc:
                logger.error("Error stopping audio detector: %s", exc, exc_info=True)
        self.audio = None
        if self.vision is not None:
            self.vision = None
