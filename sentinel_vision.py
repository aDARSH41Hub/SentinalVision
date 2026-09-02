#!/usr/bin/env python3
"""
SentinelVision Unified Multi-Sensor Surveillance System.

Phase-1 architecture:

    Camera
       |
       v
    VisionDetector
       |
       v
    VisionFeatures
       |
       +--------------------+
                            |
                            v
                       FusionEngine
                            ^
                            |
       +--------------------+
       |
    AudioFeatures
       ^
       |
    AudioDetector
       ^
       |
    Microphone

Design principles:
    - Vision and audio produce evidence.
    - FusionEngine owns multimodal decision scoring.
    - FusionConfig is the authoritative source for fusion weights,
      thresholds, EMA settings, and contextual vision scoring.
    - Missing/stale sensors are not treated as benign evidence.
    - Audio freshness is tracked explicitly as audio_age_ms.
    - Threat scores are heuristic/model scores, not calibrated probabilities.
    - Generic object detection is contextual perception, not validated
      weapon detection.
    - Sensor failures are explicitly surfaced through health telemetry.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, TypedDict

import cv2
import numpy as np

from fusion_engine import (
    AudioFeatures,
    FusionConfig,
    FusionEngine,
    VisionFeatures,
)
from audio import AudioDetector, AudioDetectorConfig
from vision import VisionDetector, VisionDetectorConfig

logger = logging.getLogger(__name__)


class DetectionResult(TypedDict):
    """
    Unified result returned by SentinelVisionUnified.process_frame().

    The result keeps raw sensor evidence, fusion output, health state,
    and performance/observation-age telemetry separately.
    """

    timestamp: float
    frame_id: int
    vision: Dict[str, Any]
    audio: Dict[str, Any]
    composite: Dict[str, Any]
    performance: Dict[str, float]
    health: Dict[str, Any]
    annotated_frame: Optional[np.ndarray]


@dataclass
class SentinelUnifiedConfig:
    """
    Runtime/orchestration configuration for SentinelVisionUnified.

    Fusion-specific settings belong to FusionConfig.

    Separation:

        SentinelUnifiedConfig
            -> runtime/orchestration behavior

        FusionConfig
            -> fusion weights, thresholds, EMA, vision scoring
    """

    # Single authoritative fusion configuration.
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Alert management.
    alert_queue_size: int = 100
    alert_cooldown_seconds: float = 2.0

    # Processing behavior.
    enable_logging: bool = True
    graceful_degradation: bool = True
    sync_timeout_ms: int = 100
    audio_max_age_sec: float = 3.0

    # Retained for compatibility/documentation.
    # These values are NOT used directly by FusionEngine.
    # FusionConfig remains authoritative for scoring.
    vision_threat_scale: Dict[str, float] = field(
        default_factory=lambda: {
            "multiple_persons": 0.3,
            "many_persons": 0.4,
            "face_detected": 0.2,
        }
    )

    @property
    def threat_thresholds(self) -> Dict[str, float]:
        """
        Compatibility view of the FusionConfig thresholds.

        These are derived from FusionConfig rather than maintained
        as a separate threshold source.
        """
        return {
            "BENIGN": self.fusion.benign_threshold,
            "SUSPICIOUS": self.fusion.suspicious_threshold,
            "THREATENING": self.fusion.threatening_threshold,
        }


@dataclass
class Alert:
    """Historical alert entry."""

    timestamp: float
    frame_id: int
    risk_label: str
    threat_score: float
    reasoning: str


class SentinelVisionUnified:
    """
    Unified multimodal surveillance orchestrator.

    Responsibilities:
        - Coordinate vision and audio detectors.
        - Enforce sensor availability/freshness semantics.
        - Pass evidence to FusionEngine.
        - Generate human-readable reasoning.
        - Maintain alerts.
        - Expose health and performance telemetry.
    """

    def __init__(
        self,
        vision_config: Optional[VisionDetectorConfig] = None,
        audio_config: Optional[AudioDetectorConfig] = None,
        audio_weight: Optional[float] = None,
        vision_weight: Optional[float] = None,
        config: Optional[SentinelUnifiedConfig] = None,
    ) -> None:
        """
        Initialize the unified detector.

        Args:
            vision_config:
                Configuration for YOLO/Haar vision detection.

            audio_config:
                Configuration for AST audio detection.

            audio_weight:
                Optional backward-compatible override for FusionConfig's
                audio weight.

            vision_weight:
                Optional backward-compatible override for FusionConfig's
                vision weight.

            config:
                SentinelUnifiedConfig.

        New code should preferably configure fusion weights through:

            SentinelUnifiedConfig(
                fusion=FusionConfig(
                    audio_weight=0.60,
                    vision_weight=0.40,
                )
            )
        """

        self.config = config or SentinelUnifiedConfig()

        # --------------------------------------------------------------
        # Resolve authoritative FusionConfig.
        # --------------------------------------------------------------
        fusion_config = self.config.fusion

        # Backward-compatible weight overrides.
        #
        # FusionConfig is frozen, so create a new instance rather than
        # mutating the existing configuration.
        if audio_weight is not None or vision_weight is not None:
            fusion_config = FusionConfig(
                **{
                    **fusion_config.__dict__,
                    "audio_weight": (
                        fusion_config.audio_weight
                        if audio_weight is None
                        else audio_weight
                    ),
                    "vision_weight": (
                        fusion_config.vision_weight
                        if vision_weight is None
                        else vision_weight
                    ),
                }
            )

            self.config.fusion = fusion_config

        self.audio_weight, self.vision_weight = (
            fusion_config.normalized_weights
        )

        logger.info(
            "Initialized FusionConfig with weights: "
            "audio=%.2f, vision=%.2f",
            self.audio_weight,
            self.vision_weight,
        )

        # --------------------------------------------------------------
        # Centralized fusion engine.
        # --------------------------------------------------------------
        self.fusion_engine = FusionEngine(
            config=fusion_config
        )

        # --------------------------------------------------------------
        # Detector configurations.
        # --------------------------------------------------------------
        vision_cfg = (
            vision_config
            or VisionDetectorConfig()
        )

        audio_cfg = (
            audio_config
            or AudioDetectorConfig()
        )

        # --------------------------------------------------------------
        # Sensor health state.
        # --------------------------------------------------------------
        self.vision_available = True
        self.audio_available = True

        self.last_vision_error: Optional[str] = None
        self.last_audio_error: Optional[str] = None

        self.vision: Optional[VisionDetector] = None
        self.audio: Optional[AudioDetector] = None

        # --------------------------------------------------------------
        # Vision initialization.
        # --------------------------------------------------------------
        try:
            logger.info(
                "Initializing vision detector..."
            )

            self.vision = VisionDetector(
                yolo_device="auto",
                yolo_imgsz=vision_cfg.YOLO_IMGSZ,
                haar_scale_width=vision_cfg.HAAR_SCALE_WIDTH,
                config=vision_cfg,
            )

            logger.info(
                "Vision detector initialized successfully"
            )

        except Exception as exc:
            logger.error(
                "Failed to initialize vision detector: %s",
                exc,
                exc_info=True,
            )

            self.vision_available = False
            self.last_vision_error = str(exc)
            self.vision = None

            if not self.config.graceful_degradation:
                raise RuntimeError(
                    f"Vision detector initialization failed: {exc}"
                ) from exc

        # --------------------------------------------------------------
        # Audio initialization.
        # --------------------------------------------------------------
        try:
            logger.info(
                "Initializing audio detector..."
            )

            self.audio = AudioDetector(
                config=audio_cfg
            )

            logger.info(
                "Audio detector initialized successfully"
            )

        except Exception as exc:
            logger.error(
                "Failed to initialize audio detector: %s",
                exc,
                exc_info=True,
            )

            self.audio_available = False
            self.last_audio_error = str(exc)
            self.audio = None

            if not self.config.graceful_degradation:
                raise RuntimeError(
                    f"Audio detector initialization failed: {exc}"
                ) from exc

        # --------------------------------------------------------------
        # At least one sensor must remain operational.
        # --------------------------------------------------------------
        if (
            not self.vision_available
            and not self.audio_available
        ):
            raise RuntimeError(
                "Both vision and audio detectors failed to "
                "initialize. Cannot proceed with unified surveillance."
            )

        # --------------------------------------------------------------
        # Thread-safe audio state.
        # --------------------------------------------------------------
        self.lock = threading.Lock()

        self.last_audio_result: Dict[str, Any] = (
            self._create_empty_audio_result()
        )

        self.last_audio_timestamp: Optional[float] = None

        # --------------------------------------------------------------
        # Performance telemetry.
        # --------------------------------------------------------------
        self.frame_count = 0

        # Total successfully processed frames.
        # Used when callers don't provide an explicit frame_id.
        self.total_frames_processed = 0

        self.last_stats_time = time.time()

        self.fps = 0.0
        self.vision_latency_ms = 0.0

        # This measures observation age, not AST inference latency.
        self.audio_age_ms = float("inf")

        # --------------------------------------------------------------
        # Alert management.
        # --------------------------------------------------------------
        self.alerts: Deque[Alert] = deque(
            maxlen=self.config.alert_queue_size
        )

        self.last_alert_time = 0.0

        logger.info(
            "SentinelVisionUnified initialized. "
            "Vision=%s, Audio=%s, Graceful degradation=%s",
            (
                "OK"
                if self.vision_available
                else "ERROR"
            ),
            (
                "OK"
                if self.audio_available
                else "ERROR"
            ),
            self.config.graceful_degradation,
        )

    # ==================================================================
    # Main processing pipeline
    # ==================================================================

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
    ) -> Optional[DetectionResult]:
        """
        Process one camera frame and fuse it with the latest audio evidence.

        Args:
            frame:
                BGR OpenCV frame.

            frame_id:
                Optional sequence number.

                If omitted, the detector's internal processed-frame counter
                is used. This preserves compatibility with simple callers
                and unit tests.

        Returns:
            DetectionResult, or None if a processing failure causes the
            method to return without a result.

        Raises:
            ValueError:
                If the input frame is invalid.
        """

        # --------------------------------------------------------------
        # Backward/simple-caller compatibility.
        # --------------------------------------------------------------
        if frame_id is None:
            frame_id = self.total_frames_processed

        # --------------------------------------------------------------
        # Validate input BEFORE touching sensors.
        # --------------------------------------------------------------
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise ValueError(
                "frame must be a color image with shape "
                "(height, width, 3)"
            )

        timestamp = time.time()

        # --------------------------------------------------------------
        # 1. Vision processing.
        # --------------------------------------------------------------
        vision_result = self._process_vision(
            frame=frame,
            timestamp=timestamp,
        )

        # --------------------------------------------------------------
        # 2. Read latest audio result.
        # --------------------------------------------------------------
        with self.lock:
            audio_result = (
                self.last_audio_result.copy()
            )

            if self.last_audio_timestamp is not None:
                self.audio_age_ms = max(
                    0.0,
                    (
                        timestamp
                        - self.last_audio_timestamp
                    )
                    * 1000.0,
                )
            else:
                self.audio_age_ms = float("inf")

        # --------------------------------------------------------------
        # 3. Refresh latest audio inference.
        # --------------------------------------------------------------
        if (
            self.audio_available
            and self.audio is not None
        ):
            try:
                new_audio = self.audio.get_result()

                if new_audio != audio_result:
                    with self.lock:
                        self.last_audio_result = (
                            new_audio.copy()
                        )

                        audio_timestamp = (
                            new_audio.get("timestamp")
                        )

                        if audio_timestamp is not None:
                            self.last_audio_timestamp = (
                                float(audio_timestamp)
                            )

                        audio_result = new_audio

                        if (
                            self.last_audio_timestamp
                            is not None
                        ):
                            self.audio_age_ms = max(
                                0.0,
                                (
                                    timestamp
                                    - self.last_audio_timestamp
                                )
                                * 1000.0,
                            )
                        else:
                            self.audio_age_ms = float(
                                "inf"
                            )

            except Exception as exc:
                logger.error(
                    "Failed to retrieve audio result: %s",
                    exc,
                    exc_info=True,
                )

                self.audio_available = False
                self.last_audio_error = str(exc)

                if not self.config.graceful_degradation:
                    raise RuntimeError(
                        f"Audio result retrieval failed: {exc}"
                    ) from exc

                audio_result = (
                    self._create_empty_audio_result()
                )

                self.audio_age_ms = float("inf")

        # --------------------------------------------------------------
        # 4. Centralized multimodal fusion.
        # --------------------------------------------------------------
        composite = (
            self._calculate_composite_threat(
                vision_result=vision_result,
                audio_result=audio_result,
            )
        )

        # --------------------------------------------------------------
        # 5. Human-readable reasoning.
        # --------------------------------------------------------------
        reasoning = self._generate_reasoning(
            vision_result=vision_result,
            audio_result=audio_result,
            composite=composite,
        )

        composite["reasoning"] = reasoning

        # --------------------------------------------------------------
        # 6. Alert management.
        # --------------------------------------------------------------
        self._check_and_queue_alert(
            frame_id=frame_id,
            timestamp=timestamp,
            composite=composite,
            reasoning=reasoning,
        )

        # --------------------------------------------------------------
        # 7. Performance metrics.
        # --------------------------------------------------------------
        self._update_performance_metrics()

        # --------------------------------------------------------------
        # 8. Track successfully processed frame.
        # --------------------------------------------------------------
        self.total_frames_processed += 1

        return DetectionResult(
            timestamp=timestamp,
            frame_id=frame_id,
            vision=vision_result,
            audio=audio_result,
            composite=composite,
            performance={
                "vision_latency_ms": (
                    self.vision_latency_ms
                ),
                "audio_age_ms": (
                    self.audio_age_ms
                ),
                "frame_fps": self.fps,
            },
            health=self._get_sensor_health(),
            annotated_frame=vision_result.get(
                "annotated_frame"
            ),
        )

    # ==================================================================
    # Vision
    # ==================================================================

    def _process_vision(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> Dict[str, Any]:
        """
        Run YOLO/Haar vision detection and expose raw contextual features.
        """

        del timestamp

        if (
            not self.vision_available
            or self.vision is None
        ):
            return self._create_empty_vision_result()

        try:
            start_time = time.time()

            annotated, features = (
                self.vision.process(frame)
            )

            self.vision_latency_ms = (
                time.time() - start_time
            ) * 1000.0

            return {
                "annotated_frame": annotated,
                "persons": features.get(
                    "person_count",
                    0,
                ),
                "faces": features.get(
                    "face_count",
                    0,
                ),
                "objects": features.get(
                    "objects",
                    [],
                ),
                "fps": features.get(
                    "fps",
                    0.0,
                ),
                "health_status": "OK",
            }

        except Exception as exc:
            logger.error(
                "Vision processing failed: %s",
                exc,
                exc_info=True,
            )

            self.last_vision_error = str(exc)

            if self.config.graceful_degradation:
                self.vision_available = False
                self.vision = None

                return (
                    self._create_empty_vision_result()
                )

            raise RuntimeError(
                f"Vision processing failed: {exc}"
            ) from exc

    # ==================================================================
    # Fusion
    # ==================================================================

    def _calculate_composite_threat(
        self,
        vision_result: Dict[str, Any],
        audio_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert sensor outputs into FusionEngine feature contracts
        and perform centralized fusion.
        """

        current_time = time.time()

        # --------------------------------------------------------------
        # Audio freshness.
        # --------------------------------------------------------------
        audio_timestamp = audio_result.get(
            "timestamp"
        )

        audio_is_fresh = (
            self.audio_available
            and audio_timestamp is not None
            and (
                current_time
                - float(audio_timestamp)
            )
            <= self.config.audio_max_age_sec
        )

        audio_features: Optional[
            AudioFeatures
        ]

        if audio_is_fresh:
            audio_features = AudioFeatures(
                dominant_class=audio_result.get(
                    "label"
                ),
                dominant_conf=float(
                    audio_result.get(
                        "confidence",
                        0.0,
                    )
                ),
                threat_score=float(
                    audio_result.get(
                        "threat_score",
                        0.0,
                    )
                ),
                top_k=audio_result.get(
                    "top_predictions",
                    [],
                ),
            )
        else:
            audio_features = None

        # --------------------------------------------------------------
        # Vision features.
        #
        # VisionDetector provides raw contextual information.
        # FusionEngine computes the contextual vision score.
        # --------------------------------------------------------------
        vision_features: Optional[
            VisionFeatures
        ]

        if self.vision_available:
            vision_features = VisionFeatures(
                person_count=int(
                    vision_result.get(
                        "persons",
                        0,
                    )
                ),
                face_count=int(
                    vision_result.get(
                        "faces",
                        0,
                    )
                ),
                other_objects=vision_result.get(
                    "objects",
                    [],
                ),
            )
        else:
            vision_features = None

        # --------------------------------------------------------------
        # Centralized fusion.
        # --------------------------------------------------------------
        fusion_result = self.fusion_engine.fuse(
            audio=audio_features,
            vision=vision_features,
            audio_available=audio_is_fresh,
            vision_available=self.vision_available,
        )

        # --------------------------------------------------------------
        # Sensor coverage.
        #
        # IMPORTANT:
        # Coverage is NOT statistical confidence.
        # --------------------------------------------------------------
        sensor_coverage = (
            int(
                fusion_result.audio_available
            )
            + int(
                fusion_result.vision_available
            )
        ) / 2.0

        return {
            "risk_label": (
                fusion_result.level.name
            ),
            "threat_score": (
                fusion_result.fused_score
            ),

            # Compatibility alias for older consumers.
            "fusion_confidence": sensor_coverage,

            "sensor_coverage": sensor_coverage,

            "audio_available": (
                fusion_result.audio_available
            ),
            "vision_available": (
                fusion_result.vision_available
            ),

            "audio_contribution": (
                fusion_result.audio_contrib
            ),
            "vision_contribution": (
                fusion_result.vision_contrib
            ),

            "raw_audio_score": (
                fusion_result.raw_audio_score
            ),
            "raw_vision_score": (
                fusion_result.raw_vision_score
            ),

            "effective_audio_weight": (
                fusion_result.effective_audio_weight
            ),
            "effective_vision_weight": (
                fusion_result.effective_vision_weight
            ),

            "reasoning": "",
        }

    # ==================================================================
    # Reasoning
    # ==================================================================

    def _generate_reasoning(
        self,
        vision_result: Dict[str, Any],
        audio_result: Dict[str, Any],
        composite: Dict[str, Any],
    ) -> str:
        """
        Generate conservative human-readable evidence reasoning.

        This function describes the evidence used by the system.

        It intentionally does NOT claim:
            - generic YOLO is weapon detection
            - heuristic scores are calibrated probabilities
            - audio AST is a validated emergency-event detector
        """

        risk_label = composite[
            "risk_label"
        ]

        parts: List[str] = []

        # --------------------------------------------------------------
        # Audio evidence.
        # --------------------------------------------------------------
        if composite.get(
            "audio_available",
            False,
        ):
            audio_label = audio_result.get(
                "label",
                "Unknown",
            )

            audio_confidence = float(
                audio_result.get(
                    "confidence",
                    0.0,
                )
            )

            audio_threat = float(
                audio_result.get(
                    "threat_score",
                    0.0,
                )
            )

            if (
                audio_label
                and audio_label != "Unknown"
            ):
                parts.append(
                    f"audio={audio_label} "
                    f"(classifier confidence="
                    f"{audio_confidence:.1%}, "
                    f"threat score="
                    f"{audio_threat:.1%})"
                )

        # --------------------------------------------------------------
        # Vision contextual evidence.
        # --------------------------------------------------------------
        if composite.get(
            "vision_available",
            False,
        ):
            persons = int(
                vision_result.get(
                    "persons",
                    0,
                )
            )

            faces = int(
                vision_result.get(
                    "faces",
                    0,
                )
            )

            if persons == 1:
                parts.append(
                    "1 person visible"
                )
            elif persons > 1:
                parts.append(
                    f"{persons} persons detected"
                )

            if faces == 1:
                parts.append(
                    "1 face detected"
                )
            elif faces > 1:
                parts.append(
                    f"{faces} faces detected"
                )

        # --------------------------------------------------------------
        # Sensor availability.
        # --------------------------------------------------------------
        unavailable: List[str] = []

        if not composite.get(
            "audio_available",
            False,
        ):
            unavailable.append("audio")

        if not composite.get(
            "vision_available",
            False,
        ):
            unavailable.append("vision")

        if unavailable:
            parts.append(
                "unavailable sensor(s): "
                + ", ".join(unavailable)
            )

        # --------------------------------------------------------------
        # Final classification explanation.
        # --------------------------------------------------------------
        if risk_label == "THREATENING":
            prefix = "THREAT"
        elif risk_label == "SUSPICIOUS":
            prefix = "SUSPICIOUS"
        else:
            prefix = "SAFE"

        if parts:
            return (
                f"{prefix}: "
                + " + ".join(parts)
            )

        if risk_label == "BENIGN":
            return (
                "SAFE: no active threat evidence"
            )

        return (
            f"{prefix}: insufficient sensor evidence"
        )

    # ==================================================================
    # Alerts
    # ==================================================================

    def _check_and_queue_alert(
        self,
        frame_id: int,
        timestamp: float,
        composite: Dict[str, Any],
        reasoning: str,
    ) -> None:
        """Queue suspicious/threatening events with cooldown."""

        risk_label = composite[
            "risk_label"
        ]

        if risk_label not in {
            "SUSPICIOUS",
            "THREATENING",
        }:
            return

        # Prevent repeated alerts for the same ongoing event.
        if (
            timestamp - self.last_alert_time
            < self.config.alert_cooldown_seconds
        ):
            return

        alert = Alert(
            timestamp=timestamp,
            frame_id=frame_id,
            risk_label=risk_label,
            threat_score=float(
                composite["threat_score"]
            ),
            reasoning=reasoning,
        )

        with self.lock:
            self.alerts.append(alert)

        self.last_alert_time = timestamp

        logger.warning(
            "[ALERT] Frame %d: %s "
            "(score=%.2f) - %s",
            frame_id,
            risk_label,
            composite["threat_score"],
            reasoning,
        )

    # ==================================================================
    # Performance
    # ==================================================================

    def _update_performance_metrics(
        self,
    ) -> None:
        """Update frame throughput metrics."""

        self.frame_count += 1

        now = time.time()
        elapsed = (
            now - self.last_stats_time
        )

        if elapsed >= 1.0:
            self.fps = (
                self.frame_count
                / elapsed
            )

            self.frame_count = 0
            self.last_stats_time = now

            logger.debug(
                "Performance: FPS=%.1f, "
                "Vision latency=%.1fms, "
                "Audio age=%.1fms, "
                "Alerts queued=%d",
                self.fps,
                self.vision_latency_ms,
                self.audio_age_ms,
                len(self.alerts),
            )

    # ==================================================================
    # Alert retrieval
    # ==================================================================

    def get_recent_alerts(
        self,
        count: int = 10,
    ) -> List[Alert]:
        """
        Return the most recent alerts.

        Non-positive counts return an empty list.
        """

        if count <= 0:
            return []

        with self.lock:
            alerts_list = list(
                self.alerts
            )

        return sorted(
            alerts_list[-count:],
            key=lambda alert: alert.timestamp,
            reverse=True,
        )

    # ==================================================================
    # Status / Health
    # ==================================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """Return current system state and telemetry."""

        health = (
            self._get_sensor_health()
        )

        return {
            "vision_available": (
                self.vision_available
            ),
            "audio_available": (
                self.audio_available
            ),

            "vision_status": (
                health["vision_status"]
            ),
            "audio_status": (
                health["audio_status"]
            ),

            "last_vision_error": (
                health["last_vision_error"]
            ),
            "last_audio_error": (
                health["last_audio_error"]
            ),

            "fps": self.fps,

            "vision_latency_ms": (
                self.vision_latency_ms
            ),

            # This is observation age,
            # not AST inference latency.
            "audio_age_ms": (
                self.audio_age_ms
            ),

            "alerts_total": (
                len(self.alerts)
            ),

            "audio_weight": (
                self.audio_weight
            ),
            "vision_weight": (
                self.vision_weight
            ),

            "graceful_degradation": (
                self.config.graceful_degradation
            ),

            "sensor_coverage": (
                (
                    int(
                        self.vision_available
                    )
                    + int(
                        self.audio_available
                    )
                )
                / 2.0
            ),
        }

    def _get_sensor_health(
        self,
    ) -> Dict[str, Any]:
        """Return a stable health summary for both sensors."""

        vision_health: Dict[
            str,
            Any,
        ] = {}

        audio_health: Dict[
            str,
            Any,
        ] = {}

        # --------------------------------------------------------------
        # Vision health.
        # --------------------------------------------------------------
        if (
            self.vision_available
            and self.vision is not None
        ):
            get_health = getattr(
                self.vision,
                "get_health",
                None,
            )

            if get_health is not None:
                try:
                    vision_health = (
                        get_health() or {}
                    )
                except Exception as exc:
                    logger.warning(
                        "Vision health query failed: %s",
                        exc,
                    )

        # --------------------------------------------------------------
        # Audio health.
        # --------------------------------------------------------------
        if (
            self.audio_available
            and self.audio is not None
        ):
            get_health = getattr(
                self.audio,
                "get_health",
                None,
            )

            if get_health is not None:
                try:
                    audio_health = (
                        get_health() or {}
                    )
                except Exception as exc:
                    logger.warning(
                        "Audio health query failed: %s",
                        exc,
                    )

        return {
            "vision_status": vision_health.get(
                "status",
                (
                    "OK"
                    if self.vision_available
                    else "ERROR"
                ),
            ),
            "audio_status": audio_health.get(
                "status",
                (
                    "OK"
                    if self.audio_available
                    else "ERROR"
                ),
            ),
            "last_vision_error": (
                self.last_vision_error
                or vision_health.get(
                    "last_error"
                )
            ),
            "last_audio_error": (
                self.last_audio_error
                or audio_health.get(
                    "last_error"
                )
            ),
        }

    # ==================================================================
    # Shutdown
    # ==================================================================

    def stop(self) -> None:
        """
        Cleanly shut down audio processing.

        Safe to call multiple times.
        """

        logger.info(
            "Stopping SentinelVisionUnified..."
        )

        if (
            self.audio_available
            and self.audio is not None
        ):
            try:
                self.audio.stop()

                logger.info(
                    "Audio detector stopped"
                )

            except Exception as exc:
                logger.error(
                    "Error stopping audio detector: %s",
                    exc,
                )

        logger.info(
            "Shutdown complete. "
            "Final stats: %d alerts queued, "
            "%.1f avg FPS",
            len(self.alerts),
            self.fps,
        )

    # ==================================================================
    # Empty sensor results
    # ==================================================================

    @staticmethod
    def _create_empty_vision_result(
    ) -> Dict[str, Any]:
        """Create an explicit unavailable/error vision result."""

        return {
            "annotated_frame": None,
            "persons": 0,
            "faces": 0,
            "objects": [],
            "fps": 0.0,
            "health_status": "ERROR",
        }

    @staticmethod
    def _create_empty_audio_result(
    ) -> Dict[str, Any]:
        """Create an explicit unavailable/error audio result."""

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
        }


# ======================================================================
# Direct module smoke test
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(name)s - "
            "%(levelname)s - %(message)s"
        ),
    )

    logger = logging.getLogger(__name__)

    logger.info(
        "Testing SentinelVisionUnified initialization..."
    )

    detector: Optional[
        SentinelVisionUnified
    ] = None

    try:
        # Fusion weights belong to FusionConfig.
        config = SentinelUnifiedConfig(
            graceful_degradation=True,
            fusion=FusionConfig(
                audio_weight=0.60,
                vision_weight=0.40,
            ),
        )

        detector = SentinelVisionUnified(
            vision_config=VisionDetectorConfig(),
            audio_config=AudioDetectorConfig(),
            config=config,
        )

        logger.info(
            "✓ Unified detector initialized successfully"
        )

        logger.info(
            "Status: %s",
            detector.get_status(),
        )

    except Exception as exc:
        logger.error(
            "✗ Initialization failed: %s",
            exc,
            exc_info=True,
        )

    finally:
        if detector is not None:
            detector.stop()