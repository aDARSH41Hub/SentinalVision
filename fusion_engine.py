"""Centralized multimodal evidence fusion for SentinelVision.

Phase-1 policy:
- Audio and vision produce evidence/features; this module owns decision scoring.
- Sensor availability is explicit and missing sensors are never treated as evidence of safety.
- Vision scoring is intentionally contextual: crowd density + low face/person ratio.
- Threat thresholds and fusion weights come from one FusionConfig.
- EMA is used for temporal smoothing, with an optional threat-event reset for response speed.

This is a heuristic baseline for research prototyping, not a calibrated probability model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ThreatLevel(IntEnum):
    BENIGN = 0
    SUSPICIOUS = 1
    THREATENING = 2


@dataclass(frozen=True)
class FusionConfig:
    """Single authoritative configuration for fusion behavior."""

    audio_weight: float = 0.60
    vision_weight: float = 0.40

    benign_threshold: float = 0.00
    suspicious_threshold: float = 0.30
    threatening_threshold: float = 0.70

    ema_alpha: float = 0.30
    ema_reset_on_threat: bool = True
    ema_reset_threshold: float = 0.50

    # Contextual vision scoring constants.
    people_medium: int = 3
    people_high: int = 5
    people_very_high: int = 8
    score_medium_crowd: float = 0.08
    score_high_crowd: float = 0.20
    score_very_high_crowd: float = 0.35
    low_face_ratio_threshold: float = 0.30
    low_face_ratio_bonus: float = 0.15
    max_vision_score: float = 0.60

    def __post_init__(self) -> None:
        if self.audio_weight < 0 or self.vision_weight < 0:
            raise ValueError("Fusion weights must be non-negative")
        if self.audio_weight + self.vision_weight <= 0:
            raise ValueError("At least one fusion weight must be positive")
        if not 0.0 <= self.benign_threshold <= self.suspicious_threshold <= self.threatening_threshold <= 1.0:
            raise ValueError(
                "Thresholds must satisfy 0 <= benign <= suspicious <= threatening <= 1"
            )
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= self.ema_reset_threshold <= 1.0:
            raise ValueError("ema_reset_threshold must be in [0, 1]")
        if self.people_medium < 1 or self.people_high < self.people_medium or self.people_very_high < self.people_high:
            raise ValueError("Crowd thresholds must be increasing positive integers")
        for value in (
            self.score_medium_crowd,
            self.score_high_crowd,
            self.score_very_high_crowd,
            self.low_face_ratio_bonus,
        ):
            if value < 0:
                raise ValueError("Vision score components must be non-negative")
        if not 0.0 < self.low_face_ratio_threshold <= 1.0:
            raise ValueError("low_face_ratio_threshold must be in (0, 1]")
        if self.max_vision_score <= 0:
            raise ValueError("max_vision_score must be positive")

    @property
    def normalized_weights(self) -> Tuple[float, float]:
        total = self.audio_weight + self.vision_weight
        return self.audio_weight / total, self.vision_weight / total


@dataclass(frozen=True)
class AudioFeatures:
    dominant_class: Optional[str]
    dominant_conf: float
    threat_score: float
    top_k: List[Tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class VisionFeatures:
    person_count: int
    face_count: int
    other_objects: List[Tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class FusionResult:
    level: ThreatLevel
    fused_score: float
    audio_contrib: float
    vision_contrib: float
    raw_audio_score: float
    raw_vision_score: float
    audio_available: bool
    vision_available: bool
    effective_audio_weight: float
    effective_vision_weight: float
    explanation: str


class FusionEngine:
    """Combine sensor evidence into one auditable risk assessment."""

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()
        self.audio_weight, self.vision_weight = self.config.normalized_weights
        self.ema_alpha = self.config.ema_alpha
        self._ema_score: Optional[float] = None

    @property
    def ema_score(self) -> Optional[float]:
        return self._ema_score

    def reset(self) -> None:
        """Reset temporal state."""
        self._ema_score = None

    def compute_vision_score(self, features: VisionFeatures) -> float:
        """Compute weak contextual visual evidence.

        Important: this function intentionally does *not* score weapons or other
        unsupported threat classes. Generic COCO-style object detection is treated
        as contextual information only.
        """
        people = max(0, int(features.person_count))
        faces = max(0, int(features.face_count))

        score = 0.0
        if people >= self.config.people_very_high:
            score += self.config.score_very_high_crowd
        elif people >= self.config.people_high:
            score += self.config.score_high_crowd
        elif people >= self.config.people_medium:
            score += self.config.score_medium_crowd

        if people > 0:
            face_ratio = min(1.0, faces / people)
            if face_ratio < self.config.low_face_ratio_threshold:
                score += self.config.low_face_ratio_bonus

        return float(np.clip(score, 0.0, self.config.max_vision_score))

    def _get_effective_weights(
        self,
        audio_available: bool,
        vision_available: bool,
    ) -> Tuple[float, float]:
        if audio_available and vision_available:
            return self.audio_weight, self.vision_weight
        if audio_available:
            return 1.0, 0.0
        if vision_available:
            return 0.0, 1.0
        return 0.0, 0.0

    def classify(self, score: float) -> ThreatLevel:
        score = float(np.clip(score, 0.0, 1.0))
        if score >= self.config.threatening_threshold:
            return ThreatLevel.THREATENING
        if score >= self.config.suspicious_threshold:
            return ThreatLevel.SUSPICIOUS
        return ThreatLevel.BENIGN

    def fuse(
        self,
        audio: Optional[AudioFeatures],
        vision: Optional[VisionFeatures],
        audio_available: bool = True,
        vision_available: bool = True,
    ) -> FusionResult:
        """Fuse currently available sensor evidence."""
        if audio_available and audio is not None:
            audio_score = float(np.clip(audio.threat_score, 0.0, 1.0))
        else:
            audio_score = 0.0
            audio_available = False

        if vision_available and vision is not None:
            vision_score = self.compute_vision_score(vision)
        else:
            vision_score = 0.0
            vision_available = False

        audio_w, vision_w = self._get_effective_weights(audio_available, vision_available)
        raw_score = float(
            np.clip(
                audio_w * audio_score + vision_w * vision_score,
                0.0,
                1.0,
            )
        )

        previous = self._ema_score
        if (
            self.config.ema_reset_on_threat
            and previous is not None
            and previous < self.config.suspicious_threshold
            and raw_score >= self.config.ema_reset_threshold
        ):
            self._ema_score = raw_score
            logger.debug(
                "EMA reset on threat event: previous=%.3f raw=%.3f",
                previous,
                raw_score,
            )
        elif self._ema_score is None:
            self._ema_score = raw_score
        else:
            alpha = self.ema_alpha
            self._ema_score = alpha * raw_score + (1.0 - alpha) * self._ema_score

        fused_score = float(np.clip(self._ema_score, 0.0, 1.0))
        level = self.classify(fused_score)

        audio_contrib = audio_w * audio_score
        vision_contrib = vision_w * vision_score

        evidence: List[str] = []
        if audio_available and audio is not None:
            label = audio.dominant_class or "Unknown"
            evidence.append(f"audio={label} score={audio_score:.3f}")
        if vision_available and vision is not None:
            evidence.append(
                f"vision={vision_score:.3f} people={vision.person_count} faces={vision.face_count}"
            )
        if not evidence:
            evidence.append("no active sensors")

        explanation = "; ".join(evidence)
        return FusionResult(
            level=level,
            fused_score=fused_score,
            audio_contrib=float(audio_contrib),
            vision_contrib=float(vision_contrib),
            raw_audio_score=float(audio_score),
            raw_vision_score=float(vision_score),
            audio_available=audio_available,
            vision_available=vision_available,
            effective_audio_weight=float(audio_w),
            effective_vision_weight=float(vision_w),
            explanation=explanation,
        )
