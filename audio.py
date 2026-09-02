"""Real-time AudioSet-tuned AST audio classification with threat scoring."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from transformers import AutoModelForAudioClassification, AutoProcessor

logger = logging.getLogger(__name__)


class AudioDetectorConfig:
    MODEL_NAME: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    SAMPLE_RATE: int = 16000
    BUFFER_SECONDS: int = 2
    MIC_DEVICE: int = 0
    BLOCKSIZE: int = 1600
    INFERENCE_INTERVAL_SECONDS: float = 1.0
    SILENCE_THRESHOLD: float = 0.01

    THREAT_SCORE_THREATENING: float = 0.70
    THREAT_SCORE_SUSPICIOUS: float = 0.30

    THREAT_CLASSES: Dict[str, float] = {
        "Gunshot, gunfire": 1.00,
        "Machine gun": 1.00,
        "Fusillade": 1.00,
        "Explosion": 1.00,
        "Glass": 0.75,
        "Shatter": 0.90,
        "Smash, crash": 0.70,
        "Breaking": 0.70,
        "Bang": 0.60,
        "Screaming": 0.85,
        "Yell": 0.45,
        "Shout": 0.40,
        "Battle cry": 0.50,
        "Children shouting": 0.40,
        "Crying, sobbing": 0.30,
        "Wail, moan": 0.30,
        "Fire alarm": 0.70,
        "Smoke detector, smoke alarm": 0.70,
        "Siren": 0.60,
        "Civil defense siren": 0.75,
        "Police car (siren)": 0.65,
        "Ambulance (siren)": 0.60,
        "Fire engine, fire truck (siren)": 0.65,
    }


class AudioDetector:
    """Background AST detector with explicit timestamps, timing and health."""

    def __init__(
        self,
        config: Optional[AudioDetectorConfig] = None,
        device: str = "cpu",
        mic_device: Optional[int] = None,
    ) -> None:
        self.config = config or AudioDetectorConfig()
        self.device = torch.device(device)
        self.mic_device = self.config.MIC_DEVICE if mic_device is None else mic_device

        if self.mic_device < 0:
            raise ValueError(f"Invalid microphone device index: {self.mic_device}")

        try:
            logger.info("Loading AST processor from %s", self.config.MODEL_NAME)
            self.processor = AutoProcessor.from_pretrained(self.config.MODEL_NAME)
            self.model = AutoModelForAudioClassification.from_pretrained(self.config.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            logger.error("Failed to load AST model: %s", exc, exc_info=True)
            raise RuntimeError(f"Could not load AST model: {exc}") from exc

        self.buffer_size = int(self.config.SAMPLE_RATE * self.config.BUFFER_SECONDS)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.lock = threading.Lock()
        self.running = True

        self.health_status = "OK"
        self.last_error: Optional[str] = None
        self.last_inference_ms: float = 0.0

        self.last_result: Dict[str, Any] = self._create_empty_result()

        try:
            self.stream = sd.InputStream(
                samplerate=self.config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=self.mic_device,
                blocksize=self.config.BLOCKSIZE,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            logger.error("Failed to open microphone: %s", exc, exc_info=True)
            raise RuntimeError(
                f"Could not open microphone device {self.mic_device}: {exc}"
            ) from exc

        self.thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="AudioInferenceThread",
        )
        self.thread.start()

    def _create_empty_result(self) -> Dict[str, Any]:
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

    def _set_health(self, status: str, error: Optional[str] = None) -> None:
        self.health_status = status
        self.last_error = error
        if error:
            logger.warning(error)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        if status:
            self._set_health("DEGRADED", f"Audio stream status: {status}")

        if indata.ndim == 1:
            new_audio = indata
        else:
            new_audio = indata[:, 0]

        new_audio = np.asarray(new_audio, dtype=np.float32)
        with self.lock:
            if len(new_audio) >= self.buffer_size:
                self.audio_buffer = new_audio[-self.buffer_size:].copy()
            else:
                self.audio_buffer = np.roll(self.audio_buffer, -len(new_audio))
                self.audio_buffer[-len(new_audio):] = new_audio

    def _inference_loop(self) -> None:
        while self.running:
            time.sleep(self.config.INFERENCE_INTERVAL_SECONDS)
            with self.lock:
                audio = self.audio_buffer.copy()

            if len(audio) < self.buffer_size:
                continue

            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio ** 2)))

            if peak < self.config.SILENCE_THRESHOLD:
                timestamp = time.time()
                with self.lock:
                    self.last_result = {
                        "label": "Silence",
                        "confidence": 1.0,
                        "threat_score": 0.0,
                        "risk_label": "BENIGN",
                        "top_predictions": [],
                        "threat_predictions": [],
                        "peak": peak,
                        "rms": rms,
                        "timestamp": timestamp,
                        "inference_latency_ms": 0.0,
                    }
                self._set_health("OK")
                continue

            try:
                self._run_inference(audio, peak, rms)
            except Exception as exc:
                message = f"AST inference failed: {exc}"
                logger.error(message, exc_info=True)
                self._set_health("ERROR", message)
                with self.lock:
                    self.last_result = {
                        **self._create_empty_result(),
                        "timestamp": None,
                    }

    def _run_inference(self, audio: np.ndarray, peak: float, rms: float) -> None:
        start = time.perf_counter()
        inputs = self.processor(
            audio,
            sampling_rate=self.config.SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        k = min(10, probabilities.shape[0])
        top_values, top_indices = torch.topk(probabilities, k=k)

        top_predictions: List[Tuple[str, float]] = []
        for probability, index in zip(top_values, top_indices):
            class_id = int(index)
            label = self.model.config.id2label[class_id]
            top_predictions.append((label, float(probability)))

        threat_predictions: List[Tuple[str, float, float, float]] = []
        weighted_scores: List[float] = []
        for label, confidence in top_predictions:
            weight = self.config.THREAT_CLASSES.get(label)
            if weight is None:
                continue
            weighted_score = confidence * weight
            threat_predictions.append((label, confidence, weight, weighted_score))
            weighted_scores.append(weighted_score)

        threat_score = min(sum(weighted_scores), 1.0)
        if threat_score >= self.config.THREAT_SCORE_THREATENING:
            risk_label = "THREATENING"
        elif threat_score >= self.config.THREAT_SCORE_SUSPICIOUS:
            risk_label = "SUSPICIOUS"
        else:
            risk_label = "BENIGN"

        label, confidence = top_predictions[0] if top_predictions else ("Unknown", 0.0)
        latency_ms = (time.perf_counter() - start) * 1000.0
        timestamp = time.time()

        with self.lock:
            self.last_result = {
                "label": label,
                "confidence": confidence,
                "threat_score": threat_score,
                "risk_label": risk_label,
                "top_predictions": top_predictions,
                "threat_predictions": threat_predictions,
                "peak": peak,
                "rms": rms,
                "timestamp": timestamp,
                "inference_latency_ms": latency_ms,
            }

        self.last_inference_ms = latency_ms
        self._set_health("OK")

    def get_result(self) -> Dict[str, Any]:
        with self.lock:
            return self.last_result.copy()

    def get_health(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "status": self.health_status,
                "last_error": self.last_error,
                "last_inference_ms": self.last_inference_ms,
            }

    def get_audio_buffer(self) -> np.ndarray:
        """Return a copy of the current 2-second rolling audio buffer."""
        with self.lock:
            return self.audio_buffer.copy()

    def save_audio_buffer(self, path: str) -> None:
        """Save the current 2-second rolling buffer as a WAV file."""
        audio = self.get_audio_buffer()
        sf.write(path, audio, self.config.SAMPLE_RATE)

    def stop(self) -> None:
        self.running = False
        stream = getattr(self, "stream", None)
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                logger.error("Error closing stream: %s", exc, exc_info=True)
        thread = getattr(self, "thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
