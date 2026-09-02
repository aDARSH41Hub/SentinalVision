"""Orchestration tests with model/sensor dependencies mocked."""

import threading
import time
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np

from sentinel_vision import Alert, SentinelUnifiedConfig, SentinelVisionUnified


class Base(unittest.TestCase):
    def make_detector(
        self,
        *,
        vision_failure=None,
        audio_failure=None,
        vision_process_return=None,
        audio_result=None,
        graceful_degradation=True,
        audio_weight=None,
        vision_weight=None,
    ):
        self.vpatch = patch("sentinel_vision.VisionDetector")
        self.apatch = patch("sentinel_vision.AudioDetector")
        self.vcls = self.vpatch.start()
        self.acls = self.apatch.start()
        self.addCleanup(self.vpatch.stop)
        self.addCleanup(self.apatch.stop)

        if vision_failure is not None:
            self.vcls.side_effect = vision_failure
        else:
            v = self.vcls.return_value
            v.process.return_value = vision_process_return or (
                np.zeros((480, 640, 3), dtype=np.uint8),
                {"person_count": 0, "face_count": 0, "objects": [], "fps": 10.0,
                 "timings": {"yolo_ms": 2.0, "haar_ms": 1.0, "vision_total_ms": 3.0}},
            )
            v.get_health.return_value = {"status": "OK", "last_error": None}

        if audio_failure is not None:
            self.acls.side_effect = audio_failure
        else:
            a = self.acls.return_value
            a.get_result.return_value = audio_result or {
                "label": "Silence", "confidence": 1.0, "threat_score": 0.0,
                "risk_label": "BENIGN", "top_predictions": [], "threat_predictions": [],
                "peak": 0.0, "rms": 0.0, "timestamp": time.time(), "inference_latency_ms": 0.0,
            }
            a.get_health.return_value = {"status": "OK", "last_error": None, "last_inference_ms": 0.0}

        self.detector = SentinelVisionUnified(
            audio_weight=audio_weight,
            vision_weight=vision_weight,
            config=SentinelUnifiedConfig(graceful_degradation=graceful_degradation, fusion=__import__("fusion_engine").FusionConfig(ema_alpha=1.0)),
        )
        self.addCleanup(self.detector.stop)
        return self.detector


class TestArchitecture(Base):
    def test_config_is_single_fusion_source(self):
        d = self.make_detector()
        self.assertIs(d.config.fusion, d.fusion_engine.config)
        self.assertAlmostEqual(d.fusion_engine.config.suspicious_threshold, 0.30)
        self.assertAlmostEqual(d.fusion_engine.config.threatening_threshold, 0.70)

    def test_no_legacy_vision_threat_helper(self):
        self.make_detector()
        self.assertFalse(hasattr(self.detector, "_calculate_vision_threat"))

    def test_weight_override(self):
        d = self.make_detector(audio_weight=0.7, vision_weight=0.3)
        self.assertAlmostEqual(d.config.fusion.audio_weight, 0.7)
        self.assertAlmostEqual(d.fusion_engine.audio_weight, 0.7)


class TestFusion(Base):
    def test_weighted_composite(self):
        d = self.make_detector(
            audio_result={
                "label": "Alarm", "confidence": 0.8, "threat_score": 0.8,
                "timestamp": time.time(), "top_predictions": [],
            }
        )
        result = d._calculate_composite_threat(
            {"persons": 0, "faces": 0, "objects": [], "health_status": "OK"},
            d.audio.get_result(),
        )
        self.assertAlmostEqual(result["threat_score"], 0.48)
        self.assertEqual(result["risk_label"], "SUSPICIOUS")
        self.assertEqual(result["sensor_coverage"], 1.0)

    def test_stale_audio_is_not_used(self):
        d = self.make_detector(
            audio_result={
                "label": "Gunshot, gunfire", "confidence": 0.99, "threat_score": 1.0,
                "timestamp": time.time() - 100, "top_predictions": [],
            }
        )
        result = d._calculate_composite_threat(
            {"persons": 0, "faces": 0, "objects": [], "health_status": "OK"},
            d.audio.get_result(),
        )
        self.assertFalse(result["audio_available"])
        self.assertEqual(result["sensor_coverage"], 0.5)


class TestHealth(Base):
    def test_vision_failure_exposed(self):
        d = self.make_detector(vision_failure=RuntimeError("vision init failed"))
        self.assertFalse(d.vision_available)
        status = d.get_status()
        self.assertEqual(status["vision_status"], "ERROR")
        self.assertIn("vision init failed", status["last_vision_error"])

    def test_audio_failure_exposed(self):
        d = self.make_detector(audio_failure=RuntimeError("audio init failed"))
        self.assertFalse(d.audio_available)
        status = d.get_status()
        self.assertEqual(status["audio_status"], "ERROR")
        self.assertIn("audio init failed", status["last_audio_error"])

    def test_both_failure_raises(self):
        with self.assertRaises(RuntimeError):
            self.make_detector(
                vision_failure=RuntimeError("vision"),
                audio_failure=RuntimeError("audio"),
            )

    def test_strict_mode_raises(self):
        with self.assertRaises(RuntimeError):
            self.make_detector(
                vision_failure=RuntimeError("vision"),
                graceful_degradation=False,
            )


class TestReasoningAndAlerts(Base):
    def test_reasoning_does_not_claim_confidence_is_threat_probability(self):
        d = self.make_detector()
        text = d._generate_reasoning(
            {"persons": 2, "faces": 1},
            {"label": "Alarm", "confidence": 0.95, "threat_score": 0.6},
            {"risk_label": "SUSPICIOUS", "threat_score": 0.5, "audio_available": True},
        )
        self.assertIn("classifier confidence", text)
        self.assertIn("threat score", text)

    def test_alert_cooldown(self):
        d = self.make_detector()
        base = time.time()
        composite = {"risk_label": "SUSPICIOUS", "threat_score": 0.5}
        d._check_and_queue_alert(1, base, composite, "Alert")
        d._check_and_queue_alert(2, base + 0.5, composite, "Alert")
        d._check_and_queue_alert(3, base + 2.1, composite, "Alert")
        alerts = d.get_recent_alerts(10)
        self.assertEqual(len(alerts), 2)
        self.assertEqual([a.frame_id for a in alerts], [3, 1])

    def test_zero_alert_count(self):
        d = self.make_detector()
        self.assertEqual(d.get_recent_alerts(0), [])


class TestProcessFrame(Base):
    def test_process_frame_contract(self):
        d = self.make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = d.process_frame(frame, frame_id=42)
        self.assertEqual(result["frame_id"], 42)
        for key in ("timestamp", "vision", "audio", "composite", "performance", "health", "annotated_frame"):
            self.assertIn(key, result)
        self.assertIn("audio_age_ms", result["performance"])
        self.assertNotIn("audio_latency_ms", result["performance"])
        self.assertIn("sensor_coverage", result["composite"])

    def test_invalid_frame_rejected(self):
        d = self.make_detector()
        with self.assertRaises(ValueError):
            d.process_frame(np.zeros((10, 10), dtype=np.uint8))


class TestDatatypes(unittest.TestCase):
    def test_alert(self):
        alert = Alert(time.time(), 7, "THREATENING", 0.85, "Threat")
        self.assertEqual(alert.frame_id, 7)

    def test_lock_exercise(self):
        lock = threading.Lock()
        value = {"count": 0}

        def worker():
            for _ in range(100):
                with lock:
                    value["count"] += 1

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(value["count"], 300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
