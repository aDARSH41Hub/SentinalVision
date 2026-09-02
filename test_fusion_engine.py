"""Deterministic unit tests for the centralized FusionEngine."""

import unittest

from fusion_engine import AudioFeatures, FusionConfig, FusionEngine, ThreatLevel, VisionFeatures


class TestFusionConfig(unittest.TestCase):
    def test_default_weights_normalize(self):
        config = FusionConfig()
        audio_w, vision_w = config.normalized_weights
        self.assertAlmostEqual(audio_w, 0.60)
        self.assertAlmostEqual(vision_w, 0.40)
        self.assertAlmostEqual(audio_w + vision_w, 1.0)

    def test_threshold_order(self):
        config = FusionConfig()
        self.assertLess(config.suspicious_threshold, config.threatening_threshold)


class TestVisionScoring(unittest.TestCase):
    def setUp(self):
        self.engine = FusionEngine(FusionConfig(ema_alpha=1.0))

    def test_empty_scene_is_zero(self):
        self.assertEqual(
            self.engine.compute_vision_score(VisionFeatures(0, 0)),
            0.0,
        )

    def test_medium_crowd_adds_context(self):
        score = self.engine.compute_vision_score(VisionFeatures(3, 3))
        self.assertAlmostEqual(score, 0.08)

    def test_high_crowd_low_face_ratio_adds_context(self):
        score = self.engine.compute_vision_score(VisionFeatures(5, 0))
        self.assertAlmostEqual(score, 0.35)

    def test_no_weapon_logic_exists(self):
        score = self.engine.compute_vision_score(
            VisionFeatures(1, 1, [("knife", 0.99), ("gun", 0.99)])
        )
        self.assertEqual(score, 0.0)


class TestAvailabilityAndFusion(unittest.TestCase):
    def setUp(self):
        self.engine = FusionEngine(FusionConfig(ema_alpha=1.0))

    def test_both_sensors_use_configured_weights(self):
        result = self.engine.fuse(
            AudioFeatures("Alarm", 0.8, 0.8),
            VisionFeatures(0, 0),
        )
        self.assertAlmostEqual(result.fused_score, 0.48)
        self.assertEqual(result.level, ThreatLevel.SUSPICIOUS)
        self.assertTrue(result.audio_available)
        self.assertTrue(result.vision_available)

    def test_audio_only_renormalizes_to_one(self):
        result = self.engine.fuse(
            AudioFeatures("Alarm", 0.8, 0.5),
            None,
            audio_available=True,
            vision_available=False,
        )
        self.assertAlmostEqual(result.fused_score, 0.5)
        self.assertAlmostEqual(result.effective_audio_weight, 1.0)
        self.assertAlmostEqual(result.effective_vision_weight, 0.0)

    def test_vision_only_renormalizes_to_one(self):
        result = self.engine.fuse(
            None,
            VisionFeatures(5, 0),
            audio_available=False,
            vision_available=True,
        )
        self.assertAlmostEqual(result.fused_score, 0.35)
        self.assertAlmostEqual(result.effective_audio_weight, 0.0)
        self.assertAlmostEqual(result.effective_vision_weight, 1.0)

    def test_missing_both_is_benign_without_claiming_confidence(self):
        result = self.engine.fuse(None, None, False, False)
        self.assertEqual(result.level, ThreatLevel.BENIGN)
        self.assertEqual(result.fused_score, 0.0)
        self.assertFalse(result.audio_available)
        self.assertFalse(result.vision_available)

    def test_thresholds(self):
        benign = self.engine.fuse(AudioFeatures("x", 1.0, 0.2), None, True, False)
        suspicious = self.engine.fuse(AudioFeatures("x", 1.0, 0.3), None, True, False)
        threatening = self.engine.fuse(AudioFeatures("x", 1.0, 1.0), None, True, False)
        self.assertEqual(benign.level, ThreatLevel.BENIGN)
        self.assertEqual(suspicious.level, ThreatLevel.SUSPICIOUS)
        self.assertEqual(threatening.level, ThreatLevel.THREATENING)

    def test_ema_threat_event_reset(self):
        engine = FusionEngine(FusionConfig(ema_alpha=0.30, ema_reset_on_threat=True))
        engine.fuse(AudioFeatures("x", 1.0, 0.0), None, True, False)
        result = engine.fuse(AudioFeatures("x", 1.0, 0.8), None, True, False)
        self.assertAlmostEqual(result.fused_score, 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
