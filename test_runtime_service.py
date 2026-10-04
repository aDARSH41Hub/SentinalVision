import unittest

from runtime_service import RuntimeConfig, RuntimeInferenceService, classify_probability


class TestRuntimeConfig(unittest.TestCase):
    def test_runtime_config_loads_frozen_values(self):
        config = RuntimeConfig.from_project_root("D:/SentinalVision")
        self.assertEqual(config.target_timesteps, 256)
        self.assertEqual(config.audio_dim, 128)
        self.assertEqual(config.rgb_dim, 1024)
        self.assertAlmostEqual(config.audio_alpha, 0.51)
        self.assertAlmostEqual(config.rgb_alpha, 0.49)
        self.assertAlmostEqual(config.threshold, 0.49)

    def test_probability_classification(self):
        self.assertEqual(classify_probability(0.10), "normal")
        self.assertEqual(classify_probability(0.49), "normal")
        self.assertEqual(classify_probability(0.50), "suspicious")
        self.assertEqual(classify_probability(0.80), "threat")


class TestRuntimeInferenceService(unittest.TestCase):
    def test_service_produces_runtime_result_schema(self):
        service = RuntimeInferenceService(project_root="D:/SentinalVision")
        result = service.build_decision_payload(
            audio_probability=0.12,
            rgb_probability=0.18,
            fused_probability=0.16,
            decision_label="normal",
            decision_score=0.16,
            model_name="TLF",
        )

        self.assertEqual(result["model_name"], "TLF")
        self.assertIn("fused_probability", result)
        self.assertIn("decision_label", result)
        self.assertIn("threshold", result)
        self.assertAlmostEqual(result["threshold"], 0.49)


if __name__ == "__main__":
    unittest.main(verbosity=2)
