import json
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[2]


class ScientificValidationAlignmentTest(unittest.TestCase):
    def test_inherits_delivered_runtime_settings(self):
        pairs = json.loads(
            Path(__file__).with_name("validation_source_pairs.json").read_text()
        )
        self.assertEqual(len(pairs), 136)
        for validation, source in pairs:
            actual = yaml.safe_load((ROOT / validation).read_text())[
                "evaluation_configs"
            ][0]
            expected = yaml.safe_load((ROOT / source).read_text())[
                "evaluation_configs"
            ][0]
            for key in (
                "usd_name",
                "robots",
                "physics_scene_config",
                "physics_dt",
                "rendering_dt",
                "warmup_steps",
                "preprocess_config",
                "object_config",
            ):
                with self.subTest(config=validation, field=key):
                    self.assertEqual(actual.get(key), expected.get(key))
