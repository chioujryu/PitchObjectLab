"""Regression tests for RF-DETR Small presets and project-local output defaults."""

import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath

import rf_detr_acceleration
import rf_detr_runtime
import train_rf_detr_model
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "config"


def load_yaml(path: Path) -> dict:
    """Load one execution preset and require a mapping root."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"Config root must be a mapping: {path.name}")
    return data


class SmallConfigPresetTests(unittest.TestCase):
    """Keep the train/test/inference Small architecture matrix synchronized."""

    VARIANTS = {
        "small": (False, False),
        "small_p2": (True, False),
        "small_tracknet_v5": (False, True),
        "small_p2_tracknet_v5": (True, True),
    }

    def test_all_twelve_small_presets_define_the_expected_architecture(self):
        for task in ("train", "test", "inference"):
            for variant, (p2_enabled, motion_enabled) in self.VARIANTS.items():
                path = CONFIG_DIR / f"rf_detr_{task}_{variant}.yaml"
                with self.subTest(task=task, variant=variant):
                    self.assertTrue(path.is_file(), path)
                    config = load_yaml(path)
                    model = config["model"]
                    self.assertEqual(model["size"], "small")
                    self.assertIsNone(model["resolution"])
                    self.assertEqual(model["p2"]["enabled"], p2_enabled)
                    self.assertEqual(model["p2"]["projector_scale"], ["P2", "P3", "P4"])
                    self.assertEqual(model["motion"]["enabled"], motion_enabled)
                    self.assertEqual(model["motion"]["type"], "tracknet_v5")
                    if task == "train" or variant == "small":
                        self.assertEqual(model["pretrain_weights"], "default")
                    else:
                        self.assertIsNone(model["pretrain_weights"])

    def test_custom_test_and_inference_presets_require_an_explicit_checkpoint(self):
        for task in ("test", "inference"):
            stock = load_yaml(CONFIG_DIR / f"rf_detr_{task}_small.yaml")
            rf_detr_runtime._require_custom_architecture_checkpoint(stock, task)
            for variant in ("small_p2", "small_tracknet_v5", "small_p2_tracknet_v5"):
                config = load_yaml(CONFIG_DIR / f"rf_detr_{task}_{variant}.yaml")
                with self.subTest(task=task, variant=variant):
                    with self.assertRaisesRegex(ValueError, "requires an explicit matching checkpoint"):
                        rf_detr_runtime._require_custom_architecture_checkpoint(config, task)
                    config["model"]["pretrain_weights"] = "matching_checkpoint.pth"
                    rf_detr_runtime._require_custom_architecture_checkpoint(config, task)

    def test_all_execution_config_output_defaults_are_project_local(self):
        for path in sorted(CONFIG_DIR.glob("rf_detr_*.yaml")):
            with self.subTest(config=path.name):
                config = load_yaml(path)
                output = config.get("output")
                self.assertIsInstance(output, dict, path.name)
                candidates = [output.get("output_dir", ""), output.get("root", "")]
                demo = config.get("demo")
                if isinstance(demo, dict):
                    candidates.append(demo.get("output_dir", ""))
                for value in candidates:
                    if value is None or not str(value).strip():
                        continue
                    text = str(value).strip()
                    self.assertFalse(PureWindowsPath(text).is_absolute(), (path.name, text))
                    self.assertFalse(PurePosixPath(text).is_absolute(), (path.name, text))
                    self.assertNotEqual(PurePosixPath(text).parts[0], "..", (path.name, text))
                    resolved = (PROJECT_DIR / text).resolve()
                    self.assertTrue(resolved.is_relative_to(PROJECT_DIR), (path.name, text))

    def test_all_tensorrt_cache_defaults_resolve_under_the_project(self):
        expected = (PROJECT_DIR / "runs" / "rf_detr" / "tensorrt_cache").resolve()
        for path in sorted(CONFIG_DIR.glob("rf_detr_*.yaml")):
            config = load_yaml(path)
            model = config.get("model", {})
            optimization = model.get("inference_optimization", {}) if isinstance(model, dict) else {}
            tensorrt = optimization.get("tensorrt", {}) if isinstance(optimization, dict) else {}
            if not isinstance(tensorrt, dict) or "cache_dir" not in tensorrt:
                continue
            with self.subTest(config=path.name):
                self.assertEqual(tensorrt["cache_dir"], "runs/rf_detr/tensorrt_cache")
                resolved = rf_detr_acceleration.resolve_tensorrt_cache_dir(tensorrt["cache_dir"])
                self.assertEqual(resolved, expected)

    def test_temporal_training_matrix_uses_memory_safe_optimizer_defaults(self):
        for size in ("small", "medium", "large"):
            for p2 in (False, True):
                suffix = "_p2_tracknet_v5" if p2 else "_tracknet_v5"
                path = CONFIG_DIR / f"rf_detr_train_{size}{suffix}.yaml"
                with self.subTest(size=size, p2=p2):
                    config = train_rf_detr_model.load_yaml(path)
                    model = config["model"]
                    motion = model["motion"]
                    self.assertEqual(model["size"], size)
                    self.assertIsNone(model["resolution"])
                    self.assertEqual(model["p2"]["enabled"], p2)
                    self.assertTrue(motion["enabled"])
                    self.assertEqual(
                        motion["temporal"]["backbone_grad_mode"],
                        "center_only",
                    )
                    self.assertEqual(
                        motion["tracknet_v5"]["rstr"]["patch_size"],
                        16,
                    )
                    self.assertTrue(motion["overrides"]["gradient_checkpointing"])
                    self.assertEqual(config["train"]["grad_accum_steps"], 1)
                    self.assertFalse(config["train"]["early_stopping"])
                    self.assertEqual(config["train"]["num_workers"], 2)
                    self.assertEqual(config["train"]["batch_size"], 4 if size == "small" else 1)


if __name__ == "__main__":
    unittest.main()
