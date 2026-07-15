from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_rf_detr_model as tester  # noqa: E402
import train_rf_detr_model as trainer  # noqa: E402

TEMP_ROOT = Path(r"C:\tmp")
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class RFDETRModelSizeTest(unittest.TestCase):
    def test_all_installed_detection_and_segmentation_sizes_resolve(self):
        expected = {
            "base": "RFDETRBase",
            "nano": "RFDETRNano",
            "small": "RFDETRSmall",
            "medium": "RFDETRMedium",
            "large": "RFDETRLarge",
            "seg-preview": "RFDETRSegPreview",
            "seg-nano": "RFDETRSegNano",
            "seg-small": "RFDETRSegSmall",
            "seg-medium": "RFDETRSegMedium",
            "seg-large": "RFDETRSegLarge",
            "seg-xlarge": "RFDETRSegXLarge",
            "seg-2xlarge": "RFDETRSeg2XLarge",
        }
        for size, class_name in expected.items():
            with self.subTest(size=size):
                self.assertEqual(trainer.get_model_class(size).__name__, class_name)

    def test_common_size_aliases_normalize(self):
        aliases = {
            "rf-detr-medium.pth": "medium",
            "RFDETRSegNano": "seg-nano",
            "seg_2xlarge": "seg-2xlarge",
            "rf-detr-seg-xxlarge.pt": "seg-2xlarge",
        }
        for value, expected in aliases.items():
            with self.subTest(value=value):
                self.assertEqual(trainer.normalize_model_size(value), expected)

    def test_cli_pretrain_false_string_disables_pretrained_weights(self):
        self.assertEqual(trainer.normalize_pretrain_weights("false"), (True, None))
        self.assertEqual(trainer.normalize_pretrain_weights("null"), (True, None))
        self.assertEqual(trainer.normalize_pretrain_weights("default"), (False, None))

    def test_default_standalone_test_config_is_test_only(self):
        config_path = PROJECT_DIR / "config" / "rf_detr_test.yaml"
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertIn("test", data)
        self.assertNotIn("train", data)
        self.assertNotIn("periodic_test", data)

    def test_standalone_test_config_accepts_test_shape_modes(self):
        for mode in ("full_image", "class_crop", "sahi"):
            with self.subTest(mode=mode):
                config = {
                    "model": {"size": "seg-small"},
                    "dataset": {"dataset_dir": "dataset"},
                    "test": {
                        "split": "valid",
                        "test_mode": {"mode": mode},
                        "sahi": {"slice_height": 320},
                        "classwise": False,
                        "error_cases": {"enabled": True, "class_names": ["ball"], "max_missed_images": 3},
                    },
                }
                internal = tester.build_internal_test_config(config)
                self.assertEqual(internal["test"]["split"], "valid")
                self.assertEqual(internal["test"]["test_mode"]["mode"], mode)
                self.assertEqual(internal["test"]["sahi"]["slice_height"], 320)
                self.assertFalse(internal["test"]["classwise"])
                self.assertTrue(internal["test"]["error_cases"]["enabled"])
                self.assertEqual(internal["periodic_test"]["test_mode"]["mode"], mode)

    def test_standalone_test_config_still_accepts_legacy_periodic_shape(self):
        config = {
            "model": {"size": "seg-small"},
            "dataset": {"dataset_dir": "dataset"},
            "periodic_test": {
                "split": "valid",
                "test_mode": {"mode": "sahi"},
                "sahi": {"slice_height": 320},
                "classwise": False,
            },
        }
        internal = tester.build_internal_test_config(config)
        self.assertEqual(internal["periodic_test"]["split"], "valid")
        self.assertEqual(internal["periodic_test"]["test_mode"]["mode"], "sahi")
        self.assertEqual(internal["periodic_test"]["sahi"]["slice_height"], 320)
        self.assertFalse(internal["periodic_test"]["classwise"])

    def test_standalone_test_config_normalizes_numeric_device(self):
        config = {
            "model": {"size": "medium", "device": 0},
            "dataset": {"dataset_dir": "dataset"},
            "test": {"split": "test", "test_mode": {"mode": "sahi"}},
        }
        internal = tester.build_internal_test_config(config)
        self.assertEqual(internal["model"]["device"], "cuda:0")
        self.assertEqual(internal["train"]["device"], "cuda:0")
        self.assertEqual(trainer.build_model_kwargs(internal)["device"], "cuda:0")

    def test_standalone_test_config_auto_device_is_not_passed_to_model_constructor(self):
        config = {
            "model": {"size": "medium", "device": "auto"},
            "dataset": {"dataset_dir": "dataset"},
            "test": {"split": "test", "test_mode": {"mode": "sahi"}},
        }
        internal = tester.build_internal_test_config(config)
        self.assertEqual(internal["model"]["device"], "auto")
        self.assertEqual(internal["train"]["device"], "auto")
        self.assertNotIn("device", trainer.build_model_kwargs(internal))

    def test_model_constructor_device_shortcuts_are_normalized(self):
        cases = [
            (0, "cuda:0"),
            ("0", "cuda:0"),
            (1, "cuda:1"),
            ("cuda:0", "cuda:0"),
            ("cpu", "cpu"),
            (-1, "cpu"),
            ("-1", "cpu"),
            ("0,1", "cuda:0"),
        ]
        for raw_device, expected in cases:
            with self.subTest(raw_device=raw_device):
                kwargs = trainer.build_model_kwargs({"model": {"device": raw_device}})
                self.assertEqual(kwargs["device"], expected)

        for raw_device in (None, "", "auto"):
            with self.subTest(raw_device=raw_device):
                kwargs = trainer.build_model_kwargs({"model": {"device": raw_device}})
                self.assertNotIn("device", kwargs)

    def test_train_device_parser_returns_rfdetr_compatible_devices(self):
        cases = [
            ("0", {"accelerator": "gpu", "devices": "0,"}),
            ("cuda:0", {"accelerator": "gpu", "devices": "0,"}),
            ("4,5", {"accelerator": "gpu", "devices": "4,5"}),
            ("-1", {"accelerator": "auto", "devices": "auto"}),
            ("cpu", {"accelerator": "cpu"}),
        ]
        for raw_device, expected in cases:
            with self.subTest(raw_device=raw_device):
                self.assertEqual(trainer.parse_device_to_trainer_kwargs(raw_device), expected)

    def test_multigpu_training_uses_find_unused_ddp_strategy(self):
        trainer_kwargs = trainer.parse_device_to_trainer_kwargs("4,5")
        trainer.apply_multigpu_ddp_strategy({"train": {"strategy": "auto"}}, trainer_kwargs, verbose=False)
        self.assertEqual(trainer_kwargs["strategy"], "ddp_find_unused_parameters_true")

    def test_multigpu_training_disables_sanity_validation_by_default(self):
        trainer_kwargs = trainer.parse_device_to_trainer_kwargs("4,5")
        trainer.apply_multigpu_validation_safety(trainer_kwargs, verbose=False)
        self.assertEqual(trainer_kwargs["num_sanity_val_steps"], 0)

    def test_multigpu_training_keeps_explicit_sanity_validation_setting(self):
        trainer_kwargs = trainer.parse_device_to_trainer_kwargs("4,5")
        trainer_kwargs["num_sanity_val_steps"] = 2
        trainer.apply_multigpu_validation_safety(trainer_kwargs, verbose=False)
        self.assertEqual(trainer_kwargs["num_sanity_val_steps"], 2)

    def test_multigpu_training_keeps_explicit_non_ddp_strategy(self):
        trainer_kwargs = trainer.parse_device_to_trainer_kwargs("4,5")
        trainer_kwargs["strategy"] = "deepspeed"
        trainer.apply_multigpu_ddp_strategy({"train": {"strategy": "auto"}}, trainer_kwargs, verbose=False)
        self.assertEqual(trainer_kwargs["strategy"], "deepspeed")

    def test_train_eval_interval_controls_lightning_validation_frequency(self):
        trainer_kwargs = {}
        trainer.apply_validation_interval_to_trainer_kwargs({"train": {"eval_interval": 5}}, trainer_kwargs, verbose=False)
        self.assertEqual(trainer_kwargs["check_val_every_n_epoch"], 5)

    def test_explicit_lightning_validation_frequency_is_preserved(self):
        trainer_kwargs = {"check_val_every_n_epoch": 2}
        trainer.apply_validation_interval_to_trainer_kwargs({"train": {"eval_interval": 5}}, trainer_kwargs, verbose=False)
        self.assertEqual(trainer_kwargs["check_val_every_n_epoch"], 2)

    def test_single_gpu_training_does_not_force_ddp_strategy(self):
        trainer_kwargs = trainer.parse_device_to_trainer_kwargs("4")
        trainer.apply_multigpu_ddp_strategy({"train": {"strategy": "auto"}}, trainer_kwargs, verbose=False)
        self.assertNotIn("strategy", trainer_kwargs)

    def test_single_gpu_training_does_not_force_sanity_validation_setting(self):
        trainer_kwargs = trainer.parse_device_to_trainer_kwargs("4")
        trainer.apply_multigpu_validation_safety(trainer_kwargs, verbose=False)
        self.assertNotIn("num_sanity_val_steps", trainer_kwargs)

    def test_standalone_test_average_inference_seconds_prefers_result_summary(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            output_dir = Path(temp)
            result = {"summary": {"avg_inference_seconds_per_image": 0.1234567}}

            self.assertAlmostEqual(tester.average_inference_seconds(result, output_dir), 0.1234567)

    def test_standalone_test_average_inference_seconds_falls_back_to_stats_csv(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            output_dir = Path(temp)
            (output_dir / "inference_stats.csv").write_text(
                "image_id,file_name,elapsed_seconds\n"
                "1,a.jpg,0.10\n"
                "2,b.jpg,0.30\n",
                encoding="utf-8",
            )

            self.assertAlmostEqual(tester.average_inference_seconds({}, output_dir), 0.20)

    def test_standalone_test_settings_map_to_evaluator_config(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            dataset_dir = Path(temp) / "dataset"
            split_dir = dataset_dir / "valid"
            split_dir.mkdir(parents=True)
            (split_dir / "_annotations.coco.json").write_text("{}", encoding="utf-8")
            config = {
                "runtime": {"verbose": False},
                "model": {"size": "medium", "pretrain_weights": "default", "confidence_threshold": 0.3},
                "train": {"device": "cpu"},
                "test": {
                    "split": "valid",
                    "test_mode": {"mode": "class_crop"},
                    "batch_size": 6,
                    "sahi": {"batch_size": 5},
                    "crop": {"class_names": ["ball"], "source_conf": 0.2},
                    "visual_samples": {
                        "enabled": True,
                        "class_names": ["player"],
                        "render_class_names": ["football"],
                    },
                    "classwise": False,
                    "error_cases": {
                        "enabled": True,
                        "class_ids": [0],
                        "render_class_names": ["player", "football"],
                        "max_missed_images": 2,
                    },
                },
            }
            evaluator_config = trainer.build_rfdetr_evaluator_config(
                merged_config=config,
                model_config=SimpleNamespace(resolution=640),
                train_config=SimpleNamespace(dataset_dir=str(dataset_dir), eval_max_dets=777),
                output_dir=Path(temp) / "out",
                split="valid",
                test_section="test",
            )

            self.assertEqual(evaluator_config["test_mode"]["mode"], "class_crop")
            self.assertEqual(evaluator_config["inference"]["batch_size"], 6)
            self.assertEqual(evaluator_config["sahi"]["batch_size"], 5)
            self.assertEqual(evaluator_config["crop"]["class_names"], ["ball"])
            self.assertFalse(evaluator_config["evaluation"]["classwise"])
            self.assertEqual(evaluator_config["evaluation"]["max_detections"], [1, 10, 777])
            self.assertEqual(evaluator_config["output"]["visual_filter_class_names"], ["player"])
            self.assertEqual(evaluator_config["output"]["visual_render_class_names"], ["football"])
            self.assertTrue(evaluator_config["output"]["error_cases"]["enabled"])
            self.assertEqual(evaluator_config["output"]["error_cases"]["render_class_names"], ["player", "football"])
            self.assertEqual(evaluator_config["output"]["error_cases"]["max_missed_images"], 2)

    def test_rfdetr_evaluator_auto_maps_contiguous_labels_to_coco_category_ids(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            dataset_dir = Path(temp) / "dataset"
            split_dir = dataset_dir / "test"
            split_dir.mkdir(parents=True)
            (split_dir / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [],
                        "annotations": [],
                        "categories": [
                            {"id": 1, "name": "standing_player"},
                            {"id": 2, "name": "football"},
                            {"id": 3, "name": "goal"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "runtime": {"verbose": False},
                "model": {"size": "medium", "pretrain_weights": "default"},
                "train": {"device": "cpu"},
                "test": {"split": "test", "test_mode": {"mode": "sahi"}},
            }
            evaluator_config = trainer.build_rfdetr_evaluator_config(
                merged_config=config,
                model_config=SimpleNamespace(resolution=640),
                train_config=SimpleNamespace(dataset_dir=str(dataset_dir), eval_max_dets=500),
                output_dir=Path(temp) / "out",
                split="test",
                test_section="test",
            )

            self.assertEqual(evaluator_config["model"]["category_remapping"], {0: 1, 1: 2, 2: 3})

    def test_rfdetr_evaluator_keeps_explicit_category_remapping(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            dataset_dir = Path(temp) / "dataset"
            split_dir = dataset_dir / "test"
            split_dir.mkdir(parents=True)
            (split_dir / "_annotations.coco.json").write_text(
                json.dumps({"images": [], "annotations": [], "categories": [{"id": 1, "name": "ball"}]}),
                encoding="utf-8",
            )
            config = {
                "runtime": {"verbose": False},
                "model": {"size": "medium", "pretrain_weights": "default", "category_remapping": {"0": "7"}},
                "train": {"device": "cpu"},
                "test": {"split": "test", "test_mode": {"mode": "sahi"}},
            }
            evaluator_config = trainer.build_rfdetr_evaluator_config(
                merged_config=config,
                model_config=SimpleNamespace(resolution=640),
                train_config=SimpleNamespace(dataset_dir=str(dataset_dir), eval_max_dets=500),
                output_dir=Path(temp) / "out",
                split="test",
                test_section="test",
            )

            self.assertEqual(evaluator_config["model"]["category_remapping"], {0: 7})


if __name__ == "__main__":
    unittest.main()
