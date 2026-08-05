"""Focused regression tests for RF-DETR train/test performance policy plumbing."""

from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_rf_detr_model as test_runner  # noqa: E402
import train_rf_detr_model as trainer  # noqa: E402


class PerformanceProfileTest(unittest.TestCase):
    def test_safe_train_profile_is_accuracy_preserving(self):
        config = {
            "runtime": {"performance_profile": "safe"},
            "model": {"extra_model_args": {"compile": True}},
            "train": {},
        }

        selected = trainer.apply_train_performance_profile(config)

        self.assertEqual(selected, "safe")
        self.assertEqual(
            config["train"],
            {
                "amp_dtype": "bf16",
                "batch_size": "auto",
                "auto_batch_target_effective": 32,
                "grad_accum_steps": 8,
                "num_workers": 2,
                "multi_scale": True,
                "expanded_scales": True,
                "ema_update_interval": 1,
                "eval_interval": 1,
                "compute_val_loss": True,
                "checkpoint_interval": 5,
                "save_last_interval": 1,
            },
        )
        self.assertFalse(config["model"]["extra_model_args"]["compile"])

    def test_fast_train_profile_enables_fixed_shape_compile_policy(self):
        config = {
            "runtime": {"performance_profile": "fast"},
            "model": {"extra_model_args": {}},
            "train": {},
        }

        selected = trainer.apply_train_performance_profile(config)

        self.assertEqual(selected, "fast")
        self.assertEqual(config["train"]["amp_dtype"], "bf16")
        self.assertFalse(config["train"]["multi_scale"])
        self.assertFalse(config["train"]["expanded_scales"])
        self.assertEqual(config["train"]["ema_update_interval"], 2)
        self.assertEqual(config["train"]["eval_interval"], 5)
        self.assertFalse(config["train"]["compute_val_loss"])
        self.assertEqual(config["train"]["checkpoint_interval"], 5)
        self.assertEqual(config["train"]["save_last_interval"], 2)
        self.assertTrue(config["model"]["extra_model_args"]["compile"])

    def test_train_autotune_defaults_match_acceptance_plan(self):
        settings = trainer.training_autotune_settings(
            {
                "runtime": {"performance_profile": "safe"},
                "train": {"batch_size": "auto"},
            }
        )

        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["candidates"], [4, 8, 16])
        self.assertEqual(settings["warmup_steps"], 50)
        self.assertEqual(settings["measure_steps"], 200)
        self.assertEqual(settings["target_effective_batch"], 32)
        self.assertEqual(settings["max_vram_fraction"], 0.9)
        self.assertEqual(settings["loader_wait_threshold"], 0.05)

    def test_microbatch_selection_requires_exact_batch_and_strict_vram_gate(self):
        selected = trainer.select_training_microbatch(
            [
                {"micro_batch": 4, "success": True, "images_per_second": 100, "projected_vram_fraction": 0.7},
                {"micro_batch": 8, "success": True, "images_per_second": 120, "projected_vram_fraction": 0.89},
                {"micro_batch": 16, "success": True, "images_per_second": 999, "projected_vram_fraction": 0.9},
                {"micro_batch": 6, "success": True, "images_per_second": 2000, "projected_vram_fraction": 0.2},
            ],
            target_effective_batch=32,
            max_vram_fraction=0.9,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["micro_batch"], 8)
        self.assertEqual(selected["grad_accum_steps"], 4)
        self.assertEqual(selected["effective_batch"], 32)

    def test_test_profiles_select_expected_backend_and_precision(self):
        safe = {"runtime": {"performance_profile": "safe"}, "model": {}}
        fast = {"runtime": {"performance_profile": "fast"}, "model": {}}

        self.assertEqual(test_runner.apply_test_performance_profile(safe), "safe")
        self.assertEqual(test_runner.apply_test_performance_profile(fast), "fast")

        safe_optimization = safe["model"]["inference_optimization"]
        fast_optimization = fast["model"]["inference_optimization"]
        self.assertEqual(safe_optimization["backend"], "pytorch")
        self.assertEqual(safe_optimization["pytorch"]["precision"], "bf16")
        self.assertEqual(fast_optimization["backend"], "tensorrt")
        self.assertEqual(fast_optimization["tensorrt"]["precision"], "fp16")

    def test_sahi_batch_cli_accepts_auto(self):
        parser = trainer.parse_scalar
        self.assertEqual(parser("auto"), "auto")


class TrainCheckpointPolicyTest(unittest.TestCase):
    def test_project_save_last_field_is_not_forwarded_to_train_config(self):
        with tempfile.TemporaryDirectory() as temp:
            config = {
                "train": {
                    "dataset_dir": temp,
                    "save_last_interval": 2,
                    "extra_train_args": {},
                }
            }

            kwargs = trainer.build_train_kwargs(config, Path(temp) / "run")

            self.assertNotIn("save_last_interval", kwargs)
            self.assertEqual(kwargs.pop("_save_last_interval"), 2)

    def test_rolling_checkpoint_has_independent_callback_state(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = ModelCheckpoint(
                dirpath=temp,
                filename="checkpoint_{epoch}",
                every_n_epochs=5,
                save_top_k=-1,
            )
            upstream_last = ModelCheckpoint(
                dirpath=temp,
                filename="last",
                every_n_epochs=1,
                save_top_k=1,
            )
            fake_trainer = SimpleNamespace(
                callbacks=[upstream_last, archive],
                default_root_dir=temp,
            )

            trainer.configure_save_last_interval(
                fake_trainer,
                save_last_interval=2,
                archive_interval=5,
            )

            rolling = fake_trainer.callbacks[0]
            self.assertIsInstance(rolling, trainer.RollingLastCheckpoint)
            self.assertEqual(rolling.every_n_epochs, 2)
            self.assertNotEqual(rolling.state_key, archive.state_key)


class TrainLoaderAndProfilerPolicyTest(unittest.TestCase):
    def test_workers_two_is_kept_when_loader_wait_is_below_five_percent(self):
        selected = trainer.select_loader_worker_count(
            [
                {"num_workers": 2, "success": True, "seconds_per_batch": 0.02},
                {"num_workers": 4, "success": True, "seconds_per_batch": 0.001},
            ],
            model_step_seconds=1.0,
            wait_threshold=0.05,
        )

        self.assertEqual(selected["num_workers"], 2)
        self.assertEqual(selected["selection_reason"], "baseline_wait_within_threshold")

    def test_more_workers_are_considered_only_after_wait_gate(self):
        fake_datamodule = SimpleNamespace(
            _num_workers=2,
            _persistent_workers=True,
            _prefetch_factor=2,
        )
        train_config = SimpleNamespace(
            num_workers=2,
            persistent_workers=None,
            prefetch_factor=None,
        )
        settings = dict(trainer.TRAIN_AUTOTUNE_DEFAULTS)

        def measured(*_args, num_workers, **_kwargs):
            return {
                "num_workers": num_workers,
                "success": True,
                "seconds_per_batch": {2: 0.2, 4: 0.08, 8: 0.1}[num_workers],
            }

        with mock.patch.object(trainer, "benchmark_dataloader_wait", side_effect=measured) as benchmark:
            report = trainer.tune_training_dataloader_workers(
                fake_datamodule,
                train_config,
                settings,
                model_step_seconds=1.0,
            )

        self.assertEqual([call.kwargs["num_workers"] for call in benchmark.call_args_list], [2, 4, 8])
        self.assertEqual(report["selected"]["num_workers"], 4)
        self.assertEqual(train_config.num_workers, 4)

    def test_loader_candidates_are_not_run_when_baseline_wait_is_small(self):
        fake_datamodule = SimpleNamespace(
            _num_workers=2,
            _persistent_workers=True,
            _prefetch_factor=2,
        )
        train_config = SimpleNamespace(num_workers=2, persistent_workers=None, prefetch_factor=None)
        settings = dict(trainer.TRAIN_AUTOTUNE_DEFAULTS)

        with mock.patch.object(
            trainer,
            "benchmark_dataloader_wait",
            return_value={"num_workers": 2, "success": True, "seconds_per_batch": 0.01},
        ) as benchmark:
            report = trainer.tune_training_dataloader_workers(
                fake_datamodule,
                train_config,
                settings,
                model_step_seconds=1.0,
            )

        self.assertEqual([call.kwargs["num_workers"] for call in benchmark.call_args_list], [2])
        self.assertEqual(report["selected"]["num_workers"], 2)

    def test_phase_profiler_is_installed_but_explicit_profiler_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            kwargs = {}
            report = trainer.configure_training_phase_profiler(
                {"runtime": {"performance_profile": "safe"}, "train": {}},
                kwargs,
                Path(temp),
            )
            self.assertTrue(report["enabled"])
            self.assertIn("profiler", kwargs)

            explicit = {"profiler": "pytorch"}
            explicit_report = trainer.configure_training_phase_profiler(
                {"runtime": {"performance_profile": "fast"}, "train": {}},
                explicit,
                Path(temp),
            )
            self.assertEqual(explicit["profiler"], "pytorch")
            self.assertEqual(explicit_report["source"], "trainer.extra_trainer_args")


class FastPostfitValidationTest(unittest.TestCase):
    def test_fast_loads_best_then_runs_full_validation(self):
        calls = []

        class FakeTrainer:
            world_size = 1
            is_global_zero = True

            def validate(self, **kwargs):
                calls.append(("validate", kwargs))

        best = Path("/tmp/best.pth")
        with mock.patch.object(
            trainer,
            "load_best_checkpoint_if_available",
            side_effect=lambda *_args: calls.append(("load_best", {})) or best,
        ):
            ran, checkpoint = trainer.run_fast_postfit_validation(
                performance_profile="fast",
                run_final_test=True,
                trainer=FakeTrainer(),
                module=object(),
                datamodule=object(),
                output_dir=Path("/tmp/run"),
                verbose=False,
            )

        self.assertTrue(ran)
        self.assertEqual(checkpoint, best)
        self.assertEqual([name for name, _ in calls], ["load_best", "validate"])
        self.assertIsNone(calls[1][1]["ckpt_path"])

    def test_safe_profile_does_not_add_postfit_validation(self):
        fake_trainer = mock.Mock()
        ran, checkpoint = trainer.run_fast_postfit_validation(
            performance_profile="safe",
            run_final_test=True,
            trainer=fake_trainer,
            module=object(),
            datamodule=object(),
            output_dir=Path("/tmp/run"),
            verbose=False,
        )

        self.assertFalse(ran)
        self.assertIsNone(checkpoint)
        fake_trainer.validate.assert_not_called()


class EvaluationSessionTest(unittest.TestCase):
    class _Module:
        def __init__(self) -> None:
            self.training = True
            self.device = torch.device("cpu")
            self.eval_calls = 0
            self.train_calls = 0

        def eval(self):
            self.eval_calls += 1
            self.training = False
            return self

        def train(self):
            self.train_calls += 1
            self.training = True
            return self

    def test_whole_evaluation_uses_one_inference_session_and_restores_training(self):
        module = self._Module()

        with trainer.lightning_evaluation_session(
            module,
            SimpleNamespace(amp=True),
            SimpleNamespace(amp_dtype="bf16"),
        ):
            self.assertFalse(module.training)
            self.assertTrue(torch.is_inference_mode_enabled())

        self.assertTrue(module.training)
        self.assertEqual(module.eval_calls, 1)
        self.assertEqual(module.train_calls, 1)


class DatasetPlanningAndMetadataTest(unittest.TestCase):
    def test_merged_config_compacts_large_filename_collections(self):
        filenames = [f"/dataset/train/image_{index:06d}.jpg" for index in range(1000)]

        compacted = trainer.compact_merged_config_snapshot(
            {"dataset": {"image_files": filenames}, "train": {"epochs": 1}}
        )

        summary = compacted["dataset"]["image_files"]
        self.assertTrue(summary["snapshot_compacted"])
        self.assertEqual(summary["item_count"], 1000)
        self.assertEqual(len(summary["sha256"]), 64)
        self.assertNotIn("image_000999.jpg", json.dumps(compacted))

    def test_required_limited_test_split_does_not_scan_train_or_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset_dir = Path(temp) / "dataset"
            for split in ("train", "valid", "test"):
                (dataset_dir / split).mkdir(parents=True)
            config = {
                "dataset": {
                    "source_format": "rfdetr",
                    "dataset_dir": str(dataset_dir),
                },
                "train": {"dataset_dir": str(dataset_dir)},
            }
            calls = []

            def count(path, max_count=None):
                calls.append((path, max_count))
                return max_count

            with mock.patch.object(trainer, "maybe_count_images", side_effect=count):
                plan = trainer.build_dataset_plan(
                    config,
                    Path(temp) / "run",
                    None,
                    required_splits=["test"],
                    required_split_limits={"test": 10},
                )

            self.assertEqual(calls, [(dataset_dir / "test", 10)])
            self.assertEqual(plan["split_counts"], {"test": 10})

    def test_adapter_metadata_stores_link_mode_counts_not_file_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = []
            for index in range(2):
                source = root / "source" / f"image_{index}.jpg"
                source.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(index, 0, 0)).save(source)
                sources.append(source)
            cache_root = root / "cache"
            plan = {
                "source_format": "coco_json",
                "cache_dir": cache_root / "prepared",
                "cache_root": cache_root,
                "link_mode": "copy",
                "refresh_cache": False,
                "fingerprint": {"hash": "abc", "dataset_limits": {}},
                "records_by_split": {
                    "test": [
                        {
                            "source_image": source,
                            "file_name": source.name,
                            "width": 8,
                            "height": 8,
                            "annotations": [],
                        }
                        for source in sources
                    ]
                },
                "categories": [{"id": 1, "name": "ball"}],
                "split_counts": {"test": 2},
                "class_names": ["ball"],
                "warnings": [],
            }
            config = {"dataset": {}, "train": {}}

            metadata = trainer.materialize_cache_dataset_plan(
                plan,
                config,
                root / "run",
                verbose=False,
            )

            self.assertEqual(metadata["link_mode_used"], {"test": {"copy": 2}})
            serialized = json.dumps(metadata)
            self.assertNotIn("image_0.jpg", serialized)
            self.assertNotIn("image_1.jpg", serialized)


if __name__ == "__main__":
    unittest.main()
