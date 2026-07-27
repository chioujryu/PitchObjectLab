"""Focused tests for standalone RF-DETR parallel test orchestration."""

from __future__ import annotations

import argparse
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import rf_detr_runtime
import test_rf_detr_model as test_runner


def _cli_args(**updates):
    values = {
        "yes": False,
        "dry_run": False,
        "model_size": None,
        "checkpoint": None,
        "resolution": None,
        "num_classes": None,
        "output_dir": None,
        "test_mode": None,
        "max_images": None,
        "batch_size": None,
        "sahi_batch_size": None,
        "chunks": None,
        "inference_backend": None,
        "inference_precision": None,
        "tensorrt_engine": None,
        "tensorrt_cache_dir": None,
        "tensorrt_force_rebuild": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _config(*, chunks=1, device="0", size="medium", mode="full_image", evaluation_type="bbox"):
    return {
        "runtime": {"time_estimate": {"use_history": False, "default_test_seconds_per_image": 1.0}},
        "model": {"size": size, "device": device, "extra_model_args": {}},
        "dataset": {"dataset_dir": "/tmp/materialized-dataset"},
        "output": {"max_model_input_batches": 0},
        "evaluation": {"type": evaluation_type, "classwise": False},
        "test": {
            "parallel": {"chunks": chunks},
            "test_mode": {"mode": mode},
            "split": "test",
            "max_images": None,
            "visual_samples": {"enabled": False},
            "error_cases": {"enabled": False},
        },
        "train": {
            "dataset_dir": "/tmp/materialized-dataset",
            "device": device,
            "eval_max_dets": 500,
            "num_workers": 2,
            "extra_train_args": {},
        },
    }


class ParallelConfigTests(unittest.TestCase):
    def test_standalone_acceleration_cli_overrides_yaml_and_derives_manifest(self):
        config = {
            "model": {
                "inference_optimization": {
                    "backend": "pytorch",
                    "pytorch": {"precision": "fp32"},
                    "tensorrt": {
                        "precision": "fp16",
                        "manifest_path": "stale.engine.manifest.json",
                    },
                }
            }
        }

        test_runner.apply_cli_overrides(
            config,
            _cli_args(
                inference_backend="tensorrt",
                inference_precision="bf16",
                tensorrt_engine="trusted.engine",
                tensorrt_cache_dir="cache",
            ),
        )

        optimization = config["model"]["inference_optimization"]
        self.assertEqual(optimization["backend"], "tensorrt")
        self.assertEqual(optimization["pytorch"]["precision"], "fp32")
        self.assertEqual(optimization["tensorrt"]["precision"], "bf16")
        self.assertEqual(optimization["tensorrt"]["engine_path"], "trusted.engine")
        self.assertEqual(optimization["tensorrt"]["manifest_path"], "")
        self.assertEqual(optimization["tensorrt"]["cache_dir"], "cache")

    def test_acceleration_profile_uses_only_active_model_call_batches(self):
        test_sahi = {
            "test": {
                "batch_size": 80,
                "test_mode": {"mode": "sahi"},
                "sahi": {"batch_size": 64},
            }
        }
        inference_sahi = {
            "inference": {"mode": "sahi", "batch_size": 80, "video": {"batch_size": 32}},
            "sahi": {"batch_size": 64},
        }
        inference_full = {
            "inference": {"mode": "full_image", "batch_size": 8, "video": {"batch_size": 5}},
            "sahi": {"batch_size": 64},
        }
        inference_sahi_inherit = {
            "inference": {"mode": "sahi", "batch_size": 12},
            "sahi": {"batch_size": None},
        }

        self.assertEqual(rf_detr_runtime.inference_acceleration_batch_sizes(test_sahi), [64])
        self.assertEqual(rf_detr_runtime.inference_acceleration_batch_sizes(inference_sahi), [64])
        self.assertEqual(rf_detr_runtime.inference_acceleration_batch_sizes(inference_full), [8, 5])
        self.assertEqual(rf_detr_runtime.inference_acceleration_batch_sizes(inference_sahi_inherit), [12])

    def test_all_standalone_test_configs_document_parallel_default(self):
        config_dir = Path(test_runner.__file__).resolve().parent / "config"
        paths = sorted(config_dir.glob("rf_detr_test*.yaml"))
        tensorrt_presets = {
            "rf_detr_test_p2_tensorrt_fp16_example.yaml",
            "rf_detr_test_tracknet_tensorrt_fp16_example.yaml",
        }
        real_temporal_tracknet_presets = {
            "rf_detr_test_small_tracknet_v5.yaml",
            "rf_detr_test_small_p2_tracknet_v5.yaml",
            "rf_detr_test_smoke_temporal_tracknet_v5.yaml",
        }
        legacy_tracknet_preset = "rf_detr_test_tracknet_tensorrt_fp16_example.yaml"
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                config = test_runner.load_yaml(path)
                self.assertEqual(config["test"]["parallel"]["chunks"], 1)
                optimization = config["model"]["inference_optimization"]
                expected_backend = "tensorrt" if path.name in tensorrt_presets else "pytorch"
                self.assertEqual(optimization["backend"], expected_backend)
                self.assertEqual(optimization["pytorch"]["precision"], "fp32")
                self.assertIn(optimization["tensorrt"]["precision"], {"fp16", "bf16"})
                if path.name == legacy_tracknet_preset:
                    self.assertFalse(config["model"]["p2"]["enabled"])
                    self.assertTrue(config["model"]["motion"]["enabled"])
                    self.assertEqual(
                        config["model"]["motion"]["temporal"]["fallback_mode"],
                        "identity",
                    )
                elif path.name in real_temporal_tracknet_presets:
                    self.assertTrue(config["model"]["motion"]["enabled"])
                    self.assertEqual(
                        config["model"]["motion"]["temporal"]["fallback_mode"],
                        "real",
                    )
                    self.assertEqual(
                        config["model"]["motion"]["temporal"]["mode"],
                        "real",
                    )
                elif path.name == "rf_detr_test_p2_tensorrt_fp16_example.yaml":
                    self.assertTrue(config["model"]["p2"]["enabled"])
                    self.assertFalse(config["model"]["motion"]["enabled"])

    def test_chunks_default_to_one_and_cli_override_is_nested(self):
        internal = test_runner.build_internal_test_config({"model": {}, "dataset": {}, "test": {}})
        self.assertEqual(internal["test"]["parallel"]["chunks"], 1)

        config = {"test": {}}
        test_runner.apply_cli_overrides(config, _cli_args(chunks=6))
        self.assertEqual(config["test"]["parallel"]["chunks"], 6)

    def test_chunks_reject_bool_zero_and_string(self):
        for value in (True, 0, -1, "6"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "test.parallel.chunks"):
                test_runner.normalize_test_parallel_chunks(value)

    def test_estimate_round_robins_devices_adds_summary_and_caps_speedup(self):
        config = _config(chunks=6, device="0,1")
        plan = {"action": "existing", "source_format": "rfdetr", "split_counts": {"test": 12}}
        estimate = test_runner.estimate_standalone_test_outputs(config, Path("/tmp/out"), plan)

        self.assertEqual(
            estimate["parallel_chunk_devices"],
            ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0", "cuda:1"],
        )
        self.assertEqual(estimate["parallel_device_worker_counts"], {"cuda:0": 3, "cuda:1": 3})
        self.assertEqual(estimate["parallel_summary_files"], 1)
        self.assertEqual(estimate["runtime_units"], 6.0)
        self.assertEqual(estimate["runtime_estimate_basis"]["parallel_speedup_cap"], 2)
        self.assertIn("VRAM", estimate["parallel_same_device_warning"])

        single = test_runner.estimate_standalone_test_outputs(
            _config(chunks=1, device="0,1"), Path("/tmp/out-single"), plan
        )
        self.assertEqual(estimate["estimated_total_files"], single["estimated_total_files"] + 1)

    def test_estimate_rejects_more_chunks_than_known_images(self):
        with self.assertRaisesRegex(ValueError, "test.parallel.chunks.*6.*5"):
            test_runner.estimate_standalone_test_outputs(
                _config(chunks=6),
                Path("/tmp/out"),
                {"action": "existing", "split_counts": {"test": 5}},
            )

    def test_missing_device_keeps_auto_assignment_in_estimate(self):
        config = _config(chunks=2)
        config["model"].pop("device")
        config["train"]["device"] = "auto"
        plan = test_runner.build_test_parallel_plan(config, image_count=2)
        self.assertEqual(plan["chunk_devices"], ["auto", "auto"])

    def test_prefixed_comma_devices_match_execution_plan(self):
        config = _config(chunks=4, device="cuda:0,cuda:1")
        plan = test_runner.build_test_parallel_plan(config, image_count=4)
        self.assertEqual(
            plan["chunk_devices"],
            ["cuda:0", "cuda:1", "cuda:0", "cuda:1"],
        )


class ParallelCompatibilityTests(unittest.TestCase):
    def test_segmentation_full_image_auto_multi_requires_bbox_or_one_chunk(self):
        config = _config(chunks=2, size="seg-medium", mode="full_image", evaluation_type="auto")
        with self.assertRaisesRegex(ValueError, r"evaluation\.type=bbox.*chunks=1"):
            test_runner.validate_parallel_test_compatibility(config)

        config["evaluation"]["type"] = "bbox"
        test_runner.validate_parallel_test_compatibility(config)

    def test_explicit_segmentation_head_override_controls_parallel_guard(self):
        detection_size = _config(chunks=2, size="medium", evaluation_type="auto")
        detection_size["model"]["extra_model_args"]["segmentation_head"] = True
        with self.assertRaisesRegex(ValueError, r"evaluation\.type=bbox"):
            test_runner.validate_parallel_test_compatibility(detection_size)

        segmentation_size = _config(chunks=2, size="seg-medium", evaluation_type="auto")
        segmentation_size["model"]["extra_model_args"]["segmentation_head"] = False
        test_runner.validate_parallel_test_compatibility(segmentation_size)


class ParallelWiringTests(unittest.TestCase):
    def test_parallel_evaluator_config_uses_spawn_factory_and_materialized_config(self):
        config = _config(chunks=3, device="0,1")
        base = {"inference": {"mode": "full_image"}, "model": {"device": "0,1"}}
        preview_model = types.SimpleNamespace(resolution=704, segmentation_head=False)
        preview_train = types.SimpleNamespace(dataset_dir="/tmp/materialized-dataset", eval_max_dets=500)

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    test_runner,
                    "build_rfdetr_evaluator_preview",
                    return_value=(preview_model, preview_train),
                )
            )
            stack.enter_context(patch.object(test_runner.trainer, "build_rfdetr_evaluator_config", return_value=base))
            result = test_runner.build_parallel_rfdetr_evaluator_config(config, Path("/tmp/out"), "test")

        self.assertEqual(result["inference"]["chunks"], 3)
        self.assertFalse(result["runtime"]["validate_devices_in_parent"])
        self.assertEqual(result["model"]["factory"], "rf_detr_runtime.build_rfdetr_evaluator_model")
        self.assertEqual(result["model"]["devices"], ["cuda:0", "cuda:1"])
        factory_config = result["model"]["factory_config"]
        self.assertEqual(Path(factory_config["output_dir"]), Path("/tmp/out"))
        self.assertEqual(
            Path(factory_config["merged_config"]["train"]["dataset_dir"]),
            Path("/tmp/materialized-dataset"),
        )
        self.assertIsNot(factory_config["merged_config"], config)

    def test_resolution_preview_matches_constructor_precedence(self):
        class FakeModelConfig:
            resolution = 576
            segmentation_head = False

        class FakeModelClass:
            _model_config_class = FakeModelConfig

        config = _config()
        config["model"].update(
            {
                "resolution": 800,
                "extra_model_args": {"resolution": 700},
                "p2": {"enabled": True, "overrides": {"resolution": 960}},
            }
        )
        with patch.object(test_runner.trainer, "get_model_class", return_value=FakeModelClass):
            model_config, train_config = test_runner.build_rfdetr_evaluator_preview(config, Path("/tmp/out"))
        self.assertEqual(model_config.resolution, 960)
        self.assertFalse(model_config.segmentation_head)
        self.assertEqual(train_config.dataset_dir, "/tmp/materialized-dataset")
        self.assertEqual(train_config.eval_max_dets, 500)


class SpawnFactoryTests(unittest.TestCase):
    def test_parallel_preparation_builds_once_per_gpu_compatibility_profile(self):
        config = _config(chunks=3, device="0,1,2")
        config["model"]["inference_optimization"] = {
            "backend": "tensorrt",
            "tensorrt": {"precision": "fp16"},
        }
        built_devices = []

        def preflight(_config, *, device):
            return {
                "backend": "tensorrt",
                "precision": "fp16",
                "device": f"cuda:{device}",
            }

        def properties(device):
            return types.SimpleNamespace(name="same-gpu" if device.index in {0, 1} else "other-gpu")

        def capability(device):
            return (8, 9) if device.index in {0, 1} else (8, 6)

        def build_runtime(_config, _output_dir, *, device):
            built_devices.append(device)
            model = types.SimpleNamespace(profile_index=len(built_devices))
            return model, object()

        def get_handle(model):
            index = model.profile_index
            return types.SimpleNamespace(
                metadata={
                    "cache_hit": False,
                    "build_seconds": 1.0,
                    "export_seconds": 0.5,
                    "load_seconds": 0.1,
                    "warmup_seconds": 0.1,
                    "engine_path": f"/cache/{index}/model.engine",
                    "manifest_path": f"/cache/{index}/model.engine.manifest.json",
                }
            )

        with patch.object(
            rf_detr_runtime,
            "validate_inference_acceleration_config",
            return_value=types.SimpleNamespace(backend="tensorrt", precision="fp16"),
        ), patch.object(
            rf_detr_runtime, "preflight_rfdetr_inference_acceleration", side_effect=preflight
        ), patch.object(
            torch.cuda, "get_device_properties", side_effect=properties
        ), patch.object(
            torch.cuda, "get_device_capability", side_effect=capability
        ), patch.object(
            rf_detr_runtime, "build_rfdetr_evaluator_runtime", side_effect=build_runtime
        ), patch.object(
            rf_detr_runtime, "get_inference_acceleration_handle", side_effect=get_handle
        ):
            result = rf_detr_runtime._prepare_parallel_tensorrt_profiles(
                config,
                "/tmp/out",
                ["0", "1", "2"],
            )

        self.assertEqual(built_devices, ["cuda:0", "cuda:2"])
        self.assertEqual(len(result["profiles"]), 2)
        self.assertEqual(result["device_artifacts"]["0"]["engine_path"], "/cache/1/model.engine")
        self.assertEqual(result["device_artifacts"]["1"]["engine_path"], "/cache/1/model.engine")
        self.assertEqual(result["device_artifacts"]["2"]["engine_path"], "/cache/2/model.engine")

    def test_factory_overrides_devices_attaches_motion_and_aligns_classes(self):
        model = MagicMock()
        model.model = object()
        model.model_config = types.SimpleNamespace(pretrain_weights="matching_motion_checkpoint.pth")
        model_cls = MagicMock(return_value=model)
        train_config = types.SimpleNamespace(dataset_dir="/tmp/materialized-dataset")
        model.get_train_config.return_value = train_config
        seen = {}

        def model_kwargs(config):
            seen["model_device"] = config["model"]["device"]
            seen["p2"] = config["model"]["p2"]
            return {"device": config["model"]["device"]}

        def train_kwargs(config, output_dir):
            seen["train_device"] = config["train"]["device"]
            seen["output_dir"] = output_dir
            return {"dataset_dir": config["train"]["dataset_dir"], "_device": config["train"]["device"]}

        merged = _config()
        merged["model"]["p2"] = {"enabled": True}
        merged["model"]["motion"] = {
            "enabled": True,
            "variant": "v5",
            "temporal": {"fallback_mode": "identity"},
        }
        factory_model_cfg = {"factory_config": {"merged_config": merged, "output_dir": "/tmp/out"}}
        motion_module = types.SimpleNamespace(
            attach_motion_module=MagicMock(),
            assert_motion_checkpoint_compatible=MagicMock(),
            load_motion_checkpoint_weights=MagicMock(),
        )
        p2_module = types.SimpleNamespace(assert_p2_checkpoint_compatible=MagicMock())

        with ExitStack() as stack:
            stack.enter_context(patch.object(rf_detr_runtime, "get_model_class", return_value=model_cls))
            stack.enter_context(patch.object(rf_detr_runtime, "build_model_kwargs", side_effect=model_kwargs))
            stack.enter_context(patch.object(rf_detr_runtime, "build_train_kwargs", side_effect=train_kwargs))
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {"rf_detr_motion": motion_module, "rf_detr_p2": p2_module},
                )
            )
            result = rf_detr_runtime.build_rfdetr_evaluator_model(factory_model_cfg, "cuda:2")

        self.assertIs(result, model)
        self.assertEqual(seen["model_device"], "cuda:2")
        self.assertEqual(seen["train_device"], "cuda:2")
        self.assertEqual(seen["output_dir"], Path("/tmp/out"))
        self.assertTrue(seen["p2"]["enabled"])
        motion_module.attach_motion_module.assert_called_once_with(model.model, merged["model"]["motion"])
        motion_module.assert_motion_checkpoint_compatible.assert_called_once()
        motion_module.load_motion_checkpoint_weights.assert_called_once_with(
            model.model,
            "matching_motion_checkpoint.pth",
        )
        p2_module.assert_p2_checkpoint_compatible.assert_called_once()
        model.get_train_config.assert_called_once_with(dataset_dir="/tmp/materialized-dataset")
        model._align_num_classes_from_dataset.assert_called_once_with("/tmp/materialized-dataset")
        self.assertEqual(merged["model"]["device"], "0")

    def test_factory_maps_prepared_engine_to_normalized_device_and_disables_rebuild(self):
        merged = _config(device="0,1")
        merged["model"]["inference_optimization"] = {
            "backend": "tensorrt",
            "tensorrt": {
                "precision": "fp16",
                "engine_path": "",
                "manifest_path": "",
                "force_rebuild": True,
            },
        }
        factory_model_cfg = {
            "factory_config": {
                "merged_config": merged,
                "output_dir": "/tmp/out",
                "prepared_tensorrt": {
                    "cuda:2": {
                        "engine_path": "/tmp/cache/model.engine",
                        "manifest_path": "/tmp/cache/model.engine.manifest.json",
                        # The worker must force this off even if either source asks to rebuild.
                        "force_rebuild": True,
                    }
                },
            }
        }
        prepared_model = object()

        with patch.object(
            rf_detr_runtime,
            "build_rfdetr_evaluator_runtime",
            return_value=(prepared_model, object()),
        ) as build_runtime:
            result = rf_detr_runtime.build_rfdetr_evaluator_model(factory_model_cfg, "2")

        self.assertIs(result, prepared_model)
        worker_config = build_runtime.call_args.args[0]
        worker_tensorrt = worker_config["model"]["inference_optimization"]["tensorrt"]
        self.assertEqual(worker_tensorrt["engine_path"], "/tmp/cache/model.engine")
        self.assertEqual(
            worker_tensorrt["manifest_path"],
            "/tmp/cache/model.engine.manifest.json",
        )
        self.assertFalse(worker_tensorrt["force_rebuild"])
        self.assertEqual(build_runtime.call_args.kwargs["device"], "2")
        self.assertTrue(
            merged["model"]["inference_optimization"]["tensorrt"]["force_rebuild"],
            "worker mapping must not mutate the shared parent config",
        )

    def test_factory_refuses_unmapped_worker_instead_of_rebuilding(self):
        merged = _config(device="0")
        merged["model"]["inference_optimization"] = {
            "backend": "tensorrt",
            "tensorrt": {"precision": "fp16", "force_rebuild": True},
        }
        factory_model_cfg = {
            "factory_config": {
                "merged_config": merged,
                "output_dir": "/tmp/out",
                "prepared_tensorrt": {
                    "cuda:0": {
                        "engine_path": "/tmp/cache/model.engine",
                        "manifest_path": "/tmp/cache/model.engine.manifest.json",
                    }
                },
            }
        }

        with patch.object(rf_detr_runtime, "build_rfdetr_evaluator_runtime") as build_runtime:
            with self.assertRaisesRegex(RuntimeError, "not allowed to rebuild"):
                rf_detr_runtime.build_rfdetr_evaluator_model(factory_model_cfg, "cuda:3")

        build_runtime.assert_not_called()


class StandaloneMainFlowTests(unittest.TestCase):
    @staticmethod
    def _main_patches(config, output_dir, evaluator_result):
        dataset_plan = {
            "action": "none",
            "source_format": "rfdetr",
            "split_counts": {"test": 4},
            "dataset_dir": Path("/tmp/resolved-dataset"),
        }
        return (
            patch.object(test_runner, "load_yaml", return_value=config),
            patch.object(test_runner.trainer, "build_output_dir", return_value=output_dir),
            patch.object(test_runner.trainer, "build_dataset_plan", return_value=dataset_plan),
            patch.object(test_runner, "confirm_test_or_exit"),
            patch.object(test_runner.trainer, "start_run_log_capture"),
            patch.object(test_runner.trainer, "materialize_dataset_plan", return_value={}),
            patch.object(test_runner.trainer, "dump_config_snapshot"),
            patch.object(test_runner, "run_evaluation", return_value=evaluator_result),
            patch.object(test_runner.trainer, "write_rfdetr_evaluator_aliases"),
            patch.object(test_runner, "print_inference_timing_summary"),
        )

    def test_parallel_main_never_constructs_parent_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "parallel-output"
            config = _config(chunks=2, evaluation_type="bbox")
            config["dataset"]["dataset_dir"] = "relative-dataset"
            config["train"]["dataset_dir"] = "relative-dataset"
            config["runtime"].update({"yes": True, "confirm_before_run": False, "verbose": False})
            evaluator_config = {"inference": {"chunks": 2}, "model": {}}
            evaluator_result = {"summary": {}, "per_class": []}
            common = self._main_patches(config, output_dir, evaluator_result)
            with ExitStack() as stack:
                started = [stack.enter_context(patcher) for patcher in common]
                build_parallel = stack.enter_context(
                    patch.object(
                        test_runner,
                        "build_parallel_rfdetr_evaluator_config",
                        return_value=evaluator_config,
                    )
                )
                get_model_class = stack.enter_context(
                    patch.object(
                        test_runner.trainer,
                        "get_model_class",
                        side_effect=AssertionError("parent model construction is forbidden"),
                    )
                )
                stack.enter_context(patch.object(sys, "argv", ["test_rf_detr_model.py", "--config", "dummy.yaml"]))
                self.assertEqual(test_runner._main_impl(), 0)

            get_model_class.assert_not_called()
            build_parallel.assert_called_once()
            built_config = build_parallel.call_args.args[0]
            self.assertEqual(Path(built_config["dataset"]["dataset_dir"]), Path("/tmp/resolved-dataset"))
            self.assertEqual(Path(built_config["train"]["dataset_dir"]), Path("/tmp/resolved-dataset"))
            self.assertIsNone(started[7].call_args.kwargs["prebuilt_model"])

    def test_single_chunk_main_keeps_prebuilt_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "single-output"
            config = _config(chunks=1, evaluation_type="bbox")
            config["model"]["motion"] = {
                "enabled": True,
                "variant": "v5",
                "temporal": {"fallback_mode": "identity"},
            }
            config["model"]["pretrain_weights"] = "matching_motion_checkpoint.pth"
            config["runtime"].update({"yes": True, "confirm_before_run": False, "verbose": False})
            evaluator_result = {"summary": {}, "per_class": []}
            common = self._main_patches(config, output_dir, evaluator_result)
            rf_model = MagicMock()
            rf_model.model_config = types.SimpleNamespace(
                resolution=576,
                segmentation_head=False,
                pretrain_weights="matching_motion_checkpoint.pth",
            )
            rf_model.get_train_config.return_value = types.SimpleNamespace(
                dataset_dir="/tmp/materialized-dataset",
                eval_max_dets=500,
            )
            model_cls = MagicMock(return_value=rf_model)
            motion_module = types.SimpleNamespace(
                attach_motion_module=MagicMock(),
                assert_motion_checkpoint_compatible=MagicMock(),
                load_motion_checkpoint_weights=MagicMock(),
            )
            with ExitStack() as stack:
                started = [stack.enter_context(patcher) for patcher in common]
                stack.enter_context(patch.object(test_runner.trainer, "get_model_class", return_value=model_cls))
                stack.enter_context(patch.object(test_runner.trainer, "build_model_kwargs", return_value={}))
                stack.enter_context(
                    patch.object(
                        test_runner.trainer,
                        "build_train_kwargs",
                        return_value={"dataset_dir": "/tmp/materialized-dataset"},
                    )
                )
                stack.enter_context(
                    patch.object(
                        test_runner.trainer,
                        "build_rfdetr_evaluator_config",
                        return_value={"inference": {}, "model": {}},
                    )
                )
                stack.enter_context(patch.object(sys, "argv", ["test_rf_detr_model.py", "--config", "dummy.yaml"]))
                stack.enter_context(patch.dict(sys.modules, {"rf_detr_motion": motion_module}))
                self.assertEqual(test_runner._main_impl(), 0)

            model_cls.assert_called_once_with()
            rf_model._align_num_classes_from_dataset.assert_called_once_with("/tmp/materialized-dataset")
            motion_module.attach_motion_module.assert_called_once_with(
                rf_model.model,
                config["model"]["motion"],
            )
            motion_module.assert_motion_checkpoint_compatible.assert_called_once()
            motion_module.load_motion_checkpoint_weights.assert_called_once_with(
                rf_model.model,
                "matching_motion_checkpoint.pth",
            )
            self.assertIs(started[7].call_args.kwargs["prebuilt_model"], rf_model)


if __name__ == "__main__":
    unittest.main()
