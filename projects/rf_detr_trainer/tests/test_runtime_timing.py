from pathlib import Path
import io
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_rf_detr_model as trainer  # noqa: E402


class RuntimeTimingTest(unittest.TestCase):
    def test_nonzero_distributed_process_detection(self):
        self.assertFalse(trainer.is_nonzero_distributed_process({}))
        self.assertFalse(trainer.is_nonzero_distributed_process({"LOCAL_RANK": "0", "RANK": "0"}))
        self.assertTrue(trainer.is_nonzero_distributed_process({"LOCAL_RANK": "2", "RANK": "2"}))
        self.assertTrue(trainer.is_nonzero_distributed_process({"GLOBAL_RANK": "1", "LOCAL_RANK": "0"}))

    def test_distributed_child_overrides_use_parent_runtime_paths(self):
        config = {
            "dataset": {
                "source_format": "ultralytics_yolo",
                "dataset_dir": "/source/yolo",
                "data_yaml": "/source/yolo/dataset.yaml",
            },
            "train": {"dataset_dir": "/source/yolo", "dataset_file": "roboflow"},
            "output": {"name": "run_{timestamp}", "exist_ok": False},
        }
        with patch.dict(
            "os.environ",
            {
                trainer.DDP_OUTPUT_DIR_ENV: "/runs/shared",
                trainer.DDP_DATASET_DIR_ENV: "/cache/rfdetr",
                trainer.DDP_DATASET_FILE_ENV: "roboflow",
                "LOCAL_RANK": "1",
            },
            clear=True,
        ):
            metadata = trainer.apply_distributed_child_runtime_overrides(config)

        self.assertTrue(metadata["distributed_child"])
        self.assertEqual(config["output"]["output_dir"], "/runs/shared")
        self.assertTrue(config["output"]["exist_ok"])
        self.assertEqual(config["dataset"]["source_format"], "rfdetr")
        self.assertEqual(config["dataset"]["data_yaml"], "")
        self.assertEqual(config["dataset"]["dataset_dir"], "/cache/rfdetr")
        self.assertEqual(config["train"]["dataset_dir"], "/cache/rfdetr")
        self.assertEqual(config["train"]["dataset_file"], "roboflow")

    def test_format_duration_hms(self):
        self.assertEqual(trainer.format_duration_hms(0), "00:00:00")
        self.assertEqual(trainer.format_duration_hms(0.2), "00:00:01")
        self.assertEqual(trainer.format_duration_hms(59), "00:00:59")
        self.assertEqual(trainer.format_duration_hms(60), "00:01:00")
        self.assertEqual(trainer.format_duration_hms(3661), "01:01:01")
        self.assertEqual(trainer.format_duration_hms(None), "unknown")

    def test_runtime_estimate_uses_default_rate_without_history(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "run"
            estimate = {}

            trainer.add_runtime_estimate(
                estimate=estimate,
                config={"runtime": {"time_estimate": {"use_history": True}}},
                output_dir=output_dir,
                task="test",
                runtime_units=4,
                default_rate_key="default_test_seconds_per_image",
                basis={"test_images": 4},
            )

            self.assertEqual(estimate["estimated_runtime_source"], "default-rate")
            self.assertEqual(estimate["estimated_runtime_seconds"], 1.0)
            self.assertEqual(estimate["estimated_runtime_hms"], "00:00:01")

    def test_runtime_estimate_prefers_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            history_dir = root / "old_run"
            history_dir.mkdir()
            config = {"runtime": {"time_estimate": {"use_history": True}}}
            trainer.write_json(
                history_dir / "run_timing.json",
                {
                    "task": "test",
                    "success": True,
                    "execution_profile": trainer.inference_execution_profile(config),
                    "throughput": {"seconds_per_runtime_unit": 2.0},
                },
            )
            estimate = {}

            trainer.add_runtime_estimate(
                estimate=estimate,
                config={"runtime": {"time_estimate": {"use_history": True}}},
                output_dir=root / "new_run",
                task="test",
                runtime_units=3,
                default_rate_key="default_test_seconds_per_image",
                basis={"test_images": 3},
            )

            self.assertEqual(estimate["estimated_runtime_source"], "history")
            self.assertEqual(estimate["estimated_runtime_seconds"], 6.0)
            self.assertEqual(estimate["estimated_runtime_hms"], "00:00:06")

    def test_runtime_estimate_ignores_different_backend_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            history_dir = root / "old_run"
            history_dir.mkdir()
            config = {"runtime": {"time_estimate": {"use_history": True}}}
            trainer.write_json(
                history_dir / "run_timing.json",
                {
                    "task": "test",
                    "success": True,
                    "execution_profile": {
                        "backend": "tensorrt",
                        "precision": "fp16",
                        "model_size": "medium",
                        "resolution": "default",
                    },
                    "throughput": {"seconds_per_runtime_unit": 9.0},
                },
            )
            estimate = {}

            trainer.add_runtime_estimate(
                estimate=estimate,
                config=config,
                output_dir=root / "new_run",
                task="test",
                runtime_units=4,
                default_rate_key="default_test_seconds_per_image",
                basis={"test_images": 4},
            )

            self.assertEqual(estimate["estimated_runtime_source"], "default-rate")

    def test_execution_profile_partitions_batch_architecture_and_workload(self):
        base = {
            "model": {
                "size": "medium",
                "pretrain_weights": "checkpoint-a.pth",
                "p2": {"enabled": True},
                "motion": {"enabled": False},
                "inference_optimization": {
                    "backend": "tensorrt",
                    "tensorrt": {
                        "precision": "fp16",
                        "profile": {
                            "min_batch_size": 1,
                            "opt_batch_size": "auto",
                            "max_batch_size": "auto",
                        },
                    },
                },
            },
            "test": {"batch_size": 7, "test_mode": {"mode": "sahi"}, "sahi": {"batch_size": 24}},
        }
        changed = json.loads(json.dumps(base))
        changed["test"]["sahi"]["batch_size"] = 64

        first = trainer.inference_execution_profile(base)
        second = trainer.inference_execution_profile(changed)

        self.assertEqual(first["tensorrt_profile"]["max_batch_size"], 24)
        self.assertEqual(second["tensorrt_profile"]["max_batch_size"], 64)
        self.assertNotEqual(first, second)

    def test_inference_execution_profile_uses_active_inference_mode_only(self):
        config = {
            "model": {
                "inference_optimization": {
                    "backend": "tensorrt",
                    "tensorrt": {
                        "precision": "fp16",
                        "profile": {"opt_batch_size": "auto", "max_batch_size": "auto"},
                    },
                }
            },
            "inference": {"mode": "sahi", "batch_size": 8, "video": {"batch_size": 64}},
            "sahi": {"batch_size": 24, "slice_height": 160, "slice_width": 160},
            # Inactive sections must not pollute inference timing history.
            "periodic_test": {"batch_size": 128},
        }

        profile = trainer.inference_execution_profile(config)

        self.assertEqual(profile["workload"]["test_mode"], "sahi")
        self.assertEqual(profile["workload"]["batch_sizes"], [24])
        self.assertEqual(profile["tensorrt_profile"]["max_batch_size"], 24)

    def test_finish_run_timing_writes_json(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "run"
            output_dir.mkdir()
            context = trainer.start_run_timing("inference", verbose=False)
            context.update(
                {
                    "output_dir": str(output_dir),
                    "outputs_created": True,
                    "success": True,
                    "estimate": {
                        "runtime_units": 2,
                        "estimated_runtime_seconds": 1.5,
                        "estimated_runtime_hms": "00:00:02",
                        "estimated_runtime_source": "default-rate",
                        "estimated_runtime_confidence": "rough",
                        "runtime_estimate_basis": {"image_sources": 2},
                    },
                    "started_at_monotonic": time.monotonic() - 0.01,
                    "execution_profile": {
                        "backend": "pytorch",
                        "precision": "bf16",
                        "model_size": "medium",
                        "resolution": 576,
                    },
                    "acceleration": {"cache_hit": False, "backend": "pytorch", "precision": "bf16"},
                }
            )

            trainer.finish_run_timing(context)

            data = json.loads((output_dir / "run_timing.json").read_text(encoding="utf-8"))
            self.assertEqual(data["task"], "inference")
            self.assertTrue(data["success"])
            self.assertEqual(data["estimated_runtime_hms"], "00:00:02")
            self.assertEqual(data["execution_profile"]["precision"], "bf16")
            self.assertEqual(data["acceleration"]["backend"], "pytorch")
            self.assertGreater(data["throughput"]["seconds_per_runtime_unit"], 0)

    def test_finish_run_timing_keeps_engine_build_out_of_steady_state_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "run"
            output_dir.mkdir()
            context = trainer.start_run_timing("inference", verbose=False)
            context.update(
                {
                    "output_dir": str(output_dir),
                    "outputs_created": True,
                    "success": True,
                    "estimate": {"runtime_units": 2},
                    "acceleration": {
                        "export_seconds": 2.0,
                        "build_seconds": 3.0,
                        "load_seconds": 1.0,
                        "warmup_seconds": 0.5,
                    },
                    "stage_timing": {"images_or_frames": 2, "total_seconds": 4.0},
                }
            )

            trainer.finish_run_timing(context)

            data = json.loads((output_dir / "run_timing.json").read_text(encoding="utf-8"))
            throughput = data["throughput"]
            self.assertEqual(throughput["source"], "stage_timing")
            self.assertEqual(throughput["steady_state_seconds"], 4.0)
            self.assertEqual(throughput["seconds_per_runtime_unit"], 2.0)
            self.assertEqual(throughput["engine_export_seconds"], 2.0)
            self.assertEqual(throughput["engine_build_seconds"], 3.0)

    def test_tee_text_stream_ignores_closed_log_file(self):
        console = io.StringIO()
        log_file = tempfile.TemporaryFile("w+", encoding="utf-8")
        try:
            tee = trainer.TeeTextStream(console, log_file)

            self.assertEqual(tee.write("normal\n"), len("normal\n"))
            tee.flush()
            log_file.seek(0)
            self.assertEqual(log_file.read(), "normal\n")

            log_file.close()
            self.assertEqual(tee.write("\x1b[0m"), len("\x1b[0m"))
            tee.flush()
            self.assertTrue(console.getvalue().endswith("\x1b[0m"))
        finally:
            log_file.close()


if __name__ == "__main__":
    unittest.main()
