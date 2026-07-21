import os
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_rf_detr_model as inference_runner  # noqa: E402
import rf_detr_acceleration as acceleration  # noqa: E402
import rf_detr_runtime as runtime  # noqa: E402
import test_rf_detr_model as test_runner  # noqa: E402
import train_rf_detr_model as trainer  # noqa: E402


@contextmanager
def working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class RFDETROutputPathTest(unittest.TestCase):
    timestamp = "20260721123456"

    def test_relative_train_test_and_inference_outputs_ignore_cwd_and_existing_paths(self):
        relative_output = Path("runs") / "rf_detr" / "same_name"
        expected = (PROJECT_DIR / relative_output).resolve()

        with tempfile.TemporaryDirectory() as temporary:
            alternate_cwd = Path(temporary)
            (alternate_cwd / relative_output).mkdir(parents=True)
            with working_directory(alternate_cwd):
                train_output = trainer.build_output_dir(
                    {"output": {"output_dir": str(relative_output)}},
                    self.timestamp,
                )
                test_config = test_runner.build_internal_test_config(
                    {"output": {"output_dir": str(relative_output)}}
                )
                test_output = runtime.build_output_dir(test_config, self.timestamp)
                inference_output = inference_runner.build_output_dir(
                    {"output": {"output_dir": str(relative_output)}},
                    self.timestamp,
                )

        self.assertEqual(train_output, expected)
        self.assertEqual(test_output, expected)
        self.assertEqual(inference_output, expected)

    def test_relative_output_root_and_name_are_project_local(self):
        train_output = trainer.build_output_dir(
            {"output": {"root": "custom_runs", "name": "train_{timestamp}"}},
            self.timestamp,
        )
        inference_output = inference_runner.build_output_dir(
            {"output": {"root": "custom_runs/inference", "name": "infer_{timestamp}"}},
            self.timestamp,
        )

        self.assertEqual(train_output, (PROJECT_DIR / "custom_runs" / f"train_{self.timestamp}").resolve())
        self.assertEqual(
            inference_output,
            (PROJECT_DIR / "custom_runs" / "inference" / f"infer_{self.timestamp}").resolve(),
        )

    def test_absolute_and_parent_relative_outputs_can_target_project_external_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            absolute_output = Path(temporary).resolve() / "absolute_run"
            self.assertEqual(
                trainer.build_output_dir(
                    {"output": {"output_dir": str(absolute_output)}},
                    self.timestamp,
                ),
                absolute_output,
            )
            self.assertEqual(
                inference_runner.build_output_dir(
                    {"output": {"output_dir": str(absolute_output)}},
                    self.timestamp,
                ),
                absolute_output,
            )

        parent_relative = Path("..") / "external_rf_detr_runs" / "run"
        expected_parent = (PROJECT_DIR / parent_relative).resolve()
        self.assertEqual(trainer.resolve_path_for_output(parent_relative), expected_parent)
        self.assertFalse(expected_parent.is_relative_to(PROJECT_DIR))

    def test_ambiguous_windows_paths_are_rejected(self):
        for value in (r"\rooted\run", r"C:drive_relative\run"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "ambiguous Windows path"):
                    trainer.resolve_path_for_output(value)
                with self.assertRaisesRegex(ValueError, "ambiguous Windows path"):
                    acceleration.resolve_tensorrt_cache_dir(value)

    def test_demo_output_is_resolved_from_project_directory(self):
        config = {
            "model": {"size": "small"},
            "train": {"epochs": 10, "batch_size": 8, "grad_accum_steps": 2},
            "output": {},
            "periodic_test": {},
            "demo": {
                "enabled": True,
                "output_dir": "demo_runs/small_{timestamp}",
                "max_epochs": 2,
                "max_batch_size": 2,
                "max_grad_accum_steps": 1,
            },
        }

        trainer.apply_demo_mode(config, self.timestamp, verbose=False)

        self.assertEqual(
            trainer.build_output_dir(config, self.timestamp),
            (PROJECT_DIR / "demo_runs" / f"small_{self.timestamp}").resolve(),
        )

    def test_empty_and_relative_tensorrt_caches_are_project_local(self):
        def settings(cache_dir):
            return acceleration.resolve_acceleration_config(
                {
                    "backend": "tensorrt",
                    "tensorrt": {"cache_dir": cache_dir},
                },
                batch_sizes=[1],
                resolution=704,
            )

        expected_default = (PROJECT_DIR / "runs" / "rf_detr" / "tensorrt_cache").resolve()
        self.assertEqual(settings("").tensorrt.cache_dir, expected_default)
        with tempfile.TemporaryDirectory() as temporary:
            alternate_cwd = Path(temporary)
            (alternate_cwd / "custom_tensorrt_cache").mkdir()
            with working_directory(alternate_cwd):
                relative_cache = settings("custom_tensorrt_cache").tensorrt.cache_dir
        self.assertEqual(relative_cache, (PROJECT_DIR / "custom_tensorrt_cache").resolve())

        config = {
            "model": {
                "resolution": 704,
                "inference_optimization": {
                    "backend": "tensorrt",
                    "tensorrt": {"cache_dir": ""},
                },
            }
        }
        estimate = runtime.estimate_tensorrt_cache_artifacts(config)
        self.assertEqual(Path(estimate["cache_dir"]), expected_default)

    def test_absolute_and_parent_relative_tensorrt_caches_can_be_external(self):
        with tempfile.TemporaryDirectory() as temporary:
            absolute_cache = Path(temporary).resolve() / "trt"
            self.assertEqual(acceleration.resolve_tensorrt_cache_dir(absolute_cache), absolute_cache)

        parent_cache = acceleration.resolve_tensorrt_cache_dir("../external_tensorrt_cache")
        self.assertEqual(parent_cache, (PROJECT_DIR / ".." / "external_tensorrt_cache").resolve())
        self.assertFalse(parent_cache.is_relative_to(PROJECT_DIR))


if __name__ == "__main__":
    unittest.main()
