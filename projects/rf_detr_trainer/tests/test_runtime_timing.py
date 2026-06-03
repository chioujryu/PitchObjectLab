from pathlib import Path
import json
import sys
import tempfile
import time
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_rf_detr_model as trainer  # noqa: E402


class RuntimeTimingTest(unittest.TestCase):
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
            trainer.write_json(
                history_dir / "run_timing.json",
                {
                    "task": "test",
                    "success": True,
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
                }
            )

            trainer.finish_run_timing(context)

            data = json.loads((output_dir / "run_timing.json").read_text(encoding="utf-8"))
            self.assertEqual(data["task"], "inference")
            self.assertTrue(data["success"])
            self.assertEqual(data["estimated_runtime_hms"], "00:00:02")
            self.assertGreater(data["throughput"]["seconds_per_runtime_unit"], 0)


if __name__ == "__main__":
    unittest.main()
