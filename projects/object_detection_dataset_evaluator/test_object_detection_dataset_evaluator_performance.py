from pathlib import Path
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from projects.object_detection_common import test_modes
from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator


class RFDetrEvaluatorPerformanceTest(unittest.TestCase):
    @staticmethod
    def detection(category_id: int = 0):
        return SimpleNamespace(
            xyxy=np.array([[1, 1, 6, 6]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([category_id], dtype=np.int64),
        )

    def test_predict_fallback_disables_source_image_copy(self):
        class Model:
            def __init__(self):
                self.include_source_image = None

            def predict(self, images, *, threshold, shape, include_source_image=True):
                self.include_source_image = include_source_image
                return [RFDetrEvaluatorPerformanceTest.detection() for _ in images]

        model = Model()
        detections, _ = evaluator.rfdetr_predict_batches(
            model,
            ["one.jpg", "two.jpg"],
            threshold=0.25,
            shape=(32, 32),
            batch_size=2,
        )

        self.assertEqual(len(detections), 2)
        self.assertIs(model.include_source_image, False)

    def test_oom_safe_batch_is_reused_by_later_calls(self):
        class Model:
            def __init__(self):
                self.calls = []

            def predict(self, images, **_kwargs):
                self.calls.append(len(images))
                if len(images) > 2:
                    raise RuntimeError("CUDA out of memory")
                return [RFDetrEvaluatorPerformanceTest.detection() for _ in images]

        model = Model()
        inputs = [Image.new("RGB", (32, 32)) for _ in range(4)]
        _, first_timings = evaluator.rfdetr_predict_batches(
            model,
            inputs,
            threshold=0.25,
            shape=(32, 32),
            batch_size=4,
        )
        first_calls = list(model.calls)
        model.calls.clear()
        _, second_timings = evaluator.rfdetr_predict_batches(
            model,
            inputs,
            threshold=0.25,
            shape=(32, 32),
            batch_size=4,
        )
        second_calls = list(model.calls)
        model.calls.clear()
        different_sized_inputs = [Image.new("RGB", (64, 64)) for _ in range(4)]
        evaluator.rfdetr_predict_batches(
            model,
            different_sized_inputs,
            threshold=0.25,
            shape=(32, 32),
            batch_size=4,
        )

        self.assertEqual(first_calls, [4, 2, 2])
        self.assertEqual(second_calls, [2, 2])
        self.assertEqual(model.calls, [4, 2, 2])
        self.assertEqual({row["effective_batch_size"] for row in first_timings}, {2})
        self.assertEqual({row["effective_batch_size"] for row in second_timings}, {2})
        self.assertTrue(all(row["safe_batch_cache_hit"] for row in second_timings))
        self.assertEqual(
            model._rf_detr_workload_counters,
            {
                "model_inputs": 12,
                "model_batches": 6,
                "oom_retries": 2,
                "full_inputs": 12,
                "full_batches": 6,
            },
        )

    def test_raw_fast_path_batches_preprocess_and_skips_predict(self):
        import torch
        from projects.rf_detr_trainer import rf_detr_acceleration

        class Handle:
            backend = "pytorch"
            precision = "bf16"
            device = torch.device("cpu")
            metadata = {"effective_backend": "pytorch", "effective_precision": "bf16"}

            def __init__(self):
                self.model = SimpleNamespace(
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5],
                    model=SimpleNamespace(resolution=8),
                )
                self.seen = None

            def prepare_batch(self, images, *, shape=None):
                return rf_detr_acceleration.prepare_inference_batch(
                    images,
                    shape=shape or 8,
                    device=self.device,
                    mean=self.model.means,
                    std=self.model.stds,
                    num_channels=3,
                )

            def infer_raw(self, tensor):
                self.seen = tensor.detach().clone()
                return {"pred_boxes": tensor[:, :1, :1, :1], "pred_logits": tensor[:, :1, :1, :1]}

            def postprocess(self, _outputs, target_sizes):
                return [
                    {
                        "boxes": torch.tensor([[0.0, 0.0, float(size[1]), float(size[0])]]),
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1]),
                    }
                    for size in target_sizes.cpu().tolist()
                ]

            @staticmethod
            def consume_forward_seconds():
                return 0.0

            @staticmethod
            def consume_postprocess_seconds():
                return 0.0

        class Model:
            def __init__(self):
                self._rf_detr_acceleration_handle = Handle()

            def predict(self, *_args, **_kwargs):
                raise AssertionError("upstream predict fallback should not run")

        model = Model()
        white = Image.new("RGB", (4, 4), color=(255, 255, 255))
        detections, timings = evaluator.rfdetr_predict_batches(
            model,
            [white, white],
            threshold=0.25,
            shape=(8, 8),
            batch_size=2,
        )

        self.assertEqual(tuple(model._rf_detr_acceleration_handle.seen.shape), (2, 3, 8, 8))
        self.assertTrue(torch.allclose(model._rf_detr_acceleration_handle.seen, torch.ones((2, 3, 8, 8))))
        self.assertEqual([row.class_id.tolist() for row in detections], [[1], [1]])
        self.assertTrue(all(row["prepare_wall_seconds"] > 0.0 for row in timings))
        self.assertTrue(all(row["device_preprocess_seconds"] == 0.0 for row in timings))
        self.assertTrue(all(row["h2d_seconds"] == 0.0 for row in timings))
        self.assertTrue(all(row["resize_normalize_seconds"] == 0.0 for row in timings))
        self.assertTrue(all(0.0 < row["host_preprocess_seconds"] <= row["preprocess_seconds"] for row in timings))
        self.assertTrue(
            all(
                abs(row["host_preprocess_seconds"] + row["orchestration_seconds"] - row["preprocess_seconds"])
                < 1e-9
                for row in timings
            )
        )

    def test_tensorrt_fast_path_reuses_output_buffers_after_sequential_postprocess(self):
        import torch

        class Handle:
            backend = "tensorrt"
            metadata = {"effective_backend": "tensorrt"}

            def __init__(self):
                self.reused = False

            @staticmethod
            def prepare_batch(_images, *, shape=None):
                return SimpleNamespace(
                    tensor=torch.zeros((1, 3, *(shape or (8, 8)))),
                    target_sizes=torch.tensor([[8, 8]]),
                )

            @staticmethod
            def infer_raw(_tensor):
                raise AssertionError("TensorRT should use its reusable output pool")

            def infer_raw_reusing_buffers(self, tensor):
                self.reused = True
                return {"tensor": tensor}

            @staticmethod
            def postprocess(_outputs, _target_sizes):
                return [
                    {
                        "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1]),
                    }
                ]

        handle = Handle()
        model = SimpleNamespace(_rf_detr_acceleration_handle=handle)
        result = evaluator._rfdetr_fast_predict_batch(
            model,
            [Image.new("RGB", (8, 8))],
            threshold=0.25,
            shape=(8, 8),
        )

        self.assertTrue(handle.reused)
        self.assertEqual(result.detections[0].class_id.tolist(), [1])

    def test_sahi_auto_benchmarks_4_8_16_and_rejects_over_90pct_vram(self):
        images = [
            evaluator.ImageRecord(
                image_id=index,
                file_name=f"{index}.jpg",
                path=f"unused_{index}.jpg",
                width=32,
                height=32,
            )
            for index in range(16)
        ]
        sources = [Image.new("RGB", (32, 32)) for _ in images]
        config = {
            "model": {"type": "rfdetr", "image_size": 32, "confidence_threshold": 0.25},
            "inference": {"batch_size": 1},
            "sahi": {"batch_size": "auto"},
        }
        model = SimpleNamespace()
        requested = []

        def fake_direct(records, _model, _config, **kwargs):
            candidate = int(kwargs["batch_size"])
            requested.append(candidate)
            stats = [
                {
                    "effective_batch_size": candidate,
                    "oom_retry_count": 0,
                }
                for _ in records
            ]
            return [[] for _ in records], stats

        # Candidate durations: b4=4s, b8=1s, b16=0.5s. b16 is rejected
        # because its peak allocation is 95%, so b8 must win.
        with patch.object(evaluator, "predict_rfdetr_direct_batch", side_effect=fake_direct), patch.object(
            evaluator, "_rfdetr_cuda_benchmark_end", side_effect=[0.5, 0.5, 0.95]
        ), patch.object(
            evaluator.time,
            "perf_counter",
            side_effect=[0.0, 0.0, 4.0, 4.0, 5.0, 5.0, 5.5, 5.5],
        ):
            selected, metadata = evaluator.autotune_rfdetr_sahi_batch_size(
                images,
                model,
                config,
                sources=sources,
                widths=[32] * len(images),
                heights=[32] * len(images),
            )

        self.assertEqual(requested, [4, 8, 16])
        self.assertEqual(selected, 8)
        self.assertEqual(metadata["selected_batch_size"], 8)
        self.assertEqual([row["eligible"] for row in metadata["candidates"]], [True, True, False])
        with patch.object(evaluator, "predict_rfdetr_direct_batch") as predict:
            cached_selected, cached = evaluator.autotune_rfdetr_sahi_batch_size(
                images,
                model,
                config,
                sources=sources,
                widths=[32] * len(images),
                heights=[32] * len(images),
            )
        self.assertEqual(cached_selected, 8)
        self.assertTrue(cached["cache_hit"])
        predict.assert_not_called()

    def test_sahi_reuses_in_memory_source_and_reports_oom_downshift(self):
        class Model:
            def __init__(self):
                self.calls = []

            def predict(self, images, **_kwargs):
                self.calls.append(len(images))
                if len(images) > 1:
                    raise RuntimeError("CUDA out of memory")
                return [RFDetrEvaluatorPerformanceTest.detection(category_id=1) for _ in images]

        image = evaluator.ImageRecord(
            image_id=1,
            file_name="not_on_disk.jpg",
            path="/definitely/not/on/disk.jpg",
            width=100,
            height=100,
        )
        config = {
            "model": {"type": "rfdetr", "confidence_threshold": 0.25},
            "dataset_categories": [{"id": 1, "name": "football"}],
            "inference": {"mode": "sahi", "batch_size": 1},
            "test_mode": {"mode": "sahi"},
            "sahi": {
                "slice_width": 50,
                "slice_height": 50,
                "overlap_width_ratio": 0.0,
                "overlap_height_ratio": 0.0,
                "standard_prediction": False,
                "postprocess_type": "GREEDYNMM",
                "postprocess_match_metric": "IOS",
                "postprocess_match_threshold": 0.5,
                "postprocess_class_agnostic": False,
                "batch_size": 3,
                "recheck": {"enabled": False},
            },
        }
        model = Model()
        source = Image.new("RGB", (100, 100), color=(255, 255, 255))

        predictions, stats = evaluator.predict_images_rfdetr_sahi(
            [image],
            model,
            config,
            sources=[source],
        )

        self.assertEqual(len(predictions[0]), 4)
        self.assertEqual(model.calls, [3, 1, 1, 1, 1])
        self.assertEqual(stats[0]["requested_slice_batch_size"], 3)
        self.assertEqual(stats[0]["slice_batch_size"], 1)
        self.assertEqual(stats[0]["effective_slice_batch_sizes"], [1])
        self.assertGreaterEqual(stats[0]["crop_seconds"], 0.0)
        exclusive_total = (
            stats[0]["autotune_seconds"]
            + stats[0]["model_forward_seconds"]
            + stats[0]["exclusive_postprocess_seconds"]
            + stats[0]["crop_seconds"]
            + stats[0]["host_preprocess_seconds"]
            + stats[0]["device_preprocess_seconds"]
            + stats[0]["orchestration_seconds"]
        )
        self.assertAlmostEqual(exclusive_total, stats[0]["elapsed_seconds"], places=5)
        self.assertEqual(model._rf_detr_workload_counters["slice_inputs"], 4)
        self.assertEqual(model._rf_detr_workload_counters["slice_batches"], 4)
        self.assertEqual(model._rf_detr_workload_counters["oom_retries"], 1)

    def test_sahi_auto_profile_controls_actual_slice_batches(self):
        class Model:
            def __init__(self):
                self.calls = []

            def predict(self, images, **_kwargs):
                self.calls.append(len(images))
                return [RFDetrEvaluatorPerformanceTest.detection(category_id=1) for _ in images]

        image = evaluator.ImageRecord(1, "memory.jpg", "/unused.jpg", 200, 200)
        config = {
            "model": {"type": "rfdetr", "confidence_threshold": 0.25},
            "dataset_categories": [{"id": 1, "name": "football"}],
            "inference": {"mode": "sahi", "batch_size": 1},
            "test_mode": {"mode": "sahi"},
            "sahi": {
                "slice_width": 50,
                "slice_height": 50,
                "overlap_width_ratio": 0.0,
                "overlap_height_ratio": 0.0,
                "standard_prediction": False,
                "postprocess_type": "GREEDYNMM",
                "postprocess_match_metric": "IOS",
                "postprocess_match_threshold": 0.5,
                "postprocess_class_agnostic": False,
                "batch_size": "auto",
                "recheck": {"enabled": False},
            },
        }
        profile = {
            "enabled": True,
            "cache_hit": False,
            "selected_batch_size": 8,
            "sample_count": 16,
            "candidates": [],
            "tuning_seconds": 0.0,
            "reason": "test",
        }
        model = Model()
        with patch.object(evaluator, "autotune_rfdetr_sahi_batch_size", return_value=(8, profile)):
            _, stats = evaluator.predict_images_rfdetr_sahi(
                [image],
                model,
                config,
                sources=[Image.new("RGB", (200, 200))],
            )

        self.assertEqual(model.calls, [8, 8])
        self.assertEqual(stats[0]["slice_batch_size_setting"], "auto")
        self.assertEqual(stats[0]["slice_batch_size"], 8)
        self.assertEqual(stats[0]["sahi_batch_autotune"]["selected_batch_size"], 8)

    def test_sahi_auto_starts_at_sixteen(self):
        config = {"model": {"batch_size": 2}, "inference": {"batch_size": 2}, "sahi": {"batch_size": "auto"}}
        self.assertEqual(evaluator.rfdetr_sahi_batch_size(config), 16)

    def test_class_crop_combines_fast_preprocess_stages_and_reports_crop_separately(self):
        image = evaluator.ImageRecord(1, "memory.jpg", "/unused.jpg", 100, 80)
        source = Image.new("RGB", (100, 80))
        config = {
            "model": {"type": "rfdetr", "confidence_threshold": 0.25},
            "inference": {"batch_size": 1},
            "dataset_categories": [{"id": 1, "name": "football"}],
        }
        prediction = {
            "image_id": 1,
            "category_id": 1,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "score": 0.9,
            "area": 12.0,
        }

        def stats(engine):
            return {
                "image_id": 1,
                "file_name": "memory.jpg",
                "width": 100,
                "height": 80,
                "predictions": 1,
                "elapsed_seconds": 0.15,
                "preprocess_seconds": 0.03,
                "host_preprocess_seconds": 0.01,
                "device_preprocess_seconds": 0.02,
                "h2d_seconds": 0.005,
                "resize_normalize_seconds": 0.015,
                "h2d_resize_normalize_seconds": 0.02,
                "orchestration_seconds": 0.0,
                "exclusive_preprocess_seconds": 0.03,
                "prepare_wall_seconds": 0.025,
                "model_forward_seconds": 0.1,
                "postprocess_seconds": 0.02,
                "inference_engine": engine,
            }

        direct_results = [
            ([[prediction]], [stats("rfdetr_class_crop_source")]),
            ([[prediction]], [stats("rfdetr_class_crop")]),
        ]
        with patch.object(
            evaluator.shared_modes,
            "select_crop_window_from_predictions",
            return_value=(10, 20, 30, 40, 1),
        ), patch.object(
            evaluator,
            "predict_rfdetr_direct_batch",
            side_effect=direct_results,
        ):
            predictions, rows = evaluator.predict_images_rfdetr_class_crop(
                [image],
                SimpleNamespace(),
                config,
                sources=[source],
            )

        row = rows[0]
        self.assertEqual(predictions[0][0]["bbox"], [11.0, 22.0, 3.0, 4.0])
        self.assertGreater(row["crop_seconds"], 0.0)
        self.assertAlmostEqual(row["host_preprocess_seconds"], 0.02)
        self.assertAlmostEqual(row["device_preprocess_seconds"], 0.04)
        self.assertAlmostEqual(row["h2d_seconds"], 0.01)
        self.assertAlmostEqual(row["resize_normalize_seconds"], 0.03)
        self.assertAlmostEqual(row["prepare_wall_seconds"], 0.05)
        exclusive_total = (
            row["crop_seconds"]
            + row["exclusive_preprocess_seconds"]
            + row["model_forward_seconds"]
            + row["exclusive_postprocess_seconds"]
        )
        self.assertAlmostEqual(exclusive_total, row["elapsed_seconds"], places=9)

    def test_model_input_manifest_is_sampled_unless_full_is_requested(self):
        images = [
            SimpleNamespace(image_id=index, width=100, height=100, file_name=f"{index}.jpg", path="missing.jpg")
            for index in range(1, 4)
        ]
        base_config = {
            "test_mode": {"mode": "sahi"},
            "sahi": {
                "slice_width": 50,
                "slice_height": 50,
                "overlap_width_ratio": 0.0,
                "overlap_height_ratio": 0.0,
            },
            "output": {"max_model_input_batches": 1, "model_input_batch_size": 2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            sampled_dir = Path(tmp) / "sampled"
            test_modes.write_model_input_artifacts(
                sampled_dir, images, [], [], [], base_config, []
            )
            sampled = json.loads((sampled_dir / "model_inputs_manifest.json").read_text())

            full_config = {**base_config, "output": {**base_config["output"], "full_model_input_manifest": True}}
            full_dir = Path(tmp) / "full"
            test_modes.write_model_input_artifacts(
                full_dir, images, [], [], [], full_config, []
            )
            full = json.loads((full_dir / "model_inputs_manifest.json").read_text())

        self.assertFalse(sampled["full_manifest"])
        self.assertEqual(len(sampled["cases"]), 2)
        self.assertTrue(full["full_manifest"])
        self.assertEqual(len(full["cases"]), 12)

    def test_annotated_jpeg_disables_pillow_optimize(self):
        image = Image.new("RGB", (8, 8), color=(255, 255, 255))
        with tempfile.TemporaryDirectory() as tmp, patch.object(Image.Image, "save") as save:
            evaluator.save_annotated_image(
                image,
                Path(tmp) / "annotated.jpg",
                {"run_id": "run", "config_hash": "hash"},
                92,
            )

        self.assertIs(save.call_args.kwargs["optimize"], False)


if __name__ == "__main__":
    unittest.main()
