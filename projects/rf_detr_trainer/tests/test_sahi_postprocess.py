import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_common import test_modes
from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator


class SahiPostprocessTest(unittest.TestCase):
    @staticmethod
    def fake_detection(category_id: int = 0, score: float = 0.9):
        return SimpleNamespace(
            xyxy=np.array([[1, 1, 6, 6]], dtype=np.float32),
            confidence=np.array([score], dtype=np.float32),
            class_id=np.array([category_id], dtype=np.int64),
        )

    def test_greedynmm_ios_merges_partial_slice_box_into_same_object(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [30, 30, 20, 20], "score": 0.8},
        ]

        merged = test_modes.postprocess_sahi_coco_predictions(
            predictions,
            postprocess_type="GREEDYNMM",
            match_metric="IOS",
            match_threshold=0.5,
            class_agnostic=False,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["category_id"], 1)
        self.assertEqual(merged[0]["score"], 0.9)
        self.assertEqual(merged[0]["merged_prediction_count"], 2)
        self.assertEqual(merged[0]["bbox"], [10.0, 10.0, 100.0, 100.0])

    def test_greedynmm_ios_merges_transitive_small_slice_duplicates(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 60, 80], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [30, 0, 60, 80], "score": 0.85},
            {"image_id": 1, "category_id": 1, "bbox": [60, 0, 60, 80], "score": 0.8},
        ]

        merged = test_modes.postprocess_sahi_coco_predictions(
            predictions,
            postprocess_type="GREEDYNMM",
            match_metric="IOS",
            match_threshold=0.5,
            class_agnostic=False,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["merged_prediction_count"], 3)
        self.assertEqual(merged[0]["bbox"], [0.0, 0.0, 120.0, 80.0])

    def test_greedynmm_ios_keeps_non_overlapping_same_class_boxes(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 80], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [50, 0, 50, 80], "score": 0.8},
        ]

        merged = test_modes.postprocess_sahi_coco_predictions(
            predictions,
            postprocess_type="GREEDYNMM",
            match_metric="IOS",
            match_threshold=0.5,
            class_agnostic=False,
        )

        self.assertEqual(len(merged), 2)

    def test_iou_metric_does_not_merge_low_iou_partial_slice_box(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [30, 30, 20, 20], "score": 0.8},
        ]

        merged = test_modes.postprocess_sahi_coco_predictions(
            predictions,
            postprocess_type="GREEDYNMM",
            match_metric="IOU",
            match_threshold=0.5,
            class_agnostic=False,
        )

        self.assertEqual(len(merged), 2)

    def test_same_location_different_classes_do_not_merge_by_default(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100], "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": [10, 10, 100, 100], "score": 0.8},
        ]

        merged = test_modes.postprocess_sahi_coco_predictions(
            predictions,
            postprocess_type="GREEDYNMM",
            match_metric="IOS",
            match_threshold=0.5,
            class_agnostic=False,
        )

        self.assertEqual(len(merged), 2)

    def test_nms_ios_drops_duplicate_without_merging_box_geometry(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [30, 30, 20, 20], "score": 0.8},
        ]

        merged = test_modes.postprocess_sahi_coco_predictions(
            predictions,
            postprocess_type="NMS",
            match_metric="IOS",
            match_threshold=0.5,
            class_agnostic=False,
        )

        self.assertEqual(len(merged), 1)
        self.assertNotIn("merged_prediction_count", merged[0])
        self.assertEqual(merged[0]["bbox"], [10, 10, 100, 100])

    def test_sahi_recheck_fuses_score_and_keeps_first_stage_geometry(self):
        class FakeModel:
            def predict(self, *_args, **_kwargs):
                return SimpleNamespace(
                    xyxy=np.array([[4, 4, 8, 8]], dtype=np.float32),
                    confidence=np.array([0.8], dtype=np.float32),
                    class_id=np.array([1], dtype=np.int64),
                )

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.jpg"
            Image.new("RGB", (100, 100), color=(255, 255, 255)).save(image_path)
            image = evaluator.ImageRecord(
                image_id=1, file_name=image_path.name, path=str(image_path), width=100, height=100
            )
            config = {
                "model": {"type": "rfdetr", "confidence_threshold": 0.25},
                "dataset_categories": [{"id": 1, "name": "football"}],
                "sahi": {
                    "slice_width": 20,
                    "slice_height": 20,
                    "recheck": {
                        "enabled": True,
                        "target_class_names": ["football"],
                        "crop_size": 20,
                        "second_confidence_threshold": 0.25,
                        "first_weight": 0.5,
                        "second_weight": 0.5,
                        "fused_confidence_threshold": 0.6,
                        "center_padding_ratio": 0.0,
                        "max_rechecks_per_image": 5,
                    },
                },
            }
            predictions = [{"image_id": 1, "category_id": 1, "bbox": [40, 40, 10, 10], "score": 0.6}]

            with Image.open(image_path) as source:
                output, stats = evaluator.apply_sahi_recheck(
                    image, FakeModel(), config, source.convert("RGB"), predictions
                )

            self.assertEqual(len(output), 1)
            self.assertEqual(output[0]["bbox"], [40, 40, 10, 10])
            self.assertAlmostEqual(output[0]["score"], 0.7)
            self.assertEqual(stats["passed"], 1)

    def test_sahi_recheck_filters_target_without_center_hit(self):
        class FakeModel:
            def predict(self, *_args, **_kwargs):
                return SimpleNamespace(
                    xyxy=np.array([[0, 0, 2, 2]], dtype=np.float32),
                    confidence=np.array([0.9], dtype=np.float32),
                    class_id=np.array([1], dtype=np.int64),
                )

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.jpg"
            Image.new("RGB", (100, 100), color=(255, 255, 255)).save(image_path)
            image = evaluator.ImageRecord(
                image_id=1, file_name=image_path.name, path=str(image_path), width=100, height=100
            )
            config = {
                "model": {"type": "rfdetr", "confidence_threshold": 0.25},
                "dataset_categories": [{"id": 0, "name": "player"}, {"id": 1, "name": "football"}],
                "sahi": {
                    "slice_width": 20,
                    "slice_height": 20,
                    "recheck": {
                        "enabled": True,
                        "target_class_names": ["football"],
                        "crop_size": 20,
                        "second_confidence_threshold": 0.25,
                        "first_weight": 0.5,
                        "second_weight": 0.5,
                        "fused_confidence_threshold": 0.25,
                        "center_padding_ratio": 0.0,
                        "max_rechecks_per_image": 5,
                    },
                },
            }
            predictions = [
                {"image_id": 1, "category_id": 1, "bbox": [40, 40, 10, 10], "score": 0.6},
                {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.6},
            ]

            with Image.open(image_path) as source:
                output, stats = evaluator.apply_sahi_recheck(
                    image, FakeModel(), config, source.convert("RGB"), predictions
                )

            self.assertEqual([row["category_id"] for row in output], [0])
            self.assertEqual(stats["filtered"], 1)

    def test_rfdetr_full_image_uses_list_input_batches(self):
        class FakeBatchModel:
            def __init__(self):
                self.calls = []

            def predict(self, images, **_kwargs):
                batch = images if isinstance(images, list) else [images]
                self.calls.append(len(batch))
                return [SahiPostprocessTest.fake_detection(category_id=0, score=0.9) for _ in batch]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for image_id in range(1, 4):
                image_path = root / f"image_{image_id}.jpg"
                Image.new("RGB", (32, 32), color=(255, 255, 255)).save(image_path)
                records.append(
                    evaluator.ImageRecord(
                        image_id=image_id, file_name=image_path.name, path=str(image_path), width=32, height=32
                    )
                )
            config = {
                "model": {"type": "rfdetr", "confidence_threshold": 0.25},
                "inference": {"mode": "full_image", "batch_size": 2},
                "test_mode": {"mode": "full_image"},
                "sahi": {"batch_size": 2},
            }

            model = FakeBatchModel()
            predictions, stats, _ = evaluator.predict_images_rfdetr(records, model, config, root)

            self.assertEqual(model.calls, [2, 1])
            self.assertEqual([len(row) for row in predictions], [1, 1, 1])
            self.assertEqual([row[0]["image_id"] for row in predictions], [1, 2, 3])
            self.assertEqual([stat["batch_size"] for stat in stats], [2, 2, 1])
            self.assertEqual([stat["batch_index"] for stat in stats], [0, 0, 1])
            for stat in stats:
                self.assertEqual(stat["base_model_forward_seconds"], stat["model_forward_seconds"])
                self.assertAlmostEqual(
                    stat["elapsed_seconds"],
                    stat["preprocess_seconds"] + stat["model_forward_seconds"] + stat["postprocess_seconds"],
                )
                self.assertAlmostEqual(
                    stat["model_forward_ratio"],
                    stat["model_forward_seconds"] / stat["elapsed_seconds"],
                )

    def test_rfdetr_direct_sahi_uses_slice_batches(self):
        class FakeBatchModel:
            def __init__(self):
                self.calls = []

            def predict(self, images, **_kwargs):
                batch = images if isinstance(images, list) else [images]
                self.calls.append(len(batch))
                return [SahiPostprocessTest.fake_detection(category_id=1, score=0.9) for _ in batch]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (100, 100), color=(255, 255, 255)).save(image_path)
            image = evaluator.ImageRecord(
                image_id=1, file_name=image_path.name, path=str(image_path), width=100, height=100
            )
            model = FakeBatchModel()
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

            predictions, stats = evaluator.predict_images_rfdetr_sahi([image], model, config)

            self.assertEqual(model.calls, [3, 1])
            self.assertEqual(len(predictions[0]), 4)
            self.assertEqual(stats[0]["slice_batch_size"], 3)

    def test_sahi_recheck_predict_timing_is_included_in_stage_totals_and_ratios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (50, 50), color=(255, 255, 255)).save(image_path)
            image = evaluator.ImageRecord(
                image_id=1,
                file_name=image_path.name,
                path=str(image_path),
                width=50,
                height=50,
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
                    "batch_size": 1,
                    "recheck": {
                        "enabled": True,
                        "target_class_names": ["football"],
                        "crop_size": 20,
                        "second_confidence_threshold": 0.25,
                        "first_weight": 0.5,
                        "second_weight": 0.5,
                        "fused_confidence_threshold": 0.25,
                        "center_padding_ratio": 0.0,
                        "max_rechecks_per_image": 1,
                    },
                },
            }

            def fake_direct_batch(*_args, **kwargs):
                if kwargs["engine"] == "rfdetr_slice":
                    return (
                        [[{"image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10], "score": 0.6}]],
                        [{"elapsed_seconds": 0.4, "model_forward_seconds": 0.4}],
                    )
                if kwargs["engine"] == "rfdetr_recheck":
                    return (
                        [[{"image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10], "score": 0.8}]],
                        [
                            {
                                "elapsed_seconds": 0.2,
                                "model_forward_seconds": 0.2,
                                "batch_index": 0,
                                "batch_size": 1,
                            }
                        ],
                    )
                self.fail(f"Unexpected RF-DETR stage: {kwargs['engine']}")

            with patch.object(evaluator, "predict_rfdetr_direct_batch", side_effect=fake_direct_batch), patch.object(
                evaluator.time,
                "perf_counter",
                side_effect=[0.0, 0.4, 0.6, 0.6, 0.9, 1.1],
            ):
                predictions, stats = evaluator.predict_images_rfdetr_sahi([image], object(), config)

            self.assertEqual(len(predictions[0]), 1)
            stat = stats[0]
            self.assertAlmostEqual(stat["base_model_forward_seconds"], 0.0)
            self.assertAlmostEqual(stat["sahi_model_forward_seconds"], 0.4)
            self.assertAlmostEqual(stat["recheck_model_forward_seconds"], 0.2)
            self.assertAlmostEqual(stat["model_forward_seconds"], 0.6)
            self.assertAlmostEqual(stat["preprocess_seconds"], 0.2)
            self.assertAlmostEqual(stat["postprocess_seconds"], 0.3)
            self.assertAlmostEqual(stat["elapsed_seconds"], 1.1)
            self.assertAlmostEqual(stat["model_forward_ratio"], 0.6 / 1.1)
            self.assertAlmostEqual(stat["sahi_model_forward_ratio"], 0.4 / 1.1)
            self.assertAlmostEqual(stat["recheck_model_forward_ratio"], 0.2 / 1.1)

            recheck = stat["sahi_recheck"]
            self.assertAlmostEqual(recheck["model_forward_seconds"], 0.2)
            self.assertAlmostEqual(recheck["postprocess_seconds"], 0.1)
            self.assertAlmostEqual(recheck["elapsed_seconds"], 0.3)
            self.assertEqual(recheck["batch_count"], 1)
            self.assertEqual(recheck["effective_batch_sizes"], [1])

            summary = evaluator.summarize_inference_timing(stats)
            self.assertAlmostEqual(summary["avg_inference_seconds_per_image"], 1.1)
            self.assertAlmostEqual(summary["avg_base_model_forward_seconds_per_image"], 0.0)
            self.assertAlmostEqual(summary["avg_sahi_model_forward_seconds_per_image"], 0.4)
            self.assertAlmostEqual(summary["avg_recheck_model_forward_seconds_per_image"], 0.2)
            self.assertAlmostEqual(summary["avg_model_forward_seconds_per_image"], 0.6)
            self.assertAlmostEqual(summary["avg_preprocess_seconds_per_image"], 0.2)
            self.assertAlmostEqual(summary["avg_postprocess_seconds_per_image"], 0.3)
            self.assertAlmostEqual(summary["model_forward_ratio"], 0.6 / 1.1)
            self.assertAlmostEqual(summary["sahi_model_forward_ratio"], 0.4 / 1.1)
            self.assertAlmostEqual(summary["recheck_model_forward_ratio"], 0.2 / 1.1)

    def test_timing_summary_preserves_legacy_schema_without_stage_timings(self):
        self.assertEqual(
            evaluator.summarize_inference_timing([{"elapsed_seconds": 0.5}]),
            {"avg_inference_seconds_per_image": 0.5},
        )

    def test_timing_summary_ratios_use_total_evaluation_elapsed_time(self):
        summary = evaluator.summarize_inference_timing(
            [
                {
                    "elapsed_seconds": 1.0,
                    "base_model_forward_seconds": 0.0,
                    "sahi_model_forward_seconds": 0.5,
                    "recheck_model_forward_seconds": 0.0,
                    "model_forward_seconds": 0.5,
                    "postprocess_seconds": 0.5,
                },
                {
                    "elapsed_seconds": 3.0,
                    "base_model_forward_seconds": 0.5,
                    "sahi_model_forward_seconds": 0.0,
                    "recheck_model_forward_seconds": 1.0,
                    "model_forward_seconds": 1.5,
                    "postprocess_seconds": 1.5,
                },
            ]
        )

        self.assertAlmostEqual(summary["avg_inference_seconds_per_image"], 2.0)
        self.assertAlmostEqual(summary["model_forward_ratio"], 2.0 / 4.0)
        self.assertAlmostEqual(summary["sahi_model_forward_ratio"], 0.5 / 4.0)
        self.assertAlmostEqual(summary["recheck_model_forward_ratio"], 1.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
