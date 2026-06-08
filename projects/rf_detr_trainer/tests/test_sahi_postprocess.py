import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_common import test_modes  # noqa: E402
from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator  # noqa: E402


class SahiPostprocessTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
