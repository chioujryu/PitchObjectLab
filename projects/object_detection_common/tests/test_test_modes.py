import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_common import test_modes


class TestModesTest(unittest.TestCase):
    def test_legacy_use_sahi_normalizes_to_sahi(self):
        self.assertEqual(test_modes.canonical_test_mode({"inference": {"use_sahi": True}}), "sahi")
        self.assertEqual(test_modes.canonical_test_mode({"inference": {"use_sahi": False}}), "full_image")

    def test_prediction_crop_window_with_padding(self):
        categories = [{"id": 1, "name": "ball"}]
        predictions = [{"image_id": 1, "category_id": 1, "bbox": [40, 30, 20, 10], "score": 0.9}]
        crop = {"class_names": ["ball"], "source_conf": 0.25, "padding_pixels": 5, "padding_ratio": 0.0}
        self.assertEqual(
            test_modes.select_crop_window_from_predictions(predictions, crop, categories, 100, 80),
            (35, 25, 30, 20, 1),
        )

    def test_crop_fallback_when_no_class_prediction(self):
        categories = [{"id": 1, "name": "ball"}]
        predictions = [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.9}]
        crop = {"class_names": ["ball"], "source_conf": 0.25}
        self.assertIsNone(test_modes.select_crop_window_from_predictions(predictions, crop, categories, 100, 80))

    def test_project_crop_predictions_back_to_original(self):
        predictions = [{"image_id": 1, "category_id": 0, "bbox": [5, 6, 10, 12], "score": 0.8}]
        projected = test_modes.project_predictions_to_original(predictions, 20, 30, 100, 80)
        self.assertEqual(projected[0]["bbox"], [25.0, 36.0, 10.0, 12.0])

    def test_class_aware_nms_keeps_different_classes(self):
        predictions = [
            {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9},
            {"image_id": 1, "category_id": 0, "bbox": [11, 11, 20, 20], "score": 0.8},
            {"image_id": 1, "category_id": 1, "bbox": [11, 11, 20, 20], "score": 0.7},
        ]
        kept = test_modes.nms_coco_predictions(predictions, iou_threshold=0.5, class_agnostic=False)
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()

