from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_common import test_modes  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
