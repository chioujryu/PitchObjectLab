import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator


class CocoMetricsSummaryTest(unittest.TestCase):
    def test_map_50_95_uses_configured_max_dets(self):
        params = SimpleNamespace(
            iouType="bbox",
            iouThrs=np.array([0.5, 0.75], dtype=np.float64),
            maxDets=[1, 10, 500],
            areaRngLbl=["all", "small", "medium", "large"],
        )
        precision = np.full((2, 2, 1, 4, 3), -1.0, dtype=np.float64)
        recall = np.full((2, 1, 4, 3), -1.0, dtype=np.float64)
        precision[0, :, :, :, 2] = 0.5
        precision[1, :, :, :, 2] = 0.3
        recall[:, :, :, 0] = 0.2
        recall[:, :, :, 1] = 0.4
        recall[:, :, :, 2] = 0.6
        coco_eval = SimpleNamespace(params=params, eval={"precision": precision, "recall": recall})

        text = evaluator.summarize_coco_eval(coco_eval)
        metrics = evaluator.coco_metrics_dict(coco_eval)

        self.assertIn("IoU=0.50:0.75", text)
        self.assertIn("maxDets=500", text.splitlines()[0])
        self.assertAlmostEqual(coco_eval.stats[0], 0.4)
        self.assertAlmostEqual(metrics["mAP50-95"], 0.4)
        self.assertAlmostEqual(metrics["mAP50"], 0.5)
        self.assertAlmostEqual(metrics["mAP75"], 0.3)

    def test_per_class_metrics_include_size_breakdowns(self):
        params = SimpleNamespace(
            iouType="bbox",
            iouThrs=np.array([0.5, 0.75], dtype=np.float64),
            maxDets=[1, 10, 500],
            catIds=[1],
            areaRngLbl=["all", "small", "medium", "large"],
        )
        precision = np.full((2, 2, 1, 4, 3), -1.0, dtype=np.float64)
        recall = np.full((2, 1, 4, 3), -1.0, dtype=np.float64)
        precision[0, :, 0, 1, 2] = 0.8
        precision[0, :, 0, 2, 2] = 0.6
        precision[0, :, 0, 3, 2] = 0.4
        precision[1, :, 0, 1, 2] = 0.4
        precision[1, :, 0, 2, 2] = 0.2
        precision[1, :, 0, 3, 2] = 0.0
        recall[:, 0, 1, 2] = 0.7
        recall[:, 0, 2, 2] = 0.5
        recall[:, 0, 3, 2] = 0.3
        coco_eval = SimpleNamespace(params=params, eval={"precision": precision, "recall": recall})

        rows = evaluator.coco_per_class_metrics(coco_eval, [{"id": 1, "name": "football"}])
        size_rows = evaluator.coco_per_class_size_metrics(coco_eval, [{"id": 1, "name": "football"}])

        self.assertIn("mAP50-95_small", rows[0])
        self.assertIn("mAP50_small", rows[0])
        self.assertEqual({row["area"] for row in size_rows}, {"small", "medium", "large"})
        small_row = next(row for row in size_rows if row["area"] == "small")
        self.assertAlmostEqual(small_row["mAP50"], 0.8)
        self.assertAlmostEqual(small_row["mAP50-95"], 0.6)

    def test_operating_metrics_by_area_bucket(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9, "area": 400},
            {"image_id": 1, "category_id": 1, "bbox": [50, 50, 20, 20], "score": 0.8, "area": 400},
        ]
        annotations = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "area": 400},
            {"image_id": 1, "category_id": 1, "bbox": [80, 80, 100, 100], "area": 10000},
        ]

        rows = evaluator.match_predictions_by_area_at_threshold(
            predictions=predictions,
            annotations=annotations,
            category_ids=[1],
            area_ranges=[1024, 9216, 10000000000],
            iou_threshold=0.5,
            confidence_threshold=0.25,
        )

        by_area = {row["area"]: row for row in rows}
        self.assertEqual(by_area["small"]["tp"], 1)
        self.assertEqual(by_area["small"]["fp"], 1)
        self.assertEqual(by_area["large"]["fn"], 1)


if __name__ == "__main__":
    unittest.main()
