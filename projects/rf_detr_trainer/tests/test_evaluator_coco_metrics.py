import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
