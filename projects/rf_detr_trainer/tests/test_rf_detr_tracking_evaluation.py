import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import rf_detr_tracking_evaluation as tracking_eval
import evaluate_rf_detr_tracking as evaluation_cli


def task(segment, frame_index, timestamp, name, boxes):
    results = []
    for label, x, y, width, height in boxes:
        results.append({
            "original_width": 200,
            "original_height": 100,
            "type": "rectanglelabels",
            "value": {
                "rectanglelabels": [label],
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
        })
    return {
        "data": {
            "image_name": name,
            "segment_id": segment,
            "frameSplit": {"frame": {"index": frame_index, "timestamp_seconds": timestamp}},
        },
        "annotations": [{"ground_truth": True, "result": results}],
    }


class LabelStudioSequenceTest(unittest.TestCase):
    def test_load_groups_sorts_converts_percent_boxes_and_ignores_empty(self):
        tasks = [
            task("a", 1, 1.1, "a1.jpg", [("soccer_ball", 10, 20, 5, 10)]),
            task("b", 0, 2.0, "b0.jpg", [("key_soccer_ball", 50, 50, 0, 5)]),
            task("a", 0, 1.0, "a0.jpg", [("key_soccer_ball", 20, 30, 10, 20)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
            sequences = tracking_eval.load_label_studio_sequences(root)

        self.assertEqual([row.frame_index for row in sequences["a"]], [0, 1])
        self.assertEqual(sequences["a"][0].boxes[0].bbox, (40.0, 30.0, 20.0, 20.0))
        self.assertEqual(sequences["a"][0].boxes[0].role, "key")
        self.assertEqual(sequences["b"][0].boxes, ())

    def test_pseudo_gt_linking_is_segment_local_and_role_aware(self):
        frames = [
            tracking_eval.FrameGT("s", 0, 0.0, "0.jpg", 100, 100, (
                tracking_eval.GTBox((5, 5, 10, 10), "key", "key_soccer_ball"),
                tracking_eval.GTBox((80, 5, 10, 10), "side", "soccer_ball"),
            )),
            tracking_eval.FrameGT("s", 1, 0.1, "1.jpg", 100, 100, (
                tracking_eval.GTBox((10, 5, 10, 10), "key", "key_soccer_ball"),
                tracking_eval.GTBox((75, 5, 10, 10), "side", "soccer_ball"),
            )),
        ]

        linked = tracking_eval.link_pseudo_gt_tracks(frames, max_distance_pixels=20)

        self.assertEqual(linked[0][0]["gt_track_id"], linked[1][0]["gt_track_id"])
        self.assertEqual(linked[0][1]["gt_track_id"], linked[1][1]["gt_track_id"])
        self.assertNotEqual(linked[0][0]["gt_track_id"], linked[0][1]["gt_track_id"])


class ProxyMetricsTest(unittest.TestCase):
    def test_metrics_count_id_switch_and_false_merge(self):
        matches = [
            {"frame_index": 0, "gt_track_id": 1, "track_id": 10},
            {"frame_index": 1, "gt_track_id": 1, "track_id": 11},
            {"frame_index": 2, "gt_track_id": 2, "track_id": 11},
        ]

        metrics = tracking_eval.association_proxy_metrics(matches, gt_count=3, prediction_count=3)

        self.assertEqual(metrics["id_switches"], 1)
        self.assertEqual(metrics["false_merges"], 1)
        self.assertFalse(metrics["official_hota"])
        self.assertIn("hota_proxy", metrics)

    def test_cache_fingerprint_changes_when_inputs_change(self):
        first = tracking_eval.cache_fingerprint({"config": "a", "checkpoint": "b", "source": ["x", 1]})
        second = tracking_eval.cache_fingerprint({"config": "a", "checkpoint": "b", "source": ["x", 2]})
        self.assertNotEqual(first, second)


class EvaluationCliSafetyTest(unittest.TestCase):
    def test_reid_baselines_require_an_existing_local_weight_file(self):
        config = SimpleNamespace(reid_weights=None)
        with self.assertRaisesRegex(ValueError, "local ReID weights"):
            evaluation_cli.require_local_reid_weights(config, ("deepocsort", "hybrid"))

    def test_motion_only_algorithms_do_not_require_reid_weights(self):
        config = SimpleNamespace(reid_weights=None)
        evaluation_cli.require_local_reid_weights(config, ("circle", "ocsort", "bytetrack", "hybrid"))

    def test_existing_local_reid_weights_are_accepted(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            weights = Path(tmp) / "reid.pt"
            weights.write_bytes(b"local")
            config = SimpleNamespace(reid_weights=str(weights))
            evaluation_cli.require_local_reid_weights(config, ("botsort",))


    def test_default_comparison_is_offline_and_motion_only(self):
        args = evaluation_cli.build_parser().parse_args(
            ["evaluate", "--source-root", ".", "--output-dir", "."]
        )
        self.assertEqual(args.algorithms, "circle,ocsort,bytetrack,hybrid")
if __name__ == "__main__":
    unittest.main()
