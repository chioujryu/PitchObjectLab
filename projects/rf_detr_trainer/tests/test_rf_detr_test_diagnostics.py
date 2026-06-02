import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_rf_detr_model as test_runner  # noqa: E402

from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator  # noqa: E402


class RfDetrTestDiagnosticsTest(unittest.TestCase):
    def test_visual_sample_count_caps_error_cases(self):
        config = {
            "model": {"confidence_threshold": 0.25},
            "dataset": {},
            "evaluation": {"max_detections": [1, 10, 500], "match_iou_threshold": 0.5},
            "test": {
                "split": "test",
                "visual_samples": {"enabled": True, "max_images": 7},
                "error_cases": {"enabled": True, "target_class_names": ["football"]},
            },
        }

        internal = test_runner.build_internal_test_config(config)

        self.assertEqual(internal["test"]["visual_samples"]["max_images"], 7)
        self.assertEqual(internal["test"]["error_cases"]["max_images"], 7)

    def test_render_football_error_cases_outputs_images_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = []
            for image_id in (1, 2, 3):
                image_path = root / f"image_{image_id}.jpg"
                Image.new("RGB", (100, 100), color=(255, 255, 255)).save(image_path)
                images.append(
                    evaluator.ImageRecord(
                        image_id=image_id,
                        file_name=image_path.name,
                        path=str(image_path),
                        width=100,
                        height=100,
                    )
                )
            dataset = evaluator.DatasetBundle(
                images=images,
                categories=[
                    {"id": 0, "name": "standing_player"},
                    {"id": 1, "name": "football"},
                ],
                annotations=[
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                    {"id": 2, "image_id": 2, "category_id": 0, "bbox": [60, 60, 20, 20], "area": 400, "iscrowd": 0},
                    {"id": 3, "image_id": 3, "category_id": 1, "bbox": [30, 30, 20, 20], "area": 400, "iscrowd": 0},
                ],
                coco={},
                source_kind="coco",
            )
            predictions = [
                {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9, "area": 400},
                {"image_id": 2, "category_id": 1, "bbox": [60, 60, 20, 20], "score": 0.8, "area": 400},
            ]
            config = {
                "runtime": {"seed": 0},
                "model": {"confidence_threshold": 0.25},
                "evaluation": {"match_iou_threshold": 0.5, "operating_confidence_threshold": 0.25},
                "progress": {"error_cases": False},
                "output": {
                    "visual_format": "jpg",
                    "visual_jpeg_quality": 92,
                    "draw_ground_truth": True,
                    "draw_predictions": True,
                    "gt_color": "green",
                    "pred_color": "red",
                    "error_cases": {
                        "enabled": True,
                        "target_class_names": ["football"],
                        "max_images": 3,
                        "output_subdir": "error_cases",
                        "format": "jpg",
                        "jpeg_quality": 92,
                    },
                },
            }
            output_info = {"run_id": "test", "config_hash": "hash"}
            manifest = []

            rows = evaluator.render_error_case_outputs(
                dataset, predictions, config, root / "out", output_info, True, manifest
            )
            events, info = evaluator.build_error_case_events(dataset, predictions, config)

            self.assertEqual(len(rows), 3)
            self.assertEqual(info["selected_event_count"], 3)
            self.assertEqual(
                {event["case_type"] for event in events},
                {"target_missed", "target_misclassified", "target_false_positive"},
            )
            self.assertTrue((root / "out" / "error_cases" / "error_cases_manifest.csv").exists())
            self.assertTrue((root / "out" / "error_cases" / "error_case_events.csv").exists())
            self.assertTrue((root / "out" / "error_cases" / "error_cases_metadata.json").exists())
            self.assertEqual(len([row for row in manifest if row["kind"] == "error_case"]), 3)

    def test_render_visual_image_limits_drawn_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (100, 100), color=(255, 255, 255)).save(image_path)
            image = evaluator.ImageRecord(
                image_id=1, file_name=image_path.name, path=str(image_path), width=100, height=100
            )
            config = {
                "output": {
                    "draw_ground_truth": True,
                    "draw_predictions": True,
                    "hide_labels": False,
                    "hide_conf": False,
                    "gt_color": "green",
                    "pred_color": "red",
                    "visual_jpeg_quality": 92,
                }
            }
            annotations = [
                {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
                {"image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10]},
            ]
            predictions = [
                {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
                {"image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10], "score": 0.8},
            ]

            with mock.patch.object(evaluator, "draw_labeled_box") as draw_box:
                evaluator.render_visual_image(
                    image=image,
                    annotations=annotations,
                    predictions=predictions,
                    categories=[{"id": 0, "name": "standing_player"}, {"id": 1, "name": "football"}],
                    config=config,
                    output_info={"run_id": "test", "config_hash": "hash"},
                    output_path=root / "visual.jpg",
                    render_class_ids=[1],
                )

            self.assertEqual([call.args[2] for call in draw_box.call_args_list], ["GT football", "P football 0.80"])

    def test_error_case_target_and_render_classes_are_independent(self):
        categories = [{"id": 0, "name": "standing_player"}, {"id": 1, "name": "football"}]

        target_ids = evaluator.resolve_error_case_class_ids(categories, {})
        render_ids = evaluator.resolve_error_case_render_class_ids(
            categories, {"render_class_names": ["standing_player"]}
        )

        self.assertEqual(target_ids, [1])
        self.assertEqual(render_ids, [0])


if __name__ == "__main__":
    unittest.main()
