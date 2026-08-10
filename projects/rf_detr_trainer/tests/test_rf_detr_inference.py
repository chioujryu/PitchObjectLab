import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inference_rf_detr_model as inference_runner  # noqa: E402


class RfDetrInferenceTest(unittest.TestCase):
    def test_peak_vram_uses_the_models_nondefault_cuda_device(self):
        import torch

        model = object()
        handle = SimpleNamespace(device=torch.device("cuda:2"))
        with patch.object(
            inference_runner.trainer,
            "get_inference_acceleration_handle",
            return_value=handle,
        ), patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda,
            "max_memory_allocated",
            return_value=123456,
        ) as peak:
            self.assertEqual(inference_runner.peak_vram_bytes_for_model(model), 123456)

        peak.assert_called_once_with(torch.device("cuda:2"))

    def test_discover_mixed_folder_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jpg").write_bytes(b"image")
            (root / "b.mp4").write_bytes(b"video")
            (root / "ignore.txt").write_text("ignore", encoding="utf-8")
            config = {"inference": {"sources": [str(root)], "recursive": True}}

            items = inference_runner.discover_sources(config)

            self.assertEqual([item.kind for item in items], ["image", "video"])

    def test_source_limits_apply_to_mixed_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.jpg", "b.jpg", "c.mp4", "d.mp4"):
                (root / name).write_bytes(b"media")
            config = {
                "inference": {
                    "sources": [str(root)],
                    "recursive": True,
                    "max_images": 1,
                    "max_videos": 1,
                    "max_sources": 2,
                }
            }

            items = inference_runner.apply_source_limits(inference_runner.discover_sources(config), config)

            self.assertEqual(len(items), 2)
            self.assertEqual([item.kind for item in items], ["image", "video"])

    def test_video_seconds_limit_to_frame_total(self):
        self.assertIsNone(inference_runner.parse_seconds_limit("all"))
        self.assertEqual(inference_runner.parse_seconds_limit("00:00:02"), 2.0)
        self.assertEqual(inference_runner.limited_video_frame_total(300, 30.0, 2.0), 60)
        self.assertEqual(inference_runner.limited_video_frame_total(30, 30.0, 2.0), 30)

    def test_video_time_parsing_accepts_hms_and_mmss(self):
        self.assertEqual(inference_runner.parse_video_time_seconds("01:02:03", "time"), 3723.0)
        self.assertEqual(inference_runner.parse_video_time_seconds("02:03", "time"), 123.0)
        self.assertEqual(inference_runner.parse_video_time_seconds(4.5, "time"), 4.5)
        self.assertIsNone(inference_runner.parse_video_time_seconds("all", "time", allow_all=True))

    def test_video_frame_window_uses_start_end_segment(self):
        window = inference_runner.video_frame_window(
            frame_count=300,
            input_fps=30.0,
            video_cfg={"start_time": "00:00:02", "end_time": "00:00:05", "max_seconds": "all"},
        )

        self.assertEqual(window.start_frame, 60)
        self.assertEqual(window.end_frame, 150)
        self.assertEqual(window.output_frames, 90)

    def test_video_frame_window_max_seconds_caps_segment(self):
        window = inference_runner.video_frame_window(
            frame_count=600,
            input_fps=30.0,
            video_cfg={"start_time": "00:00:02", "end_time": "00:00:10", "max_seconds": "00:00:03"},
        )

        self.assertEqual(window.start_frame, 60)
        self.assertEqual(window.end_frame, 150)
        self.assertEqual(window.output_frames, 90)
        self.assertEqual(window.effective_end_seconds, 5.0)

    def test_video_frame_window_rejects_end_before_start(self):
        with self.assertRaises(ValueError):
            inference_runner.video_frame_window(
                frame_count=300,
                input_fps=30.0,
                video_cfg={"start_time": "00:00:05", "end_time": "00:00:05"},
            )

    def test_class_color_is_stable(self):
        self.assertEqual(inference_runner.class_color(3), inference_runner.class_color(3))
        self.assertNotEqual(inference_runner.class_color(3), inference_runner.class_color(4))

    def test_inference_batch_size_defaults_and_video_inheritance(self):
        config = {"inference": {"batch_size": 8, "video": {"batch_size": None}}}

        self.assertEqual(inference_runner.inference_batch_size(config), 8)
        self.assertEqual(inference_runner.video_batch_size(config), 8)
        self.assertEqual(inference_runner.video_batch_size({"inference": {"batch_size": 8, "video": {"batch_size": 3}}}), 3)

    def test_sahi_workload_estimator_counts_1080p_slice_inputs(self):
        config = {
            "inference": {"mode": "sahi"},
            "sahi": {
                "slice_height": 320,
                "slice_width": 320,
                "overlap_height_ratio": 0.2,
                "overlap_width_ratio": 0.2,
                "standard_prediction": False,
                "batch_size": 4,
                "recheck": {"enabled": True, "max_rechecks_per_image": 50},
            },
        }

        workload = inference_runner.estimated_rfdetr_workload(1920, 1080, 1800, config)

        self.assertEqual(workload["slice_inputs"], 57_600)
        self.assertEqual(workload["model_inputs"], 57_600)
        self.assertEqual(workload["model_batches"], 14_400)
        self.assertEqual(workload["recheck_input_cap"], 90_000)

    def test_performance_profiles_do_not_change_slice_geometry(self):
        base = {
            "runtime": {},
            "model": {"inference_optimization": {}},
            "inference": {"video": {}, "tracking": {"hybrid": {"cmc": {}}}},
            "sahi": {"slice_width": 320, "slice_height": 320},
        }
        safe = json.loads(json.dumps(base))
        fast = json.loads(json.dumps(base))

        inference_runner.apply_performance_profile(safe, "safe")
        inference_runner.apply_performance_profile(fast, "fast")

        self.assertEqual(safe["sahi"], fast["sahi"])
        self.assertEqual(safe["model"]["inference_optimization"]["backend"], "pytorch")
        self.assertEqual(fast["model"]["inference_optimization"]["backend"], "tensorrt")
        self.assertEqual(
            fast["inference"]["tracking"]["hybrid"]["cmc"]["processing_scale"],
            0.5,
        )

    def test_medium_p2_performance_presets_keep_the_32_slice_safe_workload(self):
        safe = inference_runner.load_yaml(
            PROJECT_DIR / "config" / "rf_detr_inference_medium_p2_performance_safe.yaml"
        )
        fast = inference_runner.load_yaml(
            PROJECT_DIR / "config" / "rf_detr_inference_medium_p2_performance_fast.yaml"
        )

        for config in (safe, fast):
            workload = inference_runner.estimated_rfdetr_workload(1920, 1080, 1800, config)
            self.assertEqual(workload["slice_inputs"], 57_600)
            self.assertEqual(config["model"]["num_classes"], 1)
            self.assertEqual(config["dataset"]["class_names"], ["football"])
            self.assertFalse(config["sahi"]["standard_prediction"])
            self.assertTrue(config["inference"]["tracking"]["enabled"])
            self.assertEqual(config["inference"]["tracking"]["algorithm"], "hybrid")
            self.assertFalse(config["inference"]["tracking"]["draw_predicted_center"])
            self.assertTrue(config["inference"]["tracking"]["render_confirmed_only"])
            self.assertFalse(config["inference"]["tracking"]["export_confirmed_only"])
            self.assertEqual(config["inference"]["tracking"]["trajectory_width"], 4)
        self.assertEqual(
            safe["inference"]["tracking"]["hybrid"]["cmc"]["processing_scale"],
            1.0,
        )
        self.assertEqual(
            fast["inference"]["tracking"]["hybrid"]["cmc"]["processing_scale"],
            0.5,
        )

    def test_prediction_config_includes_batch_size(self):
        config = {"model": {"confidence_threshold": 0.3}, "inference": {"mode": "full_image", "batch_size": 7}}

        prediction_config = inference_runner.build_prediction_config(config, [{"id": 0, "name": "ball"}])

        self.assertEqual(prediction_config["inference"]["batch_size"], 7)

    def test_prediction_config_preserves_auto_sahi_batch_tuning(self):
        config = {
            "model": {"confidence_threshold": 0.3},
            "inference": {"mode": "sahi", "batch_size": 8},
            "sahi": {"batch_size": "auto"},
        }

        prediction_config = inference_runner.prediction_config_with_batch(config, 8)

        self.assertEqual(prediction_config["inference"]["batch_size"], 8)
        self.assertEqual(prediction_config["sahi"]["batch_size"], "auto")

    def test_football_output_resolves_default_name_case_insensitively(self):
        categories = [{"id": 0, "name": "Football"}, {"id": 1, "name": "player"}]

        resolved = inference_runner.parse_football_output_config(
            {"inference": {"football_output": {"enabled": True}}},
            categories,
        )

        self.assertTrue(resolved.enabled)
        self.assertEqual(resolved.target_class_ids, frozenset({0}))

    def test_football_output_ids_take_precedence_over_names(self):
        categories = [{"id": 0, "name": "football"}, {"id": 1, "name": "ball"}]
        config = {
            "inference": {
                "football_output": {
                    "enabled": True,
                    "target_class_ids": [1],
                    "target_class_names": ["does-not-exist"],
                }
            }
        }

        resolved = inference_runner.parse_football_output_config(config, categories)

        self.assertEqual(resolved.target_class_ids, frozenset({1}))

    def test_football_output_rejects_unknown_names_and_ids(self):
        categories = [{"id": 0, "name": "football"}]
        configs = [
            {"inference": {"football_output": {"target_class_names": ["soccer-ball"]}}},
            {"inference": {"football_output": {"target_class_ids": [9]}}},
        ]

        for config in configs:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "available categories"):
                inference_runner.parse_football_output_config(config, categories)

    def test_football_coordinate_conversions_preserve_float_precision(self):
        self.assertEqual(
            inference_runner.coco_xywh_to_center_xywh([10.25, 20.5, 5.5, 7.25]),
            [13.0, 24.125, 5.5, 7.25],
        )
        self.assertEqual(
            inference_runner.xyxy_to_center_xywh([10.25, 20.5, 15.75, 27.75]),
            [13.0, 24.125, 5.5, 7.25],
        )

    def test_standard_football_rows_filter_classes_and_keep_nullable_track_id(self):
        categories = [{"id": 0, "name": "football"}, {"id": 1, "name": "player"}]
        output_config = inference_runner.FootballOutputConfig(True, frozenset({0}))
        predictions = [
            {
                "image_id": 7,
                "category_id": 0,
                "bbox": [10.0, 20.0, 4.0, 6.0],
                "score": 0.8,
                "source": "image.jpg",
            },
            {
                "image_id": 8,
                "category_id": 1,
                "bbox": [0.0, 0.0, 20.0, 40.0],
                "score": 0.9,
                "source": "video.mp4",
                "frame_index": 3,
                "timestamp_seconds": 0.1,
            },
            {
                "image_id": 8,
                "category_id": 0,
                "bbox": [30.0, 40.0, 8.0, 10.0],
                "score": 0.7,
                "track_id": 5,
                "source": "video.mp4",
                "frame_index": 3,
                "timestamp_seconds": 0.1,
            },
        ]

        rows = inference_runner.build_standard_football_rows(predictions, output_config, categories)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["kind"], "image")
        self.assertEqual(rows[0]["xywh"], [12.0, 23.0, 4.0, 6.0])
        self.assertIsNone(rows[0]["frame_index"])
        self.assertIsNone(rows[0]["timestamp_seconds"])
        self.assertIsNone(rows[0]["track_id"])
        self.assertEqual(rows[1]["kind"], "video")
        self.assertEqual(rows[1]["xywh"], [34.0, 45.0, 8.0, 10.0])
        self.assertEqual(rows[1]["track_id"], 5)
        self.assertEqual(
            set(rows[0]),
            {
                "kind",
                "source",
                "image_id",
                "frame_index",
                "timestamp_seconds",
                "category_id",
                "category_name",
                "score",
                "xywh",
                "track_id",
            },
        )

    def test_temporal_football_rows_use_anchor_and_threshold(self):
        categories = [{"id": 0, "name": "football"}, {"id": 1, "name": "player"}]
        output_config = inference_runner.FootballOutputConfig(True, frozenset({0}))
        temporal_rows = [
            {
                "source": "/data/sequence/frame_0003.jpg",
                "anchor_frame_index": 3,
                "detections": {
                    "boxes": [[10.0, 20.0, 14.0, 26.0], [0.0, 0.0, 8.0, 8.0], [2.0, 4.0, 6.0, 10.0]],
                    "scores": [0.5, 0.9, 0.49],
                    "labels": [0, 1, 0],
                },
            }
        ]

        rows = inference_runner.build_temporal_football_rows(
            temporal_rows,
            output_config,
            categories,
            confidence_threshold=0.5,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "temporal")
        self.assertEqual(rows[0]["source"], "/data/sequence/frame_0003.jpg")
        self.assertEqual(rows[0]["frame_index"], 3)
        self.assertEqual(rows[0]["xywh"], [12.0, 23.0, 4.0, 6.0])
        self.assertIsNone(rows[0]["image_id"])
        self.assertIsNone(rows[0]["timestamp_seconds"])
        self.assertIsNone(rows[0]["track_id"])

    def test_write_empty_football_jsonl_creates_empty_utf8_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / inference_runner.FOOTBALL_PREDICTIONS_FILENAME

            inference_runner.write_predictions_jsonl(path, [])

            self.assertTrue(path.is_file())
            self.assertEqual(path.read_text(encoding="utf-8"), "")
            unicode_path = Path(tmp) / "unicode.jsonl"
            inference_runner.write_predictions_jsonl(unicode_path, [{"category_name": "足球"}])
            self.assertEqual(
                json.loads(unicode_path.read_text(encoding="utf-8"))["category_name"],
                "足球",
            )

    def test_write_football_output_reports_enabled_and_disabled_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enabled_summary = inference_runner.write_football_predictions_output(
                root,
                [{"category_name": "football"}],
                inference_runner.FootballOutputConfig(True, frozenset({0})),
            )
            disabled_summary = inference_runner.write_football_predictions_output(
                root / "disabled",
                [],
                inference_runner.FootballOutputConfig(False, frozenset()),
            )

            self.assertEqual(enabled_summary["football_prediction_count"], 1)
            self.assertEqual(
                enabled_summary["football_predictions_file"],
                inference_runner.FOOTBALL_PREDICTIONS_FILENAME,
            )
            self.assertTrue((root / inference_runner.FOOTBALL_PREDICTIONS_FILENAME).is_file())
            self.assertEqual(
                disabled_summary,
                {"football_prediction_count": 0, "football_predictions_file": None},
            )
            self.assertFalse((root / "disabled").exists())

    def test_football_output_adds_one_estimated_file(self):
        output_dir = PROJECT_DIR / "runs" / "test-estimate"
        disabled = {"inference": {"football_output": {"enabled": False}}}
        enabled = {"inference": {"football_output": {"enabled": True}}}

        disabled_estimate = inference_runner.estimate_outputs([], output_dir, disabled)
        enabled_estimate = inference_runner.estimate_outputs([], output_dir, enabled)

        self.assertEqual(
            enabled_estimate["estimated_total_files"],
            disabled_estimate["estimated_total_files"] + 1,
        )

    def test_canonical_output_estimates_manifest_four_bundle_files_and_media_time(self):
        item = inference_runner.SourceItem(
            source="fixture.mp4", kind="video", is_url=False, local_path=None
        )
        common_inference = {
            "video": {"max_seconds": 1, "save_video": False},
            "football_output": {"enabled": False},
        }
        disabled = {
            "runtime": {"time_estimate": {"enabled": False}},
            "inference": {**common_inference, "canonical_output": {"enabled": False}},
        }
        enabled = {
            "runtime": {"time_estimate": {"enabled": False}},
            "inference": {**common_inference, "canonical_output": {"enabled": True}},
        }

        disabled_estimate = inference_runner.estimate_outputs([item], Path("runs/disabled"), disabled)
        enabled_estimate = inference_runner.estimate_outputs([item], Path("runs/enabled"), enabled)

        self.assertEqual(
            enabled_estimate["estimated_total_files"],
            disabled_estimate["estimated_total_files"] + 5,
        )
        self.assertTrue(enabled_estimate["canonical_v2"]["enabled"])
        self.assertEqual(enabled_estimate["canonical_v2"]["video_bundles"], 1)
        self.assertGreater(enabled_estimate["canonical_v2"]["estimated_bytes"], 0)
        self.assertGreater(enabled_estimate["canonical_v2"]["estimated_transcode_seconds"], 0)

    def test_temporal_mode_excludes_canonical_video_estimate(self):
        item = inference_runner.SourceItem(
            source="fixture.mp4", kind="video", is_url=False, local_path=None
        )
        config = {
            "model": {
                "motion": {
                    "enabled": True,
                    "type": "tracknet_v5",
                    "temporal": {"mode": "real"},
                }
            },
            "inference": {
                "video": {"max_seconds": 1},
                "canonical_output": {"enabled": True},
            },
        }

        estimate = inference_runner.estimate_outputs([item], Path("runs/temporal"), config)

        self.assertFalse(estimate["canonical_v2"]["enabled"])
        self.assertEqual(estimate["canonical_v2"]["video_bundles"], 0)

    def test_decoded_timestamp_fallback_is_strictly_monotonic(self):
        class Capture:
            def __init__(self, milliseconds):
                self.milliseconds = milliseconds

            def get(self, _property):
                return self.milliseconds

        cv2_stub = SimpleNamespace(CAP_PROP_POS_MSEC=1)
        timestamp, source = inference_runner.decoded_frame_timestamp(
            Capture(0.0), cv2_stub, 30, 30.0, 1.1
        )
        self.assertEqual(source, "nominal_fps")
        self.assertAlmostEqual(timestamp, 1.1 + 1.0 / 30.0)

        timestamp, source = inference_runner.decoded_frame_timestamp(
            Capture(1200.0), cv2_stub, 31, 30.0, 1.1
        )
        self.assertEqual(source, "decoder_pts")
        self.assertAlmostEqual(timestamp, 1.2)

    def test_stage_timing_summary_includes_sahi_and_recheck_ratios(self):
        model = SimpleNamespace()
        model._rf_detr_workload_counters = {"model_batches": 3}
        inference_runner.record_inference_timing_rows(
            model,
            [
                {
                    "elapsed_seconds": 2.0,
                    "model_forward_seconds": 1.5,
                    "sahi_model_forward_seconds": 1.0,
                    "recheck_model_forward_seconds": 0.5,
                    "postprocess_seconds": 0.5,
                    "crop_seconds": 0.1,
                    "host_preprocess_seconds": 0.2,
                    "h2d_seconds": 0.03,
                    "resize_normalize_seconds": 0.07,
                    "device_preprocess_seconds": 0.1,
                    "orchestration_seconds": 0.1,
                    "exclusive_postprocess_seconds": 0.4,
                    "requested_slice_batch_size": 16,
                    "effective_slice_batch_sizes": [8],
                    "observed_slice_batch_sizes": [3, 8],
                }
            ],
        )

        summary = inference_runner.summarize_inference_timing_rows(model)

        self.assertEqual(summary["images_or_frames"], 1)
        self.assertEqual(summary["runtime_units"], 3)
        self.assertAlmostEqual(summary["model_forward_ratio"], 0.75)
        self.assertAlmostEqual(summary["sahi_model_forward_ratio"], 0.5)
        self.assertAlmostEqual(summary["recheck_model_forward_ratio"], 0.25)
        self.assertEqual(summary["requested_sahi_batch_sizes"], [16])
        self.assertEqual(summary["effective_sahi_batch_sizes"], [8])
        self.assertEqual(summary["observed_sahi_batch_sizes"], [3, 8])
        self.assertAlmostEqual(summary["crop_seconds"], 0.1)
        self.assertAlmostEqual(summary["host_preprocess_seconds"], 0.2)
        self.assertAlmostEqual(summary["h2d_seconds"], 0.03)
        self.assertAlmostEqual(summary["resize_normalize_seconds"], 0.07)
        self.assertAlmostEqual(summary["exclusive_postprocess_seconds"], 0.4)

    def test_final_prediction_filter_uses_fused_threshold_for_all_classes(self):
        config = {
            "model": {"confidence_threshold": 0.25},
            "inference": {"mode": "sahi"},
            "sahi": {"recheck": {"enabled": True, "fused_confidence_threshold": 0.5}},
        }
        predictions = [
            {"category_id": 0, "score": 0.49},
            {"category_id": 0, "score": 0.5},
            {"category_id": 1, "score": 0.8},
        ]

        filtered = inference_runner.filter_final_inference_predictions(predictions, config)

        self.assertEqual([(row["category_id"], row["score"]) for row in filtered], [(0, 0.5), (1, 0.8)])

    def test_final_prediction_filter_is_inactive_without_sahi_and_recheck(self):
        predictions = [{"category_id": 1, "score": 0.1}]
        configs = [
            {"inference": {"mode": "full_image"}, "sahi": {"recheck": {"enabled": True, "fused_confidence_threshold": 0.5}}},
            {"inference": {"mode": "sahi"}, "sahi": {"recheck": {"enabled": False, "fused_confidence_threshold": 0.5}}},
        ]

        for config in configs:
            with self.subTest(config=config):
                self.assertEqual(inference_runner.filter_final_inference_predictions(predictions, config), predictions)

    def test_image_batch_uses_same_filtered_predictions_for_output_and_render(self):
        from PIL import Image

        config = {
            "model": {"confidence_threshold": 0.25},
            "inference": {"mode": "sahi", "batch_size": 1},
            "sahi": {
                "batch_size": 1,
                "recheck": {"enabled": True, "fused_confidence_threshold": 0.5},
            },
        }
        predictions = [
            {"image_id": 1, "category_id": 0, "bbox": [1, 1, 5, 5], "score": 0.49},
            {"image_id": 1, "category_id": 1, "bbox": [2, 2, 5, 5], "score": 0.5},
        ]
        rendered_scores = []

        def capture_render(image, rows, *_args, **_kwargs):
            rendered_scores.extend(row["score"] for row in rows)
            return image.convert("RGB")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (20, 20), color=(255, 255, 255)).save(image_path)
            item = inference_runner.SourceItem(source=str(image_path), kind="image", local_path=image_path)
            with patch.object(
                inference_runner.evaluator,
                "predict_images_rfdetr",
                return_value=([predictions], [], []),
            ), patch.object(inference_runner, "draw_predictions", side_effect=capture_render):
                rows, outputs, _ = inference_runner.predict_image_files_batch(
                    [item], 1, object(), config, [], root / "outputs", [], 1
                )

        self.assertEqual([row["score"] for row in rows], [0.5])
        self.assertEqual(rendered_scores, [0.5])
        self.assertEqual(outputs[0]["predictions"], 1)

    def test_draw_predictions_without_tracks_is_unchanged(self):
        from PIL import Image

        image = Image.new("RGB", (40, 40), (10, 20, 30))
        predictions = [{"category_id": 1, "bbox": [5, 5, 10, 10], "score": 0.9}]
        categories = [{"id": 1, "name": "football"}]

        baseline = inference_runner.draw_predictions(image, predictions, categories, [])
        with_defaults = inference_runner.draw_predictions(image, predictions, categories, [], None, None)

        self.assertEqual(baseline.tobytes(), with_defaults.tobytes())

    def test_build_video_row_has_expected_keys(self):
        from types import SimpleNamespace

        window = SimpleNamespace(start_seconds=0.0, end_seconds=2.0, effective_end_seconds=2.0)
        row = inference_runner.build_video_row(
            {"category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}, "v.mp4", 3, 3, 30.0, window
        )

        self.assertEqual(row["source"], "v.mp4")
        self.assertEqual(row["frame_index"], 3)
        self.assertAlmostEqual(row["timestamp_seconds"], 0.1)
        for key in ("segment_frame_index", "segment_timestamp_seconds", "video_start_seconds", "video_end_seconds", "video_effective_end_seconds"):
            self.assertIn(key, row)

    def test_hybrid_render_and_export_confirmation_switches_are_independent(self):
        rows = [
            {
                "category_id": 1,
                "frame_index": 0,
                "track_id": 1,
                "track_final_confirmed": True,
            },
            {
                "category_id": 1,
                "frame_index": 0,
                "track_id": 2,
                "track_final_confirmed": False,
            },
            {
                "category_id": 1,
                "frame_index": 0,
                "track_id": None,
                "track_final_confirmed": False,
            },
            {"category_id": 0, "frame_index": 0, "track_id": None},
        ]
        for render_only in (False, True):
            for export_only in (False, True):
                with self.subTest(render_only=render_only, export_only=export_only):
                    cfg = inference_runner.video_tracking.TrackingConfig(
                        enabled=True,
                        algorithm="hybrid",
                        target_class_ids={1},
                        render_confirmed_only=render_only,
                        export_confirmed_only=export_only,
                    )
                    published, suppressed = inference_runner.filter_confirmed_hybrid_exports(rows, cfg)
                    visible = [
                        row
                        for row in rows
                        if inference_runner.prediction_visible_for_confirmed_render(row, cfg, {1})
                    ]

                    self.assertEqual(len(published), 2 if export_only else 4)
                    self.assertEqual(suppressed, 2 if export_only else 0)
                    self.assertEqual(len(visible), 2 if render_only else 4)

    def test_committed_first_hit_keeps_raw_lifecycle_but_renders_stable_id(self):
        from PIL import Image

        packet = {
            "confirmed_track_ids": frozenset({7}),
            "detections": [{
                "category_id": 1,
                "bbox": [5, 5, 8, 8],
                "score": 0.9,
                "track_id": 7,
                "track_confirmed": False,
                "track_status": "tentative",
            }],
        }
        rows = inference_runner.hybrid_packet_detections(packet)
        self.assertFalse(rows[0]["track_confirmed"])
        self.assertEqual(rows[0]["track_status"], "tentative")
        self.assertTrue(rows[0]["track_final_confirmed"])

        cfg = inference_runner.video_tracking.TrackingConfig(
            enabled=True,
            algorithm="hybrid",
            target_class_ids={1},
            render_confirmed_only=True,
            label_track_id=True,
        )
        draw = unittest.mock.Mock()
        draw.textbbox.return_value = (0, 0, 20, 10)
        with patch.object(inference_runner.ImageDraw, "Draw", return_value=draw):
            inference_runner.draw_predictions(
                Image.new("RGB", (30, 30)),
                rows,
                [{"id": 1, "name": "football"}],
                [],
                tracking_cfg=cfg,
                current_frame_index=0,
                confirmed_track_ids={7},
            )

        self.assertIn("football #7", draw.text.call_args.args[1])

    def test_confirmed_export_filter_keeps_images_and_non_target_video_rows(self):
        cfg = inference_runner.video_tracking.TrackingConfig(
            enabled=True,
            algorithm="hybrid",
            target_class_ids={1},
            export_confirmed_only=True,
        )
        rows = [
            {"category_id": 1, "image_id": 1, "track_id": None},
            {"category_id": 0, "frame_index": 2, "track_id": None},
            {"category_id": 1, "frame_index": 2, "track_id": None},
        ]

        published, suppressed = inference_runner.filter_confirmed_hybrid_exports(rows, cfg)

        self.assertEqual(published, rows[:2])
        self.assertEqual(suppressed, 1)

    def test_load_model_attaches_motion_only_when_enabled(self):
        rf_model = SimpleNamespace(model=object())
        model_cls = unittest.mock.Mock(return_value=rf_model)
        attached = unittest.mock.Mock()
        checkpoint_guard = unittest.mock.Mock()
        checkpoint_loader = unittest.mock.Mock()
        motion_module = ModuleType("rf_detr_motion")
        motion_module.attach_motion_module = attached
        motion_module.assert_motion_checkpoint_compatible = checkpoint_guard
        motion_module.load_motion_checkpoint_weights = checkpoint_loader

        with patch.object(inference_runner.trainer, "get_model_class", return_value=model_cls), patch.object(
            inference_runner.trainer, "build_model_kwargs", return_value={"device": "cpu"}
        ), patch.dict(sys.modules, {"rf_detr_motion": motion_module}):
            result = inference_runner.load_rfdetr_model({"model": {"size": "medium", "motion": {"enabled": True}}})

        self.assertIs(result, rf_model)
        model_cls.assert_called_once_with(device="cpu")
        attached.assert_called_once_with(rf_model.model, {"enabled": True})
        checkpoint_guard.assert_called_once()
        checkpoint_loader.assert_called_once_with(rf_model.model, None)

    def test_load_model_skips_motion_attachment_when_disabled(self):
        rf_model = SimpleNamespace(model=object())
        model_cls = unittest.mock.Mock(return_value=rf_model)

        with patch.object(inference_runner.trainer, "get_model_class", return_value=model_cls), patch.object(
            inference_runner.trainer, "build_model_kwargs", return_value={}
        ):
            result = inference_runner.load_rfdetr_model({"model": {"size": "medium", "motion": {"enabled": False}}})

        self.assertIs(result, rf_model)
        model_cls.assert_called_once_with()

    def test_all_trajectory_presets_disable_predicted_trajectory_by_default(self):
        config_dir = PROJECT_DIR / "config"
        trajectory_presets = 0
        for path in sorted(config_dir.glob("rf_detr_inference*.yaml")):
            config = inference_runner.load_yaml(path)
            tracking = config.get("inference", {}).get("tracking", {})
            if "draw_trajectory" not in tracking:
                continue
            trajectory_presets += 1
            with self.subTest(config=path.name):
                self.assertIn("draw_predicted_trajectory", tracking)
                self.assertFalse(tracking["draw_predicted_trajectory"])
        # New derived presets may inherit trajectory rendering.  The contract
        # is the per-preset safety setting above, not a frozen file count.
        self.assertGreaterEqual(trajectory_presets, 17)

    def test_all_standalone_inference_presets_declare_canonical_v2_defaults(self):
        config_dir = PROJECT_DIR / "config"
        expected = {
            "enabled": True,
            "directory": "canonical_v2",
            "ffmpeg_path": "auto",
            "ffprobe_path": "auto",
            "media": {
                "video_codec": "libx264",
                "crf": 18,
                "preset": "medium",
                "pixel_format": "yuv420p",
                "audio_codec": "aac",
                "audio_bitrate": "192k",
            },
        }
        standalone_paths = []
        overlay_paths = []

        for path in sorted(config_dir.glob("rf_detr_inference*.yaml")):
            raw_text = path.read_text(encoding="utf-8")
            if "\nextends:" in raw_text:
                overlay_paths.append(path.name)
                with self.subTest(config=path.name):
                    config = inference_runner.load_yaml(path)
                    self.assertEqual(config["inference"]["canonical_output"], expected)
                continue
            standalone_paths.append(path.name)
            with self.subTest(config=path.name):
                config = inference_runner.load_yaml(path)
                self.assertEqual(config["inference"]["canonical_output"], expected)
                self.assertEqual(raw_text.count("\n  canonical_output:\n"), 1)

        self.assertEqual(len(standalone_paths), 19)
        self.assertEqual(
            overlay_paths,
            [
                "rf_detr_inference_medium_p2_performance_fast.yaml",
                "rf_detr_inference_medium_p2_performance_safe.yaml",
                "rf_detr_inference_smoke_temporal_tracknet_v5.yaml",
            ],
        )

    def test_all_tracking_presets_explicitly_declare_confirmation_render_controls(self):
        config_dir = PROJECT_DIR / "config"
        tracking_paths = []
        hybrid_paths = set()
        required_keys = (
            "draw_predicted_center",
            "render_confirmed_only",
            "export_confirmed_only",
        )

        for path in sorted(config_dir.glob("rf_detr_inference*.yaml")):
            raw_text = path.read_text(encoding="utf-8")
            if "\n  tracking:\n" not in raw_text:
                continue
            tracking_paths.append(path)
            config = inference_runner.load_yaml(path)
            tracking = config["inference"]["tracking"]
            algorithm = tracking.get("algorithm", "circle")

            with self.subTest(config=path.name):
                for key in required_keys:
                    declaration = f"\n    {key}:"
                    self.assertEqual(raw_text.count(declaration), 1)
                    declaration_line = next(
                        index
                        for index, line in enumerate(raw_text.splitlines())
                        if line.startswith(f"    {key}:")
                    )
                    preceding_line = raw_text.splitlines()[declaration_line - 1].strip()
                    self.assertTrue(preceding_line.startswith("#"), f"{key} needs an inline schema comment")

                self.assertFalse(tracking["draw_predicted_center"])
                self.assertNotIn("beam_width", raw_text)
                if algorithm == "hybrid":
                    hybrid_paths.add(path.name)
                    self.assertTrue(tracking["render_confirmed_only"])
                    self.assertFalse(tracking["export_confirmed_only"])
                    self.assertEqual(tracking["trajectory_width"], 4)
                else:
                    self.assertFalse(tracking["render_confirmed_only"])
                    self.assertFalse(tracking["export_confirmed_only"])

        self.assertEqual(len(tracking_paths), 21)
        self.assertEqual(
            hybrid_paths,
            {
                "rf_detr_inference_medium_p2_performance_fast.yaml",
                "rf_detr_inference_medium_p2_performance_safe.yaml",
                "rf_detr_inference_medium_p2_sahi160_recheck320_hybrid.yaml",
                "rf_detr_inference_medium_p2_sahi320_recheck640_hybrid.yaml",
            },
        )

    def test_all_inference_configs_declare_optional_architecture_blocks(self):
        config_dir = PROJECT_DIR / "config"
        p2_video_preset = "rf_detr_inference_medium_p2_video_1984090152231178242_003.yaml"
        combined_preset = "rf_detr_inference_medium_p2_tensorrt_sahi160_recheck320_ocsort_6videos.yaml"
        tensorrt_presets = {
            combined_preset,
            "rf_detr_inference_medium_p2_performance_fast.yaml",
            "rf_detr_inference_medium_p2_sahi160_recheck320_hybrid.yaml",
            "rf_detr_inference_tracknet_tensorrt_fp16_example.yaml",
            "rf_detr_inference_large_p2_tensorrt_fp16_smoke.yaml",
            "rf_detr_inference_small_p2.yaml",
        }
        bf16_pytorch_presets = {
            "rf_detr_inference_medium_p2_performance_fast.yaml",
            "rf_detr_inference_medium_p2_performance_safe.yaml",
            "rf_detr_inference_medium_p2_sahi160_recheck320_hybrid.yaml",
            "rf_detr_inference_medium_p2_sahi320_recheck640_hybrid.yaml",
        }
        real_temporal_tracknet_presets = {
            "rf_detr_inference_small_tracknet_v5.yaml",
            "rf_detr_inference_small_p2_tracknet_v5.yaml",
            "rf_detr_inference_smoke_temporal_tracknet_v5.yaml",
        }
        legacy_tracknet_preset = "rf_detr_inference_tracknet_tensorrt_fp16_example.yaml"
        for path in sorted(config_dir.glob("rf_detr_inference*.yaml")):
            with self.subTest(config=path.name):
                config = inference_runner.load_yaml(path)
                model = config["model"]
                self.assertIn("p2", model)
                self.assertIn("motion", model)
                self.assertIn("inference_optimization", model)
                self.assertIn("enabled", model["p2"])
                self.assertIn("enabled", model["motion"])
                football_output = config["inference"]["football_output"]
                self.assertTrue(football_output["enabled"])
                self.assertEqual(football_output["target_class_ids"], [])
                self.assertEqual(football_output["target_class_names"], ["football"])
                resolved_football = inference_runner.parse_football_output_config(
                    config,
                    inference_runner.build_categories(config),
                )
                self.assertTrue(resolved_football.target_class_ids)
                expected_backend = "tensorrt" if path.name in tensorrt_presets else "pytorch"
                self.assertEqual(model["inference_optimization"]["backend"], expected_backend)
                expected_pytorch_precision = "bf16" if path.name in bf16_pytorch_presets else "fp32"
                self.assertEqual(
                    model["inference_optimization"]["pytorch"]["precision"],
                    expected_pytorch_precision,
                )
                self.assertIn(model["inference_optimization"]["tensorrt"]["precision"], {"fp16", "bf16"})
                if path.name == combined_preset:
                    self.assertEqual(model["inference_optimization"]["tensorrt"]["precision"], "fp16")
                    self.assertTrue(model["p2"]["enabled"])
                    self.assertFalse(model["motion"]["enabled"])
                    self.assertEqual(len(config["inference"]["sources"]), 6)
                    self.assertEqual(config["inference"]["mode"], "sahi")
                    self.assertTrue(config["inference"]["tracking"]["enabled"])
                    self.assertEqual(config["inference"]["tracking"]["algorithm"], "ocsort")
                    self.assertEqual(
                        (config["sahi"]["slice_height"], config["sahi"]["slice_width"]),
                        (160, 160),
                    )
                    self.assertTrue(config["sahi"]["recheck"]["enabled"])
                    self.assertEqual(config["sahi"]["recheck"]["crop_size"], 320)
                if path.name == p2_video_preset:
                    self.assertTrue(model["p2"]["enabled"])
                if path.name == "rf_detr_inference_tracknet_tensorrt_fp16_example.yaml":
                    self.assertFalse(model["p2"]["enabled"])
                if path.name in real_temporal_tracknet_presets:
                    self.assertTrue(model["motion"]["enabled"])
                    self.assertEqual(model["motion"]["temporal"]["fallback_mode"], "real")
                    self.assertEqual(model["motion"]["temporal"]["mode"], "real")
                elif path.name == legacy_tracknet_preset:
                    self.assertTrue(model["motion"]["enabled"])
                    self.assertEqual(model["motion"]["temporal"]["fallback_mode"], "identity")
                else:
                    self.assertFalse(model["motion"]["enabled"])

    @staticmethod
    def _cli_args(**overrides):
        base = dict(
            yes=False, dry_run=False, source=None, output_dir=None, checkpoint=None, device=None,
            confidence_threshold=None, max_sources=None, max_images=None, max_videos=None,
            batch_size=None, video_batch_size=None, max_seconds=None, video_start_time=None, video_end_time=None,
            track=False, no_track=False, track_radius=None, track_velocity=False,
            inference_backend=None, inference_precision=None, tensorrt_engine=None,
            tensorrt_cache_dir=None, tensorrt_force_rebuild=False,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_cli_track_overrides_enable_and_tune(self):
        config = {}
        inference_runner.apply_cli_overrides(config, self._cli_args(track=True, track_radius=120.0, track_velocity=True))
        tracking = config["inference"]["tracking"]
        self.assertTrue(tracking["enabled"])
        self.assertEqual(tracking["radius_pixels"], 120.0)
        self.assertTrue(tracking["use_velocity_prediction"])

    def test_cli_no_track_disables_and_wins(self):
        config = {}
        inference_runner.apply_cli_overrides(config, self._cli_args(track=True, no_track=True))
        self.assertFalse(config["inference"]["tracking"]["enabled"])

    def test_cli_without_track_flags_adds_no_tracking_key(self):
        config = {}
        inference_runner.apply_cli_overrides(config, self._cli_args())
        self.assertNotIn("tracking", config.get("inference", {}))

    def test_cli_canonical_output_overrides_are_applied_together(self):
        config = {}
        inference_runner.apply_cli_overrides(
            config,
            self._cli_args(
                canonical_output=False,
                canonical_output_dir="vlm_packets",
                ffmpeg_path="/opt/media/ffmpeg",
                ffprobe_path="/opt/media/ffprobe",
            ),
        )
        self.assertEqual(
            config["inference"]["canonical_output"],
            {
                "enabled": False,
                "directory": "vlm_packets",
                "ffmpeg_path": "/opt/media/ffmpeg",
                "ffprobe_path": "/opt/media/ffprobe",
            },
        )

    def test_cli_without_canonical_flags_adds_no_empty_mapping(self):
        config = {}
        inference_runner.apply_cli_overrides(config, self._cli_args())
        self.assertNotIn("canonical_output", config.get("inference", {}))

    def test_cli_selects_tensorrt_bf16_and_artifact_options(self):
        config = {
            "model": {
                "inference_optimization": {
                    "tensorrt": {"manifest_path": "stale.engine.manifest.json"}
                }
            }
        }
        inference_runner.apply_cli_overrides(
            config,
            self._cli_args(
                inference_backend="tensorrt",
                inference_precision="bf16",
                tensorrt_engine="model.engine",
                tensorrt_cache_dir="trt-cache",
                tensorrt_force_rebuild=True,
            ),
        )

        settings = config["model"]["inference_optimization"]
        self.assertEqual(settings["backend"], "tensorrt")
        self.assertEqual(settings["tensorrt"]["precision"], "bf16")
        self.assertEqual(settings["tensorrt"]["engine_path"], "model.engine")
        self.assertEqual(settings["tensorrt"]["manifest_path"], "")
        self.assertEqual(settings["tensorrt"]["cache_dir"], "trt-cache")
        self.assertTrue(settings["tensorrt"]["force_rebuild"])


if __name__ == "__main__":
    unittest.main()
