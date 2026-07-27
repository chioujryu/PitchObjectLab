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

    def test_prediction_config_includes_batch_size(self):
        config = {"model": {"confidence_threshold": 0.3}, "inference": {"mode": "full_image", "batch_size": 7}}

        prediction_config = inference_runner.build_prediction_config(config, [{"id": 0, "name": "ball"}])

        self.assertEqual(prediction_config["inference"]["batch_size"], 7)

    def test_stage_timing_summary_includes_sahi_and_recheck_ratios(self):
        model = SimpleNamespace()
        inference_runner.record_inference_timing_rows(
            model,
            [
                {
                    "elapsed_seconds": 2.0,
                    "model_forward_seconds": 1.5,
                    "sahi_model_forward_seconds": 1.0,
                    "recheck_model_forward_seconds": 0.5,
                    "postprocess_seconds": 0.5,
                }
            ],
        )

        summary = inference_runner.summarize_inference_timing_rows(model)

        self.assertEqual(summary["images_or_frames"], 1)
        self.assertAlmostEqual(summary["model_forward_ratio"], 0.75)
        self.assertAlmostEqual(summary["sahi_model_forward_ratio"], 0.5)
        self.assertAlmostEqual(summary["recheck_model_forward_ratio"], 0.25)

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

    def test_all_inference_configs_declare_optional_architecture_blocks(self):
        config_dir = PROJECT_DIR / "config"
        p2_video_preset = "rf_detr_inference_medium_p2_video_1984090152231178242_003.yaml"
        tensorrt_presets = {
            "rf_detr_inference_tracknet_tensorrt_fp16_example.yaml",
            "rf_detr_inference_large_p2_tensorrt_fp16_smoke.yaml",
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
                expected_backend = "tensorrt" if path.name in tensorrt_presets else "pytorch"
                self.assertEqual(model["inference_optimization"]["backend"], expected_backend)
                self.assertEqual(model["inference_optimization"]["pytorch"]["precision"], "fp32")
                self.assertIn(model["inference_optimization"]["tensorrt"]["precision"], {"fp16", "bf16"})
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
