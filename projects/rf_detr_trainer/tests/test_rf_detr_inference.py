from pathlib import Path
import sys
import tempfile
import unittest


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

    @staticmethod
    def _cli_args(**overrides):
        from types import SimpleNamespace

        base = dict(
            yes=False, dry_run=False, source=None, output_dir=None, checkpoint=None, device=None,
            confidence_threshold=None, max_sources=None, max_images=None, max_videos=None,
            batch_size=None, video_batch_size=None, max_seconds=None, video_start_time=None, video_end_time=None,
            track=False, no_track=False, track_radius=None, track_velocity=False,
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


if __name__ == "__main__":
    unittest.main()
