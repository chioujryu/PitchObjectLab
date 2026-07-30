"""Unit tests for circle-based football tracking and its video-pipeline integration.

These tests use scripted detections and a tiny synthetic video, so no RF-DETR
model is loaded.
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rf_detr_video_tracking as vt

CATEGORIES = [
    {"id": 0, "name": "standing_player"},
    {"id": 1, "name": "football"},
    {"id": 2, "name": "goal"},
]


def ball(center_x, center_y, size=10.0, score=0.9, category_id=1):
    """Build a COCO-style prediction whose bbox is centered at (center_x, center_y)."""
    return {
        "category_id": category_id,
        "bbox": [center_x - size / 2.0, center_y - size / 2.0, size, size],
        "score": score,
        "area": size * size,
    }


def make_config(**overrides):
    cfg = vt.TrackingConfig(enabled=True, target_class_ids={1}, radius_pixels=50.0)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class FootballTrackerTest(unittest.TestCase):
    def test_same_ball_inside_radius_keeps_track_id(self):
        tracker = vt.FootballTracker(make_config())
        first = tracker.update(0, [ball(100, 100)])
        second = tracker.update(1, [ball(110, 100)])  # 10 px away, inside r=50
        self.assertEqual(first[0]["track_id"], 1)
        self.assertEqual(second[0]["track_id"], 1)

    def test_recenter_follows_moving_ball(self):
        tracker = vt.FootballTracker(make_config())
        ids = [tracker.update(i, [ball(100 + 40 * i, 100)])[0]["track_id"] for i in range(4)]
        self.assertEqual(ids, [1, 1, 1, 1])  # 40 px steps stay inside the re-centered circle
        self.assertAlmostEqual(tracker.tracks[0].center_x, 100 + 40 * 3)

    def test_missing_frames_then_rematch_keeps_track_id(self):
        tracker = vt.FootballTracker(make_config())  # max_missing_frames defaults to None (never expire)
        tracker.update(0, [ball(100, 100)])
        for frame in (1, 2, 3):
            tracker.update(frame, [])  # gaps: circle held at last center
        rematch = tracker.update(4, [ball(120, 100)])  # 20 px from held center, inside r=50
        self.assertEqual(rematch[0]["track_id"], 1)

    def test_radius_growth_catches_far_jump(self):
        sequence = [(0, ball(100, 100)), (1, []), (2, []), (3, ball(200, 100))]  # 100 px jump
        grown = vt.FootballTracker(make_config(radius_growth_per_missing_frame=30.0))
        plain = vt.FootballTracker(make_config(radius_growth_per_missing_frame=0.0))
        for tracker in (grown, plain):
            rows = None
            for frame, preds in sequence:
                rows = tracker.update(frame, preds if isinstance(preds, list) else [preds])
            tracker._last_rows = rows
        # grown radius = 50 + 30*2 = 110 >= 100 -> same track; plain stays at 50 -> new track
        self.assertEqual(grown._last_rows[0]["track_id"], 1)
        self.assertEqual(plain._last_rows[0]["track_id"], 2)

    def test_velocity_gate_catches_fast_linear_ball(self):
        sequence = [(0, [ball(100, 100)]), (1, [ball(140, 100)]), (2, []), (3, [ball(220, 100)])]
        moving = vt.FootballTracker(make_config(use_velocity_prediction=True))
        static = vt.FootballTracker(make_config(use_velocity_prediction=False))
        for tracker in (moving, static):
            rows = None
            for frame, preds in sequence:
                rows = tracker.update(frame, preds)
            tracker._last_rows = rows
        # velocity 40 px/frame, elapsed 2 -> gate at 220 hits the ball; static gate stays at 140
        self.assertEqual(moving._last_rows[0]["track_id"], 1)
        self.assertEqual(static._last_rows[0]["track_id"], 2)

    def test_ball_outside_radius_creates_new_track(self):
        tracker = vt.FootballTracker(make_config())
        tracker.update(0, [ball(100, 100)])
        rows = tracker.update(1, [ball(200, 100)])  # 100 px away, outside r=50
        self.assertEqual(rows[0]["track_id"], 2)

    def test_multiple_balls_use_nearest_unused_track(self):
        tracker = vt.FootballTracker(make_config())
        tracker.update(0, [ball(100, 100), ball(200, 100)])  # track 1 @100, track 2 @200
        rows = tracker.update(1, [ball(190, 100), ball(110, 100)])
        by_center = {round(vt.bbox_center(row)[0]): row["track_id"] for row in rows}
        self.assertEqual(by_center[190], 2)  # near track 2
        self.assertEqual(by_center[110], 1)  # near track 1

    def test_ignores_non_target_classes(self):
        tracker = vt.FootballTracker(make_config())
        rows = tracker.update(0, [ball(100, 100, category_id=0), ball(300, 300, category_id=1)])
        player_row = next(row for row in rows if row["category_id"] == 0)
        football_row = next(row for row in rows if row["category_id"] == 1)
        self.assertIsNone(player_row["track_id"])
        self.assertEqual(football_row["track_id"], 1)

    def test_min_hits_confirmation(self):
        tracker = vt.FootballTracker(make_config(min_hits=3))
        confirmations = [tracker.update(i, [ball(100 + 5 * i, 100)])[0]["track_confirmed"] for i in range(3)]
        self.assertEqual(confirmations, [False, False, True])
        self.assertEqual(tracker.tracks[0].hits, 3)

    def test_velocity_union_gate_catches_stopped_ball(self):
        # O5: with velocity on, a ball that moved fast then stopped during a gap must still match.
        tracker = vt.FootballTracker(make_config(use_velocity_prediction=True))
        tracker.update(0, [ball(100, 100)])
        tracker.update(1, [ball(140, 100)])  # establishes ~40 px/frame velocity
        tracker.update(2, [])  # gap
        rows = tracker.update(3, [ball(150, 100)])  # ball stopped near 150; predicted gate is at ~220
        self.assertEqual(rows[0]["track_id"], 1)  # static half of the union gate catches it

    def test_global_association_prefers_lower_total_distance(self):
        # O6: greedy-by-score would give the on-track ball a new id; global assignment keeps it.
        tracker = vt.FootballTracker(make_config(radius_pixels=60))
        tracker.update(0, [ball(100, 100), ball(200, 100)])  # track 1 @100, track 2 @200
        rows = tracker.update(1, [ball(150, 100, score=0.9), ball(105, 100, score=0.8)])
        by_center = {round(vt.bbox_center(row)[0]): row["track_id"] for row in rows}
        self.assertEqual(by_center[105], 1)  # ball sitting on track 1 keeps id 1
        self.assertEqual(by_center[150], 2)

    def test_trajectory_points_are_bounded(self):
        # O7: stored trail is capped to trajectory_max_points (deque maxlen).
        tracker = vt.FootballTracker(make_config(trajectory_max_points=3))
        for frame in range(10):
            tracker.update(frame, [ball(100 + frame, 100)])  # 1 px steps stay in radius
        self.assertEqual(len(tracker.tracks[0].points), 3)

    def test_invalid_velocity_smoothing_raises(self):
        with self.assertRaises(ValueError):
            vt.parse_tracking_config({"inference": {"tracking": {"velocity_smoothing": 1.5}}}, CATEGORIES)


class TrackingConfigTest(unittest.TestCase):
    def test_config_defaults_to_football(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"enabled": True}}}, CATEGORIES)
        self.assertEqual(cfg.target_class_ids, {1})

    def test_empty_targets_fall_back_to_football(self):
        raw = {"inference": {"tracking": {"enabled": True, "target_class_ids": [], "target_class_names": []}}}
        cfg = vt.parse_tracking_config(raw, CATEGORIES)
        self.assertEqual(cfg.target_class_ids, {1})

    def test_all_sentinel_means_never_expire(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"max_missing_frames": "all"}}}, CATEGORIES)
        self.assertIsNone(cfg.max_missing_frames)

    def test_max_missing_frames_defaults_to_finite(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"enabled": True}}}, CATEGORIES)
        self.assertEqual(cfg.max_missing_frames, 30)  # omitted -> finite hygiene default

    def test_invalid_radius_raises(self):
        with self.assertRaises(ValueError):
            vt.parse_tracking_config({"inference": {"tracking": {"radius_pixels": 0}}}, CATEGORIES)

    def test_invalid_min_hits_raises(self):
        with self.assertRaises(ValueError):
            vt.parse_tracking_config({"inference": {"tracking": {"min_hits": 0}}}, CATEGORIES)


class BoxmotConfigTest(unittest.TestCase):
    def test_default_algorithm_is_circle(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"enabled": True}}}, CATEGORIES)
        self.assertEqual(cfg.algorithm, "circle")

    def test_algorithm_is_casefolded(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"algorithm": "DeepOcSort"}}}, CATEGORIES)
        self.assertEqual(cfg.algorithm, "deepocsort")

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError) as ctx:
            vt.parse_tracking_config({"inference": {"tracking": {"algorithm": "magic"}}}, CATEGORIES)
        self.assertIn("algorithm", str(ctx.exception))

    def test_circle_algorithm_allowed(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"algorithm": "circle"}}}, CATEGORIES)
        self.assertEqual(cfg.algorithm, "circle")

    def test_nested_boxmot_blocks_map_to_flat_fields(self):
        raw = {
            "inference": {
                "tracking": {
                    "algorithm": "ocsort",
                    "per_class": True,
                    "reid_weights": "weights/osnet_x0_25_msmt17.pt",
                    "ocsort": {
                        "max_age": 50,
                        "asso_threshold": 0.4,
                        "use_byte": True,
                        "Q_xy_scaling": 1.0,
                        "Q_s_scaling": 0.5,
                    },
                    "bytetrack": {"track_buffer": 40},
                    "botsort": {"with_reid": False},
                    "deepocsort": {"iou_threshold": 0.2},
                }
            }
        }
        cfg = vt.parse_tracking_config(raw, CATEGORIES)
        self.assertEqual(cfg.ocsort_max_age, 50)
        self.assertAlmostEqual(cfg.ocsort_asso_threshold, 0.4)
        self.assertIs(cfg.ocsort_use_byte, True)
        self.assertAlmostEqual(cfg.ocsort_q_xy_scaling, 1.0)
        self.assertAlmostEqual(cfg.ocsort_q_s_scaling, 0.5)
        self.assertEqual(cfg.bytetrack_track_buffer, 40)
        self.assertIs(cfg.botsort_with_reid, False)
        self.assertAlmostEqual(cfg.deepocsort_iou_threshold, 0.2)
        self.assertIs(cfg.per_class, True)
        self.assertEqual(cfg.reid_weights, "weights/osnet_x0_25_msmt17.pt")

    def test_missing_boxmot_blocks_use_defaults(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"algorithm": "ocsort"}}}, CATEGORIES)
        self.assertAlmostEqual(cfg.ocsort_det_thresh, 0.2)
        self.assertEqual(cfg.ocsort_max_age, 30)
        self.assertAlmostEqual(cfg.ocsort_q_xy_scaling, 0.01)
        self.assertAlmostEqual(cfg.ocsort_q_s_scaling, 0.0001)
        self.assertAlmostEqual(cfg.bytetrack_track_thresh, 0.45)

    def test_reid_weights_null_is_none(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"reid_weights": "null"}}}, CATEGORIES)
        self.assertIsNone(cfg.reid_weights)

    def test_unit_float_out_of_range_raises(self):
        with self.assertRaises(ValueError) as ctx:
            vt.parse_tracking_config({"inference": {"tracking": {"bytetrack": {"track_thresh": 1.5}}}}, CATEGORIES)
        self.assertIn("bytetrack.track_thresh", str(ctx.exception))

    def test_positive_int_rejects_zero(self):
        with self.assertRaises(ValueError) as ctx:
            vt.parse_tracking_config({"inference": {"tracking": {"botsort": {"track_buffer": 0}}}}, CATEGORIES)
        self.assertIn("botsort.track_buffer", str(ctx.exception))

    def test_positive_float_rejects_nonpositive(self):
        for bad in (0, -0.5):
            with self.assertRaises(ValueError) as ctx:
                vt.parse_tracking_config(
                    {"inference": {"tracking": {"algorithm": "ocsort", "ocsort": {"Q_xy_scaling": bad}}}},
                    CATEGORIES,
                )
            self.assertIn("ocsort.Q_xy_scaling", str(ctx.exception))

    def test_cmc_method_null_disables(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"cmc_method": "null"}}}, CATEGORIES)
        self.assertIsNone(cfg.cmc_method)

    def test_invalid_cmc_method_raises(self):
        with self.assertRaises(ValueError):
            vt.parse_tracking_config({"inference": {"tracking": {"cmc_method": "magic"}}}, CATEGORIES)

    def test_target_class_override_still_works(self):
        raw = {"inference": {"tracking": {"algorithm": "ocsort", "target_class_names": ["goal"]}}}
        cfg = vt.parse_tracking_config(raw, CATEGORIES)
        self.assertEqual(cfg.target_class_ids, {2})


class TrackingSummaryTest(unittest.TestCase):
    def test_build_tracking_summary_groups_by_track(self):
        rows = [
            {"source": "v.mp4", "frame_index": 0, "track_id": 1, "track_hits": 1, "track_confirmed": True},
            {"source": "v.mp4", "frame_index": 1, "track_id": 1, "track_hits": 2, "track_confirmed": True},
            {"source": "v.mp4", "frame_index": 0, "track_id": 2, "track_hits": 1, "track_confirmed": False},
            {"source": "v.mp4", "frame_index": 2, "track_id": None},
        ]
        summary = vt.build_tracking_summary(rows)
        self.assertEqual(summary["track_count"], 2)
        self.assertEqual(summary["confirmed_count"], 1)
        track1 = next(t for t in summary["tracks"] if t["track_id"] == 1)
        self.assertEqual(track1["first_frame_index"], 0)
        self.assertEqual(track1["last_frame_index"], 1)
        self.assertEqual(track1["num_points"], 2)
        self.assertEqual(track1["lifespan_frames"], 2)
        self.assertTrue(track1["confirmed"])


class TrackVisibilityTest(unittest.TestCase):
    def _track(self, hits=5, last_seen=40, frames=range(41)):
        from collections import deque

        return vt.TrackedBall(
            track_id=1,
            center_x=0.0,
            center_y=0.0,
            base_radius=80.0,
            hits=hits,
            first_frame_index=0,
            last_seen_frame_index=last_seen,
            points=deque([(frame, float(frame), 0.0) for frame in frames]),
        )

    def test_is_track_visible_hides_stale_track(self):
        cfg = make_config(trajectory_max_age_frames=10)
        track = self._track(hits=5, last_seen=40)
        self.assertTrue(vt.is_track_visible(track, 45, cfg))  # 5 frames since last seen -> visible
        self.assertFalse(vt.is_track_visible(track, 55, cfg))  # 15 frames -> stale, hidden
        # unconfirmed (hits < min_hits) is hidden even if recent
        unconfirmed_cfg = make_config(trajectory_max_age_frames=10, min_hits=3)
        self.assertFalse(vt.is_track_visible(self._track(hits=1, last_seen=40), 45, unconfirmed_cfg))
        # null age -> always visible when confirmed
        self.assertTrue(vt.is_track_visible(track, 9999, make_config(trajectory_max_age_frames=None)))

    def test_trail_points_age_filtered(self):
        track = self._track(frames=range(41))
        frames_drawn = [int(x) for (x, _y) in vt.trail_points(track, 40, make_config(trajectory_max_age_frames=10))]
        self.assertEqual(frames_drawn, list(range(30, 41)))  # only points within 10 frames of frame 40
        self.assertEqual(
            len(vt.trail_points(track, 40, make_config(trajectory_max_age_frames=None))), 41
        )  # null -> all

    def test_live_center_extrapolates_during_gap(self):
        track = self._track(hits=5, last_seen=40)
        track.center_x, track.center_y = 100.0, 50.0
        track.velocity_x, track.velocity_y, track.velocity_initialized = 4.0, 0.0, True
        moving = make_config(use_velocity_prediction=True)
        self.assertEqual(vt.live_center(track, 43, moving), (112.0, 50.0))  # gap 3 -> 100 + 4*3
        self.assertEqual(vt.live_center(track, 40, moving), (100.0, 50.0))  # gap 0 -> last center
        self.assertEqual(
            vt.live_center(track, 43, make_config(use_velocity_prediction=False)), (100.0, 50.0)
        )  # velocity off
        self.assertEqual(vt.live_center(track, None, moving), (100.0, 50.0))  # no current frame


# --- O4: synthetic end-to-end video test (no RF-DETR model) -----------------

import inference_rf_detr_model as inference_runner

# frame_index -> list of ball centers detected on that frame.
SCRIPT = {0: [(20, 20)], 1: [(24, 20)], 2: [(28, 20)], 3: [(32, 20)], 4: [(36, 20)], 5: [(40, 20)]}


def _frame_index_from_name(file_name):
    match = re.search(r"_frame_(\d+)", file_name)
    return int(match.group(1)) if match else -1


def _scripted_prediction(image_id, center_x, center_y):
    return {
        "image_id": image_id,
        "category_id": 1,
        "bbox": [center_x - 5.0, center_y - 5.0, 10.0, 10.0],
        "score": 0.9,
        "area": 100.0,
    }


def _fake_predict_image(record, model, prediction_config, output_dir, save_visual=False):
    centers = SCRIPT.get(_frame_index_from_name(record.file_name), [])
    return [_scripted_prediction(record.image_id, cx, cy) for cx, cy in centers], None, None


def _fake_predict_images_rfdetr(records, model, batch_config, output_dir):
    predictions = [
        [
            _scripted_prediction(record.image_id, cx, cy)
            for cx, cy in SCRIPT.get(_frame_index_from_name(record.file_name), [])
        ]
        for record in records
    ]
    return predictions, None, None


def _write_synthetic_video(path, frames=6, size=64):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (size, size))
    try:
        for _ in range(frames):
            writer.write(np.full((size, size, 3), 40, dtype=np.uint8))
    finally:
        writer.release()


def _track_ids_by_frame(rows):
    return {row["frame_index"]: row["track_id"] for row in rows if row.get("category_id") == 1}


class SyntheticVideoTrackingTest(unittest.TestCase):
    def _run(self, tmp, batch_size):
        video_path = Path(tmp) / "clip.mp4"
        _write_synthetic_video(video_path)
        output_dir = Path(tmp) / f"out_{batch_size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        item = inference_runner.SourceItem(source=str(video_path), kind="video", is_url=False, local_path=video_path)
        video_cfg = {
            "batch_size": batch_size,
            "detection_fps": None,
            "start_time": 0,
            "end_time": "all",
            "max_seconds": "all",
            "output_fps": None,
            "render_skipped_frames": True,
        }
        tracking_config = inference_runner.video_tracking.parse_tracking_config(
            {"inference": {"tracking": {"enabled": True, "algorithm": "circle", "radius_pixels": 80}}}, CATEGORIES
        )
        with mock.patch.object(inference_runner.evaluator, "predict_image", _fake_predict_image), mock.patch.object(
            inference_runner.evaluator, "predict_images_rfdetr", _fake_predict_images_rfdetr
        ):
            rows, target, _ = inference_runner.predict_video_file(
                item, 1, None, {}, CATEGORIES, output_dir, [], video_cfg, tracking_config
            )
        return rows, target

    def test_one_pass_and_batched_match_and_track_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            one_pass_rows, one_pass_video = self._run(tmp, batch_size=1)
            batched_rows, batched_video = self._run(tmp, batch_size=4)

            one_pass_ids = _track_ids_by_frame(one_pass_rows)
            batched_ids = _track_ids_by_frame(batched_rows)

            self.assertEqual(set(one_pass_ids), set(SCRIPT))  # every scripted frame produced a tracked ball
            self.assertEqual(set(one_pass_ids.values()), {1})  # one ball -> one stable id
            self.assertEqual(one_pass_ids, batched_ids)  # one-pass and batched agree
            self.assertTrue(one_pass_video.exists() and one_pass_video.stat().st_size > 0)
            self.assertTrue(batched_video.exists() and batched_video.stat().st_size > 0)
            self.assertTrue(all(row.get("track_id") is not None for row in one_pass_rows if row["category_id"] == 1))

    def test_sahi_recheck_filter_runs_before_tracker_and_render(self):
        class RecordingTracker:
            def __init__(self):
                self.seen_scores = []
                self.tracks = []

            def update(self, _frame_index, predictions, frame=None):
                self.seen_scores.append([row["score"] for row in predictions])
                return [dict(row) for row in predictions]

        def predictions_for(record):
            low = _scripted_prediction(record.image_id, 20, 20)
            low["score"] = 0.49
            boundary = _scripted_prediction(record.image_id, 40, 20)
            boundary["score"] = 0.5
            return [low, boundary]

        def fake_predict_image(record, *_args, **_kwargs):
            return predictions_for(record), None, None

        def fake_predict_batch(records, *_args, **_kwargs):
            return [predictions_for(record) for record in records], None, None

        prediction_config = {
            "model": {"confidence_threshold": 0.25},
            "inference": {"mode": "sahi", "batch_size": 4},
            "sahi": {
                "batch_size": 4,
                "recheck": {"enabled": True, "fused_confidence_threshold": 0.5},
            },
        }
        tracking_config = inference_runner.video_tracking.parse_tracking_config(
            {"inference": {"tracking": {"enabled": True, "algorithm": "circle"}}}, CATEGORIES
        )

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            _write_synthetic_video(video_path)
            item = inference_runner.SourceItem(
                source=str(video_path), kind="video", is_url=False, local_path=video_path
            )
            for batch_size in (1, 4):
                with self.subTest(batch_size=batch_size):
                    output_dir = Path(tmp) / f"filtered_{batch_size}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    video_cfg = {
                        "batch_size": batch_size,
                        "detection_fps": None,
                        "start_time": 0,
                        "end_time": "all",
                        "max_seconds": "all",
                        "output_fps": None,
                        "render_skipped_frames": True,
                    }
                    tracker = RecordingTracker()
                    rendered_scores = []

                    def capture_render(image, rows, *_args, **_kwargs):
                        rendered_scores.extend(row["score"] for row in rows)
                        return image.convert("RGB")

                    with mock.patch.object(
                        inference_runner.evaluator, "predict_image", fake_predict_image
                    ), mock.patch.object(
                        inference_runner.evaluator, "predict_images_rfdetr", fake_predict_batch
                    ), mock.patch.object(inference_runner, "create_tracker", return_value=tracker), mock.patch.object(
                        inference_runner, "draw_predictions", side_effect=capture_render
                    ):
                        rows, _, _ = inference_runner.predict_video_file(
                            item,
                            1,
                            None,
                            prediction_config,
                            CATEGORIES,
                            output_dir,
                            [],
                            video_cfg,
                            tracking_config,
                        )

                    self.assertTrue(tracker.seen_scores)
                    self.assertTrue(all(scores == [0.5] for scores in tracker.seen_scores))
                    self.assertTrue(rows)
                    self.assertTrue(all(row["score"] == 0.5 for row in rows))
                    self.assertTrue(rendered_scores)
                    self.assertTrue(all(score == 0.5 for score in rendered_scores))


if __name__ == "__main__":
    unittest.main()
