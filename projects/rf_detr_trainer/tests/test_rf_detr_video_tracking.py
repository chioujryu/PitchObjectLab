"""Unit tests for circle-based football tracking and its video-pipeline integration.

These tests use scripted detections and a tiny synthetic video, so no RF-DETR
model is loaded.
"""

from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rf_detr_video_tracking as vt  # noqa: E402

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

    def test_new_track_is_not_aged_on_its_creation_frame(self):
        tracker = vt.FootballTracker(make_config())

        tracker.update(0, [ball(100, 100)])

        self.assertEqual(tracker.tracks[0].missing_frames, 0)

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

    def test_predicted_trajectory_defaults_off_and_can_be_enabled(self):
        default_cfg = vt.parse_tracking_config({"inference": {"tracking": {"enabled": True}}}, CATEGORIES)
        enabled_cfg = vt.parse_tracking_config(
            {"inference": {"tracking": {"draw_predicted_trajectory": True}}}, CATEGORIES
        )
        self.assertFalse(default_cfg.draw_predicted_trajectory)
        self.assertTrue(enabled_cfg.draw_predicted_trajectory)

    def test_center_and_confirmation_policy_defaults_are_backward_compatible(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"enabled": True}}}, CATEGORIES)

        self.assertFalse(cfg.draw_predicted_center)
        self.assertFalse(cfg.render_confirmed_only)
        self.assertFalse(cfg.export_confirmed_only)

    def test_center_and_hybrid_confirmation_policies_are_parsed(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {
            "enabled": True,
            "algorithm": "hybrid",
            "draw_predicted_center": True,
            "render_confirmed_only": True,
            "export_confirmed_only": True,
        }}}, CATEGORIES)

        self.assertTrue(cfg.draw_predicted_center)
        self.assertTrue(cfg.render_confirmed_only)
        self.assertTrue(cfg.export_confirmed_only)

    def test_confirmation_policies_reject_non_hybrid_algorithms(self):
        for key in ("render_confirmed_only", "export_confirmed_only"):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError,
                rf"inference\.tracking\.{key}.*only supported.*hybrid",
            ):
                vt.parse_tracking_config({"inference": {"tracking": {
                    "algorithm": "circle",
                    key: True,
                }}}, CATEGORIES)

    def test_invalid_radius_raises(self):
        with self.assertRaises(ValueError):
            vt.parse_tracking_config({"inference": {"tracking": {"radius_pixels": 0}}}, CATEGORIES)

    def test_optional_radius_and_trajectory_limits_must_be_positive(self):
        for key in ("radius_scale", "max_radius_pixels", "trajectory_max_points"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                vt.parse_tracking_config({"inference": {"tracking": {key: 0}}}, CATEGORIES)

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

    def test_hybrid_algorithm_allowed(self):
        cfg = vt.parse_tracking_config({"inference": {"tracking": {"algorithm": "hybrid"}}}, CATEGORIES)
        self.assertEqual(cfg.algorithm, "hybrid")

    def test_hybrid_nested_config_is_validated_and_preserved(self):
        raw = {"inference": {"tracking": {"algorithm": "hybrid", "hybrid": {
            "candidate": {"low_confidence": 0.2, "high_confidence": 0.6},
            "hypothesis": {"lookahead_seconds": 0.25, "ambiguity_margin": 0.04},
        }}}}

        cfg = vt.parse_tracking_config(raw, CATEGORIES)

        self.assertEqual(cfg.hybrid_options["candidate"]["low_confidence"], 0.2)
        self.assertEqual(cfg.hybrid_options["hypothesis"]["lookahead_seconds"], 0.25)
        self.assertEqual(cfg.hybrid_options["hypothesis"]["ambiguity_margin"], 0.04)

    def test_hybrid_legacy_beam_width_is_explicitly_rejected(self):
        for hybrid in ({"hypothesis": {"beam_width": 8}}, {"beam_width": 8}):
            with self.subTest(hybrid=hybrid), self.assertRaisesRegex(ValueError, r"beam_width is unsupported"):
                vt.parse_tracking_config({"inference": {"tracking": {
                    "algorithm": "hybrid",
                    "hybrid": hybrid,
                }}}, CATEGORIES)

    def test_hybrid_unknown_config_key_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown.*hybrid"):
            vt.parse_tracking_config({"inference": {"tracking": {"algorithm": "hybrid", "hybrid": {"typo": 1}}}}, CATEGORIES)

    def test_nested_boxmot_blocks_map_to_flat_fields(self):
        raw = {
            "inference": {
                "tracking": {
                    "algorithm": "ocsort",
                    "per_class": True,
                    "reid_weights": "weights/osnet_x0_25_msmt17.pt",
                    "ocsort": {"max_age": 50, "asso_threshold": 0.4, "use_byte": True, "Q_xy_scaling": 1.0, "Q_s_scaling": 0.5},
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

    def test_hybrid_summary_includes_state_and_cmc_diagnostics(self):
        predictions = [{"source": "v.mp4", "frame_index": 0, "track_id": 1, "track_hits": 1}]
        states = [
            {
                "source": "v.mp4",
                "frame_index": 0,
                "track_id": 1,
                "status": "confirmed",
                "observation": "observed",
                "cmc": {"method": "sparse_optical_flow", "success": True, "reason": None},
            },
            {
                "source": "v.mp4",
                "frame_index": 1,
                "track_id": 1,
                "status": "lost",
                "observation": "predicted",
                "cmc": {"method": "identity", "success": False, "reason": "estimation_failed"},
            },
        ]

        summary = vt.build_hybrid_tracking_summary(predictions, states)

        self.assertEqual(summary["state_rows"], 2)
        self.assertEqual(summary["observation_counts"], {"observed": 1, "predicted": 1})
        self.assertEqual(summary["cmc"]["fallback_reasons"], {"estimation_failed": 1})


class TrackVisibilityTest(unittest.TestCase):
    def _track(self, hits=5, last_seen=40, frames=range(0, 41)):
        from collections import deque

        return vt.TrackedBall(
            track_id=1, center_x=0.0, center_y=0.0, base_radius=80.0,
            hits=hits, first_frame_index=0, last_seen_frame_index=last_seen,
            points=deque([(frame, float(frame), 0.0) for frame in frames]),
        )

    def test_is_track_visible_hides_stale_track(self):
        cfg = make_config(trajectory_max_age_frames=10)
        track = self._track(hits=5, last_seen=40)
        self.assertTrue(vt.is_track_visible(track, 45, cfg))   # 5 frames since last seen -> visible
        self.assertFalse(vt.is_track_visible(track, 55, cfg))  # 15 frames -> stale, hidden
        # unconfirmed (hits < min_hits) is hidden even if recent
        unconfirmed_cfg = make_config(trajectory_max_age_frames=10, min_hits=3)
        self.assertFalse(vt.is_track_visible(self._track(hits=1, last_seen=40), 45, unconfirmed_cfg))
        # null age -> always visible when confirmed
        self.assertTrue(vt.is_track_visible(track, 9999, make_config(trajectory_max_age_frames=None)))

    def test_final_hybrid_confirmation_backfills_overlay_before_top_level_min_hits(self):
        track = self._track(hits=1, last_seen=0, frames=[0])
        cfg = make_config(algorithm="hybrid", min_hits=3, render_confirmed_only=True)

        self.assertFalse(vt.is_track_visible(track, 0, cfg))
        self.assertTrue(vt.is_track_visible(track, 0, cfg, final_confirmed=True))

    def test_trail_points_age_filtered(self):
        track = self._track(frames=range(0, 41))
        frames_drawn = [int(x) for (x, _y) in vt.trail_points(track, 40, make_config(trajectory_max_age_frames=10))]
        self.assertEqual(frames_drawn, list(range(30, 41)))  # only points within 10 frames of frame 40
        self.assertEqual(len(vt.trail_points(track, 40, make_config(trajectory_max_age_frames=None))), 41)  # null -> all

    def test_live_center_extrapolates_during_gap(self):
        track = self._track(hits=5, last_seen=40)
        track.center_x, track.center_y = 100.0, 50.0
        track.velocity_x, track.velocity_y, track.velocity_initialized = 4.0, 0.0, True
        moving = make_config(use_velocity_prediction=True)
        self.assertEqual(vt.live_center(track, 43, moving), (112.0, 50.0))  # gap 3 -> 100 + 4*3
        self.assertEqual(vt.live_center(track, 40, moving), (100.0, 50.0))  # gap 0 -> last center
        self.assertEqual(vt.live_center(track, 43, make_config(use_velocity_prediction=False)), (100.0, 50.0))  # velocity off
        self.assertEqual(vt.live_center(track, None, moving), (100.0, 50.0))  # no current frame

    def test_hybrid_snapshot_center_is_not_extrapolated_twice(self):
        track = self._track(hits=5, last_seen=40)
        track.center_x, track.center_y = 112.0, 50.0
        track.velocity_x, track.velocity_y = 4.0, 0.0
        cfg = make_config(algorithm="hybrid", use_velocity_prediction=True)

        self.assertEqual(vt.live_center(track, 43, cfg), (112.0, 50.0))

    def test_observed_center_is_drawn_only_on_an_observation_frame(self):
        track = self._track(last_seen=40)
        cfg = make_config(draw_current_center=True, draw_predicted_center=False)

        self.assertTrue(vt.is_observed_on_frame(track, 40))
        self.assertTrue(vt.should_draw_observed_center(track, 40, cfg))
        self.assertFalse(vt.is_observed_on_frame(track, 41))
        self.assertFalse(vt.should_draw_observed_center(track, 41, cfg))

    def test_predicted_center_requires_independent_opt_in(self):
        track = self._track(last_seen=40)

        self.assertFalse(vt.should_draw_predicted_center(
            track, 41, make_config(draw_current_center=True, draw_predicted_center=False)
        ))
        self.assertTrue(vt.should_draw_predicted_center(
            track, 41, make_config(draw_current_center=False, draw_predicted_center=True)
        ))
        self.assertFalse(vt.should_draw_predicted_center(
            track, 40, make_config(draw_current_center=False, draw_predicted_center=True)
        ))

    def test_hybrid_predicted_center_respects_output_horizon(self):
        cfg = make_config(
            algorithm="hybrid",
            draw_predicted_center=True,
            hybrid_options={"lifecycle": {"predicted_output_seconds": 0.5}},
        )
        within = SimpleNamespace(
            last_seen_frame_index=40,
            last_timestamp=10.5,
            last_observed_timestamp=10.0,
        )
        beyond = SimpleNamespace(
            last_seen_frame_index=40,
            last_timestamp=10.5001,
            last_observed_timestamp=10.0,
        )

        self.assertTrue(vt.should_draw_predicted_center(within, 41, cfg))
        self.assertFalse(vt.should_draw_predicted_center(beyond, 41, cfg))

    def test_hybrid_predicted_center_without_timing_is_suppressed(self):
        cfg = make_config(algorithm="hybrid", draw_predicted_center=True)

        self.assertFalse(vt.should_draw_predicted_center(self._track(last_seen=40), 41, cfg))


# --- O4: synthetic end-to-end video test (no RF-DETR model) -----------------

import inference_rf_detr_model as inference_runner  # noqa: E402


class TrackOverlayRenderingTest(unittest.TestCase):
    def _predicted_track(self):
        from collections import deque

        return vt.TrackedBall(
            track_id=1,
            center_x=20.0,
            center_y=10.0,
            base_radius=50.0,
            velocity_x=5.0,
            velocity_y=0.0,
            velocity_initialized=True,
            hits=2,
            first_frame_index=0,
            last_seen_frame_index=2,
            points=deque([(0, 10.0, 10.0), (2, 20.0, 10.0)]),
        )

    def test_predicted_trajectory_off_keeps_observed_bridge_without_predicted_dot(self):
        draw = mock.Mock()
        cfg = make_config(
            draw_predicted_trajectory=False,
            trajectory_taper=False,
            draw_current_center=True,
            draw_search_circle=True,
            use_velocity_prediction=True,
        )

        inference_runner.draw_track_overlays(draw, [self._predicted_track()], cfg, [1], current_frame_index=4)

        self.assertEqual(draw.line.call_args.args[0], [(10.0, 10.0), (20.0, 10.0)])
        color = inference_runner.track_color(1)
        self.assertEqual(
            draw.ellipse.call_args_list,
            [
                mock.call([-20.0, -40.0, 80.0, 60.0], outline=color, width=1),
            ],
        )

    def test_predicted_center_opt_in_uses_fixed_three_pixel_radius(self):
        draw = mock.Mock()
        cfg = make_config(
            draw_trajectory=False,
            draw_current_center=False,
            draw_predicted_center=True,
            draw_search_circle=False,
            trajectory_width=9,
            use_velocity_prediction=True,
        )

        inference_runner.draw_track_overlays(draw, [self._predicted_track()], cfg, [1], current_frame_index=4)

        color = inference_runner.track_color(1)
        draw.ellipse.assert_called_once_with([27.0, 7.0, 33.0, 13.0], fill=color)

    def test_taper_reaches_configured_width_without_changing_center_radius(self):
        draw = mock.Mock()
        track = self._predicted_track()
        track.last_seen_frame_index = 2
        cfg = make_config(
            draw_predicted_trajectory=False,
            trajectory_taper=True,
            trajectory_width=4,
            draw_current_center=True,
            draw_predicted_center=False,
            draw_search_circle=False,
        )

        inference_runner.draw_track_overlays(draw, [track], cfg, [1], current_frame_index=2)

        self.assertEqual([call.kwargs["width"] for call in draw.line.call_args_list], [4])
        color = inference_runner.track_color(1)
        draw.ellipse.assert_called_once_with([17.0, 7.0, 23.0, 13.0], fill=color)

    def test_predicted_trajectory_on_restores_live_head(self):
        draw = mock.Mock()
        cfg = make_config(
            draw_predicted_trajectory=True,
            trajectory_taper=False,
            draw_current_center=False,
            use_velocity_prediction=True,
        )

        inference_runner.draw_track_overlays(draw, [self._predicted_track()], cfg, [1], current_frame_index=4)

        self.assertEqual(draw.line.call_args.args[0], [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0)])


class TrackerFactoryTest(unittest.TestCase):
    def test_hybrid_factory_is_lazy_and_uses_nested_options(self):
        from rf_detr_hybrid_tracker import HybridFootballTracker

        cfg = vt.parse_tracking_config({"inference": {"tracking": {
            "enabled": True,
            "algorithm": "hybrid",
            "hybrid": {"hypothesis": {"lookahead_seconds": 0.25}},
        }}}, CATEGORIES)

        tracker = inference_runner.create_tracker(cfg)
        self.assertIsInstance(tracker, HybridFootballTracker)
        self.assertAlmostEqual(tracker.cfg.lookahead_seconds, 0.25)

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
        [_scripted_prediction(record.image_id, cx, cy) for cx, cy in SCRIPT.get(_frame_index_from_name(record.file_name), [])]
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


def _video_frame_count(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()


class _RecordingCanonicalVideo:
    def __init__(self, start_kwargs):
        self.start_kwargs = dict(start_kwargs)
        self.frames = []
        self.finalize_kwargs = None

    def write_frame(self, **kwargs):
        self.frames.append(dict(kwargs))

    def finalize(self, **kwargs):
        self.finalize_kwargs = dict(kwargs)
        return {"frame_count": len(self.frames)}


class _RecordingCanonicalRun:
    def __init__(self):
        self.videos = []
        self.abort_count = 0

    def start_video(self, **kwargs):
        video = _RecordingCanonicalVideo(kwargs)
        self.videos.append(video)
        return video

    def abort_active(self):
        self.abort_count += 1


class SyntheticVideoTrackingTest(unittest.TestCase):
    @staticmethod
    def _pipeline_options(pipeline):
        if pipeline == "streaming":
            return {"streaming": True, "batch_size": 4}
        if pipeline == "one_pass":
            return {"streaming": False, "batch_size": 1}
        if pipeline == "batched":
            return {"streaming": False, "batch_size": 4}
        raise AssertionError(f"unknown test pipeline {pipeline}")

    def _run(self, tmp, pipeline):
        video_path = Path(tmp) / "clip.mp4"
        _write_synthetic_video(video_path)
        output_dir = Path(tmp) / f"out_{pipeline}"
        output_dir.mkdir(parents=True, exist_ok=True)
        item = inference_runner.SourceItem(source=str(video_path), kind="video", is_url=False, local_path=video_path)
        video_cfg = {
            "detection_fps": None,
            "start_time": 0,
            "end_time": "all",
            "max_seconds": "all",
            "output_fps": None,
            "render_skipped_frames": True,
            **self._pipeline_options(pipeline),
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

    def _run_hybrid(self, tmp, pipeline, canonical_run=None, detection_fps=None):
        video_path = Path(tmp) / "clip.mp4"
        _write_synthetic_video(video_path)
        output_dir = Path(tmp) / f"hybrid_{pipeline}"
        output_dir.mkdir(parents=True, exist_ok=True)
        item = inference_runner.SourceItem(source=str(video_path), kind="video", is_url=False, local_path=video_path)
        video_cfg = {
            "detection_fps": detection_fps,
            "start_time": 0,
            "end_time": "all",
            "max_seconds": "all",
            "output_fps": None,
            "render_skipped_frames": True,
            **self._pipeline_options(pipeline),
        }
        tracking_config = inference_runner.video_tracking.parse_tracking_config(
            {"inference": {"tracking": {
                "enabled": True,
                "algorithm": "hybrid",
                "hybrid": {
                    "hypothesis": {"lookahead_seconds": 0.1},
                    "cmc": {"cmc_enabled": False},
                    "association": {"high_gate_pixels": 80.0},
                },
            }}},
            CATEGORIES,
        )
        states = []
        with mock.patch.object(inference_runner.evaluator, "predict_image", _fake_predict_image), mock.patch.object(
            inference_runner.evaluator, "predict_images_rfdetr", _fake_predict_images_rfdetr
        ):
            rows, target, _ = inference_runner.predict_video_file(
                item,
                1,
                None,
                {},
                CATEGORIES,
                output_dir,
                [],
                video_cfg,
                tracking_config,
                "cpu",
                states,
                canonical_run=canonical_run,
            )
        return rows, states, target

    def test_hybrid_streaming_one_pass_and_batched_have_identical_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = {
                pipeline: self._run_hybrid(tmp, pipeline)
                for pipeline in ("streaming", "one_pass", "batched")
            }
            self.assertTrue(all(video.exists() for _rows, _states, video in results.values()))
            self.assertTrue(all(_video_frame_count(video) == len(SCRIPT) for _rows, _states, video in results.values()))

        ids_by_pipeline = {
            pipeline: _track_ids_by_frame(rows)
            for pipeline, (rows, _states, _video) in results.items()
        }
        self.assertEqual(set(ids_by_pipeline["streaming"]), set(SCRIPT))
        self.assertEqual(ids_by_pipeline["streaming"], ids_by_pipeline["one_pass"])
        self.assertEqual(ids_by_pipeline["streaming"], ids_by_pipeline["batched"])
        self.assertEqual(set(ids_by_pipeline["streaming"].values()), {1})
        for rows, states, _video in results.values():
            self.assertTrue(rows and states)
            self.assertEqual({row["frame_index"] for row in states}, set(SCRIPT))

    def test_hybrid_canonical_commits_keep_order_predictions_and_cmc_affine(self):
        results = {}
        with tempfile.TemporaryDirectory() as tmp:
            for pipeline in ("streaming", "one_pass", "batched"):
                canonical_run = _RecordingCanonicalRun()
                self._run_hybrid(
                    tmp,
                    pipeline,
                    canonical_run=canonical_run,
                    detection_fps=15,
                )
                self.assertEqual(canonical_run.abort_count, 0)
                self.assertEqual(len(canonical_run.videos), 1)
                canonical_video = canonical_run.videos[0]
                frames = canonical_video.frames
                self.assertEqual(
                    [row["segment_frame_index"] for row in frames], list(range(len(SCRIPT)))
                )
                self.assertEqual(
                    [row["detection_ran"] for row in frames],
                    [True, False, True, False, True, False],
                )
                self.assertEqual(
                    [len(row["detections"]) for row in frames], [1, 0, 1, 0, 1, 0]
                )
                self.assertEqual(
                    [[state["observation"] for state in row["track_states"]] for row in frames],
                    [["observed"], [], ["observed"], ["predicted"], ["observed"], ["predicted"]],
                )
                for row in frames:
                    motion = row["camera_motion"]
                    self.assertIsNotNone(motion)
                    affine = motion["affine_previous_to_current"]
                    self.assertEqual(len(affine), 2)
                    self.assertTrue(all(len(values) == 3 for values in affine))
                results[pipeline] = [
                    (
                        row["segment_frame_index"],
                        row["source_frame_index"],
                        round(float(row["source_timestamp_seconds"]), 6),
                        row["timestamp_source"],
                        row["detection_ran"],
                        len(row["detections"]),
                        [state["observation"] for state in row["track_states"]],
                    )
                    for row in frames
                ]

        self.assertEqual(results["streaming"], results["one_pass"])
        self.assertEqual(results["streaming"], results["batched"])

    def test_streaming_one_pass_and_batched_match_and_track_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = {
                pipeline: self._run(tmp, pipeline)
                for pipeline in ("streaming", "one_pass", "batched")
            }

            ids_by_pipeline = {
                pipeline: _track_ids_by_frame(rows)
                for pipeline, (rows, _video) in results.items()
            }
            self.assertEqual(set(ids_by_pipeline["streaming"]), set(SCRIPT))
            self.assertEqual(set(ids_by_pipeline["streaming"].values()), {1})
            self.assertEqual(ids_by_pipeline["streaming"], ids_by_pipeline["one_pass"])
            self.assertEqual(ids_by_pipeline["streaming"], ids_by_pipeline["batched"])
            for rows, video in results.values():
                self.assertTrue(video.exists() and video.stat().st_size > 0)
                self.assertEqual(_video_frame_count(video), len(SCRIPT))
                self.assertTrue(all(row.get("track_id") is not None for row in rows if row["category_id"] == 1))

    def test_canonical_packets_cover_every_frame_and_match_all_video_pipelines(self):
        results = {}
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "canonical_clip.mp4"
            _write_synthetic_video(video_path)
            item = inference_runner.SourceItem(
                source=str(video_path), kind="video", is_url=False, local_path=video_path
            )
            tracking_config = inference_runner.video_tracking.parse_tracking_config(
                {"inference": {"tracking": {
                    "enabled": True,
                    "algorithm": "circle",
                    "radius_pixels": 80,
                    "use_velocity_prediction": True,
                }}},
                CATEGORIES,
            )
            for pipeline in ("streaming", "one_pass", "batched"):
                output_dir = Path(tmp) / f"canonical_{pipeline}"
                output_dir.mkdir(parents=True, exist_ok=True)
                canonical_run = _RecordingCanonicalRun()
                video_cfg = {
                    **self._pipeline_options(pipeline),
                    "save_video": False,
                    "detection_fps": 15,
                    "start_time": 0,
                    "end_time": "all",
                    "max_seconds": "all",
                }
                with mock.patch.object(
                    inference_runner.evaluator, "predict_image", _fake_predict_image
                ), mock.patch.object(
                    inference_runner.evaluator,
                    "predict_images_rfdetr",
                    _fake_predict_images_rfdetr,
                ):
                    _rows, target, _next_id = inference_runner.predict_video_file(
                        item,
                        1,
                        None,
                        {},
                        CATEGORIES,
                        output_dir,
                        [],
                        video_cfg,
                        tracking_config,
                        canonical_run=canonical_run,
                    )

                self.assertIsNone(target)
                self.assertEqual(canonical_run.abort_count, 0)
                self.assertEqual(len(canonical_run.videos), 1)
                canonical_video = canonical_run.videos[0]
                self.assertIsNotNone(canonical_video.finalize_kwargs)
                self.assertIsNone(canonical_video.finalize_kwargs["annotated_output"])
                self.assertEqual(canonical_video.start_kwargs["width"], 64)
                self.assertEqual(canonical_video.start_kwargs["height"], 64)
                frames = canonical_video.frames
                self.assertEqual(
                    [row["segment_frame_index"] for row in frames], list(range(len(SCRIPT)))
                )
                self.assertEqual(
                    [row["source_frame_index"] for row in frames], list(range(len(SCRIPT)))
                )
                self.assertEqual(
                    [row["detection_ran"] for row in frames],
                    [True, False, True, False, True, False],
                )
                self.assertEqual(
                    [len(row["detections"]) for row in frames], [1, 0, 1, 0, 1, 0]
                )
                self.assertEqual(
                    [state["observation"] for row in frames for state in row["track_states"]],
                    ["observed", "predicted", "observed", "predicted", "observed", "predicted"],
                )
                self.assertTrue(
                    all(
                        "score" not in state
                        for row in frames
                        for state in row["track_states"]
                        if state["observation"] == "predicted"
                    )
                )
                timestamps = [row["source_timestamp_seconds"] for row in frames]
                self.assertTrue(all(right > left for left, right in zip(timestamps, timestamps[1:])))
                results[pipeline] = [
                    (
                        row["segment_frame_index"],
                        row["source_frame_index"],
                        round(float(row["source_timestamp_seconds"]), 6),
                        row["timestamp_source"],
                        row["detection_ran"],
                        len(row["detections"]),
                        [state["observation"] for state in row["track_states"]],
                    )
                    for row in frames
                ]

        self.assertEqual(results["streaming"], results["one_pass"])
        self.assertEqual(results["streaming"], results["batched"])

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
                    ), mock.patch.object(
                        inference_runner, "create_tracker", return_value=tracker
                    ), mock.patch.object(
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

    def test_streaming_uses_memory_sources_and_can_skip_video_output(self):
        captured = {}

        def memory_predict(records, _model, _config, _output_dir, *, sources=None):
            captured["paths"] = [record.path for record in records]
            captured["sources"] = list(sources or [])
            return ([[] for _ in records], [], [])

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            _write_synthetic_video(video_path, frames=3)
            output_dir = Path(tmp) / "json_only"
            output_dir.mkdir(parents=True, exist_ok=True)
            item = inference_runner.SourceItem(
                source=str(video_path), kind="video", is_url=False, local_path=video_path
            )
            with mock.patch.object(
                inference_runner.evaluator,
                "predict_images_rfdetr",
                side_effect=memory_predict,
            ):
                rows, target, next_image_id = inference_runner.predict_video_file(
                    item,
                    1,
                    None,
                    {},
                    CATEGORIES,
                    output_dir,
                    [],
                    {
                        "batch_size": 3,
                        "streaming": True,
                        "save_video": False,
                        "start_time": 0,
                        "end_time": "all",
                        "max_seconds": "all",
                    },
                )

        self.assertEqual(rows, [])
        self.assertIsNone(target)
        self.assertEqual(next_image_id, 4)
        self.assertTrue(all(path.startswith("memory://") for path in captured["paths"]))
        self.assertEqual(len(captured["sources"]), 3)

    def test_legacy_video_pipelines_honor_save_video_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            _write_synthetic_video(video_path, frames=3)
            item = inference_runner.SourceItem(
                source=str(video_path), kind="video", is_url=False, local_path=video_path
            )
            for pipeline in ("one_pass", "batched"):
                with self.subTest(pipeline=pipeline):
                    output_dir = Path(tmp) / f"json_only_{pipeline}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with mock.patch.object(
                        inference_runner.evaluator, "predict_image", _fake_predict_image
                    ), mock.patch.object(
                        inference_runner.evaluator,
                        "predict_images_rfdetr",
                        _fake_predict_images_rfdetr,
                    ):
                        _rows, target, _next_id = inference_runner.predict_video_file(
                            item,
                            1,
                            None,
                            {},
                            CATEGORIES,
                            output_dir,
                            [],
                            {
                                **self._pipeline_options(pipeline),
                                "save_video": False,
                                "start_time": 0,
                                "end_time": "all",
                                "max_seconds": "all",
                            },
                        )

                    self.assertIsNone(target)
                    self.assertFalse((output_dir / "videos").exists())

    def test_all_video_pipelines_advance_tracker_on_detection_gaps(self):
        class RecordingTracker:
            def __init__(self):
                self.calls = []
                self.tracks = []

            def update(self, frame_index, predictions, frame=None):
                self.calls.append((frame_index, len(predictions), frame is not None))
                return [dict(row) for row in predictions]

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            _write_synthetic_video(video_path, frames=6)
            item = inference_runner.SourceItem(
                source=str(video_path), kind="video", is_url=False, local_path=video_path
            )
            for pipeline in ("streaming", "one_pass", "batched"):
                with self.subTest(pipeline=pipeline):
                    output_dir = Path(tmp) / f"gaps_{pipeline}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    tracker = RecordingTracker()
                    video_cfg = {
                        **self._pipeline_options(pipeline),
                        "save_video": pipeline != "streaming",
                        "detection_fps": 15,
                        "start_time": 0,
                        "end_time": "all",
                        "max_seconds": "all",
                    }
                    with mock.patch.object(
                        inference_runner.evaluator,
                        "predict_image",
                        _fake_predict_image,
                    ), mock.patch.object(
                        inference_runner.evaluator,
                        "predict_images_rfdetr",
                        _fake_predict_images_rfdetr,
                    ), mock.patch.object(
                        inference_runner,
                        "create_tracker",
                        return_value=tracker,
                    ):
                        inference_runner.predict_video_file(
                            item,
                            1,
                            None,
                            {},
                            CATEGORIES,
                            output_dir,
                            [],
                            video_cfg,
                            inference_runner.video_tracking.TrackingConfig(enabled=True),
                        )

                    self.assertEqual([frame for frame, _, _ in tracker.calls], list(range(6)))
                    self.assertEqual([count for _, count, _ in tracker.calls], [1, 0, 1, 0, 1, 0])
                    self.assertTrue(all(has_frame for _, _, has_frame in tracker.calls))

    def test_streaming_queue_backpressure_flushes_before_detection_batch_is_full(self):
        observed_batch_lengths = []

        def empty_predict(records, _model, _config, _output_dir, *, sources=None):
            self.assertEqual(len(records), len(sources or []))
            observed_batch_lengths.append(len(records))
            return [list() for _ in records], [], []

        model = SimpleNamespace()
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            _write_synthetic_video(video_path, frames=8)
            output_dir = Path(tmp) / "backpressure"
            output_dir.mkdir(parents=True, exist_ok=True)
            item = inference_runner.SourceItem(
                source=str(video_path), kind="video", is_url=False, local_path=video_path
            )
            with mock.patch.object(
                inference_runner.evaluator,
                "predict_images_rfdetr",
                side_effect=empty_predict,
            ):
                inference_runner.predict_video_file(
                    item,
                    1,
                    model,
                    {},
                    CATEGORIES,
                    output_dir,
                    [],
                    {
                        "batch_size": 4,
                        "queue_size": 4,
                        "streaming": True,
                        "save_video": False,
                        "detection_fps": 15,
                        "start_time": 0,
                        "end_time": "all",
                        "max_seconds": "all",
                    },
                )

        self.assertEqual(observed_batch_lengths, [2, 2])
        timing = model._rf_detr_video_pipeline_timing
        self.assertEqual(timing["frame_queue_peak"], 4)
        self.assertEqual(timing["outer_batch_size"], 4)
        self.assertEqual(timing["source_frames"], 8)
        self.assertEqual(timing["detection_frames"], 4)
        self.assertIn("frame_conversion_seconds", timing)

    def test_streaming_fails_fast_when_video_encoder_cannot_open(self):
        import cv2

        class ClosedWriter:
            released = False

            def isOpened(self):
                return False

            def release(self):
                self.released = True

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            _write_synthetic_video(video_path, frames=2)
            output_dir = Path(tmp) / "encoder_failure"
            output_dir.mkdir(parents=True, exist_ok=True)
            item = inference_runner.SourceItem(
                source=str(video_path), kind="video", is_url=False, local_path=video_path
            )
            closed_writer = ClosedWriter()
            with mock.patch.object(cv2, "VideoWriter", return_value=closed_writer):
                with self.assertRaisesRegex(RuntimeError, "Could not create video writer"):
                    inference_runner.predict_video_file(
                        item,
                        1,
                        SimpleNamespace(),
                        {},
                        CATEGORIES,
                        output_dir,
                        [],
                        {
                            "batch_size": 1,
                            "streaming": True,
                            "save_video": True,
                            "start_time": 0,
                            "end_time": "all",
                            "max_seconds": "all",
                        },
                    )
            self.assertTrue(closed_writer.released)


if __name__ == "__main__":
    unittest.main()
