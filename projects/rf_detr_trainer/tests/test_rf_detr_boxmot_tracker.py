"""Offline unit tests for the boxmot tracking adapter (rf_detr_boxmot_tracker).

These tests never import boxmot or load a model: ``_build_boxmot_tracker`` is monkeypatched
with a FakeBoxmotTracker that returns a controlled (M,8) [x1,y1,x2,y2,id,conf,cls,det_ind]
array, so the adapter's dets-conversion, det_ind mapback, TrackedBall maintenance, and
TRACK_FIELDS emission are verified without the dependency. The one real-boxmot smoke test is
gated with skipUnless.
"""

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rf_detr_boxmot_tracker as bt  # noqa: E402
import rf_detr_video_tracking as vt  # noqa: E402

CATEGORIES = [
    {"id": 0, "name": "standing_player"},
    {"id": 1, "name": "football"},
    {"id": 2, "name": "goal"},
]
FRAME = np.zeros((16, 16, 3), dtype=np.uint8)


def ball(center_x, center_y, size=10.0, score=0.9, category_id=1):
    """Build a COCO-style prediction whose bbox is centered at (center_x, center_y)."""
    return {
        "category_id": category_id,
        "bbox": [center_x - size / 2.0, center_y - size / 2.0, size, size],
        "score": score,
        "area": size * size,
    }


def make_config(**overrides):
    cfg = vt.TrackingConfig(enabled=True, target_class_ids={1}, algorithm="ocsort", min_hits=1)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class FakeBoxmotTracker:
    """Stand-in for a boxmot tracker; records inputs and returns a scripted (M,8) array."""

    def __init__(self, responder=None):
        self.calls = []
        self._responder = responder

    def update(self, dets, img):
        dets = np.asarray(dets, dtype=float)
        self.calls.append((dets.copy(), img))
        if self._responder is not None:
            out = self._responder(dets, img, len(self.calls) - 1)
            return np.asarray(out, dtype=float) if len(out) else np.empty((0, 8), dtype=float)
        # Default: echo each det as its own track, track_id = det_ind + 1.
        rows = []
        for index, det in enumerate(dets):
            x1, y1, x2, y2, conf, cls = det[:6]
            rows.append([x1, y1, x2, y2, index + 1, conf, cls, index])
        return np.asarray(rows, dtype=float) if rows else np.empty((0, 8), dtype=float)


def build_tracker(fake, frame_size=(64, 64), **cfg_overrides):
    cfg = make_config(**cfg_overrides)
    with mock.patch.object(bt, "_build_boxmot_tracker", lambda config, device: fake):
        return bt.BoxmotTracker(cfg, device="cpu", frame_size=frame_size)


class BuildDetsTest(unittest.TestCase):
    def test_filters_targets_and_converts_xywh_to_xyxy(self):
        tracker = build_tracker(FakeBoxmotTracker())
        preds = [ball(100, 100, size=10, category_id=1), ball(50, 50, size=20, category_id=0)]
        dets, mapping = tracker.build_dets(preds)
        self.assertEqual(dets.shape, (1, 6))  # only the football survives the target filter
        self.assertEqual(list(dets[0]), [95.0, 95.0, 105.0, 105.0, 0.9, 1.0])
        self.assertEqual(mapping, [0])

    def test_empty_targets_yield_empty_dets(self):
        tracker = build_tracker(FakeBoxmotTracker())
        dets, mapping = tracker.build_dets([ball(10, 10, category_id=0)])
        self.assertEqual(dets.shape, (0, 6))
        self.assertEqual(mapping, [])


class DetIndMapbackTest(unittest.TestCase):
    def test_det_ind_maps_to_original_rows_shuffled_and_partial(self):
        # orig indices: 0=ball, 1=player(non-target), 2=ball, 3=ball -> targets map to dets [0,2,3].
        preds = [ball(10, 10), ball(0, 0, category_id=0), ball(20, 20), ball(30, 30)]

        def responder(dets, img, call):
            # Return two tracks, shuffled, M < K, referencing det_ind 2 and 0.
            return [
                [0, 0, 10, 10, 7, 0.9, 1, 2],  # det_ind 2 -> orig index 3
                [0, 0, 10, 10, 5, 0.9, 1, 0],  # det_ind 0 -> orig index 0
            ]

        tracker = build_tracker(FakeBoxmotTracker(responder))
        rows = tracker.update(0, preds, frame=FRAME)
        self.assertEqual(len(rows), 4)  # input order preserved
        self.assertEqual(rows[0]["track_id"], 5)
        self.assertEqual(rows[3]["track_id"], 7)
        self.assertIsNone(rows[2]["track_id"])  # det_ind 1 not emitted
        self.assertIsNone(rows[1]["track_id"])  # non-target player

    def test_out_of_range_det_ind_is_ignored(self):
        def responder(dets, img, call):
            return [[0, 0, 10, 10, 3, 0.9, 1, -1]]  # det_ind -1 is unmappable

        tracker = build_tracker(FakeBoxmotTracker(responder))
        rows = tracker.update(0, [ball(10, 10)], frame=FRAME)
        self.assertIsNone(rows[0]["track_id"])
        self.assertEqual(tracker.tracks, [])

    def test_seven_column_output_is_skipped(self):
        def responder(dets, img, call):
            return [[0, 0, 10, 10, 3, 0.9, 1]]  # no det_ind column

        tracker = build_tracker(FakeBoxmotTracker(responder))
        rows = tracker.update(0, [ball(10, 10)], frame=FRAME)
        self.assertIsNone(rows[0]["track_id"])


class TrackedBallMaintenanceTest(unittest.TestCase):
    def test_stable_id_accumulates_hits_and_points(self):
        tracker = build_tracker(FakeBoxmotTracker())  # echoes det 0 as track id 1
        tracker.update(0, [ball(100, 100)], frame=FRAME)
        tracker.update(1, [ball(110, 100)], frame=FRAME)
        self.assertEqual(len(tracker.tracks), 1)
        track = tracker.tracks[0]
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.hits, 2)
        self.assertEqual(len(track.points), 2)
        self.assertEqual(track.first_frame_index, 0)
        self.assertEqual(track.last_seen_frame_index, 1)
        self.assertAlmostEqual(track.center_x, 110.0)  # re-centered on the returned bbox

    def test_track_fields_populated_and_non_target_none(self):
        tracker = build_tracker(FakeBoxmotTracker())
        rows = tracker.update(3, [ball(100, 100), ball(0, 0, category_id=0)], frame=FRAME)
        for key in vt.TRACK_FIELDS:
            self.assertIn(key, rows[0])
            self.assertIn(key, rows[1])
        self.assertEqual(rows[0]["track_id"], 1)
        self.assertEqual(rows[0]["track_first_frame_index"], 3)
        self.assertEqual(rows[0]["track_age_frames"], 1)
        self.assertTrue(rows[0]["track_confirmed"])  # hits(1) >= min_hits(1)
        self.assertTrue(all(rows[1][key] is None for key in vt.TRACK_FIELDS))

    def test_confirmed_tracks_and_render_helper_compat(self):
        tracker = build_tracker(FakeBoxmotTracker())
        tracker.update(0, [ball(100, 100)], frame=FRAME)
        self.assertIsInstance(tracker.tracks[0], vt.TrackedBall)
        self.assertEqual(len(tracker.confirmed_tracks()), 1)
        # The existing render helpers must consume adapter tracks unchanged.
        track = tracker.tracks[0]
        self.assertTrue(vt.is_track_visible(track, 0, tracker.cfg))
        self.assertIsInstance(vt.trail_points(track, 0, tracker.cfg), list)
        self.assertEqual(vt.live_center(track, 0, tracker.cfg), (100.0, 100.0))
        self.assertGreater(vt.effective_radius(track, tracker.cfg), 0.0)

    def test_stale_track_pruned_after_max_missing(self):
        tracker = build_tracker(FakeBoxmotTracker(), max_missing_frames=1)
        tracker.update(0, [ball(100, 100)], frame=FRAME)  # creates track id 1
        tracker.update(1, [], frame=FRAME)  # miss -> missing_frames 1 (kept)
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update(2, [], frame=FRAME)  # miss -> missing_frames 2 (> 1, pruned)
        self.assertEqual(tracker.tracks, [])


class FrameHandlingTest(unittest.TestCase):
    def test_frame_none_synthesizes_from_frame_size(self):
        fake = FakeBoxmotTracker()
        tracker = build_tracker(fake, frame_size=(32, 24))
        tracker.update(0, [ball(10, 10)], frame=None)
        _dets, img = fake.calls[0]
        self.assertEqual(img.shape, (24, 32, 3))  # (height, width, 3)

    def test_frame_none_without_frame_size_raises(self):
        tracker = build_tracker(FakeBoxmotTracker(), frame_size=None)
        with self.assertRaises(ValueError):
            tracker.update(0, [ball(10, 10)], frame=None)


class ReidHalfTest(unittest.TestCase):
    def test_half_clamped_off_on_cpu_and_mps(self):
        cfg = make_config(reid_half=True)
        self.assertFalse(bt.effective_reid_half(cfg, "cpu"))
        self.assertFalse(bt.effective_reid_half(cfg, "mps"))
        self.assertTrue(bt.effective_reid_half(cfg, "cuda:0"))

    def test_half_false_stays_false(self):
        cfg = make_config(reid_half=False)
        self.assertFalse(bt.effective_reid_half(cfg, "cuda:0"))

    def test_resolve_reid_weights_prefers_local_path(self):
        cfg = make_config(reid_weights="weights/osnet_x0_25_msmt17.pt")
        self.assertEqual(bt.resolve_reid_weights(cfg).name, "osnet_x0_25_msmt17.pt")
        self.assertEqual(bt.resolve_reid_weights(make_config()).name, bt.DEFAULT_REID_WEIGHTS)


class ImportGuardTest(unittest.TestCase):
    @unittest.skipIf(bt.boxmot_importable(), "boxmot is installed")
    def test_missing_boxmot_raises_helpful_error(self):
        with self.assertRaises(ImportError) as ctx:
            bt._build_boxmot_tracker(make_config(algorithm="ocsort"), "cpu")
        self.assertIn("boxmot", str(ctx.exception))


class RealBoxmotSmokeTest(unittest.TestCase):
    @unittest.skipUnless(bt.boxmot_importable(), "boxmot not installed")
    def test_real_ocsort_tracks_a_moving_ball(self):
        cfg = make_config(
            algorithm="ocsort",
            ocsort_min_hits=1,
            ocsort_det_thresh=0.1,
            ocsort_q_xy_scaling=1.0,
            ocsort_q_s_scaling=0.5,
        )
        tracker = bt.BoxmotTracker(cfg, device="cpu", frame_size=(128, 128))
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        rows = []
        for index in range(4):
            rows = tracker.update(index, [ball(40 + index * 4, 60, size=14, category_id=1)], frame=frame)
        self.assertTrue(any(row.get("track_id") is not None for row in rows))


if __name__ == "__main__":
    unittest.main()
