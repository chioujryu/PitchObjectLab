import unittest

from rf_detr_hybrid_tracker import HybridFootballTracker, HybridTrackingConfig
import rf_detr_video_tracking as video_tracking


def det(cx, cy, score=0.9, *, recheck_pass=True, size=10.0):
    return {
        "bbox": [cx - size / 2.0, cy - size / 2.0, size, size],
        "score": score,
        "category_id": 0,
        "recheck_pass": recheck_pass,
    }


class HybridFootballTrackerTest(unittest.TestCase):
    def make_tracker(self, **overrides):
        values = {
            "lookahead_seconds": 0.10,
            "high_confidence": 0.50,
            "low_confidence": 0.25,
            "new_track_confidence": 0.50,
            "confirmed_hits": 2,
            "confirmation_window": 3,
            "lost_seconds": 1.0,
            "predicted_output_seconds": 0.5,
            "cmc_enabled": False,
            "high_gate_pixels": 40.0,
            "low_gate_pixels": 24.0,
        }
        values.update(overrides)
        return HybridFootballTracker(HybridTrackingConfig(**values), target_class_ids={0})

    def test_step_has_fixed_delay_and_flush_is_deterministic(self):
        tracker = self.make_tracker()

        self.assertEqual(tracker.step(0, 0.00, None, [det(10, 10)]), [])
        self.assertEqual(tracker.step(1, 0.05, None, [det(12, 10)]), [])
        committed = tracker.step(2, 0.10, None, [det(14, 10)])

        self.assertEqual([frame["frame_index"] for frame in committed], [0])
        self.assertEqual([frame["frame_index"] for frame in tracker.flush()], [1, 2])
        self.assertEqual(tracker.flush(), [])

    def test_low_confidence_detection_never_starts_a_track(self):
        tracker = self.make_tracker(lookahead_seconds=0.0)

        frame = tracker.step(0, 0.0, None, [det(20, 20, score=0.30)])[0]

        self.assertIsNone(frame["detections"][0]["track_id"])
        self.assertEqual(frame["track_states"], [])

    def test_rechecked_low_confidence_detection_can_continue_confirmed_track(self):
        tracker = self.make_tracker(lookahead_seconds=0.0)
        first = tracker.step(0, 0.0, None, [det(20, 20)])[0]
        track_id = first["detections"][0]["track_id"]
        tracker.step(1, 1 / 30, None, [det(22, 20)])

        continued = tracker.step(2, 2 / 30, None, [det(24, 20, score=0.30, recheck_pass=True)])[0]
        rejected = tracker.step(3, 3 / 30, None, [det(26, 20, score=0.30, recheck_pass=False)])[0]

        self.assertEqual(continued["detections"][0]["track_id"], track_id)
        self.assertIsNone(rejected["detections"][0]["track_id"])

    def test_crossing_tracks_keep_motion_consistent_ids(self):
        tracker = self.make_tracker(lookahead_seconds=0.0, ambiguity_margin=0.02)
        frames = [
            [det(10, 20), det(50, 20)],
            [det(18, 20), det(42, 20)],
            [det(26, 20), det(34, 20)],
            [det(34, 20), det(26, 20)],
            [det(42, 20), det(18, 20)],
        ]
        observed = []
        for index, detections in enumerate(frames):
            observed.append(tracker.step(index, index / 10, None, detections)[0]["detections"])

        right_mover = observed[0][0]["track_id"]
        left_mover = observed[0][1]["track_id"]
        self.assertEqual(observed[-1][0]["track_id"], right_mover)
        self.assertEqual(observed[-1][1]["track_id"], left_mover)
        self.assertNotEqual(right_mover, left_mover)

    def test_observation_union_gate_keeps_id_when_fast_ball_stops(self):
        tracker = self.make_tracker(lookahead_seconds=0.0, high_gate_pixels=96.0)
        first = tracker.step(0, 0.0, None, [det(100, 100)])[0]
        tracker.step(1, 1 / 30, None, [det(72, 108)])
        stopped = tracker.step(2, 2 / 30, None, [det(72, 108)])[0]

        self.assertEqual(stopped["detections"][0]["track_id"], first["detections"][0]["track_id"])
        self.assertEqual(stopped["detections"][0]["association_stage"], "high")

    def test_low_recheck_accepts_pipeline_recheck_passed_field(self):
        tracker = self.make_tracker(lookahead_seconds=0.0)
        first = tracker.step(0, 0.0, None, [det(20, 20)])[0]
        tracker.step(1, 1 / 30, None, [det(22, 20)])
        low = det(24, 20, score=0.30)
        low.pop("recheck_pass")
        low["recheck_passed"] = True
        continued = tracker.step(2, 2 / 30, None, [low])[0]

        self.assertEqual(continued["detections"][0]["track_id"], first["detections"][0]["track_id"])

    def test_every_high_confidence_ball_can_start_a_track_without_field_filter(self):
        tracker = self.make_tracker(lookahead_seconds=0.0)

        frame = tracker.step(0, 0.0, None, [det(-5, 10), det(1000, 700), det(100, 100)])[0]

        self.assertEqual(len({row["track_id"] for row in frame["detections"]}), 3)

    def test_state_output_exposes_motion_covariance_association_and_cmc(self):
        tracker = self.make_tracker(lookahead_seconds=0.0)

        frame = tracker.step(0, 0.0, None, [det(10, 10)])[0]
        state = frame["track_states"][0]

        self.assertEqual(state["observation"], "observed")
        self.assertEqual(len(state["covariance_diagonal"]), 8)
        self.assertEqual(set(state["motion_model_probabilities"]), {"constant_velocity", "constant_acceleration"})
        self.assertIn("association", state)
        self.assertEqual(frame["cmc"]["method"], "identity")

    def test_nearby_stationary_balls_do_not_churn_ids_under_default_ambiguity_policy(self):
        tracker = self.make_tracker(lookahead_seconds=0.0)
        first = tracker.step(0, 0.0, None, [det(100, 100), det(106, 100)])[0]
        expected = [row["track_id"] for row in first["detections"]]

        second = tracker.step(1, 1 / 30, None, [det(100, 100), det(106, 100)])[0]
        third = tracker.step(2, 2 / 30, None, [det(100, 100), det(106, 100)])[0]

        self.assertEqual([row["track_id"] for row in third["detections"]], expected)
        self.assertEqual([row["track_id"] for row in second["detections"]], expected)

    def test_tracks_expose_the_legacy_overlay_contract(self):
        tracker = self.make_tracker(lookahead_seconds=0.0)
        tracker.step(0, 0.0, None, [det(10, 10)])
        tracker.step(1, 1 / 30, None, [det(12, 10)])
        track = tracker.tracks[0]

        self.assertEqual((track.center_x, track.center_y), (12.0, 10.0))
        self.assertGreater(track.base_radius, 0.0)
        self.assertEqual(track.missing_frames, 0)
        self.assertEqual(video_tracking.trail_points(track, 1, video_tracking.TrackingConfig()), [(10.0, 10.0), (12.0, 10.0)])


if __name__ == "__main__":
    unittest.main()
