"""Unit tests for KalmanBoxTracker — pure numpy, no GPU required."""
import numpy as np
import pytest

from pitch_ball_tracker.tracking.kalman import KalmanBoxTracker


def _bbox(cx, cy, w, h):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class TestKalmanBoxTracker:
    def test_predict_returns_array(self):
        kf = KalmanBoxTracker(_bbox(100, 100, 30, 30))
        pred = kf.predict()
        assert pred.shape == (4,)

    def test_update_reduces_uncertainty(self):
        kf = KalmanBoxTracker(_bbox(100, 100, 30, 30))
        p_before = np.trace(kf.P)
        kf.update(_bbox(100, 100, 30, 30))
        p_after = np.trace(kf.P)
        assert p_after < p_before

    def test_tracking_stationary_object(self):
        """Kalman should converge close to true position for a stationary ball."""
        kf = KalmanBoxTracker(_bbox(100, 100, 30, 30))
        for _ in range(20):
            kf.predict()
            kf.update(_bbox(100, 100, 30, 30))
        state = kf.get_state()
        cx_est = (state[0] + state[2]) / 2
        cy_est = (state[1] + state[3]) / 2
        assert abs(cx_est - 100) < 2.0
        assert abs(cy_est - 100) < 2.0

    def test_tracking_moving_object(self):
        """Kalman should follow a linearly-moving ball reasonably well."""
        kf = KalmanBoxTracker(_bbox(0, 100, 20, 20))
        for i in range(1, 30):
            kf.predict()
            kf.update(_bbox(i * 5, 100, 20, 20))
        state = kf.get_state()
        cx_est = (state[0] + state[2]) / 2
        assert abs(cx_est - 145) < 10.0

    def test_unique_ids(self):
        before = KalmanBoxTracker._next_id
        kf1 = KalmanBoxTracker(_bbox(0, 0, 10, 10))
        kf2 = KalmanBoxTracker(_bbox(0, 0, 10, 10))
        assert kf2.id == kf1.id + 1

    def test_time_since_update_increments_on_predict(self):
        kf = KalmanBoxTracker(_bbox(50, 50, 20, 20))
        assert kf.time_since_update == 0
        kf.predict()
        assert kf.time_since_update == 1
        kf.predict()
        assert kf.time_since_update == 2
        kf.update(_bbox(50, 50, 20, 20))
        assert kf.time_since_update == 0
