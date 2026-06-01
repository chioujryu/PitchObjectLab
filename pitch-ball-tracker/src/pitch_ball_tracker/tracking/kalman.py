from __future__ import annotations

import numpy as np


class KalmanBoxTracker:
    """
    Kalman filter for a single bounding box.

    State vector  x = [cx, cy, a, h, vx, vy, va, vh]
      cx, cy : centre coordinates
      a      : aspect ratio (w / h)
      h      : height
      vx…vh  : corresponding velocities

    Observation z = [cx, cy, a, h]

    Follows the SORT / BoT-SORT convention.
    """

    _next_id: int = 0

    def __init__(self, bbox_xyxy: np.ndarray) -> None:
        # --- constant-velocity motion model ---
        # F: state transition (8×8)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0

        # H: observation matrix (4×8)
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Q: process noise
        self.Q = np.eye(8, dtype=np.float64) * 1e-2
        self.Q[4:, 4:] *= 10.0

        # R: measurement noise
        self.R = np.eye(4, dtype=np.float64) * 1e-1
        self.R[2:, 2:] *= 10.0

        # P: initial state covariance (large uncertainty in velocity)
        self.P = np.eye(8, dtype=np.float64)
        self.P[0, 0] = self.P[1, 1] = 10.0
        self.P[2, 2] = self.P[3, 3] = 10.0
        self.P[4:, 4:] = 1e4

        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[:4] = self._bbox_to_obs(bbox_xyxy)

        self.time_since_update: int = 0
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Advance state one step; return predicted bbox as xyxy."""
        # prevent negative aspect ratio
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._obs_to_bbox(self.x[:4])

    def update(self, bbox_xyxy: np.ndarray) -> None:
        """Correct the filter with a new observation."""
        z = self._bbox_to_obs(bbox_xyxy)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.time_since_update = 0

    def get_state(self) -> np.ndarray:
        """Return current bbox estimate as xyxy."""
        return self._obs_to_bbox(self.x[:4])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_to_obs(bbox: np.ndarray) -> np.ndarray:
        """Convert xyxy bbox to (cx, cy, a, h) column vector."""
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1e-3)
        h = max(y2 - y1, 1e-3)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        a = w / h
        return np.array([[cx], [cy], [a], [h]], dtype=np.float64)

    @staticmethod
    def _obs_to_bbox(obs: np.ndarray) -> np.ndarray:
        """Convert (cx, cy, a, h) to xyxy bbox."""
        cx, cy, a, h = obs[0, 0], obs[1, 0], obs[2, 0], obs[3, 0]
        h = max(h, 1e-3)
        w = max(a * h, 1e-3)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
