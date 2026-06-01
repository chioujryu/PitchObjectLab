from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import Optional

import numpy as np

from pitch_ball_tracker.tracking.kalman import KalmanBoxTracker


class TrackState(Enum):
    TENTATIVE = auto()   # not yet confirmed
    CONFIRMED = auto()   # confirmed active track
    LOST = auto()        # no detection for N frames; Kalman still predicts
    DELETED = auto()     # ready to be removed


class Tracklet:
    """
    A single object track combining a Kalman filter with a ReID embedding
    history that follows the PRTReID (tracklet-averaged) approach from
    SoccerMaster: the stored embedding is a window mean over the last
    `avg_window` frame embeddings.
    """

    def __init__(
        self,
        track_id: int,
        bbox_xyxy: np.ndarray,
        score: float,
        avg_window: int = 12,
        ema_alpha: float = 0.1,
    ) -> None:
        self.track_id = track_id
        self.state = TrackState.TENTATIVE

        self.hits: int = 1
        self.age: int = 1          # total frames since creation
        self.frames_since_update: int = 0

        self._kf = KalmanBoxTracker(bbox_xyxy)
        self.bbox: np.ndarray = bbox_xyxy.astype(np.float32)
        self.score: float = score

        # ReID embedding state
        self._avg_window = avg_window
        self._ema_alpha = ema_alpha
        self._embedding_history: deque[np.ndarray] = deque(maxlen=avg_window)
        self._ema_embedding: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Kalman interface
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        pred = self._kf.predict()
        self.age += 1
        self.frames_since_update += 1
        self.bbox = pred
        return pred

    def update(self, bbox_xyxy: np.ndarray, score: float) -> None:
        self._kf.update(bbox_xyxy)
        self.bbox = self._kf.get_state()
        self.score = score
        self.hits += 1
        self.frames_since_update = 0

    def get_bbox(self) -> np.ndarray:
        return self._kf.get_state()

    # ------------------------------------------------------------------
    # ReID embedding management
    # ------------------------------------------------------------------

    def add_embedding(self, embedding: np.ndarray) -> None:
        """
        Store a new per-frame embedding and update both the window mean and
        the exponential moving average.  The window mean is the "PRTReID"
        representative embedding; EMA provides an alternative for lookup.
        """
        self._embedding_history.append(embedding.copy())
        if self._ema_embedding is None:
            self._ema_embedding = embedding.copy()
        else:
            self._ema_embedding = (
                (1.0 - self._ema_alpha) * self._ema_embedding
                + self._ema_alpha * embedding
            )

    @property
    def reid_embedding(self) -> Optional[np.ndarray]:
        """Window-averaged embedding (PRTReID style).  None if not yet set."""
        if not self._embedding_history:
            return self._ema_embedding
        return np.mean(list(self._embedding_history), axis=0)

    def has_embedding(self) -> bool:
        return bool(self._embedding_history) or self._ema_embedding is not None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def mark_confirmed(self) -> None:
        self.state = TrackState.CONFIRMED

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_deleted(self) -> None:
        self.state = TrackState.DELETED

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED
