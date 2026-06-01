from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from omegaconf import DictConfig

from pitch_ball_tracker.detection.ball_filter import BallCandidate


class MotionFilter:
    """
    Filters ball candidates by motion magnitude computed from optical flow.

    Strategy:
      - Maintain a ring buffer of recent grayscale frames.
      - Compute Farneback dense optical flow between the oldest and newest
        buffered frame, giving a per-pixel (u, v) motion field.
      - For each candidate, average |flow| over the bounding-box region.
      - Candidates below `min_motion_px` are classified as stationary.
      - Stationary candidates outside the field are removed outright.
        Stationary candidates inside the field have their score penalised
        (configurable) so the tracker still sees them but with low confidence.

    Camera shake:
      - The global mean flow is subtracted from each candidate's flow before
        thresholding, partially compensating for panning/shaking camera.
    """

    def __init__(self, cfg: DictConfig) -> None:
        mc = cfg.motion
        self._enabled: bool = mc.enabled
        self._method: str = mc.method
        self._min_motion: float = mc.min_motion_px
        self._history: int = mc.history_frames
        self._penalise: bool = mc.penalize_stationary_inside
        self._penalty: float = mc.stationary_score_penalty

        self._gray_buffer: deque[np.ndarray] = deque(maxlen=max(self._history, 2))
        self._flow: np.ndarray | None = None   # cached flow from last compute

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Push a new frame into the buffer and recompute optical flow."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._gray_buffer.append(gray)
        if len(self._gray_buffer) >= 2:
            self._flow = self._compute_flow()

    def filter(
        self,
        candidates: list[BallCandidate],
        field_mask: np.ndarray | None,
    ) -> list[BallCandidate]:
        """
        Apply motion filtering to `candidates`.

        Args:
            candidates : output from BallFilter.
            field_mask : boolean HxW mask; True = inside field.  If None,
                         stationary candidates are never penalised (treated as
                         outside-field rule is skipped).
        Returns:
            Filtered + possibly score-penalised list of BallCandidate.
        """
        if not self._enabled or self._flow is None:
            return candidates

        flow_mag = np.sqrt(
            self._flow[..., 0] ** 2 + self._flow[..., 1] ** 2
        ).astype(np.float32)

        # Camera-shake compensation: subtract global median motion magnitude
        global_median = float(np.median(flow_mag))

        kept: list[BallCandidate] = []
        for c in candidates:
            region_mag = self._region_mean(flow_mag, c.box)
            effective_mag = max(region_mag - global_median, 0.0)

            if effective_mag >= self._min_motion:
                kept.append(c)
                continue

            # Stationary candidate
            in_field = self._is_in_field(c.box, field_mask)
            if not in_field:
                # Outside field and stationary → discard
                continue

            # Inside field and stationary → penalise score
            if self._penalise:
                penalised = BallCandidate(
                    box=c.box,
                    mask=c.mask,
                    score=c.score * self._penalty,
                    circularity=c.circularity,
                    area=c.area,
                )
                kept.append(penalised)
            # If penalise disabled, drop stationary-inside-field too

        return kept

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_flow(self) -> np.ndarray:
        """Compute optical flow between the first and last buffered frames."""
        prev = self._gray_buffer[0]
        curr = self._gray_buffer[-1]
        if self._method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )
        else:
            # Lucas-Kanade sparse → dense approximation via Farneback fallback
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
        return flow  # (H, W, 2)

    @staticmethod
    def _region_mean(mag: np.ndarray, box: np.ndarray) -> float:
        """Average flow magnitude inside a bounding box."""
        H, W = mag.shape
        x1 = int(max(box[0], 0))
        y1 = int(max(box[1], 0))
        x2 = int(min(box[2], W))
        y2 = int(min(box[3], H))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(mag[y1:y2, x1:x2].mean())

    @staticmethod
    def _is_in_field(box: np.ndarray, field_mask: np.ndarray | None) -> bool:
        """Check whether the centre of the box lies inside the field mask."""
        if field_mask is None:
            return True   # no mask → treat as inside (conservative)
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        H, W = field_mask.shape
        cx = np.clip(cx, 0, W - 1)
        cy = np.clip(cy, 0, H - 1)
        return bool(field_mask[cy, cx])
