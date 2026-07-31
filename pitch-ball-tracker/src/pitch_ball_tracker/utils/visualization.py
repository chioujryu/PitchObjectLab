from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from pitch_ball_tracker.tracking.tracklet import Tracklet


# Deterministic color palette by track ID
def _id_color(track_id: int) -> tuple[int, int, int]:
    np.random.seed(track_id * 137 + 42)
    return tuple(int(x) for x in np.random.randint(80, 255, 3))


class Visualizer:
    """Draws tracking results onto BGR frames."""

    def __init__(
        self,
        draw_masks: bool = True,
        draw_trails: bool = True,
        trail_length: int = 40,
    ) -> None:
        self._draw_masks = draw_masks
        self._draw_trails = draw_trails
        self._trail_length = trail_length
        # track_id → deque of (cx, cy) center points
        self._trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_length))

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Tracklet],
        field_mask: np.ndarray | None = None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        out = frame.copy()

        # Optional: lightly tint field mask
        if field_mask is not None:
            tint = np.zeros_like(out)
            tint[field_mask] = (0, 30, 0)
            out = cv2.addWeighted(out, 1.0, tint, 0.25, 0)

        for t in tracks:
            color = _id_color(t.track_id)
            box = t.get_bbox().astype(int)
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"ID:{t.track_id} {t.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Motion trail
            if self._draw_trails:
                self._trails[t.track_id].append((cx, cy))
                pts = list(self._trails[t.track_id])
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(out, pts[k - 1], pts[k], c, 2, cv2.LINE_AA)

        # Frame counter
        cv2.putText(out, f"Frame {frame_idx}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        return out

    def cleanup_lost_trails(self, active_ids: set[int]) -> None:
        """Remove trail history for tracks that no longer exist."""
        stale = [tid for tid in self._trails if tid not in active_ids]
        for tid in stale:
            del self._trails[tid]
