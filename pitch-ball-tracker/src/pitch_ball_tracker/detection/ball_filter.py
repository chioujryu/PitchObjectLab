from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from pitch_ball_tracker.segmentation.sam3_segmentor import SegmentResult

_DEBUG_FRAME = -1  # set to frame index to dump all detections for that frame


@dataclass
class BallCandidate:
    """A single detection that has passed ball-shape filtering."""
    box: np.ndarray       # (4,) xyxy float32
    mask: np.ndarray      # (H, W) bool
    score: float
    circularity: float
    area: float


class BallFilter:
    """
    Geometric filter that accepts SAM3 segments that look like a ball:
      - Circularity close to 1 (round shape)
      - Area within configured bounds
      - Aspect ratio not too elongated
    """

    def __init__(self, cfg: DictConfig) -> None:
        fc = cfg.ball_filter
        self._min_circ = fc.min_circularity
        self._min_area = fc.min_area_px
        self._max_area = fc.max_area_px
        self._max_ar = fc.max_aspect_ratio

    def filter(self, result: SegmentResult, frame_idx: int = -1) -> list[BallCandidate]:
        candidates: list[BallCandidate] = []
        for mask, box, score in zip(result.masks, result.boxes, result.scores):
            circ, area = _mask_circularity_area(mask)
            x1, y1, x2, y2 = box
            w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
            ar = max(w / h, h / w)
            accepted = self._accept(box, circ, area)
            if frame_idx < 10:
                reason = (
                    "OK" if accepted else
                    f"circ={circ:.2f}<{self._min_circ}" if circ < self._min_circ else
                    f"area={area:.0f} out of [{self._min_area},{self._max_area}]" if not (self._min_area <= area <= self._max_area) else
                    f"ar={ar:.2f}>{self._max_ar}"
                )
                logger.debug(
                    f"  [frame {frame_idx}] score={score:.3f} circ={circ:.2f} "
                    f"area={area:.0f} ar={ar:.2f} → {reason}"
                )
            if accepted:
                candidates.append(BallCandidate(
                    box=box.astype(np.float32),
                    mask=mask,
                    score=float(score),
                    circularity=circ,
                    area=area,
                ))
        return candidates

    def _accept(self, box: np.ndarray, circ: float, area: float) -> bool:
        if circ < self._min_circ:
            return False
        if not (self._min_area <= area <= self._max_area):
            return False
        x1, y1, x2, y2 = box
        w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
        ar = max(w / h, h / w)
        if ar > self._max_ar:
            return False
        return True


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mask_circularity_area(mask: np.ndarray) -> tuple[float, float]:
    """Return (circularity, area_px) for a boolean mask."""
    uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, closed=True)
    if perimeter < 1e-3:
        return 0.0, float(area)
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return float(np.clip(circularity, 0.0, 1.0)), float(area)
