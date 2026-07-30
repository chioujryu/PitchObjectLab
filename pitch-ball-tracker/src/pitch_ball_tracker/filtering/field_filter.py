from __future__ import annotations

import cv2
import numpy as np
from loguru import logger
from omegaconf import DictConfig


class FieldFilter:
    """Estimates the playing-field boundary and returns a binary mask (True = inside field) used by the rest of the
    pipeline.

    Two modes: auto – detect field by HSV green-colour segmentation; find the
                largest contiguous green region; build a convex hull mask.
    manual – use a polygon provided in the config (list of [x, y] points).

    The mask is cached and refreshed every `update_interval` frames to avoid recomputing on every frame (field
    boundaries move slowly).
    """

    _UPDATE_INTERVAL = 30  # re-detect every N frames

    def __init__(self, cfg: DictConfig) -> None:
        fc = cfg.field
        self._enabled: bool = fc.boundary_filter
        self._auto: bool = fc.auto_detect
        self._manual_polygon: list | None = list(fc.manual_polygon) if fc.manual_polygon is not None else None
        self._hsv_range: list[int] = list(fc.green_hsv_range)
        self._min_ratio: float = fc.min_field_area_ratio

        self._mask: np.ndarray | None = None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Update (or reuse cached) field mask for `frame_bgr`. Returns a boolean HxW mask (True = inside field). Always
        returns a valid mask even if detection fails (all-True fallback).
        """
        H, W = frame_bgr.shape[:2]
        if not self._enabled:
            return np.ones((H, W), dtype=bool)

        # Manual polygon overrides auto detection
        if self._manual_polygon is not None:
            if self._mask is None or self._mask.shape != (H, W):
                self._mask = self._poly_to_mask(self._manual_polygon, H, W)
            return self._mask

        # Auto: refresh every N frames
        if self._frame_count % self._UPDATE_INTERVAL == 0 or self._mask is None:
            detected = self._detect_field(frame_bgr)
            self._mask = detected if detected is not None else np.ones((H, W), dtype=bool)
        self._frame_count += 1
        return self._mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_field(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        H, W = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lo = np.array(self._hsv_range[0::2], dtype=np.uint8)  # H_lo, S_lo, V_lo
        hi = np.array(self._hsv_range[1::2], dtype=np.uint8)  # H_hi, S_hi, V_hi
        green_mask = cv2.inRange(hsv, lo, hi)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # Find largest contour
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("FieldFilter: no green region found; using full frame.")
            return None

        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / (H * W)
        if area_ratio < self._min_ratio:
            logger.warning(f"FieldFilter: largest green region too small ({area_ratio:.2%}); using full frame.")
            return None

        # Convex hull for a cleaner field boundary
        hull = cv2.convexHull(largest)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        logger.debug(f"FieldFilter: field detected ({area_ratio:.1%} of frame).")
        return mask.astype(bool)

    @staticmethod
    def _poly_to_mask(polygon: list, H: int, W: int) -> np.ndarray:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask.astype(bool)
