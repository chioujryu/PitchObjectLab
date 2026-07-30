"""Unit tests for BallFilter — no GPU / SAM3 required."""

import numpy as np
from omegaconf import OmegaConf

from pitch_ball_tracker.detection.ball_filter import BallFilter, _mask_circularity_area
from pitch_ball_tracker.segmentation.sam3_segmentor import SegmentResult


def _make_cfg(**overrides):
    base = {
        "ball_filter": {
            "min_circularity": 0.60,
            "min_area_px": 50,
            "max_area_px": 8000,
            "max_aspect_ratio": 1.8,
        }
    }
    cfg = OmegaConf.create(base)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v)
    return cfg


def _circle_mask(cx, cy, r, H=200, W=200):
    """Create a boolean mask with a filled circle."""
    import cv2

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask.astype(bool)


def _rect_mask(x1, y1, x2, y2, H=200, W=200):
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# ------------------------------------------------------------------


class TestMaskCircularityArea:
    def test_circle_high_circularity(self):
        mask = _circle_mask(100, 100, 30)
        circ, area = _mask_circularity_area(mask)
        assert circ > 0.85, f"Expected high circularity for circle, got {circ:.3f}"
        assert area > 2000

    def test_rect_low_circularity(self):
        mask = _rect_mask(10, 10, 50, 150)  # tall rectangle
        circ, _ = _mask_circularity_area(mask)
        assert circ < 0.65, f"Expected low circularity for rectangle, got {circ:.3f}"

    def test_empty_mask_returns_zeros(self):
        mask = np.zeros((100, 100), dtype=bool)
        circ, area = _mask_circularity_area(mask)
        assert circ == 0.0
        assert area == 0.0


class TestBallFilter:
    def _make_result(self, masks, boxes):
        scores = np.ones(len(boxes), dtype=np.float32) * 0.9
        return SegmentResult(
            masks=np.array(masks, dtype=bool),
            boxes=np.array(boxes, dtype=np.float32),
            scores=scores,
        )

    def test_keeps_circular_ball(self):
        cfg = _make_cfg()
        bf = BallFilter(cfg)
        mask = _circle_mask(100, 100, 25)
        result = self._make_result([mask], [[75, 75, 125, 125]])
        cands = bf.filter(result)
        assert len(cands) == 1
        assert cands[0].circularity > 0.60

    def test_rejects_rectangle(self):
        cfg = _make_cfg()
        bf = BallFilter(cfg)
        mask = _rect_mask(10, 10, 30, 100)  # elongated
        result = self._make_result([mask], [[10, 10, 30, 100]])
        cands = bf.filter(result)
        assert len(cands) == 0

    def test_rejects_too_small(self):
        cfg = _make_cfg()
        bf = BallFilter(cfg)
        mask = _circle_mask(100, 100, 3)  # tiny
        result = self._make_result([mask], [[97, 97, 103, 103]])
        cands = bf.filter(result)
        assert len(cands) == 0

    def test_rejects_too_large(self):
        cfg = _make_cfg()
        bf = BallFilter(cfg)
        mask = _circle_mask(100, 100, 90)  # huge
        result = self._make_result([mask], [[10, 10, 190, 190]])
        cands = bf.filter(result)
        assert len(cands) == 0

    def test_empty_input(self):
        cfg = _make_cfg()
        bf = BallFilter(cfg)
        result = SegmentResult(
            masks=np.empty((0, 200, 200), dtype=bool),
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
        )
        assert bf.filter(result) == []
