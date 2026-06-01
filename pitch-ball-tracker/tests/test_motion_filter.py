"""Unit tests for MotionFilter — requires only opencv and numpy."""
import numpy as np
import pytest
from omegaconf import OmegaConf

from pitch_ball_tracker.detection.ball_filter import BallCandidate
from pitch_ball_tracker.filtering.motion_filter import MotionFilter


def _make_cfg(**overrides):
    base = {
        "motion": {
            "enabled": True,
            "method": "farneback",
            "min_motion_px": 2.0,
            "history_frames": 3,
            "penalize_stationary_inside": True,
            "stationary_score_penalty": 0.4,
        }
    }
    cfg = OmegaConf.create(base)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v)
    return cfg


def _candidate(box, score=0.9):
    H, W = 200, 200
    import cv2
    mask = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in box]
    mask[y1:y2, x1:x2] = True
    return BallCandidate(
        box=np.array(box, dtype=np.float32),
        mask=mask,
        score=score,
        circularity=0.9,
        area=float((x2 - x1) * (y2 - y1)),
    )


def _static_frame(H=200, W=200):
    return np.zeros((H, W, 3), dtype=np.uint8)


def _moving_frame(shift_x=10, H=200, W=200):
    """Frame with a white rectangle shifted right — simulates motion."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[80:120, 80 + shift_x : 120 + shift_x] = 255
    return frame


class TestMotionFilter:
    def test_disabled_returns_all(self):
        cfg = _make_cfg(**{"motion.enabled": False})
        mf = MotionFilter(cfg)
        cands = [_candidate([70, 70, 130, 130])]
        mf.update_frame(_static_frame())
        mf.update_frame(_static_frame())
        result = mf.filter(cands, field_mask=None)
        assert len(result) == len(cands)

    def test_no_flow_yet_returns_all(self):
        """Before two frames are pushed, flow is None → no filtering."""
        cfg = _make_cfg()
        mf = MotionFilter(cfg)
        cands = [_candidate([70, 70, 130, 130])]
        mf.update_frame(_static_frame())   # only one frame pushed
        result = mf.filter(cands, field_mask=None)
        assert len(result) == len(cands)

    def test_stationary_outside_field_dropped(self):
        cfg = _make_cfg()
        mf = MotionFilter(cfg)
        # Push two identical frames → near-zero flow
        mf.update_frame(_static_frame())
        mf.update_frame(_static_frame())
        cands = [_candidate([70, 70, 130, 130])]
        # field_mask = all False → everything is "outside"
        field_mask = np.zeros((200, 200), dtype=bool)
        result = mf.filter(cands, field_mask=field_mask)
        assert len(result) == 0

    def test_stationary_inside_field_penalised(self):
        cfg = _make_cfg()
        mf = MotionFilter(cfg)
        mf.update_frame(_static_frame())
        mf.update_frame(_static_frame())
        cands = [_candidate([70, 70, 130, 130], score=1.0)]
        field_mask = np.ones((200, 200), dtype=bool)   # all inside
        result = mf.filter(cands, field_mask=field_mask)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.4, abs=0.01)

    def test_field_mask_none_keeps_stationary(self):
        """field_mask=None → stationary candidates are treated as inside field."""
        cfg = _make_cfg()
        mf = MotionFilter(cfg)
        mf.update_frame(_static_frame())
        mf.update_frame(_static_frame())
        cands = [_candidate([70, 70, 130, 130], score=1.0)]
        result = mf.filter(cands, field_mask=None)
        assert len(result) == 1   # kept but penalised
        assert result[0].score < 1.0
