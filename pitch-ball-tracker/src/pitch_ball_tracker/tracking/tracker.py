from __future__ import annotations

import numpy as np
from loguru import logger
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment

from pitch_ball_tracker.detection.ball_filter import BallCandidate
from pitch_ball_tracker.tracking.reid import BallReIDExtractor
from pitch_ball_tracker.tracking.tracklet import Tracklet, TrackState


class BotSortTracker:
    """BoT-SORT–style multi-object tracker adapted for ball tracking.

    Three-stage matching per frame: Stage 1 – High-confidence detections ↔ CONFIRMED+LOST tracks (IoU) Stage 2 –
    Low-confidence detections ↔ remaining CONFIRMED (IoU) Stage 3 – Unmatched LOST tracks ↔ unmatched detections (ReID)

    Tracks are confirmed after `min_hits` consecutive detections. LOST tracks are deleted after `max_age +
    lost_track_buffer` frames.
    """

    _HIGH_CONF = 0.5  # split between high / low confidence detections

    def __init__(self, cfg: DictConfig, reid: BallReIDExtractor | None) -> None:
        tc = cfg.tracking
        rc = cfg.reid
        self._max_age: int = tc.max_age
        self._min_hits: int = tc.min_hits
        self._iou_thresh: float = tc.iou_threshold
        self._use_reid: bool = tc.use_reid and reid is not None
        self._reid_thresh: float = rc.similarity_threshold
        self._lost_buffer: int = rc.lost_track_buffer
        self._avg_window: int = rc.avg_window
        self._ema_alpha: float = rc.ema_alpha

        self._reid = reid
        self._tracks: list[Tracklet] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        candidates: list[BallCandidate],
        frame_bgr: np.ndarray,
    ) -> list[Tracklet]:
        """Advance the tracker by one frame.

        Args:
            candidates: ball candidates from BallFilter (possibly empty).
            frame_bgr: current BGR frame (needed by ReID extractor).

        Returns:
            List of currently CONFIRMED (and predicted) tracks.
        """
        # --- predict all tracks one step forward ---
        for t in self._tracks:
            t.predict()

        # --- extract ReID embeddings for candidates ---
        boxes = np.array([c.box for c in candidates], dtype=np.float32) if candidates else np.empty((0, 4))
        scores = np.array([c.score for c in candidates], dtype=np.float32) if candidates else np.empty((0,))
        embeddings = self._reid.extract(frame_bgr, boxes) if self._use_reid and len(boxes) > 0 else None

        # --- split candidates into high / low confidence ---
        if len(candidates) > 0:
            high_mask = scores >= self._HIGH_CONF
            high_idx = np.where(high_mask)[0].tolist()
            low_idx = np.where(~high_mask)[0].tolist()
        else:
            high_idx, low_idx = [], []

        confirmed = [t for t in self._tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self._tracks if t.state == TrackState.TENTATIVE]

        # --- Stage 1: high-conf dets ↔ confirmed+lost tracks (IoU) ---
        matched1, unmatched_trk1, unmatched_det_high = self._match_iou(
            confirmed, candidates, high_idx, frame_bgr, embeddings
        )
        for ti, di in matched1:
            t = confirmed[ti]
            c = candidates[di]
            t.update(c.box, c.score)
            if embeddings is not None:
                t.add_embedding(embeddings[di])
            if t.state == TrackState.LOST:
                t.state = TrackState.CONFIRMED

        # --- Stage 2: low-conf dets ↔ still-unmatched CONFIRMED tracks (IoU) ---
        remaining_conf = [confirmed[i] for i in unmatched_trk1 if confirmed[i].is_confirmed]
        [i for i in unmatched_trk1 if confirmed[i].is_confirmed]
        lost_idx = [i for i in unmatched_trk1 if confirmed[i].is_lost]

        matched2, _unmatched_trk2, unmatched_det_low = self._match_iou(
            remaining_conf, candidates, low_idx, frame_bgr, embeddings
        )
        for ti, di in matched2:
            t = remaining_conf[ti]
            c = candidates[di]
            t.update(c.box, c.score)
            if embeddings is not None:
                t.add_embedding(embeddings[di])

        # --- Stage 3: ReID matching of LOST tracks ↔ leftover high-conf dets ---
        all_unmatched_det = unmatched_det_high + unmatched_det_low
        lost_tracks = [confirmed[i] for i in lost_idx]
        if self._use_reid and lost_tracks and all_unmatched_det and embeddings is not None:
            matched3, _, unmatched_det_final = self._match_reid(lost_tracks, candidates, all_unmatched_det, embeddings)
            for ti, di in matched3:
                t = lost_tracks[ti]
                c = candidates[di]
                t.update(c.box, c.score)
                t.add_embedding(embeddings[di])
                t.state = TrackState.CONFIRMED
        else:
            unmatched_det_final = all_unmatched_det

        # --- Stage 1+2: tentative tracks vs remaining dets (IoU only) ---
        matched_tent, _, unmatched_det_new = self._match_iou(
            tentative, candidates, unmatched_det_final, frame_bgr, embeddings
        )
        for ti, di in matched_tent:
            t = tentative[ti]
            c = candidates[di]
            t.update(c.box, c.score)
            if embeddings is not None:
                t.add_embedding(embeddings[di])

        # --- create new tentative tracks for truly unmatched detections ---
        for di in unmatched_det_new:
            c = candidates[di]
            t = Tracklet(
                track_id=self._next_id,
                bbox_xyxy=c.box,
                score=c.score,
                avg_window=self._avg_window,
                ema_alpha=self._ema_alpha,
            )
            self._next_id += 1
            if embeddings is not None:
                t.add_embedding(embeddings[di])
            self._tracks.append(t)

        # --- promote/demote track states ---
        for t in self._tracks:
            if t.state == TrackState.TENTATIVE:
                if t.hits >= self._min_hits:
                    t.mark_confirmed()
                elif t.frames_since_update > 1:
                    t.mark_deleted()
            elif t.state == TrackState.CONFIRMED:
                if t.frames_since_update > self._max_age:
                    t.mark_lost()
            elif t.state == TrackState.LOST:
                if t.frames_since_update > self._max_age + self._lost_buffer:
                    t.mark_deleted()

        # --- purge deleted tracks ---
        self._tracks = [t for t in self._tracks if not t.is_deleted]

        logger.debug(
            f"Tracker: {len(self._tracks)} active tracks ({sum(1 for t in self._tracks if t.is_confirmed)} confirmed)"
        )
        return [t for t in self._tracks if t.is_confirmed]

    @property
    def tracks(self) -> list[Tracklet]:
        return self._tracks

    # ------------------------------------------------------------------
    # Private matching helpers
    # ------------------------------------------------------------------

    def _match_iou(
        self,
        tracks: list[Tracklet],
        candidates: list[BallCandidate],
        det_indices: list[int],
        frame_bgr: np.ndarray,
        embeddings: np.ndarray | None,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not det_indices:
            return [], list(range(len(tracks))), det_indices

        pred_boxes = np.array([t.get_bbox() for t in tracks], dtype=np.float32)
        det_boxes = np.array([candidates[i].box for i in det_indices], dtype=np.float32)
        iou_mat = _box_iou(pred_boxes, det_boxes)  # (|tracks|, |dets|)
        cost = 1.0 - iou_mat

        trk_idx, det_idx = linear_sum_assignment(cost)
        matched, unmatched_trk, unmatched_det = [], [], []
        assigned_det = set()
        for ti, dj in zip(trk_idx, det_idx):
            if cost[ti, dj] > 1.0 - self._iou_thresh:
                continue
            matched.append((ti, det_indices[dj]))
            assigned_det.add(dj)
        unmatched_trk = [i for i in range(len(tracks)) if i not in {m[0] for m in matched}]
        unmatched_det = [det_indices[j] for j in range(len(det_indices)) if j not in assigned_det]
        return matched, unmatched_trk, unmatched_det

    def _match_reid(
        self,
        tracks: list[Tracklet],
        candidates: list[BallCandidate],
        det_indices: list[int],
        embeddings: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not det_indices:
            return [], list(range(len(tracks))), det_indices

        track_embs = []
        valid_trk = []
        for ti, t in enumerate(tracks):
            e = t.reid_embedding
            if e is not None:
                track_embs.append(e)
                valid_trk.append(ti)

        if not valid_trk:
            return [], list(range(len(tracks))), det_indices

        T_embs = np.array(track_embs, dtype=np.float32)
        D_embs = embeddings[det_indices]
        sim_mat = BallReIDExtractor.cosine_similarity(T_embs, D_embs)  # (|valid_trk|, |dets|)
        cost = 1.0 - sim_mat

        trk_idx, det_idx = linear_sum_assignment(cost)
        matched, assigned_trk, assigned_det = [], set(), set()
        for ti, dj in zip(trk_idx, det_idx):
            if sim_mat[ti, dj] < self._reid_thresh:
                continue
            matched.append((valid_trk[ti], det_indices[dj]))
            assigned_trk.add(ti)
            assigned_det.add(dj)

        unmatched_trk = [i for i in range(len(tracks)) if i not in {m[0] for m in matched}]
        unmatched_det = [det_indices[j] for j in range(len(det_indices)) if j not in assigned_det]
        return matched, unmatched_trk, unmatched_det


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes. Returns (M, N)."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area_a = np.maximum(ax2 - ax1, 0) * np.maximum(ay2 - ay1, 0)
    area_b = np.maximum(bx2 - bx1, 0) * np.maximum(by2 - by1, 0)

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union
