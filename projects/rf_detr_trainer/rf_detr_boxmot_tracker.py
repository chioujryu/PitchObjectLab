"""
boxmot-backed multi-object tracking adapter for RF-DETR video inference.

把 OC-SORT / Deep OC-SORT / BoT-SORT / ByteTrack（皆來自 `boxmot` 套件）包成與
`rf_detr_video_tracking.FootballTracker` 完全相同的介面，因此既有的軌跡渲染、
`predictions.jsonl` 欄位與 `tracking_summary.json` 都不需更動即可沿用。

設計重點：
- `boxmot` 只在 `_build_boxmot_tracker()` 內**延遲載入**，所以匯入本模組本身不需要 boxmot
  （測試可 monkeypatch `_build_boxmot_tracker` 注入假 tracker，完全離線執行）。
- boxmot `update(dets, img)` 的契約（已針對 boxmot==13.0.0 驗證）：
    輸入 dets：np.ndarray (N,6) = [x1,y1,x2,y2,conf,cls]（xyxy 像素）＋ BGR HxWx3 影格。
    輸出     ：np.ndarray (M,8) = [x1,y1,x2,y2,track_id,conf,cls,det_ind]，
              其中 det_ind（第 7 欄）是該輸出對應到輸入 dets 的索引——用它把 track 精準
              對回原始 prediction（M 可能 < N 且順序不同，所以絕不可用位置 zip）。
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from rf_detr_video_tracking import (
    TRACK_FIELDS,
    TrackedBall,
    TrackingConfig,
    _history_maxlen,
)

# Lightweight default ReID model boxmot uses when no local weights are configured.
DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"


def boxmot_importable() -> bool:
    """Return True when the boxmot package can be imported (used to gate optional tests)."""
    try:
        import boxmot  # noqa: F401
    except Exception:
        return False
    return True


def resolve_reid_weights(cfg: TrackingConfig) -> Path:
    """Local ReID weights path when configured, else the default model name for boxmot auto-download."""
    if cfg.reid_weights:
        return Path(str(cfg.reid_weights)).expanduser()
    return Path(DEFAULT_REID_WEIGHTS)


def effective_reid_half(cfg: TrackingConfig, device: str) -> bool:
    """FP16 ReID inference is GPU-only; force it off on cpu/mps regardless of config."""
    return bool(cfg.reid_half) and str(device).strip().lower().startswith("cuda")


def _build_boxmot_tracker(cfg: TrackingConfig, device: str) -> Any:
    """Construct the underlying boxmot tracker for ``cfg.algorithm``.

    boxmot is imported here (lazily) so importing this module never requires the dependency.
    Tests monkeypatch this function to inject a fake tracker and run fully offline.
    """
    try:
        # Pinned to boxmot==13.0.0: later releases (v19+) removed these top-level tracker
        # classes and changed the update() return contract (the Nx8 + det_ind columns this
        # adapter relies on), so do not bump the pin without re-validating the mapping below.
        from boxmot import BotSort, ByteTrack, DeepOcSort, OcSort
    except ImportError as exc:  # pragma: no cover - exercised only when boxmot is absent
        raise ImportError(
            f"inference.tracking.algorithm={cfg.algorithm!r} requires the 'boxmot' package. "
            "Install it with: uv add 'boxmot==13.0.0' "
            "(or set inference.tracking.algorithm: circle to use the built-in tracker)."
        ) from exc

    algorithm = cfg.algorithm
    if algorithm == "ocsort":
        return OcSort(
            per_class=cfg.per_class,
            det_thresh=cfg.ocsort_det_thresh,
            max_age=cfg.ocsort_max_age,
            min_hits=cfg.ocsort_min_hits,
            asso_threshold=cfg.ocsort_asso_threshold,
            delta_t=cfg.ocsort_delta_t,
            asso_func=cfg.ocsort_asso_func,
            inertia=cfg.ocsort_inertia,
            use_byte=cfg.ocsort_use_byte,
            Q_xy_scaling=cfg.ocsort_q_xy_scaling,
            Q_s_scaling=cfg.ocsort_q_s_scaling,
        )
    if algorithm == "bytetrack":
        return ByteTrack(
            track_thresh=cfg.bytetrack_track_thresh,
            match_thresh=cfg.bytetrack_match_thresh,
            track_buffer=cfg.bytetrack_track_buffer,
            frame_rate=cfg.bytetrack_frame_rate,
            per_class=cfg.per_class,
        )
    if algorithm == "deepocsort":
        return DeepOcSort(
            reid_weights=resolve_reid_weights(cfg),
            device=device,
            half=effective_reid_half(cfg, device),
            per_class=cfg.per_class,
            det_thresh=cfg.deepocsort_det_thresh,
            max_age=cfg.deepocsort_max_age,
            min_hits=cfg.deepocsort_min_hits,
            iou_threshold=cfg.deepocsort_iou_threshold,
            delta_t=cfg.deepocsort_delta_t,
            asso_func=cfg.deepocsort_asso_func,
            inertia=cfg.deepocsort_inertia,
            w_association_emb=cfg.deepocsort_w_association_emb,
            alpha_fixed_emb=cfg.deepocsort_alpha_fixed_emb,
            embedding_off=cfg.deepocsort_embedding_off,
            cmc_off=cfg.deepocsort_cmc_off,
        )
    if algorithm == "botsort":
        return BotSort(
            reid_weights=resolve_reid_weights(cfg),
            device=device,
            half=effective_reid_half(cfg, device),
            per_class=cfg.per_class,
            track_high_thresh=cfg.botsort_track_high_thresh,
            track_low_thresh=cfg.botsort_track_low_thresh,
            new_track_thresh=cfg.botsort_new_track_thresh,
            track_buffer=cfg.botsort_track_buffer,
            match_thresh=cfg.botsort_match_thresh,
            proximity_thresh=cfg.botsort_proximity_thresh,
            appearance_thresh=cfg.botsort_appearance_thresh,
            cmc_method=cfg.cmc_method or "ecc",
            fuse_first_associate=cfg.botsort_fuse_first_associate,
            with_reid=cfg.botsort_with_reid,
        )
    raise ValueError(f"Unsupported boxmot algorithm: {algorithm!r}")


class BoxmotTracker:
    """Adapter exposing the FootballTracker interface on top of a boxmot tracker.

    ``.tracks`` holds TrackedBall objects so the existing ``draw_track_overlays`` and the
    module-level render helpers (is_track_visible / trail_points / live_center / effective_radius)
    consume it unchanged, and ``update`` emits the same TRACK_FIELDS as FootballTracker.
    """

    def __init__(
        self,
        cfg: TrackingConfig,
        device: str = "cpu",
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.frame_size = frame_size  # (width, height)
        self.tracks: List[TrackedBall] = []
        self._by_id: Dict[int, TrackedBall] = {}
        self._tracker = _build_boxmot_tracker(cfg, device)

    def _is_target(self, prediction: Mapping[str, Any]) -> bool:
        return int(prediction.get("category_id", -1)) in self.cfg.target_class_ids

    def _resolve_frame(self, frame: Any) -> Any:
        """Return a BGR image for boxmot: the real frame, else a synthesized blank from frame_size."""
        if frame is not None:
            return frame
        if self.frame_size is not None:
            width, height = self.frame_size
            return np.zeros((int(height), int(width), 3), dtype=np.uint8)
        raise ValueError(
            f"BoxmotTracker (algorithm={self.cfg.algorithm}) requires the video frame; "
            "pass frame=... to update() or provide frame_size."
        )

    def _default_max_missing(self) -> int:
        """Overlay-retraction window when max_missing_frames is 'all'/null; mirror the tracker buffer."""
        cfg = self.cfg
        return {
            "ocsort": cfg.ocsort_max_age,
            "deepocsort": cfg.deepocsort_max_age,
            "botsort": cfg.botsort_track_buffer,
            "bytetrack": cfg.bytetrack_track_buffer,
        }.get(cfg.algorithm, 30)

    def build_dets(
        self, predictions: Sequence[Mapping[str, Any]]
    ) -> Tuple[np.ndarray, List[int]]:
        """Build the (K,6) [x1,y1,x2,y2,conf,cls] det array for target classes only.

        Returns the array plus ``target_local_to_orig`` mapping each det row back to the
        original prediction index (so the boxmot ``det_ind`` output column resolves cleanly).
        """
        target_local_to_orig: List[int] = []
        det_rows: List[List[float]] = []
        for index, pred in enumerate(predictions):
            if not self._is_target(pred):
                continue
            x, y, w, h = [float(value) for value in pred.get("bbox", [0, 0, 0, 0])[:4]]
            det_rows.append(
                [x, y, x + w, y + h, float(pred.get("score", 0.0)), float(pred.get("category_id", 0))]
            )
            target_local_to_orig.append(index)
        dets = np.asarray(det_rows, dtype=float) if det_rows else np.empty((0, 6), dtype=float)
        return dets, target_local_to_orig

    def update(
        self, frame_index: int, predictions: Sequence[Mapping[str, Any]], frame: Any = None
    ) -> List[Dict[str, Any]]:
        """Run boxmot for one frame and return rows (input order) with TRACK_FIELDS attached."""
        cfg = self.cfg
        dets, target_local_to_orig = self.build_dets(predictions)
        target_count = len(target_local_to_orig)
        img = self._resolve_frame(frame)

        out = self._tracker.update(dets, img)
        out_array = np.asarray(out, dtype=float) if out is not None else np.empty((0, 8), dtype=float)

        assigned: Dict[int, TrackedBall] = {}
        seen_ids: Set[int] = set()
        for row in out_array:
            if len(row) < 8:
                # 8 columns (incl. det_ind) are required to map back; skip otherwise.
                continue
            track_id = int(round(float(row[4])))
            det_ind = int(round(float(row[7])))
            if not 0 <= det_ind < target_count:
                # Predicted-but-unmatched / version drift: ignore (no original row to attach to).
                continue
            orig_index = target_local_to_orig[det_ind]
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            base_radius = max(1.0, 0.5 * max(x2 - x1, y2 - y1))
            track = self._by_id.get(track_id)
            if track is None:
                track = TrackedBall(
                    track_id=track_id,
                    center_x=center_x,
                    center_y=center_y,
                    base_radius=base_radius,
                    missing_frames=0,
                    hits=1,
                    first_frame_index=frame_index,
                    last_seen_frame_index=frame_index,
                    points=deque([(frame_index, center_x, center_y)], maxlen=_history_maxlen(cfg)),
                )
                self._by_id[track_id] = track
                self.tracks.append(track)
            else:
                track.center_x = center_x
                track.center_y = center_y
                track.base_radius = base_radius
                track.hits += 1
                track.missing_frames = 0
                track.last_seen_frame_index = frame_index
                track.points.append((frame_index, center_x, center_y))
            assigned[orig_index] = track
            seen_ids.add(track_id)

        # Age tracks not emitted this frame; prune stale ones so overlays retract.
        max_missing = cfg.max_missing_frames if cfg.max_missing_frames is not None else self._default_max_missing()
        for track in self.tracks:
            if track.track_id not in seen_ids:
                track.missing_frames += 1
        if max_missing is not None:
            survivors = [track for track in self.tracks if track.missing_frames <= max_missing]
            if len(survivors) != len(self.tracks):
                self.tracks = survivors
                self._by_id = {track.track_id: track for track in survivors}

        rows: List[Dict[str, Any]] = []
        for index, pred in enumerate(predictions):
            row = dict(pred)
            track = assigned.get(index)
            if track is None:
                for key in TRACK_FIELDS:
                    row[key] = None
            else:
                row["track_id"] = track.track_id
                row["track_center_x"] = track.center_x
                row["track_center_y"] = track.center_y
                row["track_radius_pixels"] = track.base_radius
                row["track_first_frame_index"] = track.first_frame_index
                row["track_last_seen_frame_index"] = track.last_seen_frame_index
                row["track_age_frames"] = track.last_seen_frame_index - track.first_frame_index + 1
                row["track_hits"] = track.hits
                row["track_confirmed"] = track.hits >= cfg.min_hits
            rows.append(row)
        return rows

    def confirmed_tracks(self) -> List[TrackedBall]:
        """Tracks that have reached min_hits (used for rendering and summaries)."""
        return [track for track in self.tracks if track.hits >= self.cfg.min_hits]
