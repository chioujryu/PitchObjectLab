"""
Circle-based football tracking for RF-DETR video inference.

足球圓形追蹤（純邏輯，不載入模型，可單獨單元測試）：

- 偵測到足球後，以該球中心為圓心、`radius_pixels` 為半徑建立搜尋圓。
- 下一格的足球若落在某條 track 的搜尋圓內，視為同一顆球（中心距離 vs 半徑，
  絕不使用 IoU，因為快球相鄰格的 bbox 可能完全不重疊）。
- 相隔數格漏檢時，沿用最後建立的搜尋圓繼續判斷；半徑可隨漏檢格數成長。
- 確認匹配後，搜尋圓圓心更新到最新偵測到的足球中心，再往下追蹤。

This module is intentionally dependency-light (stdlib only) so it can be unit
tested without importing the RF-DETR training stack.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# Tracking fields attached to every prediction row when tracking is enabled.
# Non-target predictions receive these same keys set to None.
TRACK_FIELDS: tuple[str, ...] = (
    "track_id",
    "track_center_x",
    "track_center_y",
    "track_radius_pixels",
    "track_first_frame_index",
    "track_last_seen_frame_index",
    "track_age_frames",
    "track_hits",
    "track_confirmed",
)

_NULL_STRINGS = {"all", "null", "none", ""}

# Per-track trail points kept in memory when trajectory_max_points is unlimited (null).
DEFAULT_HISTORY_LIMIT = 1024


def _is_null_token(value: Any) -> bool:
    """Return True for None or the all/null/none sentinel strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in _NULL_STRINGS
    return False


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "on", "y", "t"}


def _as_float(value: Any, default: float, field_name: str) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"inference.tracking.{field_name} must be a number, got {value!r}") from exc


def _as_optional_float(value: Any, field_name: str) -> float | None:
    if _is_null_token(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"inference.tracking.{field_name} must be a number or null, got {value!r}") from exc


def _as_optional_int(value: Any, field_name: str) -> int | None:
    """Parse an int that may be the all/null sentinel (returns None for 'all'/null)."""
    if _is_null_token(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"inference.tracking.{field_name} must be an integer or all/null, got {value!r}") from exc


def _as_int(value: Any, default: int, field_name: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"inference.tracking.{field_name} must be an integer, got {value!r}") from exc


def _as_str(value: Any, default: str, field_name: str) -> str:
    """Parse a non-empty string with a fallback default (used for enum-like keys)."""
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _as_optional_str(value: Any) -> str | None:
    """Return a stripped string, or None for null/none/empty sentinels."""
    if _is_null_token(value):
        return None
    return str(value).strip()


def _as_unit_float(value: Any, default: float, field_name: str) -> float:
    """Parse a float constrained to the inclusive 0.0-1.0 range."""
    parsed = _as_float(value, default, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"inference.tracking.{field_name} must be between 0 and 1, got {parsed!r}")
    return parsed


def _as_positive_int(value: Any, default: int, field_name: str) -> int:
    """Parse a strictly positive integer."""
    parsed = _as_int(value, default, field_name)
    if parsed <= 0:
        raise ValueError(f"inference.tracking.{field_name} must be a positive integer, got {parsed!r}")
    return parsed


def _as_positive_float(value: Any, default: float, field_name: str) -> float:
    """Parse a strictly positive float (may exceed 1.0, unlike _as_unit_float)."""
    parsed = _as_float(value, default, field_name)
    if parsed <= 0.0:
        raise ValueError(f"inference.tracking.{field_name} must be a positive number, got {parsed!r}")
    return parsed


# Tracking algorithms: "circle" is the built-in stdlib tracker in this module; the rest are
# provided by the boxmot adapter (rf_detr_boxmot_tracker.BoxmotTracker).
ALLOWED_ALGORITHMS: tuple[str, ...] = ("circle", "ocsort", "deepocsort", "botsort", "bytetrack")
# boxmot trackers that consume an appearance ReID model (need reid_weights / device).
APPEARANCE_ALGORITHMS: tuple[str, ...] = ("deepocsort", "botsort")
_ALLOWED_CMC_METHODS = {"ecc", "orb", "sof", "sparseoptflow", "file", "files"}


@dataclass
class TrackingConfig:
    """Parsed and validated `inference.tracking` settings."""

    enabled: bool = False
    target_class_ids: set[int] = field(default_factory=set)
    radius_pixels: float = 80.0
    radius_scale: float | None = None
    radius_growth_per_missing_frame: float = 0.0
    max_radius_pixels: float | None = None
    max_missing_frames: int | None = None
    min_hits: int = 1
    use_velocity_prediction: bool = False
    velocity_smoothing: float = 0.5
    draw_trajectory: bool = True
    trajectory_max_points: int | None = 30
    trajectory_max_age_frames: int | None = 30
    trajectory_width: int = 2
    trajectory_per_track_color: bool = True
    trajectory_taper: bool = True
    draw_current_center: bool = True
    draw_search_circle: bool = False
    label_track_id: bool = True
    # Algorithm selector: "circle" (this module) or boxmot-backed trackers.
    algorithm: str = "circle"
    # Shared boxmot ReID / camera-motion settings (only used by boxmot algorithms).
    reid_weights: str | None = None
    reid_device: str | None = None
    reid_half: bool = False
    cmc_method: str | None = "ecc"
    per_class: bool = False
    # OC-SORT parameters.
    ocsort_det_thresh: float = 0.2
    ocsort_max_age: int = 30
    ocsort_min_hits: int = 3
    ocsort_asso_threshold: float = 0.3
    ocsort_delta_t: int = 3
    ocsort_asso_func: str = "iou"
    ocsort_inertia: float = 0.2
    ocsort_use_byte: bool = False
    ocsort_q_xy_scaling: float = 0.01
    ocsort_q_s_scaling: float = 0.0001
    # Deep OC-SORT parameters.
    deepocsort_det_thresh: float = 0.3
    deepocsort_max_age: int = 30
    deepocsort_min_hits: int = 3
    deepocsort_iou_threshold: float = 0.3
    deepocsort_delta_t: int = 3
    deepocsort_asso_func: str = "iou"
    deepocsort_inertia: float = 0.2
    deepocsort_w_association_emb: float = 0.5
    deepocsort_alpha_fixed_emb: float = 0.95
    deepocsort_embedding_off: bool = False
    deepocsort_cmc_off: bool = False
    # BoT-SORT parameters.
    botsort_track_high_thresh: float = 0.5
    botsort_track_low_thresh: float = 0.1
    botsort_new_track_thresh: float = 0.6
    botsort_track_buffer: int = 30
    botsort_match_thresh: float = 0.8
    botsort_proximity_thresh: float = 0.5
    botsort_appearance_thresh: float = 0.25
    botsort_with_reid: bool = True
    botsort_fuse_first_associate: bool = False
    # ByteTrack parameters.
    bytetrack_track_thresh: float = 0.45
    bytetrack_match_thresh: float = 0.8
    bytetrack_track_buffer: int = 25
    bytetrack_frame_rate: int = 30


@dataclass
class TrackedBall:
    """A single tracked football and the search circle that follows it."""

    track_id: int
    center_x: float
    center_y: float
    base_radius: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_initialized: bool = False
    missing_frames: int = 0
    hits: int = 0
    first_frame_index: int = 0
    last_seen_frame_index: int = 0
    points: deque[tuple[int, float, float]] = field(default_factory=deque)


def bbox_center(prediction: Mapping[str, Any]) -> tuple[float, float]:
    """Return the center (cx, cy) of a COCO xywh bbox."""
    x, y, width, height = [float(value) for value in prediction.get("bbox", [0, 0, 0, 0])[:4]]
    return x + width / 2.0, y + height / 2.0


def ball_size(prediction: Mapping[str, Any]) -> float:
    """Return the larger bbox side, used for size-relative radii."""
    _, _, width, height = [float(value) for value in prediction.get("bbox", [0, 0, 0, 0])[:4]]
    return max(width, height)


def effective_radius(track: TrackedBall, cfg: TrackingConfig) -> float:
    """Search radius for matching, grown by missing frames and capped."""
    radius = track.base_radius + cfg.radius_growth_per_missing_frame * track.missing_frames
    if cfg.max_radius_pixels is not None:
        radius = min(radius, cfg.max_radius_pixels)
    return radius


def predicted_center(track: TrackedBall, cfg: TrackingConfig, elapsed_frames: int = 0) -> tuple[float, float]:
    """Gate center: extrapolate by velocity over the elapsed frames when enabled, else last center."""
    if cfg.use_velocity_prediction and elapsed_frames > 0:
        return (
            track.center_x + track.velocity_x * elapsed_frames,
            track.center_y + track.velocity_y * elapsed_frames,
        )
    return track.center_x, track.center_y


def _history_maxlen(cfg: TrackingConfig) -> int:
    """Maximum trail points kept in memory per track (bounds long-video memory)."""
    if cfg.trajectory_max_points is not None and cfg.trajectory_max_points > 0:
        return int(cfg.trajectory_max_points)
    return DEFAULT_HISTORY_LIMIT


def is_track_visible(track: TrackedBall, current_frame_index: int | None, cfg: TrackingConfig) -> bool:
    """Whether a track's overlay should be drawn: confirmed and seen within trajectory_max_age_frames."""
    if track.hits < cfg.min_hits:
        return False
    if cfg.trajectory_max_age_frames is None or current_frame_index is None:
        return True
    return current_frame_index - track.last_seen_frame_index <= cfg.trajectory_max_age_frames


def trail_points(track: TrackedBall, current_frame_index: int | None, cfg: TrackingConfig) -> list[tuple[float, float]]:
    """Trail (x, y) points to draw; points older than trajectory_max_age_frames are dropped so the trail retracts."""
    if cfg.trajectory_max_age_frames is None or current_frame_index is None:
        return [(point_x, point_y) for (_, point_x, point_y) in track.points]
    max_age = cfg.trajectory_max_age_frames
    return [
        (point_x, point_y)
        for (frame_index, point_x, point_y) in track.points
        if current_frame_index - frame_index <= max_age
    ]


def live_center(track: TrackedBall, current_frame_index: int | None, cfg: TrackingConfig) -> tuple[float, float]:
    """Current drawn position: velocity-extrapolated through a detection gap, else the last detected center."""
    if current_frame_index is None:
        return track.center_x, track.center_y
    return predicted_center(track, cfg, current_frame_index - track.last_seen_frame_index)


def _category_name_to_id(categories: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {str(category.get("name", category["id"])).casefold(): int(category["id"]) for category in categories}


def parse_tracking_config(
    config: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
) -> TrackingConfig:
    """Build a TrackingConfig from `inference.tracking`, resolving target class IDs."""
    raw = dict((config.get("inference", {}) or {}).get("tracking", {}) or {})

    target_ids: set[int] = {int(value) for value in (raw.get("target_class_ids") or [])}
    if not target_ids:
        name_to_id = _category_name_to_id(categories)
        names = [str(value).casefold() for value in (raw.get("target_class_names") or [])]
        target_ids = {name_to_id[name] for name in names if name in name_to_id}
        if not target_ids and "football" in name_to_id:
            target_ids = {name_to_id["football"]}

    radius_pixels = _as_float(raw.get("radius_pixels"), 80.0, "radius_pixels")
    trajectory_width = _as_int(raw.get("trajectory_width"), 2, "trajectory_width")
    min_hits = _as_int(raw.get("min_hits"), 1, "min_hits")
    max_missing_frames = _as_optional_int(raw.get("max_missing_frames", 30), "max_missing_frames")
    radius_growth = _as_float(raw.get("radius_growth_per_missing_frame"), 0.0, "radius_growth_per_missing_frame")
    velocity_smoothing = _as_float(raw.get("velocity_smoothing"), 0.5, "velocity_smoothing")
    trajectory_max_age_frames = _as_optional_int(raw.get("trajectory_max_age_frames", 30), "trajectory_max_age_frames")

    if radius_pixels <= 0:
        raise ValueError("inference.tracking.radius_pixels must be > 0")
    if trajectory_width <= 0:
        raise ValueError("inference.tracking.trajectory_width must be > 0")
    if min_hits < 1:
        raise ValueError("inference.tracking.min_hits must be >= 1")
    if max_missing_frames is not None and max_missing_frames < 0:
        raise ValueError("inference.tracking.max_missing_frames must be all/null or a non-negative integer")
    if radius_growth < 0:
        raise ValueError("inference.tracking.radius_growth_per_missing_frame must be non-negative")
    if not 0.0 <= velocity_smoothing <= 1.0:
        raise ValueError("inference.tracking.velocity_smoothing must be between 0 and 1")
    if trajectory_max_age_frames is not None and trajectory_max_age_frames <= 0:
        raise ValueError("inference.tracking.trajectory_max_age_frames must be all/null or a positive integer")

    algorithm = _as_str(raw.get("algorithm"), "circle", "algorithm").casefold()
    if algorithm not in ALLOWED_ALGORITHMS:
        raise ValueError(f"inference.tracking.algorithm must be one of {list(ALLOWED_ALGORITHMS)}, got {algorithm!r}")
    cmc_method = _as_optional_str(raw.get("cmc_method", "ecc"))
    if cmc_method is not None and cmc_method.casefold() not in _ALLOWED_CMC_METHODS:
        raise ValueError(
            f"inference.tracking.cmc_method must be null or one of {sorted(_ALLOWED_CMC_METHODS)}, got {cmc_method!r}"
        )
    # Per-algorithm boxmot params live in nested sub-blocks; map them to flat dataclass fields.
    ocsort = dict(raw.get("ocsort", {}) or {})
    deepocsort = dict(raw.get("deepocsort", {}) or {})
    botsort = dict(raw.get("botsort", {}) or {})
    bytetrack = dict(raw.get("bytetrack", {}) or {})

    return TrackingConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        target_class_ids=target_ids,
        radius_pixels=radius_pixels,
        radius_scale=_as_optional_float(raw.get("radius_scale"), "radius_scale"),
        radius_growth_per_missing_frame=radius_growth,
        max_radius_pixels=_as_optional_float(raw.get("max_radius_pixels"), "max_radius_pixels"),
        max_missing_frames=max_missing_frames,
        min_hits=min_hits,
        use_velocity_prediction=_as_bool(raw.get("use_velocity_prediction"), False),
        velocity_smoothing=velocity_smoothing,
        draw_trajectory=_as_bool(raw.get("draw_trajectory"), True),
        trajectory_max_points=_as_optional_int(raw.get("trajectory_max_points", 30), "trajectory_max_points"),
        trajectory_max_age_frames=trajectory_max_age_frames,
        trajectory_width=trajectory_width,
        trajectory_per_track_color=_as_bool(raw.get("trajectory_per_track_color"), True),
        trajectory_taper=_as_bool(raw.get("trajectory_taper"), True),
        draw_current_center=_as_bool(raw.get("draw_current_center"), True),
        draw_search_circle=_as_bool(raw.get("draw_search_circle"), False),
        label_track_id=_as_bool(raw.get("label_track_id"), True),
        algorithm=algorithm,
        reid_weights=_as_optional_str(raw.get("reid_weights")),
        reid_device=_as_optional_str(raw.get("reid_device")),
        reid_half=_as_bool(raw.get("reid_half"), False),
        cmc_method=cmc_method,
        per_class=_as_bool(raw.get("per_class"), False),
        ocsort_det_thresh=_as_unit_float(ocsort.get("det_thresh"), 0.2, "ocsort.det_thresh"),
        ocsort_max_age=_as_positive_int(ocsort.get("max_age"), 30, "ocsort.max_age"),
        ocsort_min_hits=_as_positive_int(ocsort.get("min_hits"), 3, "ocsort.min_hits"),
        ocsort_asso_threshold=_as_unit_float(ocsort.get("asso_threshold"), 0.3, "ocsort.asso_threshold"),
        ocsort_delta_t=_as_positive_int(ocsort.get("delta_t"), 3, "ocsort.delta_t"),
        ocsort_asso_func=_as_str(ocsort.get("asso_func"), "iou", "ocsort.asso_func"),
        ocsort_inertia=_as_unit_float(ocsort.get("inertia"), 0.2, "ocsort.inertia"),
        ocsort_use_byte=_as_bool(ocsort.get("use_byte"), False),
        ocsort_q_xy_scaling=_as_positive_float(ocsort.get("Q_xy_scaling"), 0.01, "ocsort.Q_xy_scaling"),
        ocsort_q_s_scaling=_as_positive_float(ocsort.get("Q_s_scaling"), 0.0001, "ocsort.Q_s_scaling"),
        deepocsort_det_thresh=_as_unit_float(deepocsort.get("det_thresh"), 0.3, "deepocsort.det_thresh"),
        deepocsort_max_age=_as_positive_int(deepocsort.get("max_age"), 30, "deepocsort.max_age"),
        deepocsort_min_hits=_as_positive_int(deepocsort.get("min_hits"), 3, "deepocsort.min_hits"),
        deepocsort_iou_threshold=_as_unit_float(deepocsort.get("iou_threshold"), 0.3, "deepocsort.iou_threshold"),
        deepocsort_delta_t=_as_positive_int(deepocsort.get("delta_t"), 3, "deepocsort.delta_t"),
        deepocsort_asso_func=_as_str(deepocsort.get("asso_func"), "iou", "deepocsort.asso_func"),
        deepocsort_inertia=_as_unit_float(deepocsort.get("inertia"), 0.2, "deepocsort.inertia"),
        deepocsort_w_association_emb=_as_unit_float(
            deepocsort.get("w_association_emb"), 0.5, "deepocsort.w_association_emb"
        ),
        deepocsort_alpha_fixed_emb=_as_unit_float(
            deepocsort.get("alpha_fixed_emb"), 0.95, "deepocsort.alpha_fixed_emb"
        ),
        deepocsort_embedding_off=_as_bool(deepocsort.get("embedding_off"), False),
        deepocsort_cmc_off=_as_bool(deepocsort.get("cmc_off"), False),
        botsort_track_high_thresh=_as_unit_float(botsort.get("track_high_thresh"), 0.5, "botsort.track_high_thresh"),
        botsort_track_low_thresh=_as_unit_float(botsort.get("track_low_thresh"), 0.1, "botsort.track_low_thresh"),
        botsort_new_track_thresh=_as_unit_float(botsort.get("new_track_thresh"), 0.6, "botsort.new_track_thresh"),
        botsort_track_buffer=_as_positive_int(botsort.get("track_buffer"), 30, "botsort.track_buffer"),
        botsort_match_thresh=_as_unit_float(botsort.get("match_thresh"), 0.8, "botsort.match_thresh"),
        botsort_proximity_thresh=_as_unit_float(botsort.get("proximity_thresh"), 0.5, "botsort.proximity_thresh"),
        botsort_appearance_thresh=_as_unit_float(botsort.get("appearance_thresh"), 0.25, "botsort.appearance_thresh"),
        botsort_with_reid=_as_bool(botsort.get("with_reid"), True),
        botsort_fuse_first_associate=_as_bool(botsort.get("fuse_first_associate"), False),
        bytetrack_track_thresh=_as_unit_float(bytetrack.get("track_thresh"), 0.45, "bytetrack.track_thresh"),
        bytetrack_match_thresh=_as_unit_float(bytetrack.get("match_thresh"), 0.8, "bytetrack.match_thresh"),
        bytetrack_track_buffer=_as_positive_int(bytetrack.get("track_buffer"), 25, "bytetrack.track_buffer"),
        bytetrack_frame_rate=_as_positive_int(bytetrack.get("frame_rate"), 30, "bytetrack.frame_rate"),
    )


class FootballTracker:
    """Greedy, motion-only circle tracker. One search circle follows each ball."""

    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.tracks: list[TrackedBall] = []
        self.next_id = 1

    def _is_target(self, prediction: Mapping[str, Any]) -> bool:
        return int(prediction.get("category_id", -1)) in self.cfg.target_class_ids

    def _base_radius(self, prediction: Mapping[str, Any]) -> float:
        base = self.cfg.radius_pixels
        if self.cfg.radius_scale is not None:
            base = max(base, self.cfg.radius_scale * ball_size(prediction))
        return base

    def _gate_distance(self, track: TrackedBall, center_x: float, center_y: float, frame_index: int) -> float:
        """Distance to the search gate: min of the static circle and the velocity-predicted circle."""
        static = math.hypot(center_x - track.center_x, center_y - track.center_y)
        if self.cfg.use_velocity_prediction:
            elapsed = frame_index - track.last_seen_frame_index
            if elapsed > 0:
                gate_x, gate_y = predicted_center(track, self.cfg, elapsed)
                return min(static, math.hypot(center_x - gate_x, center_y - gate_y))
        return static

    def _apply_match(
        self, track: TrackedBall, prediction: Mapping[str, Any], center_x: float, center_y: float, frame_index: int
    ) -> None:
        """Re-center the circle on the matched detection and update velocity (EMA) + trail."""
        cfg = self.cfg
        gap = max(1, frame_index - track.last_seen_frame_index)
        new_velocity_x = (center_x - track.center_x) / gap
        new_velocity_y = (center_y - track.center_y) / gap
        if track.velocity_initialized:
            smoothing = cfg.velocity_smoothing
            track.velocity_x = smoothing * track.velocity_x + (1.0 - smoothing) * new_velocity_x
            track.velocity_y = smoothing * track.velocity_y + (1.0 - smoothing) * new_velocity_y
        else:
            track.velocity_x = new_velocity_x
            track.velocity_y = new_velocity_y
            track.velocity_initialized = True
        track.center_x = center_x
        track.center_y = center_y
        track.base_radius = self._base_radius(prediction)
        track.missing_frames = 0
        track.hits += 1
        track.last_seen_frame_index = frame_index
        track.points.append((frame_index, center_x, center_y))

    def update(
        self, frame_index: int, predictions: Sequence[Mapping[str, Any]], frame: Any = None
    ) -> list[dict[str, Any]]:
        """Associate this frame's detections, returning rows (input order) with tracking fields.

        `frame` is accepted for interface parity with the boxmot adapter and ignored here
        (the circle tracker is motion/position only).
        """
        cfg = self.cfg
        # Score-desc order breaks distance ties and orders new-track creation.
        ordered_targets = sorted(
            ((index, pred) for index, pred in enumerate(predictions) if self._is_target(pred)),
            key=lambda item: -float(item[1].get("score", 0.0)),
        )
        centers: dict[int, tuple[float, float]] = {index: bbox_center(pred) for index, pred in ordered_targets}
        score_rank: dict[int, int] = {index: rank for rank, (index, _) in enumerate(ordered_targets)}

        # Global association: collect every valid detection<->track pair, then assign nearest first.
        candidates = []
        for index, _pred in ordered_targets:
            center_x, center_y = centers[index]
            for track in self.tracks:
                distance = self._gate_distance(track, center_x, center_y, frame_index)
                if distance <= effective_radius(track, cfg):
                    candidates.append((distance, track.track_id, score_rank[index], index, track))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        assigned: dict[int, TrackedBall] = {}
        for _distance, track_id, _rank, index, track in candidates:
            if track_id in used_tracks or index in used_dets:
                continue
            center_x, center_y = centers[index]
            self._apply_match(track, predictions[index], center_x, center_y, frame_index)
            used_tracks.add(track_id)
            used_dets.add(index)
            assigned[index] = track

        # Unmatched detections start new tracks in score-desc order (deterministic ids).
        for index, pred in ordered_targets:
            if index in used_dets:
                continue
            center_x, center_y = centers[index]
            track = TrackedBall(
                track_id=self.next_id,
                center_x=center_x,
                center_y=center_y,
                base_radius=self._base_radius(pred),
                missing_frames=0,
                hits=1,
                first_frame_index=frame_index,
                last_seen_frame_index=frame_index,
                points=deque([(frame_index, center_x, center_y)], maxlen=_history_maxlen(cfg)),
            )
            self.next_id += 1
            self.tracks.append(track)
            used_dets.add(index)
            assigned[index] = track

        # Age tracks that did not match this frame; the circle stays at the last center.
        for track in self.tracks:
            if track.track_id not in used_tracks:
                track.missing_frames += 1

        if cfg.max_missing_frames is not None:
            self.tracks = [track for track in self.tracks if track.missing_frames <= cfg.max_missing_frames]

        rows: list[dict[str, Any]] = []
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

    def confirmed_tracks(self) -> list[TrackedBall]:
        """Tracks that have reached min_hits (used for rendering and summaries)."""
        return [track for track in self.tracks if track.hits >= self.cfg.min_hits]


def build_tracking_summary(all_predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-(source, track_id) statistics from prediction rows."""
    groups: dict[tuple[Any, int], dict[str, Any]] = {}
    for row in all_predictions:
        track_id = row.get("track_id")
        if track_id is None:
            continue
        key = (row.get("source"), int(track_id))
        group = groups.get(key)
        if group is None:
            group = {
                "source": row.get("source"),
                "track_id": int(track_id),
                "first_frame_index": None,
                "last_frame_index": None,
                "num_points": 0,
                "hits": 0,
                "confirmed": False,
            }
            groups[key] = group
        frame_index = row.get("frame_index")
        if frame_index is not None:
            frame_index = int(frame_index)
            if group["first_frame_index"] is None or frame_index < group["first_frame_index"]:
                group["first_frame_index"] = frame_index
            if group["last_frame_index"] is None or frame_index > group["last_frame_index"]:
                group["last_frame_index"] = frame_index
        group["num_points"] += 1
        group["hits"] = max(group["hits"], int(row.get("track_hits") or 0))
        group["confirmed"] = group["confirmed"] or bool(row.get("track_confirmed"))

    tracks: list[dict[str, Any]] = []
    for group in sorted(groups.values(), key=lambda item: (str(item["source"]), item["track_id"])):
        first, last = group["first_frame_index"], group["last_frame_index"]
        group["lifespan_frames"] = (last - first + 1) if first is not None and last is not None else None
        tracks.append(group)

    return {
        "tracks": tracks,
        "track_count": len(tracks),
        "confirmed_count": sum(1 for group in tracks if group["confirmed"]),
    }
