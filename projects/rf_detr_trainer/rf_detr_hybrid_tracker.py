"""Hybrid motion-only football tracker with delayed, deterministic output.

The tracker combines confidence-tier association (ByteTrack), observation-centric
velocity correction (OC-SORT), camera-motion compensation (BoT-SORT), and an
adaptive constant-velocity/constant-acceleration motion model. It deliberately
does not use appearance, field masks, player context, or a main-ball heuristic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


_INVALID_COST = 1.0e6


@dataclass(frozen=True)
class HybridTrackingConfig:
    """Validated hybrid tracker settings; times are seconds, distances pixels."""

    low_confidence: float = 0.25
    high_confidence: float = 0.50
    new_track_confidence: float = 0.50
    low_require_recheck: bool = True
    lookahead_seconds: float = 0.50
    ambiguity_margin: float = 0.02
    confirmed_hits: int = 2
    confirmation_window: int = 3
    lost_seconds: float = 1.0
    predicted_output_seconds: float = 0.50
    high_gate_pixels: float = 72.0
    low_gate_pixels: float = 56.0
    mahalanobis_gate: float = 5.0
    weight_mahalanobis: float = 0.28
    weight_center: float = 0.28
    weight_direction: float = 0.0
    weight_size: float = 0.14
    weight_confidence: float = 0.10
    low_gate_scale: float = 0.75
    process_noise: float = 12.0
    measurement_noise: float = 8.0
    acceleration_smoothing: float = 0.70
    size_smoothing: float = 0.80
    model_probability_smoothing: float = 0.80
    cmc_enabled: bool = True
    cmc_max_corners: int = 400
    cmc_quality_level: float = 0.01
    cmc_min_distance: float = 7.0
    cmc_ransac_threshold: float = 3.0
    cmc_min_inliers: int = 12
    cmc_max_translation: float = 250.0
    cmc_min_scale: float = 0.80
    cmc_max_scale: float = 1.25
    cmc_processing_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("low_confidence", "high_confidence", "new_track_confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"hybrid.{name} must be between 0 and 1")
        if self.low_confidence > self.high_confidence:
            raise ValueError("hybrid.low_confidence must not exceed high_confidence")
        if self.new_track_confidence < self.high_confidence:
            raise ValueError("hybrid.new_track_confidence must be >= high_confidence")
        for name in ("lookahead_seconds", "lost_seconds", "predicted_output_seconds"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"hybrid.{name} must be non-negative")
        for name in ("confirmed_hits", "confirmation_window", "cmc_min_inliers"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"hybrid.{name} must be >= 1")
        if self.confirmed_hits > self.confirmation_window:
            raise ValueError("hybrid.confirmed_hits must not exceed confirmation_window")
        if not 0.0 < float(self.cmc_processing_scale) <= 1.0:
            raise ValueError("hybrid.cmc_processing_scale must be in the range (0, 1]")

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "HybridTrackingConfig":
        """Parse grouped YAML blocks while rejecting unknown settings."""
        raw = dict(raw or {})
        groups = {
            "candidate": {
                "low_confidence", "high_confidence", "new_track_confidence", "low_require_recheck"
            },
            "hypothesis": {"lookahead_seconds", "ambiguity_margin"},
            "lifecycle": {
                "confirmed_hits", "confirmation_window", "lost_seconds", "predicted_output_seconds"
            },
            "association": {
                "high_gate_pixels", "low_gate_pixels", "mahalanobis_gate", "weight_mahalanobis",
                "weight_center", "weight_direction", "weight_size", "weight_confidence", "low_gate_scale"
            },
            "motion": {
                "process_noise", "measurement_noise", "acceleration_smoothing", "size_smoothing",
                "model_probability_smoothing"
            },
            "cmc": {
                "cmc_enabled", "cmc_max_corners", "cmc_quality_level", "cmc_min_distance",
                "cmc_ransac_threshold", "cmc_min_inliers", "cmc_max_translation", "cmc_min_scale",
                "cmc_max_scale", "cmc_processing_scale", "processing_scale"
            },
        }
        values: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in groups:
                if not isinstance(value, Mapping):
                    raise ValueError(f"inference.tracking.hybrid.{key} must be a mapping")
                for child, child_value in value.items():
                    if key == "hypothesis" and child == "beam_width":
                        raise ValueError(
                            "inference.tracking.hybrid.hypothesis.beam_width is unsupported; remove it"
                        )
                    if child not in groups[key]:
                        raise ValueError(f"unknown inference.tracking.hybrid.{key}.{child}")
                    values["cmc_processing_scale" if child == "processing_scale" else child] = child_value
            elif key == "beam_width":
                raise ValueError("inference.tracking.hybrid.beam_width is unsupported; remove it")
            elif key in cls.__dataclass_fields__:
                values[key] = value
            else:
                raise ValueError(f"unknown inference.tracking.hybrid.{key}")
        return cls(**values)


@dataclass
class _Detection:
    index: int
    row: Mapping[str, Any]
    cx: float
    cy: float
    width: float
    height: float
    log_width: float
    log_height: float
    score: float


@dataclass
class _Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    first_frame_index: int
    last_frame_index: int
    first_timestamp: float
    last_timestamp: float
    last_observed_timestamp: float
    last_observed_center: np.ndarray
    hits: int = 1
    age_frames: int = 1
    misses: int = 0
    status: str = "tentative"
    cv_probability: float = 0.5
    ca_probability: float = 0.5
    last_innovation: float = 0.0
    association: Dict[str, Any] = field(default_factory=dict)
    points: Deque[Tuple[int, float, float]] = field(default_factory=deque)

    @property
    def center_x(self) -> float:
        return float(self.state[0])

    @property
    def center_y(self) -> float:
        return float(self.state[1])

    @property
    def velocity_x(self) -> float:
        return float(self.state[2])

    @property
    def velocity_y(self) -> float:
        return float(self.state[3])

    @property
    def base_radius(self) -> float:
        return float(max(math.exp(self.state[6]), math.exp(self.state[7])) / 2.0)

    @property
    def missing_frames(self) -> int:
        return int(self.misses)

    @property
    def last_seen_frame_index(self) -> int:
        return int(self.last_frame_index)


@dataclass(frozen=True)
class HybridTrackSnapshot:
    """Immutable overlay state captured at one processed frame.

    Delayed consumers must use these snapshots instead of ``tracker.tracks``:
    the latter continues advancing while an older video frame waits to commit.
    """

    track_id: int
    center_x: float
    center_y: float
    velocity_x: float
    velocity_y: float
    base_radius: float
    missing_frames: int
    hits: int
    first_frame_index: int
    last_seen_frame_index: int
    points: Tuple[Tuple[int, float, float], ...]
    status: str
    observed_this_frame: bool
    seconds_since_observed: float
    last_timestamp: float
    last_observed_timestamp: float


def _bbox_values(row: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    bbox = list(row.get("bbox", (0.0, 0.0, 0.0, 0.0)))
    if len(bbox) < 4:
        bbox += [0.0] * (4 - len(bbox))
    x, y, width, height = (float(value) for value in bbox[:4])
    return x, y, max(width, 1.0e-3), max(height, 1.0e-3)


def _recheck_passed(row: Mapping[str, Any]) -> bool:
    for key in ("recheck_passed", "recheck_pass"):
        if key in row:
            return bool(row[key])
    nested = row.get("recheck")
    if isinstance(nested, Mapping) and "passed" in nested:
        return bool(nested["passed"])
    return False


class _CameraMotionEstimator:
    """Sparse-flow affine estimation with ORB fallback and explicit diagnostics."""

    def __init__(self, cfg: HybridTrackingConfig) -> None:
        self.cfg = cfg
        self.previous_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.previous_gray = None

    def estimate(self, frame: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        if not self.cfg.cmc_enabled or frame is None:
            return identity, {"method": "identity", "success": False, "reason": "disabled_or_no_frame", "inliers": 0}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if getattr(frame, "ndim", 0) == 3 else np.asarray(frame)
        processing_scale = float(self.cfg.cmc_processing_scale)
        if processing_scale < 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=processing_scale,
                fy=processing_scale,
                interpolation=cv2.INTER_AREA,
            )
        if self.previous_gray is None:
            self.previous_gray = gray.copy()
            return identity, {"method": "identity", "success": False, "reason": "first_frame", "inliers": 0}
        previous = self.previous_gray
        self.previous_gray = gray.copy()
        affine, inliers = self._sparse_flow(previous, gray)
        method = "sparse_optical_flow"
        reason = None
        if affine is None:
            affine, inliers = self._orb(previous, gray)
            method = "orb"
        if affine is None:
            return identity, {"method": "identity", "success": False, "reason": "estimation_failed", "inliers": 0}
        if processing_scale < 1.0:
            # A uniform resize preserves the affine linear terms. Translation
            # is measured in the reduced pixel coordinate system and must be
            # mapped back before it is applied to full-resolution tracks.
            affine = np.asarray(affine, dtype=np.float64).copy()
            affine[:, 2] /= processing_scale
        valid, reason = self._validate(affine)
        if not valid:
            return identity, {
                "method": "identity", "success": False, "reason": reason,
                "inliers": int(inliers), "processing_scale": processing_scale,
            }
        return affine.astype(np.float64), {
            "method": method, "success": True, "reason": None,
            "inliers": int(inliers), "processing_scale": processing_scale,
        }

    def _sparse_flow(self, previous: np.ndarray, current: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=self.cfg.cmc_max_corners,
            qualityLevel=self.cfg.cmc_quality_level,
            minDistance=max(1.0, self.cfg.cmc_min_distance * self.cfg.cmc_processing_scale),
        )
        if points is None or len(points) < self.cfg.cmc_min_inliers:
            return None, 0
        next_points, status, _error = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
        if next_points is None or status is None:
            return None, 0
        good = status.reshape(-1).astype(bool)
        source = points.reshape(-1, 2)[good]
        target = next_points.reshape(-1, 2)[good]
        if len(source) < self.cfg.cmc_min_inliers:
            return None, 0
        affine, mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.cfg.cmc_ransac_threshold * self.cfg.cmc_processing_scale,
        )
        inliers = int(mask.sum()) if mask is not None else 0
        if affine is None or inliers < self.cfg.cmc_min_inliers:
            return None, inliers
        return affine, inliers

    def _orb(self, previous: np.ndarray, current: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        orb = cv2.ORB_create(nfeatures=self.cfg.cmc_max_corners)
        key_a, desc_a = orb.detectAndCompute(previous, None)
        key_b, desc_b = orb.detectAndCompute(current, None)
        if desc_a is None or desc_b is None:
            return None, 0
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
        matches = [first for pair in pairs if len(pair) == 2 for first, second in [pair] if first.distance < 0.75 * second.distance]
        if len(matches) < self.cfg.cmc_min_inliers:
            return None, 0
        source = np.float32([key_a[match.queryIdx].pt for match in matches])
        target = np.float32([key_b[match.trainIdx].pt for match in matches])
        affine, mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.cfg.cmc_ransac_threshold * self.cfg.cmc_processing_scale,
        )
        inliers = int(mask.sum()) if mask is not None else 0
        if affine is None or inliers < self.cfg.cmc_min_inliers:
            return None, inliers
        return affine, inliers

    def _validate(self, affine: np.ndarray) -> Tuple[bool, Optional[str]]:
        scale = math.sqrt(float(affine[0, 0]) ** 2 + float(affine[1, 0]) ** 2)
        translation = math.hypot(float(affine[0, 2]), float(affine[1, 2]))
        if not self.cfg.cmc_min_scale <= scale <= self.cfg.cmc_max_scale:
            return False, "scale_out_of_range"
        if translation > self.cfg.cmc_max_translation:
            return False, "translation_out_of_range"
        return True, None


class HybridFootballTracker:
    """All-ball, motion-only tracker with two confidence tiers and delayed commits."""

    def __init__(
        self,
        cfg: HybridTrackingConfig,
        target_class_ids: Set[int],
        history_maxlen: int = 30,
        confirmation_backfill: bool = False,
    ) -> None:
        self.cfg = cfg
        self.target_class_ids = {int(value) for value in target_class_ids}
        self.tracks: List[_Track] = []
        self._history_maxlen = max(1, int(history_maxlen))
        self._confirmation_backfill = bool(confirmation_backfill)
        self.next_id = 1
        self._pending: Deque[Tuple[int, Dict[str, Any]]] = deque()
        self._latest: Optional[Dict[str, Any]] = None
        self._last_frame_index: Optional[int] = None
        self._last_timestamp: Optional[float] = None
        self._processed_count = 0
        self._confirmed_track_ids: Set[int] = set()
        self._cmc = _CameraMotionEstimator(cfg)

    def reset(self) -> None:
        """Reset segment-local identity and temporal state."""
        self.tracks.clear()
        self.next_id = 1
        self._pending.clear()
        self._latest = None
        self._last_frame_index = None
        self._last_timestamp = None
        self._processed_count = 0
        self._confirmed_track_ids.clear()
        self._cmc.reset()

    def step(
        self,
        frame_index: int,
        timestamp: float,
        frame: Any,
        detections: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Consume one ordered frame and return frames whose lookahead has elapsed."""
        frame_index = int(frame_index)
        timestamp = float(timestamp)
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("hybrid tracker frame_index must increase within a segment")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("hybrid tracker timestamp must be monotonic within a segment")
        affine, cmc_diagnostic = self._cmc.estimate(frame)
        self._apply_camera_motion(affine)
        normalized = self._normalize(detections)
        rows, states, snapshots = self._process(frame_index, timestamp, detections, normalized)
        frame_result = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "detections": rows,
                "track_states": states,
                "track_snapshots": snapshots,
                "cmc": cmc_diagnostic,
                "cmc_affine": [
                    [float(value) for value in row]
                    for row in np.asarray(affine, dtype=np.float64).reshape(2, 3)
                ],
        }
        self._latest = frame_result
        sequence_index = self._processed_count
        self._processed_count += 1
        self._pending.append((sequence_index, frame_result))
        self._last_frame_index = frame_index
        self._last_timestamp = timestamp
        committed: List[Dict[str, Any]] = []
        epsilon = 1.0e-9
        confirmation_delay = self.cfg.confirmation_window - 1 if self._confirmation_backfill else 0
        while self._pending:
            pending_sequence, pending_frame = self._pending[0]
            time_ready = timestamp - float(pending_frame["timestamp"]) + epsilon >= self.cfg.lookahead_seconds
            confirmation_ready = sequence_index - pending_sequence >= confirmation_delay
            if not (time_ready and confirmation_ready):
                break
            self._pending.popleft()
            committed.append(self._finalize_packet(pending_frame))
        return committed

    def flush(self) -> List[Dict[str, Any]]:
        """Commit all delayed frames exactly once, preserving input order."""
        remaining = [self._finalize_packet(frame) for _sequence, frame in self._pending]
        self._pending.clear()
        return remaining

    def update(
        self, frame_index: int, predictions: Sequence[Mapping[str, Any]], frame: Any = None
    ) -> List[Dict[str, Any]]:
        """Immediate compatibility adapter for legacy video replay callers."""
        timestamp = float(frame_index) / 30.0
        previous_delay = self.cfg.lookahead_seconds
        if previous_delay > 0.0:
            raise RuntimeError("hybrid tracker integration must use step()/flush() to preserve fixed-delay output")
        committed = self.step(frame_index, timestamp, frame, predictions)
        return committed[0]["detections"] if committed else []

    def latest_frame(self) -> Optional[Dict[str, Any]]:
        """Return the newest provisional frame for rendering; exports must use committed frames."""
        return self._latest


    @property
    def confirmed_track_ids(self) -> frozenset[int]:
        """All IDs that ever reached confirmed status in the current source segment."""
        return frozenset(self._confirmed_track_ids)


    def confirmed_tracks(self) -> List[_Track]:
        return [track for track in self.tracks if track.status == "confirmed"]

    def _normalize(self, rows: Sequence[Mapping[str, Any]]) -> List[_Detection]:
        normalized: List[_Detection] = []
        for index, row in enumerate(rows):
            if int(row.get("category_id", -1)) not in self.target_class_ids:
                continue
            score = float(row.get("score", 0.0))
            if score < self.cfg.low_confidence:
                continue
            x, y, width, height = _bbox_values(row)
            normalized.append(
                _Detection(index, row, x + width / 2.0, y + height / 2.0, width, height, math.log(width), math.log(height), score)
            )
        return normalized

    def _apply_camera_motion(self, affine: np.ndarray) -> None:
        linear = affine[:, :2]
        for track in self.tracks:
            center = affine @ np.asarray([track.state[0], track.state[1], 1.0])
            velocity = linear @ track.state[2:4]
            acceleration = linear @ track.state[4:6]
            track.state[0:2] = center
            track.last_observed_center = affine @ np.asarray(
                [track.last_observed_center[0], track.last_observed_center[1], 1.0]
            )
            track.state[2:4] = velocity
            track.state[4:6] = acceleration
            jacobian = np.eye(8, dtype=np.float64)
            jacobian[0:2, 0:2] = linear
            jacobian[2:4, 2:4] = linear
            jacobian[4:6, 4:6] = linear
            track.covariance = jacobian @ track.covariance @ jacobian.T

    def _predict(self, track: _Track, timestamp: float) -> None:
        dt = max(1.0e-3, timestamp - track.last_timestamp)
        cv_center = track.state[0:2] + track.state[2:4] * dt
        ca_center = cv_center + 0.5 * track.state[4:6] * dt * dt
        track.state[0:2] = track.cv_probability * cv_center + track.ca_probability * ca_center
        track.state[2:4] += track.ca_probability * track.state[4:6] * dt
        transition = np.eye(8, dtype=np.float64)
        transition[0, 2] = transition[1, 3] = dt
        transition[0, 4] = transition[1, 5] = 0.5 * dt * dt
        transition[2, 4] = transition[3, 5] = dt
        noise = np.diag([dt * dt, dt * dt, dt, dt, 1.0, 1.0, 0.05, 0.05]) * self.cfg.process_noise
        track.covariance = transition @ track.covariance @ transition.T + noise
        track.last_timestamp = timestamp
        track.age_frames += 1

    def _process(
        self,
        frame_index: int,
        timestamp: float,
        original: Sequence[Mapping[str, Any]],
        detections: Sequence[_Detection],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Tuple[HybridTrackSnapshot, ...]]:
        for track in self.tracks:
            self._predict(track, timestamp)
            track.association = {"stage": "miss", "cost": None, "detection_index": None}

        high = [detection for detection in detections if detection.score >= self.cfg.high_confidence]
        low = [
            detection for detection in detections
            if self.cfg.low_confidence <= detection.score < self.cfg.high_confidence
            and (not self.cfg.low_require_recheck or _recheck_passed(detection.row))
        ]
        assigned: Dict[int, _Track] = {}
        active = [track for track in self.tracks if track.status != "retired"]
        first_matches, unmatched_tracks, unmatched_high = self._associate(active, high, self.cfg.high_gate_pixels, "high")
        for track, detection, cost in first_matches:
            self._correct(track, detection, frame_index, timestamp, "high", cost)
            assigned[detection.index] = track

        eligible_low = [track for track in unmatched_tracks if track.status in {"confirmed", "lost"}]
        low_matches, _unmatched_low_tracks, _unmatched_low = self._associate(
            eligible_low, low, self.cfg.low_gate_pixels * self.cfg.low_gate_scale, "low"
        )
        matched_low_ids = set()
        for track, detection, cost in low_matches:
            self._correct(track, detection, frame_index, timestamp, "low", cost)
            assigned[detection.index] = track
            matched_low_ids.add(track.track_id)

        matched_track_ids = {track.track_id for track in assigned.values()}
        for track in active:
            if track.track_id not in matched_track_ids:
                self._miss(track, timestamp)

        for detection in unmatched_high:
            if detection.score < self.cfg.new_track_confidence:
                continue
            track = self._spawn(detection, frame_index, timestamp)
            assigned[detection.index] = track

        self.tracks = [track for track in self.tracks if track.status != "retired"]
        rows: List[Dict[str, Any]] = []
        for index, original_row in enumerate(original):
            row = dict(original_row)
            track = assigned.get(index)
            if track is None:
                row.update(
                    track_id=None, track_center_x=None, track_center_y=None, track_radius_pixels=None,
                    track_first_frame_index=None, track_last_seen_frame_index=None, track_age_frames=None,
                    track_hits=None, track_confirmed=None, track_status=None, association_stage=None,
                )
            else:
                row.update(
                    track_id=track.track_id,
                    track_center_x=float(track.state[0]),
                    track_center_y=float(track.state[1]),
                    track_radius_pixels=float(max(math.exp(track.state[6]), math.exp(track.state[7])) / 2.0),
                    track_first_frame_index=track.first_frame_index,
                    track_last_seen_frame_index=track.last_frame_index,
                    track_age_frames=track.age_frames,
                    track_hits=track.hits,
                    track_confirmed=track.status == "confirmed",
                    track_status=track.status,
                    association_stage=track.association.get("stage"),
                )
            rows.append(row)
        states = [
            self._state_row(track, timestamp, observed=track.track_id in matched_track_ids or track.last_frame_index == frame_index)
            for track in self.tracks
            if track.status in {"tentative", "confirmed", "lost"}
            and (
                track.last_frame_index == frame_index
                or (track.status in {"confirmed", "lost"} and timestamp - track.last_observed_timestamp <= self.cfg.predicted_output_seconds)
            )
        ]
        states.sort(key=lambda row: row["track_id"])
        snapshots = tuple(
            self._snapshot_track(
                track,
                timestamp,
                observed=track.track_id in matched_track_ids or track.last_frame_index == frame_index,
            )
            for track in sorted(self.tracks, key=lambda item: item.track_id)
            if track.status in {"tentative", "confirmed", "lost"}
        )
        return rows, states, snapshots

    def _associate(
        self, tracks: Sequence[_Track], detections: Sequence[_Detection], gate_pixels: float, stage: str
    ) -> Tuple[List[Tuple[_Track, _Detection, float]], List[_Track], List[_Detection]]:
        if not tracks or not detections:
            return [], list(tracks), list(detections)
        costs = np.full((len(tracks), len(detections)), _INVALID_COST, dtype=np.float64)
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                costs[track_index, detection_index] = self._cost(track, detection, gate_pixels)
        row_indices, column_indices = linear_sum_assignment(costs)
        matches: List[Tuple[_Track, _Detection, float]] = []
        used_tracks: Set[int] = set()
        used_detections: Set[int] = set()
        for track_index, detection_index in zip(row_indices.tolist(), column_indices.tolist()):
            cost = float(costs[track_index, detection_index])
            if cost >= _INVALID_COST:
                continue
            row_candidates = sorted(value for value in costs[track_index, :] if value < _INVALID_COST)
            column_candidates = sorted(value for value in costs[:, detection_index] if value < _INVALID_COST)
            row_ambiguous = len(row_candidates) > 1 and row_candidates[1] - row_candidates[0] <= self.cfg.ambiguity_margin
            column_ambiguous = len(column_candidates) > 1 and column_candidates[1] - column_candidates[0] <= self.cfg.ambiguity_margin
            if row_ambiguous or column_ambiguous:
                continue
            track = tracks[track_index]
            detection = detections[detection_index]
            matches.append((track, detection, cost))
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        return (
            matches,
            [track for index, track in enumerate(tracks) if index not in used_tracks],
            [detection for index, detection in enumerate(detections) if index not in used_detections],
        )

    def _cost(self, track: _Track, detection: _Detection, gate_pixels: float) -> float:
        measured = np.asarray([detection.cx, detection.cy])
        predicted_residual = measured - track.state[0:2]
        observed_residual = measured - track.last_observed_center
        # OC-SORT union gate: rapid motion uses the prediction, while sudden stops or duplicate
        # detections can still match the latest camera-compensated observation.
        residual = min((predicted_residual, observed_residual), key=lambda value: float(np.linalg.norm(value)))
        distance = float(np.linalg.norm(residual))
        if distance > gate_pixels:
            return _INVALID_COST
        innovation_covariance = track.covariance[0:2, 0:2] + np.eye(2) * self.cfg.measurement_noise
        try:
            mahalanobis = math.sqrt(max(0.0, float(residual.T @ np.linalg.inv(innovation_covariance) @ residual)))
        except np.linalg.LinAlgError:
            return _INVALID_COST
        if mahalanobis > self.cfg.mahalanobis_gate:
            return _INVALID_COST
        speed = float(np.linalg.norm(track.state[2:4]))
        if speed > 1.0e-6 and distance > 1.0e-6:
            cosine = float(np.dot(track.state[2:4], residual) / (speed * distance))
            direction_cost = (1.0 - max(-1.0, min(1.0, cosine))) / 2.0
        else:
            direction_cost = 0.5
        size_cost = min(1.0, (abs(detection.log_width - track.state[6]) + abs(detection.log_height - track.state[7])) / 2.0)
        return (
            self.cfg.weight_mahalanobis * min(1.0, mahalanobis / self.cfg.mahalanobis_gate)
            + self.cfg.weight_center * min(1.0, distance / gate_pixels)
            + self.cfg.weight_direction * direction_cost
            + self.cfg.weight_size * size_cost
            + self.cfg.weight_confidence * (1.0 - detection.score)
        )

    def _spawn(self, detection: _Detection, frame_index: int, timestamp: float) -> _Track:
        state = np.asarray(
            [detection.cx, detection.cy, 0.0, 0.0, 0.0, 0.0, detection.log_width, detection.log_height], dtype=np.float64
        )
        covariance = np.diag([64.0, 64.0, 400.0, 400.0, 900.0, 900.0, 0.25, 0.25])
        status = "confirmed" if self.cfg.confirmed_hits <= 1 else "tentative"
        track = _Track(
            track_id=self.next_id,
            state=state,
            covariance=covariance,
            first_frame_index=frame_index,
            last_frame_index=frame_index,
            first_timestamp=timestamp,
            last_timestamp=timestamp,
            last_observed_timestamp=timestamp,
            last_observed_center=np.asarray([detection.cx, detection.cy], dtype=np.float64),
            status=status,
            association={"stage": "new", "cost": 0.0, "detection_index": detection.index},
            points=deque([(frame_index, detection.cx, detection.cy)], maxlen=self._history_maxlen),
        )
        self.next_id += 1
        self.tracks.append(track)
        if status == "confirmed":
            self._confirmed_track_ids.add(track.track_id)
        return track

    def _correct(
        self, track: _Track, detection: _Detection, frame_index: int, timestamp: float, stage: str, cost: float
    ) -> None:
        predicted_center = track.state[0:2].copy()
        measured_center = np.asarray([detection.cx, detection.cy], dtype=np.float64)
        residual = measured_center - predicted_center
        observation_dt = max(1.0e-3, timestamp - track.last_observed_timestamp)
        observed_velocity = (measured_center - track.last_observed_center) / observation_dt
        old_velocity = track.state[2:4].copy()
        observed_acceleration = (observed_velocity - old_velocity) / observation_dt
        innovation_covariance = track.covariance[0:2, 0:2] + np.eye(2) * self.cfg.measurement_noise
        gain = track.covariance[:, 0:2] @ np.linalg.inv(innovation_covariance)
        track.state += gain @ residual
        track.covariance = (np.eye(8) - gain @ np.eye(8)[0:2, :]) @ track.covariance
        track.state[0:2] = measured_center
        track.state[2:4] = 0.35 * old_velocity + 0.65 * observed_velocity
        smoothing = self.cfg.acceleration_smoothing
        track.state[4:6] = smoothing * track.state[4:6] + (1.0 - smoothing) * observed_acceleration
        track.state[6] = self.cfg.size_smoothing * track.state[6] + (1.0 - self.cfg.size_smoothing) * detection.log_width
        track.state[7] = self.cfg.size_smoothing * track.state[7] + (1.0 - self.cfg.size_smoothing) * detection.log_height
        cv_error = float(np.linalg.norm(residual))
        ca_prediction = predicted_center + 0.5 * track.state[4:6] * observation_dt * observation_dt
        ca_error = float(np.linalg.norm(measured_center - ca_prediction))
        cv_likelihood = math.exp(-cv_error / max(1.0, self.cfg.high_gate_pixels))
        ca_likelihood = math.exp(-ca_error / max(1.0, self.cfg.high_gate_pixels))
        total = max(1.0e-9, cv_likelihood + ca_likelihood)
        target_cv = cv_likelihood / total
        probability_smoothing = self.cfg.model_probability_smoothing
        track.cv_probability = probability_smoothing * track.cv_probability + (1.0 - probability_smoothing) * target_cv
        track.ca_probability = 1.0 - track.cv_probability
        track.last_innovation = cv_error
        track.last_observed_center = measured_center
        track.last_observed_timestamp = timestamp
        track.last_frame_index = frame_index
        track.last_timestamp = timestamp
        track.hits += 1
        track.misses = 0
        if track.status in {"lost", "ambiguous"}:
            track.status = "confirmed"
        elif track.status == "tentative" and track.hits >= self.cfg.confirmed_hits and track.age_frames <= self.cfg.confirmation_window:
            track.status = "confirmed"
        if track.status == "confirmed":
            self._confirmed_track_ids.add(track.track_id)
        track.association = {"stage": stage, "cost": cost, "detection_index": detection.index}
        track.points.append((frame_index, detection.cx, detection.cy))

    def _miss(self, track: _Track, timestamp: float) -> None:
        track.misses += 1
        elapsed = timestamp - track.last_observed_timestamp
        if track.status == "tentative" and track.age_frames >= self.cfg.confirmation_window:
            track.status = "retired"
        elif elapsed > self.cfg.lost_seconds:
            track.status = "retired"
        elif track.status == "confirmed":
            track.status = "lost"

    def _state_row(self, track: _Track, timestamp: float, observed: bool) -> Dict[str, Any]:
        width, height = math.exp(track.state[6]), math.exp(track.state[7])
        return {
            "track_id": track.track_id,
            "status": track.status,
            "observation": "observed" if observed else "predicted",
            "bbox": [float(track.state[0] - width / 2.0), float(track.state[1] - height / 2.0), float(width), float(height)],
            "center": [float(track.state[0]), float(track.state[1])],
            "velocity": [float(track.state[2]), float(track.state[3])],
            "acceleration": [float(track.state[4]), float(track.state[5])],
            "covariance_diagonal": [float(value) for value in np.diag(track.covariance)],
            "motion_model_probabilities": {
                "constant_velocity": float(track.cv_probability),
                "constant_acceleration": float(track.ca_probability),
            },
            "association": dict(track.association),
            "seconds_since_observed": float(timestamp - track.last_observed_timestamp),
            "hits": track.hits,
            "age_frames": track.age_frames,
        }

    def _snapshot_track(self, track: _Track, timestamp: float, observed: bool) -> HybridTrackSnapshot:
        return HybridTrackSnapshot(
            track_id=int(track.track_id),
            center_x=float(track.center_x),
            center_y=float(track.center_y),
            velocity_x=float(track.velocity_x),
            velocity_y=float(track.velocity_y),
            base_radius=float(track.base_radius),
            missing_frames=int(track.missing_frames),
            hits=int(track.hits),
            first_frame_index=int(track.first_frame_index),
            last_seen_frame_index=int(track.last_seen_frame_index),
            points=tuple((int(index), float(x), float(y)) for index, x, y in track.points),
            status=str(track.status),
            observed_this_frame=bool(observed),
            seconds_since_observed=float(timestamp - track.last_observed_timestamp),
            last_timestamp=float(timestamp),
            last_observed_timestamp=float(track.last_observed_timestamp),
        )

    def _finalize_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """Attach source-final confirmation knowledge available at commit time.

        Per-frame detection lifecycle fields remain raw.  Render/export policy
        consumes the separate final-ID registry so an unfiltered JSON export
        still records that the first hit was tentative.
        """
        confirmed_ids = frozenset(self._confirmed_track_ids)
        packet["confirmed_track_ids"] = confirmed_ids
        return packet
