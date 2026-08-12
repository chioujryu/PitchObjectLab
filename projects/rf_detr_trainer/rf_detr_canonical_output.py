"""Canonical V2 video inference output for RF-DETR.

This dependency-light module owns the public, offline-friendly video export
contract.  It deliberately does not import the RF-DETR model or a tracker: the
inference pipelines stream their committed frame packets into
``CanonicalVideoWriter`` while existing legacy JSONL rows remain untouched.

The public contract is versioned independently from the trainer.  Pixel boxes
use continuous ``[x1, y1, x2, y2]`` edges (``width = x2 - x1``); raw pixel
coordinates are retained, while normalized box and center coordinates are
clamped to the image.  Motion is recomputed from real frame timestamps rather
than copied from a tracker-specific state vector.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "2.0.0"

DEFAULT_CANONICAL_OUTPUT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "directory": "canonical_v2",
    "ffmpeg_path": "auto",
    "ffprobe_path": "auto",
    "media": {
        "video_codec": "libx264",
        "crf": 18,
        "preset": "medium",
        "pixel_format": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
    },
}

_MOTION_FIELDS: Tuple[str, ...] = (
    "delta_time_seconds",
    "frame_gap",
    "velocity_pixels_per_second",
    "velocity_normalized_per_second",
    "speed_pixels_per_second",
    "speed_normalized_per_second",
    "direction_clockwise_degrees",
    "direction_8way",
    "acceleration_pixels_per_second_squared",
    "acceleration_normalized_per_second_squared",
    "acceleration_magnitude_pixels_per_second_squared",
    "acceleration_magnitude_normalized_per_second_squared",
    "bbox_log_area_change_per_second",
)

_GEOMETRY_FIELDS: Tuple[str, ...] = (
    "bbox_xyxy_pixels",
    "bbox_xyxy_normalized",
    "center_pixels",
    "center_normalized",
    "area_pixels",
    "area_normalized",
    "was_clipped",
    "in_frame",
)

_DETECTION_RESERVED_FIELDS = {
    "detection_index",
    "category_id",
    "category_name",
    "score",
    *_GEOMETRY_FIELDS,
}

_TRACK_RESERVED_FIELDS = {
    "track_id",
    "provenance",
    "status",
    "hits",
    "age_frames",
    "seconds_since_observed",
    "detection_index",
    "category_id",
    "category_name",
    "motion",
    *_GEOMETRY_FIELDS,
}


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_CANONICAL_OUTPUT_CONFIG",
    "CanonicalMediaConfig",
    "CanonicalOutputConfig",
    "CanonicalToolchain",
    "VideoSelection",
    "parse_canonical_output_config",
    "add_canonical_output_cli_arguments",
    "apply_canonical_output_cli_overrides",
    "preflight",
    "xywh_to_xyxy",
    "geometry_from_xyxy",
    "build_detection_row",
    "build_detection_rows_from_legacy",
    "build_track_state_row",
    "build_observed_track_states",
    "build_tracker_state_rows",
    "normalize_camera_motion",
    "tracker_capabilities",
    "TrajectoryAccumulator",
    "make_video_id",
    "probe_media",
    "transcode_clean_media",
    "atomic_write_json",
    "CanonicalRunWriter",
    "CanonicalVideoWriter",
]


@dataclass(frozen=True)
class CanonicalMediaConfig:
    """Validated clean-media encoding settings."""

    video_codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"


@dataclass(frozen=True)
class CanonicalOutputConfig:
    """Validated ``inference.canonical_output`` settings."""

    enabled: bool = True
    directory: str = "canonical_v2"
    ffmpeg_path: str = "auto"
    ffprobe_path: str = "auto"
    media: CanonicalMediaConfig = field(default_factory=CanonicalMediaConfig)


@dataclass(frozen=True)
class CanonicalToolchain:
    """Resolved and preflighted media tool paths."""

    ffmpeg_path: Optional[str]
    ffprobe_path: Optional[str]
    video_encoder: Optional[str]
    audio_encoder: Optional[str]


@dataclass(frozen=True)
class VideoSelection:
    """Selected source range; frame end is always exclusive."""

    start_frame: int = 0
    end_frame_exclusive: Optional[int] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("selected start_frame must be non-negative")
        if self.end_frame_exclusive is not None and self.end_frame_exclusive <= self.start_frame:
            raise ValueError("selected end_frame_exclusive must be greater than start_frame")
        if self.start_seconds is not None and self.start_seconds < 0.0:
            raise ValueError("selected start_seconds must be non-negative")
        if self.start_seconds is not None and self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("selected end_seconds must be greater than start_seconds")

    @property
    def requested_frame_count(self) -> Optional[int]:
        if self.end_frame_exclusive is None:
            return None
        return self.end_frame_exclusive - self.start_frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


def _as_bool(value: Any, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "yes", "on", "1"}:
        return True
    if text in {"false", "no", "off", "0"}:
        return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}")


def _nonempty_text(value: Any, default: str, field_name: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _canonical_config_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "inference" in config:
        inference = config.get("inference", {}) or {}
        if not isinstance(inference, Mapping):
            raise ValueError("inference must be a mapping")
        raw = inference.get("canonical_output", {}) or {}
    elif "canonical_output" in config:
        raw = config.get("canonical_output", {}) or {}
    elif set(config).intersection(DEFAULT_CANONICAL_OUTPUT_CONFIG):
        raw = config
    else:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("inference.canonical_output must be a mapping")
    return raw


def parse_canonical_output_config(config: Mapping[str, Any]) -> CanonicalOutputConfig:
    """Parse the full run config (or the canonical sub-block) without mutation."""

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    raw = _canonical_config_mapping(config)
    unknown = set(raw) - set(DEFAULT_CANONICAL_OUTPUT_CONFIG)
    if unknown:
        raise ValueError(f"unknown inference.canonical_output.{sorted(unknown)[0]}")
    media_raw = raw.get("media", {}) or {}
    if not isinstance(media_raw, Mapping):
        raise ValueError("inference.canonical_output.media must be a mapping")
    unknown_media = set(media_raw) - set(DEFAULT_CANONICAL_OUTPUT_CONFIG["media"])
    if unknown_media:
        raise ValueError(f"unknown inference.canonical_output.media.{sorted(unknown_media)[0]}")
    defaults = DEFAULT_CANONICAL_OUTPUT_CONFIG["media"]
    try:
        crf = int(media_raw.get("crf", defaults["crf"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("inference.canonical_output.media.crf must be an integer") from exc
    if not 0 <= crf <= 51:
        raise ValueError("inference.canonical_output.media.crf must be between 0 and 51")
    media = CanonicalMediaConfig(
        video_codec=_nonempty_text(
            media_raw.get("video_codec"),
            defaults["video_codec"],
            "inference.canonical_output.media.video_codec",
        ),
        crf=crf,
        preset=_nonempty_text(
            media_raw.get("preset"),
            defaults["preset"],
            "inference.canonical_output.media.preset",
        ),
        pixel_format=_nonempty_text(
            media_raw.get("pixel_format"),
            defaults["pixel_format"],
            "inference.canonical_output.media.pixel_format",
        ),
        audio_codec=_nonempty_text(
            media_raw.get("audio_codec"),
            defaults["audio_codec"],
            "inference.canonical_output.media.audio_codec",
        ),
        audio_bitrate=_nonempty_text(
            media_raw.get("audio_bitrate"),
            defaults["audio_bitrate"],
            "inference.canonical_output.media.audio_bitrate",
        ),
    )
    directory = _nonempty_text(
        raw.get("directory"),
        "canonical_v2",
        "inference.canonical_output.directory",
    )
    directory_path = Path(directory)
    windows_path = PureWindowsPath(directory)
    if (
        directory_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in directory_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(
            "inference.canonical_output.directory must be a relative path inside the inference output directory"
        )
    return CanonicalOutputConfig(
        enabled=_as_bool(raw.get("enabled"), True, "inference.canonical_output.enabled"),
        directory=directory,
        ffmpeg_path=_nonempty_text(
            raw.get("ffmpeg_path"),
            "auto",
            "inference.canonical_output.ffmpeg_path",
        ),
        ffprobe_path=_nonempty_text(
            raw.get("ffprobe_path"),
            "auto",
            "inference.canonical_output.ffprobe_path",
        ),
        media=media,
    )


def add_canonical_output_cli_arguments(parser: Any) -> None:
    """Register Canonical V2 overrides on an ``argparse`` parser."""

    toggle = parser.add_mutually_exclusive_group()
    toggle.add_argument(
        "--canonical-output",
        dest="canonical_output",
        action="store_true",
        default=None,
        help="Enable Canonical V2 video bundles (default from YAML, normally enabled).",
    )
    toggle.add_argument(
        "--no-canonical-output",
        dest="canonical_output",
        action="store_false",
        help="Disable Canonical V2 video bundles and clean media.",
    )
    parser.add_argument(
        "--canonical-output-dir",
        default=None,
        help="Override inference.canonical_output.directory.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="Override the Canonical V2 FFmpeg executable (or 'auto').",
    )
    parser.add_argument(
        "--ffprobe-path",
        default=None,
        help="Override the Canonical V2 FFprobe executable (or 'auto').",
    )


def apply_canonical_output_cli_overrides(config: CanonicalOutputConfig, args: Any) -> CanonicalOutputConfig:
    """Return a config dataclass with non-``None`` CLI overrides applied."""

    updates: Dict[str, Any] = {}
    enabled = getattr(args, "canonical_output", None)
    if enabled is not None:
        updates["enabled"] = bool(enabled)
    directory = getattr(args, "canonical_output_dir", None)
    if directory is not None:
        updates["directory"] = _nonempty_text(directory, config.directory, "--canonical-output-dir")
    ffmpeg_path = getattr(args, "ffmpeg_path", None)
    if ffmpeg_path is not None:
        updates["ffmpeg_path"] = _nonempty_text(ffmpeg_path, config.ffmpeg_path, "--ffmpeg-path")
    ffprobe_path = getattr(args, "ffprobe_path", None)
    if ffprobe_path is not None:
        updates["ffprobe_path"] = _nonempty_text(ffprobe_path, config.ffprobe_path, "--ffprobe-path")
    return replace(config, **updates)


def _run_process(runner: Callable[..., Any], command: Sequence[str]) -> Any:
    return runner(
        [str(part) for part in command],
        capture_output=True,
        text=True,
        check=False,
    )


def _resolve_executable(value: str, default_name: str, *, required: bool) -> Optional[str]:
    text = str(value).strip()
    automatic = text.casefold() == "auto"
    candidate = default_name if automatic else text
    resolved = shutil.which(candidate)
    if resolved:
        return str(Path(resolved).resolve())
    explicit_path = Path(candidate).expanduser()
    if not automatic and explicit_path.is_file():
        return str(explicit_path.resolve())
    if required:
        field = "ffmpeg_path" if default_name == "ffmpeg" else "ffprobe_path"
        raise RuntimeError(f"Canonical V2 requires {default_name}; {field}={value!r} was not executable.")
    return None


def preflight(
    config: CanonicalOutputConfig | Mapping[str, Any],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> CanonicalToolchain:
    """Resolve FFmpeg/FFprobe and verify configured H.264/AAC encoders.

    Call this after source discovery confirms at least one video but before a
    bundle directory is created.  A disabled config performs no tool lookup.
    """

    cfg = config if isinstance(config, CanonicalOutputConfig) else parse_canonical_output_config(config)
    if not cfg.enabled:
        return CanonicalToolchain(None, None, None, None)
    ffmpeg = _resolve_executable(cfg.ffmpeg_path, "ffmpeg", required=True)
    ffprobe = _resolve_executable(
        cfg.ffprobe_path,
        "ffprobe",
        required=cfg.ffprobe_path.strip().casefold() != "auto",
    )
    assert ffmpeg is not None
    result = _run_process(runner, [ffmpeg, "-hide_banner", "-encoders"])
    if int(getattr(result, "returncode", 1)) != 0:
        detail = (getattr(result, "stderr", "") or getattr(result, "stdout", "")).strip()
        raise RuntimeError(f"FFmpeg encoder preflight failed: {detail or 'unknown error'}")
    encoders_text = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
    missing = [
        encoder
        for encoder in (cfg.media.video_codec, cfg.media.audio_codec)
        if re.search(rf"(?<![\w-]){re.escape(encoder)}(?![\w-])", encoders_text) is None
    ]
    if missing:
        raise RuntimeError("FFmpeg is missing Canonical V2 encoder(s): " + ", ".join(missing))
    return CanonicalToolchain(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        video_encoder=cfg.media.video_codec,
        audio_encoder=cfg.media.audio_codec,
    )


def _finite_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return parsed


def _optional_finite_float(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    return _finite_float(value, field_name)


def _positive_dimension(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {value!r}")
    return parsed


def _four_floats(values: Sequence[Any], field_name: str) -> List[float]:
    if values is None or len(values) < 4:
        raise ValueError(f"{field_name} must contain four values, got {values!r}")
    return [_finite_float(values[index], f"{field_name}[{index}]") for index in range(4)]


def xywh_to_xyxy(bbox_xywh: Sequence[Any]) -> List[float]:
    """Convert continuous top-left COCO xywh to continuous xyxy."""

    x, y, width, height = _four_floats(bbox_xywh, "bbox_xywh")
    if width < 0.0 or height < 0.0:
        raise ValueError("bbox_xywh width and height must be non-negative")
    return [x, y, x + width, y + height]


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def geometry_from_xyxy(bbox_xyxy: Sequence[Any], frame_width: Any, frame_height: Any) -> Dict[str, Any]:
    """Build canonical raw-pixel and clamped-normalized geometry.

    ``area_pixels`` is the un-clipped raw box area.  ``area_normalized`` is the
    visible (clipped) box area divided by frame area, so it remains in [0, 1]
    and is consistent with ``bbox_xyxy_normalized``.
    """

    x1, y1, x2, y2 = _four_floats(bbox_xyxy, "bbox_xyxy")
    width = _positive_dimension(frame_width, "frame_width")
    height = _positive_dimension(frame_height, "frame_height")
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox_xyxy must satisfy x2 >= x1 and y2 >= y1")
    clipped = [
        min(width, max(0.0, x1)),
        min(height, max(0.0, y1)),
        min(width, max(0.0, x2)),
        min(height, max(0.0, y2)),
    ]
    normalized = [
        clipped[0] / width,
        clipped[1] / height,
        clipped[2] / width,
        clipped[3] / height,
    ]
    center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
    center_normalized = [
        _clamp_unit(center[0] / width),
        _clamp_unit(center[1] / height),
    ]
    raw_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    clipped_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    was_clipped = any(
        not math.isclose(raw, bounded, rel_tol=0.0, abs_tol=1.0e-12) for raw, bounded in zip((x1, y1, x2, y2), clipped)
    )
    return {
        "bbox_xyxy_pixels": [x1, y1, x2, y2],
        "bbox_xyxy_normalized": normalized,
        "center_pixels": center,
        "center_normalized": center_normalized,
        "area_pixels": raw_area,
        "area_normalized": clipped_area / (width * height),
        "was_clipped": was_clipped,
        "in_frame": clipped[2] > clipped[0] and clipped[3] > clipped[1],
    }


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_compatible(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_compatible(item_method())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return _json_compatible(vars(value))
    return str(value)


def build_detection_row(
    *,
    detection_index: int,
    category_id: int,
    category_name: str,
    score: Any,
    bbox_xyxy: Sequence[Any],
    frame_width: Any,
    frame_height: Any,
    attributes: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one canonical detector result; bbox input is explicitly xyxy."""

    index = int(detection_index)
    if index < 0:
        raise ValueError("detection_index must be non-negative")
    parsed_score = _finite_float(score, "score")
    row: Dict[str, Any] = {
        "detection_index": index,
        "category_id": int(category_id),
        "category_name": str(category_name),
        "score": parsed_score,
        **geometry_from_xyxy(bbox_xyxy, frame_width, frame_height),
    }
    if attributes:
        row["attributes"] = _json_compatible(dict(attributes))
    else:
        row["attributes"] = {}
    return row


def _category_lookup(categories: Sequence[Mapping[str, Any]]) -> Dict[int, str]:
    return {int(category["id"]): str(category.get("name", category["id"])) for category in categories}


def _legacy_detection_attributes(row: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "bbox",
        "bbox_xyxy",
        "bbox_xyxy_pixels",
        "category_id",
        "category_name",
        "score",
        "source",
        "image_id",
        "frame_index",
        "segment_frame_index",
        "timestamp_seconds",
        "source_timestamp_seconds",
        "segment_timestamp_seconds",
        "track_id",
        "association_stage",
        "track_final_confirmed",
        "area",
        *_DETECTION_RESERVED_FIELDS,
    }
    return {
        str(key): _json_compatible(value)
        for key, value in row.items()
        if key not in excluded and not str(key).startswith("track_") and value is not None
    }


def build_detection_rows_from_legacy(
    predictions: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    frame_width: Any,
    frame_height: Any,
) -> List[Dict[str, Any]]:
    """Adapt existing COCO-xywh inference rows without changing those rows."""

    id_to_name = _category_lookup(categories)
    output: List[Dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        if "bbox_xyxy_pixels" in prediction:
            bbox = prediction["bbox_xyxy_pixels"]
        elif "bbox_xyxy" in prediction:
            bbox = prediction["bbox_xyxy"]
        else:
            bbox = xywh_to_xyxy(prediction.get("bbox", ()))
        category_id = int(prediction.get("category_id", -1))
        category_name = str(prediction.get("category_name", id_to_name.get(category_id, category_id)))
        output.append(
            build_detection_row(
                detection_index=index,
                category_id=category_id,
                category_name=category_name,
                score=prediction.get("score", 0.0),
                bbox_xyxy=bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                attributes=_legacy_detection_attributes(prediction),
            )
        )
    return output


def build_track_state_row(
    *,
    track_id: Any,
    provenance: str,
    bbox_xyxy: Optional[Sequence[Any]],
    frame_width: Any,
    frame_height: Any,
    status: Optional[str] = None,
    hits: Optional[Any] = None,
    age_frames: Optional[Any] = None,
    seconds_since_observed: Optional[Any] = None,
    detection_index: Optional[Any] = None,
    category_id: Optional[Any] = None,
    category_name: Optional[str] = None,
    center_pixels: Optional[Sequence[Any]] = None,
    tracker_native: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a canonical observed or predicted tracker state.

    Detector scores intentionally never appear on track-state rows. Scores are
    obtained only through ``detection_index``, preventing predicted states
    from acquiring a fake score.
    """

    normalized_provenance = str(provenance).strip().casefold()
    if normalized_provenance not in {"observed", "predicted"}:
        raise ValueError("track-state provenance must be 'observed' or 'predicted'")
    parsed_detection_index = None if detection_index is None else int(detection_index)
    if normalized_provenance == "predicted" and parsed_detection_index is not None:
        raise ValueError("predicted track states cannot reference a detection_index")
    if parsed_detection_index is not None and parsed_detection_index < 0:
        raise ValueError("detection_index must be non-negative or null")
    if bbox_xyxy is None:
        if center_pixels is None or len(center_pixels) < 2:
            raise ValueError("a center-only track state requires center_pixels")
        width = _positive_dimension(frame_width, "frame_width")
        height = _positive_dimension(frame_height, "frame_height")
        center = [
            _finite_float(center_pixels[0], "center_pixels[0]"),
            _finite_float(center_pixels[1], "center_pixels[1]"),
        ]
        geometry = {
            "bbox_xyxy_pixels": None,
            "bbox_xyxy_normalized": None,
            "center_pixels": center,
            "center_normalized": [
                _clamp_unit(center[0] / width),
                _clamp_unit(center[1] / height),
            ],
            "area_pixels": None,
            "area_normalized": None,
            "was_clipped": None,
            "in_frame": None,
        }
    else:
        geometry = geometry_from_xyxy(bbox_xyxy, frame_width, frame_height)
    if center_pixels is not None:
        if len(center_pixels) < 2:
            raise ValueError("center_pixels must contain two values")
        center = [
            _finite_float(center_pixels[0], "center_pixels[0]"),
            _finite_float(center_pixels[1], "center_pixels[1]"),
        ]
        width = _positive_dimension(frame_width, "frame_width")
        height = _positive_dimension(frame_height, "frame_height")
        geometry["center_pixels"] = center
        geometry["center_normalized"] = [
            _clamp_unit(center[0] / width),
            _clamp_unit(center[1] / height),
        ]
    return {
        "track_id": _json_compatible(track_id),
        "provenance": normalized_provenance,
        "status": None if status is None else str(status),
        "hits": None if hits is None else int(hits),
        "age_frames": None if age_frames is None else int(age_frames),
        "seconds_since_observed": _optional_finite_float(seconds_since_observed, "seconds_since_observed"),
        "detection_index": parsed_detection_index,
        "category_id": None if category_id is None else int(category_id),
        "category_name": None if category_name is None else str(category_name),
        **geometry,
        "motion": {field_name: None for field_name in _MOTION_FIELDS},
        "tracker_native": _json_compatible(dict(tracker_native or {})),
    }


def _bbox_xyxy_from_any(row: Mapping[str, Any], field_name: str = "bbox") -> Optional[List[float]]:
    if "bbox_xyxy_pixels" in row:
        return _four_floats(row["bbox_xyxy_pixels"], "bbox_xyxy_pixels")
    if "bbox_xyxy" in row:
        return _four_floats(row["bbox_xyxy"], "bbox_xyxy")
    if row.get(field_name) is None:
        return None
    return xywh_to_xyxy(row.get(field_name, ()))


def _observed_status(row: Mapping[str, Any]) -> Optional[str]:
    if row.get("track_status") is not None:
        return str(row["track_status"])
    if row.get("status") is not None:
        return str(row["status"])
    confirmed = row.get("track_confirmed")
    if confirmed is None:
        return None
    return "confirmed" if bool(confirmed) else "tentative"


def build_observed_track_states(
    tracked_detection_rows: Sequence[Mapping[str, Any]],
    frame_width: Any,
    frame_height: Any,
    *,
    categories: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Build generic observed states from circle/BoxMOT tracked detections."""

    id_to_name = _category_lookup(categories)
    states: List[Dict[str, Any]] = []
    for detection_index, detection in enumerate(tracked_detection_rows):
        track_id = detection.get("track_id")
        if track_id is None:
            continue
        category_id = detection.get("category_id")
        category_name = detection.get("category_name")
        if category_name is None and category_id is not None:
            category_name = id_to_name.get(int(category_id), str(category_id))
        center = None
        if detection.get("track_center_x") is not None and detection.get("track_center_y") is not None:
            center = [detection["track_center_x"], detection["track_center_y"]]
        native = {
            key: detection[key]
            for key in (
                "track_radius_pixels",
                "track_first_frame_index",
                "track_last_seen_frame_index",
                "association_stage",
                "track_final_confirmed",
            )
            if detection.get(key) is not None
        }
        states.append(
            build_track_state_row(
                track_id=track_id,
                provenance="observed",
                bbox_xyxy=_bbox_xyxy_from_any(detection),
                frame_width=frame_width,
                frame_height=frame_height,
                status=_observed_status(detection),
                hits=detection.get("track_hits", detection.get("hits")),
                age_frames=detection.get("track_age_frames", detection.get("age_frames")),
                seconds_since_observed=0.0,
                detection_index=detection_index,
                category_id=category_id,
                category_name=category_name,
                center_pixels=center,
                tracker_native=native,
            )
        )
    return states


def _tracker_native_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "bbox",
        "bbox_xyxy",
        "bbox_xyxy_pixels",
        "center",
        "track_id",
        "status",
        "observation",
        "provenance",
        "hits",
        "track_hits",
        "age_frames",
        "track_age_frames",
        "seconds_since_observed",
        "detection_index",
        "category_id",
        "category_name",
        "score",
        "confidence",
        "detector_score",
        *_TRACK_RESERVED_FIELDS,
    }
    return {
        str(key): _json_compatible(value) for key, value in row.items() if key not in excluded and value is not None
    }


def build_tracker_state_rows(
    tracker_rows: Sequence[Mapping[str, Any]],
    frame_width: Any,
    frame_height: Any,
    *,
    detections: Sequence[Mapping[str, Any]] = (),
    categories: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Adapt Hybrid or another explicit tracker-state schema.

    For observed Hybrid states, a detection's matching ``track_id`` is the
    authoritative association. This repairs Hybrid's target-local association
    index when the canonical detection list also contains non-target classes.
    """

    id_to_name = _category_lookup(categories)
    by_track_id: Dict[str, int] = {}
    for index, detection in enumerate(detections):
        detection_track_id = detection.get("track_id")
        if detection_track_id is None:
            attributes = detection.get("attributes", {}) or {}
            if isinstance(attributes, Mapping):
                detection_track_id = attributes.get("track_id")
        if detection_track_id is not None:
            by_track_id[str(detection_track_id)] = int(detection.get("detection_index", index))
    states: List[Dict[str, Any]] = []
    for raw in tracker_rows:
        provenance = str(raw.get("provenance", raw.get("observation", "observed"))).casefold()
        if provenance not in {"observed", "predicted"}:
            provenance = "observed" if bool(raw.get("observed", True)) else "predicted"
        association = raw.get("association", {}) or {}
        detection_index = raw.get("detection_index")
        if detection_index is None and isinstance(association, Mapping):
            detection_index = association.get("detection_index")
        track_key = str(raw.get("track_id"))
        if provenance == "observed" and track_key in by_track_id:
            detection_index = by_track_id[track_key]
        if provenance == "predicted":
            detection_index = None
        category_id = raw.get("category_id")
        category_name = raw.get("category_name")
        if detection_index is not None and 0 <= int(detection_index) < len(detections):
            linked = detections[int(detection_index)]
            category_id = linked.get("category_id", category_id)
            category_name = linked.get("category_name", category_name)
        if category_name is None and category_id is not None:
            category_name = id_to_name.get(int(category_id), str(category_id))
        center = raw.get("center")
        if center is None and raw.get("track_center_x") is not None and raw.get("track_center_y") is not None:
            center = [raw["track_center_x"], raw["track_center_y"]]
        states.append(
            build_track_state_row(
                track_id=raw.get("track_id"),
                provenance=provenance,
                bbox_xyxy=_bbox_xyxy_from_any(raw),
                frame_width=frame_width,
                frame_height=frame_height,
                status=raw.get("status", raw.get("track_status")),
                hits=raw.get("hits", raw.get("track_hits")),
                age_frames=raw.get("age_frames", raw.get("track_age_frames")),
                seconds_since_observed=raw.get(
                    "seconds_since_observed",
                    0.0 if provenance == "observed" else None,
                ),
                detection_index=detection_index,
                category_id=category_id,
                category_name=category_name,
                center_pixels=center,
                tracker_native=_tracker_native_fields(raw),
            )
        )
    return states


def normalize_camera_motion(
    camera_motion: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize Hybrid previous-to-current CMC diagnostics."""

    if not camera_motion:
        return None
    raw = dict(camera_motion)
    affine_value = raw.get(
        "affine_previous_to_current",
        raw.get(
            "affine_2x3_pixels",
            raw.get("affine", raw.get("cmc_affine")),
        ),
    )
    affine: Optional[List[List[float]]] = None
    translation: Optional[List[float]] = None
    scale: Optional[float] = None
    rotation: Optional[float] = None
    if affine_value is not None:
        try:
            if len(affine_value) == 2 and all(len(row) >= 3 for row in affine_value):
                affine = [
                    [
                        _finite_float(
                            row[index],
                            f"camera_motion.affine[{row_index}][{index}]",
                        )
                        for index in range(3)
                    ]
                    for row_index, row in enumerate(affine_value)
                ]
            elif len(affine_value) >= 6:
                flat = [_finite_float(affine_value[index], "camera_motion.affine") for index in range(6)]
                affine = [flat[:3], flat[3:6]]
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("camera motion affine must be a 2x3 matrix") from exc
        a, _b, tx = affine[0]
        c, _d, ty = affine[1]
        translation = [tx, ty]
        scale = math.hypot(a, c)
        rotation = math.degrees(math.atan2(c, a)) % 360.0
    elif raw.get("translation_pixels") is not None:
        values = raw["translation_pixels"]
        if len(values) < 2:
            raise ValueError("camera_motion.translation_pixels must have two values")
        translation = [
            _finite_float(values[0], "camera_motion.translation_pixels[0]"),
            _finite_float(values[1], "camera_motion.translation_pixels[1]"),
        ]
        scale = _optional_finite_float(raw.get("scale"), "camera_motion.scale")
        rotation = _optional_finite_float(
            raw.get("rotation_clockwise_degrees"),
            "camera_motion.rotation_clockwise_degrees",
        )
    return {
        "affine_previous_to_current": affine,
        "translation_pixels": translation,
        "scale": scale,
        "rotation_clockwise_degrees": rotation,
        "success": bool(raw.get("success", affine is not None)),
        "inliers": None if raw.get("inliers") is None else int(raw["inliers"]),
        "reason": None if raw.get("reason") is None else str(raw["reason"]),
        "method": None if raw.get("method") is None else str(raw["method"]),
        "processing_scale": _optional_finite_float(raw.get("processing_scale"), "camera_motion.processing_scale"),
    }


def tracker_capabilities(algorithm: Optional[str], *, enabled: bool = True) -> Dict[str, bool]:
    """Describe only state that the integration can export reliably."""

    name = str(algorithm or "none").strip().casefold()
    if not enabled or name in {"", "none", "disabled"}:
        return {
            "observed_track_states": False,
            "predicted_track_states": False,
            "predicted_center": False,
            "predicted_bbox": False,
            "camera_motion_affine": False,
            "native_covariance": False,
        }
    hybrid = name == "hybrid"
    circle = name == "circle"
    return {
        "observed_track_states": True,
        "predicted_track_states": hybrid or circle,
        "predicted_center": hybrid or circle,
        "predicted_bbox": hybrid,
        "camera_motion_affine": hybrid,
        "native_covariance": hybrid,
    }


def _empty_motion() -> Dict[str, Any]:
    return {field_name: None for field_name in _MOTION_FIELDS}


def _direction_8way(angle: Optional[float], speed: float) -> Optional[str]:
    if speed <= 1.0e-12:
        return "stationary"
    assert angle is not None
    labels = (
        "east",
        "south_east",
        "south",
        "south_west",
        "west",
        "north_west",
        "north",
        "north_east",
    )
    return labels[int((angle + 22.5) // 45.0) % 8]


@dataclass
class _TrackHistory:
    points: List[Dict[str, Any]] = field(default_factory=list)
    detector_scores: List[float] = field(default_factory=list)
    category_ids: set[int] = field(default_factory=set)
    category_names: set[str] = field(default_factory=set)
    previous_velocity_pixels: Optional[Tuple[float, float]] = None
    previous_velocity_normalized: Optional[Tuple[float, float]] = None
    path_length_pixels: float = 0.0
    path_length_normalized: float = 0.0
    max_gap_frames: int = 0
    max_gap_seconds: float = 0.0
    frame_width: Optional[float] = None
    frame_height: Optional[float] = None
    last_observed_timestamp: Optional[float] = None


class TrajectoryAccumulator:
    """Accumulate video-local trajectories and timestamp-derived motion."""

    def __init__(
        self,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> None:
        self._tracks: Dict[Any, _TrackHistory] = {}
        self._frame_width = None if frame_width is None else _positive_dimension(frame_width, "frame_width")
        self._frame_height = None if frame_height is None else _positive_dimension(frame_height, "frame_height")

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def add_state(
        self,
        state: Mapping[str, Any],
        *,
        segment_frame_index: int,
        source_frame_index: int,
        source_timestamp_seconds: Optional[float],
        segment_timestamp_seconds: Optional[float],
        detector_score: Optional[float] = None,
        continuity_known: bool = True,
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Append a point, returning the canonical real-time motion object."""

        track_id = state.get("track_id")
        if track_id is None:
            raise ValueError("track states require a non-null track_id")
        history = self._tracks.setdefault(track_id, _TrackHistory())
        resolved_width = self._frame_width if frame_width is None else _positive_dimension(frame_width, "frame_width")
        resolved_height = (
            self._frame_height if frame_height is None else _positive_dimension(frame_height, "frame_height")
        )
        if resolved_width is None or resolved_height is None:
            raise ValueError(
                "TrajectoryAccumulator requires frame_width and frame_height for unclamped normalized motion"
            )
        if history.frame_width is None:
            history.frame_width = resolved_width
            history.frame_height = resolved_height
        elif not math.isclose(history.frame_width, resolved_width) or not math.isclose(
            history.frame_height or 0.0, resolved_height
        ):
            raise ValueError("track frame dimensions changed within one video")
        center_pixels = [float(value) for value in state["center_pixels"][:2]]
        center_normalized = [float(value) for value in state["center_normalized"][:2]]
        area_pixels = None if state.get("area_pixels") is None else float(state["area_pixels"])
        area_normalized = None if state.get("area_normalized") is None else float(state["area_normalized"])
        timestamp = segment_timestamp_seconds if segment_timestamp_seconds is not None else source_timestamp_seconds
        motion = _empty_motion()
        previous = history.points[-1] if history.points else None
        valid_step = False
        if previous is not None:
            previous_timestamp = previous.get("segment_timestamp_seconds")
            if previous_timestamp is None:
                previous_timestamp = previous.get("source_timestamp_seconds")
            frame_gap = source_frame_index - int(previous["source_frame_index"])
            delta_time = (
                None
                if previous_timestamp is None or timestamp is None
                else float(timestamp) - float(previous_timestamp)
            )
            if frame_gap > 0:
                history.max_gap_frames = max(history.max_gap_frames, frame_gap - 1)
            if delta_time is not None and delta_time > 0.0:
                history.max_gap_seconds = max(history.max_gap_seconds, delta_time)
            if continuity_known and frame_gap == 1 and delta_time is not None and delta_time > 0.0:
                valid_step = True
                dx = center_pixels[0] - float(previous["center_pixels"][0])
                dy = center_pixels[1] - float(previous["center_pixels"][1])
                ndx = dx / resolved_width
                ndy = dy / resolved_height
                velocity_pixels = (dx / delta_time, dy / delta_time)
                velocity_normalized = (ndx / delta_time, ndy / delta_time)
                speed_pixels = math.hypot(*velocity_pixels)
                speed_normalized = math.hypot(*velocity_normalized)
                angle = (
                    None
                    if speed_pixels <= 1.0e-12
                    else math.degrees(math.atan2(velocity_pixels[1], velocity_pixels[0])) % 360.0
                )
                acceleration_pixels = None
                acceleration_normalized = None
                acceleration_magnitude_pixels = None
                acceleration_magnitude_normalized = None
                if history.previous_velocity_pixels is not None:
                    acceleration_pixels = [
                        (velocity_pixels[index] - history.previous_velocity_pixels[index]) / delta_time
                        for index in range(2)
                    ]
                    acceleration_magnitude_pixels = math.hypot(*acceleration_pixels)
                if history.previous_velocity_normalized is not None:
                    acceleration_normalized = [
                        (velocity_normalized[index] - history.previous_velocity_normalized[index]) / delta_time
                        for index in range(2)
                    ]
                    acceleration_magnitude_normalized = math.hypot(*acceleration_normalized)
                log_area_change = None
                previous_area = previous["area_pixels"]
                if (
                    area_pixels is not None
                    and previous_area is not None
                    and area_pixels > 0.0
                    and float(previous_area) > 0.0
                ):
                    log_area_change = math.log(area_pixels / float(previous_area)) / delta_time
                motion.update(
                    delta_time_seconds=delta_time,
                    frame_gap=frame_gap,
                    velocity_pixels_per_second=list(velocity_pixels),
                    velocity_normalized_per_second=list(velocity_normalized),
                    speed_pixels_per_second=speed_pixels,
                    speed_normalized_per_second=speed_normalized,
                    direction_clockwise_degrees=angle,
                    direction_8way=_direction_8way(angle, speed_pixels),
                    acceleration_pixels_per_second_squared=acceleration_pixels,
                    acceleration_normalized_per_second_squared=acceleration_normalized,
                    acceleration_magnitude_pixels_per_second_squared=acceleration_magnitude_pixels,
                    acceleration_magnitude_normalized_per_second_squared=acceleration_magnitude_normalized,
                    bbox_log_area_change_per_second=log_area_change,
                )
                history.previous_velocity_pixels = velocity_pixels
                history.previous_velocity_normalized = velocity_normalized
                history.path_length_pixels += math.hypot(dx, dy)
                history.path_length_normalized += math.hypot(ndx, ndy)
        if not valid_step:
            history.previous_velocity_pixels = None
            history.previous_velocity_normalized = None
        if detector_score is not None and state.get("provenance") == "observed":
            history.detector_scores.append(float(detector_score))
        if state.get("provenance") == "observed":
            computed_seconds_since_observed = 0.0
            if timestamp is not None:
                history.last_observed_timestamp = float(timestamp)
        elif timestamp is not None and history.last_observed_timestamp is not None:
            computed_seconds_since_observed = max(0.0, float(timestamp) - history.last_observed_timestamp)
        else:
            computed_seconds_since_observed = None
        if isinstance(state, MutableMapping):
            state["seconds_since_observed"] = computed_seconds_since_observed
        if state.get("category_id") is not None:
            history.category_ids.add(int(state["category_id"]))
        if state.get("category_name") is not None:
            history.category_names.add(str(state["category_name"]))
        point = {
            "segment_frame_index": int(segment_frame_index),
            "source_frame_index": int(source_frame_index),
            "source_timestamp_seconds": source_timestamp_seconds,
            "segment_timestamp_seconds": segment_timestamp_seconds,
            "provenance": str(state.get("provenance", "observed")),
            "status": state.get("status"),
            "hits": state.get("hits"),
            "age_frames": state.get("age_frames"),
            "seconds_since_observed": computed_seconds_since_observed,
            "detection_index": state.get("detection_index"),
            "center_pixels": center_pixels,
            "center_normalized": center_normalized,
            "area_pixels": area_pixels,
            "area_normalized": area_normalized,
            "detector_score": (
                float(detector_score) if detector_score is not None and state.get("provenance") == "observed" else None
            ),
            "motion": motion,
        }
        history.points.append(point)
        return motion

    def track_rows(self, video_id: str) -> List[Dict[str, Any]]:
        """Return deterministic one-row-per-track aggregates."""

        output: List[Dict[str, Any]] = []
        for track_id in sorted(self._tracks, key=lambda value: (str(type(value)), str(value))):
            history = self._tracks[track_id]
            points = history.points
            first = points[0]
            last = points[-1]
            scores = history.detector_scores
            score_stats = {
                "count": len(scores),
                "minimum": min(scores) if scores else None,
                "maximum": max(scores) if scores else None,
                "mean": (sum(scores) / len(scores)) if scores else None,
            }
            duration = None
            start_time = first.get("segment_timestamp_seconds")
            end_time = last.get("segment_timestamp_seconds")
            if start_time is not None and end_time is not None:
                duration = max(0.0, float(end_time) - float(start_time))
            net_pixels = [
                float(last["center_pixels"][index]) - float(first["center_pixels"][index]) for index in range(2)
            ]
            assert history.frame_width is not None
            assert history.frame_height is not None
            net_normalized = [
                net_pixels[0] / history.frame_width,
                net_pixels[1] / history.frame_height,
            ]
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "video_id": video_id,
                    "track_id": _json_compatible(track_id),
                    "scope": "video_id",
                    "start_segment_frame_index": first["segment_frame_index"],
                    "end_segment_frame_index": last["segment_frame_index"],
                    "start_source_frame_index": first["source_frame_index"],
                    "end_source_frame_index": last["source_frame_index"],
                    "start_timestamp_seconds": first["source_timestamp_seconds"],
                    "end_timestamp_seconds": last["source_timestamp_seconds"],
                    "duration_seconds": duration,
                    "point_count": len(points),
                    "observed_point_count": sum(point["provenance"] == "observed" for point in points),
                    "predicted_point_count": sum(point["provenance"] == "predicted" for point in points),
                    "max_gap_frames": history.max_gap_frames,
                    "max_gap_seconds": history.max_gap_seconds,
                    "detector_score_statistics": score_stats,
                    "path_length_pixels": history.path_length_pixels,
                    "path_length_normalized": history.path_length_normalized,
                    "net_displacement_pixels": math.hypot(*net_pixels),
                    "net_displacement_normalized": math.hypot(*net_normalized),
                    "net_displacement_vector_pixels": net_pixels,
                    "net_displacement_vector_normalized": net_normalized,
                    "category_ids": sorted(history.category_ids),
                    "category_names": sorted(history.category_names),
                    "points": copy.deepcopy(points),
                }
            )
        return output


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_source_identity(source: Any) -> str:
    text = str(source)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        return text
    return str(Path(text).expanduser().resolve(strict=False))


def make_video_id(
    source: Any,
    selection: VideoSelection | Mapping[str, Any],
) -> str:
    """Create a stable source-and-selected-range video identifier."""

    selected = (
        selection
        if isinstance(selection, VideoSelection)
        else VideoSelection(
            start_frame=int(selection.get("start_frame", 0)),
            end_frame_exclusive=selection.get("end_frame_exclusive", selection.get("end_frame")),
            start_seconds=selection.get("start_seconds", selection.get("start_time")),
            end_seconds=selection.get(
                "end_seconds",
                selection.get("effective_end_seconds", selection.get("end_time")),
            ),
        )
    )
    payload = {
        "source": _normalized_source_identity(source),
        "selection": selected.to_dict(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    source_name = Path(str(source).split("?", 1)[0]).stem or "video"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", source_name).strip("-._")
    slug = (slug or "video")[:48]
    return f"{slug}-{digest}"


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rational_float(value: Any) -> Optional[float]:
    if value in (None, "", "N/A", "0/0"):
        return None
    text = str(value)
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0.0:
                return None
            return float(numerator) / denominator_value
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _empty_probe(backend: str, warnings: Sequence[str] = ()) -> Dict[str, Any]:
    return {
        "probe_backend": backend,
        "format": {
            "name": None,
            "long_name": None,
            "duration_seconds": None,
            "size_bytes": None,
            "bit_rate": None,
        },
        "video": {
            "stream_index": None,
            "codec_name": None,
            "codec_long_name": None,
            "profile": None,
            "decoded_width": None,
            "decoded_height": None,
            "display_width": None,
            "display_height": None,
            "pixel_format": None,
            "fps_rational": None,
            "fps_float": None,
            "nominal_fps_rational": None,
            "time_base": None,
            "is_variable_frame_rate": None,
            "frame_count": None,
            "duration_seconds": None,
            "rotation_degrees": None,
        },
        "audio": {
            "has_audio": False,
            "stream_index": None,
            "codec_name": None,
            "sample_rate_hz": None,
            "channels": None,
            "channel_layout": None,
            "duration_seconds": None,
        },
        "probe_warnings": list(warnings),
    }


def _stream_rotation(stream: Mapping[str, Any]) -> float:
    tags = stream.get("tags", {}) or {}
    tagged = _safe_float(tags.get("rotate")) if isinstance(tags, Mapping) else None
    if tagged is not None:
        return tagged % 360.0
    for side_data in stream.get("side_data_list", []) or []:
        rotation = _safe_float(side_data.get("rotation"))
        if rotation is not None:
            return rotation % 360.0
    return 0.0


def _parse_ffprobe_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    streams = list(payload.get("streams", []) or [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not videos:
        raise RuntimeError("ffprobe found no video stream")
    video = next(
        (stream for stream in videos if int((stream.get("disposition", {}) or {}).get("default", 0)) == 1),
        videos[0],
    )
    audio = next(
        (stream for stream in audios if int((stream.get("disposition", {}) or {}).get("default", 0)) == 1),
        audios[0] if audios else None,
    )
    format_row = payload.get("format", {}) or {}
    avg_rate = str(video.get("avg_frame_rate") or "") or None
    nominal_rate = str(video.get("r_frame_rate") or "") or None
    avg_float = _rational_float(avg_rate)
    nominal_float = _rational_float(nominal_rate)
    is_vfr = None
    if avg_float is not None and nominal_float is not None:
        is_vfr = not math.isclose(avg_float, nominal_float, rel_tol=1.0e-6, abs_tol=1.0e-6)
    rotation = _stream_rotation(video)
    decoded_width = _safe_int(video.get("width"))
    decoded_height = _safe_int(video.get("height"))
    quarter_turn = int(round(rotation / 90.0)) % 2 == 1
    display_width = decoded_height if quarter_turn else decoded_width
    display_height = decoded_width if quarter_turn else decoded_height
    video_duration = _safe_float(video.get("duration"))
    format_duration = _safe_float(format_row.get("duration"))
    frame_count = _safe_int(video.get("nb_read_frames", video.get("nb_frames")))
    if frame_count is None and avg_float and (video_duration or format_duration):
        frame_count = int(round((video_duration or format_duration or 0.0) * avg_float))
    result = _empty_probe("ffprobe")
    result["format"] = {
        "name": format_row.get("format_name"),
        "long_name": format_row.get("format_long_name"),
        "duration_seconds": format_duration,
        "size_bytes": _safe_int(format_row.get("size")),
        "bit_rate": _safe_int(format_row.get("bit_rate")),
    }
    result["video"] = {
        "stream_index": _safe_int(video.get("index")),
        "codec_name": video.get("codec_name"),
        "codec_long_name": video.get("codec_long_name"),
        "profile": video.get("profile"),
        "decoded_width": decoded_width,
        "decoded_height": decoded_height,
        "display_width": display_width,
        "display_height": display_height,
        "pixel_format": video.get("pix_fmt"),
        "fps_rational": avg_rate if avg_float is not None else nominal_rate,
        "fps_float": avg_float if avg_float is not None else nominal_float,
        "nominal_fps_rational": nominal_rate,
        "time_base": video.get("time_base"),
        "is_variable_frame_rate": is_vfr,
        "frame_count": frame_count,
        "duration_seconds": video_duration or format_duration,
        "rotation_degrees": rotation,
    }
    if audio is not None:
        result["audio"] = {
            "has_audio": True,
            "stream_index": _safe_int(audio.get("index")),
            "codec_name": audio.get("codec_name"),
            "sample_rate_hz": _safe_int(audio.get("sample_rate")),
            "channels": _safe_int(audio.get("channels")),
            "channel_layout": audio.get("channel_layout"),
            "duration_seconds": _safe_float(audio.get("duration")) or format_duration,
        }
    return result


def _probe_with_ffprobe(
    source: Any,
    ffprobe_path: str,
    runner: Callable[..., Any],
    *,
    count_frames: bool,
) -> Dict[str, Any]:
    command = [ffprobe_path, "-v", "error"]
    if count_frames:
        command.append("-count_frames")
    command.extend(
        [
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            str(source),
        ]
    )
    result = _run_process(runner, command)
    if int(getattr(result, "returncode", 1)) != 0:
        raise RuntimeError((getattr(result, "stderr", "") or "ffprobe failed").strip())
    try:
        payload = json.loads(getattr(result, "stdout", "") or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc
    return _parse_ffprobe_payload(payload)


def _probe_with_ffmpeg_text(
    source: Any,
    ffmpeg_path: str,
    runner: Callable[..., Any],
) -> Optional[Dict[str, Any]]:
    result = _run_process(runner, [ffmpeg_path, "-hide_banner", "-i", str(source)])
    text = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
    video_match = re.search(
        r"Stream #(?P<input>\d+):(?P<index>\d+).*?Video:\s*(?P<codec>[^,\s]+)"
        r".*?(?P<width>\d{2,6})x(?P<height>\d{2,6}).*?"
        r"(?P<fps>\d+(?:\.\d+)?)\s+fps",
        text,
    )
    if video_match is None:
        return None
    result_probe = _empty_probe("ffmpeg_text")
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    duration = None
    if duration_match:
        duration = (
            int(duration_match.group(1)) * 3600.0 + int(duration_match.group(2)) * 60.0 + float(duration_match.group(3))
        )
    fps = float(video_match.group("fps"))
    width = int(video_match.group("width"))
    height = int(video_match.group("height"))
    result_probe["format"]["duration_seconds"] = duration
    result_probe["video"].update(
        stream_index=int(video_match.group("index")),
        codec_name=video_match.group("codec"),
        decoded_width=width,
        decoded_height=height,
        display_width=width,
        display_height=height,
        fps_rational=str(fps),
        fps_float=fps,
        frame_count=None if duration is None else int(round(duration * fps)),
        duration_seconds=duration,
    )
    audio_match = re.search(
        r"Stream #\d+:(?P<index>\d+).*?Audio:\s*(?P<codec>[^,\s]+)"
        r".*?(?P<sample_rate>\d+)\s+Hz(?:,\s*(?P<layout>[^,\r\n]+))?",
        text,
    )
    if audio_match:
        result_probe["audio"].update(
            has_audio=True,
            stream_index=int(audio_match.group("index")),
            codec_name=audio_match.group("codec"),
            sample_rate_hz=int(audio_match.group("sample_rate")),
            channel_layout=audio_match.group("layout"),
            duration_seconds=duration,
        )
    return result_probe


def _probe_with_opencv(source: Any) -> Optional[Dict[str, Any]]:
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            return None
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        return None
    result = _empty_probe("opencv")
    result["video"].update(
        decoded_width=width,
        decoded_height=height,
        display_width=width,
        display_height=height,
        fps_rational=None if fps <= 0.0 else str(fps),
        fps_float=None if fps <= 0.0 else fps,
        frame_count=None if frame_count <= 0 else frame_count,
        duration_seconds=(None if fps <= 0.0 or frame_count <= 0 else frame_count / fps),
    )
    return result


def _merge_probe_fallback(probe: Dict[str, Any], fallback_metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not fallback_metadata:
        return probe
    fallback = dict(fallback_metadata)
    aliases = {
        "decoded_width": ("decoded_width", "width", "frame_width"),
        "decoded_height": ("decoded_height", "height", "frame_height"),
        "display_width": ("display_width", "width", "frame_width"),
        "display_height": ("display_height", "height", "frame_height"),
        "fps_float": ("fps_float", "input_fps", "fps"),
        "frame_count": ("frame_count", "source_frame_count"),
        "duration_seconds": ("duration_seconds",),
    }
    for target, names in aliases.items():
        if probe["video"].get(target) is not None:
            continue
        for name in names:
            if fallback.get(name) is not None:
                probe["video"][target] = _json_compatible(fallback[name])
                break
    if probe["video"].get("fps_rational") is None and probe["video"].get("fps_float"):
        probe["video"]["fps_rational"] = str(probe["video"]["fps_float"])
    if probe["format"].get("duration_seconds") is None:
        probe["format"]["duration_seconds"] = probe["video"].get("duration_seconds")
    return probe


def probe_media(
    source: Any,
    *,
    ffprobe_path: str = "auto",
    ffmpeg_path: str = "auto",
    fallback_metadata: Optional[Mapping[str, Any]] = None,
    count_frames: bool = False,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """Probe media through FFprobe, FFmpeg text, OpenCV, then caller metadata."""

    warnings: List[str] = []
    explicit_ffprobe = str(ffprobe_path).strip().casefold() != "auto"
    resolved_ffprobe = _resolve_executable(ffprobe_path, "ffprobe", required=explicit_ffprobe)
    if resolved_ffprobe:
        try:
            return _merge_probe_fallback(
                _probe_with_ffprobe(
                    source,
                    resolved_ffprobe,
                    runner,
                    count_frames=count_frames,
                ),
                fallback_metadata,
            )
        except (OSError, RuntimeError) as exc:
            warnings.append(f"ffprobe: {exc}")
            if explicit_ffprobe:
                raise RuntimeError(f"explicit FFprobe failed: {exc}") from exc
    resolved_ffmpeg = _resolve_executable(ffmpeg_path, "ffmpeg", required=False)
    if resolved_ffmpeg:
        try:
            ffmpeg_probe = _probe_with_ffmpeg_text(source, resolved_ffmpeg, runner)
            if ffmpeg_probe is not None:
                ffmpeg_probe["probe_warnings"] = warnings
                return _merge_probe_fallback(ffmpeg_probe, fallback_metadata)
        except OSError as exc:
            warnings.append(f"ffmpeg: {exc}")
    try:
        opencv_probe = _probe_with_opencv(source)
    except Exception as exc:
        warnings.append(f"opencv: {exc}")
        opencv_probe = None
    if opencv_probe is not None:
        opencv_probe["probe_warnings"] = warnings
        return _merge_probe_fallback(opencv_probe, fallback_metadata)
    unavailable = _empty_probe("provided_metadata" if fallback_metadata else "unavailable", warnings)
    return _merge_probe_fallback(unavailable, fallback_metadata)


def _media_duration(probe: Mapping[str, Any]) -> Optional[float]:
    video = probe.get("video", {}) or {}
    format_row = probe.get("format", {}) or {}
    return _safe_float(video.get("duration_seconds")) or _safe_float(format_row.get("duration_seconds"))


def _validate_clean_media(
    probe: Mapping[str, Any],
    *,
    expected_frames: Optional[int],
    expected_width: Optional[int],
    expected_height: Optional[int],
    expected_audio: bool,
    expected_duration: Optional[float],
) -> None:
    video = probe.get("video", {}) or {}
    audio = probe.get("audio", {}) or {}
    if expected_frames is not None:
        if video.get("frame_count") is None:
            raise RuntimeError("clean media probe did not report a frame count")
        if int(video["frame_count"]) != int(expected_frames):
            raise RuntimeError(
                f"clean media frame count mismatch: expected {expected_frames}, got {video['frame_count']}"
            )
    if expected_width is not None:
        if video.get("display_width") is None:
            raise RuntimeError("clean media probe did not report display width")
        if int(video["display_width"]) != int(expected_width):
            raise RuntimeError("clean media display width mismatch")
    if expected_height is not None:
        if video.get("display_height") is None:
            raise RuntimeError("clean media probe did not report display height")
        if int(video["display_height"]) != int(expected_height):
            raise RuntimeError("clean media display height mismatch")
    if expected_audio and not bool(audio.get("has_audio")):
        raise RuntimeError("source has audio but clean media does not")
    video_duration = _media_duration(probe)
    audio_duration = _safe_float(audio.get("duration_seconds"))
    fps = _safe_float(video.get("fps_float"))
    tolerance = (1.0 / fps + 1.0e-3) if fps and fps > 0.0 else 0.05
    if expected_duration is not None:
        if video_duration is None:
            raise RuntimeError("clean media probe did not report duration")
        duration_delta = abs(video_duration - expected_duration)
        if duration_delta > tolerance:
            raise RuntimeError(
                "clean media duration differs from selected duration by more than one frame: "
                f"video={video_duration:.6f}s, selected={expected_duration:.6f}s, "
                f"difference={duration_delta:.6f}s, tolerance={tolerance:.6f}s"
            )
    if expected_audio and audio_duration is None:
        raise RuntimeError("clean media probe did not report audio duration")
    if expected_audio and video_duration is not None and audio_duration is not None:
        audio_video_delta = abs(video_duration - audio_duration)
        if audio_video_delta > tolerance:
            raise RuntimeError(
                "clean media audio/video duration differs by more than one frame: "
                f"video={video_duration:.6f}s, audio={audio_duration:.6f}s, "
                f"difference={audio_video_delta:.6f}s, tolerance={tolerance:.6f}s"
            )


def transcode_clean_media(
    source_path: Any,
    output_path: Path | str,
    selection: VideoSelection,
    media_config: CanonicalMediaConfig,
    toolchain: CanonicalToolchain,
    *,
    expected_frames: Optional[int] = None,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    source_probe: Optional[Mapping[str, Any]] = None,
    runner: Callable[..., Any] = subprocess.run,
    probe_fn: Callable[..., Dict[str, Any]] = probe_media,
) -> Dict[str, Any]:
    """Encode an exact clean selected clip with H.264 and optional AAC audio."""

    if not toolchain.ffmpeg_path:
        raise RuntimeError("clean media requires a preflighted FFmpeg toolchain")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    end_frame = selection.end_frame_exclusive
    if end_frame is None and expected_frames is not None:
        end_frame = selection.start_frame + int(expected_frames)
    if end_frame is None:
        raise ValueError("clean media requires an exclusive end frame")
    source_info = dict(source_probe or {})
    video_info = source_info.get("video", {}) or {}
    audio_info = source_info.get("audio", {}) or {}
    has_audio = bool(audio_info.get("has_audio", False))
    fps = _safe_float(video_info.get("fps_float"))
    start_seconds = selection.start_seconds
    end_seconds = selection.end_seconds
    if start_seconds is None and fps and fps > 0.0:
        start_seconds = selection.start_frame / fps
    if end_seconds is None and fps and fps > 0.0:
        end_seconds = end_frame / fps
    selected_duration = None
    if start_seconds is not None and end_seconds is not None:
        selected_duration = float(end_seconds) - float(start_seconds)
        if not math.isfinite(selected_duration) or selected_duration <= 0.0:
            raise ValueError("clean media selected duration must be a finite positive number")
    if has_audio and (start_seconds is None or end_seconds is None):
        raise RuntimeError("cannot trim source audio without selected timestamps or a valid FPS")
    video_stream_index = video_info.get("stream_index")
    video_input = f"0:{int(video_stream_index)}" if video_stream_index is not None else "0:v:0"
    video_filter = (
        f"[{video_input}]trim=start_frame={selection.start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS[v]"
    )
    filter_parts = [video_filter]
    command = [
        toolchain.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
    ]
    if has_audio:
        audio_index = audio_info.get("stream_index")
        audio_input = f"0:{int(audio_index)}" if audio_index is not None else "0:a:0"
        # AAC packet timestamps can cover less time than their decoded sample
        # count. Pad indefinitely, then trim in the reset PTS domain so the
        # audio timeline reaches the selected video endpoint exactly.
        filter_parts.append(
            f"[{audio_input}]atrim=start={float(start_seconds):.12f}:"
            f"end={float(end_seconds):.12f},asetpts=PTS-STARTPTS,"
            f"apad,atrim=duration={float(selected_duration):.12f}[a]"
        )
    command.extend(["-filter_complex", ";".join(filter_parts), "-map", "[v]"])
    if has_audio:
        command.extend(["-map", "[a]"])
    command.extend(
        [
            "-c:v",
            media_config.video_codec,
            "-crf",
            str(media_config.crf),
            "-preset",
            media_config.preset,
            "-pix_fmt",
            media_config.pixel_format,
            "-fps_mode",
            "passthrough",
        ]
    )
    if has_audio:
        command.extend(
            [
                "-c:a",
                media_config.audio_codec,
                "-b:a",
                media_config.audio_bitrate,
            ]
        )
    else:
        command.append("-an")
    command.extend(["-movflags", "+faststart", "-map_chapters", "-1", str(output)])
    result = _run_process(runner, command)
    if int(getattr(result, "returncode", 1)) != 0:
        detail = (getattr(result, "stderr", "") or "unknown FFmpeg error").strip()
        raise RuntimeError(f"clean media FFmpeg encode failed: {detail}")
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("clean media FFmpeg did not create a non-empty output")
    output_probe = probe_fn(
        output,
        ffprobe_path=toolchain.ffprobe_path or "auto",
        ffmpeg_path=toolchain.ffmpeg_path,
        count_frames=True,
        runner=runner,
    )
    _validate_clean_media(
        output_probe,
        expected_frames=expected_frames,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_audio=has_audio,
        expected_duration=selected_duration,
    )
    return output_probe


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Write UTF-8 JSON through a same-directory atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_compatible(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        _json_compatible(row),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _field_value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _selection_from_window(frame_window: Any, *, input_fps: Optional[float]) -> VideoSelection:
    start_frame = int(_field_value(frame_window, "start_frame", default=0) or 0)
    end_frame = _field_value(frame_window, "end_frame_exclusive", "end_frame", default=None)
    start_seconds = _field_value(frame_window, "start_seconds", "start_time", default=None)
    end_seconds = _field_value(
        frame_window,
        "effective_end_seconds",
        "end_seconds",
        "end_time",
        default=None,
    )
    if start_seconds is None and input_fps and input_fps > 0.0:
        start_seconds = start_frame / input_fps
    if end_seconds is None and end_frame is not None and input_fps and input_fps > 0.0:
        end_seconds = int(end_frame) / input_fps
    return VideoSelection(
        start_frame=start_frame,
        end_frame_exclusive=None if end_frame is None else int(end_frame),
        start_seconds=None if start_seconds is None else float(start_seconds),
        end_seconds=None if end_seconds is None else float(end_seconds),
    )


def _tracking_metadata(tracking_config: Any) -> Dict[str, Any]:
    configuration = _json_compatible(tracking_config or {})
    enabled = bool(_field_value(tracking_config, "enabled", default=False))
    algorithm = str(_field_value(tracking_config, "algorithm", default="none")).casefold()
    lookahead = None
    backfill = False
    if isinstance(configuration, Mapping):
        hybrid = configuration.get("hybrid_options", configuration.get("hybrid", {})) or {}
        if isinstance(hybrid, Mapping):
            hypothesis = hybrid.get("hypothesis", {}) or {}
            if isinstance(hypothesis, Mapping):
                lookahead = hypothesis.get("lookahead_seconds")
            lookahead = hybrid.get("lookahead_seconds", lookahead)
        lookahead = configuration.get("lookahead_seconds", lookahead)
        backfill = (enabled and algorithm == "hybrid") or bool(
            configuration.get(
                "confirmation_backfill",
                configuration.get("backfill", False),
            )
        )
    return {
        "enabled": enabled,
        "algorithm": algorithm,
        "capabilities": tracker_capabilities(algorithm, enabled=enabled),
        "offline": {
            "lookahead_seconds": _safe_float(lookahead),
            "confirmation_backfill": backfill,
        },
        "configuration": configuration,
    }


class CanonicalRunWriter:
    """Own a run-level manifest and one atomic bundle per selected video."""

    def __init__(
        self,
        output_dir: Path | str,
        cfg: CanonicalOutputConfig | Mapping[str, Any],
        categories: Sequence[Mapping[str, Any]],
        producer_metadata: Mapping[str, Any],
        *,
        toolchain: Optional[CanonicalToolchain] = None,
        runner: Callable[..., Any] = subprocess.run,
        probe_fn: Callable[..., Dict[str, Any]] = probe_media,
        transcode_fn: Callable[..., Dict[str, Any]] = transcode_clean_media,
    ) -> None:
        if isinstance(cfg, CanonicalOutputConfig):
            validated = parse_canonical_output_config({"canonical_output": asdict(cfg)})
        else:
            validated = parse_canonical_output_config(cfg)
        self.cfg = validated
        self.output_dir = Path(output_dir)
        self.root = self.output_dir / validated.directory
        self.categories = [_json_compatible(dict(category)) for category in categories]
        self.producer_metadata = _json_compatible(dict(producer_metadata))
        self.runner = runner
        self.probe_fn = probe_fn
        self.transcode_fn = transcode_fn
        self.toolchain = toolchain if toolchain is not None else preflight(validated, runner=runner)
        self._active: Dict[str, CanonicalVideoWriter] = {}
        self._summaries: List[Dict[str, Any]] = []
        self._manifest: Optional[Dict[str, Any]] = None

    def start_video(
        self,
        source: Any = None,
        *,
        source_path: Any = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        input_fps: Optional[float] = None,
        source_frame_count: Optional[int] = None,
        frame_window: Any = None,
        detection_fps: Optional[float] = None,
        frame_interval: int = 1,
        output_fps: Optional[float] = None,
        tracking_config: Any = None,
        selection: VideoSelection | Mapping[str, Any] | None = None,
        source_metadata: Optional[Mapping[str, Any]] = None,
        tracker_metadata: Optional[Mapping[str, Any]] = None,
        video_id: Optional[str] = None,
    ) -> "CanonicalVideoWriter":
        """Open a video-local ``frames.jsonl`` stream in a partial bundle."""

        if not self.cfg.enabled:
            raise RuntimeError("Canonical V2 output is disabled")
        if source is None:
            source = source_path
        if source is None:
            raise ValueError("start_video requires source or source_path")
        parsed_fps = None if input_fps is None else _positive_dimension(input_fps, "input_fps")
        if selection is None:
            selected = _selection_from_window(frame_window or {}, input_fps=parsed_fps)
        elif isinstance(selection, VideoSelection):
            selected = selection
        else:
            selected = _selection_from_window(selection, input_fps=parsed_fps)
        stable_id = video_id or make_video_id(source, selected)
        if stable_id in self._active or any(row["video_id"] == stable_id for row in self._summaries):
            raise ValueError(f"duplicate Canonical V2 video_id {stable_id!r}")
        merged_source_metadata = dict(source_metadata or {})
        merged_source_metadata.update(
            {
                key: value
                for key, value in {
                    "width": width,
                    "height": height,
                    "input_fps": parsed_fps,
                    "source_frame_count": source_frame_count,
                }.items()
                if value is not None
            }
        )
        tracker = _tracking_metadata(tracking_config)
        if tracker_metadata:
            tracker.update(_json_compatible(dict(tracker_metadata)))
        writer = CanonicalVideoWriter(
            run=self,
            video_id=stable_id,
            source=source,
            source_path=source_path if source_path is not None else source,
            width=width,
            height=height,
            input_fps=parsed_fps,
            source_frame_count=source_frame_count,
            selection=selected,
            detection_fps=detection_fps,
            frame_interval=frame_interval,
            output_fps=output_fps,
            source_metadata=merged_source_metadata,
            tracker_metadata=tracker,
        )
        self._active[stable_id] = writer
        return writer

    def _register_finalized(self, writer: "CanonicalVideoWriter", summary: Mapping[str, Any]) -> None:
        self._active.pop(writer.video_id, None)
        self._summaries.append(_json_compatible(dict(summary)))

    def _register_aborted(self, writer: "CanonicalVideoWriter") -> None:
        self._active.pop(writer.video_id, None)

    def abort_active(self) -> None:
        """Abort every still-partial video bundle."""

        for writer in list(self._active.values()):
            writer.abort()

    def finish_manifest(self) -> Dict[str, Any]:
        """Atomically publish the run manifest after every bundle is final."""

        if self._active:
            raise RuntimeError("cannot finish Canonical V2 manifest with active video bundles")
        self.root.mkdir(parents=True, exist_ok=True)
        summaries = sorted(self._summaries, key=lambda row: row["video_id"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "producer": self.producer_metadata,
            "video_count": len(summaries),
            "frame_count": sum(int(row["frame_count"]) for row in summaries),
            "detection_count": sum(int(row["detection_count"]) for row in summaries),
            "track_count": sum(int(row["track_count"]) for row in summaries),
            "media_count": sum(1 for row in summaries if row.get("media_path") is not None),
            "videos": summaries,
            "artifacts": {"manifest": "manifest.json"},
        }
        atomic_write_json(self.root / "manifest.json", manifest)
        self._manifest = manifest
        return copy.deepcopy(manifest)

    def finalize_manifest(self) -> Dict[str, Any]:
        """Compatibility alias for ``finish_manifest``."""

        return self.finish_manifest()


class CanonicalVideoWriter:
    """Stream one selected video and atomically publish its four-file bundle."""

    def __init__(
        self,
        *,
        run: CanonicalRunWriter,
        video_id: str,
        source: Any,
        source_path: Any,
        width: Optional[int],
        height: Optional[int],
        input_fps: Optional[float],
        source_frame_count: Optional[int],
        selection: VideoSelection,
        detection_fps: Optional[float],
        frame_interval: int,
        output_fps: Optional[float],
        source_metadata: Mapping[str, Any],
        tracker_metadata: Mapping[str, Any],
    ) -> None:
        self.run = run
        self.video_id = str(video_id)
        self.source = str(source)
        self.source_path = source_path
        self.width = int(
            _positive_dimension(
                width if width is not None else source_metadata.get("width"),
                "video width",
            )
        )
        self.height = int(
            _positive_dimension(
                height if height is not None else source_metadata.get("height"),
                "video height",
            )
        )
        self.input_fps = input_fps
        self.source_frame_count = None if source_frame_count is None else int(source_frame_count)
        self.selection = selection
        self.detection_fps = _safe_float(detection_fps)
        self.frame_interval = max(1, int(frame_interval))
        self.output_fps = _safe_float(output_fps)
        self.source_metadata = _json_compatible(dict(source_metadata))
        self.tracker_metadata = _json_compatible(dict(tracker_metadata))
        self.partial_path = self.run.root / f"{self.video_id}.partial"
        self.final_path = self.run.root / self.video_id
        if self.partial_path.exists() or self.final_path.exists():
            raise FileExistsError(f"Canonical V2 bundle already exists for {self.video_id}")
        self.run.root.mkdir(parents=True, exist_ok=True)
        self.partial_path.mkdir(parents=False, exist_ok=False)
        self._frames_handle = (self.partial_path / "frames.jsonl").open("w", encoding="utf-8", newline="\n")
        self._trajectory = TrajectoryAccumulator(self.width, self.height)
        self._last_segment_index = -1
        self._last_source_index: Optional[int] = None
        self._last_source_timestamp: Optional[float] = None
        self._last_timestamp_delta: Optional[float] = None
        self._first_source_timestamp: Optional[float] = None
        self._first_segment_timestamp: Optional[float] = None
        self._last_segment_timestamp: Optional[float] = None
        self._timestamp_source_counts: Dict[str, int] = {}
        self._frame_count = 0
        self._detection_count = 0
        self._track_state_count = 0
        self._detection_ran_count = 0
        self._finalized = False
        self._aborted = False

    def _normalize_detections(self, detections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []
        output: List[Dict[str, Any]] = []
        id_to_name = _category_lookup(self.run.categories)
        for index, raw in enumerate(detections):
            if "bbox_xyxy_pixels" not in raw:
                return build_detection_rows_from_legacy(
                    detections,
                    self.run.categories,
                    self.width,
                    self.height,
                )
            category_id = int(raw.get("category_id", -1))
            output.append(
                build_detection_row(
                    detection_index=index,
                    category_id=category_id,
                    category_name=str(
                        raw.get(
                            "category_name",
                            id_to_name.get(category_id, category_id),
                        )
                    ),
                    score=raw.get("score", 0.0),
                    bbox_xyxy=raw["bbox_xyxy_pixels"],
                    frame_width=self.width,
                    frame_height=self.height,
                    attributes=raw.get("attributes", {}),
                )
            )
        return output

    def _normalize_states(
        self,
        raw_detections: Sequence[Mapping[str, Any]],
        track_states: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if not track_states:
            return build_observed_track_states(
                raw_detections,
                self.width,
                self.height,
                categories=self.run.categories,
            )
        if all("provenance" in state for state in track_states):
            return [
                build_track_state_row(
                    track_id=state.get("track_id"),
                    provenance=state.get("provenance", "observed"),
                    bbox_xyxy=state.get("bbox_xyxy_pixels"),
                    frame_width=self.width,
                    frame_height=self.height,
                    status=state.get("status"),
                    hits=state.get("hits"),
                    age_frames=state.get("age_frames"),
                    seconds_since_observed=state.get("seconds_since_observed"),
                    detection_index=state.get("detection_index"),
                    category_id=state.get("category_id"),
                    category_name=state.get("category_name"),
                    center_pixels=state.get("center_pixels"),
                    tracker_native=state.get("tracker_native", {}),
                )
                for state in track_states
            ]
        return build_tracker_state_rows(
            track_states,
            self.width,
            self.height,
            detections=raw_detections,
            categories=self.run.categories,
        )

    def _resolve_timestamps(
        self,
        *,
        source_frame_index: int,
        source_timestamp_seconds: Optional[Any],
        segment_timestamp_seconds: Optional[Any],
        timestamp_source: Optional[str],
    ) -> Tuple[Optional[float], Optional[float], str]:
        source_time = _optional_finite_float(source_timestamp_seconds, "source_timestamp_seconds")
        segment_time = _optional_finite_float(segment_timestamp_seconds, "segment_timestamp_seconds")
        provenance = (
            str(timestamp_source)
            if timestamp_source is not None
            else ("decoder" if source_time is not None else "nominal_fps_fallback")
        )
        if source_time is None and segment_time is not None:
            if self._first_source_timestamp is not None:
                source_time = self._first_source_timestamp + segment_time
            elif self.selection.start_seconds is not None:
                source_time = self.selection.start_seconds + segment_time
        if source_time is None and self.input_fps:
            source_time = source_frame_index / self.input_fps
            provenance = "nominal_fps_fallback"
        if (
            source_time is not None
            and self._last_source_timestamp is not None
            and source_time <= self._last_source_timestamp
        ):
            if self.input_fps:
                source_time = source_frame_index / self.input_fps
                provenance = "nominal_fps_fallback_non_monotonic"
            if source_time <= self._last_source_timestamp:
                source_time = self._last_source_timestamp + (1.0 / self.input_fps if self.input_fps else 1.0e-6)
                provenance = "monotonic_repair"
        if segment_time is None and source_time is not None:
            baseline = self._first_source_timestamp if self._first_source_timestamp is not None else source_time
            segment_time = source_time - baseline
        return source_time, segment_time, provenance

    def write_frame(
        self,
        segment_frame_index: int,
        source_frame_index: Optional[int] = None,
        *,
        source_timestamp_seconds: Optional[float] = None,
        segment_timestamp_seconds: Optional[float] = None,
        timestamp_source: Optional[str] = None,
        detection_ran: bool = True,
        detections: Sequence[Mapping[str, Any]] = (),
        track_states: Optional[Sequence[Mapping[str, Any]]] = None,
        camera_motion: Optional[Mapping[str, Any]] = None,
        cmc: Optional[Mapping[str, Any]] = None,
        cmc_affine: Optional[Sequence[Sequence[Any]]] = None,
        timestamp_seconds: Optional[float] = None,
        frame_index: Optional[int] = None,
        continuity_known: bool = True,
    ) -> Dict[str, Any]:
        """Write one decoded selected frame, including empty/skipped frames."""

        if self._finalized or self._aborted:
            raise RuntimeError("cannot write to a closed Canonical V2 bundle")
        segment_index = int(segment_frame_index)
        if segment_index != self._last_segment_index + 1:
            raise ValueError("segment_frame_index must be contiguous and start at zero")
        if source_frame_index is None:
            source_frame_index = frame_index
        if source_frame_index is None:
            source_frame_index = self.selection.start_frame + segment_index
        source_index = int(source_frame_index)
        if self._last_source_index is not None and source_index != self._last_source_index + 1:
            raise ValueError("source_frame_index must be contiguous")
        if source_timestamp_seconds is None:
            source_timestamp_seconds = timestamp_seconds
        source_time, segment_time, resolved_source = self._resolve_timestamps(
            source_frame_index=source_index,
            source_timestamp_seconds=source_timestamp_seconds,
            segment_timestamp_seconds=segment_timestamp_seconds,
            timestamp_source=timestamp_source,
        )
        raw_detections = [dict(row) for row in detections]
        canonical_detections = self._normalize_detections(raw_detections)
        canonical_states = self._normalize_states(raw_detections, track_states)
        scores = {int(row["detection_index"]): float(row["score"]) for row in canonical_detections}
        for state in canonical_states:
            state.pop("score", None)
            state.pop("confidence", None)
            state.pop("detector_score", None)
            detection_index = state.get("detection_index")
            if state.get("provenance") == "predicted" or detection_index not in scores:
                state["detection_index"] = None
                detection_index = None
            state["motion"] = self._trajectory.add_state(
                state,
                segment_frame_index=segment_index,
                source_frame_index=source_index,
                source_timestamp_seconds=source_time,
                segment_timestamp_seconds=segment_time,
                detector_score=(None if detection_index is None else scores[detection_index]),
                continuity_known=continuity_known,
            )
        canonical_states.sort(key=lambda row: str(row.get("track_id")))
        camera_input: Optional[Dict[str, Any]] = None
        if camera_motion or cmc or cmc_affine is not None:
            camera_input = dict(camera_motion or cmc or {})
            if cmc_affine is not None:
                camera_input["affine_2x3_pixels"] = cmc_affine
        requested_count = self.selection.requested_frame_count
        progress = None if requested_count is None else min(1.0, max(0.0, (segment_index + 1) / requested_count))
        row = {
            "schema_version": SCHEMA_VERSION,
            "video_id": self.video_id,
            "segment_frame_index": segment_index,
            "source_frame_index": source_index,
            "source_timestamp_seconds": source_time,
            "segment_timestamp_seconds": segment_time,
            "timestamp_source": resolved_source,
            "segment_progress": progress,
            "detection_ran": bool(detection_ran),
            "detections": canonical_detections,
            "track_states": canonical_states,
            "camera_motion": normalize_camera_motion(camera_input),
        }
        self._frames_handle.write(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        if source_time is not None:
            if self._first_source_timestamp is None:
                self._first_source_timestamp = source_time
            if self._last_source_timestamp is not None:
                delta = source_time - self._last_source_timestamp
                if delta > 0.0:
                    self._last_timestamp_delta = delta
            self._last_source_timestamp = source_time
        if self._first_segment_timestamp is None:
            self._first_segment_timestamp = segment_time
        self._last_segment_timestamp = segment_time
        self._last_segment_index = segment_index
        self._last_source_index = source_index
        self._frame_count += 1
        self._detection_count += len(canonical_detections)
        self._track_state_count += len(canonical_states)
        self._detection_ran_count += int(bool(detection_ran))
        self._timestamp_source_counts[resolved_source] = self._timestamp_source_counts.get(resolved_source, 0) + 1
        return copy.deepcopy(row)

    def _actual_selection(
        self,
        override: VideoSelection | Mapping[str, Any] | None = None,
    ) -> VideoSelection:
        if override is not None:
            if isinstance(override, VideoSelection):
                return override
            return _selection_from_window(override, input_fps=self.input_fps)
        end_frame = self.selection.start_frame + self._frame_count
        start_seconds = (
            self._first_source_timestamp if self._first_source_timestamp is not None else self.selection.start_seconds
        )
        frame_duration = self._last_timestamp_delta
        if frame_duration is None and self.input_fps:
            frame_duration = 1.0 / self.input_fps
        end_seconds = (
            None
            if self._last_source_timestamp is None or frame_duration is None
            else self._last_source_timestamp + frame_duration
        )
        if end_seconds is None:
            end_seconds = self.selection.end_seconds
        return VideoSelection(
            start_frame=self.selection.start_frame,
            end_frame_exclusive=end_frame,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

    def finalize(
        self,
        actual_metadata: Optional[Mapping[str, Any]] = None,
        source_path: Any = None,
        selected_range: VideoSelection | Mapping[str, Any] | None = None,
        annotated_output: Any = None,
        *,
        write_media: bool = True,
    ) -> Dict[str, Any]:
        """Validate, write metadata/tracks/media, then atomically publish."""

        if self._finalized:
            raise RuntimeError("Canonical V2 bundle is already finalized")
        if self._aborted:
            raise RuntimeError("Canonical V2 bundle was aborted")
        if self._frame_count <= 0:
            raise RuntimeError("cannot finalize an empty Canonical V2 bundle")
        actual = dict(actual_metadata or {})
        declared_frames = _field_value(
            actual,
            "actual_decoded_frame_count",
            "decoded_frame_count",
            default=None,
        )
        if declared_frames is not None and int(declared_frames) != self._frame_count:
            raise RuntimeError("actual decoded frame count does not match frames.jsonl")
        self._frames_handle.flush()
        os.fsync(self._frames_handle.fileno())
        self._frames_handle.close()
        actual_selection = self._actual_selection(selected_range)
        tracks = self._trajectory.track_rows(self.video_id)
        _write_jsonl(self.partial_path / "tracks.jsonl", tracks)
        media_source = source_path if source_path is not None else self.source_path
        fallback_probe = dict(self.source_metadata)
        fallback_probe.update(actual)
        source_probe = self.run.probe_fn(
            media_source,
            ffprobe_path=self.run.toolchain.ffprobe_path or "auto",
            ffmpeg_path=self.run.toolchain.ffmpeg_path or "auto",
            fallback_metadata=fallback_probe,
            runner=self.run.runner,
        )
        clean_probe = None
        media_path = None
        if write_media:
            media_target = self.partial_path / "media.mp4"
            clean_probe = self.run.transcode_fn(
                media_source,
                media_target,
                actual_selection,
                self.run.cfg.media,
                self.run.toolchain,
                expected_frames=self._frame_count,
                expected_width=self.width,
                expected_height=self.height,
                source_probe=source_probe,
                runner=self.run.runner,
                probe_fn=self.run.probe_fn,
            )
            media_path = f"{self.video_id}/media.mp4"
        clean_frame_count = None if clean_probe is None else (clean_probe.get("video", {}) or {}).get("frame_count")
        selected_duration = None
        if actual_selection.start_seconds is not None and actual_selection.end_seconds is not None:
            selected_duration = actual_selection.end_seconds - actual_selection.start_seconds
        effective_detection_fps = None
        if selected_duration and selected_duration > 0.0:
            effective_detection_fps = self._detection_ran_count / selected_duration
        source_video = dict(source_probe.get("video", {}) or {})
        source_video["decoded_width"] = source_video.get("decoded_width") or self.width
        source_video["decoded_height"] = source_video.get("decoded_height") or self.height
        source_video["display_width"] = source_video.get("display_width") or self.width
        source_video["display_height"] = source_video.get("display_height") or self.height
        source_video["source_frame_count"] = (
            self.source_frame_count if self.source_frame_count is not None else source_video.get("frame_count")
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "video_id": self.video_id,
            "source": {
                "uri": self.source,
                "local_path": (None if media_source is None else str(media_source)),
                "probe_backend": source_probe.get("probe_backend"),
                "format": source_probe.get("format", {}),
            },
            "selected_segment": {
                **actual_selection.to_dict(),
                "requested": self.selection.to_dict(),
                "actual_frame_count": self._frame_count,
                "first_decoded_timestamp_seconds": self._first_source_timestamp,
                "last_decoded_timestamp_seconds": self._last_source_timestamp,
            },
            "video": source_video,
            "processing": {
                "actual_decoded_frame_count": self._frame_count,
                "decoded_frame_width": self.width,
                "decoded_frame_height": self.height,
                "detection_ran_frame_count": self._detection_ran_count,
                "detection_skipped_frame_count": (self._frame_count - self._detection_ran_count),
                "detection_count": self._detection_count,
                "track_state_count": self._track_state_count,
                "track_count": len(tracks),
                "selected_duration_seconds": selected_duration,
                "detection_frame_interval": self.frame_interval,
                "requested_detection_fps": self.detection_fps,
                "effective_detection_fps": effective_detection_fps,
                "annotated_output_fps": self.output_fps,
                "clean_media_frame_count": clean_frame_count,
                "actual": _json_compatible(actual),
            },
            "audio": source_probe.get("audio", {}),
            "timestamps": {
                "preferred_source": "decoder_pts",
                "fallback_source": "source_frame_index/input_fps",
                "row_source_counts": dict(sorted(self._timestamp_source_counts.items())),
                "probe_backend": source_probe.get("probe_backend"),
            },
            "coordinate_convention": {
                "bbox_format": "xyxy",
                "pixel_edges": "continuous",
                "x2_y2": "exclusive_continuous_boundary",
                "pixel_coordinates": "raw_unclipped",
                "normalized_bbox_and_center": "clamped_0_1",
                "normalized_motion": "signed_unclamped_image_space",
                "origin": "top_left",
                "x_axis": "right",
                "y_axis": "down",
                "direction_degrees": "clockwise_from_positive_x",
                "physical_or_pitch_coordinates": False,
            },
            "categories": self.run.categories,
            "producer": self.run.producer_metadata,
            "detector": (
                self.run.producer_metadata.get("detector", {})
                if isinstance(self.run.producer_metadata, Mapping)
                else {}
            ),
            "tracker": self.tracker_metadata,
            "artifacts": {
                "metadata": "metadata.json",
                "frames": "frames.jsonl",
                "tracks": "tracks.jsonl",
                "media": "media.mp4" if write_media else None,
            },
            "annotated_output": (None if annotated_output is None else str(annotated_output)),
            "clean_media": (
                None
                if clean_probe is None
                else {
                    "path": "media.mp4",
                    "encoding": asdict(self.run.cfg.media),
                    "probe": clean_probe,
                }
            ),
        }
        atomic_write_json(self.partial_path / "metadata.json", metadata)
        if self.final_path.exists():
            raise FileExistsError(f"final Canonical V2 bundle already exists: {self.final_path}")
        os.replace(self.partial_path, self.final_path)
        self._finalized = True
        summary = {
            "schema_version": SCHEMA_VERSION,
            "video_id": self.video_id,
            "bundle_path": self.video_id,
            "metadata_path": f"{self.video_id}/metadata.json",
            "frames_path": f"{self.video_id}/frames.jsonl",
            "tracks_path": f"{self.video_id}/tracks.jsonl",
            "media_path": media_path,
            "frame_count": self._frame_count,
            "detection_count": self._detection_count,
            "track_count": len(tracks),
            "media_frame_count": clean_frame_count,
            "duration_seconds": selected_duration,
            "has_audio": bool((source_probe.get("audio", {}) or {}).get("has_audio")),
        }
        self.run._register_finalized(self, summary)
        return copy.deepcopy(summary)

    def abort(self) -> None:
        """Close and remove only this unpublished partial bundle."""

        if self._finalized or self._aborted:
            return
        if not self._frames_handle.closed:
            self._frames_handle.close()
        if self.partial_path.exists():
            shutil.rmtree(self.partial_path)
        self._aborted = True
        self.run._register_aborted(self)
