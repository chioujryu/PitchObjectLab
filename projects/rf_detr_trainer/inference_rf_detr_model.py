"""
Run RF-DETR inference on images, videos, folders, and HTTP(S) media URLs.

Usage:
    uv run python inference_rf_detr_model.py --config config/rf_detr_inference.yaml --yes

    uv run python inference_rf_detr_model.py \\
        --config config/rf_detr_inference.yaml \\
        --source /data/mixed_media \\
        --output-dir runs/rf_detr/inference_debug \\
        --yes

Notes:
    - The runner is RF-DETR only.
    - Folders may contain both images and videos.
    - Video detection can be sampled with inference.video.detection_fps.
    - Video inference can be limited by start/end time and max seconds.
    - Every output directory includes the config that created it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import rf_detr_cpu_runtime as cpu_runtime

_CPU_BOOTSTRAP_POLICY = (
    cpu_runtime.bootstrap_from_argv(
        Path(__file__).resolve().parent / "config" / "rf_detr_inference.yaml",
        "inference",
    )
    if __name__ in {"__main__", "__mp_main__"}
    else None
)

import colorama  # noqa: E402 - CPU bootstrap must run before numerical imports.
from colorama import Fore, Style  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from tqdm import tqdm  # noqa: E402

import rf_detr_runtime as trainer  # noqa: E402
import rf_detr_canonical_output as canonical_output  # noqa: E402
import rf_detr_video_tracking as video_tracking  # noqa: E402
from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator  # noqa: E402

colorama.init(autoreset=True)
if _CPU_BOOTSTRAP_POLICY is not None:
    cpu_runtime.apply_loaded_runtime(_CPU_BOOTSTRAP_POLICY)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config" / "rf_detr_inference.yaml"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
FOOTBALL_PREDICTIONS_FILENAME = "football_predictions.jsonl"
COLOR_PALETTE = [
    (239, 68, 68),
    (34, 197, 94),
    (59, 130, 246),
    (245, 158, 11),
    (168, 85, 247),
    (6, 182, 212),
    (244, 63, 94),
    (132, 204, 22),
    (14, 165, 233),
    (217, 70, 239),
]


@dataclass(frozen=True)
class SourceItem:
    source: str
    kind: str
    is_url: bool = False
    local_path: Optional[Path] = None


@dataclass(frozen=True)
class VideoFrameWindow:
    start_seconds: float
    end_seconds: Optional[float]
    max_seconds: Optional[float]
    effective_end_seconds: Optional[float]
    start_frame: int
    end_frame: Optional[int]
    output_frames: Optional[int]


@dataclass(frozen=True)
class FootballOutputConfig:
    """Resolved settings for the concise football-coordinate JSONL artifact."""

    enabled: bool
    target_class_ids: frozenset[int]


def load_yaml(path: Path) -> Dict[str, Any]:
    return trainer.load_yaml(path)


def apply_performance_profile(config: MutableMapping[str, Any], profile: str) -> None:
    """Apply stable runtime defaults without changing model architecture or slice geometry."""
    normalized = str(profile).strip().lower()
    if normalized not in {"safe", "fast"}:
        raise ValueError("performance profile must be safe or fast.")
    runtime = config.setdefault("runtime", {})
    runtime["performance_profile"] = normalized
    model = config.setdefault("model", {})
    optimization = model.setdefault("inference_optimization", {})
    inference = config.setdefault("inference", {})
    video = inference.setdefault("video", {})
    video["streaming"] = True
    tracking = inference.setdefault("tracking", {})
    hybrid = tracking.setdefault("hybrid", {})
    cmc = hybrid.setdefault("cmc", {})
    if normalized == "safe":
        optimization["backend"] = "pytorch"
        optimization.setdefault("pytorch", {})["precision"] = "bf16"
        cmc["processing_scale"] = 1.0
    else:
        optimization["backend"] = "tensorrt"
        optimization.setdefault("tensorrt", {})["precision"] = "fp16"
        cmc["processing_scale"] = 0.5


def apply_cli_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    configured_profile = getattr(args, "performance_profile", None) or (
        config.get("runtime", {}) or {}
    ).get("performance_profile")
    if configured_profile:
        apply_performance_profile(config, str(configured_profile))
    runtime = config.setdefault("runtime", {})
    cpu_runtime.apply_cpu_cli_overrides(config, args)
    output = config.setdefault("output", {})
    inference = config.setdefault("inference", {})
    model = config.setdefault("model", {})
    if args.yes:
        runtime["yes"] = True
        runtime["confirm_before_run"] = False
    if args.dry_run:
        runtime["dry_run"] = True
    if args.source:
        inference["sources"] = args.source
    if args.output_dir:
        output["output_dir"] = args.output_dir
    if args.checkpoint:
        model["pretrain_weights"] = args.checkpoint
    if args.device:
        model["device"] = args.device
    if args.confidence_threshold is not None:
        model["confidence_threshold"] = args.confidence_threshold
    tracknet_focus = getattr(args, "tracknet_focus", None)
    if tracknet_focus is not None:
        motion = model.setdefault("motion", {})
        motion.setdefault("focus", {})["mode"] = tracknet_focus
    optimization = model.setdefault("inference_optimization", {})
    backend_override = getattr(args, "inference_backend", None)
    if backend_override is not None:
        optimization["backend"] = backend_override
    precision_override = getattr(args, "inference_precision", None)
    if precision_override is not None:
        active_backend = str(optimization.get("backend", "pytorch")).strip().lower()
        optimization.setdefault(active_backend, {})["precision"] = precision_override
    tensorrt = optimization.setdefault("tensorrt", {})
    if getattr(args, "tensorrt_engine", None) is not None:
        tensorrt["engine_path"] = args.tensorrt_engine
        # CLI has no separate manifest flag; derive the adjacent project
        # sidecar for the CLI-selected engine instead of retaining YAML state.
        tensorrt["manifest_path"] = ""
    if getattr(args, "tensorrt_cache_dir", None) is not None:
        tensorrt["cache_dir"] = args.tensorrt_cache_dir
    if getattr(args, "tensorrt_force_rebuild", False):
        tensorrt["force_rebuild"] = True
    if args.max_sources is not None:
        inference["max_sources"] = args.max_sources
    if args.max_images is not None:
        inference["max_images"] = args.max_images
    if args.max_videos is not None:
        inference["max_videos"] = args.max_videos
    if args.batch_size is not None:
        inference["batch_size"] = args.batch_size
    if args.video_batch_size is not None:
        inference.setdefault("video", {})["batch_size"] = args.video_batch_size
    if getattr(args, "sahi_batch_size", None) is not None:
        config.setdefault("sahi", {})["batch_size"] = args.sahi_batch_size
    if getattr(args, "video_streaming", None) is not None:
        inference.setdefault("video", {})["streaming"] = bool(args.video_streaming)
    if getattr(args, "save_video", None) is not None:
        inference.setdefault("video", {})["save_video"] = bool(args.save_video)
    canonical_overrides = {
        "enabled": getattr(args, "canonical_output", None),
        "directory": getattr(args, "canonical_output_dir", None),
        "ffmpeg_path": getattr(args, "ffmpeg_path", None),
        "ffprobe_path": getattr(args, "ffprobe_path", None),
    }
    if any(value is not None for value in canonical_overrides.values()):
        canonical_output = inference.setdefault("canonical_output", {})
        for key, value in canonical_overrides.items():
            if value is not None:
                canonical_output[key] = bool(value) if key == "enabled" else value
    if getattr(args, "cmc_processing_scale", None) is not None:
        if not 0.0 < float(args.cmc_processing_scale) <= 1.0:
            raise ValueError("--cmc-processing-scale must be in the range (0, 1].")
        inference.setdefault("tracking", {}).setdefault("hybrid", {}).setdefault("cmc", {})[
            "processing_scale"
        ] = args.cmc_processing_scale
    if args.max_seconds is not None:
        inference.setdefault("video", {})["max_seconds"] = args.max_seconds
    if args.video_start_time is not None:
        inference.setdefault("video", {})["start_time"] = args.video_start_time
    if args.video_end_time is not None:
        inference.setdefault("video", {})["end_time"] = args.video_end_time
    if getattr(args, "no_track", False) or getattr(args, "track", False) \
            or getattr(args, "track_radius", None) is not None or getattr(args, "track_velocity", False) \
            or getattr(args, "tracker", None) is not None or getattr(args, "reid_weights", None) is not None:
        tracking = inference.setdefault("tracking", {})
        if getattr(args, "no_track", False):
            tracking["enabled"] = False
        elif getattr(args, "track", False):
            tracking["enabled"] = True
        if getattr(args, "tracker", None) is not None:
            tracking["algorithm"] = args.tracker
        if getattr(args, "track_radius", None) is not None:
            tracking["radius_pixels"] = args.track_radius
        if getattr(args, "track_velocity", False):
            tracking["use_velocity_prediction"] = True
        if getattr(args, "reid_weights", None) is not None:
            tracking["reid_weights"] = args.reid_weights


def config_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None and item != ""]
    return [value]


def positive_batch_size(value: Any, field_name: str, default: int) -> int:
    """Parse a positive batch size; all/null/empty inherit the provided default."""
    if value is None:
        return max(1, int(default))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "all", "none", "null"}:
            return max(1, int(default))
        value = trainer.parse_scalar(value)
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    return parsed


def inference_batch_size(config: Mapping[str, Any]) -> int:
    """Return image/source inference batch size."""
    return positive_batch_size(config.get("inference", {}).get("batch_size"), "inference.batch_size", 8)


def video_batch_size(config: Mapping[str, Any]) -> int:
    """Return video detection-frame batch size."""
    inference = config.get("inference", {})
    video_cfg = dict(inference.get("video", {}) or {})
    return positive_batch_size(video_cfg.get("batch_size"), "inference.video.batch_size", inference_batch_size(config))


def parse_video_time_seconds(
    value: Any,
    field_name: str,
    *,
    allow_all: bool = False,
    default: Optional[float] = None,
    positive: bool = False,
) -> Optional[float]:
    """Parse seconds, MM:SS, or HH:MM:SS values."""
    if value is None:
        return default
    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        if lower in {"", "none", "null"}:
            return default
        if lower == "all":
            if allow_all:
                return None
            raise ValueError(f"{field_name} must be a time value, got {value!r}.")
        if ":" in text:
            parts = text.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"{field_name} must use SS, MM:SS, or HH:MM:SS format, got {value!r}.")
            try:
                numbers = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{field_name} must use numeric SS, MM:SS, or HH:MM:SS format, got {value!r}.") from exc
            if any(number < 0 for number in numbers):
                raise ValueError(f"{field_name} must be non-negative, got {value!r}.")
            if len(numbers) == 2:
                minutes, seconds_part = numbers
                hours = 0.0
            else:
                hours, minutes, seconds_part = numbers
            if minutes >= 60 or seconds_part >= 60:
                raise ValueError(f"{field_name} MM:SS/HH:MM:SS minutes and seconds must be below 60, got {value!r}.")
            seconds = hours * 3600.0 + minutes * 60.0 + seconds_part
            if positive and seconds <= 0:
                raise ValueError(f"{field_name} must be positive when set, got {value!r}.")
            return seconds
        parsed = trainer.parse_scalar(text)
    if isinstance(parsed, str):
        text = parsed.strip().lower()
        if text in {"", "none", "null"}:
            return default
        if text == "all":
            if allow_all:
                return None
            raise ValueError(f"{field_name} must be a time value, got {value!r}.")
    if isinstance(parsed, bool):
        raise ValueError(f"{field_name} must be a time value, got {value!r}.")
    try:
        seconds = float(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be SS, MM:SS, HH:MM:SS, or a numeric seconds value, got {value!r}.") from exc
    if seconds < 0 or (positive and seconds <= 0):
        comparator = "positive" if positive else "non-negative"
        raise ValueError(f"{field_name} must be {comparator}, got {value!r}.")
    return seconds


def parse_seconds_limit(value: Any, field_name: str = "inference.video.max_seconds") -> Optional[float]:
    """Parse a positive duration limit; null/all/empty means the whole selected video range."""
    try:
        return parse_video_time_seconds(value, field_name, allow_all=True, default=None, positive=True)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be 'all', null, SS, MM:SS, HH:MM:SS, or a positive seconds value, got {value!r}."
        ) from exc


def video_frame_window(frame_count: int, input_fps: float, video_cfg: Mapping[str, Any]) -> VideoFrameWindow:
    """Resolve configured video start/end/max_seconds into frame bounds."""
    fps = max(0.001, float(input_fps or 0.0))
    total_frames = max(0, int(frame_count or 0))
    start_seconds = parse_video_time_seconds(video_cfg.get("start_time", 0), "inference.video.start_time", default=0.0) or 0.0
    end_seconds = parse_video_time_seconds(video_cfg.get("end_time", "all"), "inference.video.end_time", allow_all=True, default=None)
    max_seconds = parse_seconds_limit(video_cfg.get("max_seconds"))
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("inference.video.end_time must be greater than inference.video.start_time.")

    effective_end_seconds = end_seconds
    if max_seconds is not None:
        max_end_seconds = start_seconds + max_seconds
        effective_end_seconds = min(effective_end_seconds, max_end_seconds) if effective_end_seconds is not None else max_end_seconds

    start_frame = max(0, int(math.floor(start_seconds * fps)))
    if total_frames > 0:
        start_frame = min(start_frame, total_frames)

    end_frame: Optional[int]
    if effective_end_seconds is None:
        end_frame = total_frames if total_frames > 0 else None
    else:
        end_frame = max(0, int(math.ceil(effective_end_seconds * fps)))
        if total_frames > 0:
            end_frame = min(end_frame, total_frames)
        end_frame = max(start_frame, end_frame)
    output_frames = None if end_frame is None else max(0, end_frame - start_frame)
    return VideoFrameWindow(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        max_seconds=max_seconds,
        effective_end_seconds=effective_end_seconds,
        start_frame=start_frame,
        end_frame=end_frame,
        output_frames=output_frames,
    )


def limited_video_frame_total(frame_count: int, input_fps: float, max_seconds: Optional[float]) -> Optional[int]:
    """Return the maximum number of frames to process for a video."""
    if max_seconds is None:
        return frame_count if frame_count > 0 else None
    fps = max(0.001, float(input_fps or 0.0))
    seconds_frames = max(1, int(math.ceil(max_seconds * fps)))
    return min(frame_count, seconds_frames) if frame_count > 0 else seconds_frames


def video_detection_frame_count(output_frames: int, input_fps: float, detection_fps: Any) -> int:
    """Estimate how many frames will run model prediction."""
    if output_frames <= 0:
        return 0
    if detection_fps is None:
        return output_frames
    frame_interval = max(1, int(round(max(0.001, float(input_fps or 0.0)) / max(0.001, float(detection_fps)))))
    return int(math.ceil(output_frames / frame_interval))


def estimate_video_work(item: SourceItem, video_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Estimate processed/output video frames without running inference."""
    input_fps = 30.0
    frame_count = 0
    width = 1920
    height = 1080
    metadata_source = "fallback"
    if item.local_path and item.local_path.exists():
        with contextlib.suppress(Exception):
            import cv2

            capture = cv2.VideoCapture(str(item.local_path))
            if capture.isOpened():
                input_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
                metadata_source = "local-video"
            capture.release()
    window = video_frame_window(frame_count, input_fps, video_cfg)
    output_frames = window.output_frames
    if output_frames is None:
        if window.effective_end_seconds is not None:
            fallback_seconds = max(0.0, window.effective_end_seconds - window.start_seconds)
        else:
            fallback_seconds = window.max_seconds if window.max_seconds is not None else 60.0
        output_frames = max(1, int(math.ceil(fallback_seconds * input_fps)))
    detection_frames = video_detection_frame_count(output_frames, input_fps, video_cfg.get("detection_fps"))
    return {
        "start_seconds": window.start_seconds,
        "end_seconds": window.end_seconds if window.end_seconds is not None else "all",
        "max_seconds": window.max_seconds if window.max_seconds is not None else "all",
        "effective_end_seconds": window.effective_end_seconds if window.effective_end_seconds is not None else "all",
        "start_frame": window.start_frame,
        "end_frame": window.end_frame if window.end_frame is not None else "all",
        "output_frames": output_frames,
        "detection_frames": detection_frames,
        "input_fps": input_fps,
        "source_num_frames": frame_count,
        "source_duration_seconds": (frame_count / input_fps) if frame_count > 0 else None,
        "width": width,
        "height": height,
        "metadata_source": metadata_source,
    }


def estimated_rfdetr_workload(
    width: int,
    height: int,
    units: int,
    config: Mapping[str, Any],
) -> Dict[str, int]:
    """Estimate actual model inputs/batches for full-image or SAHI inference."""
    unit_count = max(0, int(units))
    inference = dict(config.get("inference", {}) or {})
    mode = str(inference.get("mode", "full_image")).strip().lower()
    if mode != "sahi":
        batch = inference_batch_size(config)
        return {
            "source_units": unit_count,
            "slice_inputs": 0,
            "standard_inputs": unit_count,
            "recheck_input_cap": 0,
            "model_inputs": unit_count,
            "model_batches": int(math.ceil(unit_count / max(1, batch))) if unit_count else 0,
        }
    sahi = dict(config.get("sahi", {}) or {})
    windows = evaluator.shared_modes.generate_slice_windows_for_size(
        width=max(1, int(width)),
        height=max(1, int(height)),
        slice_width=int(sahi.get("slice_width", width) or width),
        slice_height=int(sahi.get("slice_height", height) or height),
        overlap_width_ratio=float(sahi.get("overlap_width_ratio", 0.2)),
        overlap_height_ratio=float(sahi.get("overlap_height_ratio", 0.2)),
    )
    raw_batch = sahi.get("batch_size", 16)
    batch = 16 if isinstance(raw_batch, str) and raw_batch.strip().lower() == "auto" else max(1, int(raw_batch or 16))
    slices_per_unit = len(windows)
    standard_per_unit = int(bool(sahi.get("standard_prediction", True)))
    recheck = dict(sahi.get("recheck", {}) or {})
    recheck_cap_per_unit = (
        max(0, int(recheck.get("max_rechecks_per_image", 50) or 0))
        if bool(recheck.get("enabled", False))
        else 0
    )
    primary_batches = math.ceil(slices_per_unit / batch) + standard_per_unit
    return {
        "source_units": unit_count,
        "slice_inputs": unit_count * slices_per_unit,
        "standard_inputs": unit_count * standard_per_unit,
        "recheck_input_cap": unit_count * recheck_cap_per_unit,
        "model_inputs": unit_count * (slices_per_unit + standard_per_unit),
        # Recheck is content-dependent; report its cap separately and avoid a
        # worst-case estimate that historically overstates the common path.
        "model_batches": unit_count * primary_batches,
    }


def media_kind_from_suffix(suffix: str, image_exts: Sequence[str], video_exts: Sequence[str]) -> Optional[str]:
    normalized = suffix.lower()
    if normalized in {ext.lower() for ext in image_exts}:
        return "image"
    if normalized in {ext.lower() for ext in video_exts}:
        return "video"
    return None


def is_url(value: str) -> bool:
    return trainer.is_url_like(value)


def discover_sources(config: Mapping[str, Any]) -> List[SourceItem]:
    inference = config.get("inference", {})
    image_exts = [str(ext).lower() for ext in (inference.get("image_extensions") or sorted(IMAGE_EXTENSIONS))]
    video_exts = [str(ext).lower() for ext in (inference.get("video_extensions") or sorted(VIDEO_EXTENSIONS))]
    recursive = bool(inference.get("recursive", True))
    items: List[SourceItem] = []
    for raw in config_list(inference.get("sources")):
        source = str(raw).strip()
        if not source:
            continue
        if is_url(source):
            suffix = Path(urlparse(source).path).suffix.lower()
            kind = media_kind_from_suffix(suffix, image_exts, video_exts) or "video"
            items.append(SourceItem(source=source, kind=kind, is_url=True))
            continue
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_dir():
            iterator: Iterable[Path] = path.rglob("*") if recursive else path.glob("*")
            for file_path in sorted(candidate for candidate in iterator if candidate.is_file()):
                kind = media_kind_from_suffix(file_path.suffix, image_exts, video_exts)
                if kind:
                    items.append(SourceItem(source=str(file_path), kind=kind, local_path=file_path))
        elif path.is_file():
            kind = media_kind_from_suffix(path.suffix, image_exts, video_exts)
            if kind:
                items.append(SourceItem(source=str(path), kind=kind, local_path=path))
        else:
            raise FileNotFoundError(f"Inference source does not exist: {source}")
    return items


def apply_source_limits(items: Sequence[SourceItem], config: Mapping[str, Any]) -> List[SourceItem]:
    """Apply first-N source limits from inference config."""
    inference = config.get("inference", {})
    max_images = trainer.parse_limit_value(inference.get("max_images"), "inference.max_images")
    max_videos = trainer.parse_limit_value(inference.get("max_videos"), "inference.max_videos")
    max_sources = trainer.parse_limit_value(inference.get("max_sources"), "inference.max_sources")
    image_count = 0
    video_count = 0
    per_type_limited: List[SourceItem] = []
    for item in items:
        if item.kind == "image":
            if max_images is not None and image_count >= max_images:
                continue
            image_count += 1
        elif item.kind == "video":
            if max_videos is not None and video_count >= max_videos:
                continue
            video_count += 1
        per_type_limited.append(item)
    return per_type_limited[:max_sources] if max_sources is not None else per_type_limited


def build_output_dir(config: Mapping[str, Any], timestamp: str) -> Path:
    output = config.get("output", {})
    exact = str(output.get("output_dir") or "").strip()
    if exact:
        return trainer.resolve_path_for_output(trainer.render_timestamped(exact, timestamp))
    root = trainer.render_timestamped(output.get("root", "runs/rf_detr/inference"), timestamp)
    name = trainer.render_timestamped(output.get("name", "rfdetr_inference_{timestamp}"), timestamp)
    return trainer.resolve_path_for_output(str(Path(str(root)) / str(name)))


def estimate_outputs(items: Sequence[SourceItem], output_dir: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    image_count = sum(1 for item in items if item.kind == "image")
    video_count = sum(1 for item in items if item.kind == "video")
    video_cfg = dict((config.get("inference", {}).get("video", {}) or {}))
    video_work = [estimate_video_work(item, video_cfg) for item in items if item.kind == "video"]
    detection_frames = sum(int(work["detection_frames"]) for work in video_work)
    output_frames = sum(int(work["output_frames"]) for work in video_work)
    video_model_work = [
        estimated_rfdetr_workload(
            int(work["width"]),
            int(work["height"]),
            int(work["detection_frames"]),
            config,
        )
        for work in video_work
    ]
    image_model_work: List[Dict[str, int]] = []
    for item in items:
        if item.kind != "image":
            continue
        width = height = int(config.get("model", {}).get("resolution") or 640)
        if item.local_path is not None:
            with contextlib.suppress(Exception), Image.open(item.local_path) as image:
                width, height = image.size
        image_model_work.append(estimated_rfdetr_workload(width, height, 1, config))
    combined_model_work = image_model_work + video_model_work
    model_inputs = sum(work["model_inputs"] for work in combined_model_work)
    model_batches = sum(work["model_batches"] for work in combined_model_work)
    slice_inputs = sum(work["slice_inputs"] for work in combined_model_work)
    recheck_input_cap = sum(work["recheck_input_cap"] for work in combined_model_work)
    start_seconds = parse_video_time_seconds(video_cfg.get("start_time", 0), "inference.video.start_time", default=0.0) or 0.0
    end_seconds = parse_video_time_seconds(video_cfg.get("end_time", "all"), "inference.video.end_time", allow_all=True, default=None)
    max_seconds = parse_seconds_limit(video_cfg.get("max_seconds"))
    local_bytes = 0
    for item in items:
        if item.local_path and item.local_path.exists():
            local_bytes += item.local_path.stat().st_size
    save_video = bool(video_cfg.get("save_video", True))
    output_files = 5 + image_count + (video_count if save_video else 0)
    if bool(config.get("inference", {}).get("save_predictions_jsonl", True)):
        output_files += 1
    if bool(
        ((config.get("inference", {}).get("football_output", {}) or {}).get("enabled", True))
    ):
        output_files += 1
    if bool((config.get("inference", {}).get("tracking", {}) or {}).get("enabled", False)):
        output_files += 1  # tracking_summary.json
    canonical_cfg = dict((config.get("inference", {}).get("canonical_output", {}) or {}))
    canonical_enabled = (
        bool(canonical_cfg.get("enabled", True))
        and video_count > 0
        and not trainer.temporal_motion_enabled(config)
    )
    canonical_estimated_bytes = 0
    if canonical_enabled:
        # One run manifest plus metadata/frames/tracks/clean media per video.
        output_files += 1 + 4 * video_count
        for work in video_work:
            frame_total = max(0, int(work["output_frames"]))
            width = max(1, int(work["width"]))
            height = max(1, int(work["height"]))
            # JSONL includes an empty-frame base row even when no detector runs.
            json_bytes = frame_total * 640
            # CRF output is content-dependent. This pixel-rate estimate is intentionally rough.
            media_bytes = max(1_000_000, int(width * height * frame_total * 0.02))
            canonical_estimated_bytes += json_bytes + media_bytes + 100_000
    tensorrt_artifacts = trainer.estimate_tensorrt_cache_artifacts(config)
    output_files += int(tensorrt_artifacts["file_count"])
    estimated_bytes = (
        max(local_bytes, output_files * 500_000)
        + canonical_estimated_bytes
        + int(tensorrt_artifacts["bytes"])
    )
    estimate = {
        "output_dir": str(output_dir),
        "sources": len(items),
        "image_sources": image_count,
        "video_sources": video_count,
        "url_sources": sum(1 for item in items if item.is_url),
        "video_start_seconds": start_seconds,
        "video_end_seconds": end_seconds if end_seconds is not None else "all",
        "video_max_seconds": max_seconds if max_seconds is not None else "all",
        "estimated_video_detection_frames": detection_frames,
        "estimated_video_output_frames": output_frames,
        "estimated_model_inputs": model_inputs,
        "estimated_model_batches": model_batches,
        "estimated_sahi_slice_inputs": slice_inputs,
        "estimated_recheck_input_cap": recheck_input_cap,
        "canonical_v2": {
            "enabled": canonical_enabled,
            "video_bundles": video_count if canonical_enabled else 0,
            "estimated_bytes": canonical_estimated_bytes,
            "estimated_disk_usage": trainer.format_bytes(canonical_estimated_bytes),
        },
        "tensorrt_cache": tensorrt_artifacts,
        "estimated_total_files": output_files,
        "estimated_disk_usage": trainer.format_bytes(estimated_bytes),
        "note": "URL/rendered/canonical-media sizes and first-run TensorRT artifacts are rough content-dependent estimates.",
    }
    settings = trainer.runtime_time_estimate_settings(config)
    per_frame_media_seconds = trainer.positive_float_setting(
        settings, "default_video_render_seconds_per_frame"
    )
    render_seconds = output_frames * per_frame_media_seconds if save_video else 0.0
    canonical_media_seconds = output_frames * per_frame_media_seconds if canonical_enabled else 0.0
    estimate["canonical_v2"]["estimated_transcode_seconds"] = canonical_media_seconds
    trainer.add_runtime_estimate(
        estimate=estimate,
        config=config,
        output_dir=output_dir,
        task="inference",
        runtime_units=float(model_batches),
        default_rate_key="default_inference_seconds_per_image",
        basis={
            "image_sources": image_count,
            "video_sources": video_count,
            "video_detection_frames": detection_frames,
            "video_output_frames": output_frames,
            "video_work": video_work,
            "model_work": combined_model_work,
        },
        extra_seconds=(
            render_seconds
            + canonical_media_seconds
            + float(tensorrt_artifacts.get("estimated_build_seconds", 0) or 0)
        ),
    )
    return estimate


def confirm_or_exit(estimate: Mapping[str, Any], verbose: bool, assume_yes: bool) -> None:
    if verbose:
        print(Fore.BLUE + Style.BRIGHT + "Output and resource estimate before RF-DETR inference:")
    print(json.dumps(dict(estimate), indent=2, ensure_ascii=False))
    if assume_yes:
        if verbose:
            print(Fore.BLUE + Style.BRIGHT + "Confirmation skipped because --yes or confirm_before_run=false is enabled.")
        return
    answer = input(Fore.BLUE + Style.BRIGHT + "Continue and start inference? [y/N]: " + Style.RESET_ALL).strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted before inference output was produced.")


def class_names_from_config(config: Mapping[str, Any]) -> Dict[int, str]:
    dataset = config.get("dataset", {})
    names = dataset.get("class_names", dataset.get("names", []))
    if not names and dataset.get("data_yaml"):
        data_yaml = Path(str(dataset["data_yaml"])).expanduser()
        if not data_yaml.is_absolute():
            data_yaml = (PROJECT_DIR / data_yaml).resolve()
        if data_yaml.exists():
            data = load_yaml(data_yaml)
            names = data.get("names", [])
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def build_categories(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    names = class_names_from_config(config)
    num_classes = config.get("model", {}).get("num_classes")
    if num_classes is not None:
        for index in range(int(num_classes)):
            names.setdefault(index, str(index))
    return [{"id": int(index), "name": name} for index, name in sorted(names.items())]


def parse_football_output_config(
    config: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
) -> FootballOutputConfig:
    """Resolve and validate the independently configured football output classes."""
    raw = dict((config.get("inference", {}) or {}).get("football_output", {}) or {})
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("inference.football_output.enabled must be true or false.")
    if not enabled:
        return FootballOutputConfig(enabled=False, target_class_ids=frozenset())

    id_to_name = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in categories
    }
    available = ", ".join(f"{category_id}={name}" for category_id, name in sorted(id_to_name.items()))
    available = available or "(none)"

    raw_ids = config_list(raw.get("target_class_ids"))
    if raw_ids:
        if any(isinstance(value, bool) for value in raw_ids):
            raise ValueError("inference.football_output.target_class_ids must contain integer category IDs.")
        try:
            target_ids = {int(value) for value in raw_ids}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "inference.football_output.target_class_ids must contain integer category IDs."
            ) from exc
        unknown_ids = sorted(target_ids.difference(id_to_name))
        if unknown_ids:
            raise ValueError(
                "inference.football_output.target_class_ids contains unknown category IDs "
                f"{unknown_ids}; available categories: {available}."
            )
        return FootballOutputConfig(enabled=True, target_class_ids=frozenset(target_ids))

    requested_names = [str(value).strip() for value in config_list(raw.get("target_class_names"))]
    if not requested_names:
        requested_names = ["football"]
    name_to_id = {name.casefold(): category_id for category_id, name in id_to_name.items()}
    missing_names = [name for name in requested_names if name.casefold() not in name_to_id]
    if missing_names:
        raise ValueError(
            "inference.football_output.target_class_names contains unknown category names "
            f"{missing_names}; available categories: {available}. "
            "Set target_class_ids explicitly when the dataset uses numeric-only class names."
        )
    return FootballOutputConfig(
        enabled=True,
        target_class_ids=frozenset(name_to_id[name.casefold()] for name in requested_names),
    )


def coco_xywh_to_center_xywh(bbox: Sequence[Any]) -> List[float]:
    """Convert absolute COCO top-left xywh to absolute center-point xywh."""
    if len(bbox) < 4:
        raise ValueError(f"Expected a four-value COCO bbox, got {bbox!r}.")
    x, y, width, height = (float(value) for value in bbox[:4])
    return [x + width / 2.0, y + height / 2.0, width, height]


def xyxy_to_center_xywh(box: Sequence[Any]) -> List[float]:
    """Convert an absolute xyxy box to absolute center-point xywh."""
    if len(box) < 4:
        raise ValueError(f"Expected a four-value xyxy box, got {box!r}.")
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    width = x2 - x1
    height = y2 - y1
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height]


def _football_prediction_row(
    *,
    kind: str,
    source: Any,
    image_id: Any,
    frame_index: Any,
    timestamp_seconds: Any,
    category_id: int,
    category_name: str,
    score: Any,
    xywh: Sequence[Any],
    track_id: Any,
) -> Dict[str, Any]:
    """Build the stable, compact football output schema."""
    return {
        "kind": str(kind),
        "source": None if source is None else str(source),
        "image_id": None if image_id is None else int(image_id),
        "frame_index": None if frame_index is None else int(frame_index),
        "timestamp_seconds": None if timestamp_seconds is None else float(timestamp_seconds),
        "category_id": int(category_id),
        "category_name": str(category_name),
        "score": float(score),
        "xywh": [float(value) for value in xywh[:4]],
        "track_id": None if track_id is None else int(track_id),
    }


def build_standard_football_rows(
    predictions: Sequence[Mapping[str, Any]],
    football_config: FootballOutputConfig,
    categories: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Select final image/video football detections after tracking and convert their boxes."""
    if not football_config.enabled:
        return []
    id_to_name = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in categories
    }
    rows: List[Dict[str, Any]] = []
    for prediction in predictions:
        category_id = int(prediction.get("category_id", -1))
        if category_id not in football_config.target_class_ids:
            continue
        is_video = prediction.get("frame_index") is not None
        rows.append(
            _football_prediction_row(
                kind="video" if is_video else "image",
                source=prediction.get("source"),
                image_id=prediction.get("image_id"),
                frame_index=prediction.get("frame_index"),
                timestamp_seconds=prediction.get("timestamp_seconds"),
                category_id=category_id,
                category_name=id_to_name[category_id],
                score=prediction.get("score", 0.0),
                xywh=coco_xywh_to_center_xywh(prediction.get("bbox", [])),
                track_id=prediction.get("track_id"),
            )
        )
    return rows


def filter_confirmed_hybrid_exports(
    predictions: Sequence[Mapping[str, Any]],
    tracking_config: Optional[Any],
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter only formal Hybrid video detections when explicitly requested.

    Image detections and non-target classes are outside the Hybrid tracker and
    remain untouched.  Diagnostic state rows never pass through this helper.
    """
    enabled = bool(
        tracking_config is not None
        and getattr(tracking_config, "enabled", False)
        and getattr(tracking_config, "algorithm", "circle") == "hybrid"
        and getattr(tracking_config, "export_confirmed_only", False)
    )
    published: List[Dict[str, Any]] = []
    suppressed = 0
    for prediction in predictions:
        row = dict(prediction)
        is_video = row.get("frame_index") is not None
        is_target = (
            enabled
            and is_video
            and int(row.get("category_id", -1)) in tracking_config.target_class_ids
        )
        if is_target and not bool(row.get("track_final_confirmed", False)):
            suppressed += 1
            continue
        published.append(row)
    return published, suppressed


def build_temporal_football_rows(
    temporal_rows: Sequence[Mapping[str, Any]],
    football_config: FootballOutputConfig,
    categories: Sequence[Mapping[str, Any]],
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    """Convert RF-DETR temporal absolute-xyxy detections for each anchor frame."""
    if not football_config.enabled:
        return []
    id_to_name = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in categories
    }
    rows: List[Dict[str, Any]] = []
    for temporal_row in temporal_rows:
        detections = dict(temporal_row.get("detections", {}) or {})
        boxes = list(detections.get("boxes", []) or [])
        scores = list(detections.get("scores", []) or [])
        labels = list(detections.get("labels", []) or [])
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError(
                "Temporal detection boxes, scores, and labels must have equal lengths, got "
                f"{len(boxes)}, {len(scores)}, and {len(labels)}."
            )
        for box, score_value, label_value in zip(boxes, scores, labels):
            category_id = int(label_value)
            score = float(score_value)
            if category_id not in football_config.target_class_ids or score < confidence_threshold:
                continue
            rows.append(
                _football_prediction_row(
                    kind="temporal",
                    source=temporal_row.get("source"),
                    image_id=None,
                    frame_index=temporal_row.get("anchor_frame_index"),
                    timestamp_seconds=None,
                    category_id=category_id,
                    category_name=id_to_name[category_id],
                    score=score,
                    xywh=xyxy_to_center_xywh(box),
                    track_id=None,
                )
            )
    return rows


def class_color(category_id: int) -> Tuple[int, int, int]:
    if category_id < len(COLOR_PALETTE):
        return COLOR_PALETTE[category_id]
    value = (int(category_id) * 2654435761) & 0xFFFFFF
    return 64 + value % 160, 64 + (value >> 8) % 160, 64 + (value >> 16) % 160


def track_color(track_id: int) -> Tuple[int, int, int]:
    """Stable, distinct color per track id for trajectory overlays."""
    value = ((int(track_id) + 1) * 2246822519) & 0xFFFFFF
    return 48 + value % 180, 48 + (value >> 8) % 180, 48 + (value >> 16) % 180


def color_map(categories: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> Dict[int, Tuple[int, int, int]]:
    ids = {int(category["id"]) for category in categories}
    ids.update(int(prediction.get("category_id", 0)) for prediction in predictions)
    return {category_id: class_color(category_id) for category_id in sorted(ids)}


def resolve_render_ids(config: Mapping[str, Any], categories: Sequence[Mapping[str, Any]]) -> List[int]:
    inference = config.get("inference", {})
    ids = [int(value) for value in config_list(inference.get("render_class_ids"))]
    if ids:
        return ids
    names = [str(value).casefold() for value in config_list(inference.get("render_class_names"))]
    if not names:
        return []
    name_to_id = {str(category.get("name", category["id"])).casefold(): int(category["id"]) for category in categories}
    return [name_to_id[name] for name in names if name in name_to_id]


def draw_track_overlays(
    draw: Any,
    tracks: Sequence[Any],
    tracking_cfg: Any,
    target_ids: Sequence[int],
    current_frame_index: Optional[int] = None,
    confirmed_track_ids: Optional[Sequence[int]] = None,
) -> None:
    """Draw trajectory trails, optional search circles, and current-center dots for visible tracks."""
    base_color = class_color(int(target_ids[0])) if target_ids else class_color(1)
    width = max(1, int(tracking_cfg.trajectory_width))
    final_ids = {int(value) for value in (confirmed_track_ids or ())}
    for track in tracks:
        if not video_tracking.is_track_visible(
            track,
            current_frame_index,
            tracking_cfg,
            final_confirmed=int(track.track_id) in final_ids,
        ):
            continue
        color = track_color(track.track_id) if tracking_cfg.trajectory_per_track_color else base_color
        # Live position is used by the optional predicted head/search gate.  The
        # observed dot below is intentionally anchored to the latest real hit.
        live_x, live_y = video_tracking.live_center(track, current_frame_index, tracking_cfg)
        # Historical observed points (age-filtered and linearly bridged), optionally plus the predicted live head.
        xy = video_tracking.trail_points(track, current_frame_index, tracking_cfg)
        if getattr(tracking_cfg, "draw_predicted_trajectory", False) and (not xy or xy[-1] != (live_x, live_y)):
            xy = xy + [(live_x, live_y)]
        if tracking_cfg.draw_trajectory and len(xy) >= 2:
            if tracking_cfg.trajectory_taper:
                count = len(xy)
                for index in range(1, count):
                    segment_width = max(1, int(round(width * (index + 1) / count)))
                    draw.line([xy[index - 1], xy[index]], fill=color, width=segment_width)
            else:
                draw.line(xy, fill=color, width=width)
        if tracking_cfg.draw_search_circle:
            radius = video_tracking.effective_radius(track, tracking_cfg)
            draw.ellipse([live_x - radius, live_y - radius, live_x + radius, live_y + radius], outline=color, width=1)
        draw_observed = video_tracking.should_draw_observed_center(
            track, current_frame_index, tracking_cfg
        )
        draw_predicted = video_tracking.should_draw_predicted_center(
            track, current_frame_index, tracking_cfg
        )
        if draw_observed or draw_predicted:
            center_x, center_y = (track.center_x, track.center_y) if draw_observed else (live_x, live_y)
            # Trajectory width affects only trajectory lines.  Center markers
            # are kept visually stable across presets.
            dot_radius = 3
            draw.ellipse(
                [
                    center_x - dot_radius,
                    center_y - dot_radius,
                    center_x + dot_radius,
                    center_y + dot_radius,
                ],
                fill=color,
            )


def hybrid_packet_confirmed_ids(packet: Mapping[str, Any]) -> frozenset[int]:
    """Return the source-local final-confirmation registry attached at commit time."""
    return frozenset(int(value) for value in packet.get("confirmed_track_ids", ()))


def hybrid_packet_detections(packet: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Copy committed detections and annotate their final confirmation outcome."""
    confirmed_ids = hybrid_packet_confirmed_ids(packet)
    rows: List[Dict[str, Any]] = []
    for prediction in packet.get("detections", ()):
        row = dict(prediction)
        track_id = row.get("track_id")
        final_confirmed = track_id is not None and int(track_id) in confirmed_ids
        row["track_final_confirmed"] = bool(final_confirmed)
        rows.append(row)
    return rows


def prediction_visible_for_confirmed_render(
    prediction: Mapping[str, Any],
    tracking_cfg: Optional[Any],
    confirmed_track_ids: Optional[Sequence[int]],
) -> bool:
    """Apply Hybrid's render-only confirmation filter to target detections."""
    if (
        tracking_cfg is None
        or not getattr(tracking_cfg, "enabled", False)
        or getattr(tracking_cfg, "algorithm", "circle") != "hybrid"
        or not getattr(tracking_cfg, "render_confirmed_only", False)
    ):
        return True
    if int(prediction.get("category_id", -1)) not in tracking_cfg.target_class_ids:
        return True
    track_id = prediction.get("track_id")
    if track_id is None:
        return False
    if confirmed_track_ids is not None:
        return int(track_id) in {int(value) for value in confirmed_track_ids}
    return bool(prediction.get("track_final_confirmed", prediction.get("track_confirmed", False)))


def draw_predictions(
    image: Image.Image,
    predictions: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    render_ids: Sequence[int],
    tracks: Optional[Sequence[Any]] = None,
    tracking_cfg: Optional[Any] = None,
    current_frame_index: Optional[int] = None,
    confirmed_track_ids: Optional[Sequence[int]] = None,
) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    id_to_name = {int(category["id"]): str(category.get("name", category["id"])) for category in categories}
    render_set = set(int(value) for value in render_ids)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    has_tracked_rows = any(prediction.get("track_id") is not None for prediction in predictions)
    tracking_active = tracking_cfg is not None and getattr(tracking_cfg, "enabled", False) and (bool(tracks) or has_tracked_rows)
    confirmed_render_ids = {
        int(value) for value in (confirmed_track_ids or ())
    } if (
        tracking_cfg is not None
        and getattr(tracking_cfg, "algorithm", "circle") == "hybrid"
        and getattr(tracking_cfg, "render_confirmed_only", False)
    ) else set()
    if tracking_active and tracks:
        overlay_tracks = list(tracks)
        if confirmed_render_ids:
            overlay_tracks = [
                track for track in overlay_tracks if int(track.track_id) in confirmed_render_ids
            ]
        elif (
            getattr(tracking_cfg, "algorithm", "circle") == "hybrid"
            and getattr(tracking_cfg, "render_confirmed_only", False)
        ):
            overlay_tracks = []
        draw_track_overlays(
            draw,
            overlay_tracks,
            tracking_cfg,
            sorted(tracking_cfg.target_class_ids),
            current_frame_index,
            confirmed_render_ids,
        )
    for prediction in predictions:
        category_id = int(prediction.get("category_id", 0))
        if render_set and category_id not in render_set:
            continue
        if not prediction_visible_for_confirmed_render(prediction, tracking_cfg, confirmed_track_ids):
            continue
        x, y, width, height = [float(value) for value in prediction.get("bbox", [0, 0, 0, 0])[:4]]
        color = class_color(category_id)
        draw.rectangle([x, y, x + width, y + height], outline=color, width=2)
        label = f"{id_to_name.get(category_id, category_id)} {float(prediction.get('score', 0.0)):.2f}"
        if (
            tracking_active
            and tracking_cfg.label_track_id
            and prediction.get("track_id") is not None
            and (
                prediction.get("track_confirmed")
                or (
                    bool(confirmed_render_ids)
                    and int(prediction["track_id"]) in confirmed_render_ids
                )
            )
        ):
            label = f"{id_to_name.get(category_id, category_id)} #{int(prediction['track_id'])} {float(prediction.get('score', 0.0)):.2f}"
        text_bbox = draw.textbbox((x + 2, y + 2), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luminance > 155 else (255, 255, 255)
        draw.text((x + 2, y + 2), label, fill=text_color, font=font)
    return canvas


def build_prediction_config(config: Mapping[str, Any], categories: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    model = config.get("model", {})
    inference = config.get("inference", {})
    mode = str(inference.get("mode", "full_image")).strip().lower()
    return {
        "model": {
            "type": "rfdetr",
            "confidence_threshold": float(model.get("confidence_threshold", 0.25)),
            "image_size": model.get("resolution"),
            "category_remapping": model.get("category_remapping", {}),
            "inference_optimization": dict(model.get("inference_optimization", {}) or {}),
        },
        "inference": {"mode": mode, "use_sahi": mode == "sahi", "batch_size": inference_batch_size(config)},
        "test_mode": {"mode": mode},
        "sahi": dict(config.get("sahi", {}) or {}),
        "crop": dict(config.get("crop", {}) or {}),
        "dataset_categories": list(categories),
        "output": {"visual_format": "jpg"},
        "progress": {"slices": False},
    }


def final_prediction_confidence_threshold(config: Mapping[str, Any]) -> Optional[float]:
    """Return the final SAHI+recheck output threshold, or None when inactive."""
    inference = dict(config.get("inference", {}) or {})
    if str(inference.get("mode", "full_image")).strip().lower() != "sahi":
        return None
    sahi = dict(config.get("sahi", {}) or {})
    recheck = dict(sahi.get("recheck", {}) or {})
    if not bool(recheck.get("enabled", False)):
        return None
    model = dict(config.get("model", {}) or {})
    return float(recheck.get("fused_confidence_threshold", model.get("confidence_threshold", 0.25)))


def filter_final_inference_predictions(
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Apply the shared final confidence gate used by inference outputs and renders."""
    threshold = final_prediction_confidence_threshold(config)
    if threshold is None:
        return [dict(prediction) for prediction in predictions]
    return [
        dict(prediction)
        for prediction in predictions
        if float(prediction.get("score", 0.0)) >= threshold
    ]


def record_inference_timing_rows(model: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """Attach evaluator timing rows to the predictor for the final run summary."""
    collected = getattr(model, "_rf_detr_inference_timing_rows", None)
    if not isinstance(collected, list):
        collected = []
        try:
            setattr(model, "_rf_detr_inference_timing_rows", collected)
        except (AttributeError, TypeError):
            # Some lightweight third-party predictors (and test doubles) do not
            # expose an instance dictionary. Prediction must remain usable even
            # when optional run-level timing cannot be attached to that object.
            return
    collected.extend(dict(row) for row in rows if isinstance(row, Mapping))


def record_video_pipeline_timing(model: Any, **values: Any) -> None:
    """Accumulate video-only wall stages and workload counters on the predictor."""
    state = getattr(model, "_rf_detr_video_pipeline_timing", None)
    if not isinstance(state, dict):
        state = {}
        try:
            setattr(model, "_rf_detr_video_pipeline_timing", state)
        except (AttributeError, TypeError):
            return
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if key.endswith(("_peak", "_batch_size", "_vram_used_bytes")) or key.startswith("peak_"):
            state[key] = max(float(state.get(key, 0.0) or 0.0), float(value))
        else:
            state[key] = float(state.get(key, 0.0) or 0.0) + float(value)


def cuda_memory_telemetry_for_model(model: Any) -> Dict[str, int]:
    """Return CUDA memory telemetry for the model's actual device.

    ``torch.cuda.max_memory_allocated()`` without a device silently queries the
    current/default GPU.  That produced a false zero whenever inference used a
    non-default device such as ``cuda:2``.
    """

    empty = {
        "peak_vram_bytes": 0,
        "peak_vram_reserved_bytes": 0,
        "device_vram_used_bytes": 0,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return empty
        device: Any = None
        with contextlib.suppress(Exception):
            device = trainer.get_inference_acceleration_handle(model).device
        if device is None:
            device = getattr(model, "device", None)
        if device is None:
            device = getattr(getattr(model, "model", None), "device", None)
        if device is None:
            return empty
        resolved = device if isinstance(device, torch.device) else torch.device(device)
        if resolved.type != "cuda":
            return empty
        result = dict(empty)
        with contextlib.suppress(Exception):
            result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(resolved))
        with contextlib.suppress(Exception):
            result["peak_vram_reserved_bytes"] = int(torch.cuda.max_memory_reserved(resolved))
        with contextlib.suppress(Exception):
            free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)
            # This device-level value includes TensorRT allocations that the
            # PyTorch caching allocator cannot observe.  On a shared GPU it may
            # also include other processes, so it is reported separately.
            result["device_vram_used_bytes"] = max(0, int(total_bytes) - int(free_bytes))
        return result
    except Exception:
        # Timing telemetry must never make an otherwise successful inference
        # run fail on a third-party predictor or a CPU-only torch build.
        return empty


def peak_vram_bytes_for_model(model: Any) -> int:
    """Backward-compatible peak PyTorch allocation accessor."""

    return cuda_memory_telemetry_for_model(model)["peak_vram_bytes"]


def summarize_inference_timing_rows(model: Any) -> Dict[str, Any]:
    """Aggregate per-image evaluator timing into stable stage totals and ratios."""
    rows = getattr(model, "_rf_detr_inference_timing_rows", [])
    if not isinstance(rows, list):
        rows = []

    def total(key: str) -> float:
        return sum(float(row.get(key, 0.0) or 0.0) for row in rows if isinstance(row, Mapping))

    evaluator_elapsed = total("elapsed_seconds")
    model_forward = total("model_forward_seconds")
    base_forward = total("base_model_forward_seconds")
    sahi_forward = total("sahi_model_forward_seconds")
    recheck_forward = total("recheck_model_forward_seconds")
    preprocess = total("preprocess_seconds")
    postprocess = total("postprocess_seconds")
    crop = total("crop_seconds")
    host_preprocess = total("host_preprocess_seconds")
    device_preprocess = total("device_preprocess_seconds")
    h2d = total("h2d_seconds")
    resize_normalize = total("resize_normalize_seconds")
    orchestration = total("orchestration_seconds")
    autotune = total("autotune_seconds")
    exclusive_postprocess = total("exclusive_postprocess_seconds")
    video_timing = getattr(model, "_rf_detr_video_pipeline_timing", {})
    if not isinstance(video_timing, Mapping):
        video_timing = {}
    video_wall = float(video_timing.get("video_pipeline_wall_seconds", 0.0) or 0.0)
    video_evaluator = float(video_timing.get("video_evaluator_seconds", 0.0) or 0.0)
    elapsed = evaluator_elapsed + max(0.0, video_wall - video_evaluator)
    slice_inputs = sum(int(row.get("slice_count", 0) or 0) for row in rows if isinstance(row, Mapping))
    recheck_inputs = sum(
        int(
            dict(row.get("sahi_recheck", {}) or {}).get(
                "rechecked",
                dict(row.get("sahi_recheck", {}) or {}).get("candidate_count", 0),
            )
            or 0
        )
        for row in rows
        if isinstance(row, Mapping)
    )
    workload = getattr(model, "_rf_detr_workload_counters", {})
    if not isinstance(workload, Mapping):
        workload = {}
    slice_inputs = int(workload.get("slice_inputs", slice_inputs) or 0)
    recheck_inputs = int(workload.get("recheck_inputs", recheck_inputs) or 0)
    model_batches = int(workload.get("model_batches", 0) or 0)

    def positive_batch_values(key: str) -> List[int]:
        values: set[int] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            raw = row.get(key)
            candidates = raw if isinstance(raw, (list, tuple, set)) else [raw]
            for candidate in candidates:
                if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                    values.add(candidate)
        return sorted(values)

    summary = {
        "images_or_frames": len(rows),
        # Timing history for SAHI inference is normalized by actual model
        # batches, matching estimate_outputs().  Falling back to frames keeps
        # external/legacy predictors usable when workload counters are absent.
        "runtime_units": model_batches if model_batches > 0 else len(rows),
        "total_seconds": elapsed,
        "evaluator_seconds": evaluator_elapsed,
        "model_forward_seconds": model_forward,
        "base_model_forward_seconds": base_forward,
        "sahi_model_forward_seconds": sahi_forward,
        "recheck_model_forward_seconds": recheck_forward,
        "preprocess_seconds": preprocess,
        "crop_seconds": crop,
        "host_preprocess_seconds": host_preprocess,
        "h2d_seconds": h2d,
        "resize_normalize_seconds": resize_normalize,
        "device_preprocess_seconds": device_preprocess,
        "h2d_resize_normalize_seconds": h2d + resize_normalize,
        "orchestration_seconds": orchestration,
        "autotune_seconds": autotune,
        "postprocess_seconds": postprocess,
        "exclusive_postprocess_seconds": exclusive_postprocess,
        "model_forward_ratio": model_forward / elapsed if elapsed > 0 else 0.0,
        "sahi_model_forward_ratio": sahi_forward / elapsed if elapsed > 0 else 0.0,
        "recheck_model_forward_ratio": recheck_forward / elapsed if elapsed > 0 else 0.0,
        "source_frames": int(video_timing.get("source_frames", 0) or 0),
        "detection_frames": int(video_timing.get("detection_frames", 0) or 0),
        "slice_inputs": slice_inputs,
        "recheck_inputs": recheck_inputs,
        "model_batches": model_batches,
        "model_inputs": int(workload.get("model_inputs", 0) or 0),
        "oom_retries": int(workload.get("oom_retries", 0) or 0),
        "slice_batches": int(workload.get("slice_batches", 0) or 0),
        "recheck_batches": int(workload.get("recheck_batches", 0) or 0),
        "standard_batches": int(workload.get("standard_batches", 0) or 0),
        "requested_sahi_batch_sizes": positive_batch_values("requested_slice_batch_size"),
        "effective_sahi_batch_sizes": positive_batch_values("effective_slice_batch_sizes")
        or positive_batch_values("slice_batch_size"),
        "observed_sahi_batch_sizes": positive_batch_values("observed_slice_batch_sizes"),
    }
    for key in (
        "video_pipeline_wall_seconds",
        "decode_seconds",
        "frame_conversion_seconds",
        "video_evaluator_seconds",
        "tracker_seconds",
        "render_seconds",
        "encode_seconds",
        "serialization_seconds",
        "frame_queue_peak",
        "peak_vram_bytes",
        "peak_vram_reserved_bytes",
        "device_vram_used_bytes",
        "outer_batch_size",
    ):
        summary[key] = float(video_timing.get(key, 0.0) or 0.0)
    return summary


def load_rfdetr_model(config: Mapping[str, Any]) -> Any:
    model_cls = trainer.get_model_class(str(config.get("model", {}).get("size", "medium")))
    rf_model = model_cls(**trainer.build_model_kwargs(config))
    p2_config = config.get('model', {}).get('p2', {}) or {}
    if bool(p2_config.get('enabled', False)):
        from rf_detr_p2 import assert_p2_checkpoint_compatible

        assert_p2_checkpoint_compatible(
            rf_model.model,
            getattr(rf_model.model_config, 'pretrain_weights', None),
            trainer.build_pitchobjectlab_architecture(config, rf_model.model_config),
        )
    motion_config = config.get("model", {}).get("motion", {}) or {}
    if trainer.motion_module_enabled(config):
        from rf_detr_motion import (
            assert_motion_checkpoint_compatible,
            attach_motion_module,
            load_motion_checkpoint_weights,
        )

        attach_motion_module(rf_model.model, motion_config)
        model_config = getattr(rf_model, "model_config", None)
        checkpoint_path = getattr(model_config, "pretrain_weights", None)
        assert_motion_checkpoint_compatible(
            rf_model.model,
            checkpoint_path,
            trainer.build_pitchobjectlab_architecture(config, model_config),
        )
        load_motion_checkpoint_weights(rf_model.model, checkpoint_path)
    accelerated_model, _ = trainer.configure_rfdetr_inference_acceleration(
        rf_model,
        config,
        device=str(config.get("model", {}).get("device", "auto")),
    )
    return accelerated_model


def resolved_tracker_device(config: Mapping[str, Any], tracking_config: Any) -> str:
    """Resolve a concrete boxmot/ReID device string from tracking.reid_device or model.device."""
    raw = getattr(tracking_config, "reid_device", None) or config.get("model", {}).get("device")
    normalized = trainer.normalize_model_constructor_device(raw)
    if normalized:
        return normalized
    # auto/None: pick the best available device.
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def warn_reid_in_restricted_region(tracking_config: Any) -> None:
    """Best-effort warning before boxmot ReID auto-download in CN/HK/MO/TW (AGENTS rule 8).

    Only triggers for appearance trackers (deepocsort, or botsort with ReID on) that have no
    local reid_weights, since those are the cases that download from Google Drive via gdown.
    """
    algorithm = getattr(tracking_config, "algorithm", "circle")
    needs_reid = algorithm == "deepocsort" or (
        algorithm == "botsort" and getattr(tracking_config, "botsort_with_reid", True)
    )
    if not needs_reid or getattr(tracking_config, "reid_weights", None):
        return
    try:
        import importlib.util

        script_path = PROJECT_DIR / "scripts" / "setup_pytorch_uv.py"
        spec = importlib.util.spec_from_file_location("rf_detr_setup_pytorch_uv", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        region, _provider = module.detect_region()
        greater_china = getattr(module, "GREATER_CHINA_REGIONS", {"CN", "HK", "MO", "TW"})
    except Exception:
        return
    if region in greater_china:
        print(
            Fore.YELLOW
            + Style.BRIGHT
            + f"Warning: tracking.algorithm={algorithm} downloads ReID weights from Google Drive (gdown), "
            + f"which is unreliable in your region ({region}). Set inference.tracking.reid_weights to a local "
            + "osnet_x0_25_msmt17.pt path, or use algorithm: ocsort (or botsort_with_reid: false) to avoid it."
        )


def create_tracker(
    tracking_config: Optional[Any],
    tracker_device: str = "cpu",
    frame_size: Optional[Tuple[int, int]] = None,
) -> Optional[Any]:
    """Build the configured tracker, or None when tracking is disabled.

    algorithm 'circle' -> built-in FootballTracker; 'hybrid' -> delayed hybrid tracker;
    'ocsort'/'deepocsort'/'botsort'/'bytetrack' -> the boxmot adapter. Optional tracker
    modules are imported lazily only when their algorithm is selected.
    """
    if tracking_config is None or not tracking_config.enabled:
        return None
    if getattr(tracking_config, "algorithm", "circle") == "circle":
        return video_tracking.FootballTracker(tracking_config)
    if getattr(tracking_config, "algorithm", "circle") == "hybrid":
        from rf_detr_hybrid_tracker import HybridFootballTracker, HybridTrackingConfig

        hybrid_cfg = HybridTrackingConfig.from_mapping(getattr(tracking_config, "hybrid_options", {}))
        return HybridFootballTracker(
            hybrid_cfg,
            target_class_ids=tracking_config.target_class_ids,
            history_maxlen=tracking_config.trajectory_max_points or 300,
            confirmation_backfill=True,
        )
    import rf_detr_boxmot_tracker as boxmot_tracking


    if getattr(tracking_config, "reid_half", False) and not boxmot_tracking.effective_reid_half(tracking_config, tracker_device):
        print(
            Fore.YELLOW
            + Style.BRIGHT
            + f"Warning: reid_half disabled on non-CUDA device ({tracker_device}); FP16 ReID is GPU-only."
        )
    return boxmot_tracking.BoxmotTracker(tracking_config, device=tracker_device, frame_size=frame_size)


def download_url(item: SourceItem, cache_dir: Path) -> SourceItem:
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(item.source).path).suffix or (".mp4" if item.kind == "video" else ".jpg")
    target = cache_dir / f"url_{abs(hash(item.source))}{suffix}"
    with requests.get(item.source, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return SourceItem(source=item.source, kind=item.kind, is_url=True, local_path=target)


def predict_image_file(
    item: SourceItem,
    image_id: int,
    model: Any,
    prediction_config: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
    output_dir: Path,
    render_ids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Path]:
    assert item.local_path is not None
    with Image.open(item.local_path) as image:
        width, height = image.size
    record = evaluator.ImageRecord(image_id=image_id, file_name=item.local_path.name, path=str(item.local_path), width=width, height=height)
    predictions, timing, _ = evaluator.predict_image(record, model, prediction_config, output_dir, save_visual=False)
    record_inference_timing_rows(model, [timing])
    predictions = filter_final_inference_predictions(predictions, prediction_config)
    with Image.open(item.local_path) as image:
        rendered = draw_predictions(image, predictions, categories, render_ids)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / f"{trainer.sanitize_name(item.local_path.stem)}_pred.jpg"
    rendered.save(target, quality=92)
    return predictions, target


def prediction_config_with_batch(prediction_config: Mapping[str, Any], batch_size: int) -> Dict[str, Any]:
    """Return a shallow prediction config copy with an RF-DETR batch size."""
    configured = dict(prediction_config)
    configured["inference"] = dict(configured.get("inference", {}) or {})
    configured["inference"]["batch_size"] = max(1, int(batch_size))
    configured["sahi"] = dict(configured.get("sahi", {}) or {})
    requested_sahi_batch = configured["sahi"].get("batch_size", batch_size)
    if isinstance(requested_sahi_batch, str) and requested_sahi_batch.strip().lower() == "auto":
        configured["sahi"]["batch_size"] = "auto"
    else:
        configured["sahi"]["batch_size"] = max(1, int(requested_sahi_batch or batch_size))
    return configured


def predict_image_files_batch(
    items: Sequence[SourceItem],
    start_image_id: int,
    model: Any,
    prediction_config: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
    output_dir: Path,
    render_ids: Sequence[int],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Predict and render image sources in RF-DETR batches."""
    records: List[evaluator.ImageRecord] = []
    for offset, item in enumerate(items):
        assert item.local_path is not None
        with Image.open(item.local_path) as image:
            width, height = image.size
        records.append(
            evaluator.ImageRecord(
                image_id=start_image_id + offset,
                file_name=item.local_path.name,
                path=str(item.local_path),
                width=width,
                height=height,
            )
        )
    batch_config = prediction_config_with_batch(prediction_config, batch_size)
    predictions_by_image, timing_rows, _ = evaluator.predict_images_rfdetr(records, model, batch_config, output_dir)
    record_inference_timing_rows(model, timing_rows)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    for item, predictions in zip(items, predictions_by_image):
        assert item.local_path is not None
        predictions = filter_final_inference_predictions(predictions, batch_config)
        with Image.open(item.local_path) as image:
            rendered = draw_predictions(image, predictions, categories, render_ids)
        target = image_dir / f"{trainer.sanitize_name(item.local_path.stem)}_pred.jpg"
        rendered.save(target, quality=92)
        for prediction in predictions:
            row = dict(prediction)
            row["source"] = item.source
            all_rows.append(row)
        outputs.append({"source": item.source, "kind": "image", "output": str(target), "predictions": len(predictions)})
    return all_rows, outputs, start_image_id + len(items)


def build_video_row(
    prediction: Mapping[str, Any],
    source: str,
    absolute_frame_index: int,
    segment_frame_index: int,
    input_fps: float,
    frame_window: Any,
) -> Dict[str, Any]:
    """Attach source, frame index, and timestamp metadata to a video prediction row."""
    row = dict(prediction)
    row["source"] = source
    row["frame_index"] = absolute_frame_index
    row["segment_frame_index"] = segment_frame_index
    row["timestamp_seconds"] = absolute_frame_index / input_fps
    row["segment_timestamp_seconds"] = segment_frame_index / input_fps
    row["video_start_seconds"] = frame_window.start_seconds
    row["video_end_seconds"] = frame_window.end_seconds
    row["video_effective_end_seconds"] = frame_window.effective_end_seconds
    return row


def decoded_frame_timestamp(
    capture: Any,
    cv2_module: Any,
    absolute_frame_index: int,
    input_fps: float,
    previous_timestamp: Optional[float],
) -> Tuple[float, str]:
    """Return a monotonic decoder timestamp with an explicit nominal fallback."""
    nominal = float(absolute_frame_index) / max(0.001, float(input_fps))
    raw_milliseconds = float(capture.get(cv2_module.CAP_PROP_POS_MSEC) or 0.0)
    decoded = raw_milliseconds / 1000.0
    valid = math.isfinite(decoded) and decoded >= 0.0
    if absolute_frame_index > 0 and decoded == 0.0:
        valid = False
    if previous_timestamp is not None and decoded <= previous_timestamp:
        valid = False
    if valid:
        return decoded, "decoder_pts"
    if previous_timestamp is not None:
        nominal = max(nominal, previous_timestamp + 1.0 / max(0.001, float(input_fps)))
    return nominal, "nominal_fps"


def canonical_track_states_for_frame(
    frame_predictions: Sequence[Mapping[str, Any]],
    tracker: Any,
    tracking_config: Optional[Any],
    categories: Sequence[Mapping[str, Any]],
    absolute_frame_index: int,
    input_fps: float,
) -> List[Dict[str, Any]]:
    """Build tracker-agnostic observed states plus reliable circle predictions.

    BoxMOT adapters intentionally expose only associated observations because
    their unmatched internal state is not part of the stable adapter API.
    """
    id_to_name = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in categories
    }
    states: List[Dict[str, Any]] = []
    observed_track_ids: set[int] = set()
    for detection_index, prediction in enumerate(frame_predictions):
        track_id = prediction.get("track_id")
        if track_id is None:
            continue
        track_id = int(track_id)
        observed_track_ids.add(track_id)
        bbox = [float(value) for value in prediction.get("bbox", [])[:4]]
        if len(bbox) == 4:
            center = [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0]
        else:
            center = [
                float(prediction.get("track_center_x") or 0.0),
                float(prediction.get("track_center_y") or 0.0),
            ]
        category_id = int(prediction.get("category_id", -1))
        confirmed = bool(
            prediction.get("track_final_confirmed", prediction.get("track_confirmed", False))
        )
        states.append(
            {
                "track_id": track_id,
                "category_id": category_id,
                "category_name": id_to_name.get(category_id, str(category_id)),
                "status": prediction.get("track_status") or ("confirmed" if confirmed else "tentative"),
                "observation": "observed",
                "bbox": bbox if len(bbox) == 4 else None,
                "center": center,
                "detection_index": detection_index,
                "hits": prediction.get("track_hits"),
                "age_frames": prediction.get("track_age_frames"),
                "seconds_since_observed": 0.0,
                "final_confirmed": confirmed,
            }
        )

    if (
        tracker is None
        or tracking_config is None
        or str(getattr(tracking_config, "algorithm", "")).casefold() != "circle"
    ):
        return states
    target_ids = sorted(int(value) for value in getattr(tracking_config, "target_class_ids", set()))
    category_id = target_ids[0] if len(target_ids) == 1 else None
    for track in getattr(tracker, "tracks", ()):  # Circle exposes a stable stdlib state object.
        track_id = int(track.track_id)
        if track_id in observed_track_ids or int(track.last_seen_frame_index) >= absolute_frame_index:
            continue
        elapsed_frames = absolute_frame_index - int(track.last_seen_frame_index)
        center_x, center_y = video_tracking.predicted_center(track, tracking_config, elapsed_frames)
        confirmed = int(track.hits) >= int(getattr(tracking_config, "min_hits", 1))
        states.append(
            {
                "track_id": track_id,
                "category_id": category_id,
                "category_name": id_to_name.get(category_id) if category_id is not None else None,
                "status": "lost" if confirmed else "tentative",
                "observation": "predicted",
                "bbox": None,
                "center": [float(center_x), float(center_y)],
                "detection_index": None,
                "hits": int(track.hits),
                "age_frames": absolute_frame_index - int(track.first_frame_index) + 1,
                "seconds_since_observed": elapsed_frames / max(0.001, float(input_fps)),
                "final_confirmed": confirmed,
            }
        )
    return states

def is_hybrid_tracker(tracker: Any) -> bool:
    """Whether a tracker uses delayed step/flush commits."""
    return tracker is not None and tracker.__class__.__name__ == "HybridFootballTracker"


def collect_hybrid_committed(
    packets: Sequence[Mapping[str, Any]],
    all_predictions: List[Dict[str, Any]],
    tracking_state_rows: Optional[List[Dict[str, Any]]],
    source: str,
    input_fps: float,
    frame_window: VideoFrameWindow,
    canonical_video_writer: Optional[Any] = None,
    canonical_frame_meta: Optional[MutableMapping[int, Mapping[str, Any]]] = None,
) -> None:
    """Attach video metadata only after hybrid frames are committed."""
    for packet in packets:
        absolute_frame_index = int(packet["frame_index"])
        segment_frame_index = absolute_frame_index - frame_window.start_frame
        for prediction in hybrid_packet_detections(packet):
            all_predictions.append(
                build_video_row(
                    prediction,
                    source,
                    absolute_frame_index,
                    segment_frame_index,
                    input_fps,
                    frame_window,
                )
            )
        if tracking_state_rows is not None:
            for state in packet.get("track_states", []):
                row = dict(state)
                row.update(
                    source=source,
                    frame_index=absolute_frame_index,
                    segment_frame_index=segment_frame_index,
                    timestamp_seconds=absolute_frame_index / input_fps,
                    segment_timestamp_seconds=segment_frame_index / input_fps,
                    cmc=dict(packet.get("cmc", {})),
                )
                if packet.get("hypothesis"):
                    row["hypothesis"] = dict(packet["hypothesis"])
                tracking_state_rows.append(row)
        if canonical_video_writer is not None:
            if canonical_frame_meta is None:
                raise RuntimeError("Hybrid Canonical V2 export requires frame metadata")
            meta = canonical_frame_meta.pop(absolute_frame_index, None)
            if meta is None:
                raise RuntimeError(
                    f"Hybrid tracker committed frame {absolute_frame_index} without Canonical metadata"
                )
            cmc = dict(packet.get("cmc", {}))
            cmc_affine = packet.get("cmc_affine")
            camera_motion = None
            if cmc or cmc_affine is not None:
                camera_motion = {
                    **cmc,
                    "affine_previous_to_current": cmc_affine,
                }
            canonical_video_writer.write_frame(
                segment_frame_index=segment_frame_index,
                source_frame_index=absolute_frame_index,
                source_timestamp_seconds=float(meta["source_timestamp_seconds"]),
                timestamp_source=str(meta["timestamp_source"]),
                detection_ran=bool(meta["detection_ran"]),
                detections=hybrid_packet_detections(packet),
                track_states=list(packet.get("track_states", [])),
                camera_motion=camera_motion,
            )


def consume_hybrid_committed(
    packets: Sequence[Mapping[str, Any]],
    *,
    frame_buffer: MutableMapping[int, Tuple[Any, bool]],
    writer: Any,
    all_predictions: List[Dict[str, Any]],
    tracking_state_rows: Optional[List[Dict[str, Any]]],
    source: str,
    input_fps: float,
    frame_window: VideoFrameWindow,
    categories: Sequence[Mapping[str, Any]],
    render_ids: Sequence[int],
    tracking_config: Any,
    canonical_video_writer: Optional[Any] = None,
    canonical_frame_meta: Optional[MutableMapping[int, Mapping[str, Any]]] = None,
) -> Tuple[float, float]:
    """Publish and render committed Hybrid packets against their original frames.

    Returns ``(render_seconds, encode_seconds)`` so the streaming pipeline can
    preserve its stage telemetry.  A frame is removed from the buffer only when
    the tracker commits the matching immutable packet.
    """
    import cv2

    collect_hybrid_committed(
        packets,
        all_predictions,
        tracking_state_rows,
        source,
        input_fps,
        frame_window,
        canonical_video_writer,
        canonical_frame_meta,
    )
    render_seconds = 0.0
    encode_seconds = 0.0
    if writer is None:
        return render_seconds, encode_seconds
    for packet in packets:
        absolute_frame_index = int(packet["frame_index"])
        buffered = frame_buffer.pop(absolute_frame_index, None)
        if buffered is None:
            raise RuntimeError(
                f"Hybrid tracker committed frame {absolute_frame_index} without its buffered source frame"
            )
        frame, should_render = buffered
        if should_render:
            render_started = time.perf_counter()
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rendered: Optional[Image.Image] = None
            try:
                confirmed_ids = hybrid_packet_confirmed_ids(packet)
                rendered = draw_predictions(
                    pil_frame,
                    hybrid_packet_detections(packet),
                    categories,
                    render_ids,
                    packet.get("track_snapshots", ()),
                    tracking_config,
                    absolute_frame_index,
                    confirmed_ids,
                )
                frame = cv2.cvtColor(np_image(rendered), cv2.COLOR_RGB2BGR)
            finally:
                if rendered is not None and rendered is not pil_frame:
                    rendered.close()
                pil_frame.close()
            render_seconds += time.perf_counter() - render_started
        encode_started = time.perf_counter()
        writer.write(frame)
        encode_seconds += time.perf_counter() - encode_started
    return render_seconds, encode_seconds



def predict_video_file_one_pass(
    item: SourceItem,
    start_image_id: int,
    model: Any,
    prediction_config: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
    output_dir: Path,
    render_ids: Sequence[int],
    video_cfg: Mapping[str, Any],
    tracking_config: Optional[Any] = None,
    tracker_device: str = "cpu",
    tracking_state_rows: Optional[List[Dict[str, Any]]] = None,
    canonical_run: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Path], int]:
    import cv2

    assert item.local_path is not None
    capture = cv2.VideoCapture(str(item.local_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {item.local_path}")
    input_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    output_fps = float(video_cfg.get("output_fps") or input_fps)
    detection_fps = video_cfg.get("detection_fps")
    frame_interval = 1
    if detection_fps is not None:
        frame_interval = max(1, int(round(input_fps / max(0.001, float(detection_fps)))))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_window = video_frame_window(frame_count, input_fps, video_cfg)
    frame_limit = frame_window.output_frames
    try:
        canonical_video = (
            canonical_run.start_video(
                source=item.source,
                source_path=item.local_path,
                width=width,
                height=height,
                input_fps=input_fps,
                source_frame_count=frame_count,
                frame_window=frame_window,
                detection_fps=detection_fps,
                frame_interval=frame_interval,
                output_fps=output_fps,
                tracking_config=tracking_config,
            )
            if canonical_run is not None
            else None
        )
    except Exception:
        capture.release()
        raise
    if frame_window.start_frame > 0 and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_window.start_frame):
        capture.release()
        raise RuntimeError(f"Could not seek video for one-pass inference: {item.local_path}")
    save_video = bool(video_cfg.get("save_video", True))
    target: Optional[Path] = None
    writer: Any = None
    if save_video:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        target = video_dir / f"{trainer.sanitize_name(item.local_path.stem)}_pred.mp4"
        writer = cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            writer.release()
            raise RuntimeError(f"Could not create video writer: {target}")

    all_predictions: List[Dict[str, Any]] = []
    last_predictions: List[Dict[str, Any]] = []
    frame_cache_dir = output_dir / "_frame_cache"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    render_skipped = bool(video_cfg.get("render_skipped_frames", True))
    try:
        tracker = create_tracker(tracking_config, tracker_device, (width, height))
    except Exception:
        capture.release()
        if writer is not None:
            writer.release()
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
        raise
    hybrid_tracking = is_hybrid_tracker(tracker)
    hybrid_frame_buffer: Dict[int, Tuple[Any, bool]] = {}
    canonical_frame_meta: Dict[int, Mapping[str, Any]] = {}
    previous_canonical_timestamp: Optional[float] = None
    iterator = tqdm(total=frame_limit, desc=f"Inference video {item.local_path.name}", unit="frame")
    segment_frame_index = 0
    absolute_frame_index = frame_window.start_frame
    image_id = start_image_id
    try:
        while True:
            if frame_limit is not None and segment_frame_index >= frame_limit:
                break
            ok, frame = capture.read()
            if not ok:
                break
            should_detect = segment_frame_index % frame_interval == 0
            source_timestamp, timestamp_source = decoded_frame_timestamp(
                capture,
                cv2,
                absolute_frame_index,
                input_fps,
                previous_canonical_timestamp,
            )
            previous_canonical_timestamp = source_timestamp
            detected: List[Dict[str, Any]] = []
            if should_detect:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = frame_cache_dir / "current_frame.jpg"
                frame_image = Image.fromarray(rgb)
                try:
                    frame_image.save(frame_path, quality=92)
                finally:
                    frame_image.close()
                record = evaluator.ImageRecord(
                    image_id=image_id,
                    file_name=f"{item.local_path.stem}_frame_{absolute_frame_index:06d}.jpg",
                    path=str(frame_path),
                    width=width,
                    height=height,
                )
                detected, timing, _ = evaluator.predict_image(
                    record, model, prediction_config, output_dir, save_visual=False
                )
                record_inference_timing_rows(model, [timing])
                detected = filter_final_inference_predictions(detected, prediction_config)
                image_id += 1
            if hybrid_tracking:
                if canonical_video is not None:
                    canonical_frame_meta[absolute_frame_index] = {
                        "source_timestamp_seconds": source_timestamp,
                        "timestamp_source": timestamp_source,
                        "detection_ran": should_detect,
                    }
                if writer is not None:
                    hybrid_frame_buffer[absolute_frame_index] = (
                        frame,
                        bool(should_detect or render_skipped),
                    )
                committed = tracker.step(
                    absolute_frame_index,
                    source_timestamp,
                    frame,
                    detected,
                )
                consume_hybrid_committed(
                    committed,
                    frame_buffer=hybrid_frame_buffer,
                    writer=writer,
                    all_predictions=all_predictions,
                    tracking_state_rows=tracking_state_rows,
                    source=item.source,
                    input_fps=input_fps,
                    frame_window=frame_window,
                    categories=categories,
                    render_ids=render_ids,
                    tracking_config=tracking_config,
                    canonical_video_writer=canonical_video,
                    canonical_frame_meta=canonical_frame_meta,
                )
            else:
                if tracker is not None:
                    # Track age is measured in source frames, including frames
                    # skipped by a reduced detection_fps setting.
                    frame_predictions = tracker.update(absolute_frame_index, detected, frame=frame)
                    for prediction in frame_predictions:
                        all_predictions.append(
                            build_video_row(
                                prediction,
                                item.source,
                                absolute_frame_index,
                                segment_frame_index,
                                input_fps,
                                frame_window,
                            )
                        )
                else:
                    frame_predictions = detected if should_detect else last_predictions
                    if should_detect:
                        for prediction in frame_predictions:
                            all_predictions.append(
                                build_video_row(
                                    prediction,
                                    item.source,
                                    absolute_frame_index,
                                    segment_frame_index,
                                    input_fps,
                                    frame_window,
                                )
                            )
                if should_detect or tracker is not None:
                    last_predictions = frame_predictions
                if canonical_video is not None:
                    canonical_detections = frame_predictions if should_detect else []
                    canonical_video.write_frame(
                        segment_frame_index=segment_frame_index,
                        source_frame_index=absolute_frame_index,
                        source_timestamp_seconds=source_timestamp,
                        timestamp_source=timestamp_source,
                        detection_ran=should_detect,
                        detections=canonical_detections,
                        track_states=canonical_track_states_for_frame(
                            canonical_detections,
                            tracker,
                            tracking_config,
                            categories,
                            absolute_frame_index,
                            input_fps,
                        ),
                        camera_motion=None,
                    )
                if writer is not None and (should_detect or render_skipped):
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    rendered: Optional[Image.Image] = None
                    try:
                        rendered = draw_predictions(
                            pil_frame,
                            frame_predictions,
                            categories,
                            render_ids,
                            tracker.tracks if tracker is not None else None,
                            tracking_config,
                            absolute_frame_index,
                        )
                        frame = cv2.cvtColor(np_image(rendered), cv2.COLOR_RGB2BGR)
                    finally:
                        if rendered is not None and rendered is not pil_frame:
                            rendered.close()
                        pil_frame.close()
                if writer is not None:
                    writer.write(frame)
            segment_frame_index += 1
            absolute_frame_index += 1
            iterator.update(1)
        if hybrid_tracking:
            consume_hybrid_committed(
                tracker.flush(),
                frame_buffer=hybrid_frame_buffer,
                writer=writer,
                all_predictions=all_predictions,
                tracking_state_rows=tracking_state_rows,
                source=item.source,
                input_fps=input_fps,
                frame_window=frame_window,
                categories=categories,
                render_ids=render_ids,
                tracking_config=tracking_config,
                canonical_video_writer=canonical_video,
                canonical_frame_meta=canonical_frame_meta,
            )
            if hybrid_frame_buffer:
                raise RuntimeError("Hybrid tracker flush did not commit every buffered source frame")
            if canonical_frame_meta:
                raise RuntimeError("Hybrid tracker flush did not commit every Canonical frame")

    finally:
        iterator.close()
        capture.release()
        if writer is not None:
            writer.release()
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
    if canonical_video is not None:
        canonical_video.finalize(annotated_output=target)
    return all_predictions, target, image_id


def predict_video_file_batched(
    item: SourceItem,
    start_image_id: int,
    model: Any,
    prediction_config: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
    output_dir: Path,
    render_ids: Sequence[int],
    video_cfg: Mapping[str, Any],
    batch_size: int,
    tracking_config: Optional[Any] = None,
    tracker_device: str = "cpu",
    tracking_state_rows: Optional[List[Dict[str, Any]]] = None,
    canonical_run: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Path], int]:
    """Batch RF-DETR detection frames, then render the selected video range."""
    import cv2

    assert item.local_path is not None
    capture = cv2.VideoCapture(str(item.local_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for batched detection: {item.local_path}")
    input_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    output_fps = float(video_cfg.get("output_fps") or input_fps)
    detection_fps = video_cfg.get("detection_fps")
    frame_interval = 1
    if detection_fps is not None:
        frame_interval = max(1, int(round(input_fps / max(0.001, float(detection_fps)))))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_window = video_frame_window(frame_count, input_fps, video_cfg)
    frame_limit = frame_window.output_frames
    try:
        canonical_video = (
            canonical_run.start_video(
                source=item.source,
                source_path=item.local_path,
                width=width,
                height=height,
                input_fps=input_fps,
                source_frame_count=frame_count,
                frame_window=frame_window,
                detection_fps=detection_fps,
                frame_interval=frame_interval,
                output_fps=output_fps,
                tracking_config=tracking_config,
            )
            if canonical_run is not None
            else None
        )
    except Exception:
        capture.release()
        raise
    if frame_window.start_frame > 0 and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_window.start_frame):
        capture.release()
        raise RuntimeError(f"Could not seek video for batched detection: {item.local_path}")

    frame_cache_dir = output_dir / "_frame_cache"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    batch_config = prediction_config_with_batch(prediction_config, batch_size)
    all_predictions: List[Dict[str, Any]] = []
    predictions_by_segment: Dict[int, List[Dict[str, Any]]] = {}
    pending_records: List[evaluator.ImageRecord] = []
    pending_meta: List[Dict[str, Any]] = []
    image_id = start_image_id

    def flush_pending() -> None:
        nonlocal pending_records, pending_meta
        if not pending_records:
            return
        predictions_by_frame, timing_rows, _ = evaluator.predict_images_rfdetr(
            pending_records, model, batch_config, output_dir
        )
        record_inference_timing_rows(model, timing_rows)
        for record, meta, frame_predictions in zip(pending_records, pending_meta, predictions_by_frame):
            predictions_by_segment[int(meta["segment_frame_index"])] = filter_final_inference_predictions(
                frame_predictions,
                batch_config,
            )
            Path(record.path).unlink(missing_ok=True)
        pending_records = []
        pending_meta = []

    segment_frame_index = 0
    absolute_frame_index = frame_window.start_frame
    detect_iterator = tqdm(total=frame_limit, desc=f"Detect video {item.local_path.name}", unit="frame")
    try:
        while True:
            if frame_limit is not None and segment_frame_index >= frame_limit:
                break
            ok, frame = capture.read()
            if not ok:
                break
            if segment_frame_index % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = frame_cache_dir / f"frame_{absolute_frame_index:010d}.jpg"
                frame_image = Image.fromarray(rgb)
                try:
                    frame_image.save(frame_path, quality=92)
                finally:
                    frame_image.close()
                pending_records.append(
                    evaluator.ImageRecord(
                        image_id=image_id,
                        file_name=f"{item.local_path.stem}_frame_{absolute_frame_index:06d}.jpg",
                        path=str(frame_path),
                        width=width,
                        height=height,
                    )
                )
                pending_meta.append({"segment_frame_index": segment_frame_index, "absolute_frame_index": absolute_frame_index})
                image_id += 1
                if len(pending_records) >= batch_size:
                    flush_pending()
            segment_frame_index += 1
            absolute_frame_index += 1
            detect_iterator.update(1)
        flush_pending()
    except Exception:
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
        raise
    finally:
        detect_iterator.close()
        capture.release()

    render_capture = cv2.VideoCapture(str(item.local_path))
    if not render_capture.isOpened():
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
        raise RuntimeError(f"Could not reopen video for batched render: {item.local_path}")
    if frame_window.start_frame > 0 and not render_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_window.start_frame):
        render_capture.release()
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
        raise RuntimeError(f"Could not seek video for batched render: {item.local_path}")
    save_video = bool(video_cfg.get("save_video", True))
    target: Optional[Path] = None
    writer: Any = None
    if save_video:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        target = video_dir / f"{trainer.sanitize_name(item.local_path.stem)}_pred.mp4"
        writer = cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
        )
        if not writer.isOpened():
            render_capture.release()
            writer.release()
            shutil.rmtree(frame_cache_dir, ignore_errors=True)
            raise RuntimeError(f"Could not create video writer: {target}")

    render_skipped = bool(video_cfg.get("render_skipped_frames", True))
    try:
        tracker = create_tracker(tracking_config, tracker_device, (width, height))
    except Exception:
        render_capture.release()
        if writer is not None:
            writer.release()
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
        raise
    hybrid_tracking = is_hybrid_tracker(tracker)
    hybrid_frame_buffer: Dict[int, Tuple[Any, bool]] = {}
    canonical_frame_meta: Dict[int, Mapping[str, Any]] = {}
    previous_canonical_timestamp: Optional[float] = None
    last_predictions: List[Dict[str, Any]] = []
    segment_frame_index = 0
    render_iterator = tqdm(total=frame_limit, desc=f"Render video {item.local_path.name}", unit="frame")
    try:
        while True:
            if frame_limit is not None and segment_frame_index >= frame_limit:
                break
            ok, frame = render_capture.read()
            if not ok:
                break
            absolute_frame_index = frame_window.start_frame + segment_frame_index
            should_detect = segment_frame_index % frame_interval == 0
            source_timestamp, timestamp_source = decoded_frame_timestamp(
                render_capture,
                cv2,
                absolute_frame_index,
                input_fps,
                previous_canonical_timestamp,
            )
            previous_canonical_timestamp = source_timestamp
            detected = predictions_by_segment.get(segment_frame_index, []) if should_detect else []
            if hybrid_tracking:
                if canonical_video is not None:
                    canonical_frame_meta[absolute_frame_index] = {
                        "source_timestamp_seconds": source_timestamp,
                        "timestamp_source": timestamp_source,
                        "detection_ran": should_detect,
                    }
                if writer is not None:
                    hybrid_frame_buffer[absolute_frame_index] = (
                        frame,
                        bool(should_detect or render_skipped),
                    )
                committed = tracker.step(
                    absolute_frame_index,
                    source_timestamp,
                    frame,
                    detected,
                )
                consume_hybrid_committed(
                    committed,
                    frame_buffer=hybrid_frame_buffer,
                    writer=writer,
                    all_predictions=all_predictions,
                    tracking_state_rows=tracking_state_rows,
                    source=item.source,
                    input_fps=input_fps,
                    frame_window=frame_window,
                    categories=categories,
                    render_ids=render_ids,
                    tracking_config=tracking_config,
                    canonical_video_writer=canonical_video,
                    canonical_frame_meta=canonical_frame_meta,
                )
            else:
                if tracker is not None:
                    frame_predictions = tracker.update(absolute_frame_index, detected, frame=frame)
                    for prediction in frame_predictions:
                        all_predictions.append(
                            build_video_row(
                                prediction,
                                item.source,
                                absolute_frame_index,
                                segment_frame_index,
                                input_fps,
                                frame_window,
                            )
                        )
                else:
                    frame_predictions = detected if should_detect else last_predictions
                    if should_detect:
                        for prediction in frame_predictions:
                            all_predictions.append(
                                build_video_row(
                                    prediction,
                                    item.source,
                                    absolute_frame_index,
                                    segment_frame_index,
                                    input_fps,
                                    frame_window,
                                )
                            )
                if should_detect or tracker is not None:
                    last_predictions = frame_predictions
                if canonical_video is not None:
                    canonical_detections = frame_predictions if should_detect else []
                    canonical_video.write_frame(
                        segment_frame_index=segment_frame_index,
                        source_frame_index=absolute_frame_index,
                        source_timestamp_seconds=source_timestamp,
                        timestamp_source=timestamp_source,
                        detection_ran=should_detect,
                        detections=canonical_detections,
                        track_states=canonical_track_states_for_frame(
                            canonical_detections,
                            tracker,
                            tracking_config,
                            categories,
                            absolute_frame_index,
                            input_fps,
                        ),
                        camera_motion=None,
                    )
                if writer is not None and (should_detect or render_skipped):
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    rendered: Optional[Image.Image] = None
                    try:
                        rendered = draw_predictions(
                            pil_frame,
                            frame_predictions,
                            categories,
                            render_ids,
                            tracker.tracks if tracker is not None else None,
                            tracking_config,
                            absolute_frame_index,
                        )
                        frame = cv2.cvtColor(np_image(rendered), cv2.COLOR_RGB2BGR)
                    finally:
                        if rendered is not None and rendered is not pil_frame:
                            rendered.close()
                        pil_frame.close()
                if writer is not None:
                    writer.write(frame)
            segment_frame_index += 1
            render_iterator.update(1)
        if hybrid_tracking:
            consume_hybrid_committed(
                tracker.flush(),
                frame_buffer=hybrid_frame_buffer,
                writer=writer,
                all_predictions=all_predictions,
                tracking_state_rows=tracking_state_rows,
                source=item.source,
                input_fps=input_fps,
                frame_window=frame_window,
                categories=categories,
                render_ids=render_ids,
                tracking_config=tracking_config,
                canonical_video_writer=canonical_video,
                canonical_frame_meta=canonical_frame_meta,
            )
            if hybrid_frame_buffer:
                raise RuntimeError("Hybrid tracker flush did not commit every buffered source frame")
            if canonical_frame_meta:
                raise RuntimeError("Hybrid tracker flush did not commit every Canonical frame")
    finally:
        render_iterator.close()
        render_capture.release()
        if writer is not None:
            writer.release()
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
    if canonical_video is not None:
        canonical_video.finalize(annotated_output=target)
    return all_predictions, target, image_id


def predict_video_file_streaming(
    item: SourceItem,
    start_image_id: int,
    model: Any,
    prediction_config: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
    output_dir: Path,
    render_ids: Sequence[int],
    video_cfg: Mapping[str, Any],
    batch_size: int,
    tracking_config: Optional[Any] = None,
    tracker_device: str = "cpu",
    tracking_state_rows: Optional[List[Dict[str, Any]]] = None,
    canonical_run: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Path], int]:
    """Decode, batch, track, render, and encode a selected video range once.

    Frames are retained only until either a model batch is full or the bounded
    queue reaches ``queue_size``. Decoded RGB images are sent directly to the
    evaluator, avoiding the historical JPEG round trip and second video decode.
    """
    import cv2

    assert item.local_path is not None
    pipeline_started = time.perf_counter()
    capture = cv2.VideoCapture(str(item.local_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for streaming inference: {item.local_path}")
    input_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    output_fps = float(video_cfg.get("output_fps") or input_fps)
    detection_fps = video_cfg.get("detection_fps")
    frame_interval = 1
    if detection_fps is not None:
        frame_interval = max(1, int(round(input_fps / max(0.001, float(detection_fps)))))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_window = video_frame_window(frame_count, input_fps, video_cfg)
    frame_limit = frame_window.output_frames
    try:
        canonical_video = (
            canonical_run.start_video(
                source=item.source,
                source_path=item.local_path,
                width=width,
                height=height,
                input_fps=input_fps,
                source_frame_count=frame_count,
                frame_window=frame_window,
                detection_fps=detection_fps,
                frame_interval=frame_interval,
                output_fps=output_fps,
                tracking_config=tracking_config,
            )
            if canonical_run is not None
            else None
        )
    except Exception:
        capture.release()
        raise
    if frame_window.start_frame > 0 and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_window.start_frame):
        capture.release()
        raise RuntimeError(f"Could not seek video for streaming inference: {item.local_path}")

    save_video = bool(video_cfg.get("save_video", True))
    target: Optional[Path] = None
    writer: Any = None
    if save_video:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        target = video_dir / f"{trainer.sanitize_name(item.local_path.stem)}_pred.mp4"
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            writer.release()
            raise RuntimeError(f"Could not create video writer: {target}")

    batch_config = prediction_config_with_batch(prediction_config, batch_size)
    queue_size = positive_batch_size(
        video_cfg.get("queue_size"),
        "inference.video.queue_size",
        max(8, batch_size * 2),
    )
    queue_size = max(batch_size, queue_size)
    render_skipped = bool(video_cfg.get("render_skipped_frames", True))
    try:
        tracker = create_tracker(tracking_config, tracker_device, (width, height))
    except Exception:
        capture.release()
        if writer is not None:
            writer.release()
        raise
    hybrid_tracking = is_hybrid_tracker(tracker)
    hybrid_frame_buffer: Dict[int, Tuple[Any, bool]] = {}
    canonical_frame_meta: Dict[int, Mapping[str, Any]] = {}
    all_predictions: List[Dict[str, Any]] = []
    last_predictions: List[Dict[str, Any]] = []
    pending_frames: List[Dict[str, Any]] = []
    pending_detection_count = 0
    image_id = start_image_id
    stage = {
        "decode_seconds": 0.0,
        "frame_conversion_seconds": 0.0,
        "video_evaluator_seconds": 0.0,
        "tracker_seconds": 0.0,
        "render_seconds": 0.0,
        "encode_seconds": 0.0,
        "source_frames": 0,
        "detection_frames": 0,
        "frame_queue_peak": 0,
    }

    def flush_pending() -> None:
        nonlocal pending_detection_count, image_id, last_predictions
        if not pending_frames:
            return
        detection_packets = [packet for packet in pending_frames if packet["should_detect"]]
        predictions_by_segment: Dict[int, List[Dict[str, Any]]] = {}
        if detection_packets:
            records: List[evaluator.ImageRecord] = []
            sources: List[Image.Image] = []
            for packet in detection_packets:
                packet_image_id = image_id + len(records)
                absolute_index = int(packet["absolute_frame_index"])
                records.append(
                    evaluator.ImageRecord(
                        image_id=packet_image_id,
                        file_name=f"{item.local_path.stem}_frame_{absolute_index:06d}.jpg",
                        # The evaluator receives ``sources`` below and must not
                        # open this descriptive, intentionally non-existent path.
                        path=f"memory://{item.local_path.stem}/{absolute_index}",
                        width=width,
                        height=height,
                    )
                )
                conversion_started = time.perf_counter()
                sources.append(
                    Image.fromarray(
                        cv2.cvtColor(packet["frame"], cv2.COLOR_BGR2RGB)
                    )
                )
                stage["frame_conversion_seconds"] += time.perf_counter() - conversion_started
            evaluate_started = time.perf_counter()
            try:
                predictions_by_frame, timing_rows, _ = evaluator.call_with_supported_kwargs(
                    evaluator.predict_images_rfdetr,
                    records,
                    model,
                    batch_config,
                    output_dir,
                    sources=sources,
                )
            finally:
                for source_image in sources:
                    source_image.close()
            stage["video_evaluator_seconds"] += time.perf_counter() - evaluate_started
            record_inference_timing_rows(model, timing_rows or [])
            for packet, frame_predictions in zip(detection_packets, predictions_by_frame):
                predictions_by_segment[int(packet["segment_frame_index"])] = (
                    filter_final_inference_predictions(frame_predictions, batch_config)
                )
            image_id += len(records)

        for packet in pending_frames:
            segment_index = int(packet["segment_frame_index"])
            absolute_index = int(packet["absolute_frame_index"])
            frame = packet["frame"]
            should_detect = bool(packet["should_detect"])
            source_timestamp = float(packet["source_timestamp_seconds"])
            timestamp_source = str(packet["timestamp_source"])
            detected = predictions_by_segment.get(segment_index, []) if should_detect else []
            frame_predictions: List[Dict[str, Any]]
            tracker_started = time.perf_counter()
            if hybrid_tracking:
                if canonical_video is not None:
                    canonical_frame_meta[absolute_index] = {
                        "source_timestamp_seconds": source_timestamp,
                        "timestamp_source": timestamp_source,
                        "detection_ran": should_detect,
                    }
                if writer is not None:
                    hybrid_frame_buffer[absolute_index] = (
                        frame,
                        bool(should_detect or render_skipped),
                    )
                committed = tracker.step(
                    absolute_index,
                    source_timestamp,
                    frame,
                    detected,
                )
                stage["tracker_seconds"] += time.perf_counter() - tracker_started
                render_elapsed, encode_elapsed = consume_hybrid_committed(
                    committed,
                    frame_buffer=hybrid_frame_buffer,
                    writer=writer,
                    all_predictions=all_predictions,
                    tracking_state_rows=tracking_state_rows,
                    source=item.source,
                    input_fps=input_fps,
                    frame_window=frame_window,
                    categories=categories,
                    render_ids=render_ids,
                    tracking_config=tracking_config,
                    canonical_video_writer=canonical_video,
                    canonical_frame_meta=canonical_frame_meta,
                )
                stage["render_seconds"] += render_elapsed
                stage["encode_seconds"] += encode_elapsed
                frame_predictions = []
            elif tracker is not None:
                # Advance the tracker for every source frame. BoxMot and the
                # built-in tracker then age tracks in source-frame units rather
                # than detection-call units when detection_fps is reduced.
                frame_predictions = tracker.update(absolute_index, detected, frame=frame)
                for prediction in frame_predictions:
                    all_predictions.append(
                        build_video_row(
                            prediction,
                            item.source,
                            absolute_index,
                            segment_index,
                            input_fps,
                            frame_window,
                        )
                    )
            else:
                frame_predictions = detected if should_detect else last_predictions
                if should_detect:
                    for prediction in frame_predictions:
                        all_predictions.append(
                            build_video_row(
                                prediction,
                                item.source,
                                absolute_index,
                                segment_index,
                                input_fps,
                                frame_window,
                            )
                        )
            if not hybrid_tracking:
                stage["tracker_seconds"] += time.perf_counter() - tracker_started
            if should_detect or tracker is not None:
                last_predictions = frame_predictions
            if canonical_video is not None and not hybrid_tracking:
                canonical_detections = frame_predictions if should_detect else []
                canonical_video.write_frame(
                    segment_frame_index=segment_index,
                    source_frame_index=absolute_index,
                    source_timestamp_seconds=source_timestamp,
                    timestamp_source=timestamp_source,
                    detection_ran=should_detect,
                    detections=canonical_detections,
                    track_states=canonical_track_states_for_frame(
                        canonical_detections,
                        tracker,
                        tracking_config,
                        categories,
                        absolute_index,
                        input_fps,
                    ),
                    camera_motion=None,
                )

            if writer is not None and not hybrid_tracking:
                render_started = time.perf_counter()
                if should_detect or render_skipped:
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    rendered: Optional[Image.Image] = None
                    try:
                        rendered = draw_predictions(
                            pil_frame,
                            frame_predictions,
                            categories,
                            render_ids,
                            tracker.tracks if tracker is not None else None,
                            tracking_config,
                            absolute_index,
                        )
                        frame = cv2.cvtColor(np_image(rendered), cv2.COLOR_RGB2BGR)
                    finally:
                        if rendered is not None and rendered is not pil_frame:
                            rendered.close()
                        pil_frame.close()
                stage["render_seconds"] += time.perf_counter() - render_started
                encode_started = time.perf_counter()
                writer.write(frame)
                stage["encode_seconds"] += time.perf_counter() - encode_started

        pending_frames.clear()
        pending_detection_count = 0

    iterator = tqdm(total=frame_limit, desc=f"Stream video {item.local_path.name}", unit="frame")
    segment_frame_index = 0
    absolute_frame_index = frame_window.start_frame
    previous_canonical_timestamp: Optional[float] = None
    try:
        while True:
            if frame_limit is not None and segment_frame_index >= frame_limit:
                break
            decode_started = time.perf_counter()
            ok, frame = capture.read()
            stage["decode_seconds"] += time.perf_counter() - decode_started
            if not ok:
                break
            should_detect = segment_frame_index % frame_interval == 0
            source_timestamp, timestamp_source = decoded_frame_timestamp(
                capture,
                cv2,
                absolute_frame_index,
                input_fps,
                previous_canonical_timestamp,
            )
            previous_canonical_timestamp = source_timestamp
            pending_frames.append(
                {
                    "segment_frame_index": segment_frame_index,
                    "absolute_frame_index": absolute_frame_index,
                    "should_detect": should_detect,
                    "source_timestamp_seconds": source_timestamp,
                    "timestamp_source": timestamp_source,
                    "frame": frame,
                }
            )
            pending_detection_count += int(should_detect)
            stage["source_frames"] += 1
            stage["detection_frames"] += int(should_detect)
            stage["frame_queue_peak"] = max(stage["frame_queue_peak"], len(pending_frames))
            if pending_detection_count >= batch_size or len(pending_frames) >= queue_size:
                flush_pending()
            segment_frame_index += 1
            absolute_frame_index += 1
            iterator.update(1)
        flush_pending()
        if hybrid_tracking:
            render_elapsed, encode_elapsed = consume_hybrid_committed(
                tracker.flush(),
                frame_buffer=hybrid_frame_buffer,
                writer=writer,
                all_predictions=all_predictions,
                tracking_state_rows=tracking_state_rows,
                source=item.source,
                input_fps=input_fps,
                frame_window=frame_window,
                categories=categories,
                render_ids=render_ids,
                tracking_config=tracking_config,
                canonical_video_writer=canonical_video,
                canonical_frame_meta=canonical_frame_meta,
            )
            stage["render_seconds"] += render_elapsed
            stage["encode_seconds"] += encode_elapsed
            if hybrid_frame_buffer:
                raise RuntimeError("Hybrid tracker flush did not commit every buffered source frame")
            if canonical_frame_meta:
                raise RuntimeError("Hybrid tracker flush did not commit every Canonical frame")
    finally:
        iterator.close()
        capture.release()
        if writer is not None:
            writer.release()
        pipeline_wall = time.perf_counter() - pipeline_started
        cuda_memory = cuda_memory_telemetry_for_model(model)
        record_video_pipeline_timing(
            model,
            video_pipeline_wall_seconds=pipeline_wall,
            decode_seconds=stage["decode_seconds"],
            frame_conversion_seconds=stage["frame_conversion_seconds"],
            video_evaluator_seconds=stage["video_evaluator_seconds"],
            tracker_seconds=stage["tracker_seconds"],
            render_seconds=stage["render_seconds"],
            encode_seconds=stage["encode_seconds"],
            source_frames=stage["source_frames"],
            detection_frames=stage["detection_frames"],
            frame_queue_peak=stage["frame_queue_peak"],
            **cuda_memory,
            outer_batch_size=batch_size,
        )
    if canonical_video is not None:
        canonical_video.finalize(annotated_output=target)
    return all_predictions, target, image_id


def predict_video_file(
    item: SourceItem,
    start_image_id: int,
    model: Any,
    prediction_config: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
    output_dir: Path,
    render_ids: Sequence[int],
    video_cfg: Mapping[str, Any],
    tracking_config: Optional[Any] = None,
    tracker_device: str = "cpu",
    tracking_state_rows: Optional[List[Dict[str, Any]]] = None,
    canonical_run: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Path], int]:
    """Predict one video with batched detection frames when configured."""
    def abort_canonical_bundle() -> None:
        if canonical_run is not None:
            with contextlib.suppress(Exception):
                canonical_run.abort_active()

    configured_batch = positive_batch_size(video_cfg.get("batch_size"), "inference.video.batch_size", 1)
    if bool(video_cfg.get("streaming", True)):
        try:
            return predict_video_file_streaming(
                item,
                start_image_id,
                model,
                prediction_config,
                categories,
                output_dir,
                render_ids,
                video_cfg,
                configured_batch,
                tracking_config,
                tracker_device,
                tracking_state_rows,
                canonical_run,
            )
        except Exception:
            abort_canonical_bundle()
            raise
    if configured_batch <= 1:
        try:
            return predict_video_file_one_pass(
                item, start_image_id, model, prediction_config, categories, output_dir, render_ids, video_cfg,
                tracking_config, tracker_device, tracking_state_rows, canonical_run,
            )
        except Exception:
            abort_canonical_bundle()
            raise
    try:
        return predict_video_file_batched(
            item,
            start_image_id,
            model,
            prediction_config,
            categories,
            output_dir,
            render_ids,
            video_cfg,
            configured_batch,
            tracking_config,
            tracker_device, tracking_state_rows, canonical_run,
        )
    except RuntimeError as exc:
        abort_canonical_bundle()
        if "batched" in str(exc).lower() or "reopen video" in str(exc).lower() or "seek video" in str(exc).lower():
            print(Fore.BLUE + Style.BRIGHT + f"Warning: batched video inference fell back to one-pass mode. {exc}")
            try:
                return predict_video_file_one_pass(
                    item, start_image_id, model, prediction_config, categories, output_dir, render_ids, video_cfg,
                    tracking_config, tracker_device, tracking_state_rows, canonical_run,
                )
            except Exception:
                abort_canonical_bundle()
                raise
        raise
    except Exception:
        abort_canonical_bundle()
        raise


def np_image(image: Image.Image) -> Any:
    import numpy as np

    return np.asarray(image.convert("RGB"))


def write_predictions_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=trainer.json_safe_value) + "\n")


def write_football_predictions_output(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    football_config: FootballOutputConfig,
) -> Dict[str, Any]:
    """Write the optional concise artifact and return its stable summary fields."""
    filename: Optional[str] = None
    count = 0
    if football_config.enabled:
        filename = FOOTBALL_PREDICTIONS_FILENAME
        write_predictions_jsonl(output_dir / filename, rows)
        count = len(rows)
    return {
        "football_prediction_count": count,
        "football_predictions_file": filename,
    }


def _main_impl(timing_context: Optional[MutableMapping[str, Any]] = None) -> int:
    parser = argparse.ArgumentParser(description="RF-DETR image/video inference runner.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to rf_detr_inference.yaml.")
    parser.add_argument("--source", action="append", help="Image/video file, folder, or URL. Can be repeated.")
    parser.add_argument("--output-dir", help="Exact output directory override.")
    parser.add_argument("--checkpoint", help="RF-DETR checkpoint/pretrain_weights override.")
    parser.add_argument("--device", help="Device override: auto, cpu, cuda, cuda:0, 0, 1.")
    parser.add_argument("--confidence-threshold", type=float, help="Model confidence threshold override.")
    parser.add_argument(
        "--tracknet-focus",
        choices=("single", "all"),
        help="Override model.motion.focus.mode for temporal TrackNet inference.",
    )
    parser.add_argument(
        "--inference-backend",
        choices=["pytorch", "tensorrt"],
        help="Inference backend override. PyTorch FP32 remains the default.",
    )
    parser.add_argument(
        "--inference-precision",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for the active inference backend.",
    )
    parser.add_argument("--tensorrt-engine", help="Trusted TensorRT engine path; requires a compatible manifest.")
    parser.add_argument("--tensorrt-cache-dir", help="TensorRT ONNX/engine cache directory override.")
    parser.add_argument(
        "--tensorrt-force-rebuild",
        action="store_true",
        help="Ignore a matching automatic TensorRT cache entry and rebuild it.",
    )
    parser.add_argument("--max-sources", type=trainer.parse_scalar, help="Maximum discovered sources to run. Use all/null for all.")
    parser.add_argument("--max-images", type=trainer.parse_scalar, help="Maximum image sources to run. Use all/null for all.")
    parser.add_argument("--max-videos", type=trainer.parse_scalar, help="Maximum video sources to run. Use all/null for all.")
    parser.add_argument("--batch-size", type=int, help="RF-DETR image-source inference batch size.")
    parser.add_argument("--video-batch-size", type=trainer.parse_scalar, help="RF-DETR video detection-frame batch size. all/null inherits --batch-size.")
    parser.add_argument(
        "--sahi-batch-size",
        type=trainer.parse_scalar,
        help="RF-DETR SAHI slice/recheck batch size: auto or a positive integer.",
    )
    parser.add_argument(
        "--performance-profile",
        choices=["safe", "fast"],
        help="Runtime profile: safe=PyTorch BF16/full-scale CMC; fast=TensorRT FP16/half-scale CMC.",
    )
    parser.add_argument(
        "--video-streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use one-decode in-memory video inference (default: true).",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write rendered MP4 output; Canonical V2 clean media is controlled separately.",
    )
    canonical_output.add_canonical_output_cli_arguments(parser)
    parser.add_argument(
        "--cmc-processing-scale",
        type=float,
        help="Hybrid tracker CMC image scale in (0, 1], e.g. 0.5 for fast profile.",
    )
    parser.add_argument("--max-seconds", type=trainer.parse_scalar, help="Maximum seconds per video to infer. Use all/null for the whole video.")
    parser.add_argument("--video-start-time", help="Video segment start time. Options: seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument("--video-end-time", help="Video segment end time. Options: all/null, seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument("--track", action="store_true", help="Enable football tracking for video inference.")
    parser.add_argument("--no-track", dest="no_track", action="store_true", help="Disable football tracking (overrides config and --track).")
    parser.add_argument("--track-radius", dest="track_radius", type=float, help="Override inference.tracking.radius_pixels (search radius in pixels).")
    parser.add_argument("--track-velocity", dest="track_velocity", action="store_true", help="Enable the velocity-predicted gate for the circle tracker.")
    parser.add_argument("--tracker", dest="tracker", choices=["circle", "ocsort", "deepocsort", "botsort", "bytetrack", "hybrid"], help="Override inference.tracking.algorithm.")
    parser.add_argument("--reid-weights", dest="reid_weights", help="Override inference.tracking.reid_weights (local ReID .pt path for deepocsort/botsort).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate outputs without inference.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    cpu_runtime.add_cpu_cli_arguments(parser)
    args = parser.parse_args()

    source_config = Path(args.config).expanduser()
    if not source_config.is_absolute():
        source_config = (Path.cwd() / source_config).resolve()
    config = load_yaml(source_config)
    apply_cli_overrides(config, args)
    canonical_cfg = canonical_output.parse_canonical_output_config(config)
    cpu_summary = cpu_runtime.validate_active_config(config, "inference", source_config)
    trainer._require_custom_architecture_checkpoint(config, "Inference")
    trainer.validate_inference_acceleration_config(config)
    categories = build_categories(config)
    football_output_config = parse_football_output_config(config, categories)
    verbose = bool(config.get("runtime", {}).get("verbose", True))
    print(cpu_runtime.format_summary(cpu_summary))
    if timing_context is not None:
        timing_context["verbose"] = verbose
        timing_context["cpu_runtime"] = cpu_summary
        timing_context["execution_profile"] = trainer.inference_execution_profile(config)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    output_dir = build_output_dir(config, timestamp)
    if timing_context is not None:
        timing_context["output_dir"] = str(output_dir)
    items = apply_source_limits(discover_sources(config), config)
    if not items and not bool(config.get("runtime", {}).get("dry_run", False)):
        raise ValueError("No inference sources were found.")
    if output_dir.exists() and not bool(config.get("output", {}).get("exist_ok", False)):
        raise FileExistsError(f"Output directory already exists and output.exist_ok=false: {output_dir}")
    estimate = estimate_outputs(items, output_dir, config)
    if timing_context is not None:
        timing_context["estimate"] = estimate
        timing_context["dry_run"] = bool(config.get("runtime", {}).get("dry_run", False))
    confirm = bool(config.get("runtime", {}).get("confirm_before_run", True))
    assume_yes = bool(config.get("runtime", {}).get("yes", False) or args.yes or not confirm)
    confirm_or_exit(estimate, verbose, assume_yes)
    canonical_video_items = (
        []
        if trainer.temporal_motion_enabled(config)
        else [item for item in items if item.kind == "video"]
    )
    canonical_toolchain = (
        canonical_output.preflight(canonical_cfg)
        if canonical_cfg.enabled and canonical_video_items
        else None
    )
    trainer.preflight_rfdetr_inference_acceleration(
        config,
        device=config.get("model", {}).get("device"),
    )
    if bool(config.get("runtime", {}).get("dry_run", False)):
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if timing_context is not None:
        timing_context["outputs_created"] = True
    trainer.start_run_log_capture(output_dir, "inference", timing_context)
    trainer.dump_config_snapshot(
        output_dir=output_dir,
        merged_config=config,
        metadata={
            "event": "inference_start",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_count": len(items),
            "execution_profile": trainer.inference_execution_profile(config),
        },
        source_config=source_config,
    )
    cache_dir = output_dir / "source_cache"
    resolved_items = [download_url(item, cache_dir) if item.is_url else item for item in items]
    prediction_config = build_prediction_config(config, categories)
    model = load_rfdetr_model(config)
    if timing_context is not None:
        acceleration_handle = trainer.get_inference_acceleration_handle(model)
        timing_context["acceleration"] = dict(acceleration_handle.metadata)

    if trainer.temporal_motion_enabled(config):
        from rf_detr_temporal_runtime import run_temporal_split

        temporal_result = run_temporal_split(
            rf_model=model,
            config=config,
            output_dir=output_dir,
            split=str(config.get("inference", {}).get("temporal_split", "test")),
            save_heatmaps=bool(config.get("inference", {}).get("save_heatmaps", True)),
        )
        football_rows = build_temporal_football_rows(
            temporal_result["rows"],
            football_output_config,
            categories,
            float(config.get("model", {}).get("confidence_threshold", 0.25)),
        )
        football_summary = write_football_predictions_output(
            output_dir,
            football_rows,
            football_output_config,
        )
        temporal_summary = dict(temporal_result["summary"])
        temporal_summary.update(football_summary)
        stage_timing = {
            "images_or_frames": int(temporal_summary.get("windows", 0)),
            "total_seconds": float(temporal_summary.get("total_seconds", 0.0)),
            "model_forward_seconds": float(temporal_summary.get("model_forward_seconds", 0.0)),
            "base_model_forward_seconds": float(temporal_summary.get("model_forward_seconds", 0.0)),
            "sahi_model_forward_seconds": 0.0,
            "recheck_model_forward_seconds": 0.0,
            "preprocess_seconds": 0.0,
            "postprocess_seconds": max(
                0.0,
                float(temporal_summary.get("total_seconds", 0.0))
                - float(temporal_summary.get("model_forward_seconds", 0.0)),
            ),
        }
        total_seconds = stage_timing["total_seconds"]
        stage_timing["model_forward_ratio"] = stage_timing["model_forward_seconds"] / total_seconds if total_seconds else 0.0
        stage_timing["sahi_model_forward_ratio"] = 0.0
        stage_timing["recheck_model_forward_ratio"] = 0.0
        if timing_context is not None:
            timing_context["stage_timing"] = stage_timing
        trainer.write_json(output_dir / "temporal_summary.json", temporal_summary)
        trainer.write_json(output_dir / "inference_summary.json", temporal_summary)
        trainer.dump_config_snapshot(
            output_dir=output_dir,
            merged_config=config,
            metadata={"event": "temporal_inference", "stage_timing": stage_timing},
            source_config=source_config,
        )
        if verbose:
            print(json.dumps(temporal_summary, indent=2, ensure_ascii=False))
            print(f"RF-DETR temporal inference output directory: {output_dir}")
        return 0

    render_ids = resolve_render_ids(config, categories)
    tracking_config = video_tracking.parse_tracking_config(config, categories)
    tracker_device = resolved_tracker_device(config, tracking_config)
    if tracking_config.enabled:
        warn_reid_in_restricted_region(tracking_config)
    all_predictions: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    track_state_rows: List[Dict[str, Any]] = []
    image_id = 1
    video_cfg = dict(config.get("inference", {}).get("video", {}) or {})
    video_cfg["batch_size"] = video_batch_size(config)
    canonical_run = (
        canonical_output.CanonicalRunWriter(
            output_dir=output_dir,
            cfg=canonical_cfg,
            categories=categories,
            producer_metadata={
                "name": "PitchObjectLab RF-DETR inference",
                "detector": {
                    "family": "rf-detr",
                    "model_size": config.get("model", {}).get("size"),
                    "checkpoint": config.get("model", {}).get("pretrain_weights"),
                    "confidence_threshold": config.get("model", {}).get("confidence_threshold"),
                    "inference_mode": config.get("inference", {}).get("mode"),
                    "inference_backend": (
                        config.get("model", {})
                        .get("inference_optimization", {})
                        .get("backend", "pytorch")
                    ),
                    "prediction_config": trainer.json_safe_value(prediction_config),
                },
                "config": {
                    "snapshot": "config/merged_config.yaml",
                    "source": str(source_config),
                },
            },
            toolchain=canonical_toolchain,
        )
        if canonical_cfg.enabled and canonical_video_items
        else None
    )
    image_batch_size = inference_batch_size(config)
    pending_images: List[SourceItem] = []

    def flush_pending_images() -> None:
        nonlocal image_id
        if not pending_images:
            return
        rows, image_outputs, image_id = predict_image_files_batch(
            pending_images,
            image_id,
            model,
            prediction_config,
            categories,
            output_dir,
            render_ids,
            image_batch_size,
        )
        all_predictions.extend(rows)
        for image_output in image_outputs:
            image_output["raw_predictions"] = int(image_output.get("predictions", 0))
            image_output["suppressed_unconfirmed"] = 0
        outputs.extend(image_outputs)
        pending_images.clear()

    iterator = tqdm(resolved_items, desc="RF-DETR inference", unit="source")
    for item in iterator:
        if item.kind == "image":
            pending_images.append(item)
            if len(pending_images) >= image_batch_size:
                flush_pending_images()
        elif item.kind == "video":
            flush_pending_images()
            predictions, path, image_id = predict_video_file(
                item,
                image_id,
                model,
                prediction_config,
                categories,
                output_dir,
                render_ids,
                video_cfg,
                tracking_config,
                tracker_device,
                track_state_rows,
                canonical_run=canonical_run,
            )
            all_predictions.extend(predictions)
            published_for_source, suppressed_for_source = filter_confirmed_hybrid_exports(
                predictions,
                tracking_config,
            )
            outputs.append(
                {
                    "source": item.source,
                    "kind": "video",
                    "output": str(path) if path is not None else None,
                    "predictions": len(published_for_source),
                    "raw_predictions": len(predictions),
                    "suppressed_unconfirmed": suppressed_for_source,
                }
            )
    flush_pending_images()

    serialization_started = time.perf_counter()
    published_predictions, suppressed_unconfirmed_count = filter_confirmed_hybrid_exports(
        all_predictions,
        tracking_config,
    )
    colors = {
        str(category_id): {"rgb": list(color), "hex": "#{:02x}{:02x}{:02x}".format(*color)}
        for category_id, color in color_map(categories, all_predictions).items()
    }
    trainer.write_json(output_dir / "class_colors.json", {"categories": categories, "colors": colors})
    if bool(config.get("inference", {}).get("save_predictions_jsonl", True)):
        write_predictions_jsonl(output_dir / "predictions.jsonl", published_predictions)
    if tracking_config.enabled and tracking_config.algorithm == "hybrid":
        write_predictions_jsonl(output_dir / "track_states.jsonl", track_state_rows)
    football_rows = build_standard_football_rows(published_predictions, football_output_config, categories)
    football_summary = write_football_predictions_output(
        output_dir,
        football_rows,
        football_output_config,
    )
    canonical_manifest = canonical_run.finish_manifest() if canonical_run is not None else None
    canonical_summary = {
        "enabled": bool(canonical_cfg.enabled),
        "manifest": (
            (Path(str(canonical_cfg.directory)) / "manifest.json").as_posix()
            if canonical_manifest is not None
            else None
        ),
        "video_count": int(canonical_manifest.get("video_count", 0)) if canonical_manifest else 0,
        "frame_count": int(canonical_manifest.get("frame_count", 0)) if canonical_manifest else 0,
        "detection_count": int(canonical_manifest.get("detection_count", 0)) if canonical_manifest else 0,
        "track_count": int(canonical_manifest.get("track_count", 0)) if canonical_manifest else 0,
        "media_count": int(canonical_manifest.get("media_count", 0)) if canonical_manifest else 0,
    }
    record_video_pipeline_timing(
        model,
        serialization_seconds=time.perf_counter() - serialization_started,
    )
    stage_timing = summarize_inference_timing_rows(model)
    if timing_context is not None:
        timing_context["stage_timing"] = stage_timing
    trainer.write_json(
        output_dir / "inference_summary.json",
        {
            "outputs": outputs,
            "prediction_count": len(published_predictions),
            "raw_prediction_count": len(all_predictions),
            "suppressed_unconfirmed_count": suppressed_unconfirmed_count,
            **football_summary,
            "canonical_v2": canonical_summary,
            "stage_timing": stage_timing,
        },
    )
    if verbose and stage_timing["images_or_frames"]:
        print(
            Fore.BLUE
            + Style.BRIGHT
            + "Inference timing: "
            + f"model={stage_timing['model_forward_ratio'] * 100.0:.2f}%, "
            + f"SAHI={stage_timing['sahi_model_forward_ratio'] * 100.0:.2f}%, "
            + f"recheck={stage_timing['recheck_model_forward_ratio'] * 100.0:.2f}%."
        )
    if tracking_config.enabled:
        tracking_summary = (
            video_tracking.build_hybrid_tracking_summary(all_predictions, track_state_rows)
            if tracking_config.algorithm == "hybrid"
            else video_tracking.build_tracking_summary(all_predictions)
        )
        trainer.write_json(output_dir / "tracking_summary.json", tracking_summary)
    trainer.dump_config_snapshot(
        output_dir=output_dir,
        merged_config=config,
        metadata={
            "event": "inference",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outputs": outputs,
            "acceleration": dict(trainer.get_inference_acceleration_handle(model).metadata),
            "canonical_v2": canonical_summary,
            "stage_timing": stage_timing,
        },
        source_config=source_config,
    )
    if verbose:
        print(f"RF-DETR inference output directory: {output_dir}")
    return 0


def main() -> int:
    """Run inference with elapsed-time reporting."""
    timing_context = trainer.start_run_timing("inference")
    timing_context["cpu_runtime"] = cpu_runtime.current_summary()
    try:
        result = _main_impl(timing_context)
        timing_context["success"] = True
        return result
    except Exception as exc:
        timing_context["success"] = False
        timing_context["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        raise
    finally:
        trainer.finish_run_timing(timing_context)


if __name__ == "__main__":
    raise SystemExit(main())
