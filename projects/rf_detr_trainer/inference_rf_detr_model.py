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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import colorama
from colorama import Fore, Style
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import rf_detr_runtime as trainer
import rf_detr_video_tracking as video_tracking
from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator

colorama.init(autoreset=True)

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


def apply_cli_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    runtime = config.setdefault("runtime", {})
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
    metadata_source = "fallback"
    if item.local_path and item.local_path.exists():
        with contextlib.suppress(Exception):
            import cv2

            capture = cv2.VideoCapture(str(item.local_path))
            if capture.isOpened():
                input_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
        "metadata_source": metadata_source,
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
    start_seconds = parse_video_time_seconds(video_cfg.get("start_time", 0), "inference.video.start_time", default=0.0) or 0.0
    end_seconds = parse_video_time_seconds(video_cfg.get("end_time", "all"), "inference.video.end_time", allow_all=True, default=None)
    max_seconds = parse_seconds_limit(video_cfg.get("max_seconds"))
    local_bytes = 0
    for item in items:
        if item.local_path and item.local_path.exists():
            local_bytes += item.local_path.stat().st_size
    output_files = 5 + image_count + video_count
    if bool(config.get("inference", {}).get("save_predictions_jsonl", True)):
        output_files += 1
    if bool(
        ((config.get("inference", {}).get("football_output", {}) or {}).get("enabled", True))
    ):
        output_files += 1
    if bool((config.get("inference", {}).get("tracking", {}) or {}).get("enabled", False)):
        output_files += 1  # tracking_summary.json
    tensorrt_artifacts = trainer.estimate_tensorrt_cache_artifacts(config)
    output_files += int(tensorrt_artifacts["file_count"])
    estimated_bytes = max(local_bytes, output_files * 500_000) + int(tensorrt_artifacts["bytes"])
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
        "tensorrt_cache": tensorrt_artifacts,
        "estimated_total_files": output_files,
        "estimated_disk_usage": trainer.format_bytes(estimated_bytes),
        "note": "URL/rendered-video sizes and first-run TensorRT artifacts in the configured cache are conservative estimates.",
    }
    settings = trainer.runtime_time_estimate_settings(config)
    render_seconds = output_frames * trainer.positive_float_setting(settings, "default_video_render_seconds_per_frame")
    trainer.add_runtime_estimate(
        estimate=estimate,
        config=config,
        output_dir=output_dir,
        task="inference",
        runtime_units=float(image_count + detection_frames),
        default_rate_key="default_inference_seconds_per_image",
        basis={
            "image_sources": image_count,
            "video_sources": video_count,
            "video_detection_frames": detection_frames,
            "video_output_frames": output_frames,
            "video_work": video_work,
        },
        extra_seconds=render_seconds + float(tensorrt_artifacts.get("estimated_build_seconds", 0) or 0),
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
) -> None:
    """Draw trajectory trails, optional search circles, and current-center dots for visible tracks."""
    base_color = class_color(int(target_ids[0])) if target_ids else class_color(1)
    width = max(1, int(tracking_cfg.trajectory_width))
    for track in tracks:
        if not video_tracking.is_track_visible(track, current_frame_index, tracking_cfg):
            continue
        color = track_color(track.track_id) if tracking_cfg.trajectory_per_track_color else base_color
        # Live position: velocity-extrapolated through detection gaps so the ball keeps moving smoothly.
        live_x, live_y = video_tracking.live_center(track, current_frame_index, tracking_cfg)
        # Historical points (age-filtered, already linearly bridged) plus the live head.
        xy = video_tracking.trail_points(track, current_frame_index, tracking_cfg)
        if not xy or xy[-1] != (live_x, live_y):
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
        if tracking_cfg.draw_current_center:
            dot = max(2, width + 1)
            draw.ellipse([live_x - dot, live_y - dot, live_x + dot, live_y + dot], fill=color)


def draw_predictions(
    image: Image.Image,
    predictions: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    render_ids: Sequence[int],
    tracks: Optional[Sequence[Any]] = None,
    tracking_cfg: Optional[Any] = None,
    current_frame_index: Optional[int] = None,
) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    id_to_name = {int(category["id"]): str(category.get("name", category["id"])) for category in categories}
    render_set = set(int(value) for value in render_ids)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    tracking_active = bool(tracks) and tracking_cfg is not None and getattr(tracking_cfg, "enabled", False)
    if tracking_active:
        draw_track_overlays(draw, tracks, tracking_cfg, sorted(tracking_cfg.target_class_ids), current_frame_index)
    for prediction in predictions:
        category_id = int(prediction.get("category_id", 0))
        if render_set and category_id not in render_set:
            continue
        x, y, width, height = [float(value) for value in prediction.get("bbox", [0, 0, 0, 0])[:4]]
        color = class_color(category_id)
        draw.rectangle([x, y, x + width, y + height], outline=color, width=2)
        label = f"{id_to_name.get(category_id, category_id)} {float(prediction.get('score', 0.0)):.2f}"
        if (
            tracking_active
            and tracking_cfg.label_track_id
            and prediction.get("track_id") is not None
            and prediction.get("track_confirmed")
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


def summarize_inference_timing_rows(model: Any) -> Dict[str, Any]:
    """Aggregate per-image evaluator timing into stable stage totals and ratios."""
    rows = getattr(model, "_rf_detr_inference_timing_rows", [])
    if not isinstance(rows, list):
        rows = []

    def total(key: str) -> float:
        return sum(float(row.get(key, 0.0) or 0.0) for row in rows if isinstance(row, Mapping))

    elapsed = total("elapsed_seconds")
    model_forward = total("model_forward_seconds")
    base_forward = total("base_model_forward_seconds")
    sahi_forward = total("sahi_model_forward_seconds")
    recheck_forward = total("recheck_model_forward_seconds")
    preprocess = total("preprocess_seconds")
    postprocess = total("postprocess_seconds")
    return {
        "images_or_frames": len(rows),
        "total_seconds": elapsed,
        "model_forward_seconds": model_forward,
        "base_model_forward_seconds": base_forward,
        "sahi_model_forward_seconds": sahi_forward,
        "recheck_model_forward_seconds": recheck_forward,
        "preprocess_seconds": preprocess,
        "postprocess_seconds": postprocess,
        "model_forward_ratio": model_forward / elapsed if elapsed > 0 else 0.0,
        "sahi_model_forward_ratio": sahi_forward / elapsed if elapsed > 0 else 0.0,
        "recheck_model_forward_ratio": recheck_forward / elapsed if elapsed > 0 else 0.0,
    }


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

    algorithm 'circle' -> built-in FootballTracker; 'ocsort'/'deepocsort'/'botsort'/'bytetrack'
    -> the boxmot adapter. boxmot is imported lazily so it is only required when a boxmot
    algorithm is actually selected.
    """
    if tracking_config is None or not tracking_config.enabled:
        return None
    if getattr(tracking_config, "algorithm", "circle") == "circle":
        return video_tracking.FootballTracker(tracking_config)
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
    configured["sahi"]["batch_size"] = max(1, int(configured["sahi"].get("batch_size", batch_size) or batch_size))
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
) -> Tuple[List[Dict[str, Any]], Path, int]:
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
    if frame_window.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_window.start_frame)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    target = video_dir / f"{trainer.sanitize_name(item.local_path.stem)}_pred.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {target}")

    all_predictions: List[Dict[str, Any]] = []
    last_predictions: List[Dict[str, Any]] = []
    frame_cache_dir = output_dir / "_frame_cache"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    render_skipped = bool(video_cfg.get("render_skipped_frames", True))
    tracker = create_tracker(tracking_config, tracker_device, (width, height))
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
            frame_predictions = last_predictions
            if should_detect:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = frame_cache_dir / "current_frame.jpg"
                Image.fromarray(rgb).save(frame_path, quality=92)
                record = evaluator.ImageRecord(
                    image_id=image_id,
                    file_name=f"{item.local_path.stem}_frame_{absolute_frame_index:06d}.jpg",
                    path=str(frame_path),
                    width=width,
                    height=height,
                )
                frame_predictions, timing, _ = evaluator.predict_image(
                    record, model, prediction_config, output_dir, save_visual=False
                )
                record_inference_timing_rows(model, [timing])
                frame_predictions = filter_final_inference_predictions(frame_predictions, prediction_config)
                if tracker is not None:
                    frame_predictions = tracker.update(absolute_frame_index, frame_predictions, frame=frame)
                for prediction in frame_predictions:
                    all_predictions.append(
                        build_video_row(prediction, item.source, absolute_frame_index, segment_frame_index, input_fps, frame_window)
                    )
                last_predictions = frame_predictions
                image_id += 1
            if should_detect or render_skipped:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = cv2.cvtColor(
                    np_image(
                        draw_predictions(
                            pil_frame,
                            frame_predictions,
                            categories,
                            render_ids,
                            tracker.tracks if tracker is not None else None,
                            tracking_config,
                            absolute_frame_index,
                        )
                    ),
                    cv2.COLOR_RGB2BGR,
                )
            writer.write(frame)
            segment_frame_index += 1
            absolute_frame_index += 1
            iterator.update(1)
    finally:
        iterator.close()
        capture.release()
        writer.release()
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
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
) -> Tuple[List[Dict[str, Any]], Path, int]:
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
                Image.fromarray(rgb).save(frame_path, quality=92)
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
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    target = video_dir / f"{trainer.sanitize_name(item.local_path.stem)}_pred.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        render_capture.release()
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
        raise RuntimeError(f"Could not create video writer: {target}")

    render_skipped = bool(video_cfg.get("render_skipped_frames", True))
    tracker = create_tracker(tracking_config, tracker_device, (width, height))
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
            frame_predictions = last_predictions
            if should_detect:
                frame_predictions = predictions_by_segment.get(segment_frame_index, [])
                if tracker is not None:
                    frame_predictions = tracker.update(absolute_frame_index, frame_predictions, frame=frame)
                for prediction in frame_predictions:
                    all_predictions.append(
                        build_video_row(prediction, item.source, absolute_frame_index, segment_frame_index, input_fps, frame_window)
                    )
                last_predictions = frame_predictions
            if should_detect or render_skipped:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = cv2.cvtColor(
                    np_image(
                        draw_predictions(
                            pil_frame,
                            frame_predictions,
                            categories,
                            render_ids,
                            tracker.tracks if tracker is not None else None,
                            tracking_config,
                            absolute_frame_index,
                        )
                    ),
                    cv2.COLOR_RGB2BGR,
                )
            writer.write(frame)
            segment_frame_index += 1
            render_iterator.update(1)
    finally:
        render_iterator.close()
        render_capture.release()
        writer.release()
        shutil.rmtree(frame_cache_dir, ignore_errors=True)
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
) -> Tuple[List[Dict[str, Any]], Path, int]:
    """Predict one video with batched detection frames when configured."""
    configured_batch = positive_batch_size(video_cfg.get("batch_size"), "inference.video.batch_size", 1)
    if configured_batch <= 1:
        return predict_video_file_one_pass(
            item, start_image_id, model, prediction_config, categories, output_dir, render_ids, video_cfg,
            tracking_config, tracker_device,
        )
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
            tracker_device,
        )
    except RuntimeError as exc:
        if "batched" in str(exc).lower() or "reopen video" in str(exc).lower() or "seek video" in str(exc).lower():
            print(Fore.BLUE + Style.BRIGHT + f"Warning: batched video inference fell back to one-pass mode. {exc}")
            return predict_video_file_one_pass(
                item, start_image_id, model, prediction_config, categories, output_dir, render_ids, video_cfg,
                tracking_config, tracker_device,
            )
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
    parser.add_argument("--max-seconds", type=trainer.parse_scalar, help="Maximum seconds per video to infer. Use all/null for the whole video.")
    parser.add_argument("--video-start-time", help="Video segment start time. Options: seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument("--video-end-time", help="Video segment end time. Options: all/null, seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument("--track", action="store_true", help="Enable football tracking for video inference.")
    parser.add_argument("--no-track", dest="no_track", action="store_true", help="Disable football tracking (overrides config and --track).")
    parser.add_argument("--track-radius", dest="track_radius", type=float, help="Override inference.tracking.radius_pixels (search radius in pixels).")
    parser.add_argument("--track-velocity", dest="track_velocity", action="store_true", help="Enable the velocity-predicted gate for the circle tracker.")
    parser.add_argument("--tracker", dest="tracker", choices=["circle", "ocsort", "deepocsort", "botsort", "bytetrack"], help="Override inference.tracking.algorithm.")
    parser.add_argument("--reid-weights", dest="reid_weights", help="Override inference.tracking.reid_weights (local ReID .pt path for deepocsort/botsort).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate outputs without inference.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    args = parser.parse_args()

    source_config = Path(args.config).expanduser()
    if not source_config.is_absolute():
        source_config = (Path.cwd() / source_config).resolve()
    config = load_yaml(source_config)
    apply_cli_overrides(config, args)
    trainer._require_custom_architecture_checkpoint(config, "Inference")
    trainer.validate_inference_acceleration_config(config)
    categories = build_categories(config)
    football_output_config = parse_football_output_config(config, categories)
    verbose = bool(config.get("runtime", {}).get("verbose", True))
    if timing_context is not None:
        timing_context["verbose"] = verbose
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
    image_id = 1
    video_cfg = dict(config.get("inference", {}).get("video", {}) or {})
    video_cfg["batch_size"] = video_batch_size(config)
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
            predictions, path, image_id = predict_video_file(item, image_id, model, prediction_config, categories, output_dir, render_ids, video_cfg, tracking_config, tracker_device)
            all_predictions.extend(predictions)
            outputs.append({"source": item.source, "kind": "video", "output": str(path), "predictions": len(predictions)})
    flush_pending_images()

    colors = {
        str(category_id): {"rgb": list(color), "hex": "#{:02x}{:02x}{:02x}".format(*color)}
        for category_id, color in color_map(categories, all_predictions).items()
    }
    trainer.write_json(output_dir / "class_colors.json", {"categories": categories, "colors": colors})
    if bool(config.get("inference", {}).get("save_predictions_jsonl", True)):
        write_predictions_jsonl(output_dir / "predictions.jsonl", all_predictions)
    football_rows = build_standard_football_rows(all_predictions, football_output_config, categories)
    football_summary = write_football_predictions_output(
        output_dir,
        football_rows,
        football_output_config,
    )
    stage_timing = summarize_inference_timing_rows(model)
    if timing_context is not None:
        timing_context["stage_timing"] = stage_timing
    trainer.write_json(
        output_dir / "inference_summary.json",
        {
            "outputs": outputs,
            "prediction_count": len(all_predictions),
            **football_summary,
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
        trainer.write_json(output_dir / "tracking_summary.json", video_tracking.build_tracking_summary(all_predictions))
    trainer.dump_config_snapshot(
        output_dir=output_dir,
        merged_config=config,
        metadata={
            "event": "inference",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outputs": outputs,
            "acceleration": dict(trainer.get_inference_acceleration_handle(model).metadata),
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
