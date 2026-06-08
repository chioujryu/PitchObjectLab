r"""
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
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import colorama
import rf_detr_runtime as trainer
import yaml
from colorama import Fore, Style
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator

colorama.init(autoreset=True)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config" / "rf_detr_inference.yaml"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
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
    local_path: Path | None = None


@dataclass(frozen=True)
class VideoFrameWindow:
    start_seconds: float
    end_seconds: float | None
    max_seconds: float | None
    effective_end_seconds: float | None
    start_frame: int
    end_frame: int | None
    output_frames: int | None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


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
    if args.max_sources is not None:
        inference["max_sources"] = args.max_sources
    if args.max_images is not None:
        inference["max_images"] = args.max_images
    if args.max_videos is not None:
        inference["max_videos"] = args.max_videos
    if args.max_seconds is not None:
        inference.setdefault("video", {})["max_seconds"] = args.max_seconds
    if args.video_start_time is not None:
        inference.setdefault("video", {})["start_time"] = args.video_start_time
    if args.video_end_time is not None:
        inference.setdefault("video", {})["end_time"] = args.video_end_time


def config_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None and item != ""]
    return [value]


def parse_video_time_seconds(
    value: Any,
    field_name: str,
    *,
    allow_all: bool = False,
    default: float | None = None,
    positive: bool = False,
) -> float | None:
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
                raise ValueError(
                    f"{field_name} must use numeric SS, MM:SS, or HH:MM:SS format, got {value!r}."
                ) from exc
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
        raise ValueError(
            f"{field_name} must be SS, MM:SS, HH:MM:SS, or a numeric seconds value, got {value!r}."
        ) from exc
    if seconds < 0 or (positive and seconds <= 0):
        comparator = "positive" if positive else "non-negative"
        raise ValueError(f"{field_name} must be {comparator}, got {value!r}.")
    return seconds


def parse_seconds_limit(value: Any, field_name: str = "inference.video.max_seconds") -> float | None:
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
    start_seconds = (
        parse_video_time_seconds(video_cfg.get("start_time", 0), "inference.video.start_time", default=0.0) or 0.0
    )
    end_seconds = parse_video_time_seconds(
        video_cfg.get("end_time", "all"), "inference.video.end_time", allow_all=True, default=None
    )
    max_seconds = parse_seconds_limit(video_cfg.get("max_seconds"))
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("inference.video.end_time must be greater than inference.video.start_time.")

    effective_end_seconds = end_seconds
    if max_seconds is not None:
        max_end_seconds = start_seconds + max_seconds
        effective_end_seconds = (
            min(effective_end_seconds, max_end_seconds) if effective_end_seconds is not None else max_end_seconds
        )

    start_frame = max(0, math.floor(start_seconds * fps))
    if total_frames > 0:
        start_frame = min(start_frame, total_frames)

    end_frame: int | None
    if effective_end_seconds is None:
        end_frame = total_frames if total_frames > 0 else None
    else:
        end_frame = max(0, math.ceil(effective_end_seconds * fps))
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


def limited_video_frame_total(frame_count: int, input_fps: float, max_seconds: float | None) -> int | None:
    """Return the maximum number of frames to process for a video."""
    if max_seconds is None:
        return frame_count if frame_count > 0 else None
    fps = max(0.001, float(input_fps or 0.0))
    seconds_frames = max(1, math.ceil(max_seconds * fps))
    return min(frame_count, seconds_frames) if frame_count > 0 else seconds_frames


def video_detection_frame_count(output_frames: int, input_fps: float, detection_fps: Any) -> int:
    """Estimate how many frames will run model prediction."""
    if output_frames <= 0:
        return 0
    if detection_fps is None:
        return output_frames
    frame_interval = max(1, round(max(0.001, float(input_fps or 0.0)) / max(0.001, float(detection_fps))))
    return math.ceil(output_frames / frame_interval)


def estimate_video_work(item: SourceItem, video_cfg: Mapping[str, Any]) -> dict[str, Any]:
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
        output_frames = max(1, math.ceil(fallback_seconds * input_fps))
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


def media_kind_from_suffix(suffix: str, image_exts: Sequence[str], video_exts: Sequence[str]) -> str | None:
    normalized = suffix.lower()
    if normalized in {ext.lower() for ext in image_exts}:
        return "image"
    if normalized in {ext.lower() for ext in video_exts}:
        return "video"
    return None


def is_url(value: str) -> bool:
    return trainer.is_url_like(value)


def discover_sources(config: Mapping[str, Any]) -> list[SourceItem]:
    inference = config.get("inference", {})
    image_exts = [str(ext).lower() for ext in (inference.get("image_extensions") or sorted(IMAGE_EXTENSIONS))]
    video_exts = [str(ext).lower() for ext in (inference.get("video_extensions") or sorted(VIDEO_EXTENSIONS))]
    recursive = bool(inference.get("recursive", True))
    items: list[SourceItem] = []
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


def apply_source_limits(items: Sequence[SourceItem], config: Mapping[str, Any]) -> list[SourceItem]:
    """Apply first-N source limits from inference config."""
    inference = config.get("inference", {})
    max_images = trainer.parse_limit_value(inference.get("max_images"), "inference.max_images")
    max_videos = trainer.parse_limit_value(inference.get("max_videos"), "inference.max_videos")
    max_sources = trainer.parse_limit_value(inference.get("max_sources"), "inference.max_sources")
    image_count = 0
    video_count = 0
    per_type_limited: list[SourceItem] = []
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
        return trainer.resolve_path_for_output(trainer.render_timestamped(exact, timestamp), [Path.cwd(), PROJECT_DIR])
    root = trainer.render_timestamped(output.get("root", "runs/rf_detr/inference"), timestamp)
    name = trainer.render_timestamped(output.get("name", "rfdetr_inference_{timestamp}"), timestamp)
    return trainer.resolve_path_for_output(str(Path(str(root)) / str(name)), [Path.cwd(), PROJECT_DIR])


def estimate_outputs(items: Sequence[SourceItem], output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    image_count = sum(1 for item in items if item.kind == "image")
    video_count = sum(1 for item in items if item.kind == "video")
    video_cfg = dict(config.get("inference", {}).get("video", {}) or {})
    video_work = [estimate_video_work(item, video_cfg) for item in items if item.kind == "video"]
    detection_frames = sum(int(work["detection_frames"]) for work in video_work)
    output_frames = sum(int(work["output_frames"]) for work in video_work)
    start_seconds = (
        parse_video_time_seconds(video_cfg.get("start_time", 0), "inference.video.start_time", default=0.0) or 0.0
    )
    end_seconds = parse_video_time_seconds(
        video_cfg.get("end_time", "all"), "inference.video.end_time", allow_all=True, default=None
    )
    max_seconds = parse_seconds_limit(video_cfg.get("max_seconds"))
    local_bytes = 0
    for item in items:
        if item.local_path and item.local_path.exists():
            local_bytes += item.local_path.stat().st_size
    output_files = 5 + image_count + video_count
    if bool(config.get("inference", {}).get("save_predictions_jsonl", True)):
        output_files += 1
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
        "estimated_total_files": output_files,
        "estimated_disk_usage": trainer.format_bytes(max(local_bytes, output_files * 500_000)),
        "note": "URL sizes and rendered video sizes are estimated before download/encoding.",
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
        extra_seconds=render_seconds,
    )
    return estimate


def confirm_or_exit(estimate: Mapping[str, Any], verbose: bool, assume_yes: bool) -> None:
    if verbose:
        print(Fore.BLUE + Style.BRIGHT + "Output and resource estimate before RF-DETR inference:")
    print(json.dumps(dict(estimate), indent=2, ensure_ascii=False))
    if assume_yes:
        if verbose:
            print(
                Fore.BLUE + Style.BRIGHT + "Confirmation skipped because --yes or confirm_before_run=false is enabled."
            )
        return
    answer = input(Fore.BLUE + Style.BRIGHT + "Continue and start inference? [y/N]: " + Style.RESET_ALL).strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted before inference output was produced.")


def class_names_from_config(config: Mapping[str, Any]) -> dict[int, str]:
    dataset = config.get("dataset", {})
    names = dataset.get("class_names", dataset.get("names", []))
    if not names and dataset.get("data_yaml"):
        data_yaml = Path(str(dataset["data_yaml"])).expanduser()
        if not data_yaml.is_absolute():
            data_yaml = (Path.cwd() / data_yaml).resolve()
        if data_yaml.exists():
            data = load_yaml(data_yaml)
            names = data.get("names", [])
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def build_categories(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = class_names_from_config(config)
    num_classes = config.get("model", {}).get("num_classes")
    if num_classes is not None:
        for index in range(int(num_classes)):
            names.setdefault(index, str(index))
    return [{"id": int(index), "name": name} for index, name in sorted(names.items())]


def class_color(category_id: int) -> tuple[int, int, int]:
    if category_id < len(COLOR_PALETTE):
        return COLOR_PALETTE[category_id]
    value = (int(category_id) * 2654435761) & 0xFFFFFF
    return 64 + value % 160, 64 + (value >> 8) % 160, 64 + (value >> 16) % 160


def color_map(
    categories: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> dict[int, tuple[int, int, int]]:
    ids = {int(category["id"]) for category in categories}
    ids.update(int(prediction.get("category_id", 0)) for prediction in predictions)
    return {category_id: class_color(category_id) for category_id in sorted(ids)}


def resolve_render_ids(config: Mapping[str, Any], categories: Sequence[Mapping[str, Any]]) -> list[int]:
    inference = config.get("inference", {})
    ids = [int(value) for value in config_list(inference.get("render_class_ids"))]
    if ids:
        return ids
    names = [str(value).casefold() for value in config_list(inference.get("render_class_names"))]
    if not names:
        return []
    name_to_id = {str(category.get("name", category["id"])).casefold(): int(category["id"]) for category in categories}
    return [name_to_id[name] for name in names if name in name_to_id]


def draw_predictions(
    image: Image.Image,
    predictions: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    render_ids: Sequence[int],
) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    id_to_name = {int(category["id"]): str(category.get("name", category["id"])) for category in categories}
    render_set = set(int(value) for value in render_ids)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for prediction in predictions:
        category_id = int(prediction.get("category_id", 0))
        if render_set and category_id not in render_set:
            continue
        x, y, width, height = [float(value) for value in prediction.get("bbox", [0, 0, 0, 0])[:4]]
        color = class_color(category_id)
        draw.rectangle([x, y, x + width, y + height], outline=color, width=2)
        label = f"{id_to_name.get(category_id, category_id)} {float(prediction.get('score', 0.0)):.2f}"
        text_bbox = draw.textbbox((x + 2, y + 2), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luminance > 155 else (255, 255, 255)
        draw.text((x + 2, y + 2), label, fill=text_color, font=font)
    return canvas


def build_prediction_config(config: Mapping[str, Any], categories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model = config.get("model", {})
    inference = config.get("inference", {})
    mode = str(inference.get("mode", "full_image")).strip().lower()
    return {
        "model": {
            "type": "rfdetr",
            "confidence_threshold": float(model.get("confidence_threshold", 0.25)),
            "image_size": model.get("resolution"),
            "category_remapping": model.get("category_remapping", {}),
        },
        "inference": {"mode": mode, "use_sahi": mode == "sahi"},
        "test_mode": {"mode": mode},
        "sahi": dict(config.get("sahi", {}) or {}),
        "crop": dict(config.get("crop", {}) or {}),
        "dataset_categories": list(categories),
        "output": {"visual_format": "jpg"},
        "progress": {"slices": False},
    }


def load_rfdetr_model(config: Mapping[str, Any]) -> Any:
    model_cls = trainer.get_model_class(str(config.get("model", {}).get("size", "medium")))
    return model_cls(**trainer.build_model_kwargs(config))


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
) -> tuple[list[dict[str, Any]], Path]:
    assert item.local_path is not None
    with Image.open(item.local_path) as image:
        width, height = image.size
    record = evaluator.ImageRecord(
        image_id=image_id, file_name=item.local_path.name, path=str(item.local_path), width=width, height=height
    )
    predictions, _, _ = evaluator.predict_image(record, model, prediction_config, output_dir, save_visual=False)
    with Image.open(item.local_path) as image:
        rendered = draw_predictions(image, predictions, categories, render_ids)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / f"{trainer.sanitize_name(item.local_path.stem)}_pred.jpg"
    rendered.save(target, quality=92)
    return predictions, target


def predict_video_file(
    item: SourceItem,
    start_image_id: int,
    model: Any,
    prediction_config: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
    output_dir: Path,
    render_ids: Sequence[int],
    video_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Path, int]:
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
        frame_interval = max(1, round(input_fps / max(0.001, float(detection_fps))))
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

    all_predictions: list[dict[str, Any]] = []
    last_predictions: list[dict[str, Any]] = []
    frame_cache_dir = output_dir / "_frame_cache"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    render_skipped = bool(video_cfg.get("render_skipped_frames", True))
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
                frame_predictions, _, _ = evaluator.predict_image(
                    record, model, prediction_config, output_dir, save_visual=False
                )
                for prediction in frame_predictions:
                    row = dict(prediction)
                    row["source"] = item.source
                    row["frame_index"] = absolute_frame_index
                    row["segment_frame_index"] = segment_frame_index
                    row["timestamp_seconds"] = absolute_frame_index / input_fps
                    row["segment_timestamp_seconds"] = segment_frame_index / input_fps
                    row["video_start_seconds"] = frame_window.start_seconds
                    row["video_end_seconds"] = frame_window.end_seconds
                    row["video_effective_end_seconds"] = frame_window.effective_end_seconds
                    all_predictions.append(row)
                last_predictions = frame_predictions
                image_id += 1
            if should_detect or render_skipped:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = cv2.cvtColor(
                    np_image(draw_predictions(pil_frame, frame_predictions, categories, render_ids)), cv2.COLOR_RGB2BGR
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


def np_image(image: Image.Image) -> Any:
    import numpy as np

    return np.asarray(image.convert("RGB"))


def write_predictions_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=trainer.json_safe_value) + "\n")


def _main_impl(timing_context: MutableMapping[str, Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RF-DETR image/video inference runner.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to rf_detr_inference.yaml.")
    parser.add_argument("--source", action="append", help="Image/video file, folder, or URL. Can be repeated.")
    parser.add_argument("--output-dir", help="Exact output directory override.")
    parser.add_argument("--checkpoint", help="RF-DETR checkpoint/pretrain_weights override.")
    parser.add_argument("--device", help="Device override: auto, cpu, cuda, cuda:0, 0, 1.")
    parser.add_argument("--confidence-threshold", type=float, help="Model confidence threshold override.")
    parser.add_argument(
        "--max-sources", type=trainer.parse_scalar, help="Maximum discovered sources to run. Use all/null for all."
    )
    parser.add_argument(
        "--max-images", type=trainer.parse_scalar, help="Maximum image sources to run. Use all/null for all."
    )
    parser.add_argument(
        "--max-videos", type=trainer.parse_scalar, help="Maximum video sources to run. Use all/null for all."
    )
    parser.add_argument(
        "--max-seconds",
        type=trainer.parse_scalar,
        help="Maximum seconds per video to infer. Use all/null for the whole video.",
    )
    parser.add_argument("--video-start-time", help="Video segment start time. Options: seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument(
        "--video-end-time", help="Video segment end time. Options: all/null, seconds, MM:SS, or HH:MM:SS."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate outputs without inference.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    args = parser.parse_args()

    source_config = Path(args.config).expanduser()
    if not source_config.is_absolute():
        source_config = (Path.cwd() / source_config).resolve()
    config = load_yaml(source_config)
    apply_cli_overrides(config, args)
    verbose = bool(config.get("runtime", {}).get("verbose", True))
    if timing_context is not None:
        timing_context["verbose"] = verbose
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
    if bool(config.get("runtime", {}).get("dry_run", False)):
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if timing_context is not None:
        timing_context["outputs_created"] = True
    cache_dir = output_dir / "source_cache"
    resolved_items = [download_url(item, cache_dir) if item.is_url else item for item in items]
    categories = build_categories(config)
    prediction_config = build_prediction_config(config, categories)
    model = load_rfdetr_model(config)
    render_ids = resolve_render_ids(config, categories)
    all_predictions: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    image_id = 1
    video_cfg = dict(config.get("inference", {}).get("video", {}) or {})

    iterator = tqdm(resolved_items, desc="RF-DETR inference", unit="source")
    for item in iterator:
        if item.kind == "image":
            predictions, path = predict_image_file(
                item, image_id, model, prediction_config, categories, output_dir, render_ids
            )
            for prediction in predictions:
                row = dict(prediction)
                row["source"] = item.source
                all_predictions.append(row)
            outputs.append(
                {"source": item.source, "kind": "image", "output": str(path), "predictions": len(predictions)}
            )
            image_id += 1
        elif item.kind == "video":
            predictions, path, image_id = predict_video_file(
                item, image_id, model, prediction_config, categories, output_dir, render_ids, video_cfg
            )
            all_predictions.extend(predictions)
            outputs.append(
                {"source": item.source, "kind": "video", "output": str(path), "predictions": len(predictions)}
            )

    colors = {
        str(category_id): {"rgb": list(color), "hex": "#{:02x}{:02x}{:02x}".format(*color)}
        for category_id, color in color_map(categories, all_predictions).items()
    }
    trainer.write_json(output_dir / "class_colors.json", {"categories": categories, "colors": colors})
    if bool(config.get("inference", {}).get("save_predictions_jsonl", True)):
        write_predictions_jsonl(output_dir / "predictions.jsonl", all_predictions)
    trainer.write_json(
        output_dir / "inference_summary.json", {"outputs": outputs, "prediction_count": len(all_predictions)}
    )
    trainer.dump_config_snapshot(
        output_dir=output_dir,
        merged_config=config,
        metadata={"event": "inference", "created_at": datetime.now().isoformat(timespec="seconds"), "outputs": outputs},
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
