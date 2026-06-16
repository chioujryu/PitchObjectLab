"""
Run object-detection inference on a dataset and export COCO/Ultralytics-style validation metrics.

This script is a config-first evaluator for YOLO-format or COCO-format detection datasets. It supports:

1. Optional SAHI sliced inference with configurable slice/window size, overlap, merge strategy, and batch size.
2. Direct Ultralytics full-image inference when SAHI is disabled.
3. COCO mAP metrics: mAP50-95, mAP50, mAP75, AP small/medium/large, AR@maxDets, and per-class AP.
4. Operating-point Precision, Recall, F1, TP, FP, FN, per-image metrics, and confusion matrices.
5. Recall-Precision, Precision/Recall/F1 confidence curves, CSV/JSON tables, optional random visual samples,
   and dataset case images under output/datasets.
6. A pre-run resource estimate and developer confirmation before output files are written.
7. Reproducible output folders that include resolved config snapshots and config hashes.
8. Linux and Windows paths.

Example usage:

    uv run python object_detection_dataset_evaluator.py \
        --config config/object_detection_dataset_evaluate.yaml \
        --yes

    uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py \
        --config projects/object_detection_dataset_evaluator/config/object_detection_dataset_evaluate.yaml \
        --use-sahi \
        --model-path yolo26m.pt \
        --data-yaml ultralytics/cfg/datasets/coco8.yaml \
        --split val \
        --device cuda:0 \
        --slice-height 640 \
        --slice-width 640 \
        --overlap-height-ratio 0.2 \
        --overlap-width-ratio 0.2 \
        --save-visuals \
        --max-visuals 20

    uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py \
        --config projects/object_detection_dataset_evaluator/config/object_detection_dataset_evaluate.yaml \
        --no-sahi \
        --dataset-format coco \
        --coco-json /datasets/my_coco/annotations/instances_test.json \
        --image-dir /datasets/my_coco/images/test \
        --model-path runs/detect/train/weights/best.pt \
        --device cpu \
        --demo \
        --yes

Notes:
    - For best confidence curves, use a low model.confidence_threshold such as 0.001 or 0.01.
      Higher thresholds are faster and produce smaller JSON files, but curves only cover predictions
      that survived the model threshold.
    - Multi-GPU mode uses image-level multiprocessing. Each device loads its own model copy.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import io
import json
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml
import colorama
from colorama import Fore, Style
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from tqdm import tqdm

colorama.init(autoreset=True)

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_common import test_modes as shared_modes  # noqa: E402

DEFAULT_CONFIG = PROJECT_DIR / "config" / "object_detection_dataset_evaluate.yaml"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_.-]+)\}")
BACKGROUND_LABEL = "__background__"
FOOTBALL_ALIASES = ("football", "soccer_ball", "soccer ball", "ball", "足球")
VISUAL_COLOR_NAMES = {
    "red": (255, 56, 56),
    "green": (20, 184, 116),
    "blue": (59, 130, 246),
    "yellow": (245, 158, 11),
    "cyan": (6, 182, 212),
    "magenta": (217, 70, 239),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


@dataclass(frozen=True)
class ImageRecord:
    """Resolved image record used for inference."""

    image_id: int
    file_name: str
    path: str
    width: int
    height: int


@dataclass
class DatasetBundle:
    """In-memory COCO-compatible dataset plus resolved image paths."""

    images: List[ImageRecord]
    categories: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    coco: Dict[str, Any]
    source_kind: str


def blue(message: str, verbose: bool = True, force: bool = False) -> None:
    """Print blue English status text."""
    if force or verbose:
        print(Fore.BLUE + Style.BRIGHT + message)


def parse_scalar(value: str) -> Any:
    """Parse CLI key=value values with YAML scalar/list/dict support."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def json_default(value: Any) -> Any:
    """JSON serializer for numpy/path values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return repr(value)


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge dictionaries in-place."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_nested(mapping: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested dictionary value with dot notation."""
    target: MutableMapping[str, Any] = mapping
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError("Empty override key.")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML into a dictionary."""
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def save_yaml(path: Path, data: Mapping[str, Any]) -> None:
    """Write YAML with stable key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(data), file, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: Any) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=json_default)


def is_url_like(value: str) -> bool:
    """Detect URLs that should not be path-normalized."""
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value))


def is_abs_any_os(value: str) -> bool:
    """Return True for Windows or POSIX absolute paths regardless of current OS."""
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def resolve_path(value: Any, bases: Sequence[Path], must_exist: bool = False) -> Path:
    """Resolve a local path against likely base directories."""
    if value is None:
        raise ValueError("Cannot resolve a null path.")
    text = str(value).strip()
    if not text:
        raise ValueError("Cannot resolve an empty path.")
    path = Path(text).expanduser()
    if path.exists() or is_abs_any_os(text):
        resolved = path.resolve() if path.exists() else path
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        return resolved

    candidates = [(base / text).expanduser() for base in bases]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Path does not exist: {text}. Checked: {checked}")
    return candidates[0]


def resolve_existing_or_raw(value: Any, bases: Sequence[Path]) -> Any:
    """Resolve an existing local path; leave model names and URLs unchanged."""
    if value is None or isinstance(value, (bool, int, float, list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text or is_url_like(text):
        return value
    try:
        return str(resolve_path(text, bases, must_exist=True))
    except FileNotFoundError:
        return value


def render_template(value: Any, replacements: Mapping[str, Any]) -> Any:
    """Render folder-name placeholders in strings."""
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in replacements:
            return match.group(0)
        return str(replacements[key])

    return TEMPLATE_PLACEHOLDER_RE.sub(replace, value)


def sanitize_name(text: str) -> str:
    """Make a safe filename fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "run"


def template_fragment(value: Any) -> str:
    """Convert a config value into a safe folder-name fragment."""
    if value is None:
        text = "none"
    elif isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, (list, tuple)):
        text = "-".join(template_fragment(item) for item in value) or "empty"
    elif isinstance(value, Mapping):
        text = "-".join(f"{key}_{template_fragment(item)}" for key, item in value.items()) or "mapping"
    else:
        text = str(value)
    return sanitize_name(text)


def build_template_replacements(config: Mapping[str, Any], extra: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """Build placeholder values from config dot-paths plus unambiguous leaf aliases."""
    replacements: Dict[str, str] = {}
    leaf_values: Dict[str, List[Any]] = {}

    def visit(node: Any, prefix: str = "") -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                visit(item, child)
            return
        if not prefix:
            return
        replacements[prefix] = template_fragment(node)
        leaf_values.setdefault(prefix.rsplit(".", 1)[-1], []).append(node)

    visit(config)
    for key, values in leaf_values.items():
        if key not in replacements and len(values) == 1:
            replacements[key] = template_fragment(values[0])
    if extra:
        for key, value in extra.items():
            replacements[str(key)] = template_fragment(value)
    return replacements


def increment_path(path: Path) -> Path:
    """Return a non-existing path by appending _2, _3, ... when needed."""
    if not path.exists():
        return path
    parent = path.parent
    stem = path.name
    for index in range(2, 10000):
        candidate = parent / f"{stem}_{index}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique output directory under {parent}")


def resolve_output_root(value: Any, source_config: Path) -> Path:
    """Resolve output roots with sensible repo/project-relative behavior."""
    text = str(value or "outputs").strip()
    if not text:
        text = "outputs"
    if is_abs_any_os(text):
        return Path(text).expanduser()
    normalized = text.replace("\\", "/")
    first_part = normalized.split("/", 1)[0]
    if first_part in {"projects", "ultralytics", "docs", "examples", "tests", "runs"}:
        return (REPO_ROOT / text).expanduser()
    if first_part in {"outputs", "demo_outputs", "generated", "tmp"}:
        return (PROJECT_DIR / text).expanduser()
    return resolve_path(text, [PROJECT_DIR, REPO_ROOT, source_config.parent, Path.cwd()], must_exist=False)


def normalize_device(value: Any) -> str:
    """Normalize device strings for SAHI/PyTorch."""
    text = str(value).strip()
    if not text:
        return "cpu"
    lower = text.lower()
    if lower == "cuda":
        return "cuda:0"
    if lower in {"cpu", "mps"} or lower.startswith("cuda:"):
        return lower
    if re.fullmatch(r"\d+", text):
        return f"cuda:{text}"
    return text


def parse_devices(model_cfg: Mapping[str, Any]) -> List[str]:
    """Return one or more normalized devices."""
    devices = model_cfg.get("devices") or []
    if isinstance(devices, str):
        devices = [part.strip() for part in devices.split(",") if part.strip()]
    if devices:
        return [normalize_device(device) for device in devices]
    device = str(model_cfg.get("device", "cpu"))
    if "," in device and not device.lower().startswith("cuda:"):
        return [normalize_device(part) for part in device.split(",") if part.strip()]
    return [normalize_device(device)]


def cuda_index(device: str) -> Optional[int]:
    """Return the CUDA ordinal from a normalized device string."""
    text = str(device).strip().lower()
    if text == "cuda":
        return 0
    match = re.fullmatch(r"cuda:(\d+)", text)
    return int(match.group(1)) if match else None


def validate_runtime_devices(config: Mapping[str, Any], verbose: bool = True) -> None:
    """Fail early with a clear message when configured CUDA devices are not visible."""
    devices = parse_devices(config["model"])
    requested = [(device, cuda_index(device)) for device in devices]
    requested_cuda = [(device, index) for device, index in requested if index is not None]
    if not requested_cuda:
        return

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "CUDA device was requested, but PyTorch could not be imported. "
            "Run `uv sync` in projects/object_detection_dataset_evaluator, or set model.device: cpu."
        ) from exc

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is false. "
            f"Requested devices: {', '.join(device for device, _ in requested_cuda)}. "
            f"CUDA_VISIBLE_DEVICES={visible_env or '<not set>'}. "
            "Set model.device: cpu, or fix the CUDA/PyTorch environment."
        )

    count = int(torch.cuda.device_count())
    available = []
    for index in range(count):
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:
            name = "unknown"
        available.append(f"cuda:{index}={name}")

    invalid = [device for device, index in requested_cuda if index is None or index < 0 or index >= count]
    if invalid:
        available_text = ", ".join(available) if available else "<none>"
        raise RuntimeError(
            "Invalid CUDA device config. "
            f"Requested: {', '.join(invalid)}. "
            f"PyTorch can see {count} CUDA device(s): {available_text}. "
            f"CUDA_VISIBLE_DEVICES={visible_env or '<not set>'}. "
            "Device IDs are relative to CUDA_VISIBLE_DEVICES. "
            "For example, if you run `CUDA_VISIBLE_DEVICES=6 ...`, set model.device to cuda:0 inside this config. "
            "Otherwise choose one of the visible devices above, or use model.device: cpu."
        )

    blue(f"CUDA devices verified: {', '.join(device for device, _ in requested_cuda)}", verbose=verbose)


def config_hash(config: Mapping[str, Any]) -> str:
    """Create a short stable config hash."""
    payload = json.dumps(config, sort_keys=True, default=json_default, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def use_sahi_inference(config: Mapping[str, Any]) -> bool:
    """Return whether inference should use SAHI."""
    return shared_modes.is_sahi_mode(config)


def inference_engine_name(config: Mapping[str, Any]) -> str:
    """Return the normalized inference engine name."""
    mode = shared_modes.canonical_test_mode(config)
    if mode == shared_modes.SAHI_MODE:
        return "sahi"
    if mode == shared_modes.CLASS_CROP_MODE:
        return "class_crop"
    return str(config.get("model", {}).get("type", "ultralytics")).strip().lower()


def call_with_supported_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a function after dropping unsupported keyword arguments."""
    signature = inspect.signature(func)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return func(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(*args, **supported)


def image_size(path: Path) -> Tuple[int, int]:
    """Read image width and height with PIL."""
    with Image.open(path) as image:
        return image.size


def normalize_names(names: Any) -> Dict[int, str]:
    """Normalize YOLO names list/dict to {category_id: name}."""
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    raise ValueError("Dataset names must be a list or mapping.")


def categories_from_names(names: Mapping[int, str]) -> List[Dict[str, Any]]:
    """Create COCO categories from class names."""
    return [{"id": int(category_id), "name": str(name), "supercategory": "object"} for category_id, name in sorted(names.items())]


def resolve_yolo_root(yolo_cfg: Mapping[str, Any], yaml_path: Optional[Path]) -> Path:
    """Resolve the dataset root from a YOLO data YAML."""
    bases = [PROJECT_DIR, REPO_ROOT, Path.cwd()]
    if yaml_path is not None:
        bases.insert(0, yaml_path.parent)
    raw_root = yolo_cfg.get("path") or (yaml_path.parent if yaml_path else Path.cwd())
    if isinstance(raw_root, (list, tuple)):
        raw_root = raw_root[0]
    if str(raw_root).strip():
        return resolve_path(raw_root, bases, must_exist=False)
    return yaml_path.parent if yaml_path else Path.cwd()


def resolve_split_item(item: Any, root: Path, extra_bases: Sequence[Path]) -> Path:
    """Resolve one YOLO split source item."""
    text = str(item).strip()
    if is_abs_any_os(text):
        return Path(text).expanduser()
    for base in [root, *extra_bases, PROJECT_DIR, REPO_ROOT, Path.cwd()]:
        candidate = (base / text).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return (root / text).expanduser()


def list_images_from_source(source: Any, root: Path, extra_bases: Sequence[Path], sort_images: bool) -> List[Path]:
    """Collect images from a YOLO split source folder, text file, or list."""
    if isinstance(source, (list, tuple)):
        images: List[Path] = []
        for item in source:
            images.extend(list_images_from_source(item, root, extra_bases, sort_images=False))
        return sorted(images) if sort_images else images

    source_path = resolve_split_item(source, root, extra_bases)
    if source_path.is_file() and source_path.suffix.lower() == ".txt":
        images = []
        lines = source_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_path = Path(line).expanduser()
            if not image_path.exists() and not is_abs_any_os(line):
                for base in [source_path.parent, root, *extra_bases]:
                    candidate = (base / line).expanduser()
                    if candidate.exists():
                        image_path = candidate.resolve()
                        break
            images.append(image_path)
        return sorted(images) if sort_images else images

    if source_path.is_dir():
        images = [path for path in source_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(images) if sort_images else images

    if source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [source_path]

    raise FileNotFoundError(f"Could not collect images from split source: {source_path}")


def replace_images_with_labels(path: Path) -> Optional[Path]:
    """Infer a YOLO label path by replacing the last images path component with labels."""
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if "images" not in lowered:
        return None
    index = len(lowered) - 1 - lowered[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def infer_label_path(image_path: Path, labels_dir: Optional[Path]) -> Path:
    """Resolve a YOLO label path for an image."""
    if labels_dir is not None:
        direct = labels_dir / f"{image_path.stem}.txt"
        if direct.exists():
            return direct
        inferred = replace_images_with_labels(image_path)
        if inferred is not None:
            lowered = [part.lower() for part in inferred.parts]
            if "labels" in lowered:
                label_index = len(lowered) - 1 - lowered[::-1].index("labels")
                rel = Path(*inferred.parts[label_index + 1 :])
            else:
                rel = Path(inferred.name)
            candidate = labels_dir / rel
            if candidate.exists():
                return candidate
        return direct
    inferred = replace_images_with_labels(image_path)
    if inferred is not None:
        return inferred
    return image_path.with_suffix(".txt")


def yolo_line_to_coco_bbox(line: str, width: int, height: int) -> Optional[Tuple[int, List[float], float]]:
    """Convert one YOLO label line to COCO category_id, bbox, area."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        category_id = int(float(parts[0]))
        x_center, y_center, box_width, box_height = map(float, parts[1:5])
    except ValueError:
        return None

    x = (x_center - box_width / 2.0) * width
    y = (y_center - box_height / 2.0) * height
    w = box_width * width
    h = box_height * height
    x = max(0.0, min(float(width), x))
    y = max(0.0, min(float(height), y))
    w = max(0.0, min(float(width) - x, w))
    h = max(0.0, min(float(height) - y, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return category_id, [x, y, w, h], w * h


def load_yolo_dataset(config: Mapping[str, Any], output_info: Mapping[str, Any]) -> DatasetBundle:
    """Load YOLO-format labels and convert them to an in-memory COCO dataset."""
    dataset_cfg = config["dataset"]
    bases = [PROJECT_DIR, REPO_ROOT, Path.cwd()]
    yolo_cfg: Dict[str, Any]
    yaml_path: Optional[Path] = None

    data_yaml = str(dataset_cfg.get("data_yaml") or "").strip()
    if data_yaml:
        yaml_path = resolve_path(data_yaml, bases, must_exist=True)
        yolo_cfg = load_yaml(yaml_path)
    else:
        yolo_cfg = {
            "path": dataset_cfg.get("path") or "",
            "train": dataset_cfg.get("train"),
            "val": dataset_cfg.get("val"),
            "test": dataset_cfg.get("test"),
            "names": dataset_cfg.get("names"),
        }

    root = resolve_yolo_root(yolo_cfg, yaml_path)
    split = str(dataset_cfg.get("split", "test"))
    if split not in yolo_cfg:
        raise KeyError(f"Split {split!r} was not found in YOLO config.")
    split_source = yolo_cfg.get(split)
    if split_source is None or split_source == "":
        raise ValueError(f"Split {split!r} is empty in the YOLO config. Choose another split or fill this split path.")
    names = normalize_names(yolo_cfg.get("names") or dataset_cfg.get("names"))
    categories = categories_from_names(names)
    labels_dir_value = str(dataset_cfg.get("labels_dir") or "").strip()
    labels_dir = resolve_path(labels_dir_value, [root, PROJECT_DIR, REPO_ROOT, Path.cwd()], must_exist=False) if labels_dir_value else None

    image_paths = list_images_from_source(split_source, root, [yaml_path.parent] if yaml_path else [], bool(dataset_cfg.get("sort_images", True)))
    max_images = dataset_cfg.get("max_images")
    if max_images is not None:
        image_paths = image_paths[: int(max_images)]

    include_empty = bool(dataset_cfg.get("include_empty_images", True))
    images: List[ImageRecord] = []
    annotations: List[Dict[str, Any]] = []
    annotation_id = 1
    image_id = 1

    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Image does not exist: {path}")
        width, height = image_size(path)
        label_path = infer_label_path(path, labels_dir)
        image_annotations: List[Dict[str, Any]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                converted = yolo_line_to_coco_bbox(line, width, height)
                if converted is None:
                    continue
                category_id, bbox, area = converted
                image_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [float(v) for v in bbox],
                        "area": float(area),
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                annotation_id += 1

        if image_annotations or include_empty:
            try:
                file_name = path.relative_to(root).as_posix()
            except ValueError:
                file_name = path.name
            images.append(ImageRecord(image_id=image_id, file_name=file_name, path=str(path.resolve()), width=width, height=height))
            annotations.extend(image_annotations)
            image_id += 1

    coco = {
        "info": {
            "description": "Generated from YOLO labels by object_detection_dataset_evaluator.py",
            "source": str(yaml_path or root),
            "split": split,
            "run_id": output_info["run_id"],
            "config_hash": output_info["config_hash"],
        },
        "licenses": [],
        "images": [
            {"id": image.image_id, "file_name": image.file_name, "width": image.width, "height": image.height}
            for image in images
        ],
        "annotations": annotations,
        "categories": categories,
    }
    return DatasetBundle(images=images, categories=categories, annotations=annotations, coco=coco, source_kind="yolo")


def resolve_coco_image_path(file_name: str, image_dir: Path, annotation_dir: Path) -> Path:
    """Resolve one COCO image path."""
    raw = Path(file_name).expanduser()
    if raw.exists():
        return raw.resolve()
    candidates = [image_dir / file_name, annotation_dir / file_name, image_dir / raw.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve COCO image: {file_name}")


def load_coco_dataset(config: Mapping[str, Any], output_info: Mapping[str, Any]) -> DatasetBundle:
    """Load a COCO-format dataset."""
    dataset_cfg = config["dataset"]
    bases = [PROJECT_DIR, REPO_ROOT, Path.cwd()]
    coco_json = resolve_path(dataset_cfg.get("coco_json"), bases, must_exist=True)
    image_dir = resolve_path(dataset_cfg.get("image_dir"), [coco_json.parent, PROJECT_DIR, REPO_ROOT, Path.cwd()], must_exist=True)
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    selected_images = list(data.get("images", []))
    if bool(dataset_cfg.get("sort_images", True)):
        selected_images = sorted(selected_images, key=lambda item: str(item.get("file_name", "")))
    max_images = dataset_cfg.get("max_images")
    if max_images is not None:
        selected_images = selected_images[: int(max_images)]
    selected_ids = {image["id"] for image in selected_images}

    images: List[ImageRecord] = []
    for image in selected_images:
        path = resolve_coco_image_path(str(image["file_name"]), image_dir, coco_json.parent)
        width = image.get("width")
        height = image.get("height")
        if width is None or height is None:
            width, height = image_size(path)
        images.append(
            ImageRecord(
                image_id=int(image["id"]),
                file_name=str(image["file_name"]),
                path=str(path),
                width=int(width),
                height=int(height),
            )
        )

    annotations = [annotation for annotation in data.get("annotations", []) if annotation.get("image_id") in selected_ids]
    categories = list(data.get("categories", []))
    coco = {
        "info": {
            **(data.get("info") or {}),
            "description": "Normalized COCO subset by object_detection_dataset_evaluator.py",
            "source": str(coco_json),
            "run_id": output_info["run_id"],
            "config_hash": output_info["config_hash"],
        },
        "licenses": data.get("licenses", []),
        "images": [
            {"id": image.image_id, "file_name": image.file_name, "width": image.width, "height": image.height}
            for image in images
        ],
        "annotations": annotations,
        "categories": categories,
    }
    return DatasetBundle(images=images, categories=categories, annotations=annotations, coco=coco, source_kind="coco")


def load_dataset(config: Mapping[str, Any], output_info: Mapping[str, Any]) -> DatasetBundle:
    """Load either a YOLO or COCO dataset."""
    dataset_format = str(config["dataset"].get("format", "yolo")).lower()
    if dataset_format == "yolo":
        return load_yolo_dataset(config, output_info)
    if dataset_format == "coco":
        return load_coco_dataset(config, output_info)
    raise ValueError(f"Unsupported dataset.format: {dataset_format}")


def prediction_to_clean_coco(prediction: Mapping[str, Any], image_id: int) -> Dict[str, Any]:
    """Normalize one SAHI COCO prediction to JSON-friendly values."""
    bbox = [float(value) for value in prediction.get("bbox", [0, 0, 0, 0])]
    return {
        "image_id": int(prediction.get("image_id", image_id)),
        "category_id": int(prediction.get("category_id", 0)),
        "bbox": bbox,
        "score": float(prediction.get("score", 0.0)),
        "area": float(prediction.get("area", bbox[2] * bbox[3] if len(bbox) == 4 else 0.0)),
    }


def config_list(value: Any) -> List[Any]:
    """Normalize YAML/CLI scalar or sequence values to a flat list."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parsed = parse_scalar(value)
        if parsed is not value and not isinstance(parsed, str):
            return config_list(parsed)
        return [part.strip() for part in re.split(r"[,;\n]", value) if part.strip()]
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None and item != ""]
    return [value]


def category_maps(categories: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Return category id/name lookup maps."""
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    for category in categories:
        category_id = int(category["id"])
        name = str(category.get("name", category_id))
        id_to_name[category_id] = name
        name_to_id[name.casefold()] = category_id
    return id_to_name, name_to_id


def resolve_category_class_ids(
    categories: Sequence[Mapping[str, Any]],
    class_ids: Any,
    class_names: Any,
    context: str,
    default_names: Optional[Sequence[str]] = None,
) -> List[int]:
    """Resolve category ids from explicit ids/names with optional football-friendly defaults."""
    id_to_name, name_to_id = category_maps(categories)
    selected: List[int] = []
    seen = set()

    for value in config_list(class_ids):
        category_id = int(value)
        if category_id not in id_to_name:
            raise ValueError(f"Unknown {context} category id: {category_id}")
        if category_id not in seen:
            selected.append(category_id)
            seen.add(category_id)

    names = config_list(class_names)
    if not selected and not names and default_names:
        names = list(default_names)
    for value in names:
        name = str(value).strip()
        category_id = name_to_id.get(name.casefold())
        if category_id is None and name.casefold() == "football":
            for alias in FOOTBALL_ALIASES:
                category_id = name_to_id.get(alias.casefold())
                if category_id is not None:
                    break
        if category_id is None:
            available = ", ".join(id_to_name.values())
            raise ValueError(f"Unknown {context} category name: {name!r}. Available names: {available}")
        if category_id not in seen:
            selected.append(category_id)
            seen.add(category_id)

    return selected


def resolve_visual_filter_class_ids(categories: Sequence[Mapping[str, Any]], output_cfg: Mapping[str, Any]) -> List[int]:
    """Resolve visual class filters from ids and names."""
    return resolve_category_class_ids(
        categories,
        output_cfg.get("visual_filter_class_ids"),
        output_cfg.get("visual_filter_class_names"),
        "visual filter",
    )


def resolve_visual_render_class_ids(categories: Sequence[Mapping[str, Any]], output_cfg: Mapping[str, Any]) -> List[int]:
    """Resolve classes drawn in visual sample images; empty means draw every class."""
    return resolve_category_class_ids(
        categories,
        output_cfg.get("visual_render_class_ids"),
        output_cfg.get("visual_render_class_names"),
        "visual render",
    )


def resolve_error_case_render_class_ids(categories: Sequence[Mapping[str, Any]], error_cfg: Mapping[str, Any]) -> List[int]:
    """Resolve classes drawn in error-case images; empty means draw every class."""
    return resolve_category_class_ids(
        categories,
        error_cfg.get("render_class_ids"),
        error_cfg.get("render_class_names"),
        "error-case render",
    )


def build_category_presence_index(
    rows: Sequence[Mapping[str, Any]],
    score_threshold: Optional[float] = None,
) -> Tuple[Dict[int, set], Dict[int, int]]:
    """Build image_id -> category set/count indexes in one pass."""
    category_sets: Dict[int, set] = {}
    counts: Dict[int, int] = {}
    for row in rows:
        if score_threshold is not None and float(row.get("score", 1.0)) < score_threshold:
            continue
        image_id = int(row["image_id"])
        category_id = int(row["category_id"])
        category_sets.setdefault(image_id, set()).add(category_id)
        counts[image_id] = counts.get(image_id, 0) + 1
    return category_sets, counts


def class_filter_matches(present_ids: set, required_ids: Sequence[int], mode: str) -> bool:
    """Return whether an image category set matches the requested class filter."""
    if not required_ids:
        return True
    required = set(int(category_id) for category_id in required_ids)
    if mode == "all":
        return required.issubset(present_ids)
    return bool(required.intersection(present_ids))


def get_visual_seed(config: Mapping[str, Any]) -> int:
    """Return the visual sampling seed."""
    output_cfg = config["output"]
    value = output_cfg.get("visual_random_seed")
    if value is None:
        value = config.get("runtime", {}).get("seed", 0)
    return int(value)


def select_visual_images(
    dataset: DatasetBundle,
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[List[ImageRecord], Dict[str, Any]]:
    """Select visual output images from class-filtered candidates."""
    output_cfg = config["output"]
    if not bool(output_cfg.get("save_visuals", False)):
        return [], {"enabled": False, "candidate_count": 0, "selected_count": 0}

    filter_source = str(output_cfg.get("visual_filter_source", "ground_truth")).strip().lower()
    if filter_source not in {"ground_truth", "prediction", "either", "both"}:
        raise ValueError("output.visual_filter_source must be one of: ground_truth, prediction, either, both.")
    filter_match = str(output_cfg.get("visual_filter_match", "any")).strip().lower()
    if filter_match not in {"any", "all"}:
        raise ValueError("output.visual_filter_match must be one of: any, all.")

    required_ids = resolve_visual_filter_class_ids(dataset.categories, output_cfg)
    id_to_name, _ = category_maps(dataset.categories)
    min_gt = max(0, int(output_cfg.get("visual_min_gt_instances", 0)))
    min_pred = max(0, int(output_cfg.get("visual_min_predictions", 0)))
    prediction_score = output_cfg.get("visual_filter_min_score")
    prediction_score = float(prediction_score) if prediction_score is not None else None

    gt_sets, gt_counts = build_category_presence_index(dataset.annotations)
    pred_sets, pred_counts = build_category_presence_index(predictions, prediction_score)

    candidates: List[ImageRecord] = []
    candidate_details: Dict[int, Dict[str, Any]] = {}
    for image in dataset.images:
        image_id = int(image.image_id)
        gt_ids = gt_sets.get(image_id, set())
        pred_ids = pred_sets.get(image_id, set())
        gt_ok = gt_counts.get(image_id, 0) >= min_gt and class_filter_matches(gt_ids, required_ids, filter_match)
        pred_ok = pred_counts.get(image_id, 0) >= min_pred and class_filter_matches(pred_ids, required_ids, filter_match)

        if filter_source == "ground_truth":
            keep = gt_ok
        elif filter_source == "prediction":
            keep = pred_ok
        elif filter_source == "both":
            keep = gt_ok and pred_ok
        else:
            keep = gt_ok or pred_ok

        if keep:
            candidates.append(image)
            candidate_details[image_id] = {
                "gt_count": gt_counts.get(image_id, 0),
                "prediction_count": pred_counts.get(image_id, 0),
                "gt_category_ids": sorted(int(category_id) for category_id in gt_ids),
                "prediction_category_ids": sorted(int(category_id) for category_id in pred_ids),
            }

    max_visuals = output_cfg.get("max_visuals")
    if max_visuals is None:
        selected_count = len(candidates)
    else:
        selected_count = min(len(candidates), max(0, int(max_visuals)))

    sampling_mode = str(output_cfg.get("visual_sampling_mode", "random")).strip().lower()
    if sampling_mode not in {"random", "first", "last"}:
        raise ValueError("output.visual_sampling_mode must be one of: random, first, last.")
    if sampling_mode == "first":
        selected = candidates[:selected_count]
    elif sampling_mode == "last":
        selected = candidates[-selected_count:] if selected_count else []
    else:
        selected = random.Random(get_visual_seed(config)).sample(candidates, selected_count) if selected_count else []

    sample_order = str(output_cfg.get("visual_sample_order", "sample")).strip().lower()
    if sample_order == "dataset":
        selected_ids = {image.image_id for image in selected}
        selected = [image for image in dataset.images if image.image_id in selected_ids]
    elif sample_order != "sample":
        raise ValueError("output.visual_sample_order must be either sample or dataset.")

    selected_ids = [int(image.image_id) for image in selected]
    info = {
        "enabled": True,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_image_ids": selected_ids,
        "requested_max_visuals": max_visuals,
        "sampling_mode": sampling_mode,
        "sample_order": sample_order,
        "visual_random_seed": get_visual_seed(config),
        "filter_source": filter_source,
        "filter_match": filter_match,
        "filter_class_ids": [int(category_id) for category_id in required_ids],
        "filter_class_names": [id_to_name.get(int(category_id), str(category_id)) for category_id in required_ids],
        "visual_filter_min_score": prediction_score,
        "visual_min_gt_instances": min_gt,
        "visual_min_predictions": min_pred,
        "candidate_details": {str(image_id): candidate_details[image_id] for image_id in selected_ids if image_id in candidate_details},
    }
    return selected, info


def normalize_visual_format(value: Any) -> str:
    """Normalize image export extension."""
    text = str(value or "jpg").strip().lower().lstrip(".")
    if text == "jpeg":
        return "jpg"
    if text not in {"jpg", "png"}:
        raise ValueError("output.visual_format must be jpg or png.")
    return text


def parse_rgb_color(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Parse color config as name, hex, comma string, or RGB list."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        text = value.strip()
        named = VISUAL_COLOR_NAMES.get(text.casefold())
        if named is not None:
            return named
        if re.fullmatch(r"#?[0-9A-Fa-f]{6}", text):
            text = text.lstrip("#")
            return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
        parts = config_list(text)
    else:
        parts = config_list(value)
    if len(parts) != 3:
        raise ValueError(f"RGB color must have exactly 3 values, got: {value!r}")
    rgb = tuple(max(0, min(255, int(float(part)))) for part in parts)
    return rgb  # type: ignore[return-value]


def visual_font(text_size: float) -> ImageFont.ImageFont:
    """Load a portable label font."""
    size = max(8, int(round(14 * max(0.1, text_size))))
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    label: str,
    color: Tuple[int, int, int],
    line_width: int,
    font: ImageFont.ImageFont,
    image_size_value: Tuple[int, int],
) -> None:
    """Draw one clipped xywh box with an optional label."""
    width, height = image_size_value
    x, y, box_width, box_height = [float(value) for value in box[:4]]
    x1 = max(0.0, min(float(width - 1), x))
    y1 = max(0.0, min(float(height - 1), y))
    x2 = max(0.0, min(float(width - 1), x + box_width))
    y2 = max(0.0, min(float(height - 1), y + box_height))
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle([x1, y1, x2, y2], outline=color, width=max(1, int(line_width)))
    if not label:
        return

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_x = int(x1)
    label_y = int(max(0, y1 - text_height - 4))
    if label_x + text_width + 6 > width:
        label_x = max(0, width - text_width - 6)
    label_box = [label_x, label_y, label_x + text_width + 6, label_y + text_height + 4]
    draw.rectangle(label_box, fill=color)
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    text_color = (0, 0, 0) if luminance > 155 else (255, 255, 255)
    draw.text((label_x + 3, label_y + 1), label, fill=text_color, font=font)


def save_annotated_image(image: Image.Image, path: Path, output_info: Mapping[str, Any], quality: int) -> None:
    """Save an annotated image with lightweight metadata when supported."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".png":
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("run_id", str(output_info["run_id"]))
        png_info.add_text("config_hash", str(output_info["config_hash"]))
        image.save(path, pnginfo=png_info)
        return
    image.convert("RGB").save(path, quality=max(1, min(100, int(quality))), optimize=True)


def render_visual_image(
    image: ImageRecord,
    annotations: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_info: Mapping[str, Any],
    output_path: Path,
    render_class_ids: Optional[Sequence[int]] = None,
) -> None:
    """Render GT and prediction boxes for one sampled image."""
    output_cfg = config["output"]
    id_to_name, _ = category_maps(categories)
    render_ids = set(int(category_id) for category_id in (render_class_ids or []))
    gt_color = parse_rgb_color(output_cfg.get("gt_color"), VISUAL_COLOR_NAMES["green"])
    pred_color = parse_rgb_color(output_cfg.get("pred_color"), VISUAL_COLOR_NAMES["red"])
    line_width = int(output_cfg.get("rect_th", 2))
    font = visual_font(float(output_cfg.get("text_size", 0.8)))
    hide_labels = bool(output_cfg.get("hide_labels", False))
    hide_conf = bool(output_cfg.get("hide_conf", False))
    draw_gt = bool(output_cfg.get("draw_ground_truth", True))
    draw_pred = bool(output_cfg.get("draw_predictions", True))
    draw_order = str(output_cfg.get("visual_draw_order", "ground_truth_first")).strip().lower()
    draw_min_score = output_cfg.get("visual_draw_min_score")
    draw_min_score = float(draw_min_score) if draw_min_score is not None else None

    with Image.open(image.path) as source:
        canvas = source.convert("RGB")
    drawer = ImageDraw.Draw(canvas)
    size = canvas.size

    def draw_ground_truth() -> None:
        if not draw_gt:
            return
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if render_ids and category_id not in render_ids:
                continue
            label = "" if hide_labels else f"GT {id_to_name.get(category_id, category_id)}"
            draw_labeled_box(drawer, annotation.get("bbox", [0, 0, 0, 0]), label, gt_color, line_width, font, size)

    def draw_predictions() -> None:
        if not draw_pred:
            return
        for prediction in predictions:
            score = float(prediction.get("score", 0.0))
            if draw_min_score is not None and score < draw_min_score:
                continue
            category_id = int(prediction["category_id"])
            if render_ids and category_id not in render_ids:
                continue
            label = ""
            if not hide_labels:
                label = f"P {id_to_name.get(category_id, category_id)}"
                if not hide_conf:
                    label = f"{label} {score:.2f}"
            draw_labeled_box(drawer, prediction.get("bbox", [0, 0, 0, 0]), label, pred_color, line_width, font, size)

    if draw_order == "predictions_first":
        draw_predictions()
        draw_ground_truth()
    else:
        draw_ground_truth()
        draw_predictions()

    save_annotated_image(canvas, output_path, output_info, int(output_cfg.get("visual_jpeg_quality", 92)))


def render_visual_outputs(
    dataset: DatasetBundle,
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
    output_info: Mapping[str, Any],
    quiet: bool,
    manifest: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Render random visual samples after predictions are available."""
    output_cfg = config["output"]
    if not bool(output_cfg.get("save_visuals", False)):
        return []

    selected, selection_info = select_visual_images(dataset, predictions, config)
    visuals_dir = output_dir / str(output_cfg.get("visual_output_subdir", "visuals"))
    visuals_dir.mkdir(parents=True, exist_ok=True)
    visual_format = normalize_visual_format(output_cfg.get("visual_format", "jpg"))
    render_class_ids = resolve_visual_render_class_ids(dataset.categories, output_cfg)
    id_to_name, _ = category_maps(dataset.categories)
    render_class_names = [id_to_name.get(int(category_id), str(category_id)) for category_id in render_class_ids]

    annotations_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for annotation in dataset.annotations:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    predictions_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for prediction in predictions:
        predictions_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)

    blue("SELECTING VISUAL SAMPLES", verbose=not quiet)
    blue(
        f"Visual candidates: {selection_info['candidate_count']}; rendering: {selection_info['selected_count']}",
        verbose=not quiet,
    )

    visual_rows: List[Dict[str, Any]] = []
    iterator: Iterable[ImageRecord] = selected
    if bool(config["progress"].get("visuals", True)) and not quiet and selected:
        iterator = tqdm(selected, desc="Rendering visuals", unit="image")

    for index, image in enumerate(iterator, start=1):
        image_id = int(image.image_id)
        file_stem = sanitize_name(Path(image.file_name).stem) + f"_{image_id}"
        visual_path = visuals_dir / f"{index:04d}_{file_stem}.{visual_format}"
        image_annotations = annotations_by_image.get(image_id, [])
        image_predictions = predictions_by_image.get(image_id, [])
        render_visual_image(
            image=image,
            annotations=image_annotations,
            predictions=image_predictions,
            categories=dataset.categories,
            config=config,
            output_info=output_info,
            output_path=visual_path,
            render_class_ids=render_class_ids,
        )
        detail = selection_info.get("candidate_details", {}).get(str(image_id), {})
        row = {
            "image_id": image_id,
            "file_name": image.file_name,
            "visual_path": str(visual_path),
            "selection_index": index,
            "gt_count": len(image_annotations),
            "prediction_count": len(image_predictions),
            "gt_category_ids": json.dumps(detail.get("gt_category_ids", [])),
            "prediction_category_ids": json.dumps(detail.get("prediction_category_ids", [])),
            "sampling_mode": selection_info["sampling_mode"],
            "filter_source": selection_info["filter_source"],
            "filter_class_ids": json.dumps(selection_info["filter_class_ids"]),
            "filter_class_names": json.dumps(selection_info["filter_class_names"]),
            "render_class_ids": json.dumps([int(category_id) for category_id in render_class_ids]),
            "render_class_names": json.dumps(render_class_names, ensure_ascii=False),
            "visual_random_seed": selection_info["visual_random_seed"],
        }
        visual_rows.append(row)
        manifest.append({"path": str(visual_path), "kind": "visual", "description": "Rendered GT/prediction visual sample.", "config_hash": output_info["config_hash"]})

    metadata_path = visuals_dir / "visuals_metadata.json"
    write_json(
        metadata_path,
        {
            "metadata": output_info,
            "config": config,
            "selection": {
                **{key: value for key, value in selection_info.items() if key != "candidate_details"},
                "render_class_ids": [int(category_id) for category_id in render_class_ids],
                "render_class_names": render_class_names,
            },
            "images": visual_rows,
        },
    )
    manifest.append({"path": str(metadata_path), "kind": "visuals", "description": "Visual sampling metadata and full config."})
    return visual_rows


def resolve_error_case_class_ids(categories: Sequence[Mapping[str, Any]], error_cfg: Mapping[str, Any]) -> List[int]:
    """Resolve classes targeted by error-case diagnostics; default to football."""
    return resolve_category_class_ids(
        categories,
        error_cfg.get("target_class_ids", error_cfg.get("class_ids")),
        error_cfg.get("target_class_names", error_cfg.get("class_names")),
        "error-case",
        default_names=["football"],
    )


def error_case_max_images(error_cfg: Mapping[str, Any]) -> int:
    """Return the shared image cap for all error-case categories."""
    if error_cfg.get("max_images") is not None:
        return max(0, int(error_cfg["max_images"]))
    legacy_values = [
        error_cfg.get("max_missed_images"),
        error_cfg.get("max_false_positive_images"),
    ]
    legacy_ints = [int(value) for value in legacy_values if value is not None]
    return max(legacy_ints) if legacy_ints else 25


def build_error_case_events(
    dataset: DatasetBundle,
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Find missed, misclassified, and false-positive target-class cases."""
    output_cfg = config["output"]
    error_cfg = dict(output_cfg.get("error_cases", {}) or {})
    if not bool(error_cfg.get("enabled", False)):
        return [], {"enabled": False, "candidate_image_count": 0, "selected_image_ids": []}

    target_ids = set(resolve_error_case_class_ids(dataset.categories, error_cfg))
    id_to_name, _ = category_maps(dataset.categories)
    eval_cfg = config.get("evaluation", {})
    confidence = error_cfg.get("confidence_threshold")
    if confidence is None:
        confidence = eval_cfg.get("operating_confidence_threshold")
    if confidence is None:
        confidence = config.get("model", {}).get("confidence_threshold", 0.25)
    confidence = float(confidence)
    iou_threshold = error_cfg.get("match_iou_threshold")
    if iou_threshold is None:
        iou_threshold = eval_cfg.get("match_iou_threshold", 0.5)
    iou_threshold = float(iou_threshold)

    annotations_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for annotation in dataset.annotations:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    predictions_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= confidence:
            predictions_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)

    events: List[Dict[str, Any]] = []
    priority = {"target_missed": 0, "target_misclassified": 1, "target_false_positive": 2}
    for image in dataset.images:
        image_id = int(image.image_id)
        annotations = annotations_by_image.get(image_id, [])
        image_predictions = sorted(
            predictions_by_image.get(image_id, []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        target_annotations = [
            (index, annotation)
            for index, annotation in enumerate(annotations)
            if int(annotation.get("category_id", -1)) in target_ids
        ]
        target_predictions = [
            (index, prediction)
            for index, prediction in enumerate(image_predictions)
            if int(prediction.get("category_id", -1)) in target_ids
        ]

        matched_gt_indexes: set[int] = set()
        matched_pred_indexes: set[int] = set()
        if target_annotations and target_predictions:
            gt_boxes = np.stack([xywh_to_xyxy(annotation["bbox"]) for _, annotation in target_annotations]).astype(np.float32)
            for pred_index, prediction in target_predictions:
                ious = bbox_iou_one_to_many(xywh_to_xyxy(prediction["bbox"]), gt_boxes)
                for local_index, (gt_index, _) in enumerate(target_annotations):
                    if gt_index in matched_gt_indexes:
                        ious[local_index] = -1.0
                best_local = int(np.argmax(ious)) if len(ious) else -1
                if best_local >= 0 and float(ious[best_local]) >= iou_threshold:
                    matched_gt_indexes.add(target_annotations[best_local][0])
                    matched_pred_indexes.add(pred_index)

        for gt_index, annotation in target_annotations:
            if gt_index in matched_gt_indexes:
                continue
            gt_box = xywh_to_xyxy(annotation["bbox"])
            best_prediction: Optional[Mapping[str, Any]] = None
            best_iou = 0.0
            for _, prediction in enumerate(image_predictions):
                if int(prediction.get("category_id", -1)) in target_ids:
                    continue
                iou = float(bbox_iou_one_to_many(gt_box, np.expand_dims(xywh_to_xyxy(prediction["bbox"]), axis=0))[0])
                if iou > best_iou:
                    best_iou = iou
                    best_prediction = prediction
            case_type = "target_misclassified" if best_prediction is not None and best_iou >= iou_threshold else "target_missed"
            pred_category = int(best_prediction["category_id"]) if best_prediction is not None else None
            events.append(
                {
                    "image_id": image_id,
                    "file_name": image.file_name,
                    "case_type": case_type,
                    "target_category_id": int(annotation["category_id"]),
                    "target_category_name": id_to_name.get(int(annotation["category_id"]), str(annotation["category_id"])),
                    "gt_category_id": int(annotation["category_id"]),
                    "gt_category_name": id_to_name.get(int(annotation["category_id"]), str(annotation["category_id"])),
                    "pred_category_id": pred_category,
                    "pred_category_name": id_to_name.get(pred_category, "") if pred_category is not None else "",
                    "score": float(best_prediction.get("score", 0.0)) if best_prediction is not None else None,
                    "iou": best_iou if best_prediction is not None else 0.0,
                    "gt_bbox": json.dumps(annotation.get("bbox", []), ensure_ascii=False),
                    "pred_bbox": json.dumps(best_prediction.get("bbox", []), ensure_ascii=False) if best_prediction is not None else "",
                    "priority": priority[case_type],
                }
            )

        if target_predictions:
            all_gt_boxes = np.stack([xywh_to_xyxy(annotation["bbox"]) for annotation in annotations]).astype(np.float32) if annotations else np.zeros((0, 4), dtype=np.float32)
            for pred_index, prediction in target_predictions:
                if pred_index in matched_pred_indexes:
                    continue
                best_gt: Optional[Mapping[str, Any]] = None
                best_iou = 0.0
                if len(all_gt_boxes):
                    ious = bbox_iou_one_to_many(xywh_to_xyxy(prediction["bbox"]), all_gt_boxes)
                    best_index = int(np.argmax(ious)) if len(ious) else -1
                    if best_index >= 0:
                        best_iou = float(ious[best_index])
                        if best_iou >= iou_threshold:
                            best_gt = annotations[best_index]
                gt_category = int(best_gt["category_id"]) if best_gt is not None else None
                events.append(
                    {
                        "image_id": image_id,
                        "file_name": image.file_name,
                        "case_type": "target_false_positive",
                        "target_category_id": int(prediction["category_id"]),
                        "target_category_name": id_to_name.get(int(prediction["category_id"]), str(prediction["category_id"])),
                        "gt_category_id": gt_category,
                        "gt_category_name": id_to_name.get(gt_category, "") if gt_category is not None else "",
                        "pred_category_id": int(prediction["category_id"]),
                        "pred_category_name": id_to_name.get(int(prediction["category_id"]), str(prediction["category_id"])),
                        "score": float(prediction.get("score", 0.0)),
                        "iou": best_iou,
                        "gt_bbox": json.dumps(best_gt.get("bbox", []), ensure_ascii=False) if best_gt is not None else "",
                        "pred_bbox": json.dumps(prediction.get("bbox", []), ensure_ascii=False),
                        "priority": priority["target_false_positive"],
                    }
                )

    events.sort(key=lambda item: (int(item["image_id"]), int(item["priority"]), -float(item["score"] or 0.0)))
    max_images = error_case_max_images(error_cfg)
    selected_image_ids: List[int] = []
    seen_ids = set()
    for event in events:
        image_id = int(event["image_id"])
        if image_id in seen_ids:
            continue
        if len(selected_image_ids) >= max_images:
            break
        selected_image_ids.append(image_id)
        seen_ids.add(image_id)

    selected_events = [event for event in events if int(event["image_id"]) in seen_ids]
    info = {
        "enabled": True,
        "target_class_ids": sorted(int(category_id) for category_id in target_ids),
        "target_class_names": [id_to_name.get(int(category_id), str(category_id)) for category_id in sorted(target_ids)],
        "confidence_threshold": confidence,
        "match_iou_threshold": iou_threshold,
        "requested_max_images": max_images,
        "candidate_image_count": len({int(event["image_id"]) for event in events}),
        "candidate_event_count": len(events),
        "selected_image_ids": selected_image_ids,
        "selected_event_count": len(selected_events),
    }
    return selected_events, info


def render_error_case_outputs(
    dataset: DatasetBundle,
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
    output_info: Mapping[str, Any],
    quiet: bool,
    manifest: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Render target-class missed, misclassified, and false-positive diagnostic images."""
    output_cfg = config["output"]
    error_cfg = dict(output_cfg.get("error_cases", {}) or {})
    if not bool(error_cfg.get("enabled", False)):
        return []

    events, selection_info = build_error_case_events(dataset, predictions, config)
    selected_ids = set(int(image_id) for image_id in selection_info.get("selected_image_ids", []))
    if not selected_ids:
        return []

    case_dir = output_dir / str(error_cfg.get("output_subdir", "error_cases"))
    case_dir.mkdir(parents=True, exist_ok=True)
    case_format = normalize_visual_format(error_cfg.get("format", output_cfg.get("visual_format", "jpg")))
    render_class_ids = resolve_error_case_render_class_ids(dataset.categories, error_cfg)
    id_to_name, _ = category_maps(dataset.categories)
    render_class_names = [id_to_name.get(int(category_id), str(category_id)) for category_id in render_class_ids]
    selection_info = {
        **selection_info,
        "render_class_ids": [int(category_id) for category_id in render_class_ids],
        "render_class_names": render_class_names,
    }

    annotations_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for annotation in dataset.annotations:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    predictions_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    confidence = float(selection_info.get("confidence_threshold", config.get("model", {}).get("confidence_threshold", 0.25)))
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= confidence:
            predictions_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)

    images_by_id = {int(image.image_id): image for image in dataset.images}
    events_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for event in events:
        events_by_image.setdefault(int(event["image_id"]), []).append(event)

    blue("RENDERING ERROR CASES", verbose=not quiet)
    blue(
        f"Error-case candidate images: {selection_info['candidate_image_count']}; rendering: {len(selected_ids)}",
        verbose=not quiet,
    )

    selected_images = [images_by_id[image_id] for image_id in selection_info["selected_image_ids"] if image_id in images_by_id]
    iterator: Iterable[ImageRecord] = selected_images
    if bool(config["progress"].get("error_cases", True)) and not quiet and selected_images:
        iterator = tqdm(selected_images, desc="Rendering error cases", unit="image")

    image_rows: List[Dict[str, Any]] = []
    for index, image in enumerate(iterator, start=1):
        image_id = int(image.image_id)
        file_stem = sanitize_name(Path(image.file_name).stem) + f"_{image_id}"
        case_path = case_dir / f"{index:04d}_{file_stem}.{case_format}"
        image_events = events_by_image.get(image_id, [])
        render_visual_image(
            image=image,
            annotations=annotations_by_image.get(image_id, []),
            predictions=predictions_by_image.get(image_id, []),
            categories=dataset.categories,
            config=config,
            output_info=output_info,
            output_path=case_path,
            render_class_ids=render_class_ids,
        )
        case_types = sorted({str(event["case_type"]) for event in image_events})
        row = {
            "image_id": image_id,
            "file_name": image.file_name,
            "error_case_path": str(case_path),
            "selection_index": index,
            "case_types": json.dumps(case_types, ensure_ascii=False),
            "event_count": len(image_events),
            "target_class_names": json.dumps(selection_info["target_class_names"], ensure_ascii=False),
            "render_class_names": json.dumps(render_class_names, ensure_ascii=False),
        }
        image_rows.append(row)
        manifest.append({"path": str(case_path), "kind": "error_case", "description": "Rendered target-class GT/prediction diagnostic image.", "config_hash": output_info["config_hash"]})

    metadata = {"run_id": output_info["run_id"], "config_hash": output_info["config_hash"]}
    manifest_path = case_dir / "error_cases_manifest.csv"
    write_table(manifest_path, image_rows, metadata)
    manifest.append({"path": str(manifest_path), "kind": "error_cases", "description": "Error-case image manifest with config hash."})

    events_path = case_dir / "error_case_events.csv"
    write_table(events_path, events, metadata)
    manifest.append({"path": str(events_path), "kind": "error_cases", "description": "Per-event error-case details."})

    metadata_path = case_dir / "error_cases_metadata.json"
    write_json(
        metadata_path,
        {
            "metadata": output_info,
            "config": config,
            "selection": selection_info,
            "images": image_rows,
            "events": events,
        },
    )
    manifest.append({"path": str(metadata_path), "kind": "error_cases", "description": "Error-case metadata and full config."})
    return image_rows


def get_dataset_case_seed(config: Mapping[str, Any]) -> int:
    """Return the dataset case sampling seed."""
    output_cfg = config["output"]
    value = output_cfg.get("dataset_case_random_seed")
    if value is None:
        value = config.get("runtime", {}).get("seed", 0)
    return int(value)


def select_dataset_case_images(
    dataset: DatasetBundle,
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[List[ImageRecord], Dict[str, Any]]:
    """Select input images for output/datasets case examples."""
    output_cfg = config["output"]
    if not bool(output_cfg.get("save_dataset_cases", True)):
        return [], {"enabled": False, "candidate_count": 0, "selected_count": 0}

    filter_source = str(output_cfg.get("visual_filter_source", "ground_truth")).strip().lower()
    if filter_source not in {"ground_truth", "prediction", "either", "both"}:
        raise ValueError("output.visual_filter_source must be one of: ground_truth, prediction, either, both.")
    filter_match = str(output_cfg.get("visual_filter_match", "any")).strip().lower()
    if filter_match not in {"any", "all"}:
        raise ValueError("output.visual_filter_match must be one of: any, all.")

    required_ids = resolve_visual_filter_class_ids(dataset.categories, output_cfg)
    id_to_name, _ = category_maps(dataset.categories)
    min_gt = max(0, int(output_cfg.get("visual_min_gt_instances", 0)))
    min_pred = max(0, int(output_cfg.get("visual_min_predictions", 0)))
    prediction_score = output_cfg.get("visual_filter_min_score")
    prediction_score = float(prediction_score) if prediction_score is not None else None

    gt_sets, gt_counts = build_category_presence_index(dataset.annotations)
    pred_sets, pred_counts = build_category_presence_index(predictions, prediction_score)

    candidates: List[ImageRecord] = []
    candidate_details: Dict[int, Dict[str, Any]] = {}
    for image in dataset.images:
        image_id = int(image.image_id)
        gt_ids = gt_sets.get(image_id, set())
        pred_ids = pred_sets.get(image_id, set())
        gt_ok = gt_counts.get(image_id, 0) >= min_gt and class_filter_matches(gt_ids, required_ids, filter_match)
        pred_ok = pred_counts.get(image_id, 0) >= min_pred and class_filter_matches(pred_ids, required_ids, filter_match)

        if filter_source == "ground_truth":
            keep = gt_ok
        elif filter_source == "prediction":
            keep = pred_ok
        elif filter_source == "both":
            keep = gt_ok and pred_ok
        else:
            keep = gt_ok or pred_ok

        if keep:
            candidates.append(image)
            candidate_details[image_id] = {
                "gt_count": gt_counts.get(image_id, 0),
                "prediction_count": pred_counts.get(image_id, 0),
                "gt_category_ids": sorted(int(category_id) for category_id in gt_ids),
                "prediction_category_ids": sorted(int(category_id) for category_id in pred_ids),
            }

    max_cases = output_cfg.get("max_dataset_case_images")
    if max_cases is None:
        selected_count = len(candidates)
    else:
        selected_count = min(len(candidates), max(0, int(max_cases)))

    sampling_mode = str(output_cfg.get("dataset_case_sampling_mode", output_cfg.get("visual_sampling_mode", "random"))).strip().lower()
    if sampling_mode not in {"random", "first", "last"}:
        raise ValueError("output.dataset_case_sampling_mode must be one of: random, first, last.")
    if sampling_mode == "first":
        selected = candidates[:selected_count]
    elif sampling_mode == "last":
        selected = candidates[-selected_count:] if selected_count else []
    else:
        selected = random.Random(get_dataset_case_seed(config)).sample(candidates, selected_count) if selected_count else []

    selected_ids = [int(image.image_id) for image in selected]
    info = {
        "enabled": True,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_image_ids": selected_ids,
        "requested_max_cases": max_cases,
        "sampling_mode": sampling_mode,
        "dataset_case_random_seed": get_dataset_case_seed(config),
        "filter_source": filter_source,
        "filter_match": filter_match,
        "filter_class_ids": [int(category_id) for category_id in required_ids],
        "filter_class_names": [id_to_name.get(int(category_id), str(category_id)) for category_id in required_ids],
        "visual_filter_min_score": prediction_score,
        "visual_min_gt_instances": min_gt,
        "visual_min_predictions": min_pred,
        "candidate_details": {str(image_id): candidate_details[image_id] for image_id in selected_ids if image_id in candidate_details},
    }
    return selected, info


def slice_axis_starts(length: int, window: int, overlap_ratio: float) -> List[int]:
    """Return deterministic slice starts that cover one image axis."""
    length = max(1, int(length))
    window = max(1, min(int(window), length))
    if window >= length:
        return [0]
    overlap_ratio = max(0.0, min(0.95, float(overlap_ratio)))
    step = max(1, int(round(window * (1.0 - overlap_ratio))))
    last_start = length - window
    starts = list(range(0, last_start + 1, step))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def generate_slice_windows(image: ImageRecord, config: Mapping[str, Any]) -> List[Tuple[int, int, int, int]]:
    """Generate SAHI-style slice windows as (x, y, width, height)."""
    sahi_cfg = config.get("sahi", {})
    slice_height = int(sahi_cfg.get("slice_height", image.height))
    slice_width = int(sahi_cfg.get("slice_width", image.width))
    y_starts = slice_axis_starts(image.height, slice_height, float(sahi_cfg.get("overlap_height_ratio", 0.2)))
    x_starts = slice_axis_starts(image.width, slice_width, float(sahi_cfg.get("overlap_width_ratio", 0.2)))
    width = min(slice_width, image.width)
    height = min(slice_height, image.height)
    return [(x, y, min(width, image.width - x), min(height, image.height - y)) for y in y_starts for x in x_starts]


def bbox_intersects_window(bbox: Sequence[float], window: Tuple[int, int, int, int]) -> bool:
    """Return whether a COCO xywh bbox intersects a slice window."""
    x, y, width, height = [float(value) for value in bbox[:4]]
    wx, wy, ww, wh = [float(value) for value in window]
    return x < wx + ww and x + width > wx and y < wy + wh and y + height > wy


def select_slice_windows(
    image: ImageRecord,
    annotations: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> List[Tuple[int, int, int, int]]:
    """Select a bounded set of useful slice windows for dataset case output."""
    windows = generate_slice_windows(image, config)
    max_slices = int(config["output"].get("max_slices_per_image", 12))
    if max_slices <= 0:
        return []
    scored = []
    for index, window in enumerate(windows):
        object_count = 0
        for annotation in annotations:
            if bbox_intersects_window(annotation.get("bbox", [0, 0, 0, 0]), window):
                object_count += 1
        for prediction in predictions:
            if bbox_intersects_window(prediction.get("bbox", [0, 0, 0, 0]), window):
                object_count += 1
        x, y, _, _ = window
        scored.append((object_count, y, x, index, window))
    if any(item[0] for item in scored):
        scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    else:
        scored.sort(key=lambda item: item[3])
    return [item[4] for item in scored[: min(max_slices, len(scored))]]


def render_dataset_case_outputs(
    dataset: DatasetBundle,
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
    output_info: Mapping[str, Any],
    quiet: bool,
    manifest: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Write raw dataset case images under output/datasets."""
    output_cfg = config["output"]
    if not bool(output_cfg.get("save_dataset_cases", True)):
        return []

    selected, selection_info = select_dataset_case_images(dataset, predictions, config)
    case_root = output_dir / str(output_cfg.get("dataset_cases_subdir", "datasets"))
    case_type = "sliced" if use_sahi_inference(config) else "original"
    case_dir = case_root / f"{case_type}_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_format = normalize_visual_format(output_cfg.get("dataset_case_format", output_cfg.get("visual_format", "jpg")))
    quality = int(output_cfg.get("dataset_case_jpeg_quality", output_cfg.get("visual_jpeg_quality", 92)))

    annotations_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for annotation in dataset.annotations:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    predictions_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for prediction in predictions:
        predictions_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)

    blue("EXPORTING DATASET CASES", verbose=not quiet)
    blue(
        f"Dataset case candidates: {selection_info['candidate_count']}; selected images: {selection_info['selected_count']}; mode: {case_type}",
        verbose=not quiet,
    )

    rows: List[Dict[str, Any]] = []
    iterator: Iterable[ImageRecord] = selected
    if bool(config["progress"].get("dataset_cases", True)) and not quiet and selected:
        iterator = tqdm(selected, desc="Writing dataset cases", unit="image")

    for image_index, image in enumerate(iterator, start=1):
        image_id = int(image.image_id)
        file_stem = sanitize_name(Path(image.file_name).stem) + f"_{image_id}"
        image_annotations = annotations_by_image.get(image_id, [])
        image_predictions = predictions_by_image.get(image_id, [])
        with Image.open(image.path) as source:
            source_rgb = source.convert("RGB")
            if use_sahi_inference(config):
                windows = select_slice_windows(image, image_annotations, image_predictions, config)
                for slice_index, (x, y, width, height) in enumerate(windows, start=1):
                    case_path = case_dir / f"{image_index:04d}_{slice_index:03d}_{file_stem}_x{x}_y{y}.{case_format}"
                    crop = source_rgb.crop((x, y, x + width, y + height))
                    save_annotated_image(crop, case_path, output_info, quality)
                    row = {
                        "image_id": image_id,
                        "file_name": image.file_name,
                        "case_type": "sliced",
                        "case_path": str(case_path),
                        "image_index": image_index,
                        "slice_index": slice_index,
                        "slice_x": x,
                        "slice_y": y,
                        "slice_width": width,
                        "slice_height": height,
                        "gt_count": sum(1 for item in image_annotations if bbox_intersects_window(item.get("bbox", [0, 0, 0, 0]), (x, y, width, height))),
                        "prediction_count": sum(1 for item in image_predictions if bbox_intersects_window(item.get("bbox", [0, 0, 0, 0]), (x, y, width, height))),
                        "inference_engine": inference_engine_name(config),
                    }
                    rows.append(row)
                    manifest.append({"path": str(case_path), "kind": "dataset_case", "description": "Raw SAHI slice case image.", "config_hash": output_info["config_hash"]})
            else:
                case_path = case_dir / f"{image_index:04d}_{file_stem}.{case_format}"
                save_annotated_image(source_rgb, case_path, output_info, quality)
                row = {
                    "image_id": image_id,
                    "file_name": image.file_name,
                    "case_type": "original",
                    "case_path": str(case_path),
                    "image_index": image_index,
                    "slice_index": 0,
                    "slice_x": 0,
                    "slice_y": 0,
                    "slice_width": image.width,
                    "slice_height": image.height,
                    "gt_count": len(image_annotations),
                    "prediction_count": len(image_predictions),
                    "inference_engine": inference_engine_name(config),
                }
                rows.append(row)
                manifest.append({"path": str(case_path), "kind": "dataset_case", "description": "Raw full-image dataset case.", "config_hash": output_info["config_hash"]})

    metadata = {"run_id": output_info["run_id"], "config_hash": output_info["config_hash"]}
    manifest_path = case_root / "dataset_cases_manifest.csv"
    write_table(manifest_path, rows, metadata)
    manifest.append({"path": str(manifest_path), "kind": "dataset_cases", "description": "Dataset case image manifest with config hash."})

    metadata_path = case_root / "dataset_cases_metadata.json"
    write_json(
        metadata_path,
        {
            "metadata": output_info,
            "config": config,
            "case_type": case_type,
            "selection": {key: value for key, value in selection_info.items() if key != "candidate_details"},
            "cases": rows,
        },
    )
    manifest.append({"path": str(metadata_path), "kind": "dataset_cases", "description": "Dataset case metadata and full config."})
    return rows


def build_detection_model(model_cfg: Mapping[str, Any], device: str) -> Any:
    """Build a SAHI detection model."""
    from sahi import AutoDetectionModel

    kwargs: Dict[str, Any] = {
        "model_type": model_cfg.get("type", "ultralytics"),
        "model_path": model_cfg.get("path"),
        "model_config_path": model_cfg.get("config_path") or None,
        "confidence_threshold": float(model_cfg.get("confidence_threshold", 0.25)),
        "device": device,
        "category_mapping": model_cfg.get("category_mapping") or None,
        "category_remapping": model_cfg.get("category_remapping") or None,
        "load_at_init": bool(model_cfg.get("load_at_init", True)),
        "image_size": model_cfg.get("image_size"),
    }
    kwargs.update(model_cfg.get("extra_model_args") or {})
    kwargs = {key: value for key, value in kwargs.items() if value is not None and value != ""}
    return call_with_supported_kwargs(AutoDetectionModel.from_pretrained, **kwargs)


def build_ultralytics_model(model_cfg: Mapping[str, Any], device: str) -> Any:
    """Build a direct Ultralytics model for full-image inference."""
    from ultralytics import YOLO

    model = YOLO(model_cfg.get("path"))
    try:
        model.to(device)
    except Exception:
        pass
    try:
        setattr(model, "_evaluator_device", device)
    except Exception:
        pass
    return model


def build_rfdetr_model(model_cfg: Mapping[str, Any], device: str) -> Any:
    """Build an RF-DETR model for direct image-level inference."""
    import torch
    import rfdetr

    size = str(model_cfg.get("size", "medium")).strip().lower().replace("_", "-")
    classes = {
        "base": rfdetr.RFDETRBase,
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
        "large": rfdetr.RFDETRLarge,
        "seg-preview": rfdetr.RFDETRSegPreview,
        "seg-nano": rfdetr.RFDETRSegNano,
        "seg-small": rfdetr.RFDETRSegSmall,
        "seg-medium": rfdetr.RFDETRSegMedium,
        "seg-large": rfdetr.RFDETRSegLarge,
        "seg-xlarge": rfdetr.RFDETRSegXLarge,
        "seg-2xlarge": rfdetr.RFDETRSeg2XLarge,
    }
    model_cls = classes.get(size)
    if model_cls is None:
        raise ValueError(f"Unsupported RF-DETR size={size!r}.")
    kwargs = dict(model_cfg.get("extra_model_args") or {})
    if model_cfg.get("path") and "pretrain_weights" not in kwargs:
        kwargs["pretrain_weights"] = model_cfg.get("path")
    model = model_cls(**kwargs)
    try:
        target = torch.device(normalize_device(device))
        model.model.model = model.model.model.to(target)
        model.model.device = target
    except Exception:
        pass
    return model


def build_direct_model(model_cfg: Mapping[str, Any], device: str) -> Any:
    """Build a non-SAHI direct inference model."""
    model_type = str(model_cfg.get("type", "ultralytics")).strip().lower()
    if model_type == "ultralytics":
        return build_ultralytics_model(model_cfg, device)
    if model_type in {"rfdetr", "rf-detr", "rf_detr"}:
        return build_rfdetr_model(model_cfg, device)
    raise ValueError("Direct full_image/class_crop inference supports model.type: ultralytics or rfdetr.")


def build_inference_model(config: Mapping[str, Any], device: str, prebuilt_model: Any = None) -> Any:
    """Build the configured inference model."""
    if prebuilt_model is not None:
        return prebuilt_model
    if use_sahi_inference(config):
        model_type = str(config["model"].get("type", "ultralytics")).strip().lower()
        if model_type in {"rfdetr", "rf-detr", "rf_detr"}:
            return build_rfdetr_model(config["model"], device)
        return build_detection_model(config["model"], device)
    return build_direct_model(config["model"], device)


def remap_model_category_id(category_id: int, model_cfg: Mapping[str, Any]) -> int:
    """Apply optional model class to dataset category remapping."""
    remapping = model_cfg.get("category_remapping") or {}
    if not isinstance(remapping, Mapping):
        return int(category_id)
    for key in (category_id, str(category_id)):
        if key in remapping:
            return int(remapping[key])
    return int(category_id)


def xyxy_to_coco_bbox(box: Sequence[float], width: int, height: int) -> Optional[Tuple[List[float], float]]:
    """Convert clipped xyxy values to COCO xywh and area."""
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    bbox = [x1, y1, x2 - x1, y2 - y1]
    return bbox, bbox[2] * bbox[3]


def to_numpy_array(value: Any) -> np.ndarray:
    """Convert tensors or array-likes to a CPU numpy array."""
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


RFDETR_MODEL_TYPES = {"rfdetr", "rf-detr", "rf_detr"}


def is_rfdetr_model_type(model_cfg: Mapping[str, Any]) -> bool:
    """Return True when the configured direct model is RF-DETR."""
    return str(model_cfg.get("type", "ultralytics")).strip().lower() in RFDETR_MODEL_TYPES


def positive_int_setting(value: Any, default: int, field_name: str) -> int:
    """Parse a positive integer setting with all/null inheriting the default."""
    if value is None:
        return max(1, int(default))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "all", "none", "null"}:
            return max(1, int(default))
        value = parse_scalar(value)
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    return parsed


def rfdetr_image_batch_size(config: Mapping[str, Any]) -> int:
    """Return the image-level RF-DETR batch size."""
    inference_cfg = config.get("inference", {})
    model_cfg = config.get("model", {})
    default = positive_int_setting(model_cfg.get("batch_size", 1), 1, "model.batch_size")
    return positive_int_setting(inference_cfg.get("batch_size"), default, "inference.batch_size")


def rfdetr_sahi_batch_size(config: Mapping[str, Any]) -> int:
    """Return the RF-DETR SAHI slice/recheck batch size."""
    return positive_int_setting(config.get("sahi", {}).get("batch_size"), rfdetr_image_batch_size(config), "sahi.batch_size")


def batched_sequence(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    """Yield non-empty batches from a sequence."""
    size = max(1, int(batch_size))
    for index in range(0, len(values), size):
        yield values[index : index + size]


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Detect CUDA OOM errors without requiring torch at import time."""
    text = str(exc).lower()
    if "out of memory" not in text and "cuda error: out of memory" not in text:
        return False
    with contextlib.suppress(Exception):
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    return "cuda" in text or "cudnn" in text or "out of memory" in text


def clear_cuda_cache_if_available() -> None:
    """Release cached CUDA blocks after an OOM retry."""
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextlib.contextmanager
def torch_inference_context() -> Iterable[None]:
    """Use torch.inference_mode when torch is available."""
    try:
        import torch
    except Exception:
        yield
        return
    with torch.inference_mode():
        yield


def normalize_detection_list(result: Any, expected: int) -> List[Any]:
    """Normalize RF-DETR model.predict output to one detections object per input."""
    if isinstance(result, tuple):
        result = list(result)
    if isinstance(result, list):
        if len(result) != expected:
            raise ValueError(f"RF-DETR returned {len(result)} prediction result(s) for {expected} input image(s).")
        return result
    if expected == 1:
        return [result]
    raise ValueError(f"RF-DETR returned a single prediction result for {expected} input image(s).")


def rfdetr_predict_batches(
    model: Any,
    inputs: Sequence[Any],
    *,
    threshold: float,
    shape: Optional[Tuple[int, int]],
    batch_size: int,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Run RF-DETR model.predict on input batches with CUDA OOM downshift."""
    detections: List[Any] = []
    timing_rows: List[Dict[str, Any]] = []
    active_batch_size = max(1, int(batch_size))
    index = 0
    while index < len(inputs):
        current_size = min(active_batch_size, len(inputs) - index)
        current_inputs = list(inputs[index : index + current_size])
        start = time.perf_counter()
        try:
            with torch_inference_context():
                result = call_with_supported_kwargs(
                    model.predict,
                    current_inputs,
                    threshold=threshold,
                    shape=shape,
                )
        except Exception as exc:
            if current_size > 1 and is_cuda_oom_error(exc):
                clear_cuda_cache_if_available()
                active_batch_size = max(1, current_size // 2)
                blue(
                    f"RF-DETR CUDA OOM at batch_size={current_size}; retrying with batch_size={active_batch_size}.",
                    verbose=True,
                    force=True,
                )
                continue
            raise
        elapsed = time.perf_counter() - start
        result_list = normalize_detection_list(result, len(current_inputs))
        per_image_elapsed = elapsed / max(1, len(current_inputs))
        detections.extend(result_list)
        timing_rows.extend(
            {
                "elapsed_seconds": per_image_elapsed,
                "batch_elapsed_seconds": elapsed,
                "batch_size": len(current_inputs),
            }
            for _ in current_inputs
        )
        index += len(current_inputs)
    return detections, timing_rows


def rfdetr_detections_to_predictions(
    detections: Any,
    image: ImageRecord,
    model_cfg: Mapping[str, Any],
    infer_width: int,
    infer_height: int,
) -> List[Dict[str, Any]]:
    """Convert one RF-DETR Detections object to COCO prediction rows."""
    predictions: List[Dict[str, Any]] = []
    xyxy = to_numpy_array(getattr(detections, "xyxy", []))
    conf = to_numpy_array(getattr(detections, "confidence", []))
    cls = to_numpy_array(getattr(detections, "class_id", []))
    if cls.size == 0 and len(xyxy):
        cls = np.zeros((len(xyxy),), dtype=np.int64)
    if conf.size == 0 and len(xyxy):
        conf = np.ones((len(xyxy),), dtype=np.float32)
    for box, score, category in zip(xyxy, conf, cls):
        converted = xyxy_to_coco_bbox(box, infer_width, infer_height)
        if converted is None:
            continue
        bbox, area = converted
        predictions.append(
            {
                "image_id": int(image.image_id),
                "category_id": remap_model_category_id(int(category), model_cfg),
                "bbox": [float(value) for value in bbox],
                "score": float(score),
                "area": float(area),
            }
        )
    return predictions


def predict_rfdetr_direct_batch(
    images: Sequence[ImageRecord],
    model: Any,
    config: Mapping[str, Any],
    *,
    sources: Optional[Sequence[Any]] = None,
    widths: Optional[Sequence[int]] = None,
    heights: Optional[Sequence[int]] = None,
    confidence: Optional[float] = None,
    batch_size: Optional[int] = None,
    engine: str = "rfdetr",
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Run direct RF-DETR prediction for multiple images/crops."""
    if not images:
        return [], []
    model_cfg = config["model"]
    threshold = float(model_cfg.get("confidence_threshold", 0.25) if confidence is None else confidence)
    shape = None
    if model_cfg.get("image_size") is not None:
        image_size_value = int(model_cfg.get("image_size"))
        shape = (image_size_value, image_size_value)
    inputs = list(sources) if sources is not None else [image.path for image in images]
    infer_widths = [int(value) for value in widths] if widths is not None else [int(image.width) for image in images]
    infer_heights = [int(value) for value in heights] if heights is not None else [int(image.height) for image in images]
    if len(inputs) != len(images) or len(infer_widths) != len(images) or len(infer_heights) != len(images):
        raise ValueError("RF-DETR batch inputs, images, widths, and heights must have the same length.")
    detections, timings = rfdetr_predict_batches(
        model,
        inputs,
        threshold=threshold,
        shape=shape,
        batch_size=batch_size or rfdetr_image_batch_size(config),
    )
    predictions_by_image: List[List[Dict[str, Any]]] = []
    stats: List[Dict[str, Any]] = []
    for image, detection, infer_width, infer_height, timing in zip(images, detections, infer_widths, infer_heights, timings):
        predictions = rfdetr_detections_to_predictions(detection, image, model_cfg, infer_width, infer_height)
        predictions_by_image.append(predictions)
        stats.append(
            {
                "image_id": image.image_id,
                "file_name": image.file_name,
                "width": infer_width,
                "height": infer_height,
                "predictions": len(predictions),
                "elapsed_seconds": timing["elapsed_seconds"],
                "batch_elapsed_seconds": timing["batch_elapsed_seconds"],
                "batch_size": timing["batch_size"],
                "inference_engine": engine,
            }
        )
    return predictions_by_image, stats


def predict_image_sahi(
    image: ImageRecord,
    detection_model: Any,
    config: Mapping[str, Any],
    output_dir: Path,
    save_visual: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]]]:
    """Run SAHI prediction for one image."""
    model_type = str(config["model"].get("type", "ultralytics")).strip().lower()
    if model_type in {"rfdetr", "rf-detr", "rf_detr"}:
        return predict_image_sliced_direct(image, detection_model, config)

    from sahi.predict import get_prediction, get_sliced_prediction

    sahi_cfg = config["sahi"]
    output_cfg = config["output"]
    progress_cfg = config["progress"]
    start = time.perf_counter()

    if bool(sahi_cfg.get("sliced_prediction", True)):
        kwargs: Dict[str, Any] = {
            "slice_height": int(sahi_cfg.get("slice_height", 640)),
            "slice_width": int(sahi_cfg.get("slice_width", 640)),
            "overlap_height_ratio": float(sahi_cfg.get("overlap_height_ratio", 0.2)),
            "overlap_width_ratio": float(sahi_cfg.get("overlap_width_ratio", 0.2)),
            "perform_standard_pred": bool(sahi_cfg.get("standard_prediction", True)),
            "postprocess_type": sahi_cfg.get("postprocess_type", "GREEDYNMM"),
            "postprocess_match_metric": sahi_cfg.get("postprocess_match_metric", "IOS"),
            "postprocess_match_threshold": float(sahi_cfg.get("postprocess_match_threshold", 0.5)),
            "postprocess_class_agnostic": bool(sahi_cfg.get("postprocess_class_agnostic", False)),
            "auto_slice_resolution": bool(sahi_cfg.get("auto_slice_resolution", False)),
            "merge_buffer_length": sahi_cfg.get("merge_buffer_length"),
            "verbose": int(sahi_cfg.get("verbose", 0)),
            "progress_bar": bool(progress_cfg.get("slices", False)),
            "batch_size": int(sahi_cfg.get("batch_size", 1)),
            "exclude_classes_by_name": sahi_cfg.get("exclude_classes_by_name") or None,
            "exclude_classes_by_id": sahi_cfg.get("exclude_classes_by_id") or None,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        result = call_with_supported_kwargs(get_sliced_prediction, image.path, detection_model, **kwargs)
    else:
        result = call_with_supported_kwargs(get_prediction, image.path, detection_model)

    elapsed = time.perf_counter() - start
    predictions = [prediction_to_clean_coco(item, image.image_id) for item in result.to_coco_predictions(image_id=image.image_id)]
    durations = getattr(result, "durations_in_seconds", {}) or {}
    stat = {
        "image_id": image.image_id,
        "file_name": image.file_name,
        "width": image.width,
        "height": image.height,
        "predictions": len(predictions),
        "elapsed_seconds": elapsed,
        "inference_engine": "sahi",
        "sahi_durations": durations,
    }

    visual_row: Optional[Dict[str, Any]] = None
    if save_visual:
        visuals_dir = output_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        file_stem = sanitize_name(Path(image.file_name).stem) + f"_{image.image_id}"
        result.export_visuals(
            export_dir=str(visuals_dir),
            file_name=file_stem,
            export_format=str(output_cfg.get("visual_format", "jpg")),
            hide_labels=bool(output_cfg.get("hide_labels", False)),
            hide_conf=bool(output_cfg.get("hide_conf", False)),
            rect_th=int(output_cfg.get("rect_th", 2)),
            text_size=float(output_cfg.get("text_size", 0.8)),
            text_th=int(output_cfg.get("text_th", 1)),
        )
        visual_path = visuals_dir / f"{file_stem}.{str(output_cfg.get('visual_format', 'jpg')).lower()}"
        visual_row = {"image_id": image.image_id, "file_name": image.file_name, "visual_path": str(visual_path)}

    return predictions, stat, visual_row


def predict_image_ultralytics(
    image: ImageRecord,
    model: Any,
    config: Mapping[str, Any],
    source: Any = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    confidence: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
    """Run direct Ultralytics full-image prediction for one image."""
    model_cfg = config["model"]
    infer_width = int(width or image.width)
    infer_height = int(height or image.height)
    start = time.perf_counter()
    predict_kwargs: Dict[str, Any] = {
        "source": source if source is not None else image.path,
        "conf": float(model_cfg.get("confidence_threshold", 0.25) if confidence is None else confidence),
        "device": getattr(model, "_evaluator_device", model_cfg.get("device", "cpu")),
        "verbose": False,
    }
    if model_cfg.get("image_size") is not None:
        predict_kwargs["imgsz"] = int(model_cfg.get("image_size"))
    predict_kwargs.update(model_cfg.get("extra_predict_args") or {})
    results = call_with_supported_kwargs(model.predict, **predict_kwargs)
    elapsed = time.perf_counter() - start

    result = results[0] if results else None
    predictions: List[Dict[str, Any]] = []
    speed: Dict[str, Any] = {}
    if result is not None:
        speed = dict(getattr(result, "speed", {}) or {})
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = to_numpy_array(getattr(boxes, "xyxy", []))
            conf = to_numpy_array(getattr(boxes, "conf", []))
            cls = to_numpy_array(getattr(boxes, "cls", []))
            for box, score, category in zip(xyxy, conf, cls):
                converted = xyxy_to_coco_bbox(box, infer_width, infer_height)
                if converted is None:
                    continue
                bbox, area = converted
                predictions.append(
                    {
                        "image_id": int(image.image_id),
                        "category_id": remap_model_category_id(int(category), model_cfg),
                        "bbox": [float(value) for value in bbox],
                        "score": float(score),
                        "area": float(area),
                    }
                )

    stat = {
        "image_id": image.image_id,
        "file_name": image.file_name,
        "width": infer_width,
        "height": infer_height,
        "predictions": len(predictions),
        "elapsed_seconds": elapsed,
        "inference_engine": "ultralytics",
        "ultralytics_speed": speed,
    }
    return predictions, stat, None


def predict_image_rfdetr(
    image: ImageRecord,
    model: Any,
    config: Mapping[str, Any],
    source: Any = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    confidence: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
    """Run direct RF-DETR prediction for one image or crop."""
    predictions_by_image, stats = predict_rfdetr_direct_batch(
        [image],
        model,
        config,
        sources=[source] if source is not None else None,
        widths=[int(width or image.width)],
        heights=[int(height or image.height)],
        confidence=confidence,
        batch_size=1,
        engine="rfdetr",
    )
    return predictions_by_image[0], stats[0], None


def predict_image_direct(
    image: ImageRecord,
    model: Any,
    config: Mapping[str, Any],
    source: Any = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    confidence: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
    """Run a direct full-image/crop prediction with the configured model type."""
    model_type = str(config["model"].get("type", "ultralytics")).strip().lower()
    if model_type in {"rfdetr", "rf-detr", "rf_detr"}:
        return predict_image_rfdetr(image, model, config, source=source, width=width, height=height, confidence=confidence)
    return predict_image_ultralytics(image, model, config, source=source, width=width, height=height, confidence=confidence)


def coco_bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Return the center point of a COCO xywh box."""
    x, y, width, height = [float(value) for value in bbox[:4]]
    return x + width / 2.0, y + height / 2.0


def center_in_coco_bbox(center: Tuple[float, float], bbox: Sequence[float], padding_ratio: float = 0.0) -> bool:
    """Return True when a center point falls inside a COCO xywh box plus optional relative padding."""
    x, y, width, height = [float(value) for value in bbox[:4]]
    pad_x = max(0.0, float(width) * float(padding_ratio))
    pad_y = max(0.0, float(height) * float(padding_ratio))
    cx, cy = center
    return (x - pad_x) <= cx <= (x + width + pad_x) and (y - pad_y) <= cy <= (y + height + pad_y)


def centered_square_window(
    center: Tuple[float, float],
    crop_size: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Build a clamped square crop window centered on an object."""
    size = max(1, int(crop_size))
    size = min(size, max(1, int(image_width)), max(1, int(image_height)))
    cx, cy = center
    x = int(round(float(cx) - size / 2.0))
    y = int(round(float(cy) - size / 2.0))
    x = max(0, min(max(0, int(image_width) - size), x))
    y = max(0, min(max(0, int(image_height) - size), y))
    return x, y, size, size


def resolve_recheck_target_class_ids(config: Mapping[str, Any], recheck_cfg: Mapping[str, Any]) -> List[int]:
    """Resolve SAHI recheck target class IDs, defaulting to football aliases."""
    return resolve_category_class_ids(
        config.get("dataset_categories", []),
        recheck_cfg.get("target_class_ids"),
        recheck_cfg.get("target_class_names"),
        "SAHI recheck",
        default_names=["football"],
    )


def apply_sahi_recheck_batch(
    images: Sequence[ImageRecord],
    model: Any,
    config: Mapping[str, Any],
    source_rgbs: Sequence[Image.Image],
    predictions_by_image: Sequence[Sequence[Mapping[str, Any]]],
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Verify target-class SAHI boxes in batches with centered second-pass crops."""
    sahi_cfg = config["sahi"]
    recheck_cfg = dict(sahi_cfg.get("recheck", {}) or {})
    if not bool(recheck_cfg.get("enabled", False)):
        return [[dict(prediction) for prediction in predictions] for predictions in predictions_by_image], [
            {"enabled": False} for _ in predictions_by_image
        ]

    target_ids = set(resolve_recheck_target_class_ids(config, recheck_cfg))
    if not target_ids:
        return [[dict(prediction) for prediction in predictions] for predictions in predictions_by_image], [
            {"enabled": True, "target_class_ids": []} for _ in predictions_by_image
        ]

    crop_size = int(recheck_cfg.get("crop_size", max(int(sahi_cfg.get("slice_width", 640)), int(sahi_cfg.get("slice_height", 640)))))
    second_conf = float(recheck_cfg.get("second_confidence_threshold", config["model"].get("confidence_threshold", 0.25)))
    first_weight = float(recheck_cfg.get("first_weight", 0.5))
    second_weight = float(recheck_cfg.get("second_weight", 0.5))
    fused_threshold = float(recheck_cfg.get("fused_confidence_threshold", config["model"].get("confidence_threshold", 0.25)))
    center_padding_ratio = float(recheck_cfg.get("center_padding_ratio", 0.0))
    max_rechecks = int(recheck_cfg.get("max_rechecks_per_image", 50) or 0)

    stats = [
        {
            "enabled": True,
            "target_class_ids": sorted(target_ids),
            "requested": 0,
            "rechecked": 0,
            "passed": 0,
            "filtered": 0,
        }
        for _ in predictions_by_image
    ]
    task_records: List[ImageRecord] = []
    task_sources: List[Image.Image] = []
    task_widths: List[int] = []
    task_heights: List[int] = []
    task_meta: List[Dict[str, Any]] = []
    rechecked_keys: set = set()

    for image_index, (image, source_rgb, predictions) in enumerate(zip(images, source_rgbs, predictions_by_image)):
        indexed_targets = [
            (prediction_index, prediction)
            for prediction_index, prediction in enumerate(predictions)
            if int(prediction.get("category_id", -1)) in target_ids
        ]
        selected_indices = {
            prediction_index
            for prediction_index, _ in sorted(indexed_targets, key=lambda item: float(item[1].get("score", 0.0)), reverse=True)[:max_rechecks]
        } if max_rechecks > 0 else set()
        stats[image_index]["requested"] = len(indexed_targets)
        stats[image_index]["rechecked"] = len(selected_indices)

        for prediction_index, prediction in indexed_targets:
            if prediction_index not in selected_indices:
                continue
            center = coco_bbox_center(prediction.get("bbox", [0, 0, 0, 0]))
            x, y, width, height = centered_square_window(center, crop_size, image.width, image.height)
            crop = source_rgb.crop((x, y, x + width, y + height))
            task_records.append(image)
            task_sources.append(crop)
            task_widths.append(width)
            task_heights.append(height)
            task_meta.append(
                {
                    "image_index": image_index,
                    "prediction_index": prediction_index,
                    "category_id": int(prediction.get("category_id", -1)),
                    "first_prediction": prediction,
                    "crop": [int(x), int(y), int(width), int(height)],
                }
            )
            rechecked_keys.add((image_index, prediction_index))

    passed_rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if task_records:
        second_predictions_by_task, _ = predict_rfdetr_direct_batch(
            task_records,
            model,
            config,
            sources=task_sources,
            widths=task_widths,
            heights=task_heights,
            confidence=second_conf,
            batch_size=rfdetr_sahi_batch_size(config),
            engine="rfdetr_recheck",
        )
        for meta, second_predictions in zip(task_meta, second_predictions_by_task):
            image_index = int(meta["image_index"])
            prediction_index = int(meta["prediction_index"])
            image = images[image_index]
            first_prediction = meta["first_prediction"]
            x, y, width, height = meta["crop"]
            projected = shared_modes.project_predictions_to_original(second_predictions, x, y, image.width, image.height)
            matching = [
                item
                for item in projected
                if int(item.get("category_id", -1)) == int(meta["category_id"])
                and center_in_coco_bbox(
                    coco_bbox_center(item.get("bbox", [0, 0, 0, 0])),
                    first_prediction.get("bbox", [0, 0, 0, 0]),
                    center_padding_ratio,
                )
            ]
            if not matching:
                stats[image_index]["filtered"] += 1
                continue
            best_second = max(matching, key=lambda item: float(item.get("score", 0.0)))
            fused_score = first_weight * float(first_prediction.get("score", 0.0)) + second_weight * float(best_second.get("score", 0.0))
            if fused_score < fused_threshold:
                stats[image_index]["filtered"] += 1
                continue
            row = dict(first_prediction)
            row["score"] = float(fused_score)
            row["first_stage_score"] = float(first_prediction.get("score", 0.0))
            row["second_stage_score"] = float(best_second.get("score", 0.0))
            row["recheck_crop"] = [int(x), int(y), int(width), int(height)]
            row["recheck_passed"] = True
            passed_rows[(image_index, prediction_index)] = row
            stats[image_index]["passed"] += 1

    output_by_image: List[List[Dict[str, Any]]] = []
    for image_index, predictions in enumerate(predictions_by_image):
        output_predictions: List[Dict[str, Any]] = []
        for prediction_index, prediction in enumerate(predictions):
            key = (image_index, prediction_index)
            if key in rechecked_keys:
                if key in passed_rows:
                    output_predictions.append(passed_rows[key])
                continue
            output_predictions.append(dict(prediction))
        output_by_image.append(output_predictions)
    return output_by_image, stats


def apply_sahi_recheck(
    image: ImageRecord,
    model: Any,
    config: Mapping[str, Any],
    source_rgb: Image.Image,
    predictions: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Verify target-class SAHI boxes with a centered second pass and fused confidence."""
    output, stats = apply_sahi_recheck_batch([image], model, config, [source_rgb], [predictions])
    return output[0], stats[0]


def predict_images_rfdetr_full(
    images: Sequence[ImageRecord],
    model: Any,
    config: Mapping[str, Any],
    *,
    batch_size: Optional[int] = None,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Run batched RF-DETR full-image inference."""
    return predict_rfdetr_direct_batch(
        images,
        model,
        config,
        batch_size=batch_size or rfdetr_image_batch_size(config),
        engine="rfdetr",
    )


def predict_images_rfdetr_sahi(
    images: Sequence[ImageRecord],
    model: Any,
    config: Mapping[str, Any],
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Run batched direct RF-DETR SAHI-style sliced prediction."""
    if not images:
        return [], []
    sahi_cfg = config["sahi"]
    slice_batch_size = rfdetr_sahi_batch_size(config)
    windows_by_image: List[List[Tuple[int, int, int, int]]] = []
    source_rgbs: List[Image.Image] = []
    task_records: List[ImageRecord] = []
    task_sources: List[Image.Image] = []
    task_widths: List[int] = []
    task_heights: List[int] = []
    task_meta: List[Tuple[int, int, int, int, int]] = []

    for image_index, image in enumerate(images):
        windows = shared_modes.generate_slice_windows_for_size(
            width=image.width,
            height=image.height,
            slice_width=int(sahi_cfg.get("slice_width", image.width)),
            slice_height=int(sahi_cfg.get("slice_height", image.height)),
            overlap_width_ratio=float(sahi_cfg.get("overlap_width_ratio", 0.2)),
            overlap_height_ratio=float(sahi_cfg.get("overlap_height_ratio", 0.2)),
        )
        windows_by_image.append(windows)
        with Image.open(image.path) as source_image:
            source_rgb = source_image.convert("RGB")
        source_rgbs.append(source_rgb)
        for x, y, width, height in windows:
            task_records.append(image)
            task_sources.append(source_rgb.crop((x, y, x + width, y + height)))
            task_widths.append(width)
            task_heights.append(height)
            task_meta.append((image_index, x, y, width, height))

    elapsed_by_image = [0.0 for _ in images]
    all_predictions: List[List[Dict[str, Any]]] = [[] for _ in images]
    if task_records:
        slice_predictions_by_task, slice_stats = predict_rfdetr_direct_batch(
            task_records,
            model,
            config,
            sources=task_sources,
            widths=task_widths,
            heights=task_heights,
            batch_size=slice_batch_size,
            engine="rfdetr_slice",
        )
        for meta, crop_predictions, stat in zip(task_meta, slice_predictions_by_task, slice_stats):
            image_index, x, y, _, _ = meta
            image = images[image_index]
            all_predictions[image_index].extend(
                shared_modes.project_predictions_to_original(crop_predictions, x, y, image.width, image.height)
            )
            elapsed_by_image[image_index] += float(stat.get("elapsed_seconds", 0.0))

    if bool(sahi_cfg.get("standard_prediction", True)):
        full_predictions_by_image, full_stats = predict_rfdetr_direct_batch(
            images,
            model,
            config,
            batch_size=slice_batch_size,
            engine="rfdetr_standard",
        )
        for image_index, (full_predictions, stat) in enumerate(zip(full_predictions_by_image, full_stats)):
            all_predictions[image_index].extend(full_predictions)
            elapsed_by_image[image_index] += float(stat.get("elapsed_seconds", 0.0))

    postprocess_start = time.perf_counter()
    postprocessed: List[List[Dict[str, Any]]] = []
    for predictions in all_predictions:
        postprocessed.append(
            shared_modes.postprocess_sahi_coco_predictions(
                predictions,
                postprocess_type=str(sahi_cfg.get("postprocess_type", "GREEDYNMM")),
                match_metric=str(sahi_cfg.get("postprocess_match_metric", "IOS")),
                match_threshold=float(sahi_cfg.get("postprocess_match_threshold", 0.5)),
                class_agnostic=bool(sahi_cfg.get("postprocess_class_agnostic", False)),
            )
        )
    postprocess_elapsed = (time.perf_counter() - postprocess_start) / max(1, len(images))
    recheck_stats: List[Dict[str, Any]] = [{"enabled": False} for _ in images]
    if bool(dict(sahi_cfg.get("recheck", {}) or {}).get("enabled", False)):
        postprocessed, recheck_stats = apply_sahi_recheck_batch(images, model, config, source_rgbs, postprocessed)

    stats: List[Dict[str, Any]] = []
    for image_index, image in enumerate(images):
        stats.append(
            {
                "image_id": image.image_id,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
                "predictions": len(postprocessed[image_index]),
                "elapsed_seconds": elapsed_by_image[image_index] + postprocess_elapsed,
                "inference_engine": "direct_sahi",
                "slice_count": len(windows_by_image[image_index]),
                "slice_batch_size": slice_batch_size,
                "sahi_recheck": recheck_stats[image_index],
            }
        )
    return postprocessed, stats


def predict_images_rfdetr_class_crop(
    images: Sequence[ImageRecord],
    model: Any,
    config: Mapping[str, Any],
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Run batched RF-DETR class-crop inference."""
    if not images:
        return [], []
    crop_cfg = shared_modes.default_crop_config(config)
    source_predictions_by_image, source_stats = predict_rfdetr_direct_batch(
        images,
        model,
        config,
        confidence=float(crop_cfg.get("source_conf", config["model"].get("confidence_threshold", 0.25))),
        batch_size=rfdetr_image_batch_size(config),
        engine="rfdetr_class_crop_source",
    )
    output_predictions: List[Optional[List[Dict[str, Any]]]] = [None for _ in images]
    output_stats: List[Optional[Dict[str, Any]]] = [None for _ in images]
    fallback_indices: List[int] = []
    crop_records: List[ImageRecord] = []
    crop_sources: List[Image.Image] = []
    crop_widths: List[int] = []
    crop_heights: List[int] = []
    crop_meta: List[Dict[str, Any]] = []

    for image_index, (image, source_predictions) in enumerate(zip(images, source_predictions_by_image)):
        window = shared_modes.select_crop_window_from_predictions(
            source_predictions,
            crop_cfg,
            config.get("dataset_categories", []),
            image.width,
            image.height,
        )
        if window is None:
            fallback_indices.append(image_index)
            continue
        crop_x, crop_y, crop_w, crop_h, matches = window
        with Image.open(image.path) as source_image:
            crop_image = source_image.convert("RGB").crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        crop_records.append(image)
        crop_sources.append(crop_image)
        crop_widths.append(crop_w)
        crop_heights.append(crop_h)
        crop_meta.append(
            {
                "image_index": image_index,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "matches": matches,
            }
        )

    if fallback_indices:
        fallback_images = [images[index] for index in fallback_indices]
        fallback_predictions_by_image, fallback_stats = predict_rfdetr_direct_batch(
            fallback_images,
            model,
            config,
            batch_size=rfdetr_image_batch_size(config),
            engine="rfdetr",
        )
        for image_index, predictions, stat in zip(fallback_indices, fallback_predictions_by_image, fallback_stats):
            image = images[image_index]
            stat.update(
                {
                    "elapsed_seconds": float(source_stats[image_index].get("elapsed_seconds", 0.0)) + float(stat.get("elapsed_seconds", 0.0)),
                    "test_mode": "class_crop",
                    "model_input_type": "full_image",
                    "crop_fallback": True,
                    "crop_x": 0,
                    "crop_y": 0,
                    "crop_width": image.width,
                    "crop_height": image.height,
                    "crop_source_matches": 0,
                }
            )
            output_predictions[image_index] = predictions
            output_stats[image_index] = stat

    if crop_records:
        crop_predictions_by_task, crop_stats = predict_rfdetr_direct_batch(
            crop_records,
            model,
            config,
            sources=crop_sources,
            widths=crop_widths,
            heights=crop_heights,
            batch_size=rfdetr_image_batch_size(config),
            engine="rfdetr_class_crop",
        )
        for meta, crop_predictions, stat in zip(crop_meta, crop_predictions_by_task, crop_stats):
            image_index = int(meta["image_index"])
            image = images[image_index]
            crop_x = int(meta["crop_x"])
            crop_y = int(meta["crop_y"])
            projected = shared_modes.project_predictions_to_original(crop_predictions, crop_x, crop_y, image.width, image.height)
            stat.update(
                {
                    "width": image.width,
                    "height": image.height,
                    "predictions": len(projected),
                    "elapsed_seconds": float(source_stats[image_index].get("elapsed_seconds", 0.0)) + float(stat.get("elapsed_seconds", 0.0)),
                    "test_mode": "class_crop",
                    "model_input_type": "class_crop",
                    "crop_fallback": False,
                    "crop_x": crop_x,
                    "crop_y": crop_y,
                    "crop_width": int(meta["crop_width"]),
                    "crop_height": int(meta["crop_height"]),
                    "crop_source_matches": int(meta["matches"]),
                }
            )
            output_predictions[image_index] = projected
            output_stats[image_index] = stat

    return [predictions or [] for predictions in output_predictions], [
        stat or {
            "image_id": image.image_id,
            "file_name": image.file_name,
            "width": image.width,
            "height": image.height,
            "predictions": 0,
            "elapsed_seconds": float(source_stats[index].get("elapsed_seconds", 0.0)),
            "inference_engine": "rfdetr_class_crop",
            "test_mode": "class_crop",
        }
        for index, (image, stat) in enumerate(zip(images, output_stats))
    ]


def predict_images_rfdetr(
    images: Sequence[ImageRecord],
    inference_model: Any,
    config: Mapping[str, Any],
    output_dir: Path,
    visual_image_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run batched RF-DETR inference for the configured test mode."""
    _ = output_dir
    _ = visual_image_ids
    mode = shared_modes.canonical_test_mode(config)
    if mode == shared_modes.SAHI_MODE:
        predictions, stats = predict_images_rfdetr_sahi(images, inference_model, config)
    elif mode == shared_modes.CLASS_CROP_MODE:
        predictions, stats = predict_images_rfdetr_class_crop(images, inference_model, config)
    else:
        predictions, stats = predict_images_rfdetr_full(images, inference_model, config)
    return predictions, stats, []


def predict_image_sliced_direct(
    image: ImageRecord,
    model: Any,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
    """Run SAHI-style sliced prediction using the direct model adapter."""
    predictions_by_image, stats = predict_images_rfdetr_sahi([image], model, config)
    return predictions_by_image[0], stats[0], None


def predict_image_class_crop(
    image: ImageRecord,
    model: Any,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
    """Run prediction-crop-prediction and project crop detections to original coordinates."""
    crop_cfg = shared_modes.default_crop_config(config)
    start = time.perf_counter()
    source_predictions, _, _ = predict_image_direct(
        image=image,
        model=model,
        config=config,
        confidence=float(crop_cfg.get("source_conf", config["model"].get("confidence_threshold", 0.25))),
    )
    window = shared_modes.select_crop_window_from_predictions(
        source_predictions,
        crop_cfg,
        config.get("dataset_categories", []),
        image.width,
        image.height,
    )
    fallback = window is None
    if fallback:
        predictions, direct_stat, _ = predict_image_direct(image=image, model=model, config=config)
        crop_x, crop_y, crop_w, crop_h, matches = 0, 0, image.width, image.height, 0
        input_type = "full_image"
        elapsed = time.perf_counter() - start
        direct_stat.update(
            {
                "elapsed_seconds": elapsed,
                "test_mode": "class_crop",
                "model_input_type": input_type,
                "crop_fallback": True,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "crop_source_matches": matches,
            }
        )
        return predictions, direct_stat, None

    crop_x, crop_y, crop_w, crop_h, matches = window
    with Image.open(image.path) as source_image:
        crop_image = source_image.convert("RGB").crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    crop_predictions, crop_stat, _ = predict_image_direct(
        image=image,
        model=model,
        config=config,
        source=crop_image,
        width=crop_w,
        height=crop_h,
    )
    predictions = shared_modes.project_predictions_to_original(crop_predictions, crop_x, crop_y, image.width, image.height)
    elapsed = time.perf_counter() - start
    crop_stat.update(
        {
            "width": image.width,
            "height": image.height,
            "predictions": len(predictions),
            "elapsed_seconds": elapsed,
            "test_mode": "class_crop",
            "model_input_type": "class_crop",
            "crop_fallback": False,
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_width": crop_w,
            "crop_height": crop_h,
            "crop_source_matches": matches,
        }
    )
    return predictions, crop_stat, None


def predict_image(
    image: ImageRecord,
    inference_model: Any,
    config: Mapping[str, Any],
    output_dir: Path,
    save_visual: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]]]:
    """Run configured prediction for one image."""
    mode = shared_modes.canonical_test_mode(config)
    if mode == shared_modes.SAHI_MODE:
        return predict_image_sahi(image, inference_model, config, output_dir, save_visual)
    if mode == shared_modes.CLASS_CROP_MODE:
        return predict_image_class_crop(image, inference_model, config)
    return predict_image_direct(image, inference_model, config)


def inference_worker(
    records: List[Dict[str, Any]],
    config: Mapping[str, Any],
    output_dir_text: str,
    device: str,
    visual_image_ids: Sequence[int],
) -> Dict[str, Any]:
    """Multiprocessing-safe worker for image-level inference."""
    output_dir = Path(output_dir_text)
    inference_model = build_inference_model(config, device)
    visual_ids = set(int(image_id) for image_id in visual_image_ids)
    all_predictions: List[Dict[str, Any]] = []
    all_stats: List[Dict[str, Any]] = []
    visual_rows: List[Dict[str, Any]] = []
    image_records = [ImageRecord(**record) for record in records]

    if is_rfdetr_model_type(config.get("model", {})):
        predictions, stats, visuals = run_batched_rfdetr_records(
            image_records,
            inference_model,
            config,
            output_dir,
            list(visual_ids),
        )
        return {"predictions": predictions, "stats": stats, "visuals": visuals, "device": device}

    for image in image_records:
        predictions, stat, visual_row = predict_image(
            image=image,
            inference_model=inference_model,
            config=config,
            output_dir=output_dir,
            save_visual=image.image_id in visual_ids,
        )
        all_predictions.extend(predictions)
        all_stats.append(stat)
        if visual_row is not None:
            visual_rows.append(visual_row)
    return {"predictions": all_predictions, "stats": all_stats, "visuals": visual_rows, "device": device}


def chunk_records(records: Sequence[ImageRecord], chunks: int) -> List[List[Dict[str, Any]]]:
    """Split image records into balanced chunks."""
    if chunks <= 1:
        return [[record.__dict__ for record in records]]
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(chunks)]
    for index, record in enumerate(records):
        buckets[index % chunks].append(record.__dict__)
    return [bucket for bucket in buckets if bucket]


def run_batched_rfdetr_records(
    records: Sequence[ImageRecord],
    inference_model: Any,
    config: Mapping[str, Any],
    output_dir: Path,
    visual_image_ids: Sequence[int],
    progress_bar: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run RF-DETR records in image batches and flatten evaluator outputs."""
    all_predictions: List[Dict[str, Any]] = []
    all_stats: List[Dict[str, Any]] = []
    visual_rows: List[Dict[str, Any]] = []
    for batch in batched_sequence(list(records), rfdetr_image_batch_size(config)):
        predictions_by_image, stats, batch_visual_rows = predict_images_rfdetr(
            images=batch,
            inference_model=inference_model,
            config=config,
            output_dir=output_dir,
            visual_image_ids=visual_image_ids,
        )
        for predictions in predictions_by_image:
            all_predictions.extend(predictions)
        all_stats.extend(stats)
        visual_rows.extend(batch_visual_rows)
        if progress_bar is not None:
            progress_bar.update(len(batch))
    return all_predictions, all_stats, visual_rows


def run_inference(
    dataset: DatasetBundle,
    config: Mapping[str, Any],
    output_dir: Path,
    quiet: bool,
    prebuilt_model: Any = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run inference across all images."""
    devices = parse_devices(config["model"])
    if prebuilt_model is not None and len(devices) != 1:
        devices = [devices[0]]
    engine = inference_engine_name(config)
    progress_enabled = bool(config["progress"].get("images", True)) and not quiet
    # Visuals are rendered after inference so sampling can use GT and/or prediction filters.
    visual_image_ids: set = set()

    if len(devices) == 1:
        inference_model = build_inference_model(config, devices[0], prebuilt_model=prebuilt_model)
        if is_rfdetr_model_type(config.get("model", {})):
            progress_bar = tqdm(total=len(dataset.images), desc=f"{engine} inference", unit="image") if progress_enabled else None
            try:
                return run_batched_rfdetr_records(
                    dataset.images,
                    inference_model,
                    config,
                    output_dir,
                    list(visual_image_ids),
                    progress_bar=progress_bar,
                )
            finally:
                if progress_bar is not None:
                    progress_bar.close()
        all_predictions: List[Dict[str, Any]] = []
        all_stats: List[Dict[str, Any]] = []
        visual_rows: List[Dict[str, Any]] = []
        iterator: Iterable[ImageRecord] = dataset.images
        if progress_enabled:
            iterator = tqdm(dataset.images, desc=f"{engine} inference", unit="image")
        for image in iterator:
            predictions, stat, visual_row = predict_image(
                image=image,
                inference_model=inference_model,
                config=config,
                output_dir=output_dir,
                save_visual=image.image_id in visual_image_ids,
            )
            all_predictions.extend(predictions)
            all_stats.append(stat)
            if visual_row is not None:
                visual_rows.append(visual_row)
        return all_predictions, all_stats, visual_rows

    chunks = chunk_records(dataset.images, len(devices))
    blue(f"Using image-level multiprocessing on devices: {', '.join(devices)}", verbose=not quiet)
    all_predictions = []
    all_stats = []
    visual_rows = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = []
        for index, records in enumerate(chunks):
            device = devices[index % len(devices)]
            worker_visual_ids = [record["image_id"] for record in records if record["image_id"] in visual_image_ids]
            futures.append(executor.submit(inference_worker, records, config, str(output_dir), device, worker_visual_ids))
        iterator = as_completed(futures)
        if progress_enabled:
            iterator = tqdm(iterator, total=len(futures), desc=f"{engine} device chunks", unit="chunk")
        for future in iterator:
            result = future.result()
            all_predictions.extend(result["predictions"])
            all_stats.extend(result["stats"])
            visual_rows.extend(result["visuals"])
    return all_predictions, all_stats, visual_rows


def xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    """Convert COCO xywh box to xyxy numpy array."""
    x, y, width, height = map(float, box[:4])
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def area_bucket_from_area(area: float, area_ranges: Sequence[float]) -> str:
    """Return the COCO-style small/medium/large area bucket for an object."""
    small, medium, large = [float(value) for value in list(area_ranges)[:3]]
    value = float(area)
    if value < small:
        return "small"
    if value < medium:
        return "medium"
    if value < large:
        return "large"
    return "large"


def bbox_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Vectorized IoU between one xyxy box and many xyxy boxes."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def build_gt_indexes(annotations: Sequence[Mapping[str, Any]]) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], List[int]], Dict[int, int], Dict[int, int]]:
    """Build fast ground-truth indexes by image/class."""
    grouped_boxes: Dict[Tuple[int, int], List[np.ndarray]] = {}
    grouped_ann_ids: Dict[Tuple[int, int], List[int]] = {}
    gt_by_class: Dict[int, int] = {}
    gt_by_image: Dict[int, int] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        key = (image_id, category_id)
        grouped_boxes.setdefault(key, []).append(xywh_to_xyxy(annotation["bbox"]))
        grouped_ann_ids.setdefault(key, []).append(int(annotation.get("id", len(grouped_ann_ids.get(key, [])))))
        gt_by_class[category_id] = gt_by_class.get(category_id, 0) + 1
        gt_by_image[image_id] = gt_by_image.get(image_id, 0) + 1
    gt_boxes = {key: np.stack(value).astype(np.float32) for key, value in grouped_boxes.items()}
    return gt_boxes, grouped_ann_ids, gt_by_class, gt_by_image


def match_predictions_at_threshold(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    category_ids: Sequence[int],
    iou_threshold: float,
    confidence_threshold: float,
) -> Dict[str, Any]:
    """Compute operating-point TP/FP/FN metrics with greedy score-ordered matching."""
    gt_boxes, _, gt_by_class, gt_by_image = build_gt_indexes(annotations)
    matched = {key: np.zeros(len(boxes), dtype=bool) for key, boxes in gt_boxes.items()}
    rows = sorted(
        (prediction for prediction in predictions if float(prediction.get("score", 0.0)) >= confidence_threshold),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )

    tp_by_class = {category_id: 0 for category_id in category_ids}
    fp_by_class = {category_id: 0 for category_id in category_ids}
    per_image = {
        int(image_id): {
            "image_id": int(image_id),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ground_truth": gt_by_image.get(int(image_id), 0),
            "predictions": 0,
        }
        for image_id in image_ids
    }

    for prediction in rows:
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        per_image.setdefault(
            image_id,
            {"image_id": image_id, "tp": 0, "fp": 0, "fn": 0, "ground_truth": 0, "predictions": 0},
        )
        per_image[image_id]["predictions"] += 1
        key = (image_id, category_id)
        boxes = gt_boxes.get(key)
        is_tp = False
        if boxes is not None and len(boxes):
            ious = bbox_iou_one_to_many(xywh_to_xyxy(prediction["bbox"]), boxes)
            ious[matched[key]] = -1.0
            best_index = int(np.argmax(ious)) if len(ious) else -1
            if best_index >= 0 and float(ious[best_index]) >= iou_threshold:
                matched[key][best_index] = True
                is_tp = True
        if is_tp:
            tp_by_class[category_id] = tp_by_class.get(category_id, 0) + 1
            per_image[image_id]["tp"] += 1
        else:
            fp_by_class[category_id] = fp_by_class.get(category_id, 0) + 1
            per_image[image_id]["fp"] += 1

    fn_by_class = {category_id: max(0, gt_by_class.get(category_id, 0) - tp_by_class.get(category_id, 0)) for category_id in category_ids}
    for key, boxes in gt_boxes.items():
        image_id, _ = key
        unmatched = int((~matched[key]).sum())
        per_image.setdefault(
            image_id,
            {"image_id": image_id, "tp": 0, "fp": 0, "fn": 0, "ground_truth": gt_by_image.get(image_id, 0), "predictions": 0},
        )
        per_image[image_id]["fn"] += unmatched

    tp = int(sum(tp_by_class.values()))
    fp = int(sum(fp_by_class.values()))
    fn = int(sum(fn_by_class.values()))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    per_class = []
    for category_id in category_ids:
        cls_tp = int(tp_by_class.get(category_id, 0))
        cls_fp = int(fp_by_class.get(category_id, 0))
        cls_fn = int(fn_by_class.get(category_id, 0))
        cls_precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) else 0.0
        cls_recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) else 0.0
        cls_f1 = (2.0 * cls_precision * cls_recall / (cls_precision + cls_recall)) if (cls_precision + cls_recall) else 0.0
        per_class.append(
            {
                "category_id": category_id,
                "tp": cls_tp,
                "fp": cls_fp,
                "fn": cls_fn,
                "precision": cls_precision,
                "recall": cls_recall,
                "f1": cls_f1,
                "instances": int(gt_by_class.get(category_id, 0)),
            }
        )

    for row in per_image.values():
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if (row["tp"] + row["fp"]) else 0.0
        row["recall"] = row["tp"] / (row["tp"] + row["fn"]) if (row["tp"] + row["fn"]) else 0.0
        row["f1"] = (2.0 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])) if (row["precision"] + row["recall"]) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
        "per_image": list(per_image.values()),
    }


def match_predictions_by_area_at_threshold(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    category_ids: Sequence[int],
    area_ranges: Sequence[float],
    iou_threshold: float,
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    """Compute operating-point metrics by category and small/medium/large area bucket."""
    gt_boxes, _, _, _ = build_gt_indexes(annotations)
    matched = {key: np.zeros(len(boxes), dtype=bool) for key, boxes in gt_boxes.items()}
    gt_meta: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    counts: Dict[Tuple[int, str], Dict[str, int]] = {
        (int(category_id), bucket): {"tp": 0, "fp": 0, "fn": 0, "instances": 0}
        for category_id in category_ids
        for bucket in ("small", "medium", "large")
    }

    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        bucket = area_bucket_from_area(float(annotation.get("area", 0.0)), area_ranges)
        counts.setdefault((category_id, bucket), {"tp": 0, "fp": 0, "fn": 0, "instances": 0})
        counts[(category_id, bucket)]["instances"] += 1
        gt_meta.setdefault((image_id, category_id), []).append(annotation)

    rows = sorted(
        (prediction for prediction in predictions if float(prediction.get("score", 0.0)) >= confidence_threshold),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    for prediction in rows:
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        key = (image_id, category_id)
        boxes = gt_boxes.get(key)
        best_index = -1
        is_tp = False
        if boxes is not None and len(boxes):
            ious = bbox_iou_one_to_many(xywh_to_xyxy(prediction["bbox"]), boxes)
            ious[matched[key]] = -1.0
            best_index = int(np.argmax(ious)) if len(ious) else -1
            if best_index >= 0 and float(ious[best_index]) >= iou_threshold:
                matched[key][best_index] = True
                is_tp = True
        if is_tp:
            annotation = gt_meta.get(key, [])[best_index]
            bucket = area_bucket_from_area(float(annotation.get("area", 0.0)), area_ranges)
            counts.setdefault((category_id, bucket), {"tp": 0, "fp": 0, "fn": 0, "instances": 0})
            counts[(category_id, bucket)]["tp"] += 1
        else:
            pred_area = float(prediction.get("area", 0.0))
            if pred_area <= 0.0:
                bbox = prediction.get("bbox", [0, 0, 0, 0])
                pred_area = float(bbox[2]) * float(bbox[3]) if len(bbox) >= 4 else 0.0
            bucket = area_bucket_from_area(pred_area, area_ranges)
            counts.setdefault((category_id, bucket), {"tp": 0, "fp": 0, "fn": 0, "instances": 0})
            counts[(category_id, bucket)]["fp"] += 1

    for key in gt_boxes:
        image_id, category_id = key
        annotations_for_key = gt_meta.get((image_id, category_id), [])
        for index, is_matched in enumerate(matched[key]):
            if bool(is_matched):
                continue
            annotation = annotations_for_key[index] if index < len(annotations_for_key) else {}
            bucket = area_bucket_from_area(float(annotation.get("area", 0.0)), area_ranges)
            counts.setdefault((category_id, bucket), {"tp": 0, "fp": 0, "fn": 0, "instances": 0})
            counts[(category_id, bucket)]["fn"] += 1

    output_rows: List[Dict[str, Any]] = []
    for category_id in category_ids:
        for bucket in ("small", "medium", "large"):
            row = counts.get((int(category_id), bucket), {"tp": 0, "fp": 0, "fn": 0, "instances": 0})
            tp = int(row["tp"])
            fp = int(row["fp"])
            fn = int(row["fn"])
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            output_rows.append(
                {
                    "category_id": int(category_id),
                    "area": bucket,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "instances": int(row.get("instances", 0)),
                }
            )
    return output_rows


def threshold_sweep(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    category_ids: Sequence[int],
    thresholds: Sequence[float],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    """Compute P/R/F1 as confidence threshold decreases using incremental matching."""
    gt_boxes, _, gt_by_class, _ = build_gt_indexes(annotations)
    total_gt = int(sum(gt_by_class.values()))
    matched = {key: np.zeros(len(boxes), dtype=bool) for key, boxes in gt_boxes.items()}
    sorted_predictions = sorted(predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    thresholds_desc = sorted({float(threshold) for threshold in thresholds}, reverse=True)
    pointer = 0
    tp = 0
    fp = 0
    rows: List[Dict[str, Any]] = []

    for threshold in thresholds_desc:
        while pointer < len(sorted_predictions) and float(sorted_predictions[pointer].get("score", 0.0)) >= threshold:
            prediction = sorted_predictions[pointer]
            pointer += 1
            category_id = int(prediction.get("category_id", -1))
            if category_id not in category_ids:
                fp += 1
                continue
            key = (int(prediction["image_id"]), category_id)
            boxes = gt_boxes.get(key)
            is_tp = False
            if boxes is not None and len(boxes):
                ious = bbox_iou_one_to_many(xywh_to_xyxy(prediction["bbox"]), boxes)
                ious[matched[key]] = -1.0
                best_index = int(np.argmax(ious)) if len(ious) else -1
                if best_index >= 0 and float(ious[best_index]) >= iou_threshold:
                    matched[key][best_index] = True
                    is_tp = True
            if is_tp:
                tp += 1
            else:
                fp += 1
        fn = max(0, total_gt - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "confidence": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    return sorted(rows, key=lambda item: item["confidence"])


def build_confusion_matrix(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    category_ids: Sequence[int],
    iou_threshold: float,
    confidence_threshold: float,
) -> Tuple[np.ndarray, List[int]]:
    """Build an object-detection confusion matrix with a background row/column."""
    cat_to_index = {category_id: index for index, category_id in enumerate(category_ids)}
    bg_index = len(category_ids)
    matrix = np.zeros((len(category_ids) + 1, len(category_ids) + 1), dtype=np.int64)

    gt_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for annotation in annotations:
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(
            {
                "category_id": int(annotation["category_id"]),
                "box": xywh_to_xyxy(annotation["bbox"]),
                "matched": False,
            }
        )

    preds_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= confidence_threshold:
            preds_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)

    for image_id, image_predictions in preds_by_image.items():
        gts = gt_by_image.get(image_id, [])
        image_predictions = sorted(image_predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        for prediction in image_predictions:
            pred_category = int(prediction.get("category_id", -1))
            pred_index = cat_to_index.get(pred_category, bg_index)
            pred_box = xywh_to_xyxy(prediction["bbox"])
            best_iou = -1.0
            best_gt_index = -1
            for index, gt in enumerate(gts):
                if gt["matched"]:
                    continue
                iou = float(bbox_iou_one_to_many(pred_box, np.expand_dims(gt["box"], axis=0))[0])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = index
            if best_gt_index >= 0 and best_iou >= iou_threshold:
                true_category = int(gts[best_gt_index]["category_id"])
                true_index = cat_to_index.get(true_category, bg_index)
                matrix[true_index, pred_index] += 1
                gts[best_gt_index]["matched"] = True
            else:
                matrix[bg_index, pred_index] += 1

    for gts in gt_by_image.values():
        for gt in gts:
            if not gt["matched"]:
                true_index = cat_to_index.get(int(gt["category_id"]), bg_index)
                matrix[true_index, bg_index] += 1
    return matrix, list(category_ids)


def capture_coco_eval(gt_path: Path, predictions: List[Dict[str, Any]], config: Mapping[str, Any], image_ids: Sequence[int], category_ids: Sequence[int], quiet: bool) -> Tuple[Any, str]:
    """Run pycocotools COCOeval and capture its text summary."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    eval_cfg = config["evaluation"]
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        coco_gt = COCO(str(gt_path))
        if predictions:
            coco_dt = coco_gt.loadRes(predictions)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {"images": coco_gt.dataset.get("images", []), "categories": coco_gt.dataset.get("categories", []), "annotations": []}
            coco_dt.createIndex()
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=str(eval_cfg.get("type", "bbox")))
        coco_eval.params.imgIds = [int(image_id) for image_id in image_ids]
        coco_eval.params.catIds = [int(category_id) for category_id in category_ids]
        coco_eval.params.maxDets = [int(value) for value in eval_cfg.get("max_detections", [1, 10, 100])]
        iou_thresholds = eval_cfg.get("iou_thresholds")
        if iou_thresholds:
            coco_eval.params.iouThrs = np.array([float(value) for value in iou_thresholds], dtype=np.float64)
        recall_thresholds = eval_cfg.get("recall_thresholds")
        if recall_thresholds:
            coco_eval.params.recThrs = np.array([float(value) for value in recall_thresholds], dtype=np.float64)
        areas = [float(value) for value in eval_cfg.get("area_ranges", [1024, 9216, 10000000000])]
        coco_eval.params.areaRng = [[0, areas[2]], [0, areas[0]], [areas[0], areas[1]], [areas[1], areas[2]]]
        coco_eval.params.areaRngLbl = ["all", "small", "medium", "large"]
        coco_eval.evaluate()
        coco_eval.accumulate()
        print(summarize_coco_eval(coco_eval), end="")
    text = stdout.getvalue()
    if not quiet and text:
        print(text, end="")
    return coco_eval, text


def format_coco_summary_line(title: str, metric_type: str, iou_label: str, area_label: str, max_det: int, value: float) -> str:
    """Format one COCO-style summary row."""
    return f" {title:<18} {metric_type} @[ IoU={iou_label:<9} | area={area_label:>6s} | maxDets={max_det:>3d} ] = {value:0.3f}"


def summarize_coco_eval(coco_eval: Any) -> str:
    """Summarize COCOeval with the configured maxDets instead of pycocotools' hard-coded AP maxDets=100."""
    p = coco_eval.params
    if not getattr(coco_eval, "eval", None):
        raise RuntimeError("Please run accumulate() before summarize_coco_eval().")

    if str(p.iouType) not in {"bbox", "segm"}:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            coco_eval.summarize()
        return stdout.getvalue()

    max_det_1 = int(p.maxDets[0])
    max_det_10 = int(p.maxDets[1]) if len(p.maxDets) > 1 else max_det_1
    max_det = int(p.maxDets[-1])
    iou_range_label = f"{float(p.iouThrs[0]):0.2f}:{float(p.iouThrs[-1]):0.2f}"
    stats = np.zeros((12,), dtype=np.float64)
    rows = [
        ("Average Precision", "(AP)", iou_range_label, "all", max_det, mean_coco_precision(coco_eval, max_det=max_det), 0),
        ("Average Precision", "(AP)", "0.50", "all", max_det, mean_coco_precision(coco_eval, iou_threshold=0.5, max_det=max_det), 1),
        ("Average Precision", "(AP)", "0.75", "all", max_det, mean_coco_precision(coco_eval, iou_threshold=0.75, max_det=max_det), 2),
        ("Average Precision", "(AP)", iou_range_label, "small", max_det, mean_coco_precision(coco_eval, area_label="small", max_det=max_det), 3),
        ("Average Precision", "(AP)", iou_range_label, "medium", max_det, mean_coco_precision(coco_eval, area_label="medium", max_det=max_det), 4),
        ("Average Precision", "(AP)", iou_range_label, "large", max_det, mean_coco_precision(coco_eval, area_label="large", max_det=max_det), 5),
        ("Average Recall", "(AR)", iou_range_label, "all", max_det_1, mean_coco_recall(coco_eval, max_det=max_det_1), 6),
        ("Average Recall", "(AR)", iou_range_label, "all", max_det_10, mean_coco_recall(coco_eval, max_det=max_det_10), 7),
        ("Average Recall", "(AR)", iou_range_label, "all", max_det, mean_coco_recall(coco_eval, max_det=max_det), 8),
        ("Average Recall", "(AR)", iou_range_label, "small", max_det, mean_coco_recall(coco_eval, area_label="small", max_det=max_det), 9),
        ("Average Recall", "(AR)", iou_range_label, "medium", max_det, mean_coco_recall(coco_eval, area_label="medium", max_det=max_det), 10),
        ("Average Recall", "(AR)", iou_range_label, "large", max_det, mean_coco_recall(coco_eval, area_label="large", max_det=max_det), 11),
    ]
    lines = []
    for title, metric_type, iou_label, area_label, row_max_det, value, index in rows:
        stats[index] = float(value)
        lines.append(format_coco_summary_line(title, metric_type, iou_label, area_label, row_max_det, float(value)))
    coco_eval.stats = stats
    return "\n".join(lines) + "\n"


def mean_coco_precision(coco_eval: Any, category_index: Optional[int] = None, iou_threshold: Optional[float] = None, area_label: str = "all", max_det: Optional[int] = None) -> float:
    """Mean precision from COCOeval precision tensor."""
    precision = coco_eval.eval.get("precision")
    if precision is None:
        return -1.0
    p = coco_eval.params
    area_index = list(p.areaRngLbl).index(area_label)
    max_det = max_det if max_det is not None else p.maxDets[-1]
    max_index = list(p.maxDets).index(max_det)
    values = precision
    if iou_threshold is not None:
        iou_indices = np.where(np.isclose(p.iouThrs, float(iou_threshold)))[0]
        if len(iou_indices) == 0:
            return -1.0
        values = values[iou_indices]
    if category_index is not None:
        values = values[:, :, category_index : category_index + 1, area_index : area_index + 1, max_index : max_index + 1]
    else:
        values = values[:, :, :, area_index : area_index + 1, max_index : max_index + 1]
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else -1.0


def mean_coco_recall(coco_eval: Any, category_index: Optional[int] = None, iou_threshold: Optional[float] = None, area_label: str = "all", max_det: Optional[int] = None) -> float:
    """Mean recall from COCOeval recall tensor."""
    recall = coco_eval.eval.get("recall")
    if recall is None:
        return -1.0
    p = coco_eval.params
    area_index = list(p.areaRngLbl).index(area_label)
    max_det = max_det if max_det is not None else p.maxDets[-1]
    max_index = list(p.maxDets).index(max_det)
    values = recall
    if iou_threshold is not None:
        iou_indices = np.where(np.isclose(p.iouThrs, float(iou_threshold)))[0]
        if len(iou_indices) == 0:
            return -1.0
        values = values[iou_indices]
    if category_index is not None:
        values = values[:, category_index : category_index + 1, area_index : area_index + 1, max_index : max_index + 1]
    else:
        values = values[:, :, area_index : area_index + 1, max_index : max_index + 1]
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else -1.0


def coco_metrics_dict(coco_eval: Any) -> Dict[str, float]:
    """Collect standard and extended COCO metrics."""
    max_det = int(coco_eval.params.maxDets[-1])
    stats = getattr(coco_eval, "stats", np.full(12, -1.0))
    metrics = {
        "mAP50-95": mean_coco_precision(coco_eval, max_det=max_det),
        "mAP50": float(stats[1]) if len(stats) > 1 else -1.0,
        "mAP75": float(stats[2]) if len(stats) > 2 else -1.0,
        "mAP50-95_small": float(stats[3]) if len(stats) > 3 else -1.0,
        "mAP50-95_medium": float(stats[4]) if len(stats) > 4 else -1.0,
        "mAP50-95_large": float(stats[5]) if len(stats) > 5 else -1.0,
        f"AR@{int(coco_eval.params.maxDets[0])}": float(stats[6]) if len(stats) > 6 else -1.0,
        f"AR@{int(coco_eval.params.maxDets[1])}": float(stats[7]) if len(stats) > 7 else -1.0,
        f"AR@{max_det}": float(stats[8]) if len(stats) > 8 else -1.0,
        f"AR@{max_det}_small": float(stats[9]) if len(stats) > 9 else -1.0,
        f"AR@{max_det}_medium": float(stats[10]) if len(stats) > 10 else -1.0,
        f"AR@{max_det}_large": float(stats[11]) if len(stats) > 11 else -1.0,
        "mAP50_small": mean_coco_precision(coco_eval, iou_threshold=0.5, area_label="small", max_det=max_det),
        "mAP50_medium": mean_coco_precision(coco_eval, iou_threshold=0.5, area_label="medium", max_det=max_det),
        "mAP50_large": mean_coco_precision(coco_eval, iou_threshold=0.5, area_label="large", max_det=max_det),
    }
    return metrics


def coco_per_class_metrics(coco_eval: Any, categories: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collect per-class COCO AP/AR metrics."""
    cat_ids = list(coco_eval.params.catIds)
    cat_id_to_name = {int(category["id"]): str(category.get("name", category["id"])) for category in categories}
    rows = []
    max_det = int(coco_eval.params.maxDets[-1])
    for index, category_id in enumerate(cat_ids):
        rows.append(
            {
                "category_id": int(category_id),
                "class": cat_id_to_name.get(int(category_id), str(category_id)),
                "AP50-95": mean_coco_precision(coco_eval, category_index=index, max_det=max_det),
                "AP50": mean_coco_precision(coco_eval, category_index=index, iou_threshold=0.5, max_det=max_det),
                "AP75": mean_coco_precision(coco_eval, category_index=index, iou_threshold=0.75, max_det=max_det),
                f"AR@{max_det}": mean_coco_recall(coco_eval, category_index=index, max_det=max_det),
                "mAP50-95": mean_coco_precision(coco_eval, category_index=index, max_det=max_det),
                "mAP50": mean_coco_precision(coco_eval, category_index=index, iou_threshold=0.5, max_det=max_det),
                "mAP50-95_small": mean_coco_precision(coco_eval, category_index=index, area_label="small", max_det=max_det),
                "mAP50-95_medium": mean_coco_precision(coco_eval, category_index=index, area_label="medium", max_det=max_det),
                "mAP50-95_large": mean_coco_precision(coco_eval, category_index=index, area_label="large", max_det=max_det),
                "mAP50_small": mean_coco_precision(coco_eval, category_index=index, iou_threshold=0.5, area_label="small", max_det=max_det),
                "mAP50_medium": mean_coco_precision(coco_eval, category_index=index, iou_threshold=0.5, area_label="medium", max_det=max_det),
                "mAP50_large": mean_coco_precision(coco_eval, category_index=index, iou_threshold=0.5, area_label="large", max_det=max_det),
                f"AR@{max_det}_small": mean_coco_recall(coco_eval, category_index=index, area_label="small", max_det=max_det),
                f"AR@{max_det}_medium": mean_coco_recall(coco_eval, category_index=index, area_label="medium", max_det=max_det),
                f"AR@{max_det}_large": mean_coco_recall(coco_eval, category_index=index, area_label="large", max_det=max_det),
            }
        )
    return rows


def coco_per_class_size_metrics(coco_eval: Any, categories: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collect per-class COCO metrics as one row per small/medium/large bucket."""
    cat_ids = list(coco_eval.params.catIds)
    cat_id_to_name = {int(category["id"]): str(category.get("name", category["id"])) for category in categories}
    max_det = int(coco_eval.params.maxDets[-1])
    rows: List[Dict[str, Any]] = []
    for index, category_id in enumerate(cat_ids):
        for area_label in ("small", "medium", "large"):
            rows.append(
                {
                    "category_id": int(category_id),
                    "class": cat_id_to_name.get(int(category_id), str(category_id)),
                    "area": area_label,
                    "mAP50-95": mean_coco_precision(coco_eval, category_index=index, area_label=area_label, max_det=max_det),
                    "mAP50": mean_coco_precision(coco_eval, category_index=index, iou_threshold=0.5, area_label=area_label, max_det=max_det),
                    f"AR@{max_det}": mean_coco_recall(coco_eval, category_index=index, area_label=area_label, max_det=max_det),
                }
            )
    return rows


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    """Write rows to CSV, adding run metadata columns to every row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    fieldnames: List[str] = list(metadata.keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**metadata, **dict(row)})


def write_metrics_tables(
    output_dir: Path,
    summary: Mapping[str, Any],
    per_class: Sequence[Mapping[str, Any]],
    per_image: Sequence[Mapping[str, Any]],
    sweep_rows: Sequence[Mapping[str, Any]],
    stats_rows: Sequence[Mapping[str, Any]],
    visual_rows: Sequence[Mapping[str, Any]],
    output_info: Mapping[str, Any],
    manifest: List[Dict[str, Any]],
) -> None:
    """Write metric JSON and CSV outputs."""
    metadata = {"run_id": output_info["run_id"], "config_hash": output_info["config_hash"]}
    metrics_json = output_dir / "metrics_summary.json"
    write_json(metrics_json, {"metadata": output_info, "metrics": summary})
    manifest.append({"path": str(metrics_json), "kind": "metrics", "description": "Summary metrics JSON."})

    metrics_csv = output_dir / "metrics_summary.csv"
    write_table(metrics_csv, [{"metric": key, "value": value} for key, value in summary.items()], metadata)
    manifest.append({"path": str(metrics_csv), "kind": "metrics", "description": "Summary metrics CSV."})

    if per_class:
        path = output_dir / "per_class_metrics.csv"
        write_table(path, per_class, metadata)
        manifest.append({"path": str(path), "kind": "metrics", "description": "Per-class AP/P/R/F1 metrics."})
    if per_image:
        path = output_dir / "per_image_metrics.csv"
        write_table(path, per_image, metadata)
        manifest.append({"path": str(path), "kind": "metrics", "description": "Per-image TP/FP/FN metrics."})
    if sweep_rows:
        path = output_dir / "threshold_sweep.csv"
        write_table(path, sweep_rows, metadata)
        manifest.append({"path": str(path), "kind": "metrics", "description": "Confidence-threshold precision/recall/F1 sweep."})
    if stats_rows:
        path = output_dir / "inference_stats.csv"
        write_table(path, stats_rows, metadata)
        manifest.append({"path": str(path), "kind": "runtime", "description": "Per-image inference runtime statistics."})
    if visual_rows:
        path = output_dir / "visuals_manifest.csv"
        write_table(path, visual_rows, metadata)
        manifest.append({"path": str(path), "kind": "visuals", "description": "Visual output sidecar manifest with config hash."})


def setup_matplotlib() -> Any:
    """Import matplotlib with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_plot(fig: Any, path: Path, config_hash_value: str, manifest: List[Dict[str, Any]], description: str) -> None:
    """Save a matplotlib figure with config metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", metadata={"Description": f"config_hash={config_hash_value}"})
    manifest.append({"path": str(path), "kind": "plot", "description": description, "config_hash": config_hash_value})


def plot_pr_curve(coco_eval: Any, categories: Sequence[Mapping[str, Any]], output_dir: Path, output_info: Mapping[str, Any], manifest: List[Dict[str, Any]]) -> None:
    """Plot COCO PR curve at IoU 0.50."""
    plt = setup_matplotlib()
    p = coco_eval.params
    iou_indices = np.where(np.isclose(p.iouThrs, 0.5))[0]
    if not len(iou_indices):
        return
    iou_index = int(iou_indices[0])
    area_index = list(p.areaRngLbl).index("all")
    max_index = len(p.maxDets) - 1
    precision = coco_eval.eval["precision"][iou_index, :, :, area_index, max_index]
    recall = np.array(p.recThrs)
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = precision > -1
    mean_precision = np.divide(
        np.where(valid, precision, 0).sum(axis=1),
        np.maximum(valid.sum(axis=1), 1),
    )
    ax.plot(recall, mean_precision, linewidth=2.5, label="all")
    cat_id_to_name = {int(category["id"]): str(category.get("name", category["id"])) for category in categories}
    for index, category_id in enumerate(p.catIds[:10]):
        values = precision[:, index]
        if np.any(values > -1):
            ax.plot(recall, np.maximum(values, 0), linewidth=1.0, alpha=0.75, label=cat_id_to_name.get(int(category_id), str(category_id)))
    ax.set_title(f"PR Curve @ IoU 0.50\nconfig {output_info['config_hash']}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    save_plot(fig, output_dir / "PR_curve.png", output_info["config_hash"], manifest, "COCO precision-recall curve at IoU 0.50.")
    plt.close(fig)


def plot_threshold_curves(sweep_rows: Sequence[Mapping[str, Any]], output_dir: Path, output_info: Mapping[str, Any], manifest: List[Dict[str, Any]]) -> None:
    """Plot Precision, Recall, and F1 vs confidence."""
    if not sweep_rows:
        return
    plt = setup_matplotlib()
    confidence = [float(row["confidence"]) for row in sweep_rows]
    for key, title, file_name in [
        ("precision", "Precision Curve", "P_curve.png"),
        ("recall", "Recall Curve", "R_curve.png"),
        ("f1", "F1 Curve", "F1_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(confidence, [float(row[key]) for row in sweep_rows], linewidth=2.5)
        ax.set_title(f"{title}\nconfig {output_info['config_hash']}")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel(key.upper() if key != "f1" else "F1")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        save_plot(fig, output_dir / file_name, output_info["config_hash"], manifest, f"{title} vs confidence threshold.")
        plt.close(fig)


def plot_confusion(matrix: np.ndarray, labels: Sequence[str], output_dir: Path, output_info: Mapping[str, Any], manifest: List[Dict[str, Any]], normalized: bool) -> None:
    """Plot raw or normalized confusion matrix."""
    plt = setup_matplotlib()
    values = matrix.astype(np.float64)
    title = "Confusion Matrix"
    file_name = "confusion_matrix.png"
    if normalized:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
        title = "Normalized Confusion Matrix"
        file_name = "confusion_matrix_normalized.png"

    fig_size = max(7, min(18, 0.45 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(values, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{title}\nconfig {output_info['config_hash']}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    if len(labels) <= 25:
        threshold = values.max() / 2.0 if values.size else 0.0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                text = f"{values[row, col]:.2f}" if normalized else str(int(values[row, col]))
                ax.text(col, row, text, ha="center", va="center", color="white" if values[row, col] > threshold else "black", fontsize=7)
    save_plot(fig, output_dir / file_name, output_info["config_hash"], manifest, title)
    plt.close(fig)


def write_plots(
    coco_eval: Any,
    categories: Sequence[Mapping[str, Any]],
    sweep_rows: Sequence[Mapping[str, Any]],
    confusion_matrix: Optional[np.ndarray],
    confusion_labels: Optional[List[str]],
    output_dir: Path,
    output_info: Mapping[str, Any],
    manifest: List[Dict[str, Any]],
) -> None:
    """Write all plot outputs."""
    plot_pr_curve(coco_eval, categories, output_dir, output_info, manifest)
    plot_threshold_curves(sweep_rows, output_dir, output_info, manifest)
    if confusion_matrix is not None and confusion_labels is not None:
        plot_confusion(confusion_matrix, confusion_labels, output_dir, output_info, manifest, normalized=False)
        plot_confusion(confusion_matrix, confusion_labels, output_dir, output_info, manifest, normalized=True)


def estimate_resources(dataset: DatasetBundle, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Estimate output file count and disk usage before writing outputs."""
    output_cfg = config["output"]
    eval_cfg = config["evaluation"]
    engine = inference_engine_name(config)
    image_count = len(dataset.images)
    annotation_count = len(dataset.annotations)
    max_dets = int((eval_cfg.get("max_detections") or [1, 10, 100])[-1])
    sampled_sizes = []
    sampled_areas = []
    for image in dataset.images[: min(20, image_count)]:
        try:
            sampled_sizes.append(Path(image.path).stat().st_size)
        except OSError:
            pass
        sampled_areas.append(max(1, int(image.width) * int(image.height)))
    avg_image_size = int(sum(sampled_sizes) / len(sampled_sizes)) if sampled_sizes else 300_000
    avg_image_area = int(sum(sampled_areas) / len(sampled_areas)) if sampled_areas else 640 * 640

    visual_count = 0
    visual_candidate_count: Optional[int] = 0
    visual_estimate_note = "disabled"
    if bool(output_cfg.get("save_visuals", False)):
        if str(output_cfg.get("visual_filter_source", "ground_truth")).strip().lower() == "ground_truth":
            _, visual_selection = select_visual_images(dataset, [], config)
            visual_candidate_count = int(visual_selection["candidate_count"])
            visual_count = int(visual_selection["selected_count"])
            visual_estimate_note = "exact before inference"
        else:
            max_visuals = output_cfg.get("max_visuals")
            visual_count = image_count if max_visuals is None else min(image_count, int(max_visuals))
            visual_candidate_count = None
            visual_estimate_note = "upper bound; prediction-based filters are known after inference"

    dataset_case_count = 0
    dataset_case_candidate_count: Optional[int] = 0
    dataset_case_estimate_note = "disabled"
    if bool(output_cfg.get("save_dataset_cases", True)):
        case_filter_source = str(output_cfg.get("visual_filter_source", "ground_truth")).strip().lower()
        max_cases = output_cfg.get("max_dataset_case_images")
        max_case_images = image_count if max_cases is None else min(image_count, max(0, int(max_cases)))
        if case_filter_source == "ground_truth":
            selected_cases, case_selection = select_dataset_case_images(dataset, [], config)
            dataset_case_candidate_count = int(case_selection["candidate_count"])
            if use_sahi_inference(config):
                dataset_case_count = sum(
                    min(int(output_cfg.get("max_slices_per_image", 12)), len(generate_slice_windows(image, config)))
                    for image in selected_cases
                )
            else:
                dataset_case_count = len(selected_cases)
            dataset_case_estimate_note = "exact before inference"
        else:
            dataset_case_candidate_count = None
            if use_sahi_inference(config):
                dataset_case_count = max_case_images * max(0, int(output_cfg.get("max_slices_per_image", 12)))
            else:
                dataset_case_count = max_case_images
            dataset_case_estimate_note = "upper bound; prediction-based filters are known after inference"

    error_case_count = 0
    error_case_estimate_note = "disabled"
    error_cfg = output_cfg.get("error_cases", {}) or {}
    if bool(error_cfg.get("enabled", False)):
        error_case_count = min(image_count, error_case_max_images(error_cfg))
        error_case_estimate_note = "upper bound; known after inference"

    plot_count = 0
    if bool(output_cfg.get("save_plots", True)):
        plot_count = 1
        if bool(eval_cfg.get("curves", True)):
            plot_count += 3
        if bool(eval_cfg.get("confusion_matrix", True)):
            plot_count += 2

    file_count = 0
    file_count += 4 if bool(output_cfg.get("save_config", True)) else 0
    file_count += 2 if bool(output_cfg.get("save_predictions_json", True)) else 0
    file_count += 2 if bool(output_cfg.get("save_ground_truth_json", True)) else 0
    if bool(output_cfg.get("save_metrics", True)):
        file_count += 2  # metrics_summary.json and metrics_summary.csv
        file_count += 1 if bool(eval_cfg.get("classwise", True)) else 0
        file_count += 1 if bool(eval_cfg.get("per_image_metrics", True)) else 0
        file_count += 1 if bool(eval_cfg.get("curves", True)) else 0
        file_count += 1  # inference_stats.csv
    file_count += 1 if bool(eval_cfg.get("confusion_matrix", True)) else 0
    file_count += 1 if bool(eval_cfg.get("save_coco_summary_text", True)) else 0
    file_count += plot_count
    file_count += visual_count
    file_count += 2 if bool(output_cfg.get("save_visuals", False)) else 0
    file_count += dataset_case_count
    file_count += 2 if bool(output_cfg.get("save_dataset_cases", True)) else 0
    file_count += error_case_count
    file_count += 3 if bool(error_cfg.get("enabled", False)) else 0
    file_count += 1 if bool(output_cfg.get("save_output_manifest", True)) else 0

    prediction_json_bytes = image_count * max_dets * 170
    gt_json_bytes = max(20_000, annotation_count * 180 + image_count * 120)
    csv_bytes = max(20_000, image_count * 180 + len(dataset.categories) * 600)
    plot_bytes = plot_count * 350_000
    visual_bytes = visual_count * avg_image_size
    if use_sahi_inference(config):
        slice_area = int(output_cfg.get("max_slices_per_image", 12)) * int(config.get("sahi", {}).get("slice_height", 640)) * int(config.get("sahi", {}).get("slice_width", 640))
        slice_ratio = max(0.05, min(1.0, slice_area / max(1, avg_image_area * max(1, int(output_cfg.get("max_slices_per_image", 12))))))
        dataset_case_bytes = int(dataset_case_count * avg_image_size * slice_ratio)
    else:
        dataset_case_bytes = dataset_case_count * avg_image_size
    error_case_bytes = error_case_count * avg_image_size
    config_bytes = 60_000
    total_bytes = prediction_json_bytes + gt_json_bytes + csv_bytes + plot_bytes + visual_bytes + dataset_case_bytes + error_case_bytes + config_bytes

    return {
        "inference_engine": engine,
        "image_count": image_count,
        "annotation_count": annotation_count,
        "visual_count": visual_count,
        "visual_candidate_count": visual_candidate_count,
        "visual_estimate_note": visual_estimate_note,
        "dataset_case_count": dataset_case_count,
        "dataset_case_candidate_count": dataset_case_candidate_count,
        "dataset_case_estimate_note": dataset_case_estimate_note,
        "error_case_count": error_case_count,
        "error_case_estimate_note": error_case_estimate_note,
        "plot_count": plot_count,
        "estimated_file_count": file_count,
        "estimated_bytes": int(total_bytes),
        "estimated_mb": round(total_bytes / (1024 * 1024), 2),
        "avg_sampled_image_mb": round(avg_image_size / (1024 * 1024), 2),
        "max_detections_assumption": max_dets,
    }


def print_resource_estimate(estimate: Mapping[str, Any], output_dir: Path, verbose: bool) -> None:
    """Print the pre-run output estimate."""
    blue("OUTPUT RESOURCE ESTIMATE", verbose=verbose, force=True)
    print(f"Output directory: {output_dir}")
    print(f"Inference engine: {estimate['inference_engine']}")
    print(f"Images to evaluate: {estimate['image_count']}")
    print(f"Ground-truth annotations: {estimate['annotation_count']}")
    print(f"Estimated output files: {estimate['estimated_file_count']}")
    print(f"Estimated disk usage: {estimate['estimated_mb']} MB")
    print(f"Visual images to save: {estimate['visual_count']}")
    print(f"Visual candidate estimate: {estimate['visual_candidate_count'] if estimate['visual_candidate_count'] is not None else 'unknown'} ({estimate['visual_estimate_note']})")
    print(f"Dataset case images to save: {estimate['dataset_case_count']}")
    print(f"Dataset case candidate estimate: {estimate['dataset_case_candidate_count'] if estimate['dataset_case_candidate_count'] is not None else 'unknown'} ({estimate['dataset_case_estimate_note']})")
    print(f"Error-case images to save: {estimate['error_case_count']} ({estimate['error_case_estimate_note']})")
    print(f"Plot files to save: {estimate['plot_count']}")


def confirm_or_exit(config: Mapping[str, Any], estimate: Mapping[str, Any], output_dir: Path, verbose: bool) -> None:
    """Ask for developer confirmation before producing outputs."""
    runtime = config["runtime"]
    print_resource_estimate(estimate, output_dir, verbose)
    if bool(runtime.get("dry_run", False)):
        blue("DRY RUN COMPLETE", verbose=verbose, force=True)
        raise SystemExit(0)
    if bool(runtime.get("yes", False)) or not bool(runtime.get("confirm_before_run", True)):
        return
    response = input("Type YES to start inference and write these outputs: ").strip()
    if response != "YES":
        raise SystemExit("Cancelled by developer before inference.")


def add_config_metadata_to_coco(coco: Mapping[str, Any], output_info: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach reproducibility info to a COCO dictionary."""
    data = dict(coco)
    info = dict(data.get("info") or {})
    info.update({"run_id": output_info["run_id"], "config_hash": output_info["config_hash"], "config": config})
    data["info"] = info
    return data


def write_config_outputs(output_dir: Path, config: Mapping[str, Any], source_config: Path, output_info: Mapping[str, Any], manifest: List[Dict[str, Any]]) -> None:
    """Write resolved config and metadata outputs."""
    config_dir = output_dir / "config"
    resolved = config_dir / "resolved_config.yaml"
    save_yaml(resolved, config)
    manifest.append({"path": str(resolved), "kind": "config", "description": "Resolved config used to create this run."})
    if source_config.exists():
        copied = config_dir / "source_config.yaml"
        shutil.copy2(source_config, copied)
        manifest.append({"path": str(copied), "kind": "config", "description": "Original source config file."})
    metadata_path = config_dir / "run_metadata.json"
    write_json(metadata_path, output_info)
    manifest.append({"path": str(metadata_path), "kind": "config", "description": "Run metadata and config hash."})


def build_arg_parser(usage_text: str) -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate an object-detection dataset with optional SAHI sliced inference.",
        epilog=usage_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to config YAML.")
    parser.add_argument("--yes", action="store_true", help="Skip resource confirmation prompt.")
    parser.add_argument("--demo", action="store_true", help="Enable demo mode with fewer images/files.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate outputs and exit before inference.")
    parser.add_argument("--quiet", action="store_true", help="Suppress most status text.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose wrapper text.")
    parser.add_argument("--use-sahi", action="store_true", help="Use SAHI inference. This writes sliced dataset cases.")
    parser.add_argument("--no-sahi", action="store_true", help="Disable SAHI and use direct Ultralytics full-image inference.")
    parser.add_argument("--test-mode", choices=["full_image", "sahi", "class_crop"], help="Inference/test mode.")
    parser.add_argument("--dataset-format", choices=["yolo", "coco"], help="Dataset format override.")
    parser.add_argument("--data-yaml", type=str, help="YOLO data.yaml override.")
    parser.add_argument("--split", type=str, help="YOLO split override: train, val, test, or custom key.")
    parser.add_argument("--coco-json", type=str, help="COCO annotation JSON override.")
    parser.add_argument("--image-dir", type=str, help="COCO image directory override.")
    parser.add_argument("--model-path", type=str, help="Model weights/path override.")
    parser.add_argument("--model-type", type=str, help="SAHI model type override, e.g. ultralytics.")
    parser.add_argument("--device", type=str, help="Device override: cpu, cuda:0, 0, etc.")
    parser.add_argument("--devices", type=str, help="Comma-separated devices for multi-process inference, e.g. cuda:0,cuda:1.")
    parser.add_argument("--conf", type=float, help="Model confidence threshold override.")
    parser.add_argument("--slice-height", type=int, help="SAHI slice/window height.")
    parser.add_argument("--slice-width", type=int, help="SAHI slice/window width.")
    parser.add_argument("--overlap-height-ratio", type=float, help="SAHI height overlap ratio.")
    parser.add_argument("--overlap-width-ratio", type=float, help="SAHI width overlap ratio.")
    parser.add_argument("--batch-size", type=int, help="RF-DETR image and SAHI slice batch size.")
    parser.add_argument("--crop-class-names", type=str, help="Comma-separated class names used to find class_crop windows.")
    parser.add_argument("--crop-class-ids", type=str, help="Comma-separated class IDs used to find class_crop windows.")
    parser.add_argument("--crop-source-conf", type=float, help="Confidence threshold for crop-window source predictions.")
    parser.add_argument("--crop-padding-pixels", type=int, help="Fixed padding around class_crop union box.")
    parser.add_argument("--crop-padding-ratio", type=float, help="Ratio padding around class_crop union box.")
    parser.add_argument("--postprocess-type", type=str, choices=["GREEDYNMM", "NMM", "NMS", "LSNMS"], help="SAHI postprocess type.")
    parser.add_argument("--postprocess-match-metric", type=str, choices=["IOS", "IOU"], help="SAHI postprocess match metric.")
    parser.add_argument("--postprocess-match-threshold", type=float, help="SAHI postprocess match threshold.")
    parser.add_argument("--output-dir", type=str, help="Output root directory override.")
    parser.add_argument("--name", type=str, help="Output run name override.")
    parser.add_argument("--max-images", type=int, help="Limit evaluated images.")
    parser.add_argument("--classwise", action="store_true", help="Write per-class AP/P/R/F1 metrics.")
    parser.add_argument("--no-classwise", action="store_true", help="Disable per-class metrics table.")
    parser.add_argument("--save-visuals", action="store_true", help="Save prediction visualization images.")
    parser.add_argument("--no-save-visuals", action="store_true", help="Disable prediction visualization images.")
    parser.add_argument("--max-visuals", type=int, help="Maximum visualization images to save.")
    parser.add_argument("--save-dataset-cases", action="store_true", help="Save raw evaluated image cases under output/datasets.")
    parser.add_argument("--no-save-dataset-cases", action="store_true", help="Disable output/datasets case images.")
    parser.add_argument("--max-dataset-case-images", type=int, help="Maximum source images used for output/datasets cases.")
    parser.add_argument("--max-slices-per-image", type=int, help="Maximum slice case crops saved per source image when SAHI is enabled.")
    parser.add_argument("--visual-sampling-mode", choices=["random", "first", "last"], help="How to sample visual images from candidates.")
    parser.add_argument("--visual-seed", type=int, help="Random seed for visual sampling.")
    parser.add_argument("--visual-filter-class-names", type=str, help="Comma-separated class names that visual samples must contain.")
    parser.add_argument("--visual-filter-class-ids", type=str, help="Comma-separated class IDs that visual samples must contain.")
    parser.add_argument("--visual-filter-source", choices=["ground_truth", "prediction", "either", "both"], help="Use GT, predictions, either, or both for visual class filtering.")
    parser.add_argument("--visual-filter-match", choices=["any", "all"], help="Whether any or all requested classes must be present.")
    parser.add_argument("--visual-min-gt-instances", type=int, help="Minimum GT annotations required for a visual candidate.")
    parser.add_argument("--visual-min-predictions", type=int, help="Minimum predictions required for a visual candidate.")
    parser.add_argument("--visual-filter-min-score", type=float, help="Minimum prediction score used only for visual candidate filtering.")
    parser.add_argument("--visual-draw-min-score", type=float, help="Minimum prediction score drawn in visual images.")
    parser.add_argument("--draw-ground-truth", action="store_true", help="Draw ground-truth boxes in visual images.")
    parser.add_argument("--no-draw-ground-truth", action="store_true", help="Do not draw ground-truth boxes in visual images.")
    parser.add_argument("--draw-predictions", action="store_true", help="Draw prediction boxes in visual images.")
    parser.add_argument("--no-draw-predictions", action="store_true", help="Do not draw prediction boxes in visual images.")
    parser.add_argument("--extra", action="append", default=[], help="Dot override like section.key=value. Can be repeated.")
    return parser


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate CLI arguments into config overrides."""
    overrides: Dict[str, Any] = {}
    if args.yes:
        set_nested(overrides, "runtime.yes", True)
    if args.demo:
        set_nested(overrides, "demo.enabled", True)
    if args.dry_run:
        set_nested(overrides, "runtime.dry_run", True)
    if args.quiet:
        set_nested(overrides, "runtime.quiet", True)
    if args.verbose:
        set_nested(overrides, "runtime.verbose", True)
        set_nested(overrides, "runtime.quiet", False)
    if args.use_sahi and args.no_sahi:
        raise ValueError("Choose only one of --use-sahi or --no-sahi.")
    if args.test_mode:
        set_nested(overrides, "inference.mode", args.test_mode)
        set_nested(overrides, "test_mode.mode", args.test_mode)
        set_nested(overrides, "inference.use_sahi", args.test_mode == "sahi")
        set_nested(overrides, "sahi.enabled", args.test_mode == "sahi")
    if args.use_sahi:
        set_nested(overrides, "inference.mode", "sahi")
        set_nested(overrides, "test_mode.mode", "sahi")
        set_nested(overrides, "inference.use_sahi", True)
        set_nested(overrides, "sahi.enabled", True)
    if args.no_sahi:
        set_nested(overrides, "inference.mode", "full_image")
        set_nested(overrides, "test_mode.mode", "full_image")
        set_nested(overrides, "inference.use_sahi", False)
        set_nested(overrides, "sahi.enabled", False)
    simple = {
        "dataset.format": args.dataset_format,
        "dataset.data_yaml": args.data_yaml,
        "dataset.split": args.split,
        "dataset.coco_json": args.coco_json,
        "dataset.image_dir": args.image_dir,
        "dataset.max_images": args.max_images,
        "model.path": args.model_path,
        "model.type": args.model_type,
        "model.device": args.device,
        "model.confidence_threshold": args.conf,
        "inference.batch_size": args.batch_size,
        "sahi.slice_height": args.slice_height,
        "sahi.slice_width": args.slice_width,
        "sahi.overlap_height_ratio": args.overlap_height_ratio,
        "sahi.overlap_width_ratio": args.overlap_width_ratio,
        "sahi.batch_size": args.batch_size,
        "crop.source_conf": args.crop_source_conf,
        "crop.padding_pixels": args.crop_padding_pixels,
        "crop.padding_ratio": args.crop_padding_ratio,
        "sahi.postprocess_type": args.postprocess_type,
        "sahi.postprocess_match_metric": args.postprocess_match_metric,
        "sahi.postprocess_match_threshold": args.postprocess_match_threshold,
        "output.dir": args.output_dir,
        "output.name": args.name,
        "output.max_visuals": args.max_visuals,
        "output.max_dataset_case_images": args.max_dataset_case_images,
        "output.max_slices_per_image": args.max_slices_per_image,
        "output.visual_sampling_mode": args.visual_sampling_mode,
        "output.visual_random_seed": args.visual_seed,
        "output.visual_filter_source": args.visual_filter_source,
        "output.visual_filter_match": args.visual_filter_match,
        "output.visual_min_gt_instances": args.visual_min_gt_instances,
        "output.visual_min_predictions": args.visual_min_predictions,
        "output.visual_filter_min_score": args.visual_filter_min_score,
        "output.visual_draw_min_score": args.visual_draw_min_score,
    }
    for key, value in simple.items():
        if value is not None:
            set_nested(overrides, key, value)
    if args.devices:
        set_nested(overrides, "model.devices", [part.strip() for part in args.devices.split(",") if part.strip()])
    if args.save_visuals:
        set_nested(overrides, "output.save_visuals", True)
    if args.no_save_visuals:
        set_nested(overrides, "output.save_visuals", False)
    if args.save_dataset_cases:
        set_nested(overrides, "output.save_dataset_cases", True)
    if args.no_save_dataset_cases:
        set_nested(overrides, "output.save_dataset_cases", False)
    if args.classwise:
        set_nested(overrides, "evaluation.classwise", True)
    if args.no_classwise:
        set_nested(overrides, "evaluation.classwise", False)
    if args.visual_filter_class_names:
        set_nested(overrides, "output.visual_filter_class_names", config_list(args.visual_filter_class_names))
    if args.visual_filter_class_ids:
        set_nested(overrides, "output.visual_filter_class_ids", config_list(args.visual_filter_class_ids))
    if args.crop_class_names:
        set_nested(overrides, "crop.class_names", config_list(args.crop_class_names))
    if args.crop_class_ids:
        set_nested(overrides, "crop.class_ids", config_list(args.crop_class_ids))
    if args.draw_ground_truth:
        set_nested(overrides, "output.draw_ground_truth", True)
    if args.no_draw_ground_truth:
        set_nested(overrides, "output.draw_ground_truth", False)
    if args.draw_predictions:
        set_nested(overrides, "output.draw_predictions", True)
    if args.no_draw_predictions:
        set_nested(overrides, "output.draw_predictions", False)
    for item in args.extra:
        if "=" not in item:
            raise ValueError(f"--extra must be key=value, got: {item}")
        key, value = item.split("=", 1)
        set_nested(overrides, key, parse_scalar(value))
    return overrides


def normalize_config(config: MutableMapping[str, Any], source_config: Path) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    """Resolve paths, demo settings, output name, run_id, and config hash."""
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    config = dict(config)

    runtime = config.setdefault("runtime", {})
    inference_cfg = config.setdefault("inference", {})
    dataset_cfg = config.setdefault("dataset", {})
    model_cfg = config.setdefault("model", {})
    sahi_cfg = config.setdefault("sahi", {})
    output_cfg = config.setdefault("output", {})
    eval_cfg = config.setdefault("evaluation", {})
    progress_cfg = config.setdefault("progress", {})
    demo_cfg = config.setdefault("demo", {})

    quiet = bool(runtime.get("quiet", False))
    if quiet:
        runtime["verbose"] = False
    if "mode" not in inference_cfg:
        if isinstance(config.get("test_mode"), Mapping) and config["test_mode"].get("mode"):
            inference_cfg["mode"] = config["test_mode"]["mode"]
        elif "use_sahi" in inference_cfg:
            inference_cfg["mode"] = "sahi" if bool(inference_cfg.get("use_sahi")) else "full_image"
        elif "enabled" in sahi_cfg:
            inference_cfg["mode"] = "sahi" if bool(sahi_cfg.get("enabled")) else "full_image"
        else:
            inference_cfg["mode"] = "full_image"
    mode = shared_modes.canonical_test_mode(config)
    config.setdefault("test_mode", {})["mode"] = mode
    inference_cfg["mode"] = mode
    inference_cfg["use_sahi"] = mode == shared_modes.SAHI_MODE
    sahi_cfg["enabled"] = mode == shared_modes.SAHI_MODE
    engine = inference_engine_name(config)
    inference_cfg["engine"] = engine
    inference_cfg["batch_size"] = positive_int_setting(inference_cfg.get("batch_size"), 1, "inference.batch_size")
    sahi_cfg["batch_size"] = positive_int_setting(sahi_cfg.get("batch_size"), inference_cfg["batch_size"], "sahi.batch_size")
    model_type = str(model_cfg.get("type", "ultralytics")).strip().lower()
    if mode != shared_modes.SAHI_MODE and model_type not in {"ultralytics", "rfdetr", "rf-detr", "rf_detr"}:
        raise ValueError("Direct full_image/class_crop inference supports model.type: ultralytics or rfdetr.")

    bases = [source_config.parent, PROJECT_DIR, REPO_ROOT, Path.cwd()]
    for key in ["data_yaml", "coco_json", "image_dir", "labels_dir"]:
        if dataset_cfg.get(key):
            dataset_cfg[key] = str(resolve_path(dataset_cfg[key], bases, must_exist=key in {"data_yaml", "coco_json", "image_dir"}))
    model_cfg["path"] = resolve_existing_or_raw(model_cfg.get("path"), bases)
    if model_cfg.get("config_path"):
        model_cfg["config_path"] = str(resolve_path(model_cfg["config_path"], bases, must_exist=True))
    model_cfg["device"] = normalize_device(model_cfg.get("device", "cpu"))
    model_cfg["devices"] = parse_devices(model_cfg) if model_cfg.get("devices") else []
    model_cfg.setdefault("extra_predict_args", {})
    max_detections = [int(value) for value in (eval_cfg.get("max_detections") or [1, 10, 100])]
    if len(max_detections) == 1:
        max_detections = [1, 10, max_detections[0]]
    elif len(max_detections) == 2:
        max_detections = [max_detections[0], max_detections[1], max_detections[1]]
    eval_cfg["max_detections"] = sorted(max_detections[:3])
    eval_cfg.setdefault("classwise", True)
    progress_cfg.setdefault("images", True)
    progress_cfg.setdefault("slices", False)
    progress_cfg.setdefault("visuals", True)
    progress_cfg.setdefault("dataset_cases", True)
    progress_cfg.setdefault("error_cases", True)
    output_cfg.setdefault("visual_output_subdir", "visuals")
    output_cfg.setdefault("save_dataset_cases", True)
    output_cfg.setdefault("dataset_cases_subdir", "datasets")
    output_cfg.setdefault("max_dataset_case_images", 5)
    output_cfg.setdefault("max_slices_per_image", 12)
    output_cfg.setdefault("dataset_case_sampling_mode", "random")
    output_cfg.setdefault("dataset_case_random_seed", None)
    output_cfg.setdefault("dataset_case_format", "jpg")
    output_cfg.setdefault("dataset_case_jpeg_quality", 92)
    output_cfg.setdefault("save_model_input_batches", True)
    output_cfg.setdefault("max_model_input_batches", 3)
    output_cfg.setdefault("model_input_batch_size", 9)
    output_cfg.setdefault("visual_sampling_mode", "random")
    output_cfg.setdefault("visual_sample_order", "sample")
    output_cfg.setdefault("visual_filter_source", "ground_truth")
    output_cfg.setdefault("visual_filter_match", "any")
    output_cfg.setdefault("visual_render_class_ids", [])
    output_cfg.setdefault("visual_render_class_names", [])
    output_cfg.setdefault("draw_ground_truth", True)
    output_cfg.setdefault("draw_predictions", True)
    output_cfg.setdefault("gt_color", "green")
    output_cfg.setdefault("pred_color", "red")
    if output_cfg.get("visual_draw_min_score") is None:
        output_cfg["visual_draw_min_score"] = float(model_cfg.get("confidence_threshold", 0.25))
    output_cfg.setdefault("visual_jpeg_quality", 92)
    output_cfg.setdefault("error_cases", {})
    if not isinstance(output_cfg["error_cases"], MutableMapping):
        output_cfg["error_cases"] = {}
    output_cfg["error_cases"].setdefault("enabled", False)
    has_error_class_filter = any(
        output_cfg["error_cases"].get(key) not in (None, "", [])
        for key in ("target_class_ids", "target_class_names", "class_ids", "class_names")
    )
    output_cfg["error_cases"].setdefault("target_class_names", [] if has_error_class_filter else ["football"])
    output_cfg["error_cases"].setdefault("render_class_ids", [])
    output_cfg["error_cases"].setdefault("render_class_names", [])
    output_cfg["error_cases"].setdefault("output_subdir", "error_cases")
    output_cfg["error_cases"].setdefault("max_images", None)
    output_cfg["error_cases"].setdefault("format", output_cfg.get("visual_format", "jpg"))
    output_cfg["visual_format"] = normalize_visual_format(output_cfg.get("visual_format", "jpg"))
    output_cfg["dataset_case_format"] = normalize_visual_format(output_cfg.get("dataset_case_format", output_cfg["visual_format"]))
    output_cfg["error_cases"]["format"] = normalize_visual_format(output_cfg["error_cases"].get("format", output_cfg["visual_format"]))
    if output_cfg.get("visual_random_seed") is None:
        output_cfg["visual_random_seed"] = int(runtime.get("seed", 0))
    if output_cfg.get("dataset_case_random_seed") is None:
        output_cfg["dataset_case_random_seed"] = int(runtime.get("seed", 0))
    sahi_cfg.setdefault("recheck", {})
    if not isinstance(sahi_cfg["recheck"], MutableMapping):
        sahi_cfg["recheck"] = {}
    sahi_cfg["recheck"].setdefault("enabled", False)
    sahi_cfg["recheck"].setdefault("target_class_ids", [])
    sahi_cfg["recheck"].setdefault("target_class_names", [])
    sahi_cfg["recheck"].setdefault("crop_size", max(int(sahi_cfg.get("slice_width", 640)), int(sahi_cfg.get("slice_height", 640))))
    sahi_cfg["recheck"].setdefault("second_confidence_threshold", float(model_cfg.get("confidence_threshold", 0.25)))
    sahi_cfg["recheck"].setdefault("first_weight", 0.5)
    sahi_cfg["recheck"].setdefault("second_weight", 0.5)
    sahi_cfg["recheck"].setdefault("fused_confidence_threshold", float(model_cfg.get("confidence_threshold", 0.25)))
    sahi_cfg["recheck"].setdefault("center_padding_ratio", 0.0)
    sahi_cfg["recheck"].setdefault("max_rechecks_per_image", 50)

    if bool(demo_cfg.get("enabled", False)):
        demo_max_images = int(demo_cfg.get("max_images", 8))
        if dataset_cfg.get("max_images") is None:
            dataset_cfg["max_images"] = demo_max_images
        else:
            dataset_cfg["max_images"] = min(int(dataset_cfg["max_images"]), demo_max_images)
        output_cfg["dir"] = demo_cfg.get("output_dir", output_cfg.get("dir", "demo_outputs"))
        output_cfg["save_visuals"] = bool(demo_cfg.get("save_visuals", True))
        output_cfg["save_dataset_cases"] = bool(demo_cfg.get("save_dataset_cases", True))
        demo_max_visuals = int(demo_cfg.get("max_visuals", 4))
        if output_cfg.get("max_visuals") is None:
            output_cfg["max_visuals"] = demo_max_visuals
        else:
            output_cfg["max_visuals"] = min(int(output_cfg["max_visuals"]), demo_max_visuals)
        demo_max_cases = int(demo_cfg.get("max_dataset_case_images", 3))
        if output_cfg.get("max_dataset_case_images") is None:
            output_cfg["max_dataset_case_images"] = demo_max_cases
        else:
            output_cfg["max_dataset_case_images"] = min(int(output_cfg["max_dataset_case_images"]), demo_max_cases)
        output_cfg["max_slices_per_image"] = min(int(output_cfg.get("max_slices_per_image", 12)), int(demo_cfg.get("max_slices_per_image", 4)))

    split = dataset_cfg.get("split", "coco")
    model_name = sanitize_name(Path(str(model_cfg.get("path", "model"))).stem)
    replacements = build_template_replacements(
        config,
        {"timestamp": timestamp, "date": timestamp[:8], "model": model_name, "split": split, "engine": engine},
    )
    exact_output = output_cfg.get("output_dir")
    if exact_output:
        output_dir = resolve_output_root(render_template(exact_output, replacements), source_config)
        rendered_name = output_dir.name
    else:
        output_root = resolve_output_root(render_template(output_cfg.get("dir", "outputs"), replacements), source_config)
        run_name = str(output_cfg.get("name", "det_eval_{engine}_{model}_{split}_{timestamp}"))
        rendered_name = sanitize_name(str(render_template(run_name, replacements)))
        output_dir = output_root / rendered_name
        if not bool(output_cfg.get("exist_ok", False)):
            output_dir = increment_path(output_dir)
    output_cfg["dataset_cases_subdir"] = sanitize_name(str(render_template(output_cfg["dataset_cases_subdir"], replacements)))
    output_cfg["visual_output_subdir"] = sanitize_name(str(render_template(output_cfg["visual_output_subdir"], replacements)))
    output_cfg["resolved_name"] = rendered_name
    output_cfg["resolved_dir"] = str(output_dir)
    runtime["timestamp"] = timestamp
    runtime["source_config"] = str(source_config)

    resolved = json.loads(json.dumps(config, default=json_default))
    hash_value = config_hash(resolved)
    run_id = f"{rendered_name}_{hash_value}"
    output_info = {
        "run_id": run_id,
        "config_hash": hash_value,
        "timestamp": timestamp,
        "inference_engine": engine,
        "source_config": str(source_config),
        "output_dir": str(output_dir),
    }
    return resolved, output_dir, output_info


def default_curve_thresholds() -> List[float]:
    """Return 0.00:0.01:1.00 thresholds."""
    return [round(index / 100.0, 2) for index in range(0, 101)]


def run_evaluation(
    config: MutableMapping[str, Any],
    source_config: Path,
    prebuilt_model: Any = None,
    already_normalized: bool = False,
    print_summary: bool = True,
) -> Dict[str, Any]:
    """Run a full evaluator pass from an in-memory config."""
    if already_normalized:
        output_dir = Path(config["output"]["resolved_dir"])
        output_info = {
            "run_id": config.get("runtime", {}).get("run_id", output_dir.name),
            "config_hash": config_hash(config),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    else:
        config, output_dir, output_info = normalize_config(config, source_config)
    verbose = bool(config["runtime"].get("verbose", True)) and not bool(config["runtime"].get("quiet", False))
    quiet = bool(config["runtime"].get("quiet", False))

    engine = inference_engine_name(config)
    blue(str(config["runtime"].get("banner", "OBJECT DETECTION DATASET EVALUATION")), verbose=verbose, force=not quiet)
    validate_runtime_devices(config, verbose=verbose)
    blue("Loading dataset and building COCO ground truth in memory...", verbose=verbose)
    dataset = load_dataset(config, output_info)
    config["dataset_categories"] = dataset.categories
    estimate = estimate_resources(dataset, config)
    confirm_or_exit(config, estimate, output_dir, verbose)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    if bool(config["output"].get("save_config", True)):
        write_config_outputs(output_dir, config, source_config, output_info, manifest)

    save_ground_truth = bool(config["output"].get("save_ground_truth_json", True))
    gt_path = output_dir / ("ground_truth_coco.json" if save_ground_truth else "_tmp_ground_truth_coco.json")
    if save_ground_truth:
        write_json(gt_path, add_config_metadata_to_coco(dataset.coco, output_info, config))
        manifest.append({"path": str(gt_path), "kind": "dataset", "description": "COCO ground truth used for evaluation."})
    else:
        write_json(gt_path, add_config_metadata_to_coco(dataset.coco, output_info, config))

    blue(f"Running {engine} inference...", verbose=verbose)
    predictions, stats_rows, _ = run_inference(dataset, config, output_dir, quiet=quiet, prebuilt_model=prebuilt_model)
    predictions = sorted(predictions, key=lambda item: (int(item["image_id"]), int(item["category_id"]), -float(item["score"])))

    predictions_path = output_dir / "predictions_coco.json"
    if bool(config["output"].get("save_predictions_json", True)):
        write_json(predictions_path, predictions)
        manifest.append({"path": str(predictions_path), "kind": "predictions", "description": "COCO result predictions JSON."})
        metadata_path = output_dir / "predictions_coco.metadata.json"
        write_json(metadata_path, {"metadata": output_info, "config": config, "prediction_count": len(predictions)})
        manifest.append({"path": str(metadata_path), "kind": "predictions", "description": "Prediction JSON sidecar metadata."})

    if bool(config["output"].get("save_model_input_batches", True)):
        manifest.extend(
            shared_modes.write_model_input_artifacts(
                output_dir=output_dir,
                images=dataset.images,
                categories=dataset.categories,
                annotations=dataset.annotations,
                predictions=predictions,
                config=config,
                stats_rows=stats_rows,
                prefix="test",
            )
        )

    dataset_case_rows = render_dataset_case_outputs(dataset, predictions, config, output_dir, output_info, quiet, manifest)
    visual_rows = render_visual_outputs(dataset, predictions, config, output_dir, output_info, quiet, manifest)
    error_case_rows = render_error_case_outputs(dataset, predictions, config, output_dir, output_info, quiet, manifest)

    blue("Computing COCO and operating-point metrics...", verbose=verbose)
    category_ids = [int(category["id"]) for category in dataset.categories]
    image_ids = [image.image_id for image in dataset.images]
    coco_eval, coco_text = capture_coco_eval(gt_path, predictions, config, image_ids, category_ids, quiet=quiet)
    if not save_ground_truth:
        gt_path.unlink(missing_ok=True)
    if bool(config["evaluation"].get("save_coco_summary_text", True)):
        text_path = output_dir / "coco_eval_summary.txt"
        text_path.write_text(coco_text, encoding="utf-8")
        manifest.append({"path": str(text_path), "kind": "metrics", "description": "Raw COCOeval text summary."})

    eval_cfg = config["evaluation"]
    operating_conf = eval_cfg.get("operating_confidence_threshold")
    if operating_conf is None:
        operating_conf = float(config["model"].get("confidence_threshold", 0.25))
    operating_conf = float(operating_conf)
    match_iou = float(eval_cfg.get("match_iou_threshold", 0.5))

    operating = match_predictions_at_threshold(
        predictions=predictions,
        annotations=dataset.annotations,
        image_ids=image_ids,
        category_ids=category_ids,
        iou_threshold=match_iou,
        confidence_threshold=operating_conf,
    )
    curve_thresholds = eval_cfg.get("curve_confidence_thresholds") or default_curve_thresholds()
    sweep_rows = threshold_sweep(predictions, dataset.annotations, category_ids, curve_thresholds, match_iou) if bool(eval_cfg.get("curves", True)) else []
    best_f1 = max(sweep_rows, key=lambda row: row["f1"], default={"confidence": operating_conf, "f1": operating["f1"]})

    coco_summary = coco_metrics_dict(coco_eval)
    summary: Dict[str, Any] = {
        **coco_summary,
        "Precision": operating["precision"],
        "Recall": operating["recall"],
        "F1": operating["f1"],
        "TP": operating["tp"],
        "FP": operating["fp"],
        "FN": operating["fn"],
        "operating_confidence_threshold": operating_conf,
        "match_iou_threshold": match_iou,
        "best_F1": best_f1.get("f1", 0.0),
        "best_F1_confidence": best_f1.get("confidence", operating_conf),
        "images": len(dataset.images),
        "instances": len(dataset.annotations),
        "predictions": len(predictions),
        "inference_engine": engine,
        "test_mode": shared_modes.canonical_test_mode(config),
        "dataset_case_samples": len(dataset_case_rows),
        "visual_samples": len(visual_rows),
        "error_case_samples": len(error_case_rows),
        "avg_inference_seconds_per_image": float(np.mean([row["elapsed_seconds"] for row in stats_rows])) if stats_rows else 0.0,
    }

    cat_name = {int(category["id"]): str(category.get("name", category["id"])) for category in dataset.categories}
    per_class_rows: List[Dict[str, Any]] = []
    per_class_size_rows: List[Dict[str, Any]] = []
    if bool(eval_cfg.get("classwise", True)):
        coco_per_class = {row["category_id"]: row for row in coco_per_class_metrics(coco_eval, dataset.categories)}
        area_ranges = [float(value) for value in eval_cfg.get("area_ranges", [1024, 9216, 10000000000])]
        coco_size = {
            (int(row["category_id"]), str(row["area"])): row
            for row in coco_per_class_size_metrics(coco_eval, dataset.categories)
        }
        operating_size = match_predictions_by_area_at_threshold(
            predictions=predictions,
            annotations=dataset.annotations,
            category_ids=category_ids,
            area_ranges=area_ranges,
            iou_threshold=match_iou,
            confidence_threshold=operating_conf,
        )
        class_names = {int(category["id"]): str(category.get("name", category["id"])) for category in dataset.categories}
        for row in operating_size:
            category_id = int(row["category_id"])
            area_label = str(row["area"])
            per_class_size_rows.append(
                {
                    "category_id": category_id,
                    "class": class_names.get(category_id, str(category_id)),
                    **{key: value for key, value in coco_size.get((category_id, area_label), {}).items() if key not in {"category_id", "class", "area"}},
                    **row,
                }
            )
        for row in operating["per_class"]:
            category_id = int(row["category_id"])
            per_class_rows.append(
                {
                    "category_id": category_id,
                    "class": cat_name.get(category_id, str(category_id)),
                    **{key: value for key, value in coco_per_class.get(category_id, {}).items() if key not in {"category_id", "class"}},
                    **{key: value for key, value in row.items() if key != "category_id"},
                }
            )

    per_image_rows = operating["per_image"] if bool(eval_cfg.get("per_image_metrics", True)) else []
    if per_image_rows:
        file_name_by_id = {image.image_id: image.file_name for image in dataset.images}
        for row in per_image_rows:
            row["file_name"] = file_name_by_id.get(int(row["image_id"]), "")

    if bool(config["output"].get("save_metrics", True)):
        write_metrics_tables(output_dir, summary, per_class_rows, per_image_rows, sweep_rows, stats_rows, visual_rows, output_info, manifest)
        if per_class_size_rows:
            metadata = {"run_id": output_info["run_id"], "config_hash": output_info["config_hash"]}
            per_class_size_path = output_dir / "per_class_size_metrics.csv"
            write_table(per_class_size_path, per_class_size_rows, metadata)
            manifest.append({"path": str(per_class_size_path), "kind": "metrics", "description": "Per-class small/medium/large AP/P/R/F1 metrics."})
            per_class_size_json_path = output_dir / "per_class_size_metrics.json"
            write_json(per_class_size_json_path, {"metadata": output_info, "metrics": per_class_size_rows})
            manifest.append({"path": str(per_class_size_json_path), "kind": "metrics", "description": "Per-class small/medium/large metrics JSON."})
    elif visual_rows:
        path = output_dir / "visuals_manifest.csv"
        write_table(path, visual_rows, {"run_id": output_info["run_id"], "config_hash": output_info["config_hash"]})
        manifest.append({"path": str(path), "kind": "visuals", "description": "Visual output sidecar manifest with config hash."})

    confusion = None
    confusion_labels = None
    if bool(eval_cfg.get("confusion_matrix", True)):
        matrix, ordered_ids = build_confusion_matrix(predictions, dataset.annotations, category_ids, match_iou, operating_conf)
        confusion = matrix
        confusion_labels = [cat_name.get(category_id, str(category_id)) for category_id in ordered_ids] + [BACKGROUND_LABEL]
        path = output_dir / "confusion_matrix.csv"
        write_table(
            path,
            [
                {"actual": confusion_labels[row], **{f"pred_{confusion_labels[col]}": int(matrix[row, col]) for col in range(matrix.shape[1])}}
                for row in range(matrix.shape[0])
            ],
            {"run_id": output_info["run_id"], "config_hash": output_info["config_hash"]},
        )
        manifest.append({"path": str(path), "kind": "metrics", "description": "Raw confusion matrix CSV."})

    if bool(config["output"].get("save_plots", True)):
        write_plots(coco_eval, dataset.categories, sweep_rows, confusion, confusion_labels, output_dir, output_info, manifest)

    if bool(config["output"].get("save_output_manifest", True)):
        manifest_path = output_dir / "output_manifest.json"
        write_json(manifest_path, {"metadata": output_info, "config": config, "outputs": manifest})

    if print_summary:
        blue("OBJECT DETECTION DATASET EVALUATION COMPLETE", verbose=verbose, force=not quiet)
        print(f"Output directory: {output_dir}")
        print(f"mAP50: {summary['mAP50']:.4f}")
        print(f"mAP50-95: {summary['mAP50-95']:.4f}")
        print(f"Precision: {summary['Precision']:.4f}")
        print(f"Recall: {summary['Recall']:.4f}")
        print(f"F1: {summary['F1']:.4f}")

    return {
        "config": config,
        "output_dir": output_dir,
        "output_info": output_info,
        "summary": summary,
        "per_class": per_class_rows,
        "per_class_size": per_class_size_rows,
        "per_image": per_image_rows,
        "stats": stats_rows,
        "predictions": predictions,
        "manifest": manifest,
    }


def main() -> None:
    usage = """
Detailed usage:

  1. Edit config/object_detection_dataset_evaluate.yaml first. The key fields are:
       inference.use_sahi: true for SAHI sliced inference, false for direct Ultralytics inference
       dataset.format: yolo or coco
       dataset.data_yaml: Ultralytics data.yaml when using YOLO labels
       dataset.coco_json and dataset.image_dir: COCO ground truth and image folder
       model.path: YOLO/SAHI-supported model weights
       model.device: cpu, cuda:0, cuda:1, 0, 1, or mps
       sahi.slice_height / sahi.slice_width: SAHI window size
       sahi.overlap_height_ratio / sahi.overlap_width_ratio: slice overlap
       output.save_visuals: whether to save sampled GT/prediction visual images
       output.save_dataset_cases: whether to save raw evaluated cases under output/datasets
       output.max_dataset_case_images: how many source images to export as dataset cases
       output.max_slices_per_image: how many slice crops to export per image when SAHI is enabled
       output.max_visuals: how many matching images to render
       output.visual_sampling_mode: random, first, or last
       output.visual_filter_class_names / output.visual_filter_class_ids: optional class filters
       output.visual_filter_source: ground_truth, prediction, either, or both
       evaluation.classwise: whether to write per-class AP/P/R/F1 metrics

  2. Run a dry estimate:
       uv run python object_detection_dataset_evaluator.py --dry-run

  3. Run with confirmation:
       uv run python object_detection_dataset_evaluator.py --config config/object_detection_dataset_evaluate.yaml

  4. Run without confirmation after checking the estimate:
       uv run python object_detection_dataset_evaluator.py --config config/object_detection_dataset_evaluate.yaml --yes

Example usage:

  YOLO dataset with SAHI sliced inference:
       uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py \\
         --config projects/object_detection_dataset_evaluator/config/object_detection_dataset_evaluate.yaml \\
         --use-sahi \\
         --data-yaml ultralytics/cfg/datasets/coco8.yaml \\
         --split val \\
         --model-path yolo26n.pt \\
         --device cuda:0 \\
         --slice-height 640 \\
         --slice-width 640 \\
         --save-visuals \\
         --max-visuals 20 \\
         --visual-sampling-mode random \\
         --visual-seed 42 \\
         --visual-filter-class-names person \\
         --yes

  COCO dataset without SAHI:
       uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py \\
         --no-sahi \\
         --dataset-format coco \\
         --coco-json /datasets/coco/annotations/instances_val.json \\
         --image-dir /datasets/coco/val2017 \\
         --model-path runs/detect/train/weights/best.pt \\
         --device cpu \\
         --yes

  Small demo output:
       uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py --demo --yes

  Prediction-filtered random visuals:
       uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py \\
         --save-visuals \\
         --max-visuals 30 \\
         --visual-filter-source prediction \\
         --visual-filter-class-ids 0,1 \\
         --visual-filter-min-score 0.25 \\
         --yes

  Extra config override:
       uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py \\
         --extra evaluation.match_iou_threshold=0.75 \\
         --extra output.save_plots=true
"""
    parser = build_arg_parser(usage)
    args = parser.parse_args()
    source_config = resolve_path(args.config, [PROJECT_DIR, REPO_ROOT, Path.cwd()], must_exist=True)
    config = load_yaml(source_config)
    deep_update(config, cli_overrides(args))
    run_evaluation(config, source_config)
    return
    config, output_dir, output_info = normalize_config(config, source_config)
    verbose = bool(config["runtime"].get("verbose", True)) and not bool(config["runtime"].get("quiet", False))
    quiet = bool(config["runtime"].get("quiet", False))

    engine = inference_engine_name(config)
    blue(str(config["runtime"].get("banner", "OBJECT DETECTION DATASET EVALUATION")), verbose=verbose, force=not quiet)
    validate_runtime_devices(config, verbose=verbose)
    blue("Loading dataset and building COCO ground truth in memory...", verbose=verbose)
    dataset = load_dataset(config, output_info)
    estimate = estimate_resources(dataset, config)
    confirm_or_exit(config, estimate, output_dir, verbose)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    if bool(config["output"].get("save_config", True)):
        write_config_outputs(output_dir, config, source_config, output_info, manifest)

    save_ground_truth = bool(config["output"].get("save_ground_truth_json", True))
    gt_path = output_dir / ("ground_truth_coco.json" if save_ground_truth else "_tmp_ground_truth_coco.json")
    if bool(config["output"].get("save_ground_truth_json", True)):
        write_json(gt_path, add_config_metadata_to_coco(dataset.coco, output_info, config))
        manifest.append({"path": str(gt_path), "kind": "dataset", "description": "COCO ground truth used for evaluation."})
    else:
        write_json(gt_path, add_config_metadata_to_coco(dataset.coco, output_info, config))

    blue(f"Running {engine} inference...", verbose=verbose)
    predictions, stats_rows, _ = run_inference(dataset, config, output_dir, quiet=quiet)
    predictions = sorted(predictions, key=lambda item: (int(item["image_id"]), int(item["category_id"]), -float(item["score"])))

    predictions_path = output_dir / "predictions_coco.json"
    if bool(config["output"].get("save_predictions_json", True)):
        write_json(predictions_path, predictions)
        manifest.append({"path": str(predictions_path), "kind": "predictions", "description": "COCO result predictions JSON."})
        metadata_path = output_dir / "predictions_coco.metadata.json"
        write_json(metadata_path, {"metadata": output_info, "config": config, "prediction_count": len(predictions)})
        manifest.append({"path": str(metadata_path), "kind": "predictions", "description": "Prediction JSON sidecar metadata."})

    dataset_case_rows = render_dataset_case_outputs(dataset, predictions, config, output_dir, output_info, quiet, manifest)
    visual_rows = render_visual_outputs(dataset, predictions, config, output_dir, output_info, quiet, manifest)

    blue("Computing COCO and operating-point metrics...", verbose=verbose)
    category_ids = [int(category["id"]) for category in dataset.categories]
    image_ids = [image.image_id for image in dataset.images]
    coco_eval, coco_text = capture_coco_eval(gt_path, predictions, config, image_ids, category_ids, quiet=quiet)
    if not save_ground_truth:
        gt_path.unlink(missing_ok=True)
    if bool(config["evaluation"].get("save_coco_summary_text", True)):
        text_path = output_dir / "coco_eval_summary.txt"
        text_path.write_text(coco_text, encoding="utf-8")
        manifest.append({"path": str(text_path), "kind": "metrics", "description": "Raw COCOeval text summary."})

    eval_cfg = config["evaluation"]
    operating_conf = eval_cfg.get("operating_confidence_threshold")
    if operating_conf is None:
        operating_conf = float(config["model"].get("confidence_threshold", 0.25))
    operating_conf = float(operating_conf)
    match_iou = float(eval_cfg.get("match_iou_threshold", 0.5))

    operating = match_predictions_at_threshold(
        predictions=predictions,
        annotations=dataset.annotations,
        image_ids=image_ids,
        category_ids=category_ids,
        iou_threshold=match_iou,
        confidence_threshold=operating_conf,
    )
    curve_thresholds = eval_cfg.get("curve_confidence_thresholds") or default_curve_thresholds()
    sweep_rows = threshold_sweep(predictions, dataset.annotations, category_ids, curve_thresholds, match_iou) if bool(eval_cfg.get("curves", True)) else []
    best_f1 = max(sweep_rows, key=lambda row: row["f1"], default={"confidence": operating_conf, "f1": operating["f1"]})

    coco_summary = coco_metrics_dict(coco_eval)
    summary: Dict[str, Any] = {
        **coco_summary,
        "Precision": operating["precision"],
        "Recall": operating["recall"],
        "F1": operating["f1"],
        "TP": operating["tp"],
        "FP": operating["fp"],
        "FN": operating["fn"],
        "operating_confidence_threshold": operating_conf,
        "match_iou_threshold": match_iou,
        "best_F1": best_f1.get("f1", 0.0),
        "best_F1_confidence": best_f1.get("confidence", operating_conf),
        "images": len(dataset.images),
        "instances": len(dataset.annotations),
        "predictions": len(predictions),
        "inference_engine": engine,
        "dataset_case_samples": len(dataset_case_rows),
        "visual_samples": len(visual_rows),
        "avg_inference_seconds_per_image": float(np.mean([row["elapsed_seconds"] for row in stats_rows])) if stats_rows else 0.0,
    }

    cat_name = {int(category["id"]): str(category.get("name", category["id"])) for category in dataset.categories}
    per_class_rows: List[Dict[str, Any]] = []
    if bool(eval_cfg.get("classwise", True)):
        coco_per_class = {row["category_id"]: row for row in coco_per_class_metrics(coco_eval, dataset.categories)}
        for row in operating["per_class"]:
            category_id = int(row["category_id"])
            per_class_rows.append(
                {
                    "category_id": category_id,
                    "class": cat_name.get(category_id, str(category_id)),
                    **{key: value for key, value in coco_per_class.get(category_id, {}).items() if key not in {"category_id", "class"}},
                    **{key: value for key, value in row.items() if key != "category_id"},
                }
            )

    per_image_rows = operating["per_image"] if bool(eval_cfg.get("per_image_metrics", True)) else []
    if per_image_rows:
        file_name_by_id = {image.image_id: image.file_name for image in dataset.images}
        for row in per_image_rows:
            row["file_name"] = file_name_by_id.get(int(row["image_id"]), "")

    if bool(config["output"].get("save_metrics", True)):
        write_metrics_tables(output_dir, summary, per_class_rows, per_image_rows, sweep_rows, stats_rows, visual_rows, output_info, manifest)
    elif visual_rows:
        path = output_dir / "visuals_manifest.csv"
        write_table(path, visual_rows, {"run_id": output_info["run_id"], "config_hash": output_info["config_hash"]})
        manifest.append({"path": str(path), "kind": "visuals", "description": "Visual output sidecar manifest with config hash."})

    confusion = None
    confusion_labels = None
    if bool(eval_cfg.get("confusion_matrix", True)):
        matrix, ordered_ids = build_confusion_matrix(predictions, dataset.annotations, category_ids, match_iou, operating_conf)
        confusion = matrix
        confusion_labels = [cat_name.get(category_id, str(category_id)) for category_id in ordered_ids] + [BACKGROUND_LABEL]
        write_table(
            output_dir / "confusion_matrix.csv",
            [
                {"actual": confusion_labels[row], **{f"pred_{confusion_labels[col]}": int(matrix[row, col]) for col in range(matrix.shape[1])}}
                for row in range(matrix.shape[0])
            ],
            {"run_id": output_info["run_id"], "config_hash": output_info["config_hash"]},
        )
        manifest.append({"path": str(output_dir / "confusion_matrix.csv"), "kind": "metrics", "description": "Raw confusion matrix CSV."})

    if bool(config["output"].get("save_plots", True)):
        write_plots(coco_eval, dataset.categories, sweep_rows, confusion, confusion_labels, output_dir, output_info, manifest)

    if bool(config["output"].get("save_output_manifest", True)):
        manifest_path = output_dir / "output_manifest.json"
        write_json(manifest_path, {"metadata": output_info, "config": config, "outputs": manifest})

    blue("OBJECT DETECTION DATASET EVALUATION COMPLETE", verbose=verbose, force=not quiet)
    print(f"Output directory: {output_dir}")
    print(f"mAP50: {summary['mAP50']:.4f}")
    print(f"mAP50-95: {summary['mAP50-95']:.4f}")
    print(f"Precision: {summary['Precision']:.4f}")
    print(f"Recall: {summary['Recall']:.4f}")
    print(f"F1: {summary['F1']:.4f}")


if __name__ == "__main__":
    main()
