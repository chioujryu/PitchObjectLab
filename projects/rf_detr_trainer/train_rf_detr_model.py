"""
Train RF-DETR with a config-first, Ultralytics-style workflow.

This project wraps the official RF-DETR PyTorch Lightning training stack and adds:

1. Custom output paths using output.output_dir or output.root + output.name with placeholders.
2. Configurable validation interval through RF-DETR's eval_interval.
3. Scheduled test-set evaluation every N epochs or N minutes.
4. Final test-set evaluation after training.
5. Per-epoch validation metrics written as JSON and CSV.
6. Overall and per-class test metrics written as JSON and CSV.
7. Config snapshots inside every training/test output directory.
8. Train batch grids plus validation label/prediction grids for label checks.
9. A resource/file estimate and developer confirmation before heavy output is created.
10. Linux and Windows path support.

Example usage:

    uv run python train_rf_detr_model.py --config config/rf_detr_train.yaml

    uv run python train_rf_detr_model.py \\
        --config config/rf_detr_train.yaml \\
        --dataset-dir D:/datasets/my_rf_detr_dataset \\
        --model-size medium \\
        --device 0 \\
        --epochs 100 \\
        --eval-interval 5 \\
        --test-interval-epochs 10 \\
        --output-dir D:/runs/rf_detr/my_experiment \\
        --yes

    uv run python train_rf_detr_model.py --demo --dry-run --yes

    uv run python train_rf_detr_model.py \
        --resume runs/rf_detr/train/example/checkpoint_2.ckpt \
        --output-dir runs/rf_detr/train/example \
        --exist-ok true

Notes:
    - RF-DETR dataset_file="roboflow" auto-detects Roboflow COCO and YOLO layouts.
    - dataset.source_format="auto" detects RF-DETR/Roboflow, Ultralytics YOLO,
      COCO JSON, Pascal VOC, DOTA, and LabelMe JSON datasets.
    - Non-RF-DETR sources are converted into reusable Roboflow COCO caches under
      projects/rf_detr_trainer/dataset_cache by default.
    - Scheduled in-training test is designed for single-process training. Final test
      can still be run after multi-GPU training, but periodic in-fit test is skipped
      when trainer.world_size > 1 to avoid distributed synchronization issues.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from copy import deepcopy
from datetime import datetime
from functools import wraps
from pathlib import Path, PurePosixPath, PureWindowsPath
from string import Formatter
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import colorama
import yaml
from colorama import Fore, Style
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

colorama.init(autoreset=True)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_common import test_modes as shared_modes  # noqa: E402

DEFAULT_CONFIG = PROJECT_DIR / "config" / "rf_detr_train.yaml"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
DDP_TIMESTAMP_ENV = "RF_DETR_TRAIN_TIMESTAMP"
DDP_OUTPUT_DIR_ENV = "RF_DETR_TRAIN_OUTPUT_DIR"
DDP_DATASET_DIR_ENV = "RF_DETR_TRAIN_DATASET_DIR"
DDP_DATASET_FILE_ENV = "RF_DETR_TRAIN_DATASET_FILE"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PITCHOBJECTLAB_ARCHITECTURE_KEY = 'pitchobjectlab_architecture'
PITCHOBJECTLAB_ARCHITECTURE_SCHEMA_VERSION = 3
PITCHOBJECTLAB_TENSORRT_EXPORT_ABI = 3
LEGACY_TRACKNET_CONFIG_ALIASES = {
    "rf_detr_train_motion_v5_medium.yaml": "rf_detr_train_medium_p2_tracknet_v5.yaml",
    "rf_detr_train_motion_v5_medium_no-p2.yaml": "rf_detr_train_medium_tracknet_v5.yaml",
    "rf_detr_train_motion_v5_large.yaml": "rf_detr_train_large_p2_tracknet_v5.yaml",
    "rf_detr_train_motion_v5_large_no-p2.yaml": "rf_detr_train_large_tracknet_v5.yaml",
    "rf_detr_train_motion_v5_large_v2.yaml": "rf_detr_train_large_p2_tracknet_v5.yaml",
    "rf_detr_train_smoke_motion_v5_medium.yaml": "rf_detr_train_smoke_temporal_tracknet_v5.yaml",
    "rf_detr_train_smoke_motion_v5_p2_medium.yaml": "rf_detr_train_smoke_temporal_tracknet_v5.yaml",
}
DATASET_SOURCE_FORMATS = {
    "auto",
    "rfdetr",
    "roboflow",
    "spatiotemporal_yolo",
    "ultralytics_yolo",
    "coco_json",
    "pascal_voc",
    "dota",
    "labelme_json",
}
CONVERTER_VERSION = "2026-05-25.1"
DATA_YAML_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
DOTA_CLASS_NAMES = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
    "container crane",
    "airport",
    "helipad",
]
MODEL_SIZE_CLASS_NAMES = {
    "base": "RFDETRBase",
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "seg-preview": "RFDETRSegPreview",
    "seg-nano": "RFDETRSegNano",
    "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium",
    "seg-large": "RFDETRSegLarge",
    "seg-xlarge": "RFDETRSegXLarge",
    "seg-2xlarge": "RFDETRSeg2XLarge",
}
MODEL_SIZE_ALIASES = {
    "rf-detr-base": "base",
    "rfdetr-base": "base",
    "rfdetrbase": "base",
    "rf-detr-nano": "nano",
    "rfdetr-nano": "nano",
    "rfdetrnano": "nano",
    "rf-detr-small": "small",
    "rfdetr-small": "small",
    "rfdetrsmall": "small",
    "rf-detr-medium": "medium",
    "rfdetr-medium": "medium",
    "rfdetrmedium": "medium",
    "rf-detr-large": "large",
    "rf-detr-large-2026": "large",
    "rfdetr-large": "large",
    "rfdetrlarge": "large",
    "rf-detr-seg-preview": "seg-preview",
    "rfdetr-seg-preview": "seg-preview",
    "rfdetrsegpreview": "seg-preview",
    "rf-detr-seg-nano": "seg-nano",
    "rfdetr-seg-nano": "seg-nano",
    "rfdetrsegnano": "seg-nano",
    "rf-detr-seg-small": "seg-small",
    "rfdetr-seg-small": "seg-small",
    "rfdetrsegsmall": "seg-small",
    "rf-detr-seg-medium": "seg-medium",
    "rfdetr-seg-medium": "seg-medium",
    "rfdetrsegmedium": "seg-medium",
    "rf-detr-seg-large": "seg-large",
    "rfdetr-seg-large": "seg-large",
    "rfdetrseglarge": "seg-large",
    "rf-detr-seg-xlarge": "seg-xlarge",
    "rfdetr-seg-xlarge": "seg-xlarge",
    "rfdetrsegxlarge": "seg-xlarge",
    "seg-xl": "seg-xlarge",
    "seg-x-large": "seg-xlarge",
    "rf-detr-seg-2xlarge": "seg-2xlarge",
    "rf-detr-seg-xxlarge": "seg-2xlarge",
    "rfdetr-seg-2xlarge": "seg-2xlarge",
    "rfdetr-seg-xxlarge": "seg-2xlarge",
    "rfdetrseg2xlarge": "seg-2xlarge",
    "rfdetrsegxxlarge": "seg-2xlarge",
    "seg-xxlarge": "seg-2xlarge",
    "seg-2xl": "seg-2xlarge",
}
PER_CLASS_FIELDS = ["class_id", "class", "ap", "ar", "f1", "precision", "recall"]
METRIC_FIELDS = ["metric", "value"]
EPOCH_VAL_METRIC_FIELDS = ["epoch", "global_step", "created_at", "metric", "value"]
TRAIN_BATCH_GRID_MAX_BATCHES = 3
VALIDATION_PREDICTION_GRID_MAX_BATCHES = 3
VALIDATION_PREDICTION_GRID_MIN_SCORE = 0.25
BATCH_GRID_TILE_SIZE = 320
BATCH_GRID_CAPTION_HEIGHT = 28
BATCH_GRID_COLUMNS = 3
BATCH_GRID_MAX_ITEMS = 9
_RFDETR_INVALID_BBOX_FILTER_PATCHED = False
_RFDETR_DETECTION_HFLIP_PATCHED = False


def blue(message: str, verbose: bool = True, force: bool = False) -> None:
    """Print blue English status text."""
    if verbose or force:
        print(Fore.BLUE + Style.BRIGHT + message)


class TeeTextStream:
    """Write console output to the original stream and an output run log."""

    def __init__(self, stream: Any, log_file: Any) -> None:
        self.stream = stream
        self.log_file = log_file

    @staticmethod
    def _is_closed(stream: Any) -> bool:
        """Return whether a stream is unavailable for writes."""
        return stream is None or bool(getattr(stream, "closed", False))

    @classmethod
    def _write_if_open(cls, stream: Any, data: str) -> Optional[int]:
        """Best-effort write for exit-time cleanup paths."""
        if cls._is_closed(stream):
            return None
        try:
            return stream.write(data)
        except (OSError, ValueError):
            return None

    @classmethod
    def _flush_if_open(cls, stream: Any) -> None:
        """Best-effort flush for streams that may already be closed."""
        if cls._is_closed(stream):
            return
        try:
            stream.flush()
        except (OSError, ValueError):
            return

    def write(self, data: str) -> int:
        written = self._write_if_open(self.stream, data)
        self._write_if_open(self.log_file, data)
        return len(data) if written is None else written

    def flush(self) -> None:
        self._flush_if_open(self.stream)
        self._flush_if_open(self.log_file)

    def isatty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


def start_run_log_capture(
    output_dir: Path,
    task: str,
    context: Optional[MutableMapping[str, Any]] = None,
    enabled: bool = True,
) -> Optional[Path]:
    """Mirror stdout/stderr into output_dir/run.log for output-producing runs."""
    if not enabled:
        return None
    if context is not None and context.get("run_log_capture"):
        return Path(str(context.get("log_path")))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    capture = {
        "stdout": sys.stdout,
        "stderr": sys.stderr,
        "log_file": log_file,
        "log_path": str(log_path),
    }
    sys.stdout = TeeTextStream(sys.stdout, log_file)  # type: ignore[assignment]
    sys.stderr = TeeTextStream(sys.stderr, log_file)  # type: ignore[assignment]
    if context is not None:
        context["run_log_capture"] = capture
        context["log_path"] = str(log_path)
    print(f"Run log: {log_path}")
    print(f"Run log started for {task} at {datetime.now().isoformat(timespec='seconds')}")
    return log_path


def stop_run_log_capture(context: Optional[MutableMapping[str, Any]]) -> None:
    """Restore stdout/stderr and close the active run log capture."""
    if context is None:
        return
    capture = context.pop("run_log_capture", None)
    if not capture:
        return
    try:
        print(f"Run log ended at {datetime.now().isoformat(timespec='seconds')}")
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = capture["stdout"]
        sys.stderr = capture["stderr"]
        capture["log_file"].close()


def parse_bool(value: Any) -> bool:
    """Parse a human-friendly boolean value."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_scalar(value: str) -> Any:
    """Parse CLI key=value values with YAML scalar/list/dict support."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def parse_limit_value(value: Any, field_name: str = "limit") -> Optional[int]:
    """Parse a positive count limit; null/all/empty means no limit."""
    if value is None:
        return None
    parsed = parse_scalar(value) if isinstance(value, str) else value
    if isinstance(parsed, str):
        text = parsed.strip().lower()
        if text in {"", "all", "none", "null"}:
            return None
        parsed = text
    if isinstance(parsed, bool):
        raise ValueError(f"{field_name} must be 'all', null, or a positive integer.")
    if isinstance(parsed, float) and not parsed.is_integer():
        raise ValueError(f"{field_name} must be an integer count when set, got {value!r}.")
    try:
        limit = int(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be 'all', null, or a positive integer, got {value!r}.") from exc
    if limit <= 0:
        raise ValueError(f"{field_name} must be positive when set, got {value!r}.")
    return limit


def parse_extra_args(values: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Parse repeated key=value CLI entries."""
    parsed: Dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected key=value, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"Expected non-empty key in {item!r}.")
        parsed[key] = parse_scalar(value)
    return parsed


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge dictionaries in-place."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path, _seen: Optional[set[Path]] = None) -> Dict[str, Any]:
    """Load a YAML mapping, resolving an optional relative ``extends`` chain."""
    path = path.expanduser().resolve()
    if _seen is None and path.name in LEGACY_TRACKNET_CONFIG_ALIASES:
        warnings.warn(
            f"{path.name} is deprecated; use "
            f"{LEGACY_TRACKNET_CONFIG_ALIASES[path.name]} instead.",
            FutureWarning,
            stacklevel=2,
        )
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"Config extends cycle detected: {chain}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    parent = data.pop("extends", None)
    if parent not in (None, ""):
        parent_path = Path(str(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = (path.parent / parent_path).resolve()
        base = load_yaml(parent_path, seen)
        return dict(deep_update(base, data))
    return data


def save_yaml(path: Path, data: Mapping[str, Any]) -> None:
    """Write YAML with stable key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(data), file, sort_keys=False, allow_unicode=True)


def parse_rank_value(value: Any) -> Optional[int]:
    """Parse a distributed rank environment value."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    with contextlib.suppress(ValueError):
        return int(text)
    return None


def is_nonzero_distributed_process(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return True for Lightning/DDP child processes that should not run parent-only setup."""
    source = os.environ if env is None else env
    for key in ("RANK", "GLOBAL_RANK", "LOCAL_RANK"):
        rank = parse_rank_value(source.get(key))
        if rank is not None and rank > 0:
            return True
    return False


def apply_distributed_child_runtime_overrides(config: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Use parent-exported runtime paths in a re-launched non-zero DDP rank."""
    output_dir = os.environ.get(DDP_OUTPUT_DIR_ENV)
    if output_dir:
        output = config.setdefault("output", {})
        output["output_dir"] = output_dir
        output["exist_ok"] = True

    dataset_dir = os.environ.get(DDP_DATASET_DIR_ENV)
    dataset_file = os.environ.get(DDP_DATASET_FILE_ENV, "roboflow")
    if dataset_dir:
        dataset = config.setdefault("dataset", {})
        train = config.setdefault("train", {})
        dataset["source_format"] = "rfdetr"
        dataset["dataset_dir"] = dataset_dir
        dataset["data_yaml"] = ""
        dataset["dataset_file"] = dataset_file
        train["dataset_dir"] = dataset_dir
        train["dataset_file"] = dataset_file

    return {
        "distributed_child": True,
        "output_dir": output_dir,
        "dataset_dir": dataset_dir,
        "dataset_file": dataset_file,
        "rank": os.environ.get("RANK") or os.environ.get("GLOBAL_RANK") or os.environ.get("LOCAL_RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
    }


def export_distributed_child_runtime(config: Mapping[str, Any], output_dir: Path, timestamp: str) -> None:
    """Export resolved parent paths so Lightning DDP child ranks do not redo setup."""
    os.environ[DDP_TIMESTAMP_ENV] = timestamp
    os.environ[DDP_OUTPUT_DIR_ENV] = str(output_dir)
    dataset = config.get("dataset", {})
    train = config.get("train", {})
    dataset_dir = train.get("dataset_dir") or dataset.get("dataset_dir")
    if dataset_dir:
        os.environ[DDP_DATASET_DIR_ENV] = str(dataset_dir)
    dataset_file = train.get("dataset_file") or dataset.get("dataset_file") or "roboflow"
    os.environ[DDP_DATASET_FILE_ENV] = str(dataset_file)


def is_abs_any_os(value: str) -> bool:
    """Return True for Windows or POSIX absolute paths regardless of host OS."""
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def is_url_like(value: str) -> bool:
    """Detect URLs and hosted weight keys that should not be normalized as local paths."""
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value))


def resolve_existing_or_raw(value: Any, bases: Sequence[Path]) -> Any:
    """Resolve local paths if they exist; otherwise return the original value."""
    if value is None or isinstance(value, (bool, int, float, list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text or is_url_like(text):
        return value

    current = Path(text).expanduser()
    if current.exists():
        return str(current.resolve())

    if not is_abs_any_os(text):
        for base in bases:
            candidate = (base / text).expanduser()
            if candidate.exists():
                return str(candidate.resolve())
    return value


def resolve_path_for_output(value: Any, bases: Sequence[Path] = ()) -> Path:
    """Resolve relative output paths from the RF-DETR trainer project directory.

    ``bases`` is retained for compatibility with older callers, but output
    location no longer depends on the process CWD or whether a candidate path
    already exists. Absolute paths remain unchanged, while ``..`` components
    in relative paths intentionally allow an explicit project-external target.
    """
    del bases
    text = str(value).strip()
    if not text:
        raise ValueError("Output path cannot be empty.")
    path = Path(text).expanduser()
    if is_abs_any_os(text):
        return path
    windows_path = PureWindowsPath(text)
    if windows_path.drive or (windows_path.root and not PurePosixPath(text).is_absolute()):
        raise ValueError(
            "Output paths must be project-relative, fully absolute, or start with '../'; "
            f"ambiguous Windows path is not supported: {text}"
        )
    return (PROJECT_DIR / path).resolve()


def render_timestamped(value: Any, timestamp: str) -> Any:
    """Replace timestamp placeholders in strings."""
    if isinstance(value, str):
        return value.replace("{timestamp}", timestamp).replace("{date}", timestamp[:8])
    return value


def sanitize_name(value: str) -> str:
    """Create a filesystem-safe name fragment."""
    text = re.sub(r"\s+", "_", str(value).strip())
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    return text.strip(" ._") or "run"


def safe_placeholder_value(value: Any) -> str:
    """Convert a config value into a path-fragment-safe placeholder string."""
    if value is None or value == "":
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:g}"
    elif isinstance(value, (list, tuple)):
        text = "-".join(safe_placeholder_value(item) for item in value)
    elif isinstance(value, Mapping):
        text = "-".join(f"{safe_placeholder_value(k)}_{safe_placeholder_value(v)}" for k, v in sorted(value.items()))
    else:
        text = str(value)
    return sanitize_name(text)


def path_name_or_default(value: Any, default: str) -> str:
    """Return the final path component for use in output folder templates."""
    if value is None or str(value).strip() == "":
        return default
    return Path(str(value)).expanduser().name or default


def build_output_template_context(config: Mapping[str, Any], timestamp: str) -> Dict[str, str]:
    """Build supported placeholders for output.root/output.name/output.output_dir templates."""
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    train = config.get("train", {})
    periodic = config.get("periodic_test", {})

    dataset_dir = train.get("dataset_dir") or dataset.get("dataset_dir") or ""
    batch = train.get("batch_size", "auto")
    grad_accum = train.get("grad_accum_steps", 1)
    try:
        effective_batch: Any = int(batch) * int(grad_accum)
    except (TypeError, ValueError):
        effective_batch = f"{batch}x{grad_accum}"

    raw_context: Dict[str, Any] = {
        "timestamp": timestamp,
        "date": timestamp[:8],
        "time": timestamp[8:],
        "model_size": model.get("size", "model"),
        "resolution": model.get("resolution") or "default",
        "tracknet_focus": (
            ((model.get("motion", {}) or {}).get("focus", {}) or {}).get("mode", "disabled")
            if bool((model.get("motion", {}) or {}).get("enabled", False)) else "disabled"
        ),
        "pretrain": path_name_or_default(model.get("pretrain_weights", "default"), "default"),
        "num_classes": model.get("num_classes") or "auto",
        "dataset_name": path_name_or_default(dataset_dir, "dataset"),
        "dataset_file": train.get("dataset_file") or dataset.get("dataset_file") or "roboflow",
        "source_format": dataset.get("source_format", "auto"),
        "device": train.get("device", "auto"),
        "epochs": train.get("epochs", "epochs"),
        "batch_size": batch,
        "grad_accum_steps": grad_accum,
        "effective_batch": effective_batch,
        "lr": train.get("lr", "lr"),
        "lr_encoder": train.get("lr_encoder", "lrenc"),
        "weight_decay": train.get("weight_decay", "wd"),
        "workers": train.get("num_workers", "workers"),
        "checkpoint_interval": train.get("checkpoint_interval", "ckpt"),
        "eval_interval": train.get("eval_interval", "eval"),
        "test_interval_epochs": periodic.get("test_interval_epochs", "test"),
        "test_interval_minutes": periodic.get("test_interval_minutes", "testmin"),
        "test_split": periodic.get("split", "test"),
        "logger_project": train.get("project") or "project",
        "logger_run": train.get("run") or "run",
    }
    return {key: safe_placeholder_value(value) for key, value in raw_context.items()}


def render_output_template(value: Any, config: Mapping[str, Any], timestamp: str) -> Any:
    """Render output folder templates with supported config placeholders."""
    if not isinstance(value, str):
        return value
    fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(value)
        if field_name
    }
    if not fields:
        return value

    context = build_output_template_context(config, timestamp)
    unknown = sorted(field for field in fields if field not in context)
    if unknown:
        available = ", ".join(sorted(context))
        raise ValueError(
            f"Unknown output path placeholder(s): {', '.join(unknown)}. "
            f"Available placeholders: {available}."
        )
    return value.format_map(context)


def json_safe_value(value: Any) -> Any:
    """Convert common numpy/torch/path/scalar objects into JSON-safe values."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return json_safe_value(value.tolist())
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            return repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return repr(value)


def write_json(path: Path, data: Any) -> None:
    """Write JSON with readable Unicode output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe_value(data), file, indent=2, ensure_ascii=False)


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe_value(row.get(key)) for key in fieldnames})


def metric_scalar_value(value: Any) -> Optional[Any]:
    """Return a JSON-safe scalar metric value, or None for non-scalar objects."""
    safe = json_safe_value(value)
    if isinstance(safe, list) and len(safe) == 1:
        safe = safe[0]
    if isinstance(safe, (bool, int, float, str)) or safe is None:
        return safe
    return None


class EpochValidationResultsCallback(Callback):
    """Persist and print detector/TrackNet diagnostics for every training epoch."""

    def __init__(self, output_dir: Path, verbose: bool) -> None:
        self.output_dir = output_dir
        self.verbose = verbose
        self.results_dir = output_dir / "epoch_results"
        self.rows: List[Dict[str, Any]] = self._load_existing_rows()
        self.written_epochs: set[int] = set()
        self._first_batch_started_at: Optional[float] = None
        self._first_batch_seconds: Dict[int, float] = {}
        self._train_diagnostic_sums: Dict[int, Dict[str, float]] = {}
        self._train_diagnostic_counts: Dict[int, int] = {}

    def _load_existing_rows(self) -> List[Dict[str, Any]]:
        """Load existing epoch metric rows so resumed runs preserve earlier results."""
        path = self.results_dir / "epoch_metrics.csv"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return [
                {field: row.get(field) for field in EPOCH_VAL_METRIC_FIELDS}
                for row in reader
            ]

    @staticmethod
    def _row_epoch(row: Mapping[str, Any]) -> int:
        """Parse an epoch number from a summary CSV row."""
        try:
            return int(row.get("epoch", -1) or -1)
        except (TypeError, ValueError):
            return -1

    def _collect_epoch_metrics(self, trainer: Any) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for source in (getattr(trainer, "callback_metrics", {}), getattr(trainer, "logged_metrics", {})):
            if not isinstance(source, Mapping):
                continue
            for key, value in source.items():
                metric_name = str(key)
                if not metric_name.startswith(("train/", "val/")):
                    continue
                scalar = metric_scalar_value(value)
                if scalar is not None:
                    metrics[metric_name] = scalar
        return dict(sorted(metrics.items()))

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Start timing the first complete temporal micro-batch of each epoch."""
        del trainer, pl_module, batch
        if batch_idx == 0:
            self._first_batch_started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record first-batch latency so a slow batch is not mistaken for a hang."""
        del outputs, batch
        epoch_number = int(getattr(trainer, "current_epoch", -1)) + 1
        if batch_idx == 0 and self._first_batch_started_at is not None:
            self._first_batch_seconds[epoch_number] = time.perf_counter() - self._first_batch_started_at
            self._first_batch_started_at = None
        diagnostics = getattr(pl_module, "_pitchobjectlab_last_train_diagnostics", {})
        if isinstance(diagnostics, Mapping) and diagnostics:
            sums = self._train_diagnostic_sums.setdefault(epoch_number, {})
            for key, value in diagnostics.items():
                scalar = metric_scalar_value(value)
                if isinstance(scalar, (int, float)):
                    sums[str(key)] = sums.get(str(key), 0.0) + float(scalar)
            self._train_diagnostic_counts[epoch_number] = (
                self._train_diagnostic_counts.get(epoch_number, 0) + 1
            )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Write per-epoch validation metrics after RF-DETR logs them."""
        del pl_module
        if bool(getattr(trainer, "sanity_checking", False)):
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return

        epoch_number = int(getattr(trainer, "current_epoch", -1)) + 1
        if epoch_number <= 0 or epoch_number in self.written_epochs:
            return

        metrics = self._collect_epoch_metrics(trainer)
        diagnostic_count = self._train_diagnostic_counts.get(epoch_number, 0)
        if diagnostic_count:
            metrics.update(
                {
                    f"train/{key}": value / diagnostic_count
                    for key, value in self._train_diagnostic_sums[epoch_number].items()
                }
            )
        if not metrics:
            return

        created_at = datetime.now().isoformat(timespec="seconds")
        global_step = int(getattr(trainer, "global_step", 0) or 0)
        first_batch_seconds = self._first_batch_seconds.get(epoch_number)
        if first_batch_seconds is not None:
            metrics["train/first_batch_seconds"] = first_batch_seconds
        payload = {
            "epoch": epoch_number,
            "global_step": global_step,
            "created_at": created_at,
            "metrics": metrics,
        }
        epoch_dir = self.results_dir / f"epoch_{epoch_number:04d}"
        write_json(epoch_dir / "val_metrics.json", payload)

        epoch_rows = [
            {
                "epoch": epoch_number,
                "global_step": global_step,
                "created_at": created_at,
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics.items()
        ]
        write_rows(epoch_dir / "val_metrics.csv", epoch_rows, EPOCH_VAL_METRIC_FIELDS)

        self.rows = [row for row in self.rows if self._row_epoch(row) != epoch_number]
        self.rows.extend(epoch_rows)
        self.rows.sort(key=lambda row: (self._row_epoch(row), str(row.get("metric", ""))))
        write_rows(self.results_dir / "epoch_metrics.csv", self.rows, EPOCH_VAL_METRIC_FIELDS)
        write_json(self.results_dir / "latest_val_metrics.json", payload)
        self.written_epochs.add(epoch_number)
        blue(f"Saved epoch {epoch_number} validation metrics to {epoch_dir}.", verbose=self.verbose)
        summary_keys = (
            "train/loss",
            "train/detector_loss",
            "train/heatmap_loss",
            "train/best_box_iou",
            "train/top_query_score",
            "train/first_batch_seconds",
        )
        summary = ", ".join(
            f"{key.removeprefix('train/')}={metrics[key]:.6g}"
            for key in summary_keys
            if isinstance(metrics.get(key), (int, float))
        )
        blue(
            f"Epoch {epoch_number} summary: global_step={global_step}"
            + (f", {summary}" if summary else ""),
            verbose=self.verbose,
            force=True,
        )


def copy_if_exists(src: Optional[Path], dst: Path) -> None:
    """Copy a file if it exists."""
    if src and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def dump_config_snapshot(
    output_dir: Path,
    merged_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    source_config: Optional[Path],
    train_config: Optional[Any] = None,
    model_config: Optional[Any] = None,
) -> None:
    """Save reproducibility config files inside an output folder."""
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config_dir / "merged_config.yaml", merged_config)
    write_json(config_dir / "run_metadata.json", metadata)
    copy_if_exists(source_config, config_dir / "source_config.yaml")
    if train_config is not None:
        train_dump = train_config.model_dump() if hasattr(train_config, "model_dump") else train_config
        save_yaml(config_dir / "rfdetr_train_config.yaml", json_safe_value(train_dump))
    if model_config is not None:
        model_dump = model_config.model_dump() if hasattr(model_config, "model_dump") else model_config
        save_yaml(config_dir / "rfdetr_model_config.yaml", json_safe_value(model_dump))


def format_bytes(num_bytes: Optional[float]) -> str:
    """Format bytes for display."""
    if num_bytes is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


RUNTIME_TIME_ESTIMATE_DEFAULTS = {
    "enabled": True,
    "use_history": True,
    "default_train_seconds_per_batch": 0.7,
    "default_test_seconds_per_image": 0.25,
    "default_inference_seconds_per_image": 0.25,
    "default_video_render_seconds_per_frame": 0.005,
}


def format_duration_hms(seconds: Optional[float]) -> str:
    """Format seconds as HH:MM:SS for estimates and elapsed runtime."""
    if seconds is None:
        return "unknown"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value):
        return "unknown"
    total = int(math.ceil(max(0.0, value)))
    if 0.0 < value < 1.0:
        total = 1
    hours, remainder = divmod(total, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def runtime_time_estimate_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return merged runtime.time_estimate settings with stable defaults."""
    settings = dict(RUNTIME_TIME_ESTIMATE_DEFAULTS)
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    configured = runtime.get("time_estimate", {}) if isinstance(runtime.get("time_estimate", {}), Mapping) else {}
    settings.update(configured)
    return settings


def positive_float_setting(settings: Mapping[str, Any], key: str) -> float:
    """Read a positive float timing setting."""
    value = settings.get(key, RUNTIME_TIME_ESTIMATE_DEFAULTS[key])
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"runtime.time_estimate.{key} must be a positive number, got {value!r}.") from exc
    if parsed <= 0 or not math.isfinite(parsed):
        raise ValueError(f"runtime.time_estimate.{key} must be a positive finite number, got {value!r}.")
    return parsed


def inference_execution_profile(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the stable backend/model profile used to partition timing history."""
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    optimization = model.get("inference_optimization", {})
    if not isinstance(optimization, Mapping):
        optimization = {}
    backend = str(optimization.get("backend", "pytorch")).strip().lower()
    backend_settings = optimization.get(backend, {})
    if not isinstance(backend_settings, Mapping):
        backend_settings = {}
    default_precision = "fp32" if backend == "pytorch" else "fp16"
    batch_values: List[int] = []

    def add_batch(value: Any) -> None:
        if isinstance(value, bool):
            return
        with contextlib.suppress(TypeError, ValueError):
            parsed = int(value)
            if parsed > 0:
                batch_values.append(parsed)

    test_settings = config.get("test", {})
    if not isinstance(test_settings, Mapping):
        test_settings = {}
    inference_settings = config.get("inference", {})
    if not isinstance(inference_settings, Mapping):
        inference_settings = {}

    if test_settings:
        test_mode_settings = test_settings.get("test_mode", config.get("test_mode", {}))
        if not isinstance(test_mode_settings, Mapping):
            test_mode_settings = {}
        workload_mode = str(test_mode_settings.get("mode", "full_image")).strip().lower()
        sahi_settings = test_settings.get("sahi", config.get("sahi", {}))
        if not isinstance(sahi_settings, Mapping):
            sahi_settings = {}
        if workload_mode == "sahi":
            add_batch(sahi_settings.get("batch_size") or test_settings.get("batch_size"))
        else:
            add_batch(test_settings.get("batch_size"))
    else:
        workload_mode = str(inference_settings.get("mode", "full_image")).strip().lower()
        test_mode_settings = {}
        sahi_settings = config.get("sahi", {})
        if not isinstance(sahi_settings, Mapping):
            sahi_settings = {}
        if workload_mode == "sahi":
            add_batch(sahi_settings.get("batch_size") or inference_settings.get("batch_size"))
        else:
            add_batch(inference_settings.get("batch_size"))
            video_settings = inference_settings.get("video", {})
            if isinstance(video_settings, Mapping):
                add_batch(video_settings.get("batch_size") or inference_settings.get("batch_size"))

    recheck_settings = sahi_settings.get("recheck", {})
    if not isinstance(recheck_settings, Mapping):
        recheck_settings = {}

    automatic_opt_batch = batch_values[0] if batch_values else 1
    automatic_max_batch = max(batch_values, default=1)
    trt_settings = optimization.get("tensorrt", {})
    if not isinstance(trt_settings, Mapping):
        trt_settings = {}
    configured_profile = trt_settings.get("profile", {})
    if not isinstance(configured_profile, Mapping):
        configured_profile = {}

    def resolved_profile_value(key: str, default: int) -> Any:
        value = configured_profile.get(key, default)
        if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
            return default
        return value

    extra_model_args = model.get("extra_model_args", {})
    if not isinstance(extra_model_args, Mapping):
        extra_model_args = {}
    return {
        "backend": backend,
        "precision": str(backend_settings.get("precision", default_precision)).strip().lower(),
        "model_size": str(model.get("size", "medium")),
        "resolution": model.get("resolution") or "default",
        "checkpoint": str(model.get("pretrain_weights") or "default"),
        "num_classes": model.get("num_classes") or "checkpoint",
        "segmentation_head": bool(extra_model_args.get("segmentation_head", False)),
        "p2": json_safe_value(model.get("p2", {}) or {}),
        "motion": json_safe_value(model.get("motion", {}) or {}),
        "tensorrt_profile": {
            "min_batch_size": resolved_profile_value("min_batch_size", 1),
            "opt_batch_size": resolved_profile_value("opt_batch_size", automatic_opt_batch),
            "max_batch_size": resolved_profile_value("max_batch_size", automatic_max_batch),
        },
        "workload": {
            "test_mode": workload_mode,
            "batch_sizes": sorted(set(batch_values)) or [1],
            "sahi_slice_height": sahi_settings.get("slice_height"),
            "sahi_slice_width": sahi_settings.get("slice_width"),
            "sahi_standard_prediction": bool(sahi_settings.get("standard_prediction", False)),
            "recheck_enabled": bool(recheck_settings.get("enabled", False)),
            "recheck_crop_size": recheck_settings.get("crop_size"),
        },
    }


def load_latest_timing_history(
    output_root: Path,
    task: str,
    execution_profile: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Load the newest sibling timing record matching task and execution profile."""
    if not output_root.exists():
        return None
    candidates = sorted(
        output_root.glob("*/run_timing.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for path in candidates:
        with contextlib.suppress(Exception):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("task") != task or data.get("success") is not True:
                continue
            if execution_profile is not None and data.get("execution_profile") != dict(execution_profile):
                continue
            if data.get("task") == task and data.get("success") is True:
                return data
    return None


def add_runtime_estimate(
    estimate: MutableMapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    task: str,
    runtime_units: float,
    default_rate_key: str,
    basis: Mapping[str, Any],
    extra_seconds: float = 0.0,
) -> MutableMapping[str, Any]:
    """Attach a rough HH:MM:SS runtime estimate to an output estimate."""
    settings = runtime_time_estimate_settings(config)
    units = max(0.0, float(runtime_units or 0.0))
    estimate["runtime_units"] = units
    estimate_basis = dict(basis)
    execution_profile = inference_execution_profile(config) if task in {"test", "inference"} else None
    if execution_profile is not None:
        estimate_basis["execution_profile"] = execution_profile
    estimate["runtime_estimate_basis"] = estimate_basis
    if not bool(settings.get("enabled", True)):
        estimate.update(
            {
                "estimated_runtime_seconds": None,
                "estimated_runtime_hms": "unknown",
                "estimated_runtime_source": "disabled",
                "estimated_runtime_confidence": "unknown",
            }
        )
        return estimate

    rate = positive_float_setting(settings, default_rate_key)
    source = "default-rate"
    if bool(settings.get("use_history", True)):
        history = load_latest_timing_history(output_dir.parent, task, execution_profile)
        throughput = history.get("throughput", {}) if isinstance(history, Mapping) else {}
        history_rate = throughput.get("seconds_per_runtime_unit") if isinstance(throughput, Mapping) else None
        with contextlib.suppress(TypeError, ValueError):
            parsed_history_rate = float(history_rate)
            if parsed_history_rate > 0 and math.isfinite(parsed_history_rate):
                rate = parsed_history_rate
                source = "history"

    seconds = units * rate + max(0.0, float(extra_seconds or 0.0))
    estimate.update(
        {
            "estimated_runtime_seconds": round(seconds, 3),
            "estimated_runtime_hms": format_duration_hms(seconds),
            "estimated_runtime_source": source,
            "estimated_runtime_confidence": "rough",
        }
    )
    return estimate


def start_run_timing(task: str, verbose: bool = True) -> Dict[str, Any]:
    """Create a mutable timing context for an entrypoint run."""
    return {
        "task": task,
        "verbose": verbose,
        "started_at_monotonic": time.monotonic(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": False,
        "outputs_created": False,
        "success": False,
    }


def finish_run_timing(context: MutableMapping[str, Any]) -> None:
    """Print elapsed time and write run_timing.json when an output dir exists."""
    try:
        ended_at_monotonic = time.monotonic()
        elapsed_seconds = max(0.0, ended_at_monotonic - float(context.get("started_at_monotonic", ended_at_monotonic)))
        elapsed_hms = format_duration_hms(elapsed_seconds)
        verbose = bool(context.get("verbose", True))
        if verbose:
            blue(f"Elapsed time: {elapsed_hms}", verbose=True, force=True)
        else:
            print(f"Elapsed time: {elapsed_hms}")

        output_dir = context.get("output_dir")
        dry_run = bool(context.get("dry_run", False))
        if output_dir is None or dry_run or not bool(context.get("outputs_created", False)):
            return
        output_path = Path(str(output_dir))
        if not output_path.exists():
            return

        estimate = dict(context.get("estimate", {}) or {})
        units = float(estimate.get("runtime_units") or 0.0)
        acceleration = context.get("acceleration", {})
        if not isinstance(acceleration, Mapping):
            acceleration = {}

        def acceleration_seconds(key: str) -> float:
            with contextlib.suppress(TypeError, ValueError):
                return max(0.0, float(acceleration.get(key, 0.0) or 0.0))
            return 0.0

        export_seconds = acceleration_seconds("export_seconds")
        build_seconds = acceleration_seconds("build_seconds")
        load_seconds = acceleration_seconds("load_seconds")
        warmup_seconds = acceleration_seconds("warmup_seconds")
        one_time_acceleration_seconds = export_seconds + build_seconds + load_seconds + warmup_seconds

        throughput_units = units
        steady_state_seconds = max(0.0, elapsed_seconds - one_time_acceleration_seconds)
        throughput_source = "entrypoint-minus-acceleration-setup"
        stage_timing = context.get("stage_timing")
        if isinstance(stage_timing, Mapping):
            with contextlib.suppress(TypeError, ValueError):
                stage_total = float(stage_timing.get("total_seconds", 0.0) or 0.0)
                stage_units = float(stage_timing.get("images_or_frames", 0.0) or 0.0)
                if stage_total >= 0.0 and stage_units > 0.0:
                    steady_state_seconds = stage_total
                    throughput_units = stage_units
                    throughput_source = "stage_timing"

        throughput: Dict[str, Any] = {
            "runtime_units": throughput_units,
            "requested_runtime_units": units,
            "steady_state_seconds": steady_state_seconds,
            "source": throughput_source,
            "engine_export_seconds": export_seconds,
            "engine_build_seconds": build_seconds,
            "engine_load_seconds": load_seconds,
            "warmup_seconds": warmup_seconds,
        }
        if throughput_units > 0:
            throughput["seconds_per_runtime_unit"] = steady_state_seconds / throughput_units

        payload = {
            "task": context.get("task"),
            "success": bool(context.get("success", False)),
            "started_at": context.get("started_at"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "elapsed_hms": elapsed_hms,
            "estimated_runtime_seconds": estimate.get("estimated_runtime_seconds"),
            "estimated_runtime_hms": estimate.get("estimated_runtime_hms"),
            "estimated_runtime_source": estimate.get("estimated_runtime_source"),
            "estimated_runtime_confidence": estimate.get("estimated_runtime_confidence"),
            "runtime_estimate_basis": estimate.get("runtime_estimate_basis", {}),
            "throughput": throughput,
        }
        execution_profile = context.get("execution_profile")
        if execution_profile is None:
            basis = estimate.get("runtime_estimate_basis", {})
            if isinstance(basis, Mapping):
                execution_profile = basis.get("execution_profile")
        if execution_profile is not None:
            payload["execution_profile"] = json_safe_value(execution_profile)
        if context.get("acceleration") is not None:
            payload["acceleration"] = json_safe_value(context["acceleration"])
        if context.get("stage_timing") is not None:
            payload["stage_timing"] = json_safe_value(context["stage_timing"])
        if context.get("error"):
            payload["error"] = context["error"]
        if context.get("log_path"):
            payload["log_path"] = context["log_path"]
        write_json(output_path / "run_timing.json", payload)
    finally:
        stop_run_log_capture(context)


def maybe_count_images(path: Optional[Path]) -> Optional[int]:
    """Count image files under a local directory, returning None when unavailable."""
    if path is None or not path.exists():
        return None
    if path.is_file():
        return 1 if path.suffix.lower() in IMAGE_EXTENSIONS else 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            total += 1
    return total


def dataset_split_dir(dataset_dir: Path, split: str) -> Optional[Path]:
    """Return a likely split image directory for Roboflow COCO/YOLO layouts."""
    candidates = [
        dataset_dir / split,
        dataset_dir / split / "images",
        dataset_dir / ("valid" if split == "val" else split),
        dataset_dir / ("valid" if split == "val" else split) / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_dataset_dir_value(config: Mapping[str, Any]) -> str:
    """Return the configured dataset root from train or dataset sections."""
    dataset = config.get("dataset", {})
    train = config.get("train", {})
    return str(train.get("dataset_dir") or dataset.get("dataset_dir") or "").strip()


def config_path_bases(source_config: Optional[Path] = None) -> List[Path]:
    """Return common bases for resolving local config paths."""
    bases = [Path.cwd(), PROJECT_DIR, REPO_ROOT]
    if source_config is not None:
        bases.insert(0, source_config.parent)
    return bases


def resolve_existing_path(value: Any, bases: Sequence[Path], field_name: str) -> Path:
    """Resolve an existing local path or raise a helpful error."""
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required.")
    resolved = resolve_existing_or_raw(value, bases)
    path = Path(str(resolved)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path.resolve()


def resolve_optional_existing_path(value: Any, bases: Sequence[Path], field_name: str) -> Optional[Path]:
    """Resolve an optional existing local path."""
    if value is None or str(value).strip() == "":
        return None
    return resolve_existing_path(value, bases, field_name)


def resolve_cache_root(value: Any, source_config: Optional[Path] = None) -> Path:
    """Resolve dataset cache root. Relative values default inside this trainer project."""
    text = str(value or "dataset_cache").strip()
    if not text:
        text = "dataset_cache"
    path = Path(text).expanduser()
    if is_abs_any_os(text):
        return path.resolve()
    return (PROJECT_DIR / path).resolve()


def find_dataset_yaml(config: Mapping[str, Any], source_config: Optional[Path] = None) -> Optional[Path]:
    """Find an Ultralytics-style dataset YAML from config or dataset_dir."""
    dataset = config.get("dataset", {})
    bases = config_path_bases(source_config)

    data_yaml = dataset.get("data_yaml") or dataset.get("dataset_yaml")
    if data_yaml:
        return resolve_existing_path(data_yaml, bases, "dataset.data_yaml")

    dataset_dir_value = get_dataset_dir_value(config)
    if not dataset_dir_value:
        return None
    dataset_dir = resolve_existing_path(dataset_dir_value, bases, "dataset.dataset_dir/train.dataset_dir")
    for name in DATA_YAML_NAMES:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate.resolve()
    return None


def normalize_source_format(value: Any) -> str:
    """Normalize dataset.source_format with a few human-friendly aliases."""
    text = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "coco": "coco_json",
        "cocojson": "coco_json",
        "labelme": "labelme_json",
        "labelmejson": "labelme_json",
        "pascal": "pascal_voc",
        "voc": "pascal_voc",
        "yolo": "ultralytics_yolo",
        "ultralytics": "ultralytics_yolo",
        "roboflow": "rfdetr",
    }
    return aliases.get(text, text)


def is_rfdetr_coco_layout(dataset_dir: Path) -> bool:
    """Return True when a dataset already looks like RF-DETR/Roboflow COCO layout."""
    return (
        (dataset_dir / "train" / "_annotations.coco.json").exists()
        and (dataset_dir / "valid" / "_annotations.coco.json").exists()
    )


def is_ultralytics_yolo_layout(dataset_dir: Path, data_yaml: Optional[Path]) -> bool:
    """Return True when a dataset looks like an Ultralytics YOLO directory layout."""
    if data_yaml is None or not data_yaml.exists():
        return False
    return (
        (dataset_dir / "images" / "train").exists()
        and (dataset_dir / "labels" / "train").exists()
        and ((dataset_dir / "images" / "val").exists() or (dataset_dir / "images" / "valid").exists())
    )


def yaml_looks_like_yolo_dataset(data_yaml: Optional[Path]) -> bool:
    """Return True when a YAML has the expected Ultralytics YOLO dataset fields."""
    if data_yaml is None or not data_yaml.exists():
        return False
    with contextlib.suppress(Exception):
        data = load_yaml(data_yaml)
        return bool(data.get("names") is not None and (data.get("train") is not None or data.get("val") is not None))
    return False


def is_rfdetr_yolo_layout(dataset_dir: Path) -> bool:
    """Return True when a dataset already looks like RF-DETR/Roboflow YOLO layout."""
    has_yaml = any((dataset_dir / name).exists() for name in ("data.yaml", "data.yml"))
    return (
        has_yaml
        and (dataset_dir / "train" / "images").exists()
        and (dataset_dir / "train" / "labels").exists()
        and (dataset_dir / "valid" / "images").exists()
        and (dataset_dir / "valid" / "labels").exists()
    )


def is_coco_json_file(path: Path) -> bool:
    """Return True when a JSON file looks like COCO annotations."""
    if not path.exists() or path.suffix.lower() != ".json":
        return False
    with contextlib.suppress(Exception):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return all(key in data for key in ("images", "annotations", "categories"))
    return False


def is_labelme_json_file(path: Path) -> bool:
    """Return True when a JSON file looks like LabelMe annotations."""
    if not path.exists() or path.suffix.lower() != ".json":
        return False
    with contextlib.suppress(Exception):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return isinstance(data.get("shapes"), list) and "imagePath" in data
    return False


def looks_like_pascal_voc_layout(dataset_dir: Path) -> bool:
    """Return True when a directory contains Pascal VOC XML annotations."""
    candidates = [dataset_dir / "Annotations", dataset_dir / "annotations"]
    return any(path.exists() and any(path.glob("*.xml")) for path in candidates)


def looks_like_dota_layout(dataset_dir: Path) -> bool:
    """Return True when a directory contains common DOTA image/label folders."""
    label_roots = [dataset_dir / "labelTxt", dataset_dir / "labels"]
    image_roots = [dataset_dir / "images", dataset_dir]
    has_labels = any(root.exists() and any(root.rglob("*.txt")) for root in label_roots)
    has_images = any(root.exists() and any(file.suffix.lower() in IMAGE_EXTENSIONS for file in root.rglob("*")) for root in image_roots)
    return has_labels and has_images


def looks_like_labelme_layout(dataset_dir: Path) -> bool:
    """Return True when a directory contains LabelMe JSON files."""
    for path in dataset_dir.rglob("*.json"):
        if is_labelme_json_file(path):
            return True
    return False


def find_coco_jsons(config: Mapping[str, Any], dataset_dir: Optional[Path], source_config: Optional[Path]) -> Dict[str, Path]:
    """Find COCO JSON annotation files from explicit config or common directories."""
    dataset = config.get("dataset", {})
    bases = config_path_bases(source_config)
    explicit = resolve_optional_existing_path(dataset.get("coco_json"), bases, "dataset.coco_json")
    if explicit is not None:
        return {split_from_path_or_name(explicit) or "all": explicit}
    if dataset_dir is None:
        return {}

    candidates: List[Path] = []
    for root in (dataset_dir / "annotations", dataset_dir):
        if root.exists():
            candidates.extend(sorted(root.glob("*.json")))
    result: Dict[str, Path] = {}
    for path in candidates:
        if not is_coco_json_file(path):
            continue
        split = split_from_path_or_name(path) or "all"
        result.setdefault(split, path.resolve())
    return result


def resolve_dataset_source_format(
    config: Mapping[str, Any],
    dataset_dir: Optional[Path],
    data_yaml: Optional[Path],
    source_config: Optional[Path] = None,
) -> str:
    """Resolve dataset.source_format, with auto-detection for supported adapters."""
    dataset = config.get("dataset", {})
    requested = normalize_source_format(dataset.get("source_format", "auto"))
    if requested not in DATASET_SOURCE_FORMATS:
        raise ValueError(
            f"Unsupported dataset.source_format={requested!r}. "
            f"Options: {', '.join(sorted(DATASET_SOURCE_FORMATS))}."
        )
    if requested != "auto":
        return requested
    if dataset_dir is not None and (is_rfdetr_coco_layout(dataset_dir) or is_rfdetr_yolo_layout(dataset_dir)):
        return "rfdetr"
    if find_coco_jsons(config, dataset_dir, source_config):
        return "coco_json"
    if dataset_dir is not None and looks_like_pascal_voc_layout(dataset_dir):
        return "pascal_voc"
    if dataset_dir is not None and is_ultralytics_yolo_layout(dataset_dir, data_yaml):
        return "ultralytics_yolo"
    if yaml_looks_like_yolo_dataset(data_yaml):
        return "ultralytics_yolo"
    if dataset_dir is not None and looks_like_dota_layout(dataset_dir):
        return "dota"
    if dataset_dir is not None and looks_like_labelme_layout(dataset_dir):
        return "labelme_json"
    return "rfdetr"


def resolve_path_from_yaml(value: Any, base_dir: Path, field_name: str) -> Path:
    """Resolve a dataset YAML path value relative to the YAML dataset base."""
    if isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} list paths are not supported yet; use a directory path.")
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required.")
    text = str(value).strip()
    if text.endswith(".txt"):
        raise ValueError(f"{field_name} text-file splits are not supported yet; use a directory path.")
    path = Path(text).expanduser()
    if is_abs_any_os(text):
        return path
    return (base_dir / text).resolve()


def resolve_yaml_path_values(value: Any, base_dir: Path, field_name: str) -> List[Path]:
    """Resolve YAML split values, including list and .txt image-list formats."""
    if value is None or str(value).strip() == "":
        return []
    if isinstance(value, (list, tuple)):
        paths: List[Path] = []
        for index, item in enumerate(value):
            paths.extend(resolve_yaml_path_values(item, base_dir, f"{field_name}[{index}]"))
        return paths

    text = str(value).strip()
    path = Path(text).expanduser()
    resolved = path if is_abs_any_os(text) else (base_dir / path).resolve()
    if resolved.suffix.lower() == ".txt":
        output: List[Path] = []
        with resolved.open("r", encoding="utf-8") as file:
            for line in file:
                item = line.strip()
                if not item:
                    continue
                item_path = Path(item).expanduser()
                output.append(item_path if is_abs_any_os(item) else (resolved.parent / item_path).resolve())
        return output
    return [resolved]


def replace_path_part(path: Path, old: str, new: str) -> Optional[Path]:
    """Replace one path component case-insensitively."""
    parts = list(path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == old.lower():
            parts[index] = new
            return Path(*parts)
    return None


def infer_yolo_label_dir(image_dir: Path) -> Path:
    """Infer an Ultralytics YOLO labels directory from an images directory."""
    replaced = replace_path_part(image_dir, "images", "labels")
    if replaced is not None:
        return replaced
    if image_dir.name.lower() == "images":
        return image_dir.parent / "labels"
    return image_dir.parent.parent / "labels" / image_dir.name


def normalize_yolo_names(names: Any) -> Any:
    """Normalize YOLO class names for a generated data.yaml file."""
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return [str(item) for item in names]
    raise ValueError("Ultralytics dataset YAML must contain names as a list or mapping.")


def coerce_class_names(names: Any, default: Optional[Sequence[str]] = None) -> List[str]:
    """Coerce list/dict class names into an ordered list."""
    if names is None:
        if default is None:
            return []
        return [str(item) for item in default]
    if isinstance(names, Mapping):
        def key_fn(item: Tuple[Any, Any]) -> Tuple[int, Any]:
            key = item[0]
            try:
                return (0, int(key))
            except (TypeError, ValueError):
                return (1, str(key))

        return [str(value) for _, value in sorted(names.items(), key=key_fn)]
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return [str(item) for item in names]
    raise ValueError("Class names must be a list or mapping.")


def normalize_split_name(value: Any) -> str:
    """Normalize split labels to RF-DETR/Roboflow split names."""
    text = str(value or "all").strip().lower().replace("_", "-")
    mapping = {
        "all": "all",
        "unsplit": "all",
        "val": "valid",
        "validation": "valid",
        "valid": "valid",
        "train": "train",
        "training": "train",
        "test": "test",
        "testing": "test",
        "testoriginal": "test-original",
        "test-original": "test-original",
    }
    return mapping.get(text, text or "all")


def split_from_path_or_name(path: Path) -> Optional[str]:
    """Infer split name from a path's filename or directory components."""
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    for split in ("train", "valid", "val", "validation", "test"):
        if re.search(rf"(^|[^a-z]){re.escape(split)}([^a-z]|$)", name) or split in parts:
            return normalize_split_name(split)
    return None


def iter_image_files(path: Path) -> List[Path]:
    """Return image files under a file or directory path."""
    if path.is_file():
        return [path.absolute()] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    if not path.exists():
        return []
    return sorted(file.absolute() for file in path.rglob("*") if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)


def read_image_size(path: Path) -> Tuple[int, int]:
    """Read image width and height without mutating the source file."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to convert datasets because image sizes are needed.") from exc

    with Image.open(path) as image:
        width, height = image.size
        if image.format == "JPEG":
            with contextlib.suppress(Exception):
                rotation = image.getexif().get(274, None)
                if rotation in {6, 8}:
                    width, height = height, width
        return int(width), int(height)


def clamp_coco_bbox(x: float, y: float, width: float, height: float, image_width: int, image_height: int) -> Optional[List[float]]:
    """Clamp a COCO xywh bbox to image boundaries and drop degenerate boxes."""
    x1 = max(0.0, min(float(image_width), float(x)))
    y1 = max(0.0, min(float(image_height), float(y)))
    x2 = max(0.0, min(float(image_width), float(x) + float(width)))
    y2 = max(0.0, min(float(image_height), float(y) + float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 6), round(y1, 6), round(x2 - x1, 6), round(y2 - y1, 6)]


def bbox_from_points(points: Sequence[Sequence[float]], image_width: int, image_height: int) -> Optional[List[float]]:
    """Build an axis-aligned COCO bbox from polygon points."""
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return clamp_coco_bbox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), image_width, image_height)


def make_record(source_image: Path, annotations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Create a source image record used by the cache materializer."""
    width, height = read_image_size(source_image)
    return {
        "source_image": source_image.resolve(),
        "width": width,
        "height": height,
        "annotations": annotations or [],
    }


def add_record(records_by_split: Dict[str, List[Dict[str, Any]]], split: str, record: Dict[str, Any]) -> None:
    """Append a record to a normalized split bucket."""
    records_by_split.setdefault(normalize_split_name(split), []).append(record)


def parse_split_ratio(value: Any) -> Tuple[int, int, int]:
    """Parse dataset.split_ratio into train/valid/test integer weights."""
    if value in (None, ""):
        return (8, 1, 1)
    if isinstance(value, str):
        parsed = parse_scalar(value)
    else:
        parsed = value
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)) or len(parsed) != 3:
        raise ValueError("dataset.split_ratio must contain exactly three numbers: train, valid, test.")
    ratio = tuple(int(item) for item in parsed)
    if any(item < 0 for item in ratio) or sum(ratio) <= 0:
        raise ValueError("dataset.split_ratio values must be non-negative and sum to a positive number.")
    return ratio  # type: ignore[return-value]


def parse_split_ratio_arg(value: Any) -> List[int]:
    """Parse CLI split ratio into a YAML-friendly list."""
    return list(parse_split_ratio(value))


def deterministic_unsplit(records: Sequence[Dict[str, Any]], ratio: Tuple[int, int, int], seed: Any = 0) -> Dict[str, List[Dict[str, Any]]]:
    """Split unsplit records deterministically according to train/valid/test weights."""
    positive_splits = sum(1 for item in ratio if item > 0)
    if len(records) < positive_splits:
        raise ValueError(
            f"Cannot split {len(records)} image(s) with dataset.split_ratio={list(ratio)}. "
            f"At least {positive_splits} images are required."
        )
    sorted_records = sorted(
        records,
        key=lambda record: hashlib.sha256(f"{seed}:{record['source_image']}".encode("utf-8")).hexdigest(),
    )
    total = len(sorted_records)
    ratio_sum = sum(ratio)
    counts = [max(1, int(math.floor(total * weight / ratio_sum))) if weight > 0 else 0 for weight in ratio]
    while sum(counts) > total:
        largest = max((count, index) for index, count in enumerate(counts) if count > 1)[1]
        counts[largest] -= 1
    remainders = [total * weight / ratio_sum - math.floor(total * weight / ratio_sum) for weight in ratio]
    while sum(counts) < total:
        index = max(range(3), key=lambda item: (remainders[item], ratio[item]))
        counts[index] += 1
        remainders[index] = 0

    train_end = counts[0]
    valid_end = train_end + counts[1]
    return {
        "train": list(sorted_records[:train_end]),
        "valid": list(sorted_records[train_end:valid_end]),
        "test": list(sorted_records[valid_end:]),
    }


def ensure_split_records(
    records_by_split: Mapping[str, List[Dict[str, Any]]],
    ratio: Tuple[int, int, int],
    seed: Any = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Preserve explicit train/valid splits or split unsplit data deterministically."""
    normalized: Dict[str, List[Dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    extra: Dict[str, List[Dict[str, Any]]] = {}
    unsplit: List[Dict[str, Any]] = []
    for split, records in records_by_split.items():
        normalized_split = normalize_split_name(split)
        if normalized_split in normalized:
            normalized[normalized_split].extend(records)
        elif normalized_split == "all":
            unsplit.extend(records)
        else:
            extra.setdefault(normalized_split, []).extend(records)

    if normalized["train"] and normalized["valid"]:
        result = {split: records for split, records in normalized.items() if records}
        if unsplit:
            for split, records in deterministic_unsplit(unsplit, ratio, seed).items():
                result.setdefault(split, []).extend(records)
        result.update({split: records for split, records in extra.items() if records})
        return result

    all_records = unsplit + normalized["train"] + normalized["valid"] + normalized["test"]
    if not all_records:
        raise ValueError("No images were found in the dataset source.")
    result = deterministic_unsplit(all_records, ratio, seed)
    result.update({split: records for split, records in extra.items() if records})
    return result


def dataset_limit_config(config: Mapping[str, Any]) -> Dict[str, Optional[int]]:
    """Return per-split dataset image limits from dataset.* max fields."""
    dataset = config.get("dataset", {})
    default_limit = parse_limit_value(dataset.get("max_images"), "dataset.max_images")

    def split_limit(key: str, field_name: str) -> Optional[int]:
        if key not in dataset or dataset.get(key) is None:
            return default_limit
        return parse_limit_value(dataset.get(key), field_name)

    train_limit = split_limit("max_train_images", "dataset.max_train_images")
    valid_limit = split_limit("max_val_images", "dataset.max_val_images")
    test_limit = split_limit("max_test_images", "dataset.max_test_images")
    return {
        "train": train_limit,
        "valid": valid_limit,
        "val": valid_limit,
        "test": test_limit,
        "test-original": test_limit,
        "test_original": test_limit,
    }


def has_dataset_limits(config: Mapping[str, Any]) -> bool:
    """Return True when dataset config limits at least one split."""
    return any(limit is not None for limit in dataset_limit_config(config).values())


def limit_records_by_split(
    records_by_split: Mapping[str, List[Dict[str, Any]]],
    config: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Apply dataset first-N image limits after split assignment."""
    limits = dataset_limit_config(config)
    limited: Dict[str, List[Dict[str, Any]]] = {}
    for split, records in records_by_split.items():
        normalized = normalize_split_name(split)
        limit = limits.get(normalized)
        limited[split] = list(records[:limit] if limit is not None else records)
    return limited


def assign_cache_file_names(records_by_split: Mapping[str, List[Dict[str, Any]]]) -> None:
    """Assign unique flat file names for each cache split directory."""
    for records in records_by_split.values():
        used: set[str] = set()
        for record in records:
            source = Path(record["source_image"])
            stem = sanitize_name(source.stem)
            suffix = source.suffix.lower() or ".jpg"
            name = f"{stem}{suffix}"
            if name.lower() in used:
                digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:10]
                name = f"{stem}_{digest}{suffix}"
            used.add(name.lower())
            record["file_name"] = name


def finalize_categories(records_by_split: Mapping[str, List[Dict[str, Any]]], class_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Create COCO categories and attach category_id to every annotation."""
    names: List[str] = []
    seen: set[str] = set()
    for name in class_names:
        text = str(name)
        if text not in seen:
            names.append(text)
            seen.add(text)
    for records in records_by_split.values():
        for record in records:
            for annotation in record.get("annotations", []):
                name = str(annotation.get("category_name", "object"))
                if name not in seen:
                    names.append(name)
                    seen.add(name)
    if not names:
        names = ["object"]
    name_to_id = {name: index + 1 for index, name in enumerate(names)}
    for records in records_by_split.values():
        for record in records:
            for annotation in record.get("annotations", []):
                annotation["category_id"] = name_to_id[str(annotation.get("category_name", "object"))]
    return [{"id": index + 1, "name": name, "supercategory": "object"} for index, name in enumerate(names)]


def directory_file_stats(paths: Iterable[Path]) -> Tuple[int, int]:
    """Count files and bytes under unique directories."""
    total_files = 0
    total_bytes = 0
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_files += 1
                with contextlib.suppress(OSError):
                    total_bytes += file_path.stat().st_size
    return total_files, total_bytes


def unique_source_images(records_by_split: Mapping[str, List[Dict[str, Any]]]) -> List[Path]:
    """Return unique source image paths from split records."""
    paths: Dict[str, Path] = {}
    for records in records_by_split.values():
        for record in records:
            path = Path(record["source_image"]).resolve()
            paths[str(path).lower()] = path
    return list(paths.values())


def file_stats_digest(paths: Iterable[Path]) -> Tuple[str, int]:
    """Hash file paths, sizes, and mtimes without reading file contents."""
    entries: List[Dict[str, Any]] = []
    for path in sorted({str(Path(item).resolve()) for item in paths}):
        file_path = Path(path)
        stat = file_path.stat()
        entries.append(
            {
                "path": str(file_path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(entries)


def image_bytes(paths: Iterable[Path]) -> int:
    """Sum image file sizes for copy-mode estimates."""
    total = 0
    for path in paths:
        with contextlib.suppress(OSError):
            total += path.stat().st_size
    return total


def infer_yolo_label_file(image_file: Path) -> Path:
    """Infer a YOLO label path from an image path."""
    replaced_parent = replace_path_part(image_file.parent, "images", "labels")
    if replaced_parent is not None:
        return (replaced_parent / image_file.name).with_suffix(".txt")
    return (image_file.parent / "labels" / image_file.name).with_suffix(".txt")


def yolo_annotation_from_values(
    values: Sequence[float],
    image_width: int,
    image_height: int,
) -> Optional[Tuple[List[float], Optional[List[List[float]]]]]:
    """Convert YOLO bbox/segment/OBB coordinates into COCO bbox and optional polygon."""
    if len(values) == 4:
        cx, cy, width, height = [float(item) for item in values]
        if max(abs(cx), abs(cy), abs(width), abs(height)) <= 1.5:
            x = (cx - width / 2.0) * image_width
            y = (cy - height / 2.0) * image_height
            bbox_width = width * image_width
            bbox_height = height * image_height
        else:
            x = cx - width / 2.0
            y = cy - height / 2.0
            bbox_width = width
            bbox_height = height
        bbox = clamp_coco_bbox(x, y, bbox_width, bbox_height, image_width, image_height)
        return (bbox, None) if bbox is not None else None

    if len(values) >= 6 and len(values) % 2 == 0:
        coords = [float(item) for item in values]
        normalized = max(abs(item) for item in coords) <= 1.5
        points = []
        for x, y in zip(coords[0::2], coords[1::2]):
            points.append([x * image_width if normalized else x, y * image_height if normalized else y])
        bbox = bbox_from_points(points, image_width, image_height)
        return (bbox, points) if bbox is not None else None
    return None


def parse_yolo_label_file(label_file: Path, class_names: Sequence[str], image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """Parse one YOLO label file into cache annotations."""
    annotations: List[Dict[str, Any]] = []
    if not label_file.exists():
        return annotations
    with label_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            with contextlib.suppress(ValueError, IndexError):
                class_index = int(float(parts[0]))
                converted = yolo_annotation_from_values([float(item) for item in parts[1:]], image_width, image_height)
                if converted is None:
                    continue
                bbox, polygon = converted
                name = class_names[class_index] if 0 <= class_index < len(class_names) else str(class_index)
                annotation = {
                    "category_name": name,
                    "bbox": bbox,
                    "source": f"{label_file}:{line_number}",
                }
                if polygon and len(polygon) >= 3:
                    annotation["segmentation"] = [[coord for point in polygon for coord in point]]
                annotations.append(annotation)
    return annotations


def build_ultralytics_yolo_source(
    config: Mapping[str, Any],
    source_config: Optional[Path],
) -> Dict[str, Any]:
    """Read an Ultralytics YOLO dataset into generic split records."""
    data_yaml = find_dataset_yaml(config, source_config)
    if data_yaml is None:
        raise ValueError("dataset.data_yaml or a dataset YAML inside dataset_dir/train.dataset_dir is required.")

    data = load_yaml(data_yaml)
    yaml_base = data_yaml.parent
    warnings: List[str] = []
    configured_dataset_dir: Optional[Path] = None
    dataset_dir_value = get_dataset_dir_value(config)
    if dataset_dir_value:
        with contextlib.suppress(Exception):
            configured_dataset_dir = resolve_existing_path(
                dataset_dir_value,
                config_path_bases(source_config),
                "dataset.dataset_dir/train.dataset_dir",
            )
    dataset_base_raw = data.get("path", "")
    if dataset_base_raw:
        dataset_base_candidate = resolve_path_from_yaml(dataset_base_raw, yaml_base, "path")
        if dataset_base_candidate.exists():
            dataset_base = dataset_base_candidate
        else:
            dataset_base = configured_dataset_dir or yaml_base
            warnings.append(
                f"YOLO dataset YAML path={dataset_base_raw!r} does not exist on this host; "
                f"using {dataset_base} as dataset base."
            )
    else:
        dataset_base = yaml_base

    class_names = coerce_class_names(data.get("names"))
    split_values = {"train": data.get("train"), "valid": data.get("val", data.get("valid")), "test": data.get("test")}
    for key in ("test_original", "test-original", "testoriginal"):
        if data.get(key) is not None:
            split_values["test-original"] = data.get(key)
            break
    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    annotation_files: List[Path] = [data_yaml]

    for split, split_value in split_values.items():
        for split_path in resolve_yaml_path_values(split_value, dataset_base, split):
            image_files = iter_image_files(split_path)
            if not image_files:
                warnings.append(f"No images found for YOLO {split} split at {split_path}.")
            for image_file in image_files:
                width, height = read_image_size(image_file)
                label_file = infer_yolo_label_file(image_file)
                if label_file.exists():
                    annotation_files.append(label_file)
                annotations = parse_yolo_label_file(label_file, class_names, width, height)
                add_record(
                    records_by_split,
                    split,
                    {
                        "source_image": image_file.resolve(),
                        "width": width,
                        "height": height,
                        "annotations": annotations,
                    },
                )

    return {
        "source_format": "ultralytics_yolo",
        "source_name": data_yaml.stem,
        "data_yaml": data_yaml,
        "dataset_base": dataset_base,
        "records_by_split": records_by_split,
        "class_names": class_names,
        "annotation_files": annotation_files,
        "warnings": warnings,
    }


def resolve_coco_image_path(
    file_name: str,
    dataset_dir: Optional[Path],
    image_dir: Optional[Path],
    annotation_file: Path,
    split: str,
) -> Path:
    """Resolve an image referenced by a COCO JSON annotation."""
    raw = Path(str(file_name)).expanduser()
    if is_abs_any_os(str(file_name)) and raw.exists():
        return raw.resolve()
    candidates: List[Path] = []
    if image_dir is not None:
        candidates.append(image_dir / raw)
        candidates.append(image_dir / raw.name)
    if dataset_dir is not None:
        candidates.extend(
            [
                dataset_dir / raw,
                dataset_dir / raw.name,
                dataset_dir / split / raw,
                dataset_dir / split / raw.name,
                dataset_dir / ("val" if split == "valid" else split) / raw,
                dataset_dir / ("val" if split == "valid" else split) / raw.name,
                dataset_dir / "images" / split / raw,
                dataset_dir / "images" / split / raw.name,
                dataset_dir / "images" / ("val" if split == "valid" else split) / raw,
                dataset_dir / "images" / ("val" if split == "valid" else split) / raw.name,
                dataset_dir / f"{'val' if split == 'valid' else split}2017" / raw,
                dataset_dir / f"{'val' if split == 'valid' else split}2017" / raw.name,
            ]
        )
    candidates.extend([annotation_file.parent / raw, annotation_file.parent.parent / raw, annotation_file.parent / raw.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    search_root = image_dir or dataset_dir
    if search_root and search_root.exists():
        matches = list(search_root.rglob(raw.name))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(f"Could not resolve COCO image {file_name!r} referenced by {annotation_file}.")


def build_coco_json_source(config: Mapping[str, Any], dataset_dir: Optional[Path], source_config: Optional[Path]) -> Dict[str, Any]:
    """Read COCO JSON annotations into generic split records."""
    dataset = config.get("dataset", {})
    image_dir = resolve_optional_existing_path(dataset.get("image_dir"), config_path_bases(source_config), "dataset.image_dir")
    annotation_files = find_coco_jsons(config, dataset_dir, source_config)
    if not annotation_files:
        raise ValueError("No COCO JSON annotations found. Set dataset.coco_json or dataset.dataset_dir.")

    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    class_names: List[str] = []
    class_seen: set[str] = set()
    warnings: List[str] = []
    source_files: List[Path] = list(annotation_files.values())

    for raw_split, annotation_file in annotation_files.items():
        split = normalize_split_name(raw_split)
        with annotation_file.open("r", encoding="utf-8") as file:
            coco = json.load(file)
        source_categories = {int(cat["id"]): str(cat["name"]) for cat in coco.get("categories", [])}
        for _, name in sorted(source_categories.items()):
            if name not in class_seen:
                class_names.append(name)
                class_seen.add(name)
        annotations_by_image: Dict[int, List[Mapping[str, Any]]] = {}
        for annotation in coco.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        for image in coco.get("images", []):
            image_id = int(image["id"])
            source_image = resolve_coco_image_path(str(image["file_name"]), dataset_dir, image_dir, annotation_file, split)
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if width <= 0 or height <= 0:
                width, height = read_image_size(source_image)
            annotations: List[Dict[str, Any]] = []
            for annotation in annotations_by_image.get(image_id, []):
                bbox_raw = annotation.get("bbox")
                if not bbox_raw or len(bbox_raw) != 4:
                    continue
                bbox = clamp_coco_bbox(float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]), width, height)
                if bbox is None:
                    continue
                name = source_categories.get(int(annotation.get("category_id", 0)), str(annotation.get("category_id", "object")))
                converted = {
                    "category_name": name,
                    "bbox": bbox,
                    "iscrowd": int(annotation.get("iscrowd", 0) or 0),
                    "source": f"{annotation_file}:annotation:{annotation.get('id')}",
                }
                if annotation.get("segmentation"):
                    converted["segmentation"] = annotation["segmentation"]
                annotations.append(converted)
            add_record(
                records_by_split,
                split,
                {"source_image": source_image, "width": width, "height": height, "annotations": annotations},
            )
            source_files.append(source_image)
    return {
        "source_format": "coco_json",
        "source_name": next(iter(annotation_files.values())).stem,
        "records_by_split": records_by_split,
        "class_names": class_names,
        "annotation_files": source_files,
        "warnings": warnings,
    }


def find_image_by_stem(stem: str, roots: Sequence[Path], fallback_name: Optional[str] = None) -> Path:
    """Find an image by stem or filename under candidate roots."""
    names = [fallback_name] if fallback_name else []
    names.extend(f"{stem}{suffix}" for suffix in sorted(IMAGE_EXTENSIONS))
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate.resolve()
        matches = [path for path in root.rglob("*") if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS]
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(f"Could not find image for annotation stem {stem!r}.")


def build_pascal_voc_source(config: Mapping[str, Any], dataset_dir: Optional[Path]) -> Dict[str, Any]:
    """Read Pascal VOC XML annotations into generic split records."""
    if dataset_dir is None:
        raise ValueError("dataset.dataset_dir is required for Pascal VOC conversion.")
    import xml.etree.ElementTree as ET

    annotation_dir = next((path for path in (dataset_dir / "Annotations", dataset_dir / "annotations") if path.exists()), None)
    if annotation_dir is None:
        raise FileNotFoundError(f"Pascal VOC annotations directory not found under {dataset_dir}.")
    image_roots = [path for path in (dataset_dir / "JPEGImages", dataset_dir / "images", dataset_dir) if path.exists()]
    split_dir = dataset_dir / "ImageSets" / "Main"
    split_ids: Dict[str, List[str]] = {}
    for split_name, file_name in {"train": "train.txt", "valid": "val.txt", "test": "test.txt"}.items():
        path = split_dir / file_name
        if path.exists():
            split_ids[split_name] = [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    xml_by_stem = {path.stem: path.resolve() for path in sorted(annotation_dir.glob("*.xml"))}
    split_to_stems = split_ids or {"all": sorted(xml_by_stem)}
    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    class_names: List[str] = []
    seen: set[str] = set()
    annotation_files: List[Path] = []

    for split, stems in split_to_stems.items():
        for stem in stems:
            xml_file = xml_by_stem.get(stem)
            if xml_file is None:
                continue
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename = root.findtext("filename") or f"{stem}.jpg"
            source_image = find_image_by_stem(stem, image_roots, filename)
            size = root.find("size")
            if size is not None and size.findtext("width") and size.findtext("height"):
                width, height = int(float(size.findtext("width"))), int(float(size.findtext("height")))
            else:
                width, height = read_image_size(source_image)
            annotations: List[Dict[str, Any]] = []
            for obj in root.findall("object"):
                if obj.findtext("difficult", "0") == "1":
                    continue
                name = str(obj.findtext("name") or "object")
                if name not in seen:
                    class_names.append(name)
                    seen.add(name)
                box = obj.find("bndbox")
                if box is None:
                    continue
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
                bbox = clamp_coco_bbox(xmin, ymin, xmax - xmin, ymax - ymin, width, height)
                if bbox is not None:
                    annotations.append({"category_name": name, "bbox": bbox, "source": str(xml_file)})
            add_record(records_by_split, split, {"source_image": source_image, "width": width, "height": height, "annotations": annotations})
            annotation_files.append(xml_file)
            annotation_files.append(source_image)
    return {
        "source_format": "pascal_voc",
        "source_name": dataset_dir.name,
        "records_by_split": records_by_split,
        "class_names": class_names,
        "annotation_files": annotation_files,
        "warnings": [],
    }


def dota_label_dirs(dataset_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Find common DOTA split image and label directories."""
    result: Dict[str, Tuple[Path, Path]] = {}
    split_aliases = {"train": ("train",), "valid": ("valid", "val"), "test": ("test",)}
    for split, aliases in split_aliases.items():
        image_dir = next(
            (
                path
                for alias in aliases
                for path in (dataset_dir / "images" / alias, dataset_dir / alias / "images", dataset_dir / alias)
                if path.exists()
            ),
            None,
        )
        label_dir = next(
            (
                path
                for alias in aliases
                for path in (
                    dataset_dir / "labels" / alias,
                    dataset_dir / "labels" / f"{alias}_original",
                    dataset_dir / "labelTxt" / alias,
                    dataset_dir / alias / "labelTxt",
                    dataset_dir / alias / "labels",
                )
                if path.exists()
            ),
            None,
        )
        if image_dir is not None and label_dir is not None:
            result[split] = (image_dir, label_dir)
    if result:
        return result
    image_dir = next((path for path in (dataset_dir / "images", dataset_dir) if path.exists()), None)
    label_dir = next((path for path in (dataset_dir / "labelTxt", dataset_dir / "labels") if path.exists()), None)
    if image_dir is not None and label_dir is not None:
        return {"all": (image_dir, label_dir)}
    return {}


def normalize_dota_class_name(value: str) -> str:
    """Normalize DOTA class tokens to display names."""
    return value.strip().replace("-", " ")


def build_dota_source(config: Mapping[str, Any], dataset_dir: Optional[Path], data_yaml: Optional[Path]) -> Dict[str, Any]:
    """Read native DOTA annotations and reduce OBB boxes to axis-aligned bboxes."""
    if dataset_dir is None:
        raise ValueError("dataset.dataset_dir is required for DOTA conversion.")
    names_from_yaml = coerce_class_names(load_yaml(data_yaml).get("names"), DOTA_CLASS_NAMES) if data_yaml else list(DOTA_CLASS_NAMES)
    split_dirs = dota_label_dirs(dataset_dir)
    if not split_dirs:
        raise FileNotFoundError(f"DOTA images/labels were not found under {dataset_dir}.")

    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    class_names = list(names_from_yaml)
    seen = set(class_names)
    annotation_files: List[Path] = []
    warnings = ["DOTA oriented boxes were converted to axis-aligned enclosing boxes for RF-DETR detection training."]
    for split, (image_dir, label_dir) in split_dirs.items():
        for label_file in sorted(label_dir.glob("*.txt")):
            source_image = find_image_by_stem(label_file.stem, [image_dir])
            width, height = read_image_size(source_image)
            annotations: List[Dict[str, Any]] = []
            with label_file.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    parts = line.strip().split()
                    if len(parts) < 9 or parts[0].lower() == "imagesource:" or parts[0].lower() == "gsd:":
                        continue
                    with contextlib.suppress(ValueError):
                        coords = [float(item) for item in parts[:8]]
                        points = [[coords[index], coords[index + 1]] for index in range(0, 8, 2)]
                        bbox = bbox_from_points(points, width, height)
                        if bbox is None:
                            continue
                        name = normalize_dota_class_name(parts[8])
                        if name not in seen:
                            class_names.append(name)
                            seen.add(name)
                        annotations.append(
                            {
                                "category_name": name,
                                "bbox": bbox,
                                "segmentation": [[coord for point in points for coord in point]],
                                "source": f"{label_file}:{line_number}",
                                "dota_obb_points": points,
                            }
                        )
            add_record(records_by_split, split, {"source_image": source_image, "width": width, "height": height, "annotations": annotations})
            annotation_files.extend([label_file, source_image])
    return {
        "source_format": "dota",
        "source_name": dataset_dir.name,
        "records_by_split": records_by_split,
        "class_names": class_names,
        "annotation_files": annotation_files,
        "warnings": warnings,
    }


def build_labelme_source(config: Mapping[str, Any], dataset_dir: Optional[Path]) -> Dict[str, Any]:
    """Read LabelMe JSON annotations into generic split records."""
    if dataset_dir is None:
        raise ValueError("dataset.dataset_dir is required for LabelMe JSON conversion.")
    json_files = [path.resolve() for path in sorted(dataset_dir.rglob("*.json")) if is_labelme_json_file(path)]
    if not json_files:
        raise FileNotFoundError(f"No LabelMe JSON files found under {dataset_dir}.")

    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    class_names: List[str] = []
    seen: set[str] = set()
    source_files: List[Path] = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        image_path = Path(str(data.get("imagePath", ""))).expanduser()
        source_image = image_path if is_abs_any_os(str(image_path)) else (json_file.parent / image_path)
        if not source_image.exists():
            source_image = find_image_by_stem(json_file.stem, [json_file.parent, dataset_dir], image_path.name if image_path.name else None)
        source_image = source_image.resolve()
        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            width, height = read_image_size(source_image)
        annotations: List[Dict[str, Any]] = []
        for index, shape in enumerate(data.get("shapes", []), start=1):
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            name = str(shape.get("label") or "object")
            if name not in seen:
                class_names.append(name)
                seen.add(name)
            bbox = bbox_from_points(points, width, height)
            if bbox is None:
                continue
            annotation = {
                "category_name": name,
                "bbox": bbox,
                "source": f"{json_file}:shape:{index}",
            }
            if len(points) >= 3:
                annotation["segmentation"] = [[float(coord) for point in points for coord in point]]
            annotations.append(annotation)
        split = split_from_path_or_name(json_file) or "all"
        add_record(records_by_split, split, {"source_image": source_image, "width": width, "height": height, "annotations": annotations})
        source_files.extend([json_file, source_image])
    return {
        "source_format": "labelme_json",
        "source_name": dataset_dir.name,
        "records_by_split": records_by_split,
        "class_names": class_names,
        "annotation_files": source_files,
        "warnings": [],
    }


def build_cache_source(
    config: Mapping[str, Any],
    source_format: str,
    dataset_dir: Optional[Path],
    data_yaml: Optional[Path],
    source_config: Optional[Path],
) -> Dict[str, Any]:
    """Read a supported source dataset into generic split records."""
    if source_format == "ultralytics_yolo":
        return build_ultralytics_yolo_source(config, source_config)
    if source_format == "coco_json":
        return build_coco_json_source(config, dataset_dir, source_config)
    if source_format == "pascal_voc":
        return build_pascal_voc_source(config, dataset_dir)
    if source_format == "dota":
        return build_dota_source(config, dataset_dir, data_yaml)
    if source_format == "labelme_json":
        return build_labelme_source(config, dataset_dir)
    raise ValueError(f"No cache converter is available for dataset.source_format={source_format!r}.")


def build_source_fingerprint(
    source_format: str,
    source: Mapping[str, Any],
    records_by_split: Mapping[str, List[Dict[str, Any]]],
    categories: Sequence[Mapping[str, Any]],
    split_ratio: Tuple[int, int, int],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a stable fingerprint from converter version, files, classes, and split policy."""
    files = list(source.get("annotation_files", [])) + unique_source_images(records_by_split)
    files_digest, file_count = file_stats_digest(Path(path) for path in files)
    payload = {
        "converter_version": CONVERTER_VERSION,
        "source_format": source_format,
        "source_name": source.get("source_name"),
        "split_ratio": list(split_ratio),
        "split_seed": config.get("dataset", {}).get("split_seed", 0),
        "dataset_limits": dataset_limit_config(config),
        "categories": list(categories),
        "files_digest": files_digest,
        "file_count": file_count,
    }
    payload["hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return payload


def normalize_link_mode(value: Any) -> str:
    """Validate dataset.link_mode for cache images."""
    mode = str(value or "auto").strip().lower()
    if mode not in {"auto", "hardlink", "symlink", "junction", "copy"}:
        raise ValueError("dataset.link_mode must be one of: auto, hardlink, symlink, junction, copy.")
    return mode


def build_cache_dataset_plan(
    config: Mapping[str, Any],
    source_format: str,
    dataset_dir: Optional[Path],
    data_yaml: Optional[Path],
    source_config: Optional[Path],
) -> Dict[str, Any]:
    """Build a no-write plan for converting a source dataset into RF-DETR cache."""
    dataset = config.get("dataset", {})
    source = build_cache_source(config, source_format, dataset_dir, data_yaml, source_config)
    split_ratio = parse_split_ratio(dataset.get("split_ratio", [8, 1, 1]))
    records_by_split = ensure_split_records(
        source["records_by_split"],
        split_ratio,
        seed=dataset.get("split_seed", 0),
    )
    records_by_split = limit_records_by_split(records_by_split, config)
    assign_cache_file_names(records_by_split)
    categories = finalize_categories(records_by_split, source.get("class_names", []))
    fingerprint = build_source_fingerprint(source_format, source, records_by_split, categories, split_ratio, config)
    cache_root = resolve_cache_root(dataset.get("cache_root", "dataset_cache"), source_config)
    source_name = sanitize_name(str(source.get("source_name") or (dataset_dir.name if dataset_dir else source_format)))
    cache_dir = cache_root / f"{source_format}_{source_name}_{fingerprint['hash'][:16]}"
    link_mode = normalize_link_mode(dataset.get("link_mode", "auto"))
    images = unique_source_images(records_by_split)
    copy_bytes = image_bytes(images) if link_mode == "copy" else 0
    return {
        "source_format": source_format,
        "action": "prepare_cache",
        "cache_root": cache_root,
        "cache_dir": cache_dir,
        "fingerprint": fingerprint,
        "records_by_split": records_by_split,
        "categories": categories,
        "split_counts": {split: len(records) for split, records in records_by_split.items()},
        "class_names": [category["name"] for category in categories],
        "link_mode": link_mode,
        "copy_file_count": len(images) if link_mode == "copy" else 0,
        "copy_bytes": copy_bytes,
        "cache_file_count": len(images) + len(records_by_split) + 2,
        "refresh_cache": bool(dataset.get("refresh_cache", False)),
        "warnings": list(source.get("warnings", [])),
    }


def find_rfdetr_coco_split_files(dataset_dir: Path) -> Dict[str, Path]:
    """Find RF-DETR/Roboflow COCO split annotation files."""
    candidates = [
        ("train", "train"),
        ("valid", "valid"),
        ("val", "valid"),
        ("test", "test"),
        ("test-original", "test-original"),
        ("test_original", "test-original"),
    ]
    found: Dict[str, Path] = {}
    for folder_name, split in candidates:
        if split in found:
            continue
        annotation_file = dataset_dir / folder_name / "_annotations.coco.json"
        if annotation_file.exists():
            found[split] = annotation_file.resolve()
    return found


def build_rfdetr_coco_split_source(dataset_dir: Path) -> Dict[str, Any]:
    """Read an existing RF-DETR COCO split dataset into cache records."""
    annotation_files = find_rfdetr_coco_split_files(dataset_dir)
    if not annotation_files:
        raise ValueError(
            "Limiting an existing RF-DETR dataset requires split _annotations.coco.json files "
            "or a Roboflow/RF-DETR YOLO layout with data.yaml."
        )

    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    class_names: List[str] = []
    class_seen: set[str] = set()
    source_files: List[Path] = list(annotation_files.values())

    for split, annotation_file in annotation_files.items():
        split_dir = annotation_file.parent
        with annotation_file.open("r", encoding="utf-8") as file:
            coco = json.load(file)
        source_categories = {int(category["id"]): str(category["name"]) for category in coco.get("categories", [])}
        for _, name in sorted(source_categories.items()):
            if name not in class_seen:
                class_names.append(name)
                class_seen.add(name)

        annotations_by_image: Dict[int, List[Mapping[str, Any]]] = {}
        for annotation in coco.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

        images = sorted(coco.get("images", []), key=lambda image: str(image.get("file_name", "")))
        for image in images:
            image_id = int(image["id"])
            source_image = resolve_coco_image_path(str(image["file_name"]), dataset_dir, split_dir, annotation_file, split)
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if width <= 0 or height <= 0:
                width, height = read_image_size(source_image)
            annotations: List[Dict[str, Any]] = []
            for annotation in annotations_by_image.get(image_id, []):
                bbox_raw = annotation.get("bbox")
                if not bbox_raw or len(bbox_raw) != 4:
                    continue
                bbox = clamp_coco_bbox(float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]), width, height)
                if bbox is None:
                    continue
                category_id = int(annotation.get("category_id", 0))
                converted = {
                    "category_name": source_categories.get(category_id, str(category_id)),
                    "bbox": bbox,
                    "iscrowd": int(annotation.get("iscrowd", 0) or 0),
                    "source": f"{annotation_file}:annotation:{annotation.get('id')}",
                }
                if annotation.get("segmentation"):
                    converted["segmentation"] = annotation["segmentation"]
                annotations.append(converted)
            add_record(
                records_by_split,
                split,
                {"source_image": source_image, "width": width, "height": height, "annotations": annotations},
            )
            source_files.append(source_image)

    return {
        "source_format": "rfdetr_coco",
        "source_name": dataset_dir.name,
        "records_by_split": records_by_split,
        "class_names": class_names,
        "annotation_files": source_files,
        "warnings": [],
    }


def build_rfdetr_limited_dataset_plan(
    config: Mapping[str, Any],
    dataset_dir: Optional[Path],
    source_config: Optional[Path],
) -> Dict[str, Any]:
    """Build a cache plan that limits an existing RF-DETR-readable dataset."""
    if dataset_dir is None:
        raise ValueError("dataset.dataset_dir/train.dataset_dir is required when dataset limits are enabled.")
    dataset = config.get("dataset", {})
    source = build_ultralytics_yolo_source(config, source_config) if is_rfdetr_yolo_layout(dataset_dir) else build_rfdetr_coco_split_source(dataset_dir)
    records_by_split = limit_records_by_split(source["records_by_split"], config)
    assign_cache_file_names(records_by_split)
    categories = finalize_categories(records_by_split, source.get("class_names", []))
    split_ratio = parse_split_ratio(dataset.get("split_ratio", [8, 1, 1]))
    fingerprint = build_source_fingerprint("rfdetr_limited", source, records_by_split, categories, split_ratio, config)
    cache_root = resolve_cache_root(dataset.get("cache_root", "dataset_cache"), source_config)
    source_name = sanitize_name(str(source.get("source_name") or dataset_dir.name))
    cache_dir = cache_root / f"rfdetr_limited_{source_name}_{fingerprint['hash'][:16]}"
    link_mode = normalize_link_mode(dataset.get("link_mode", "auto"))
    images = unique_source_images(records_by_split)
    copy_bytes = image_bytes(images) if link_mode == "copy" else 0
    return {
        "source_format": "rfdetr",
        "action": "prepare_cache",
        "dataset_dir": dataset_dir,
        "cache_root": cache_root,
        "cache_dir": cache_dir,
        "fingerprint": fingerprint,
        "records_by_split": records_by_split,
        "categories": categories,
        "split_counts": {split: len(records) for split, records in records_by_split.items()},
        "class_names": [category["name"] for category in categories],
        "link_mode": link_mode,
        "copy_file_count": len(images) if link_mode == "copy" else 0,
        "copy_bytes": copy_bytes,
        "cache_file_count": len(images) + len(records_by_split) + 2,
        "refresh_cache": bool(dataset.get("refresh_cache", False)),
        "warnings": list(source.get("warnings", [])),
    }


def motion_module_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether the optional TrackNet module is actually requested."""
    model = config.get("model", {})
    motion = model.get("motion", {}) if isinstance(model, Mapping) else {}
    if not isinstance(motion, Mapping) or not bool(motion.get("enabled", False)):
        return False
    return str(motion.get("type", "tracknet_v5")).strip().lower() != "none"


def temporal_motion_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether this run requires real indexed temporal windows."""

    if not motion_module_enabled(config):
        return False
    motion = config["model"]["motion"]
    temporal = motion.get("temporal", {}) or {}
    mode = str(temporal.get("mode", temporal.get("fallback_mode", "real"))).strip().lower()
    return mode == "real"


def _align_temporal_num_classes(rf_model: Any, class_names: Sequence[str]) -> None:
    """Align an unset RF-DETR class head to the temporal dataset vocabulary."""

    if not class_names:
        raise ValueError("Temporal training requires at least one dataset class name.")
    model_config = rf_model.model_config
    dataset_num_classes = len(class_names)
    model_num_classes = int(model_config.num_classes)
    user_overrode = "num_classes" in getattr(model_config, "model_fields_set", set())
    model_args = getattr(getattr(rf_model, "model", None), "args", None)

    if dataset_num_classes == model_num_classes:
        if model_args is not None:
            model_args.num_classes = dataset_num_classes
        return
    if not user_overrode:
        model_config.num_classes = dataset_num_classes
        if model_args is not None:
            model_args.num_classes = dataset_num_classes
        return
    warnings.warn(
        "Temporal dataset has "
        f"{dataset_num_classes} classes but model.num_classes was explicitly set to "
        f"{model_num_classes}; preserving the explicit model setting.",
        UserWarning,
        stacklevel=2,
    )


def build_dataset_plan(
    config: Mapping[str, Any],
    output_dir: Path,
    source_config: Optional[Path],
) -> Dict[str, Any]:
    """Build a dataset preparation plan without writing output."""
    dataset_dir_value = get_dataset_dir_value(config)
    dataset_dir = None
    bases = config_path_bases(source_config)
    if dataset_dir_value:
        dataset_dir = resolve_existing_path(
            dataset_dir_value,
            bases,
            "dataset.dataset_dir/train.dataset_dir",
        )
    data_yaml = find_dataset_yaml(config, source_config)
    if temporal_motion_enabled(config):
        if dataset_dir is None:
            raise ValueError("Real TrackNet temporal mode requires dataset.dataset_dir.")
        if data_yaml is None:
            raise ValueError("Real TrackNet temporal mode requires dataset.data_yaml or dataset.yaml.")
        temporal = ((config.get("model", {}) or {}).get("motion", {}) or {}).get("temporal", {}) or {}
        num_frames = int(temporal.get("num_frames", 3))
        frame_stride = int(temporal.get("frame_stride", 1))
        from rf_detr_temporal_data import temporal_split_window_counts

        split_counts = temporal_split_window_counts(
            data_yaml,
            num_frames=num_frames,
            stride=frame_stride,
        )
        complete_split_counts = dict(split_counts)
        temporal_dataset = (config.get("dataset", {}) or {}).get("temporal", {}) or {}
        max_windows = (
            temporal_dataset.get("max_windows_per_split", {})
            if isinstance(temporal_dataset, Mapping)
            else {}
        )
        if isinstance(max_windows, int):
            max_windows = {
                "train": max_windows,
                "val": max_windows,
                "test": max_windows,
            }
        if not isinstance(max_windows, Mapping):
            raise ValueError(
                "dataset.temporal.max_windows_per_split must be an integer or mapping"
            )
        normalised_limits: Dict[str, int] = {}
        for raw_split, raw_limit in max_windows.items():
            split = str(raw_split).strip().lower()
            if split == "valid":
                split = "val"
            limit = int(raw_limit)
            if limit < 1:
                raise ValueError(
                    "dataset.temporal.max_windows_per_split values must be positive"
                )
            normalised_limits[split] = limit
        split_counts = {
            split: min(int(count), normalised_limits.get(split, int(count)))
            for split, count in complete_split_counts.items()
        }
        return {
            "source_format": "spatiotemporal_yolo",
            "action": "temporal_direct",
            "dataset_dir": dataset_dir,
            "data_yaml": data_yaml,
            "temporal_index": dataset_dir / "metadata" / "temporal_index.jsonl",
            "split_counts": split_counts,
            "complete_split_counts": complete_split_counts,
            "copy_file_count": 0,
            "copy_bytes": 0,
            "num_frames": num_frames,
            "frame_stride": frame_stride,
        }
    source_format = resolve_dataset_source_format(config, dataset_dir, data_yaml, source_config)
    if source_format != "rfdetr":
        return build_cache_dataset_plan(config, source_format, dataset_dir, data_yaml, source_config)
    if has_dataset_limits(config):
        return build_rfdetr_limited_dataset_plan(config, dataset_dir, source_config)
    split_counts: Dict[str, Optional[int]] = {}
    if dataset_dir is not None:
        split_counts = {
            split: maybe_count_images(dataset_split_dir(dataset_dir, split))
            for split in ("train", "valid", "val", "test")
        }
    return {
        "source_format": source_format,
        "action": "none",
        "dataset_dir": dataset_dir,
        "data_yaml": data_yaml,
        "split_counts": split_counts,
        "copy_file_count": 0,
        "copy_bytes": 0,
    }


def assert_within_directory(path: Path, parent: Path, label: str) -> None:
    """Ensure a path resolves inside a parent directory."""
    resolved = path.resolve()
    resolved_parent = parent.resolve()
    try:
        resolved.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {resolved_parent}, got {resolved}") from exc


def clean_prepared_dir(prepared_dir: Path, output_dir: Path, overwrite: bool) -> None:
    """Create or optionally clear the prepared dataset directory safely."""
    assert_within_directory(prepared_dir, output_dir, "prepared dataset directory")
    if not prepared_dir.exists():
        prepared_dir.mkdir(parents=True, exist_ok=True)
        return
    existing = list(prepared_dir.iterdir())
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(
            f"Prepared dataset directory already exists and dataset.overwrite_prepared_dataset=false: {prepared_dir}"
        )
    for child in existing:
        assert_within_directory(child, output_dir, "prepared dataset child")
        is_junction = bool(getattr(child, "is_junction", lambda: False)())
        if is_junction:
            child.rmdir()
        elif child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def create_directory_symlink(source: Path, target: Path) -> None:
    """Create a directory symlink."""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source), str(target), target_is_directory=True)


def create_directory_junction(source: Path, target: Path) -> None:
    """Create a Windows directory junction."""
    if os.name != "nt":
        raise OSError("junction link mode is only available on Windows.")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(target), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise OSError(f"Failed to create junction {target} -> {source}: {message}")


def create_directory_link(source: Path, target: Path, mode: str) -> str:
    """Create a directory link and return the actual mode used."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Prepared dataset target already exists: {target}")
    if mode == "copy":
        raise ValueError("create_directory_link does not handle copy mode.")
    if mode == "symlink":
        create_directory_symlink(source, target)
        return "symlink"
    if mode == "junction":
        create_directory_junction(source, target)
        return "junction"

    errors: List[str] = []
    preferred = ("junction", "symlink") if os.name == "nt" else ("symlink",)
    for candidate in preferred:
        try:
            if candidate == "junction":
                create_directory_junction(source, target)
            else:
                create_directory_symlink(source, target)
            return candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError(
        "Could not create dataset links automatically. "
        "Set dataset.link_mode=copy to copy files after reviewing the output estimate, "
        f"or fix link permissions. Details: {'; '.join(errors)}"
    )


def copy_directory_contents(source: Path, target: Path, progress: tqdm) -> None:
    """Copy directory contents with a shared progress bar."""
    if target.exists():
        raise FileExistsError(f"Prepared dataset target already exists: {target}")
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        destination = target / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        progress.update(1)


def clean_cache_dir(cache_dir: Path, cache_root: Path) -> None:
    """Remove an existing cache directory after verifying it is under cache_root."""
    assert_within_directory(cache_dir, cache_root, "cache dataset directory")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def create_file_symlink(source: Path, target: Path) -> None:
    """Create a file symlink."""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source), str(target), target_is_directory=False)


def create_file_link(source: Path, target: Path, mode: str) -> str:
    """Create a file link/copy and return the actual mode used."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Cache image target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
        return "copy"
    if mode == "hardlink":
        os.link(source, target)
        return "hardlink"
    if mode == "symlink":
        create_file_symlink(source, target)
        return "symlink"

    errors: List[str] = []
    preferred = ("hardlink", "symlink") if mode in {"auto", "junction"} else (mode,)
    for candidate in preferred:
        try:
            if candidate == "hardlink":
                os.link(source, target)
            else:
                create_file_symlink(source, target)
            return candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
    raise OSError(
        "Could not link cache image automatically. "
        "Set dataset.link_mode=copy to copy files after reviewing the output estimate. "
        f"Details: {'; '.join(errors)}"
    )


def cache_is_ready(plan: Mapping[str, Any]) -> bool:
    """Return True when an existing cache matches the current source fingerprint."""
    cache_dir = Path(plan["cache_dir"])
    fingerprint_path = cache_dir / "source_fingerprint.json"
    if not fingerprint_path.exists():
        return False
    with contextlib.suppress(Exception):
        data = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if data.get("hash") != plan.get("fingerprint", {}).get("hash"):
            return False
        for split in plan.get("records_by_split", {}):
            if not (cache_dir / split / "_annotations.coco.json").exists():
                return False
        return True
    return False


def write_cache_split_coco(split_dir: Path, records: Sequence[Mapping[str, Any]], categories: Sequence[Mapping[str, Any]]) -> None:
    """Write one split's Roboflow COCO annotation file."""
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    annotation_id = 1
    for image_id, record in enumerate(records, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": str(record["file_name"]),
                "width": int(record["width"]),
                "height": int(record["height"]),
            }
        )
        for annotation in record.get("annotations", []):
            bbox = [float(item) for item in annotation["bbox"]]
            converted = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(annotation["category_id"]),
                "bbox": bbox,
                "area": round(float(bbox[2]) * float(bbox[3]), 6),
                "iscrowd": int(annotation.get("iscrowd", 0) or 0),
            }
            if annotation.get("segmentation"):
                converted["segmentation"] = annotation["segmentation"]
            annotations.append(converted)
            annotation_id += 1
    write_json(
        split_dir / "_annotations.coco.json",
        {
            "info": {"description": "RF-DETR cached dataset", "converter_version": CONVERTER_VERSION},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": list(categories),
        },
    )


def materialize_cache_dataset_plan(
    plan: Mapping[str, Any],
    config: MutableMapping[str, Any],
    output_dir: Path,
    verbose: bool,
) -> Dict[str, Any]:
    """Create or reuse an RF-DETR-readable Roboflow COCO cache dataset."""
    cache_dir = Path(plan["cache_dir"])
    cache_root = Path(plan["cache_root"])
    link_mode = str(plan.get("link_mode", "auto"))
    cache_reused = cache_is_ready(plan) and not bool(plan.get("refresh_cache", False))
    image_modes: Dict[str, Dict[str, str]] = {}
    if cache_reused:
        blue(f"Reusing RF-DETR dataset cache at {cache_dir}.", verbose)
    else:
        blue(f"Preparing RF-DETR dataset cache at {cache_dir}.", verbose)
        clean_cache_dir(cache_dir, cache_root)
        total = sum(len(records) for records in plan.get("records_by_split", {}).values())
        with tqdm(total=total, desc="Cache dataset", unit="image") as progress:
            for split, records in plan.get("records_by_split", {}).items():
                split_dir = cache_dir / split
                split_dir.mkdir(parents=True, exist_ok=True)
                split_modes: Dict[str, str] = {}
                for record in records:
                    source_image = Path(record["source_image"])
                    target_image = split_dir / str(record["file_name"])
                    split_modes[str(record["file_name"])] = create_file_link(source_image, target_image, link_mode)
                    progress.update(1)
                write_cache_split_coco(split_dir, records, plan["categories"])
                image_modes[split] = split_modes
        write_json(cache_dir / "source_fingerprint.json", plan["fingerprint"])

    metadata = {
        "source_format": plan.get("source_format"),
        "cache_reused": cache_reused,
        "cache_dir": str(cache_dir),
        "cache_root": str(cache_root),
        "fingerprint_hash": plan.get("fingerprint", {}).get("hash"),
        "link_mode_requested": link_mode,
        "link_mode_used": image_modes,
        "split_counts": dict(plan.get("split_counts", {})),
        "dataset_limits": plan.get("fingerprint", {}).get("dataset_limits", {}),
        "class_names": list(plan.get("class_names", [])),
        "warnings": list(plan.get("warnings", [])),
    }
    if cache_reused and (cache_dir / "adapter_metadata.json").exists():
        with contextlib.suppress(Exception):
            metadata = json.loads((cache_dir / "adapter_metadata.json").read_text(encoding="utf-8"))
            metadata["cache_reused"] = True
    write_json(cache_dir / "adapter_metadata.json", metadata)

    dataset_cfg = config.setdefault("dataset", {})
    train_cfg = config.setdefault("train", {})
    dataset_cfg["cache_dataset_dir"] = str(cache_dir)
    dataset_cfg["adapter_metadata"] = metadata
    dataset_cfg["dataset_dir"] = str(cache_dir)
    train_cfg["dataset_dir"] = str(cache_dir)
    train_cfg["dataset_file"] = "roboflow"
    return metadata


def materialize_dataset_plan(
    plan: Mapping[str, Any],
    config: MutableMapping[str, Any],
    output_dir: Path,
    verbose: bool,
) -> Dict[str, Any]:
    """Create any dataset adapter outputs required by the plan."""
    if plan.get("action") == "prepare_cache":
        return materialize_cache_dataset_plan(plan, config, output_dir, verbose)
    if plan.get("action") == "temporal_direct":
        return {
            "source_format": "spatiotemporal_yolo",
            "dataset_dir": str(plan.get("dataset_dir")),
            "data_yaml": str(plan.get("data_yaml")),
            "temporal_index": str(plan.get("temporal_index")),
            "split_window_counts": dict(plan.get("split_counts", {})),
            "num_frames": int(plan.get("num_frames", 3)),
            "frame_stride": int(plan.get("frame_stride", 1)),
        }
    return {}


def estimate_periodic_tests(config: Mapping[str, Any]) -> Tuple[Optional[int], str]:
    """Estimate how many scheduled test runs will be produced."""
    periodic = config.get("periodic_test", {})
    if not periodic.get("enabled", True):
        return 0, "disabled"
    train = config.get("train", {})
    epochs = train.get("epochs")
    by_epoch = int(periodic.get("test_interval_epochs") or 0)
    by_minutes = float(periodic.get("test_interval_minutes") or 0.0)
    if epochs and by_epoch > 0:
        return int(math.floor(float(epochs) / by_epoch)), "estimated from epochs"
    if train.get("max_time_minutes") and by_minutes > 0:
        return int(math.floor(float(train["max_time_minutes"]) / by_minutes)), "estimated from time"
    if by_minutes > 0:
        return None, "unknown because runtime depends on training speed"
    return 0, "no interval configured"


def estimate_outputs(
    config: Mapping[str, Any],
    output_dir: Path,
    periodic_count: Optional[int],
    dataset_plan: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Estimate output files and disk usage before training."""
    train = config.get("train", {})
    periodic = config.get("periodic_test", {})
    if dataset_plan is not None and dataset_plan.get("split_counts"):
        split_counts = dict(dataset_plan.get("split_counts", {}))
    else:
        dataset_dir_value = get_dataset_dir_value(config)
        dataset_dir = Path(str(dataset_dir_value)).expanduser()
        split_counts = {
            split: maybe_count_images(dataset_split_dir(dataset_dir, split)) if dataset_dir_value else None
            for split in ("train", "valid", "val", "test")
        }

    epochs = int(train.get("epochs") or 100)
    checkpoint_interval = max(1, int(train.get("checkpoint_interval") or 10))
    checkpoint_files = 4 + int(math.ceil(epochs / checkpoint_interval))
    logger_files = 4
    train_batch_grid_files = TRAIN_BATCH_GRID_MAX_BATCHES
    validation_label_grid_files = VALIDATION_PREDICTION_GRID_MAX_BATCHES
    validation_prediction_grid_files = VALIDATION_PREDICTION_GRID_MAX_BATCHES
    batch_grid_files = train_batch_grid_files + validation_label_grid_files + validation_prediction_grid_files
    epoch_validation_result_files = epochs * 2 + 2
    per_test_files = 6 if bool(periodic.get("classwise", True)) else 4
    periodic_files = None if periodic_count is None else periodic_count * per_test_files
    final_files = per_test_files if bool(periodic.get("run_final_test", True)) else 0
    total_known = (
        checkpoint_files
        + logger_files
        + batch_grid_files
        + epoch_validation_result_files
        + final_files
        + (periodic_files or 0)
    )
    approx_bytes = checkpoint_files * 250 * 1024 * 1024
    approx_bytes += epoch_validation_result_files * 8 * 1024
    approx_bytes += batch_grid_files * 2 * 1024 * 1024
    dataset_cache_files = 0
    dataset_cache_bytes = 0
    dataset_source_format = dataset_plan.get("source_format") if dataset_plan else "unknown"
    dataset_link_mode = dataset_plan.get("link_mode") if dataset_plan else None
    if dataset_plan and dataset_plan.get("action") == "prepare_cache":
        dataset_cache_files = int(dataset_plan.get("cache_file_count", 0) or 0)
        dataset_cache_bytes = int(dataset_plan.get("copy_bytes", 0) or 0)
        total_known += dataset_cache_files
        approx_bytes += dataset_cache_bytes

    train_images = split_counts.get("train") or 0
    try:
        batch_size = int(train.get("batch_size") if train.get("batch_size") != "auto" else train.get("auto_batch_target_effective", 1))
    except (TypeError, ValueError):
        batch_size = int(train.get("auto_batch_target_effective", 1) or 1)
    batch_size = max(1, batch_size)
    train_batches = max(1, int(math.ceil(float(train_images or 0) / batch_size))) if train_images else 1
    try:
        grad_accum_steps = max(1, int(train.get("grad_accum_steps", 1) or 1))
    except (TypeError, ValueError):
        grad_accum_steps = 1
    optimizer_steps_per_epoch = int(
        math.ceil(float(train_batches) / grad_accum_steps)
    )
    total_optimizer_steps = epochs * optimizer_steps_per_epoch
    training_sanity_warnings: List[str] = []
    if grad_accum_steps > train_batches:
        training_sanity_warnings.append(
            "grad_accum_steps exceeds train batches per epoch; this produces "
            "only one optimizer update per epoch."
        )
    if optimizer_steps_per_epoch < 5:
        training_sanity_warnings.append(
            "Fewer than five optimizer updates are expected per epoch; "
            "loss/mAP may appear stationary for many epochs."
        )
    if total_optimizer_steps < 100:
        training_sanity_warnings.append(
            "Fewer than 100 total optimizer updates are expected; this is "
            "appropriate only for a smoke test or a deliberate micro-run."
        )
    runtime_units = float(epochs * train_batches)

    estimate = {
        "output_dir": str(output_dir),
        "dataset_source_format": dataset_source_format,
        "dataset_link_mode": dataset_link_mode,
        "dataset_cache_dir": str(dataset_plan.get("cache_dir")) if dataset_plan and dataset_plan.get("cache_dir") else None,
        "dataset_cache_files": dataset_cache_files,
        "dataset_cache_disk_usage": format_bytes(dataset_cache_bytes),
        "split_image_counts": split_counts,
        "checkpoint_files": checkpoint_files,
        "epoch_validation_result_files": epoch_validation_result_files,
        "epoch_validation_results_dir": str(output_dir / "epoch_results"),
        "batch_grid_files": batch_grid_files,
        "train_batch_grid_files": train_batch_grid_files,
        "validation_label_grid_files": validation_label_grid_files,
        "validation_prediction_grid_files": validation_prediction_grid_files,
        "batch_grid_dir": str(output_dir),
        "periodic_test_files": periodic_files,
        "final_test_files": final_files,
        "estimated_total_files": total_known,
        "estimated_disk_usage": format_bytes(approx_bytes),
        "micro_batches_per_epoch": train_batches,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "training_sanity_warnings": training_sanity_warnings,
        "note": "Estimates are conservative approximations. Batch grid images overwrite fixed filenames.",
    }
    if dataset_plan and dataset_plan.get("action") == "temporal_direct":
        estimate["split_window_counts"] = split_counts
        estimate["complete_temporal_window_counts"] = dict(
            dataset_plan.get("complete_split_counts", split_counts)
        )
    add_runtime_estimate(
        estimate=estimate,
        config=config,
        output_dir=output_dir,
        task="train",
        runtime_units=runtime_units,
        default_rate_key="default_train_seconds_per_batch",
        basis={
            "epochs": epochs,
            "train_images": train_images,
            "batch_size": batch_size,
            "train_batches_per_epoch": train_batches,
            "grad_accum_steps": grad_accum_steps,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
        },
    )
    return estimate


def confirm_or_exit(estimate: Mapping[str, Any], verbose: bool, assume_yes: bool) -> None:
    """Ask for confirmation before heavy output is created."""
    blue("Output and resource estimate before training:", verbose=verbose, force=True)
    print(json.dumps(dict(estimate), indent=2, ensure_ascii=False))
    if assume_yes:
        blue("Confirmation skipped because --yes or confirm_before_run=false is enabled.", verbose=verbose, force=True)
        return
    answer = input(Fore.BLUE + Style.BRIGHT + "Continue and start training? [y/N]: " + Style.RESET_ALL).strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted by developer before heavy output was produced.")


def normalize_model_size(size: Any) -> str:
    """Normalize RF-DETR size aliases, class names, and hosted weight names."""
    text = str(size or "").strip()
    if not text:
        raise ValueError("model.size must not be empty.")
    text = re.split(r"[\\/]", text)[-1].strip().lower().replace("_", "-")
    for suffix in (".pth", ".pt", ".ckpt"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = MODEL_SIZE_ALIASES.get(text, text)
    if text in MODEL_SIZE_CLASS_NAMES:
        return text
    return MODEL_SIZE_ALIASES.get(text, text)


def get_available_model_classes() -> Dict[str, Any]:
    """Return canonical RF-DETR model sizes supported by the installed package."""
    import rfdetr

    mapping: Dict[str, Any] = {}
    for size, class_name in MODEL_SIZE_CLASS_NAMES.items():
        model_cls = getattr(rfdetr, class_name, None)
        if model_cls is not None:
            mapping[size] = model_cls
    return mapping


def available_model_size_options() -> str:
    """Return a user-facing list of model.size options."""
    return ", ".join(get_available_model_classes().keys())


def is_segmentation_model_size(size: Any) -> bool:
    """Return whether a model.size value points at an RF-DETR segmentation model."""
    try:
        normalized = normalize_model_size(size)
    except ValueError:
        return False
    return normalized.startswith("seg-")


def get_model_class(size: str) -> Any:
    """Resolve an RF-DETR model class by friendly size name."""
    normalized = normalize_model_size(size)
    mapping = get_available_model_classes()
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"Unsupported RF-DETR model size {size!r}. Options for this rfdetr install: {available_model_size_options()}.")


def build_output_dir(config: Mapping[str, Any], timestamp: str) -> Path:
    """Build the final output directory."""
    output = config.get("output", {})
    exact = render_output_template(output.get("output_dir", ""), config, timestamp)
    if exact:
        return resolve_path_for_output(exact)
    root = render_output_template(output.get("root", "runs/rf_detr"), config, timestamp)
    name = render_output_template(output.get("name", "rf_detr_{timestamp}"), config, timestamp)
    return resolve_path_for_output(root) / sanitize_name(str(name))


def normalize_pretrain_weights(value: Any) -> Tuple[bool, Optional[str]]:
    """Return whether to pass pretrain_weights plus its normalized value."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            return False, None
        if text in {"none", "null", "false", "0", "no", "off"}:
            return True, None
        if text in {"true", "1", "yes", "on"}:
            return True, "default"
    if value is None:
        return True, None
    if isinstance(value, bool):
        return True, None if not value else "default"
    resolved = resolve_existing_or_raw(value, [Path.cwd(), PROJECT_DIR, REPO_ROOT])
    return True, str(resolved)


def temporal_dataset_manifest_sha256(config: Mapping[str, Any]) -> Optional[str]:
    """Hash the canonical temporal index for checkpoint provenance."""

    if not temporal_motion_enabled(config):
        return None
    dataset = config.get("dataset", {})
    if not isinstance(dataset, Mapping):
        return None
    yaml_value = str(dataset.get("data_yaml") or "").strip()
    if yaml_value:
        yaml_path = Path(yaml_value).expanduser()
        if not yaml_path.is_absolute():
            yaml_path = (Path.cwd() / yaml_path).resolve()
        dataset_root = yaml_path.parent
    else:
        root_value = str(dataset.get("dataset_dir") or "").strip()
        if not root_value:
            return None
        dataset_root = Path(root_value).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (Path.cwd() / dataset_root).resolve()
    manifest = dataset_root / "metadata" / "temporal_index.jsonl"
    if not manifest.is_file():
        return None
    digest = hashlib.sha256()
    with manifest.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pitchobjectlab_architecture(
    config: Mapping[str, Any],
    model_config: Optional[Any] = None,
) -> Dict[str, Any]:
    '''Serialize only graph-affecting project architecture settings.'''

    model = config.get('model', {})
    if not isinstance(model, Mapping):
        model = {}
    p2 = model.get('p2', {}) or {}
    motion = model.get('motion', {}) or {}
    projector = p2.get('projector', {}) or {}
    p2_enabled = bool(p2.get('enabled', False))
    motion_enabled = motion_module_enabled(config)
    resolved_resolution = (
        getattr(model_config, 'resolution', None)
        if model_config is not None
        else model.get('resolution')
    )
    resolved_num_classes = (
        getattr(model_config, 'num_classes', None)
        if model_config is not None
        else model.get('num_classes')
    )
    resolved_num_queries = (
        getattr(model_config, 'num_queries', None)
        if model_config is not None
        else (
            (motion.get('overrides', {}) or {}).get('num_queries')
            or (p2.get('overrides', {}) or {}).get('num_queries')
            or (model.get('extra_model_args', {}) or {}).get('num_queries')
        )
    )
    resolved_num_select = (
        getattr(model_config, 'num_select', None)
        if model_config is not None
        else (
            (motion.get('overrides', {}) or {}).get('num_select')
            or (p2.get('overrides', {}) or {}).get('num_select')
            or (model.get('extra_model_args', {}) or {}).get('num_select')
        )
    )
    resolved_dec_n_points = (
        getattr(model_config, 'dec_n_points', None)
        if model_config is not None
        else (p2.get('overrides', {}) or {}).get('dec_n_points')
    )
    projector_scale = (
        list(p2.get('projector_scale', ['P2', 'P3', 'P4']) or [])
        if p2_enabled
        else []
    )
    p2_metadata = {
        'enabled': p2_enabled,
        'projector_scale': projector_scale,
        'projector': {
            key: deepcopy(projector.get(key))
            for key in (
                'num_blocks',
                'survival_prob',
                'force_drop_last_n_features',
                'layer_norm',
                'rms_norm',
            )
            if projector.get(key) is not None
        },
    }
    motion_metadata = {
        'enabled': motion_enabled,
        'type': str(motion.get('type', 'tracknet_v5')),
        'temporal': deepcopy(dict(motion.get('temporal', {}) or {})),
        'focus': deepcopy(dict(motion.get('focus', {}) or {})),
        'tracknet_v5': deepcopy(dict(motion.get('tracknet_v5', {}) or {})),
        'loss': deepcopy(dict(motion.get('loss', {}) or {})),
    }
    graph = {
        'model_size': str(model.get('size', 'medium')).strip().lower(),
        'resolution': resolved_resolution,
        'num_queries': resolved_num_queries,
        'num_select': resolved_num_select,
        'num_classes': resolved_num_classes,
        'dec_n_points': resolved_dec_n_points,
        'p2': p2_metadata,
        'motion': motion_metadata,
    }
    graph_json = json.dumps(
        graph,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    return {
        'schema_version': PITCHOBJECTLAB_ARCHITECTURE_SCHEMA_VERSION,
        'model_size': str(model.get('size', 'medium')),
        'rf_detr_version': '1.8.3',
        'graph': graph,
        'architecture_fingerprint': hashlib.sha256(
            graph_json.encode('utf-8')
        ).hexdigest(),
        'p2': p2_metadata,
        'motion': motion_metadata,
        'dataset_manifest_sha256': temporal_dataset_manifest_sha256(config),
        'tensorrt_export_abi': PITCHOBJECTLAB_TENSORRT_EXPORT_ABI,
    }


def install_best_checkpoint_metadata(trainer: Any, metadata: Mapping[str, Any]) -> None:
    '''Inject architecture metadata into regular and EMA checkpoint payloads.'''

    installed = False
    for callback in getattr(trainer, 'callbacks', []):
        if type(callback).__name__ != 'BestModelCallback':
            continue
        original = callback._build_checkpoint_payload
        frozen_metadata = deepcopy(dict(metadata))

        def build_payload(*args: Any, _original: Any = original, **kwargs: Any) -> Dict[str, Any]:
            payload = dict(_original(*args, **kwargs))
            payload[PITCHOBJECTLAB_ARCHITECTURE_KEY] = deepcopy(frozen_metadata)
            args_payload = payload.get('args')
            if isinstance(args_payload, Mapping):
                nested_args = dict(args_payload)
                nested_args[PITCHOBJECTLAB_ARCHITECTURE_KEY] = deepcopy(frozen_metadata)
                payload['args'] = nested_args
            return payload

        callback._build_checkpoint_payload = build_payload
        installed = True
    if not installed:
        raise RuntimeError(
            'RF-DETR trainer has no BestModelCallback; cannot preserve project architecture metadata.'
        )


def enrich_best_checkpoint_metadata(output_dir: Path) -> None:
    '''Restore top-level metadata after RF-DETR strips the total checkpoint.'''

    import torch

    for name in (
        'checkpoint_best_regular.pth',
        'checkpoint_best_ema.pth',
        'checkpoint_best_total.pth',
    ):
        path = output_dir / name
        if not path.is_file():
            continue
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location='cpu')
        if not isinstance(checkpoint, MutableMapping):
            continue
        metadata = checkpoint.get(PITCHOBJECTLAB_ARCHITECTURE_KEY)
        args_payload = checkpoint.get('args')
        if metadata is None and isinstance(args_payload, Mapping):
            metadata = args_payload.get(PITCHOBJECTLAB_ARCHITECTURE_KEY)
        if not isinstance(metadata, Mapping):
            continue
        checkpoint[PITCHOBJECTLAB_ARCHITECTURE_KEY] = deepcopy(dict(metadata))
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f'.{path.name}.',
            suffix='.metadata.tmp',
            dir=path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def build_model_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build RF-DETR model constructor kwargs."""
    model_cfg = deepcopy(config.get("model", {}))
    kwargs = deepcopy(model_cfg.get("extra_model_args", {}) or {})
    if model_cfg.get("resolution") is not None:
        kwargs["resolution"] = int(model_cfg["resolution"])
    if model_cfg.get("num_classes") is not None:
        kwargs["num_classes"] = int(model_cfg["num_classes"])
    device = normalize_model_constructor_device(model_cfg.get("device"))
    if device is not None:
        kwargs["device"] = device
    if model_cfg.get("amp") is not None:
        kwargs["amp"] = bool(model_cfg["amp"])
    # Optional pluggable P2 (stride-4) feature level. Fully inert unless model.p2.enabled.
    # ensure_p2_support patches rfdetr in-process (relax projector_scale Literal + teach the
    # backbone P2=4.0); projector_scale + overrides become real ModelConfig kwargs. This is the
    # shared model-build choke point, so this also covers test/inference/tracking entrypoints.
    p2_cfg = model_cfg.get("p2", {}) or {}
    from rf_detr_p2 import apply_p2_overrides, ensure_p2_support, resolve_p2_projector_scale

    # Install the version-pinned Backbone export path for every architecture.
    # Stock/TrackNet models also need the static DINO -> projector reshape
    # boundary; P2 scale and mismatch behavior remain conditional below.
    ensure_p2_support(p2_cfg)
    if bool(p2_cfg.get("enabled", False)):
        kwargs["projector_scale"] = resolve_p2_projector_scale(p2_cfg)
        apply_p2_overrides(kwargs, p2_cfg)
    # Optional TrackNet integration is imported and installed only for enabled models.
    # A disabled build therefore remains the exact upstream LWDETR path.
    motion_cfg = model_cfg.get("motion", {}) or {"enabled": False}
    if motion_module_enabled(config):
        from rf_detr_motion import apply_motion_overrides, ensure_motion_support

        ensure_motion_support(motion_cfg)
        apply_motion_overrides(kwargs, motion_cfg)
    should_pass, pretrain = normalize_pretrain_weights(model_cfg.get("pretrain_weights", "default"))
    if should_pass:
        if pretrain == "default":
            pass
        else:
            kwargs["pretrain_weights"] = pretrain
    return kwargs


def _rfdetr_keypoint_flip_pairs_for_task(args: Any) -> Optional[List[int]]:
    """Return RF-DETR keypoint flip pairs only for keypoint tasks."""
    if bool(getattr(args, "use_grouppose_keypoints", False)):
        return list(getattr(args, "keypoint_flip_pairs", []) or [])
    return None


def ensure_rfdetr_invalid_bbox_filter_support() -> None:
    """Filter bboxes that become degenerate during Albumentations normalization.

    RF-DETR pre-filters boxes whose absolute coordinates are already degenerate,
    but a sub-pixel box can still collapse when Albumentations normalizes its
    float32 coordinates. Enabling the processor's native invalid-box filter
    drops those boxes together with the configured label fields before
    validation, instead of crashing a DataLoader worker.
    """
    global _RFDETR_INVALID_BBOX_FILTER_PATCHED
    if _RFDETR_INVALID_BBOX_FILTER_PATCHED:
        return

    from rfdetr.datasets.transforms import AlbumentationsWrapper

    current_init = AlbumentationsWrapper.__init__
    if getattr(current_init, "_pitchobjectlab_invalid_bbox_filter_patch", False):
        _RFDETR_INVALID_BBOX_FILTER_PATCHED = True
        return

    @wraps(current_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        current_init(self, *args, **kwargs)
        processors = getattr(getattr(self, "transform", None), "processors", {})
        bbox_processor = processors.get("bboxes") if isinstance(processors, Mapping) else None
        bbox_params = getattr(bbox_processor, "params", None)
        if bbox_params is None:
            return
        if not hasattr(bbox_params, "filter_invalid_bboxes"):
            raise RuntimeError(
                "The installed Albumentations version does not support "
                "BboxParams.filter_invalid_bboxes; run `uv sync` to restore "
                "the project-locked dependencies."
            )
        bbox_params.filter_invalid_bboxes = True

    patched_init._pitchobjectlab_invalid_bbox_filter_patch = True
    AlbumentationsWrapper.__init__ = patched_init
    _RFDETR_INVALID_BBOX_FILTER_PATCHED = True


def ensure_rfdetr_detection_hflip_support() -> None:
    """Install RF-DETR CPU augmentation compatibility fixes."""
    ensure_rfdetr_invalid_bbox_filter_support()

    global _RFDETR_DETECTION_HFLIP_PATCHED
    if _RFDETR_DETECTION_HFLIP_PATCHED:
        return

    import rfdetr.datasets as datasets_module
    import rfdetr.datasets.coco as coco_module

    if getattr(coco_module.build_roboflow_from_coco, "_pitchobjectlab_detection_hflip_patch", False):
        _RFDETR_DETECTION_HFLIP_PATCHED = True
        return

    def build_coco(image_set: str, args: Any, resolution: int) -> Any:
        root = Path(getattr(args, "dataset_dir", None) or args.coco_path)
        if not root.exists():
            coco_module.logger.error(f"COCO path {root} does not exist")
            raise FileNotFoundError(f"COCO path {root} does not exist")

        has_keypoints = getattr(args, "use_grouppose_keypoints", False)
        mode = "person_keypoints" if has_keypoints else "instances"
        paths = {
            "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
            "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
            "test": (root / "test2017", root / "annotations" / "image_info_test-dev2017.json"),
        }
        img_folder, ann_file = paths[image_set.split("_")[0]]

        include_keypoints = has_keypoints
        resolved_backend = coco_module._resolve_runtime_augmentation_backend(getattr(args, "augmentation_backend", "cpu"))
        gpu_postprocess = resolved_backend != "cpu"
        make_transforms = (
            coco_module.make_coco_transforms_square_div_64
            if getattr(args, "square_resize_div_64", False)
            else coco_module.make_coco_transforms
        )
        if getattr(args, "square_resize_div_64", False):
            coco_module.logger.info(f"Building COCO {image_set} dataset with square resize at resolution {resolution}")
        else:
            coco_module.logger.info(f"Building COCO {image_set} dataset at resolution {resolution}")
        return coco_module.CocoDetection(
            img_folder,
            ann_file,
            transforms=make_transforms(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                skip_random_resize=not args.do_random_resize_via_padding,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
                aug_config=getattr(args, "aug_config", None),
                gpu_postprocess=gpu_postprocess,
                keypoint_flip_pairs=_rfdetr_keypoint_flip_pairs_for_task(args),
            ),
            include_masks=getattr(args, "segmentation_head", False),
            include_keypoints=include_keypoints,
            num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
            remap_category_ids=include_keypoints,
        )

    def build_roboflow_from_coco(image_set: str, args: Any, resolution: int) -> Any:
        root = Path(args.dataset_dir)
        if not root.exists():
            coco_module.logger.error(f"Roboflow dataset path {root} does not exist")
            raise FileNotFoundError(f"Roboflow dataset path {root} does not exist")

        paths = {
            "train": (root / "train", root / "train" / "_annotations.coco.json"),
            "val": (root / "valid", root / "valid" / "_annotations.coco.json"),
            "test": (root / "test", root / "test" / "_annotations.coco.json"),
        }
        img_folder, ann_file = paths[image_set.split("_")[0]]

        resolved_backend = coco_module._resolve_runtime_augmentation_backend(getattr(args, "augmentation_backend", "cpu"))
        gpu_postprocess = resolved_backend != "cpu"
        make_transforms = (
            coco_module.make_coco_transforms_square_div_64
            if getattr(args, "square_resize_div_64", False)
            else coco_module.make_coco_transforms
        )
        if getattr(args, "square_resize_div_64", False):
            coco_module.logger.info(
                f"Building Roboflow {image_set} dataset with square resize at resolution {resolution}"
            )
        else:
            coco_module.logger.info(f"Building Roboflow {image_set} dataset at resolution {resolution}")
        return coco_module.CocoDetection(
            img_folder,
            ann_file,
            transforms=make_transforms(
                image_set,
                resolution,
                multi_scale=getattr(args, "multi_scale", False),
                expanded_scales=getattr(args, "expanded_scales", False),
                skip_random_resize=not getattr(args, "do_random_resize_via_padding", False),
                patch_size=getattr(args, "patch_size", 16),
                num_windows=getattr(args, "num_windows", 4),
                aug_config=getattr(args, "aug_config", None),
                gpu_postprocess=gpu_postprocess,
                keypoint_flip_pairs=_rfdetr_keypoint_flip_pairs_for_task(args),
            ),
            include_masks=getattr(args, "segmentation_head", False),
            include_keypoints=getattr(args, "use_grouppose_keypoints", False),
            num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
            remap_category_ids=True,
        )

    build_coco._pitchobjectlab_detection_hflip_patch = True
    build_roboflow_from_coco._pitchobjectlab_detection_hflip_patch = True
    coco_module.build_coco = build_coco
    coco_module.build_roboflow_from_coco = build_roboflow_from_coco
    datasets_module.build_coco = build_coco
    datasets_module.build_roboflow_from_coco = build_roboflow_from_coco
    _RFDETR_DETECTION_HFLIP_PATCHED = True


def normalize_model_constructor_device(value: Any) -> Optional[str]:
    """Normalize config/CLI shortcuts into RF-DETR model constructor device strings."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return "cpu" if value < 0 else f"cuda:{value}"
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower == "auto":
        return None
    if lower == "-1":
        return "cpu"
    if lower.isdecimal():
        return f"cuda:{int(lower)}"
    if "," in lower:
        first = lower.split(",", 1)[0].strip()
        if first.isdecimal():
            return f"cuda:{int(first)}"
    if lower in {"cpu", "cuda", "mps"}:
        return lower
    if lower.startswith("cuda:") and lower.split(":", 1)[1].isdecimal():
        return lower
    return text


def build_train_kwargs(config: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Build RF-DETR TrainConfig kwargs."""
    train = deepcopy(config.get("train", {}))
    extra = train.pop("extra_train_args", {}) or {}
    device = train.pop("device", None)
    train.pop("max_time_minutes", None)
    train_kwargs: Dict[str, Any] = {}
    train_kwargs.update(train)
    train_kwargs.update(extra)
    train_kwargs.setdefault("num_workers", 2)
    train_kwargs["dataset_dir"] = str(resolve_existing_or_raw(train_kwargs["dataset_dir"], [Path.cwd(), REPO_ROOT, PROJECT_DIR]))
    train_kwargs["output_dir"] = str(output_dir)
    if device not in (None, "", "auto"):
        train_kwargs["_device"] = str(device)
    train_kwargs["run_test"] = bool(train_kwargs.get("run_test", False))
    return train_kwargs


def resolve_train_resume_checkpoint(
    config: MutableMapping[str, Any],
    source_config: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve and validate ``train.resume`` before any run output is created."""
    train = config.setdefault("train", {})
    resume = train.get("resume")
    if resume is None or (isinstance(resume, str) and not resume.strip()):
        train["resume"] = None
        return None

    bases = [source_config.parent] if source_config is not None else []
    bases.extend([Path.cwd(), PROJECT_DIR, REPO_ROOT])
    checkpoint = resolve_existing_path(resume, bases, "train.resume")
    if checkpoint.suffix.lower() != ".ckpt":
        raise ValueError(
            f"train.resume must point to a PyTorch Lightning .ckpt file, got: {checkpoint}"
        )
    train["resume"] = str(checkpoint)
    return checkpoint


def size_hw(value: Any) -> Optional[Tuple[float, float]]:
    """Return a positive (height, width) pair from tensor/list-like values."""
    values = flatten_tensor(value)
    if len(values) < 2:
        return None
    try:
        height = float(values[0])
        width = float(values[1])
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    return height, width


def tensor_image_size_hw(tensor: Any) -> Optional[Tuple[float, float]]:
    """Return (height, width) from a CHW/HW image tensor-like value."""
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        dims = tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None
    if len(dims) >= 3:
        height, width = dims[-2], dims[-1]
    elif len(dims) == 2:
        height, width = dims
    else:
        return None
    if height <= 0 or width <= 0:
        return None
    return float(height), float(width)


def mask_valid_size_hw(mask: Any) -> Optional[Tuple[float, float]]:
    """Return the visible (height, width) from an RF-DETR padding mask."""
    if mask is None:
        return None
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None and torch.is_tensor(mask):
        mask_tensor = mask.detach().cpu().bool()
        if mask_tensor.ndim < 2:
            return None
        if mask_tensor.ndim > 2:
            mask_tensor = mask_tensor.reshape(-1, *mask_tensor.shape[-2:])[-1]
        valid = ~mask_tensor
        if not bool(valid.any()):
            return None
        ys, xs = valid.nonzero(as_tuple=True)
        return float(int(ys.max().item()) + 1), float(int(xs.max().item()) + 1)

    safe = json_safe_value(mask)
    if not isinstance(safe, list) or not safe:
        return None
    valid_height = 0
    valid_width = 0
    for y, row in enumerate(safe):
        if not isinstance(row, list):
            continue
        row_valid = [index for index, value in enumerate(row) if not bool(value)]
        if not row_valid:
            continue
        valid_height = y + 1
        valid_width = max(valid_width, max(row_valid) + 1)
    if valid_height <= 0 or valid_width <= 0:
        return None
    return float(valid_height), float(valid_width)


def batch_item_valid_size_hw(single_image: Any, mask: Any = None, target: Optional[Mapping[str, Any]] = None) -> Optional[Tuple[float, float]]:
    """Return the coordinate size that matches a rendered batch-grid image."""
    return mask_valid_size_hw(mask) or tensor_image_size_hw(single_image) or (
        size_hw(target.get("size")) if isinstance(target, Mapping) else None
    )


def validation_tensor_to_image(tensor: Any) -> Any:
    """Convert a normalized CHW model-input tensor into a PIL RGB image."""
    import numpy as np
    from PIL import Image

    array = tensor.detach().cpu().float().numpy() if hasattr(tensor, "detach") else np.asarray(tensor, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
        array = array[:3]
        if array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        if array.shape[0] == 3:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            array = array * std + mean
        array = np.clip(array, 0.0, 1.0)
        array = (array.transpose(1, 2, 0) * 255).astype(np.uint8)
    elif array.ndim == 2:
        array = np.clip(array, 0.0, 1.0)
        array = (np.repeat(array[:, :, None], 3, axis=2) * 255).astype(np.uint8)
    else:
        array = np.clip(array, 0.0, 1.0)
        if array.ndim == 3 and array.shape[-1] >= 3:
            array = (array[:, :, :3] * 255).astype(np.uint8)
        else:
            array = np.zeros((320, 320, 3), dtype=np.uint8)
    return Image.fromarray(array).convert("RGB")


def batch_grid_font() -> Any:
    """Load a small font for batch grid labels."""
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arial.ttf", 12)
    except Exception:
        return ImageFont.load_default()


def make_batch_grid_tile(image: Any) -> Tuple[Any, Any, Any, float, int, int]:
    """Create a fixed-size tile and return drawing primitives for annotation."""
    from PIL import Image, ImageDraw

    tile_size = BATCH_GRID_TILE_SIZE
    scale = min(tile_size / max(1, image.width), tile_size / max(1, image.height))
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    tile = Image.new("RGB", (tile_size, tile_size), color=(20, 20, 20))
    paste_x = (tile_size - resized.width) // 2
    paste_y = (tile_size - resized.height) // 2
    tile.paste(resized, (paste_x, paste_y))
    draw = ImageDraw.Draw(tile)
    font = batch_grid_font()
    return tile, draw, font, scale, paste_x, paste_y


def box_rows(value: Any) -> List[List[float]]:
    """Return a list of [x1, y1, x2, y2]-style box rows from tensor/list values."""
    safe = json_safe_value(value)
    if not isinstance(safe, list) or not safe:
        return []
    raw_rows = safe if isinstance(safe[0], list) else [safe[index : index + 4] for index in range(0, len(safe), 4)]
    rows: List[List[float]] = []
    for row in raw_rows:
        if not isinstance(row, list) or len(row) < 4:
            continue
        try:
            rows.append([float(item) for item in row[:4]])
        except (TypeError, ValueError):
            continue
    return rows


def labels_list(value: Any) -> List[int]:
    """Return integer labels from tensor/list values."""
    labels: List[int] = []
    for item in flatten_tensor(value):
        try:
            labels.append(int(item))
        except (TypeError, ValueError):
            continue
    return labels


def scores_list(value: Any) -> List[float]:
    """Return float scores from tensor/list values."""
    scores: List[float] = []
    for item in flatten_tensor(value):
        try:
            scores.append(float(item))
        except (TypeError, ValueError):
            continue
    return scores


def prediction_rows_at_min_score(
    result: Mapping[str, Any],
    min_score: Optional[float] = VALIDATION_PREDICTION_GRID_MIN_SCORE,
) -> Tuple[List[List[float]], List[int], List[float]]:
    """Return prediction boxes, labels, and scores whose score passes the grid threshold."""
    boxes = box_rows(result.get("boxes", []))
    labels = labels_list(result.get("labels", []))
    scores = scores_list(result.get("scores", []))
    if min_score is None:
        return boxes, labels, scores

    threshold = float(min_score)
    kept_boxes: List[List[float]] = []
    kept_labels: List[int] = []
    kept_scores: List[float] = []
    for index, box in enumerate(boxes):
        if index >= len(scores):
            continue
        score = float(scores[index])
        if score < threshold:
            continue
        kept_boxes.append(box)
        kept_labels.append(int(labels[index]) if index < len(labels) else 0)
        kept_scores.append(score)
    return kept_boxes, kept_labels, kept_scores


def normalized_cxcywh_to_xyxy(
    box: Sequence[float],
    valid_size: Optional[Tuple[float, float]],
) -> Optional[List[float]]:
    """Convert RF-DETR normalized cxcywh target boxes to xyxy pixels."""
    if len(box) < 4:
        return None
    valid_h, valid_w = valid_size or (1.0, 1.0)
    cx, cy, width, height = [float(item) for item in box[:4]]
    if max(abs(cx), abs(cy), abs(width), abs(height)) <= 2.0:
        x1 = (cx - width / 2.0) * valid_w
        y1 = (cy - height / 2.0) * valid_h
        x2 = (cx + width / 2.0) * valid_w
        y2 = (cy + height / 2.0) * valid_h
    else:
        x1, y1, x2, y2 = cx, cy, width, height
    x1, x2 = sorted((max(0.0, min(valid_w, x1)), max(0.0, min(valid_w, x2))))
    y1, y2 = sorted((max(0.0, min(valid_h, y1)), max(0.0, min(valid_h, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def target_boxes_xyxy(target: Mapping[str, Any]) -> List[List[float]]:
    """Return RF-DETR target boxes as xyxy pixels in target.size coordinates."""
    valid_size = size_hw(target.get("size")) or size_hw(target.get("orig_size"))
    boxes: List[List[float]] = []
    for box in box_rows(target.get("boxes", [])):
        converted = normalized_cxcywh_to_xyxy(box, valid_size)
        if converted is not None:
            boxes.append(converted)
    return boxes


def draw_boxes_on_tile(
    draw: Any,
    font: Any,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Optional[Sequence[float]],
    source_size: Optional[Tuple[float, float]],
    valid_size: Optional[Tuple[float, float]],
    scale: float,
    paste_x: int,
    paste_y: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw boxes in source coordinates onto a model-input tile."""
    source_h, source_w = source_size or valid_size or (1.0, 1.0)
    valid_h, valid_w = valid_size or source_size or (source_h, source_w)
    x_to_image = valid_w / max(1.0, source_w)
    y_to_image = valid_h / max(1.0, source_h)

    for index, box in enumerate(boxes[:100]):
        if len(box) < 4:
            continue
        label = int(labels[index]) if index < len(labels) else 0
        x1 = float(box[0]) * x_to_image * scale + paste_x
        y1 = float(box[1]) * y_to_image * scale + paste_y
        x2 = float(box[2]) * x_to_image * scale + paste_x
        y2 = float(box[3]) * y_to_image * scale + paste_y
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if scores is None:
            text = str(label)
        else:
            score = float(scores[index]) if index < len(scores) else 0.0
            text = f"{label} {score:.2f}"
        text_bbox = draw.textbbox((x1 + 2, y1 + 2), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 255), font=font)


def draw_validation_predictions_on_tile(
    image: Any,
    result: Mapping[str, Any],
    source_size: Optional[Tuple[float, float]],
    valid_size: Optional[Tuple[float, float]],
    min_score: float = VALIDATION_PREDICTION_GRID_MIN_SCORE,
) -> Any:
    """Draw one validation prediction result onto a fixed-size tile."""
    tile, draw, font, scale, paste_x, paste_y = make_batch_grid_tile(image)
    boxes, labels, scores = prediction_rows_at_min_score(result, min_score)

    draw_boxes_on_tile(
        draw=draw,
        font=font,
        boxes=boxes,
        labels=labels,
        scores=scores,
        source_size=source_size,
        valid_size=valid_size,
        scale=scale,
        paste_x=paste_x,
        paste_y=paste_y,
        color=(239, 68, 68),
    )
    return tile


def draw_target_labels_on_tile(
    image: Any,
    target: Mapping[str, Any],
    valid_size: Optional[Tuple[float, float]] = None,
    color: Tuple[int, int, int] = (37, 99, 235),
) -> Any:
    """Draw one RF-DETR target label set onto a fixed-size tile."""
    tile, draw, font, scale, paste_x, paste_y = make_batch_grid_tile(image)
    valid_size = valid_size or size_hw(target.get("size")) or size_hw(target.get("orig_size")) or (float(image.height), float(image.width))
    boxes: List[List[float]] = []
    for box in box_rows(target.get("boxes", [])):
        converted = normalized_cxcywh_to_xyxy(box, valid_size)
        if converted is not None:
            boxes.append(converted)
    draw_boxes_on_tile(
        draw=draw,
        font=font,
        boxes=boxes,
        labels=labels_list(target.get("labels", [])),
        scores=None,
        source_size=valid_size,
        valid_size=valid_size,
        scale=scale,
        paste_x=paste_x,
        paste_y=paste_y,
        color=color,
    )
    return tile


def make_batch_grid_canvas() -> Tuple[Any, Any]:
    """Create a 3x3 batch grid canvas."""
    from PIL import Image, ImageDraw

    row_height = BATCH_GRID_TILE_SIZE + BATCH_GRID_CAPTION_HEIGHT
    canvas = Image.new(
        "RGB",
        (BATCH_GRID_COLUMNS * BATCH_GRID_TILE_SIZE, BATCH_GRID_COLUMNS * row_height),
        color=(245, 245, 245),
    )
    return canvas, ImageDraw.Draw(canvas)


def save_batch_label_grid(batch: Any, output_dir: Path, path_name: str, caption_prefix: str) -> Path:
    """Render one batch's target labels and overwrite a fixed output path."""
    samples, targets = batch
    tensors = getattr(samples, "tensors", samples)
    masks = getattr(samples, "mask", None)
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas, draw = make_batch_grid_canvas()
    row_height = BATCH_GRID_TILE_SIZE + BATCH_GRID_CAPTION_HEIGHT
    for sample_index, (single_image, target) in enumerate(zip(tensors[:BATCH_GRID_MAX_ITEMS], targets[:BATCH_GRID_MAX_ITEMS])):
        col = sample_index % BATCH_GRID_COLUMNS
        row = sample_index // BATCH_GRID_COLUMNS
        x0 = col * BATCH_GRID_TILE_SIZE
        y0 = row * row_height
        image = validation_tensor_to_image(single_image)
        mask = masks[sample_index] if masks is not None and sample_index < len(masks) else None
        valid_size = batch_item_valid_size_hw(single_image, mask, target if isinstance(target, Mapping) else None)
        tile = draw_target_labels_on_tile(image, target if isinstance(target, Mapping) else {}, valid_size=valid_size)
        canvas.paste(tile, (x0, y0))
        draw.text((x0 + 6, y0 + BATCH_GRID_TILE_SIZE + 6), f"{caption_prefix} item={sample_index}", fill=(20, 20, 20))
    path = output_dir / path_name
    canvas.save(path, quality=92)
    return path


def save_train_batch_grid(batch: Any, output_dir: Path, batch_idx: int) -> Path:
    """Render one training batch label grid."""
    return save_batch_label_grid(batch, output_dir, f"train_batch{batch_idx}.jpg", f"train batch={batch_idx}")


def save_validation_label_grid(batch: Any, output_dir: Path, batch_idx: int) -> Path:
    """Render one validation batch label grid."""
    return save_batch_label_grid(batch, output_dir, f"val_batch{batch_idx}_labels.jpg", f"val labels batch={batch_idx}")


def save_validation_prediction_grid(batch: Any, outputs: Mapping[str, Any], output_dir: Path, batch_idx: int) -> Optional[Path]:
    """Render one validation batch prediction grid and overwrite its fixed output path."""
    results = outputs.get("results") if isinstance(outputs, Mapping) else None
    if results is None:
        return None
    samples, targets = batch
    tensors = getattr(samples, "tensors", samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas, draw = make_batch_grid_canvas()
    row_height = BATCH_GRID_TILE_SIZE + BATCH_GRID_CAPTION_HEIGHT
    for sample_index, (single_image, result, target) in enumerate(zip(tensors[:BATCH_GRID_MAX_ITEMS], results[:BATCH_GRID_MAX_ITEMS], targets[:BATCH_GRID_MAX_ITEMS])):
        col = sample_index % BATCH_GRID_COLUMNS
        row = sample_index // BATCH_GRID_COLUMNS
        x0 = col * BATCH_GRID_TILE_SIZE
        y0 = row * row_height
        image = validation_tensor_to_image(single_image)
        source_size = size_hw(target.get("orig_size")) if isinstance(target, Mapping) else None
        valid_size = size_hw(target.get("size")) if isinstance(target, Mapping) else None
        tile = draw_validation_predictions_on_tile(image, result, source_size, valid_size)
        canvas.paste(tile, (x0, y0))
        draw.text((x0 + 6, y0 + BATCH_GRID_TILE_SIZE + 6), f"val pred batch={batch_idx} item={sample_index}", fill=(20, 20, 20))
    path = output_dir / f"val_batch{batch_idx}_pred.jpg"
    canvas.save(path, quality=92)
    return path


class TrainBatchGridCallback(Callback):
    """Save the first few training batch label grids for label/augmentation checks."""

    def __init__(self, output_dir: Path, verbose: bool, max_batches: int = TRAIN_BATCH_GRID_MAX_BATCHES) -> None:
        self.output_dir = output_dir
        self.verbose = verbose
        self.max_batches = max(0, int(max_batches))
        self.saved_indices: set[int] = set()
        self.error_reported = False

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs
        if batch_idx >= self.max_batches or batch_idx in self.saved_indices:
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        try:
            path = save_train_batch_grid(batch, self.output_dir, batch_idx)
        except Exception as exc:
            if not self.error_reported:
                blue(f"Warning: training batch grids could not be saved; training will continue. {exc}", force=True)
                self.error_reported = True
            return
        self.saved_indices.add(batch_idx)
        blue(f"Saved training batch grid image: {path}", verbose=self.verbose)


class ValidationPredictionGridCallback(Callback):
    """Refresh top-level validation label and prediction grid images after each validation."""

    def __init__(self, output_dir: Path, verbose: bool, max_batches: int = VALIDATION_PREDICTION_GRID_MAX_BATCHES) -> None:
        self.output_dir = output_dir
        self.verbose = verbose
        self.max_batches = max(0, int(max_batches))
        self.saved_paths: List[Path] = []
        self.error_reported = False

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        self.saved_paths = []
        if bool(getattr(trainer, "sanity_checking", False)):
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        for index in range(self.max_batches):
            for suffix in ("labels", "pred"):
                with contextlib.suppress(FileNotFoundError):
                    (self.output_dir / f"val_batch{index}_{suffix}.jpg").unlink()

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module
        if dataloader_idx != 0 or batch_idx >= self.max_batches:
            return
        if bool(getattr(trainer, "sanity_checking", False)):
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        try:
            label_path = save_validation_label_grid(batch, self.output_dir, batch_idx)
            pred_path = save_validation_prediction_grid(batch, outputs, self.output_dir, batch_idx)
        except Exception as exc:
            if not self.error_reported:
                blue(f"Warning: validation batch grids could not be saved; training will continue. {exc}", force=True)
                self.error_reported = True
            return
        self.saved_paths.append(label_path)
        if pred_path is not None:
            self.saved_paths.append(pred_path)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        if self.saved_paths:
            blue(
                f"Updated {len(self.saved_paths)} validation batch grid image(s).",
                verbose=self.verbose,
                force=True,
            )


def parse_device_to_trainer_kwargs(device: Optional[str]) -> Dict[str, Any]:
    """Parse CPU/GPU device strings into PyTorch Lightning kwargs."""
    if device is None or str(device).strip().lower() in {"", "auto"}:
        return {}
    text = str(device).strip().lower()
    if text == "cpu":
        return {"accelerator": "cpu"}
    if text in {"cuda", "gpu"}:
        return {"accelerator": "gpu"}
    if text == "mps":
        return {"accelerator": "mps"}
    if text == "-1":
        return {"accelerator": "auto", "devices": "auto"}
    if "," in text:
        ids = [str(int(part.strip())) for part in text.split(",") if part.strip()]
        return {"accelerator": "gpu", "devices": ",".join(ids)}
    if text.isdigit():
        return {"accelerator": "gpu", "devices": f"{int(text)},"}
    if text.startswith("cuda:") and text.split(":", 1)[1].isdigit():
        return {"accelerator": "gpu", "devices": f"{int(text.split(':', 1)[1])},"}
    return {"accelerator": text}


def uses_multiple_gpu_devices(trainer_kwargs: Mapping[str, Any]) -> bool:
    """Return True when resolved Trainer kwargs request more than one CUDA device."""
    accelerator = str(trainer_kwargs.get("accelerator", "")).strip().lower()
    if accelerator != "gpu":
        return False
    devices = trainer_kwargs.get("devices")
    if isinstance(devices, int):
        return devices > 1
    if isinstance(devices, str):
        text = devices.strip().lower()
        if text in {"", "auto"}:
            return False
        if "," in text:
            return len([item for item in text.split(",") if item.strip()]) > 1
        with contextlib.suppress(ValueError):
            return int(text) > 1
        return False
    if isinstance(devices, Sequence) and not isinstance(devices, (str, bytes, bytearray)):
        return len(devices) > 1
    return False


def strategy_allows_find_unused_override(value: Any) -> bool:
    """Return True for strategy settings where the wrapper may select RF-DETR-safe DDP."""
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"", "auto", "ddp"}


def apply_multigpu_ddp_strategy(config: Mapping[str, Any], trainer_kwargs: MutableMapping[str, Any], verbose: bool) -> None:
    """Use DDP unused-parameter detection for RF-DETR multi-GPU training unless explicitly overridden."""
    explicit_trainer_strategy = trainer_kwargs.get("strategy")
    train_strategy = config.get("train", {}).get("strategy", "auto")
    if explicit_trainer_strategy is not None:
        if not strategy_allows_find_unused_override(explicit_trainer_strategy):
            return
    elif not strategy_allows_find_unused_override(train_strategy):
        return
    if not uses_multiple_gpu_devices(trainer_kwargs):
        return
    trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
    blue("Multi-GPU RF-DETR training uses strategy=ddp_find_unused_parameters_true.", verbose)


def apply_multigpu_validation_safety(trainer_kwargs: MutableMapping[str, Any], verbose: bool) -> None:
    """Avoid RF-DETR DDP hangs in Lightning's initial validation sanity check."""
    if not uses_multiple_gpu_devices(trainer_kwargs):
        return
    if "num_sanity_val_steps" in trainer_kwargs:
        return
    trainer_kwargs["num_sanity_val_steps"] = 0
    blue(
        "Multi-GPU RF-DETR training disables Lightning sanity validation "
        "(num_sanity_val_steps=0) to avoid distributed mAP sync timeouts before training.",
        verbose,
    )


def apply_validation_interval_to_trainer_kwargs(
    config: Mapping[str, Any],
    trainer_kwargs: MutableMapping[str, Any],
    verbose: bool,
) -> None:
    """Make train.eval_interval control Lightning's validation loop frequency."""
    if "check_val_every_n_epoch" in trainer_kwargs:
        return
    raw_interval = config.get("train", {}).get("eval_interval", 1)
    try:
        interval = max(1, int(raw_interval or 1))
    except (TypeError, ValueError):
        return
    trainer_kwargs["check_val_every_n_epoch"] = interval
    if interval > 1:
        blue(f"Lightning validation loader will run every {interval} epoch(s).", verbose)


def flatten_tensor(value: Any) -> List[Any]:
    """Return a 1-D Python list from a scalar/list/tensor-like value."""
    safe = json_safe_value(value)
    if isinstance(safe, list):
        if safe and isinstance(safe[0], list):
            return [item for sub in safe for item in (sub if isinstance(sub, list) else [sub])]
        return safe
    if safe is None:
        return []
    return [safe]


def scalar_metric(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read one scalar metric from torchmetrics output."""
    value = json_safe_value(metrics.get(key, default))
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def cat_id_to_name_from_dataset(dataset: Any, datamodule: Any) -> Dict[int, str]:
    """Build an O(1) class-id to class-name lookup table."""
    coco = getattr(dataset, "coco", None)
    if coco is not None and hasattr(coco, "cats"):
        if hasattr(coco, "label2cat"):
            return {int(label): str(coco.cats[cat_id]["name"]) for label, cat_id in coco.label2cat.items()}
        return {int(key): str(value["name"]) for key, value in coco.cats.items()}
    names = getattr(datamodule, "class_names", None) or []
    return {index: str(name) for index, name in enumerate(names)}


def build_eval_dataloader(datamodule: Any, model_config: Any, train_config: Any, split: str) -> Tuple[Any, Any]:
    """Build an evaluation dataloader for a specific split."""
    import torch
    from torch.utils.data import DataLoader

    from rfdetr._namespace import build_namespace
    from rfdetr.datasets import build_dataset
    from rfdetr.utilities.tensors import collate_fn

    dataset = build_dataset(split, build_namespace(model_config, train_config), model_config.resolution)
    num_workers = int(getattr(train_config, "num_workers", 2) or 2)
    pin_memory = getattr(datamodule, "_pin_memory", False)
    persistent_workers = bool(num_workers > 0 and getattr(datamodule, "_persistent_workers", False))
    prefetch_factor = getattr(datamodule, "_prefetch_factor", None) if num_workers > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(train_config, "batch_size", 1) or 1),
        sampler=torch.utils.data.SequentialSampler(dataset),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return dataset, loader


class SimpleDetections:
    """Small detection container matching the evaluator's expected attributes."""

    def __init__(self, xyxy: Any, confidence: Any, class_id: Any) -> None:
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id


class RFDETRLightningPredictor:
    """Image-level predictor backed by a live RF-DETR Lightning module."""

    def __init__(self, pl_module: Any, model_config: Any) -> None:
        self.pl_module = pl_module
        self.model_config = model_config

    def predict(
        self,
        images: Any,
        threshold: float = 0.5,
        shape: Optional[Tuple[int, int]] = None,
        **_: Any,
    ) -> Any:
        """Predict boxes for PIL/path/numpy/tensor inputs."""
        import numpy as np
        import torch
        import torchvision.transforms.functional as F
        from PIL import Image

        single = not isinstance(images, list)
        image_list = [images] if single else images
        pil_images = []
        for image in image_list:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, Path):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            pil_images.append(image.convert("RGB") if hasattr(image, "convert") else image)

        resize_to = shape or (int(getattr(self.model_config, "resolution", 560)), int(getattr(self.model_config, "resolution", 560)))
        tensors = []
        orig_sizes = []
        for image in pil_images:
            tensor = F.to_tensor(image)
            h, w = tensor.shape[1:]
            orig_sizes.append((h, w))
            tensor = F.resize(tensor, list(resize_to))
            tensor = F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            tensors.append(tensor)

        was_training = bool(self.pl_module.training)
        self.pl_module.eval()
        device = self.pl_module.device
        batch = torch.stack(tensors).to(device)
        target_sizes = torch.tensor(orig_sizes, device=device)
        with torch.no_grad():
            outputs = self.pl_module.model(batch)
            results = self.pl_module.postprocess(outputs, target_sizes)
        if was_training:
            self.pl_module.train()

        detections = []
        for result in results:
            scores = result["scores"].detach().cpu().numpy()
            labels = result["labels"].detach().cpu().numpy()
            boxes = result["boxes"].detach().cpu().numpy()
            keep = scores > float(threshold)
            detections.append(SimpleDetections(xyxy=boxes[keep], confidence=scores[keep], class_id=labels[keep]))
        return detections[0] if single else detections


def periodic_test_mode(periodic: Mapping[str, Any]) -> str:
    """Return configured RF-DETR test mode."""
    mode_cfg = periodic.get("test_mode", {"mode": periodic.get("mode", "full_image")})
    return shared_modes.canonical_test_mode({"test_mode": mode_cfg})


def normalize_rfdetr_test_settings(merged_config: Mapping[str, Any], section: str = "periodic_test") -> Dict[str, Any]:
    """Normalize periodic or standalone RF-DETR test settings into one shape."""
    source = dict(merged_config.get(section, {}) or {})
    model = merged_config.get("model", {})
    dataset = merged_config.get("dataset", {})
    evaluation = merged_config.get("evaluation", {})
    mode = periodic_test_mode(source)
    split = str(source.get("split") or dataset.get("split") or "test")
    max_dets = source.get("max_dets")
    if max_dets is None:
        max_values = evaluation.get("max_detections")
        if isinstance(max_values, (list, tuple)) and max_values:
            max_dets = max_values[-1]
        elif max_values is not None:
            max_dets = max_values
    max_images = parse_limit_value(source.get("max_images"), f"{section}.max_images")
    return {
        **source,
        "split": split,
        "max_images": max_images,
        "test_mode": {"mode": mode},
        "crop": dict(source.get("crop", {}) or {}),
        "sahi": dict(source.get("sahi", {}) or {}),
        "classwise": bool(source.get("classwise", evaluation.get("classwise", True))),
        "progress_bar": bool(source.get("progress_bar", True)),
        "plots": bool(source.get("plots", False)),
        "save_dataset_cases": bool(source.get("save_dataset_cases", False)),
        "visual_samples": dict(source.get("visual_samples", {}) or {}),
        "model_input_batch_size": int(source.get("model_input_batch_size", 9) or 9),
        "batch_size": int(source.get("batch_size", 4) or 4),
        "error_cases": dict(source.get("error_cases", {}) or {}),
        "conf": source.get("conf", model.get("confidence_threshold", 0.25)),
        "match_iou_threshold": source.get("match_iou_threshold", evaluation.get("match_iou_threshold", 0.5)),
        "max_dets": int(max_dets) if max_dets is not None else None,
    }


def resolve_eval_dataset_config(train_config: Any, split: str) -> Dict[str, Any]:
    """Build evaluator dataset config from an RF-DETR-readable dataset directory."""
    dataset_dir = Path(str(getattr(train_config, "dataset_dir"))).expanduser()
    split_aliases = {
        "val": ["valid", "val"],
        "valid": ["valid", "val"],
        "validation": ["valid", "val"],
        "test": ["test"],
        "test-original": ["test-original", "test_original"],
        "test_original": ["test-original", "test_original"],
        "train": ["train"],
    }
    for candidate_split in split_aliases.get(split.lower(), [split]):
        split_dir = dataset_dir / candidate_split
        coco_json = split_dir / "_annotations.coco.json"
        if coco_json.exists():
            return {"format": "coco", "coco_json": str(coco_json), "image_dir": str(split_dir), "split": split}
    for name in DATA_YAML_NAMES:
        data_yaml = dataset_dir / name
        if data_yaml.exists():
            return {"format": "yolo", "data_yaml": str(data_yaml), "split": "val" if split in {"valid", "validation"} else split}
    raise FileNotFoundError(f"Could not find COCO annotations or data.yaml for RF-DETR split={split!r} in {dataset_dir}.")


def normalize_category_remapping(value: Mapping[Any, Any]) -> Dict[int, int]:
    """Normalize model label to dataset category-id remapping keys and values."""
    return {int(key): int(remapped) for key, remapped in value.items()}


def infer_rfdetr_category_remapping(merged_config: Mapping[str, Any], dataset_cfg: Mapping[str, Any]) -> Dict[int, int]:
    """Map RF-DETR contiguous output labels back to evaluator dataset category ids."""
    explicit = merged_config.get("model", {}).get("category_remapping")
    if explicit is False:
        return {}
    if isinstance(explicit, Mapping) and explicit:
        return normalize_category_remapping(explicit)
    if str(dataset_cfg.get("format", "")).strip().lower() != "coco":
        return {}

    coco_json = dataset_cfg.get("coco_json")
    if not coco_json:
        return {}
    data = json.loads(Path(str(coco_json)).read_text(encoding="utf-8"))
    categories = data.get("categories", []) if isinstance(data, Mapping) else []
    category_ids = sorted({int(category["id"]) for category in categories if "id" in category})
    return {label: category_id for label, category_id in enumerate(category_ids)}


def build_rfdetr_evaluator_config(
    merged_config: Mapping[str, Any],
    model_config: Any,
    train_config: Any,
    output_dir: Path,
    split: str,
    test_section: str = "periodic_test",
) -> Dict[str, Any]:
    """Build shared evaluator config for RF-DETR tests."""
    test_settings = normalize_rfdetr_test_settings(merged_config, test_section)
    mode = periodic_test_mode(test_settings)
    max_dets_value = test_settings.get("max_dets")
    if max_dets_value is None:
        max_dets_value = getattr(train_config, "eval_max_dets", 500)
    max_dets = int(max_dets_value or 500)
    conf = float(test_settings.get("conf", merged_config.get("model", {}).get("confidence_threshold", 0.25)) or 0.25)
    resolution = int(getattr(model_config, "resolution", merged_config.get("model", {}).get("resolution", 560)) or 560)
    sahi_cfg = {
        "slice_height": int(test_settings.get("slice_height", resolution)),
        "slice_width": int(test_settings.get("slice_width", resolution)),
        "overlap_height_ratio": float(test_settings.get("overlap_height_ratio", 0.2)),
        "overlap_width_ratio": float(test_settings.get("overlap_width_ratio", 0.2)),
        "standard_prediction": bool(test_settings.get("standard_prediction", True)),
        "postprocess_type": str(test_settings.get("postprocess_type", "GREEDYNMM")),
        "postprocess_match_metric": str(test_settings.get("postprocess_match_metric", "IOS")),
        "postprocess_match_threshold": float(test_settings.get("postprocess_match_threshold", 0.5)),
        "postprocess_class_agnostic": bool(test_settings.get("postprocess_class_agnostic", False)),
        "batch_size": int(test_settings.get("batch", test_settings.get("batch_size", 4)) or 4),
    }
    sahi_cfg.update(test_settings.get("sahi", {}) or {})
    dataset_cfg = resolve_eval_dataset_config(train_config, split)
    dataset_cfg.update({"include_empty_images": True, "sort_images": True, "max_images": test_settings.get("max_images")})
    category_remapping = infer_rfdetr_category_remapping(merged_config, dataset_cfg)
    visual_cfg = dict(test_settings.get("visual_samples", {}) or {})
    visual_max_images = visual_cfg.get("max_images")
    error_cases_cfg = dict(test_settings.get("error_cases", {}) or {})
    if visual_max_images is not None and error_cases_cfg.get("max_images") is None:
        error_cases_cfg["max_images"] = int(visual_max_images)
    if visual_max_images is not None:
        visual_cfg["max_images"] = int(visual_max_images)
    return {
        "runtime": {
            "verbose": bool(merged_config.get("runtime", {}).get("verbose", True)),
            "quiet": False,
            "confirm_before_run": False,
            "yes": True,
            "banner": "RF-DETR TEST EVALUATION",
            "seed": int(merged_config.get("runtime", {}).get("seed", 0) or 0),
        },
        "inference": {"mode": mode, "use_sahi": mode == shared_modes.SAHI_MODE, "batch_size": int(test_settings.get("batch_size", 4) or 4)},
        "test_mode": {"mode": mode},
        "dataset": dataset_cfg,
        "model": {
            "type": "rfdetr",
            "path": merged_config.get("model", {}).get("pretrain_weights", ""),
            "size": merged_config.get("model", {}).get("size", "medium"),
            "confidence_threshold": conf,
            "device": merged_config.get("train", {}).get("device", "cpu"),
            "image_size": resolution,
            "category_remapping": category_remapping,
            "inference_optimization": deepcopy(
                dict(merged_config.get("model", {}).get("inference_optimization", {}) or {})
            ),
        },
        "sahi": sahi_cfg,
        "crop": test_settings.get("crop", {}) or {},
        "evaluation": {
            "type": "bbox",
            "max_detections": [1, 10, max_dets],
            "match_iou_threshold": float(test_settings.get("match_iou_threshold", 0.5)),
            "operating_confidence_threshold": conf,
            "classwise": bool(test_settings.get("classwise", True)),
            "per_image_metrics": True,
            "confusion_matrix": False,
            "curves": False,
            "save_coco_summary_text": True,
        },
        "output": {
            "output_dir": str(output_dir),
            "exist_ok": True,
            "save_config": True,
            "save_predictions_json": True,
            "save_ground_truth_json": True,
            "save_metrics": True,
            "save_plots": bool(test_settings.get("plots", False)),
            "save_visuals": bool(visual_cfg.get("enabled", False)),
            "max_visuals": visual_cfg.get("max_images"),
            "visual_output_subdir": visual_cfg.get("output_subdir", "visuals"),
            "visual_sampling_mode": visual_cfg.get("sampling_mode", "first"),
            "visual_random_seed": visual_cfg.get("random_seed", merged_config.get("runtime", {}).get("seed", 0)),
            "visual_filter_source": visual_cfg.get("filter_source", "ground_truth"),
            "visual_filter_match": visual_cfg.get("filter_match", "any"),
            "visual_filter_class_ids": visual_cfg.get("class_ids", []),
            "visual_filter_class_names": visual_cfg.get("class_names", []),
            "visual_render_class_ids": visual_cfg.get("render_class_ids", []),
            "visual_render_class_names": visual_cfg.get("render_class_names", []),
            "visual_min_gt_instances": visual_cfg.get("min_gt_instances", 0),
            "visual_min_predictions": visual_cfg.get("min_predictions", 0),
            "visual_filter_min_score": visual_cfg.get("filter_min_score"),
            "visual_draw_min_score": visual_cfg.get("draw_min_score"),
            "save_dataset_cases": bool(test_settings.get("save_dataset_cases", False)),
            "error_cases": error_cases_cfg,
            "save_model_input_batches": True,
            "max_model_input_batches": 3,
            "model_input_batch_size": int(test_settings.get("model_input_batch_size", 9) or 9),
        },
        "progress": {
            "images": bool(test_settings.get("progress_bar", True)),
            "slices": False,
            "dataset_cases": False,
            "visuals": False,
            "error_cases": bool(test_settings.get("progress_bar", True)),
        },
    }


def write_rfdetr_evaluator_aliases(output_dir: Path, result: Mapping[str, Any]) -> None:
    """Write RF-DETR legacy metric file aliases from evaluator results."""
    summary = dict(result.get("summary", {}))
    metric_rows = [{"metric": key, "value": value} for key, value in summary.items()]
    write_json(output_dir / "test_metrics.json", {"overall": summary, "metrics": summary})
    write_rows(output_dir / "test_metrics.csv", metric_rows, METRIC_FIELDS)
    per_class = list(result.get("per_class", []) or [])
    if per_class:
        fields = sorted({key for row in per_class for key in row.keys()})
        write_json(output_dir / "test_per_class_metrics.json", per_class)
        write_rows(output_dir / "test_per_class_metrics.csv", per_class, fields)


def run_rfdetr_shared_evaluation(
    pl_module: Any,
    model_config: Any,
    train_config: Any,
    output_dir: Path,
    split: str,
    event: str,
    metadata: Mapping[str, Any],
    merged_config: Mapping[str, Any],
    source_config: Optional[Path],
    verbose: bool,
) -> Dict[str, Any]:
    """Run RF-DETR test through the shared image-level evaluator."""
    from projects.object_detection_dataset_evaluator.object_detection_dataset_evaluator import run_evaluation

    evaluator_config = build_rfdetr_evaluator_config(merged_config, model_config, train_config, output_dir, split)
    predictor = RFDETRLightningPredictor(pl_module, model_config)
    result = run_evaluation(evaluator_config, source_config or DEFAULT_CONFIG, prebuilt_model=predictor, print_summary=False)
    write_rfdetr_evaluator_aliases(output_dir, result)
    dump_config_snapshot(
        output_dir=output_dir,
        merged_config=merged_config,
        metadata={**dict(metadata), "event": event, "split": split, "output_dir": str(output_dir), "shared_evaluator": True},
        source_config=source_config,
        train_config=train_config,
        model_config=model_config,
    )
    blue(f"{event} {periodic_test_mode(merged_config.get('periodic_test', {}))} metrics saved to {output_dir}.", verbose)
    return {
        "event": event,
        "split": split,
        "metadata": dict(metadata),
        "overall": result.get("summary", {}),
        "per_class": result.get("per_class", []),
        "output_dir": str(output_dir),
    }


def compute_f1_by_class(f1_local: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Compute macro F1/precision/recall and per-class values from matching data."""
    import numpy as np

    from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds

    if not f1_local:
        return {"F1": 0.0, "Precision": 0.0, "Recall": 0.0}, {}
    sorted_ids = sorted(f1_local.keys())
    per_class_list = [f1_local[cid] for cid in sorted_ids]
    classes_with_gt = [i for i, cid in enumerate(sorted_ids) if f1_local[cid]["total_gt"] > 0]
    if not classes_with_gt:
        return {"F1": 0.0, "Precision": 0.0, "Recall": 0.0}, {}
    best = max(sweep_confidence_thresholds(per_class_list, np.linspace(0, 1, 101), classes_with_gt), key=lambda row: row["macro_f1"])
    overall = {
        "F1": float(best["macro_f1"]),
        "Precision": float(best["macro_precision"]),
        "Recall": float(best["macro_recall"]),
    }
    per_class: Dict[int, Dict[str, float]] = {}
    for index, class_id in enumerate(sorted_ids):
        per_class[class_id] = {
            "f1": float(best["per_class_f1"][index]),
            "precision": float(best["per_class_prec"][index]),
            "recall": float(best["per_class_rec"][index]),
        }
    return overall, per_class


def manual_test_evaluation(
    trainer: Any,
    pl_module: Any,
    datamodule: Any,
    model_config: Any,
    train_config: Any,
    output_dir: Path,
    split: str,
    event: str,
    metadata: Mapping[str, Any],
    merged_config: Mapping[str, Any],
    source_config: Optional[Path],
    verbose: bool,
    progress_bar: bool,
    inference_runtime: Any = None,
) -> Dict[str, Any]:
    """Run a single-process RF-DETR test evaluation and write metrics.

    ``inference_runtime`` is reserved for the standalone test entrypoint.  When
    supplied, raw batches run through the selected FP32/BF16/TensorRT backend;
    scheduled and final training evaluations keep the original Lightning path.
    """
    periodic = merged_config.get("periodic_test", {})
    segmentation = bool(getattr(model_config, "segmentation_head", False))
    mode = periodic_test_mode(periodic)
    eval_type = str(merged_config.get("evaluation", {}).get("type", "auto")).strip().lower()
    wants_segmentation_metrics = eval_type in {"auto", "segm", "segment", "segmentation", "mask", "masks"}
    use_shared_evaluator = not bool(periodic.get("legacy_manual_eval", False))
    if segmentation and mode == shared_modes.FULL_IMAGE_MODE and wants_segmentation_metrics:
        use_shared_evaluator = False
    if use_shared_evaluator:
        output_dir.mkdir(parents=True, exist_ok=True)
        return run_rfdetr_shared_evaluation(
            pl_module=pl_module,
            model_config=model_config,
            train_config=train_config,
            output_dir=output_dir,
            split=split,
            event=event,
            metadata=metadata,
            merged_config=merged_config,
            source_config=source_config,
            verbose=verbose,
        )

    import torch
    from torchmetrics.detection import MeanAveragePrecision

    from rfdetr.evaluation.matching import build_matching_data, init_matching_accumulator, merge_matching_data
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

    output_dir.mkdir(parents=True, exist_ok=True)
    blue(f"Running {event} evaluation on split={split}.", verbose)
    dataset, loader = build_eval_dataloader(datamodule, model_config, train_config, split)
    cat_id_to_name = cat_id_to_name_from_dataset(dataset, datamodule)
    max_dets = int(getattr(train_config, "eval_max_dets", 500) or 500)
    iou_type: Any = ["bbox", "segm"] if segmentation else "bbox"
    map_metric = MeanAveragePrecision(
        iou_type=iou_type,
        class_metrics=True,
        max_detection_thresholds=[1, 10, max_dets],
        backend="faster_coco_eval",
    )
    converter = COCOEvalCallback(max_dets=max_dets, segmentation=segmentation, eval_interval=1, log_per_class_metrics=True)
    f1_local = init_matching_accumulator()
    using_inference_runtime = inference_runtime is not None
    if using_inference_runtime:
        was_training = False
        device = getattr(inference_runtime, "device", None)
        if device is None:
            raise RuntimeError("Standalone inference runtime did not expose its execution device.")
    else:
        if pl_module is None:
            raise ValueError("pl_module is required when inference_runtime is not provided.")
        was_training = bool(pl_module.training)
        pl_module.eval()
        device = getattr(pl_module, "device", None)
    if device is not None:
        map_metric = map_metric.to(device)

    iterable: Iterable[Any] = loader
    if progress_bar:
        iterable = tqdm(loader, desc=f"{event} {split}", leave=False)
    standalone_forward_seconds = 0.0
    standalone_preprocess_seconds = 0.0
    standalone_postprocess_seconds = 0.0
    standalone_image_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterable):
            runtime_batch_started = time.perf_counter() if using_inference_runtime else 0.0
            runtime_preprocess_seconds = 0.0
            runtime_forward_seconds = 0.0
            if using_inference_runtime:
                samples, targets = batch
                samples = samples.to(device)
                targets = [
                    {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
                    for target in targets
                ]
                if str(getattr(inference_runtime, "backend", "pytorch")) == "tensorrt":
                    tensors = samples.tensors
                    expected_resolution = int(getattr(model_config, "resolution"))
                    if tensors.shape[-2:] != (expected_resolution, expected_resolution):
                        raise ValueError(
                            "TensorRT segmentation evaluation requires fixed square inputs at the model "
                            f"resolution {expected_resolution}, got {tuple(tensors.shape[-2:])}. "
                            "Keep square_resize_div_64 enabled."
                        )
                    if samples.mask is not None and bool(samples.mask.any().item()):
                        raise ValueError(
                            "TensorRT segmentation evaluation received padded/non-square samples. "
                            "Keep square_resize_div_64 enabled."
                        )
                consume_forward = getattr(inference_runtime, "consume_forward_seconds", None)
                consume_postprocess = getattr(inference_runtime, "consume_postprocess_seconds", None)
                if callable(consume_forward):
                    consume_forward()
                if callable(consume_postprocess):
                    consume_postprocess()
                forward_started = time.perf_counter()
                runtime_preprocess_seconds = forward_started - runtime_batch_started
                outputs = inference_runtime.infer_raw(samples.tensors)
                forward_wall_seconds = time.perf_counter() - forward_started
                recorded_forward_seconds = float(consume_forward()) if callable(consume_forward) else 0.0
                runtime_forward_seconds = recorded_forward_seconds or forward_wall_seconds
                standalone_forward_seconds += runtime_forward_seconds
                standalone_preprocess_seconds += runtime_preprocess_seconds
                standalone_image_count += int(samples.tensors.shape[0])
            else:
                batch = datamodule.transfer_batch_to_device(batch, pl_module.device, 0)
                samples, targets = batch
                outputs = pl_module.model(samples)
            orig_sizes = torch.stack([target["orig_size"] for target in targets])
            if using_inference_runtime:
                results = inference_runtime.postprocess(outputs, orig_sizes)
                if callable(consume_postprocess):
                    consume_postprocess()
            else:
                results = pl_module.postprocess(outputs, orig_sizes)
            converted_preds = converter._convert_preds(results)
            converted_targets = converter._convert_targets(targets)
            if using_inference_runtime:
                runtime_batch_elapsed = time.perf_counter() - runtime_batch_started
                standalone_postprocess_seconds += max(
                    0.0,
                    runtime_batch_elapsed - runtime_preprocess_seconds - runtime_forward_seconds,
                )
            map_metric.update(converted_preds, converted_targets)
            matching = build_matching_data(
                converted_preds,
                converted_targets,
                iou_threshold=0.5,
                iou_type="segm" if segmentation else "bbox",
            )
            merge_matching_data(f1_local, matching)
    if not using_inference_runtime and was_training:
        pl_module.train()

    raw_metrics = map_metric.compute()
    pfx = "bbox_" if segmentation else ""
    mar_key = f"{pfx}mar_{max_dets}"
    overall: Dict[str, float] = {
        "mAP_50_95": scalar_metric(raw_metrics, f"{pfx}map"),
        "mAP_50": scalar_metric(raw_metrics, f"{pfx}map_50"),
        "mAP_75": scalar_metric(raw_metrics, f"{pfx}map_75"),
        "mAR": scalar_metric(raw_metrics, mar_key),
    }
    if segmentation:
        overall["segm_mAP_50_95"] = scalar_metric(raw_metrics, "segm_map")
        overall["segm_mAP_50"] = scalar_metric(raw_metrics, "segm_map_50")
    f1_overall, f1_by_class = compute_f1_by_class(f1_local)
    overall.update(f1_overall)

    class_ids = [int(item) for item in flatten_tensor(raw_metrics.get("classes", []))]
    ap_values = [float(item) for item in flatten_tensor(raw_metrics.get(f"{pfx}map_per_class", []))]
    ar_values = [float(item) for item in flatten_tensor(raw_metrics.get(f"{pfx}mar_{max_dets}_per_class", []))]
    ar_by_class = {class_id: ar_values[index] for index, class_id in enumerate(class_ids) if index < len(ar_values)}
    per_class_rows: List[Dict[str, Any]] = []
    for index, class_id in enumerate(class_ids):
        ap_value = ap_values[index] if index < len(ap_values) else float("nan")
        ar_value = ar_by_class.get(class_id, float("nan"))
        if (not math.isfinite(ap_value) or ap_value < 0) and (not math.isfinite(ar_value) or ar_value < 0):
            continue
        f1_values = f1_by_class.get(class_id, {})
        per_class_rows.append(
            {
                "class_id": class_id,
                "class": cat_id_to_name.get(class_id, str(class_id)),
                "ap": ap_value,
                "ar": ar_value,
                "f1": f1_values.get("f1"),
                "precision": f1_values.get("precision"),
                "recall": f1_values.get("recall"),
            }
        )

    metric_rows = [{"metric": key, "value": value} for key, value in overall.items()]
    write_classwise = bool(merged_config.get("periodic_test", {}).get("classwise", True))
    payload = {
        "event": event,
        "split": split,
        "metadata": dict(metadata),
        "overall": overall,
        "per_class": per_class_rows,
        "raw_torchmetrics": json_safe_value(raw_metrics),
    }
    if using_inference_runtime:
        timed_total = (
            standalone_preprocess_seconds
            + standalone_forward_seconds
            + standalone_postprocess_seconds
        )
        payload["stage_timing"] = {
            "images_or_frames": standalone_image_count,
            "total_seconds": timed_total,
            "preprocess_seconds": standalone_preprocess_seconds,
            "model_forward_seconds": standalone_forward_seconds,
            "base_model_forward_seconds": standalone_forward_seconds,
            "sahi_model_forward_seconds": 0.0,
            "recheck_model_forward_seconds": 0.0,
            "postprocess_seconds": standalone_postprocess_seconds,
            "model_forward_ratio": standalone_forward_seconds / timed_total if timed_total > 0 else 0.0,
            "sahi_model_forward_ratio": 0.0,
            "recheck_model_forward_ratio": 0.0,
        }
    write_json(output_dir / "test_metrics.json", payload)
    write_rows(output_dir / "test_metrics.csv", metric_rows, METRIC_FIELDS)
    if write_classwise:
        write_json(output_dir / "test_per_class_metrics.json", per_class_rows)
        write_rows(output_dir / "test_per_class_metrics.csv", per_class_rows, PER_CLASS_FIELDS)
    dump_config_snapshot(
        output_dir=output_dir,
        merged_config=merged_config,
        metadata={**dict(metadata), "event": event, "split": split, "output_dir": str(output_dir)},
        source_config=source_config,
        train_config=train_config,
        model_config=model_config,
    )
    blue(f"{event} metrics saved to {output_dir}.", verbose)
    return payload


class PeriodicManualTestCallback(Callback):
    """PyTorch Lightning callback that runs scheduled manual test evaluation."""

    def __init__(
        self,
        merged_config: Mapping[str, Any],
        source_config: Optional[Path],
        output_dir: Path,
        model_config: Any,
        train_config: Any,
        timestamp: str,
        verbose: bool,
    ) -> None:
        self.merged_config = merged_config
        self.source_config = source_config
        self.output_dir = output_dir
        self.model_config = model_config
        self.train_config = train_config
        self.timestamp = timestamp
        self.verbose = verbose
        self.started_at = time.monotonic()
        self.last_test_time = self.started_at
        self.tested_epochs: set[int] = set()
        self.ddp_skip_reported = False

    def should_run(self, epoch_number: int) -> bool:
        """Return True if epoch or time interval says to run test now."""
        periodic = self.merged_config.get("periodic_test", {})
        if not periodic.get("enabled", True):
            return False
        by_epoch = int(periodic.get("test_interval_epochs") or 0)
        by_minutes = float(periodic.get("test_interval_minutes") or 0.0)
        if by_epoch > 0 and epoch_number % by_epoch == 0:
            return True
        if by_minutes > 0 and (time.monotonic() - self.last_test_time) >= by_minutes * 60.0:
            return True
        return False

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Run test evaluation at configured intervals."""
        if not getattr(trainer, "is_global_zero", True):
            return
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        if world_size > 1:
            if not self.ddp_skip_reported:
                blue("Scheduled test is skipped during multi-process training; final test still runs after fit.", self.verbose)
                self.ddp_skip_reported = True
            return
        epoch_number = int(getattr(trainer, "current_epoch", -1)) + 1
        if epoch_number in self.tested_epochs or not self.should_run(epoch_number):
            return
        periodic = self.merged_config.get("periodic_test", {})
        split = str(periodic.get("split", "test"))
        output_dir_name = render_output_template(
            str(periodic.get("output_dir_name", "periodic_tests")),
            self.merged_config,
            self.timestamp,
        )
        save_dir = self.output_dir / sanitize_name(str(output_dir_name)) / f"epoch_{epoch_number:04d}"
        metadata = {
            "epoch": epoch_number,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "global_step": int(getattr(trainer, "global_step", 0) or 0),
        }
        try:
            manual_test_evaluation(
                trainer=trainer,
                pl_module=pl_module,
                datamodule=trainer.datamodule,
                model_config=self.model_config,
                train_config=self.train_config,
                output_dir=save_dir,
                split=split,
                event="periodic_test",
                metadata=metadata,
                merged_config=self.merged_config,
                source_config=self.source_config,
                verbose=self.verbose,
                progress_bar=bool(periodic.get("progress_bar", True)),
            )
            self.last_test_time = time.monotonic()
        finally:
            self.tested_epochs.add(epoch_number)


def load_best_checkpoint_if_available(pl_module: Any, output_dir: Path, verbose: bool) -> Optional[Path]:
    """Load RF-DETR best-total weights into the live module when available."""
    import torch

    checkpoint = output_dir / "checkpoint_best_total.pth"
    if not checkpoint.exists():
        checkpoint = output_dir / "checkpoint_best_ema.pth"
    if not checkpoint.exists():
        checkpoint = output_dir / "checkpoint_best_regular.pth"
    if not checkpoint.exists():
        return None
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = getattr(pl_module.model, "_orig_mod", pl_module.model)
    raw.load_state_dict(ckpt["model"], strict=True)
    blue(f"Loaded best checkpoint for final test: {checkpoint}", verbose)
    return checkpoint


def validate_requirements(
    config: Mapping[str, Any],
    output_dir: Path,
    dataset_plan: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return warnings for ambiguous or risky settings."""
    warnings: List[str] = []
    dataset = config.get("dataset", {})
    train = config.get("train", {})
    periodic = config.get("periodic_test", {})
    model_size = config.get("model", {}).get("size", "medium")
    try:
        normalized_model_size = normalize_model_size(model_size)
        get_model_class(str(model_size))
    except Exception as exc:
        normalized_model_size = str(model_size)
        warnings.append(str(exc))
    dataset_dir_value = train.get("dataset_dir") or dataset.get("dataset_dir", "")
    dataset_dir = Path(str(dataset_dir_value)).expanduser()
    if dataset_plan and dataset_plan.get("action") == "prepare_cache":
        warnings.extend(str(warning) for warning in dataset_plan.get("warnings", []))
        if "test" not in dataset_plan.get("split_counts", {}) and (
            periodic.get("enabled", True) or periodic.get("run_final_test", True)
        ):
            warnings.append("Converted dataset has no test split but periodic/final test is enabled.")
        if dataset_plan.get("link_mode") == "copy":
            warnings.append("dataset.link_mode=copy will duplicate image files into the dataset cache.")
    elif not dataset_dir_value:
        warnings.append("dataset.dataset_dir is empty; training will fail until it points to an RF-DETR dataset root.")
    elif not dataset_dir.exists():
        warnings.append(f"dataset.dataset_dir does not exist on this machine: {dataset_dir}")
    if int(train.get("eval_interval") or 1) < 1:
        warnings.append("train.eval_interval must be >= 1 for RF-DETR.")
    if periodic.get("enabled", True) and not periodic.get("test_interval_epochs") and not periodic.get("test_interval_minutes"):
        warnings.append("periodic_test.enabled=true but no interval is configured; only final test can run.")
    if periodic.get("enabled", True) and "," in str(train.get("device", "")):
        warnings.append("Scheduled in-fit test is skipped for multi-GPU training; final test still runs.")
    if (
        not (dataset_plan and dataset_plan.get("action") == "prepare_cache")
        and dataset_dir_value
        and dataset_dir.exists()
        and (periodic.get("enabled", True) or periodic.get("run_final_test", True))
    ):
        split = str(periodic.get("split", "test"))
        split_dir = dataset_split_dir(dataset_dir, split)
        if split_dir is not None and not split_dir.exists():
            warnings.append(f"Configured test split may be missing: split={split}, expected near {split_dir}")
    if (
        not (dataset_plan and dataset_plan.get("action") == "prepare_cache")
        and str(train.get("dataset_file", "roboflow")) != "roboflow"
    ):
        warnings.append("For RF-DETR test split support, dataset_file=roboflow is recommended because it auto-detects COCO/YOLO.")
    if output_dir.exists() and not bool(config.get("output", {}).get("exist_ok", False)):
        warnings.append(f"output directory already exists and output.exist_ok=false: {output_dir}")
    if str(normalized_model_size).startswith("seg-"):
        try:
            mode = periodic_test_mode(periodic)
        except ValueError as exc:
            warnings.append(str(exc))
            mode = shared_modes.FULL_IMAGE_MODE
        if mode != shared_modes.FULL_IMAGE_MODE:
            warnings.append("Segmentation mask metrics are available for full_image tests; sahi/class_crop tests evaluate boxes only.")
    return warnings


def apply_demo_mode(config: MutableMapping[str, Any], timestamp: str, verbose: bool) -> None:
    """Clamp output and training settings for small demo runs."""
    demo = config.get("demo", {})
    if not demo.get("enabled", False):
        return
    train = config.setdefault("train", {})
    output = config.setdefault("output", {})
    periodic = config.setdefault("periodic_test", {})
    train["epochs"] = min(int(train.get("epochs") or demo.get("max_epochs", 2)), int(demo.get("max_epochs", 2)))
    train["batch_size"] = min(int(train.get("batch_size") or demo.get("max_batch_size", 2)), int(demo.get("max_batch_size", 2)))
    train["grad_accum_steps"] = min(
        int(train.get("grad_accum_steps") or demo.get("max_grad_accum_steps", 1)),
        int(demo.get("max_grad_accum_steps", 1)),
    )
    train["checkpoint_interval"] = max(1, int(demo.get("checkpoint_interval", 1)))
    train["progress_bar"] = demo.get("progress_bar", "tqdm")
    train["tensorboard"] = False
    train["wandb"] = False
    train["mlflow"] = False
    output["output_dir"] = render_output_template(
        demo.get("output_dir", str(PROJECT_DIR / "demo_runs" / "demo_{timestamp}")),
        config,
        timestamp,
    )
    periodic["test_interval_epochs"] = int(demo.get("test_interval_epochs", 1))
    blue("Demo mode enabled: epochs, batch, logging, and output folder were clamped.", verbose)


def apply_cli_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    """Apply argparse overrides to the loaded config."""
    runtime = config.setdefault("runtime", {})
    model = config.setdefault("model", {})
    dataset = config.setdefault("dataset", {})
    train = config.setdefault("train", {})
    output = config.setdefault("output", {})
    periodic = config.setdefault("periodic_test", {})
    demo = config.setdefault("demo", {})

    for key in ("dry_run", "verbose", "confirm_before_run"):
        value = getattr(args, key, None)
        if value is not None:
            runtime[key] = value
    if args.demo is not None:
        demo["enabled"] = args.demo
    if args.model_size is not None:
        model["size"] = args.model_size
    if args.resolution is not None:
        model["resolution"] = args.resolution
    if args.pretrain_weights is not None:
        model["pretrain_weights"] = args.pretrain_weights
    if args.num_classes is not None:
        model["num_classes"] = args.num_classes
    tracknet_focus = getattr(args, "tracknet_focus", None)
    if tracknet_focus is not None:
        motion = model.setdefault("motion", {})
        motion.setdefault("focus", {})["mode"] = tracknet_focus
    if args.dataset_dir is not None:
        dataset["dataset_dir"] = args.dataset_dir
        train["dataset_dir"] = args.dataset_dir
    if args.data_yaml is not None:
        dataset["data_yaml"] = args.data_yaml
    if args.coco_json is not None:
        dataset["coco_json"] = args.coco_json
    if args.image_dir is not None:
        dataset["image_dir"] = args.image_dir
    if args.dataset_source_format is not None:
        dataset["source_format"] = args.dataset_source_format
    if args.dataset_link_mode is not None:
        dataset["link_mode"] = args.dataset_link_mode
    if args.cache_root is not None:
        dataset["cache_root"] = args.cache_root
    if args.refresh_cache is not None:
        dataset["refresh_cache"] = args.refresh_cache
    if args.split_ratio is not None:
        dataset["split_ratio"] = args.split_ratio
    if args.split_seed is not None:
        dataset["split_seed"] = args.split_seed
    if args.prepared_dir_name is not None:
        dataset["prepared_dir_name"] = args.prepared_dir_name
    if args.overwrite_prepared_dataset is not None:
        dataset["overwrite_prepared_dataset"] = args.overwrite_prepared_dataset
    if args.dataset_file is not None:
        train["dataset_file"] = args.dataset_file
    if args.output_dir is not None:
        output["output_dir"] = args.output_dir
    if args.project is not None:
        output["root"] = args.project
    if args.name is not None:
        output["name"] = args.name
    if args.exist_ok is not None:
        output["exist_ok"] = args.exist_ok

    train_overrides = {
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "num_workers": args.workers,
        "checkpoint_interval": args.checkpoint_interval,
        "resume": getattr(args, "resume", None),
        "eval_interval": args.eval_interval,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "progress_bar": args.progress_bar,
    }
    for key, value in train_overrides.items():
        if value is not None:
            train[key] = value

    if args.periodic_test is not None:
        periodic["enabled"] = args.periodic_test
    if args.test_interval_epochs is not None:
        periodic["test_interval_epochs"] = args.test_interval_epochs
    if args.test_interval_minutes is not None:
        periodic["test_interval_minutes"] = args.test_interval_minutes
    if args.test_split is not None:
        periodic["split"] = args.test_split
    if args.final_test is not None:
        periodic["run_final_test"] = args.final_test
    if args.classwise is not None:
        periodic["classwise"] = args.classwise

    extra = parse_extra_args(args.extra)
    if extra:
        train.setdefault("extra_train_args", {}).update(extra)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    usage = """
Detailed usage:
  Configure persistent settings in config/rf_detr_train.yaml, then override common fields from CLI.
  Use --yes for non-interactive execution after accepting the resource estimate.
  Use --dry-run to validate config and estimate outputs without training.
  Use --demo to write to a small demo output folder and clamp epochs/batch/logging.

Example usage:
  uv run python train_rf_detr_model.py --config config/rf_detr_train.yaml

  uv run python train_rf_detr_model.py --dataset-dir /data/my_dataset --device 0 --epochs 100 --yes

  uv run python train_rf_detr_model.py --eval-interval 5 --test-interval-epochs 10 --final-test --yes

  uv run python train_rf_detr_model.py --output-dir D:/runs/rf_detr/custom_run --yes

    uv run python train_rf_detr_model.py --demo --dry-run --yes

  uv run python train_rf_detr_model.py --extra lr_drop=50 --extra use_ema=true --yes

  # Auto-detected source dataset: convert/reuse RF-DETR cache before training
  uv run python train_rf_detr_model.py \\
      --dataset-source-format auto \\
      --dataset-dir D:/datasets/my_dataset \\
      --dataset-link-mode auto \\
      --yes
"""
    parser = argparse.ArgumentParser(
        description="RF-DETR trainer with Ultralytics-style config, output folders, scheduled test, and per-class metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=usage,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Validate and estimate outputs only.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None, help="Enable blue wrapper logs.")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Disable blue wrapper logs.")
    parser.add_argument("--confirm-before-run", type=parse_bool, default=None, help="Ask before heavy output is created.")
    parser.add_argument("--demo", action="store_true", default=None, help="Enable demo mode.")
    parser.add_argument("--no-demo", dest="demo", action="store_false", help="Disable demo mode.")

    parser.add_argument("--model-size", default=None, help="RF-DETR size: nano, small, medium, large, seg-small, etc.")
    parser.add_argument("--resolution", type=int, default=None, help="Input resolution override.")
    parser.add_argument("--pretrain-weights", default=None, help="default, null/false, hosted key, or local .pth path.")
    parser.add_argument("--num-classes", type=int, default=None, help="Optional class count override.")
    parser.add_argument(
        "--tracknet-focus",
        choices=("single", "all"),
        default=None,
        help="Override model.motion.focus.mode for TrackNet heatmap supervision.",
    )

    parser.add_argument("--dataset-dir", default=None, help="Dataset root or RF-DETR dataset root.")
    parser.add_argument("--data-yaml", default=None, help="Ultralytics YOLO dataset YAML path.")
    parser.add_argument("--coco-json", default=None, help="COCO JSON annotation file for single-file COCO datasets.")
    parser.add_argument("--image-dir", default=None, help="Image directory used with --coco-json when image paths are relative.")
    parser.add_argument(
        "--dataset-source-format",
        choices=sorted(DATASET_SOURCE_FORMATS),
        default=None,
        help="Dataset source format before RF-DETR cache conversion.",
    )
    parser.add_argument(
        "--dataset-link-mode",
        choices=["auto", "hardlink", "symlink", "junction", "copy"],
        default=None,
        help="How to store cache images: auto, hardlink, symlink, junction alias, or copy.",
    )
    parser.add_argument("--cache-root", default=None, help="Dataset cache root. Relative paths resolve under rf_detr_trainer.")
    parser.add_argument("--refresh-cache", type=parse_bool, default=None, help="Rebuild the dataset cache even if fingerprint matches.")
    parser.add_argument("--split-ratio", type=parse_split_ratio_arg, default=None, help="Unsplit data split ratio as train,valid,test, e.g. 8,1,1.")
    parser.add_argument("--split-seed", type=int, default=None, help="Deterministic split seed for unsplit datasets.")
    parser.add_argument("--prepared-dir-name", default=None, help="Legacy option retained for old configs; cache conversion ignores it.")
    parser.add_argument(
        "--overwrite-prepared-dataset",
        type=parse_bool,
        default=None,
        help="Legacy option retained for old configs; use --refresh-cache for cache rebuilds.",
    )
    parser.add_argument("--dataset-file", choices=["roboflow", "coco", "yolo", "o365"], default=None, help="RF-DETR dataset_file.")
    parser.add_argument("--output-dir", default=None, help="Exact custom output directory. Supports output placeholders.")
    parser.add_argument("--project", default=None, help="Output root when output-dir is not used. Supports output placeholders.")
    parser.add_argument("--name", default=None, help="Run name when output-dir is not used. Supports output placeholders.")
    parser.add_argument("--exist-ok", type=parse_bool, default=None, help="Allow existing output directory.")

    parser.add_argument("--device", default=None, help="auto, cpu, cuda, 0, 0,1, cuda:0, mps, or -1.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch-size", type=parse_scalar, default=None, help="Micro batch size or auto.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=None, help="Decoder/head learning rate.")
    parser.add_argument("--lr-encoder", type=float, default=None, help="Encoder learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers.")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Save checkpoint every N epochs.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a PyTorch Lightning .ckpt checkpoint, restoring optimizer, scheduler, and callback state.",
    )
    parser.add_argument("--eval-interval", type=int, default=None, help="Run validation metrics every N epochs.")
    parser.add_argument("--early-stopping", type=parse_bool, default=None, help="Enable early stopping.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--progress-bar", choices=["tqdm", "rich", "none"], default=None, help="RF-DETR progress bar style.")

    parser.add_argument("--periodic-test", type=parse_bool, default=None, help="Enable scheduled in-training test.")
    parser.add_argument("--test-interval-epochs", type=int, default=None, help="Run test every N epochs.")
    parser.add_argument("--test-interval-minutes", type=float, default=None, help="Run test every N minutes.")
    parser.add_argument("--test-split", default=None, help="Dataset split used for scheduled/final test, usually test.")
    parser.add_argument("--final-test", dest="final_test", action="store_true", default=None, help="Run final test.")
    parser.add_argument("--no-final-test", dest="final_test", action="store_false", help="Skip final test.")
    parser.add_argument("--classwise", dest="classwise", action="store_true", default=None, help="Write per-class metrics.")
    parser.add_argument("--no-classwise", dest="classwise", action="store_false", help="Skip per-class metrics files.")
    parser.add_argument("--extra", action="append", default=None, help="Additional RF-DETR TrainConfig key=value. Repeatable.")
    return parser


def _main_impl(timing_context: Optional[MutableMapping[str, Any]] = None) -> int:
    """
    Main entry point for RF-DETR training.

    Usage:
      1. Edit config/rf_detr_train.yaml.
      2. Run a dry run:
         uv run python train_rf_detr_model.py --dry-run --yes
      3. Start training:
         uv run python train_rf_detr_model.py

    Example usage:
      uv run python train_rf_detr_model.py --config config/rf_detr_train.yaml --yes

      uv run python train_rf_detr_model.py \\
          --dataset-dir /data/football_dataset \\
          --model-size medium \\
          --device 0 \\
          --epochs 300 \\
          --batch-size 4 \\
          --grad-accum-steps 4 \\
          --eval-interval 5 \\
          --test-interval-epochs 30 \\
          --output-dir /runs/rf_detr/football_medium \\
          --yes

      uv run python train_rf_detr_model.py --demo --dry-run --yes

      # Auto-detect source layout and prepare/reuse an RF-DETR dataset cache
      uv run python train_rf_detr_model.py \\
          --dataset-source-format auto \\
          --dataset-dir D:/datasets/football_dataset \\
          --dataset-link-mode auto \\
          --yes
    """
    parser = build_parser()
    args = parser.parse_args()
    source_config = Path(args.config).expanduser()
    if not source_config.is_absolute():
        source_config = (Path.cwd() / source_config).resolve()
    if not source_config.exists():
        raise FileNotFoundError(f"Config file not found: {source_config}")

    distributed_child = is_nonzero_distributed_process()
    timestamp = (
        os.environ.get(DDP_TIMESTAMP_ENV, "").strip()
        if distributed_child and os.environ.get(DDP_TIMESTAMP_ENV)
        else datetime.now().strftime(TIMESTAMP_FORMAT)
    )
    if not distributed_child:
        os.environ[DDP_TIMESTAMP_ENV] = timestamp

    with tqdm(total=8, desc="Preparing", disable=distributed_child) as bar:
        config = load_yaml(source_config)
        apply_cli_overrides(config, args)
        verbose = bool(config.get("runtime", {}).get("verbose", True))
        if timing_context is not None:
            timing_context["verbose"] = verbose and not distributed_child
        apply_demo_mode(config, timestamp, verbose)
        resolve_train_resume_checkpoint(config, source_config)
        child_metadata: Dict[str, Any] = {}
        if distributed_child:
            child_metadata = apply_distributed_child_runtime_overrides(config)
        bar.update(1)

        if config.get("train", {}).get("progress_bar") == "none":
            config["train"]["progress_bar"] = None
        if config.get("dataset", {}).get("dataset_dir") and not config.get("train", {}).get("dataset_dir"):
            config.setdefault("train", {})["dataset_dir"] = config["dataset"]["dataset_dir"]

        output_dir = build_output_dir(config, timestamp)
        if timing_context is not None:
            timing_context["output_dir"] = str(output_dir)
        if not distributed_child:
            export_distributed_child_runtime(config, output_dir, timestamp)

        if distributed_child:
            dataset_metadata = child_metadata
            bar.set_description("Estimating")
            bar.update(2)
            bar.set_description("Dataset")
            bar.update(1)
        else:
            dataset_plan = build_dataset_plan(config, output_dir, source_config)
            warnings = validate_requirements(config, output_dir, dataset_plan)
            for warning in warnings:
                blue(f"Requirement check warning: {warning}", verbose=verbose, force=True)
            if output_dir.exists() and not bool(config.get("output", {}).get("exist_ok", False)):
                raise FileExistsError(f"Output directory already exists and output.exist_ok=false: {output_dir}")
            bar.set_description("Estimating")
            bar.update(1)

            periodic_count, periodic_note = estimate_periodic_tests(config)
            estimate = estimate_outputs(config, output_dir, periodic_count, dataset_plan)
            estimate["periodic_test_count"] = periodic_count if periodic_count is not None else periodic_note
            estimate["model_size"] = config.get("model", {}).get("size")
            estimate["eval_interval"] = config.get("train", {}).get("eval_interval")
            if timing_context is not None:
                timing_context["estimate"] = estimate
                timing_context["dry_run"] = bool(config.get("runtime", {}).get("dry_run", False))
            confirm = bool(config.get("runtime", {}).get("confirm_before_run", True))
            assume_yes = bool(args.yes or not confirm)
            confirm_or_exit(estimate, verbose=verbose, assume_yes=assume_yes)
            bar.set_description("Confirmed")
            bar.update(1)

            if bool(config.get("runtime", {}).get("dry_run", False)):
                blue("Dry run complete. Training was not started.", verbose=verbose, force=True)
                bar.update(bar.total - bar.n)
                return 0

            output_dir.mkdir(parents=True, exist_ok=True)
            if timing_context is not None:
                timing_context["outputs_created"] = True
            start_run_log_capture(output_dir, "train", timing_context, enabled=not distributed_child)
            bar.set_description("Dataset")
            dataset_metadata = materialize_dataset_plan(dataset_plan, config, output_dir, verbose)
            export_distributed_child_runtime(config, output_dir, timestamp)
            bar.update(1)

            dump_config_snapshot(
                output_dir=output_dir,
                merged_config=config,
                metadata={
                    "event": "pre_import",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "argv": sys.argv,
                    "cwd": str(Path.cwd()),
                    "output_dir": str(output_dir),
                    "dataset_metadata": dataset_metadata,
                },
                source_config=source_config,
            )
        bar.set_description("Importing")
        bar.update(1)

        blue("Importing RF-DETR training stack.", verbose)
        ensure_rfdetr_detection_hflip_support()
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
        from rfdetr.training.auto_batch import resolve_auto_batch_config

        model_cls = get_model_class(str(config.get("model", {}).get("size", "medium")))
        model_kwargs = build_model_kwargs(config)
        train_kwargs = build_train_kwargs(config, output_dir)
        device_value = train_kwargs.pop("_device", None)
        trainer_kwargs = parse_device_to_trainer_kwargs(device_value)
        trainer_kwargs.update(config.get("trainer", {}).get("extra_trainer_args", {}) or {})
        apply_validation_interval_to_trainer_kwargs(config, trainer_kwargs, verbose)
        apply_multigpu_ddp_strategy(config, trainer_kwargs, verbose)
        apply_multigpu_validation_safety(trainer_kwargs, verbose)

        blue(f"Creating RF-DETR model: {model_cls.__name__}.", verbose)
        rf_model = model_cls(**model_kwargs)
        # Validate stock -> P2 before project-owned modules add their own state.
        # TrackNet is intentionally absent from the official RF-DETR checkpoint.
        _p2_cfg = config.get('model', {}).get('p2', {}) or {}
        if bool(_p2_cfg.get('enabled', False)):
            from rf_detr_p2 import assert_p2_training_checkpoint_compatible

            pretrain_path = getattr(rf_model.model_config, 'pretrain_weights', None)
            if pretrain_path is not None and Path(str(pretrain_path)).is_file():
                assert_p2_training_checkpoint_compatible(
                    rf_model.model,
                    pretrain_path,
                    build_pitchobjectlab_architecture(config, rf_model.model_config),
                    allow_stock_initialization=True,
                )
        # Attach the TrackNetV5 motion module after the backbone is built so that
        # the projector output channel count is available for MotionModule sizing.
        _motion_cfg = config.get("model", {}).get("motion", {}) or {}
        _motion_enabled = motion_module_enabled(config)
        if _motion_enabled:
            from rf_detr_motion import attach_motion_module

            attach_motion_module(rf_model.model, _motion_cfg)
            blue("TrackNetV5 motion module (MDD + R-STR) attached to LWDETR.", verbose)
        train_config = rf_model.get_train_config(**train_kwargs)
        if train_config.batch_size == "auto":
            from rfdetr.detr import _ensure_model_on_device

            _ensure_model_on_device(rf_model.model)
            auto_batch = resolve_auto_batch_config(
                model_context=rf_model.model,
                model_config=rf_model.model_config,
                train_config=train_config,
            )
            train_config.batch_size = auto_batch.safe_micro_batch
            train_config.grad_accum_steps = auto_batch.recommended_grad_accum_steps
            blue(
                f"Auto batch resolved: batch_size={train_config.batch_size}, grad_accum_steps={train_config.grad_accum_steps}.",
                verbose,
            )

        _temporal_enabled = temporal_motion_enabled(config)
        if not _temporal_enabled:
            rf_model._align_num_classes_from_dataset(train_config.dataset_dir)

        if _temporal_enabled:
            from rf_detr_temporal_runtime import (
                build_temporal_datamodule,
                build_temporal_model_module,
            )

            datamodule = build_temporal_datamodule(config, rf_model.model_config, train_config)
            _align_temporal_num_classes(rf_model, datamodule.class_names)
            module = build_temporal_model_module(rf_model.model_config, train_config, config)
            blue("Using complete three-frame temporal windows and TrackNet heatmap loss.", verbose)
        else:
            module = RFDETRModelModule(rf_model.model_config, train_config)
            datamodule = RFDETRDataModule(rf_model.model_config, train_config)

        if _motion_enabled and train_config.resume:
            from rf_detr_motion import assert_motion_checkpoint_compatible

            assert_motion_checkpoint_compatible(
                module.model,
                train_config.resume,
                build_pitchobjectlab_architecture(config, rf_model.model_config),
            )
        if bool(_p2_cfg.get('enabled', False)) and train_config.resume:
            assert_p2_training_checkpoint_compatible(
                module.model,
                train_config.resume,
                build_pitchobjectlab_architecture(config, rf_model.model_config),
                allow_stock_initialization=False,
            )

        trainer = build_trainer(train_config, rf_model.model_config, **trainer_kwargs)
        install_best_checkpoint_metadata(
            trainer,
            build_pitchobjectlab_architecture(config, rf_model.model_config),
        )
        trainer.callbacks.append(
            PeriodicManualTestCallback(
                merged_config=config,
                source_config=source_config,
                output_dir=output_dir,
                model_config=rf_model.model_config,
                train_config=train_config,
                timestamp=timestamp,
                verbose=verbose,
            )
        )
        trainer.callbacks.append(TrainBatchGridCallback(output_dir=output_dir, verbose=verbose))
        trainer.callbacks.append(ValidationPredictionGridCallback(output_dir=output_dir, verbose=verbose))
        trainer.callbacks.append(EpochValidationResultsCallback(output_dir=output_dir, verbose=verbose))
        if not distributed_child:
            dump_config_snapshot(
                output_dir=output_dir,
                merged_config=config,
                metadata={
                    "event": "train_start",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "argv": sys.argv,
                    "cwd": str(Path.cwd()),
                    "output_dir": str(output_dir),
                    "device": device_value or "auto",
                    "train_batch_grid_files": [
                        str(output_dir / f"train_batch{index}.jpg")
                        for index in range(TRAIN_BATCH_GRID_MAX_BATCHES)
                    ],
                    "validation_label_grid_files": [
                        str(output_dir / f"val_batch{index}_labels.jpg")
                        for index in range(VALIDATION_PREDICTION_GRID_MAX_BATCHES)
                    ],
                    "validation_prediction_grid_files": [
                        str(output_dir / f"val_batch{index}_pred.jpg")
                        for index in range(VALIDATION_PREDICTION_GRID_MAX_BATCHES)
                    ],
                    "validation_prediction_grid_min_score": VALIDATION_PREDICTION_GRID_MIN_SCORE,
                },
                source_config=source_config,
                train_config=train_config,
                model_config=rf_model.model_config,
            )
        bar.set_description("Training")
        bar.update(1)

        blue("Starting RF-DETR training.", verbose)
        trainer.fit(module, datamodule, ckpt_path=train_config.resume or None)
        if bool(getattr(trainer, 'is_global_zero', True)):
            enrich_best_checkpoint_metadata(output_dir)
        rf_model.model.model = module.model
        if getattr(datamodule, "class_names", None) is not None:
            rf_model.model.class_names = datamodule.class_names
        bar.set_description("Final test")
        bar.update(1)

        periodic = config.get("periodic_test", {})
        is_global_zero = bool(getattr(trainer, "is_global_zero", not distributed_child))
        if bool(periodic.get("run_final_test", True)) and is_global_zero:
            best_checkpoint = load_best_checkpoint_if_available(module, output_dir, verbose)
            final_output_name = render_output_template(
                str(periodic.get("final_output_dir_name", "final_test")),
                config,
                timestamp,
            )
            final_dir = output_dir / sanitize_name(str(final_output_name))
            if _temporal_enabled:
                from rf_detr_temporal_runtime import run_temporal_split

                run_temporal_split(
                    rf_model=rf_model,
                    config=config,
                    output_dir=final_dir,
                    split=str(periodic.get("split", "test")),
                    save_heatmaps=True,
                )
            else:
                manual_test_evaluation(
                    trainer=trainer,
                    pl_module=module,
                    datamodule=datamodule,
                    model_config=rf_model.model_config,
                    train_config=train_config,
                    output_dir=final_dir,
                    split=str(periodic.get("split", "test")),
                    event="final_test",
                    metadata={
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
                        "global_step": int(getattr(trainer, "global_step", 0) or 0),
                    },
                    merged_config=config,
                    source_config=source_config,
                    verbose=verbose,
                    progress_bar=bool(periodic.get("progress_bar", True)),
                )
        elif is_global_zero:
            blue("Final test skipped by config.", verbose)

        bar.set_description("Done")
        bar.update(1)

    if not distributed_child:
        blue(f"Training output directory: {output_dir}", verbose=verbose, force=True)
        blue("Done.", verbose=verbose, force=True)
    return 0


def main() -> int:
    """Run training with elapsed-time reporting."""
    timing_context = start_run_timing("train", verbose=not is_nonzero_distributed_process())
    try:
        result = _main_impl(timing_context)
        timing_context["success"] = True
        return result
    except Exception as exc:
        timing_context["success"] = False
        timing_context["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        raise
    finally:
        finish_run_timing(timing_context)


if __name__ == "__main__":
    raise SystemExit(main())
