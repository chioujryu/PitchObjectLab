"""
Train any local Ultralytics model config with configurable per-epoch validation interval,
periodic test-split evaluation on a schedule, and a final test evaluation after training.

This script is a configurable Python wrapper around the Ultralytics Python API. It supports:

1. Ultralytics training from any YAML under ultralytics/cfg/models, plus normal .pt weights.
2. Configurable validation interval (val_interval): run val every N epochs instead of every epoch.
   The interval affects Ultralytics early stopping and best.pt selection.
3. Periodic evaluation on the dataset YAML `test` split every N epochs or every N minutes.
4. Final test evaluation after training completes.
5. A pre-run output/resource estimate and developer confirmation before heavy output is written.
6. Config snapshots inside every run/test output folder for reproducibility.
7. Linux and Windows path support.

Validation interval behaviour
-------------------------------
  val_interval=1 (default)  Run validation every epoch (Ultralytics default behaviour).
  val_interval=5             Run validation every 5th epoch. On non-val epochs the previous
                             val result is reused so early stopping patience and best.pt still
                             track real val performance without saving non-validated weights.
  val_interval=0             Disable validation entirely (equivalent to val=false).

Note on early stopping with val_interval > 1
---------------------------------------------
  Patience counts every training epoch, including epochs where val is skipped.
  On skipped epochs the previous val fitness is returned unchanged, so the stopper treats
  them as "no improvement" epochs. To wait for N_val val evaluations before stopping,
  set patience = N_val * val_interval.
  Example: val_interval=5, patience=150 gives 30 val evaluations before early stop.

Example usage:

    uv run python projects/yolo_trainer/train_ultralytics_model.py \\
        --config projects/yolo_trainer/config/yolo_train.yaml \\
        --yes

    uv run python projects/yolo_trainer/train_ultralytics_model.py \\
        --config projects/yolo_trainer/config/yolo_train.yaml \\
        --model rtdetr-l.yaml \\
        --task auto \\
        --device 7 \\
        --epochs 1000 \\
        --imgsz 640 \\
        --batch 32 \\
        --val-interval 5 \\
        --test-interval-epochs 10 \\
        --name yolo26m-p2_imgsz640_val5_{timestamp}

    uv run python projects/yolo_trainer/train_ultralytics_model.py \\
        --config projects/yolo_trainer/config/yolo_train.yaml \\
        --demo \\
        --yes

Notes:
    - For periodic tests during training, prefer one GPU or CPU. Ultralytics multi-GPU DDP training runs in
      subprocesses, so in-process Python callbacks are not guaranteed to run in the parent process.
    - Use --extra key=value for any Ultralytics argument that is not exposed as a first-class CLI flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import colorama
import yaml
from colorama import Fore, Style
from tqdm import tqdm

colorama.init(autoreset=True)

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_common import test_modes as shared_modes

DEFAULT_CONFIG = PROJECT_DIR / "config" / "yolo_train.yaml"
MODEL_CONFIG_ROOT = REPO_ROOT / "ultralytics" / "cfg" / "models"
IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_.-]+)\}")
MODEL_YAML_SUFFIXES = {".yaml", ".yml"}
MODEL_CONFIG_PREFIXES = ("ultralytics/cfg/models/", "cfg/models/", "models/")
SUPPORTED_TASKS = {"auto", "detect", "segment", "classify", "pose", "obb"}
_MODEL_CONFIG_INDEX: Optional[Dict[str, Path]] = None
_MODEL_CONFIG_DUPLICATES: Optional[Dict[str, List[Path]]] = None
PER_CLASS_METRIC_FIELDNAMES = [
    "class_id",
    "class",
    "images",
    "instances",
    "precision",
    "recall",
    "f1",
    "mAP50",
    "mAP75",
    "mAP50-95",
]


def blue(message: str, verbose: bool = True, force: bool = False) -> None:
    """Print blue English status text."""
    if verbose or force:
        print(Fore.BLUE + Style.BRIGHT + message)


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


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge dictionaries in-place."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


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
        yaml.safe_dump(dict(data), file, sort_keys=False, allow_unicode=False)


def is_abs_any_os(value: str) -> bool:
    """Return True for Windows or POSIX absolute paths regardless of current OS."""
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def is_url_like(value: str) -> bool:
    """Detect URL/model sources that should not be path-normalized."""
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value))


def resolve_existing_or_raw(value: Any, bases: Sequence[Path]) -> Any:
    """Resolve a local path if it exists; otherwise keep the original value for Ultralytics to resolve."""
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


def normalize_model_lookup_key(value: str) -> str:
    """Normalize a model config lookup key for fast dictionary access."""
    text = str(value).strip().strip("'\"").replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = text.lower()
    for prefix in MODEL_CONFIG_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text


def add_model_lookup_key(index: Dict[str, Path], duplicates: Dict[str, List[Path]], key: str, path: Path) -> None:
    """Add one model lookup key while tracking ambiguous aliases."""
    normalized = normalize_model_lookup_key(key)
    if not normalized:
        return
    existing = index.get(normalized)
    if existing is None:
        index[normalized] = path
        return
    if existing != path:
        duplicates.setdefault(normalized, [existing]).append(path)


def build_model_config_index() -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """Build a cached O(1) lookup table for ultralytics/cfg/models YAML files."""
    global _MODEL_CONFIG_INDEX, _MODEL_CONFIG_DUPLICATES
    if _MODEL_CONFIG_INDEX is not None and _MODEL_CONFIG_DUPLICATES is not None:
        return _MODEL_CONFIG_INDEX, _MODEL_CONFIG_DUPLICATES

    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    if MODEL_CONFIG_ROOT.exists():
        for path in sorted(MODEL_CONFIG_ROOT.rglob("*.yaml")) + sorted(MODEL_CONFIG_ROOT.rglob("*.yml")):
            rel = path.relative_to(MODEL_CONFIG_ROOT).as_posix()
            stem_rel = str(Path(rel).with_suffix("")).replace("\\", "/")
            for key in (
                path.name,
                path.stem,
                rel,
                stem_rel,
                f"ultralytics/cfg/models/{rel}",
                f"cfg/models/{rel}",
                f"models/{rel}",
                f"ultralytics/cfg/models/{stem_rel}",
                f"cfg/models/{stem_rel}",
                f"models/{stem_rel}",
            ):
                add_model_lookup_key(index, duplicates, key, path.resolve())

    for key in duplicates:
        index.pop(key, None)
    _MODEL_CONFIG_INDEX = index
    _MODEL_CONFIG_DUPLICATES = duplicates
    return index, duplicates


def model_lookup_candidates(value: str) -> List[str]:
    """Return likely aliases for a model config name, preserving most-specific candidates first."""
    key = normalize_model_lookup_key(value)
    candidates = [key]
    suffix = Path(key).suffix.lower()
    if suffix not in MODEL_YAML_SUFFIXES:
        candidates.extend([f"{key}.yaml", f"{key}.yml"])
    seen = set()
    unique: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_model_source(value: Any, bases: Sequence[Path]) -> Any:
    """Resolve model YAML aliases from ultralytics/cfg/models, or keep raw weights/URLs for Ultralytics."""
    resolved = resolve_existing_or_raw(value, bases)
    if value is None or isinstance(value, (bool, int, float, list, tuple, dict)):
        return resolved

    text = str(value).strip()
    if not text or is_url_like(text):
        return resolved

    resolved_path = Path(str(resolved)).expanduser()
    if resolved_path.exists():
        return str(resolved_path.resolve())

    suffix = Path(text).suffix.lower()
    if suffix and suffix not in MODEL_YAML_SUFFIXES:
        return resolved
    if is_abs_any_os(text):
        return resolved

    index, duplicates = build_model_config_index()
    for candidate in model_lookup_candidates(text):
        if candidate in duplicates:
            matches = ", ".join(str(path) for path in duplicates[candidate])
            raise ValueError(f"Ambiguous model config alias {value!r}. Use one exact relative path. Matches: {matches}")
        match = index.get(candidate)
        if match:
            return str(match)
    return resolved


def relative_to_repo(path_value: Any) -> str:
    """Return a compact repo-relative path when possible."""
    try:
        path = Path(str(path_value)).resolve()
        return path.relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        return str(path_value)


def module_name_from_head_entry(entry: Any) -> str:
    """Extract the Ultralytics module name from a model YAML head entry."""
    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) < 3:
        return ""
    module = entry[-2]
    if isinstance(module, str):
        return module.rsplit(".", 1)[-1].lower()
    return str(module).rsplit(".", 1)[-1].lower()


def task_from_module_name(module_name: str) -> Optional[str]:
    """Map a final model-head module name to an Ultralytics task."""
    module = module_name.lower()
    if module in {"classify", "classifier", "cls", "fc"}:
        return "classify"
    if "rtdetr" in module or "detect" in module:
        return "detect"
    if "segment" in module:
        return "segment"
    if "pose" in module:
        return "pose"
    if "obb" in module:
        return "obb"
    return None


def task_from_model_name(value: Any) -> Optional[str]:
    """Infer task from a model filename/path when YAML inspection is unavailable."""
    path = Path(str(value).replace("\\", "/"))
    stem = path.stem.lower()
    parts = {part.lower() for part in path.parts}
    if "-seg" in stem or "segment" in parts:
        return "segment"
    if "-cls" in stem or "classify" in parts:
        return "classify"
    if "-pose" in stem or "pose" in parts:
        return "pose"
    if "-obb" in stem or "obb" in parts:
        return "obb"
    if "rtdetr" in stem or "rt-detr" in parts or "detect" in parts:
        return "detect"
    return None


def infer_task_from_model_source(model_source: Any, task_override: str = "auto") -> str:
    """Infer the effective task before model construction, with RT-DETR YAML support."""
    task = str(task_override or "auto").lower()
    if task != "auto":
        return task

    path = Path(str(model_source)).expanduser()
    if path.suffix.lower() in MODEL_YAML_SUFFIXES and path.exists():
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if isinstance(data, Mapping):
            head = data.get("head")
            if isinstance(head, Sequence) and not isinstance(head, (str, bytes)) and head:
                detected = task_from_module_name(module_name_from_head_entry(head[-1]))
                if detected:
                    return detected

    detected = task_from_model_name(model_source)
    return detected or "detect"


def is_rtdetr_model_source(model_source: Any) -> bool:
    """Return True when a model source should be constructed with ultralytics.RTDETR."""
    path = Path(str(model_source).replace("\\", "/"))
    stem = path.stem.lower()
    parts = {part.lower() for part in path.parts}
    if "rtdetr" in stem or "rt-detr" in parts:
        return True

    actual_path = Path(str(model_source)).expanduser()
    if actual_path.suffix.lower() in MODEL_YAML_SUFFIXES and actual_path.exists():
        try:
            with actual_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            head = data.get("head") if isinstance(data, Mapping) else None
            return (
                isinstance(head, Sequence)
                and not isinstance(head, (str, bytes))
                and bool(head)
                and "rtdetr" in module_name_from_head_entry(head[-1])
            )
        except (OSError, yaml.YAMLError):
            return False
    return False


def create_ultralytics_model(model_source: Any, task_override: str, verbose: bool) -> Tuple[Any, str]:
    """Create the right Ultralytics model wrapper and return it with the detected task."""
    detected_task = infer_task_from_model_source(model_source, task_override)
    if is_rtdetr_model_source(model_source):
        from ultralytics import RTDETR

        return RTDETR(str(model_source)), "detect"

    from ultralytics import YOLO

    return YOLO(model_source, task=detected_task, verbose=verbose), detected_task


def render_config_template(value: Any, replacements: Mapping[str, Any]) -> Any:
    """Render folder-name placeholders in strings."""
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in replacements:
            return match.group(0)
        return str(replacements[key])

    return TEMPLATE_PLACEHOLDER_RE.sub(replace, value)


def render_timestamped(text: Any, timestamp: str) -> Any:
    """Replace timestamp placeholders in strings."""
    return render_config_template(text, {"timestamp": timestamp, "date": timestamp[:8]})


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


def path_stem_fragment(value: Any, default: str) -> str:
    """Return a path/model-like value as a compact safe fragment."""
    text = str(value or "").strip()
    if not text:
        return default
    normalized = text.replace("\\", "/").rstrip("/")
    path = Path(normalized)
    return template_fragment(path.stem or path.name or normalized or default)


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


def build_output_template_replacements(
    config: Mapping[str, Any],
    timestamp: str,
    train_args: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Build output folder placeholders from config, train args, and runtime aliases."""
    replacements = build_template_replacements(config)
    for key, value in (train_args or {}).items():
        key_text = str(key)
        if key_text.startswith("_"):
            continue
        fragment = template_fragment(value)
        replacements[f"train.{key_text}"] = fragment
        replacements[key_text] = fragment

    dataset = config.get("dataset", {})
    train = config.get("train", {})
    model_value = config.get("model", (train_args or {}).get("_model_source", "model"))
    data_yaml = dataset.get("data_yaml") or (train_args or {}).get("data")
    dataset_source = data_yaml or dataset.get("path") or "dataset"
    task_value = (train_args or {}).get("task") or config.get("task", "auto")
    split_value = (train_args or {}).get("split") or train.get("split")

    replacements.update(
        {
            "timestamp": timestamp,
            "date": timestamp[:8],
            "model": path_stem_fragment(model_value, "model"),
            "dataset": path_stem_fragment(dataset_source, "dataset"),
            "task": template_fragment(task_value),
        }
    )
    if data_yaml:
        replacements["data_yaml"] = path_stem_fragment(data_yaml, "data")
    if split_value is not None:
        replacements["split"] = template_fragment(split_value)
    if train_args and train_args.get("name") is not None:
        replacements["run_name"] = template_fragment(train_args["name"])
        replacements["name"] = template_fragment(train_args["name"])
    if extra:
        for key, value in extra.items():
            replacements[str(key)] = template_fragment(value)
    return replacements


def render_output_template(
    value: Any,
    config: Mapping[str, Any],
    timestamp: str,
    train_args: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Render an output path/name template using config-derived placeholders."""
    return render_config_template(value, build_output_template_replacements(config, timestamp, train_args, extra))


def render_output_dir_name(
    value: Any,
    config: Mapping[str, Any],
    timestamp: str,
    train_args: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render and sanitize a single output directory name."""
    return sanitize_name(str(render_output_template(value, config, timestamp, train_args, extra)))


def metrics_to_dict(metrics: Any) -> Dict[str, Any]:
    """Convert Ultralytics metric objects or dicts into a JSON-friendly dictionary."""
    if metrics is None:
        return {}
    if isinstance(metrics, Mapping):
        raw = dict(metrics)
    elif hasattr(metrics, "results_dict"):
        raw = dict(metrics.results_dict)
    else:
        raw = {"repr": repr(metrics)}

    result: Dict[str, Any] = {}
    for key, value in raw.items():
        value = json_safe_value(value)
        if isinstance(value, (int, float, str, bool)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = repr(value)
    return result


def json_safe_value(value: Any) -> Any:
    """Convert common numpy/path/scalar values into JSON-safe values."""
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


def sequence_item(values: Any, index: int, default: Any = None) -> Any:
    """Return values[index] from list-like/numpy-like containers."""
    safe_values = json_safe_value(values)
    if isinstance(safe_values, Sequence) and not isinstance(safe_values, (str, bytes)):
        return safe_values[index] if 0 <= index < len(safe_values) else default
    return default


def class_name_from_metrics(metrics: Any, class_id: int) -> str:
    """Resolve a class name from an Ultralytics metrics object."""
    names = getattr(metrics, "names", {})
    if isinstance(names, Mapping):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def per_class_metrics_rows(metrics: Any) -> List[Dict[str, Any]]:
    """Extract per-class detection metrics from an Ultralytics DetMetrics object."""
    if metrics is None or not hasattr(metrics, "ap_class_index"):
        return []

    class_indices = json_safe_value(getattr(metrics, "ap_class_index", []))
    if not isinstance(class_indices, Sequence) or isinstance(class_indices, (str, bytes)):
        return []

    box_metrics = getattr(metrics, "box", None)
    f1_values = getattr(box_metrics, "f1", [])
    all_ap = json_safe_value(getattr(box_metrics, "all_ap", []))
    nt_per_image = getattr(metrics, "nt_per_image", [])
    nt_per_class = getattr(metrics, "nt_per_class", [])
    rows: List[Dict[str, Any]] = []

    for metric_index, raw_class_id in enumerate(class_indices):
        class_id = int(raw_class_id)
        result = list(json_safe_value(metrics.class_result(metric_index))) if hasattr(metrics, "class_result") else []
        ap_row = sequence_item(all_ap, metric_index, [])
        rows.append(
            {
                "class_id": class_id,
                "class": class_name_from_metrics(metrics, class_id),
                "images": int(sequence_item(nt_per_image, class_id, 0) or 0),
                "instances": int(sequence_item(nt_per_class, class_id, 0) or 0),
                "precision": sequence_item(result, 0, 0.0),
                "recall": sequence_item(result, 1, 0.0),
                "f1": sequence_item(f1_values, metric_index, 0.0),
                "mAP50": sequence_item(result, 2, 0.0),
                "mAP75": sequence_item(ap_row, 5, None),
                "mAP50-95": sequence_item(result, 3, 0.0),
            }
        )
    return [{key: json_safe_value(value) for key, value in row.items()} for row in rows]


def get_with_fallback(mapping: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Return a config value, treating explicit YAML null the same as an omitted key."""
    value = mapping.get(key)
    return fallback if value is None else value


def write_metrics(output_dir: Path, metrics: Mapping[str, Any], prefix: str = "metrics") -> None:
    """Write metrics as JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(dict(metrics), file, indent=2, ensure_ascii=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def write_rows(output_dir: Path, rows: Sequence[Mapping[str, Any]], prefix: str, fieldnames: Sequence[str]) -> None:
    """Write a list of row dictionaries as JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_rows = [{str(key): json_safe_value(value) for key, value in row.items()} for row in rows]
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(safe_rows, file, indent=2, ensure_ascii=True)

    columns = list(fieldnames)
    for row in safe_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in safe_rows:
            writer.writerow(row)


def write_per_class_metrics(output_dir: Path, metrics: Any, prefix: str = "test_per_class_metrics") -> None:
    """Write per-class metrics extracted from an Ultralytics metrics object."""
    write_rows(output_dir, per_class_metrics_rows(metrics), prefix, PER_CLASS_METRIC_FIELDNAMES)


def copy_if_exists(src: Optional[Path], dst: Path) -> None:
    """Copy a file if it exists."""
    if src and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def dump_config_snapshot(
    output_dir: Path,
    merged_config: Mapping[str, Any],
    train_args: Mapping[str, Any],
    metadata: Mapping[str, Any],
    source_config: Optional[Path] = None,
    generated_dataset_yaml: Optional[Path] = None,
) -> None:
    """Save reproducibility config files inside an output folder."""
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config_dir / "merged_config.yaml", merged_config)
    save_yaml(config_dir / "ultralytics_train_args.yaml", train_args)
    with (config_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(dict(metadata), file, indent=2, ensure_ascii=True)
    copy_if_exists(source_config, config_dir / "source_config.yaml")
    copy_if_exists(generated_dataset_yaml, config_dir / "dataset.yaml")


def maybe_count_images(path: Optional[Path]) -> Optional[int]:
    """Count image files under a local directory, returning None if the path is unavailable."""
    if path is None or not path.exists():
        return None
    if path.is_file():
        if path.suffix.lower() in {".txt"}:
            try:
                with path.open("r", encoding="utf-8") as file:
                    return sum(1 for line in file if line.strip())
            except OSError:
                return None
        return 1 if path.suffix.lower() in IMAGE_EXTENSIONS else 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            total += 1
    return total


def local_dataset_path(root: Any, split: Any) -> Optional[Path]:
    """Resolve a dataset split to a local Path when possible."""
    if split is None or split == "":
        return None
    if isinstance(split, (list, tuple)):
        return None
    split_text = str(split)
    if is_abs_any_os(split_text):
        candidate = Path(split_text).expanduser()
    elif root:
        candidate = Path(str(root)).expanduser() / split_text
    else:
        candidate = Path(split_text).expanduser()
    return candidate


def load_dataset_info_from_yaml(data_yaml: Any) -> Dict[str, Any]:
    """Load basic dataset info from a YOLO data YAML when it is local."""
    resolved = resolve_existing_or_raw(data_yaml, [Path.cwd(), PROJECT_DIR, REPO_ROOT])
    path = Path(str(resolved)).expanduser()
    if not path.exists():
        return {}
    try:
        data = load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    data["_yaml_file"] = str(path)
    return data


def estimate_periodic_tests(train_args: Mapping[str, Any], periodic: Mapping[str, Any]) -> Tuple[Optional[int], str]:
    """Estimate how many scheduled test runs will be produced."""
    if not periodic.get("enabled", True):
        return 0, "disabled"

    epochs = train_args.get("epochs")
    epoch_interval = int(periodic.get("test_interval_epochs") or 0)
    minute_interval = float(periodic.get("test_interval_minutes") or 0.0)
    time_hours = train_args.get("time")

    by_epoch: Optional[int] = None
    by_time: Optional[int] = None
    if epochs and epoch_interval > 0:
        by_epoch = int(math.floor(float(epochs) / epoch_interval))
    if time_hours and minute_interval > 0:
        by_time = int(math.floor((float(time_hours) * 60.0) / minute_interval))

    candidates = [x for x in (by_epoch, by_time) if x is not None]
    if candidates:
        return max(candidates), "estimated from configured interval"
    if minute_interval > 0:
        return None, "unknown because training duration is epoch-based"
    return 0, "no interval configured"


def find_weight_size_bytes(model_value: Any, pretrained_value: Any) -> Optional[int]:
    """Find a likely checkpoint size from model/pretrained paths."""
    for value in (pretrained_value, model_value):
        if not isinstance(value, str) or not value.endswith(".pt"):
            continue
        resolved = resolve_existing_or_raw(value, [Path.cwd(), PROJECT_DIR, REPO_ROOT])
        path = Path(str(resolved)).expanduser()
        if path.exists():
            return path.stat().st_size
    return None


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


def estimate_outputs(
    config: Mapping[str, Any],
    train_args: Mapping[str, Any],
    data_yaml: Any,
    periodic_count: Optional[int],
) -> Dict[str, Any]:
    """Estimate output files and disk usage before training."""
    dataset_section = config.get("dataset", {})
    dataset_info = load_dataset_info_from_yaml(data_yaml) if data_yaml else {}
    dataset_root = dataset_info.get("path", dataset_section.get("path"))
    split_counts: Dict[str, Optional[int]] = {}
    for split_name in ("train", "val", "test"):
        split_value = dataset_info.get(split_name, dataset_section.get(split_name))
        split_counts[split_name] = maybe_count_images(local_dataset_path(dataset_root, split_value))

    epochs = int(train_args.get("epochs") or 100)
    save = bool(train_args.get("save", True))
    save_period = int(train_args.get("save_period", -1) or -1)
    checkpoint_files = 0
    if save:
        checkpoint_files = 2
        if save_period > 0:
            checkpoint_files += epochs // save_period

    periodic_cfg = config.get("periodic_test", {})
    plots = bool(train_args.get("plots", True))
    test_plots = bool(periodic_cfg.get("plots", plots))
    classwise_files = 2 if bool(periodic_cfg.get("classwise", False)) else 0
    base_train_files = 5 + checkpoint_files + (16 if plots else 0)
    per_test_files = 4 + classwise_files + (8 if test_plots else 0)
    periodic_files = None if periodic_count is None else periodic_count * per_test_files
    final_files = per_test_files if periodic_cfg.get("run_final_test", True) else 0
    total_known_files = base_train_files + final_files + (periodic_files or 0)

    weight_size = find_weight_size_bytes(str(config.get("model", "")), str(train_args.get("pretrained", "")))
    checkpoint_bytes = None
    if weight_size is not None:
        checkpoint_bytes = checkpoint_files * weight_size * 2.2
    plot_count_estimate = (16 if plots else 0) + (periodic_count or 0) * (8 if test_plots else 0) + ((8 if test_plots else 0) if final_files else 0)
    plot_bytes = plot_count_estimate * 1.5 * 1024 * 1024
    total_bytes = None if checkpoint_bytes is None else checkpoint_bytes + plot_bytes

    return {
        "split_image_counts": split_counts,
        "checkpoint_files": checkpoint_files,
        "periodic_test_files": periodic_files,
        "final_test_files": final_files,
        "estimated_total_files": total_known_files,
        "estimated_disk_usage": format_bytes(total_bytes),
        "note": "File and disk estimates are conservative approximations; label txt/json output depends on dataset size.",
    }


def confirm_or_exit(estimate: Mapping[str, Any], verbose: bool, assume_yes: bool) -> None:
    """Ask the developer for confirmation before running heavy output steps."""
    blue("Output and resource estimate before training:", verbose=verbose, force=True)
    print(json.dumps(dict(estimate), indent=2, ensure_ascii=True))
    if assume_yes:
        blue("Confirmation skipped because --yes or confirm_before_run=false is enabled.", verbose=verbose, force=True)
        return
    answer = input(Fore.BLUE + Style.BRIGHT + "Continue and start training? [y/N]: " + Style.RESET_ALL).strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted by developer before heavy output was produced.")


def make_generated_dataset_yaml(config: Mapping[str, Any], run_name: str) -> Optional[Path]:
    """Create a YOLO dataset YAML from the dataset section when data_yaml is not provided."""
    dataset = config.get("dataset", {})
    if dataset.get("data_yaml"):
        return None

    names = dataset.get("names")
    if not names:
        raise ValueError("dataset.names is required when dataset.data_yaml is empty.")

    generated_dir = PROJECT_DIR / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = generated_dir / f"{sanitize_name(run_name)}_dataset.yaml"
    data = {
        "path": dataset.get("path", ""),
        "train": dataset.get("train", "images/train"),
        "val": dataset.get("val", "images/val"),
        "test": dataset.get("test", "images/test"),
        "names": names,
    }
    if dataset.get("download"):
        data["download"] = dataset.get("download")
    save_yaml(output_path, data)
    return output_path


def parse_extra_args(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Parse --extra key=value items into Ultralytics training args."""
    parsed: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--extra must use key=value format, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"--extra contains an empty key: {item!r}.")
        parsed[key] = parse_scalar(value)
    return parsed


def apply_cli_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    """Apply first-class CLI overrides to the loaded config."""
    runtime = config.setdefault("runtime", {})
    dataset = config.setdefault("dataset", {})
    train = config.setdefault("train", {})
    periodic = config.setdefault("periodic_test", {})
    demo = config.setdefault("demo", {})

    direct_runtime = {
        "verbose": args.verbose,
        "dry_run": args.dry_run,
        "confirm_before_run": args.confirm_before_run,
    }
    for key, value in direct_runtime.items():
        if value is not None:
            runtime[key] = value

    for key in ("data_yaml", "path", "train", "val", "test"):
        value = getattr(args, f"dataset_{key}" if key == "path" else key, None)
        if value is not None:
            dataset[key] = value

    if args.model is not None:
        config["model"] = args.model
    if args.pretrained is not None:
        train["pretrained"] = args.pretrained

    train_overrides = {
        "epochs": args.epochs,
        "time": args.time,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
        "close_mosaic": args.close_mosaic,
        "save_period": args.save_period,
        "fraction": args.fraction,
        "multi_scale": args.multi_scale,
        "freeze": args.freeze,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "val_interval": args.val_interval,
    }
    for key, value in train_overrides.items():
        if value is not None:
            train[key] = value

    bool_train_overrides = {
        "save": args.save,
        "cache": args.cache,
        "exist_ok": args.exist_ok,
        "amp": args.amp,
        "plots": args.plots,
        "val": args.val_enabled,
        "rect": args.rect,
        "cos_lr": args.cos_lr,
        "resume": args.resume,
        "deterministic": args.deterministic,
        "half": args.half,
    }
    for key, value in bool_train_overrides.items():
        if value is not None:
            train[key] = value

    if args.test_interval_epochs is not None:
        periodic["test_interval_epochs"] = args.test_interval_epochs
    if args.test_interval_minutes is not None:
        periodic["test_interval_minutes"] = args.test_interval_minutes
    if args.final_test is not None:
        periodic["run_final_test"] = args.final_test
    if args.periodic_test is not None:
        periodic["enabled"] = args.periodic_test
    if args.periodic_test_classwise is not None:
        periodic["classwise"] = args.periodic_test_classwise

    if args.demo is not None:
        demo["enabled"] = args.demo

    extra_args = parse_extra_args(args.extra)
    if extra_args:
        train.setdefault("extra_yolo_args", {}).update(extra_args)

    if getattr(args, "task", None) is not None:
        config["task"] = args.task


def apply_demo_mode(config: MutableMapping[str, Any], timestamp: str, verbose: bool) -> None:
    """Clamp output and training settings for small demo runs."""
    demo = config.get("demo", {})
    if not demo.get("enabled", False):
        return

    train = config.setdefault("train", {})
    periodic = config.setdefault("periodic_test", {})
    max_epochs = int(demo.get("max_epochs", 2))
    max_fraction = float(demo.get("max_fraction", 0.05))
    max_batch = demo.get("max_batch", 4)

    train["epochs"] = min(int(train.get("epochs") or max_epochs), max_epochs)
    train["fraction"] = min(float(train.get("fraction", 1.0) or 1.0), max_fraction)
    if max_batch is not None:
        try:
            train["batch"] = min(int(train.get("batch") or max_batch), int(max_batch))
        except (TypeError, ValueError):
            train["batch"] = max_batch
    train["project"] = demo.get("project", str(PROJECT_DIR / "demo_runs"))
    train["name"] = f"{render_timestamped(train.get('name', 'demo_{timestamp}'), timestamp)}_demo"
    train["save_period"] = -1
    train["plots"] = bool(demo.get("plots", False))
    periodic["test_interval_epochs"] = int(demo.get("test_interval_epochs", 1))
    blue("Demo mode enabled: epochs, fraction, batch, plots, and output folder were clamped.", verbose)


def build_train_args(config: Mapping[str, Any], data_yaml: Any, timestamp: str) -> Dict[str, Any]:
    """Build Ultralytics train keyword arguments from config."""
    train = deepcopy(config.get("train", {}))
    extra = train.pop("extra_yolo_args", {}) or {}

    # val_interval is our custom parameter; extract before assembling Ultralytics args.
    val_interval = int(train.pop("val_interval", None) or 1)
    task_value = str(config.get("task", train.pop("task", "auto")) or "auto").strip().lower()
    if task_value not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task={task_value!r}. Options: {', '.join(sorted(SUPPORTED_TASKS))}.")

    model_value = resolve_model_source(config.get("model"), [Path.cwd(), PROJECT_DIR, REPO_ROOT])

    train_args: Dict[str, Any] = {}
    train_args.update(train)
    train_args.update(extra)
    train_args["data"] = str(data_yaml)
    if task_value != "auto":
        train_args["task"] = task_value
    train_args["mode"] = "train"
    train_args["name"] = render_output_template(train_args.get("name"), config, timestamp)
    train_args["project"] = render_output_template(train_args.get("project"), config, timestamp, train_args)
    train_args["pretrained"] = resolve_existing_or_raw(train_args.get("pretrained"), [Path.cwd(), PROJECT_DIR, REPO_ROOT])

    for key in ("cfg", "project"):
        train_args[key] = resolve_existing_or_raw(train_args.get(key), [Path.cwd(), PROJECT_DIR, REPO_ROOT])

    # Internal keys prefixed with _ are popped by main() before model.train().
    train_args["_model_source"] = model_value
    train_args["_task"] = task_value
    train_args["_val_interval"] = val_interval
    return train_args


def validate_requirements(config: Mapping[str, Any], train_args: Mapping[str, Any], data_yaml: Any) -> List[str]:
    """Check ambiguous or contradictory settings before training."""
    warnings: List[str] = []
    dataset = config.get("dataset", {})
    periodic = config.get("periodic_test", {})
    model_source = str(train_args.get("_model_source", ""))
    task_value = str(train_args.get("_task", "auto")).lower()
    data_info = load_dataset_info_from_yaml(data_yaml)

    if not MODEL_CONFIG_ROOT.exists():
        warnings.append(f"Model config root was not found: {MODEL_CONFIG_ROOT}")
    if dataset.get("data_yaml") and any(dataset.get(k) for k in ("path", "train", "val", "test")):
        warnings.append("Both dataset.data_yaml and dataset path/split fields are set; data_yaml has priority.")
    if periodic.get("enabled", True) and not periodic.get("test_interval_epochs") and not periodic.get("test_interval_minutes"):
        warnings.append("periodic_test.enabled=true but no interval is configured; only final test can run.")
    if periodic.get("enabled", True) and data_info and not data_info.get("test"):
        warnings.append("The selected dataset YAML does not define a test split; periodic test will be skipped.")
    if model_source == "yolo26m-p2.yaml":
        warnings.append(
            "The repo contains ultralytics/cfg/models/26/yolo26-p2.yaml, while yolo26m-p2.yaml is resolved by "
            "Ultralytics scale inference. If resolution fails, use model=ultralytics/cfg/models/26/yolo26-p2.yaml."
        )
    if model_source.endswith((".yaml", ".yml")) and not Path(model_source).expanduser().exists() and not is_url_like(model_source):
        warnings.append(
            f"Model YAML {model_source!r} was not found by the wrapper index. Ultralytics will try its own resolver."
        )
    if task_value != "auto" and task_value not in SUPPORTED_TASKS:
        warnings.append(f"task={task_value!r} is not one of {sorted(SUPPORTED_TASKS)}.")
    if task_value == "classify" and data_info.get("names"):
        warnings.append(
            "task=classify normally expects a classification dataset directory. A detection-style YAML may not work."
        )
    if task_value in {"segment", "pose", "obb"} and data_info:
        warnings.append(
            f"task={task_value} requires labels in that task format; the wrapper can load the model config but cannot "
            "convert detection labels automatically."
        )
    if periodic.get("classwise", False) and task_value == "classify":
        warnings.append("periodic_test.classwise is detection-style; classification metrics will still be saved normally.")
    device = str(train_args.get("device", ""))
    if periodic.get("enabled", True) and "," in device and device.lower() not in {"cpu", "mps"}:
        warnings.append(
            "Periodic in-process test callbacks are safest on one GPU/CPU. Ultralytics DDP multi-GPU training may "
            "run callbacks only inside subprocesses."
        )
    if train_args.get("save_txt") or train_args.get("save_json") or train_args.get("plots"):
        warnings.append("Plots/json/txt outputs can grow with test image count; the confirmation step includes this risk.")

    val_interval = int(train_args.get("_val_interval") or 1)
    if val_interval > 1:
        patience = int(train_args.get("patience") or 0)
        if 0 < patience < val_interval * 3:
            warnings.append(
                f"val_interval={val_interval} with patience={patience}: patience counts every training epoch. "
                f"Early stopping may fire after only {max(1, patience // val_interval)} val evaluations. "
                f"Consider setting patience to a multiple of val_interval (e.g. {val_interval * 10})."
            )

    return warnings


def _make_interval_val_trainer(val_interval: int, base_trainer_cls: type) -> type:
    """Return a task-specific Trainer subclass that runs validation only every val_interval epochs.

    On skipped epochs the previous val result is returned unchanged so that Ultralytics early
    stopping and best.pt selection track real val performance without saving non-validated weights.

    The returned fitness on skipped epochs is nudged one epsilon below best_fitness so that
    save_model() never overwrites best.pt with weights that have not been validated this epoch.
    """
    class IntervalValTrainer(base_trainer_cls):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._val_interval: int = val_interval
            self._cached_val_metrics: Optional[Dict[str, Any]] = None
            self._cached_val_fitness: Optional[float] = None

        def validate(self) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
            epoch_1 = getattr(self, "epoch", 0) + 1
            is_final = bool(getattr(self, "final_epoch", False))
            possible_stop = bool(getattr(getattr(self, "stopper", None), "possible_stop", False))
            must_run = (
                self._cached_val_metrics is None
                or is_final
                or possible_stop
                or epoch_1 % self._val_interval == 0
            )
            if must_run:
                m, f = super().validate()
                self._cached_val_metrics = m
                self._cached_val_fitness = f
                return m, f
            # Skipped epoch: return cached fitness nudged below best_fitness so save_model()
            # does not overwrite best.pt with non-validated epoch weights.
            cached_f = self._cached_val_fitness
            best = getattr(self, "best_fitness", None)
            if cached_f is not None and best is not None and cached_f >= best:
                cached_f = best - 1e-9
            return self._cached_val_metrics, cached_f

    IntervalValTrainer.__name__ = f"IntervalVal{base_trainer_cls.__name__}"
    return IntervalValTrainer


def periodic_test_mode(periodic: Mapping[str, Any]) -> str:
    """Return the configured periodic/final test mode."""
    return shared_modes.canonical_test_mode({"test_mode": periodic.get("test_mode", {"mode": "full_image"})})


def copy_test_batch_aliases(output_dir: Path) -> None:
    """Copy Ultralytics val_batch plots to test_batch names for test outputs."""
    for index in range(3):
        for suffix in ("labels", "pred"):
            src = output_dir / f"val_batch{index}_{suffix}.jpg"
            dst = output_dir / f"test_batch{index}_{suffix}.jpg"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)


def resolve_trainer_weight_path(trainer: Any, prefer_best: bool = False) -> Optional[Path]:
    """Find a saved Ultralytics checkpoint for image-level evaluator tests."""
    candidates = []
    if prefer_best:
        candidates.extend([getattr(trainer, "best", None), getattr(trainer, "last", None)])
    else:
        candidates.extend([getattr(trainer, "last", None), getattr(trainer, "best", None)])
    weights_dir = Path(getattr(trainer, "save_dir", ".")) / "weights"
    candidates.extend([weights_dir / "last.pt", weights_dir / "best.pt"])
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    return None


def build_yolo_evaluator_config(
    config: Mapping[str, Any],
    train_args: Mapping[str, Any],
    periodic: Mapping[str, Any],
    output_dir: Path,
    model_path: Path,
    mode: str,
) -> Dict[str, Any]:
    """Build a standalone evaluator config for YOLO non-full-image test modes."""
    max_det = get_with_fallback(periodic, "max_det", train_args.get("max_det", 300))
    conf = get_with_fallback(periodic, "conf", train_args.get("conf"))
    if conf is None:
        conf = 0.25
    iou = get_with_fallback(periodic, "iou", train_args.get("iou", 0.7))
    sahi_cfg = {
        "slice_height": int(periodic.get("slice_height", train_args.get("imgsz") or 640)),
        "slice_width": int(periodic.get("slice_width", train_args.get("imgsz") or 640)),
        "overlap_height_ratio": float(periodic.get("overlap_height_ratio", 0.2)),
        "overlap_width_ratio": float(periodic.get("overlap_width_ratio", 0.2)),
        "standard_prediction": bool(periodic.get("standard_prediction", True)),
        "postprocess_match_threshold": float(periodic.get("postprocess_match_threshold", 0.5)),
        "postprocess_class_agnostic": bool(periodic.get("postprocess_class_agnostic", False)),
        "batch_size": int(periodic.get("batch", 1) or 1),
    }
    sahi_cfg.update(periodic.get("sahi", {}) or {})
    extra_predict_args = {key: value for key, value in {"iou": iou, "max_det": max_det}.items() if value is not None}
    return {
        "runtime": {
            "verbose": bool(train_args.get("verbose", True)),
            "quiet": False,
            "confirm_before_run": False,
            "yes": True,
            "banner": "YOLO TEST EVALUATION",
            "seed": int(train_args.get("seed", 0) or 0),
        },
        "inference": {"mode": mode, "use_sahi": mode == shared_modes.SAHI_MODE},
        "test_mode": {"mode": mode},
        "dataset": {
            "format": "yolo",
            "split": str(periodic.get("split", "test")),
            "data_yaml": str(train_args["data"]),
            "include_empty_images": True,
            "sort_images": True,
            "max_images": periodic.get("max_images"),
        },
        "model": {
            "type": "ultralytics",
            "path": str(model_path),
            "confidence_threshold": float(conf),
            "device": train_args.get("device", "cpu"),
            "image_size": train_args.get("imgsz"),
            "extra_predict_args": extra_predict_args,
        },
        "sahi": sahi_cfg,
        "crop": periodic.get("crop", {}) or {},
        "evaluation": {
            "type": "bbox",
            "max_detections": [1, 10, int(max_det or 300)],
            "match_iou_threshold": 0.5,
            "operating_confidence_threshold": float(conf),
            "classwise": bool(periodic.get("classwise", True)),
            "per_image_metrics": True,
            "confusion_matrix": False,
            "curves": False,
            "save_coco_summary_text": True,
        },
        "output": {
            "output_dir": str(output_dir),
            "exist_ok": True,
            "save_config": True,
            "save_predictions_json": bool(periodic.get("save_json", True)),
            "save_ground_truth_json": True,
            "save_metrics": True,
            "save_plots": bool(periodic.get("plots", False)),
            "save_visuals": False,
            "save_dataset_cases": bool(periodic.get("save_dataset_cases", False)),
            "save_model_input_batches": True,
            "max_model_input_batches": 3,
            "model_input_batch_size": int(periodic.get("model_input_batch_size", 9) or 9),
        },
        "progress": {"images": True, "slices": False, "dataset_cases": False, "visuals": False},
    }


def write_evaluator_metric_aliases(output_dir: Path, result: Mapping[str, Any]) -> None:
    """Write legacy trainer metric filenames from evaluator results."""
    summary = dict(result.get("summary", {}))
    write_metrics(output_dir, summary, prefix="test_metrics")
    per_class = list(result.get("per_class", []) or [])
    if per_class:
        json_path = output_dir / "test_per_class_metrics.json"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(per_class, file, indent=2, ensure_ascii=True)
        fields = sorted({key for row in per_class for key in row.keys()})
        csv_path = output_dir / "test_per_class_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in per_class:
                writer.writerow(row)


def run_yolo_image_level_test(
    config: Mapping[str, Any],
    source_config: Optional[Path],
    train_args: Mapping[str, Any],
    periodic: Mapping[str, Any],
    output_dir: Path,
    model_path: Path,
    mode: str,
) -> None:
    """Run SAHI/class_crop test through the shared evaluator."""
    from projects.object_detection_dataset_evaluator.object_detection_dataset_evaluator import run_evaluation

    evaluator_config = build_yolo_evaluator_config(config, train_args, periodic, output_dir, model_path, mode)
    result = run_evaluation(evaluator_config, source_config or DEFAULT_CONFIG, print_summary=False)
    write_evaluator_metric_aliases(output_dir, result)


class PeriodicTestRunner:
    """Ultralytics callback object that runs test split evaluation after selected fit epochs."""

    def __init__(
        self,
        config: Mapping[str, Any],
        source_config: Optional[Path],
        generated_dataset_yaml: Optional[Path],
        train_args: Mapping[str, Any],
        verbose: bool,
    ) -> None:
        self.config = config
        self.source_config = source_config
        self.generated_dataset_yaml = generated_dataset_yaml
        self.train_args = train_args
        self.verbose = verbose
        self.started_at = time.monotonic()
        self.last_test_time = self.started_at
        self.tested_epochs: set = set()
        self.test_loader = None

    def should_run(self, epoch_number: int) -> bool:
        """Return True if epoch or time interval says to run test now."""
        periodic = self.config.get("periodic_test", {})
        if not periodic.get("enabled", True):
            return False
        by_epoch = int(periodic.get("test_interval_epochs") or 0)
        by_minutes = float(periodic.get("test_interval_minutes") or 0.0)
        if by_epoch > 0 and epoch_number % by_epoch == 0:
            return True
        if by_minutes > 0 and (time.monotonic() - self.last_test_time) >= by_minutes * 60.0:
            return True
        return False

    def __call__(self, trainer: Any) -> None:
        """Run test evaluation from an on_fit_epoch_end callback."""
        from ultralytics.utils import RANK

        if RANK not in {-1, 0}:
            return

        epoch_number = int(getattr(trainer, "epoch", -1)) + 1
        if epoch_number in self.tested_epochs or not self.should_run(epoch_number):
            return
        if not getattr(trainer, "data", {}).get("test"):
            blue("Periodic test skipped because dataset has no test split.", self.verbose)
            self.tested_epochs.add(epoch_number)
            return

        periodic = self.config.get("periodic_test", {})
        timestamp = str(self.config.get("runtime", {}).get("timestamp") or datetime.now().strftime(TIMESTAMP_FORMAT))
        output_name = render_output_dir_name(
            periodic.get("output_dir_name", "periodic_tests"),
            self.config,
            timestamp,
            self.train_args,
            {"event": "periodic_test", "epoch": epoch_number, "epoch4": f"{epoch_number:04d}"},
        )
        save_dir = Path(trainer.save_dir) / output_name / f"epoch_{epoch_number:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        blue(f"Running scheduled test split evaluation at epoch {epoch_number}.", self.verbose)

        mode = periodic_test_mode(periodic)
        if mode != shared_modes.FULL_IMAGE_MODE:
            model_path = resolve_trainer_weight_path(trainer, prefer_best=False)
            if model_path is None:
                blue("Scheduled image-level test skipped because no saved last.pt/best.pt checkpoint was found.", force=True)
                self.tested_epochs.add(epoch_number)
                return
            run_yolo_image_level_test(
                config=self.config,
                source_config=self.source_config,
                train_args=self.train_args,
                periodic=periodic,
                output_dir=save_dir,
                model_path=model_path,
                mode=mode,
            )
            self.tested_epochs.add(epoch_number)
            self.last_test_time = time.monotonic()
            blue(f"Scheduled {mode} test metrics saved to {save_dir}.", self.verbose)
            return

        if self.test_loader is None or not periodic.get("reuse_test_loader", True):
            batch_size = getattr(getattr(trainer, "test_loader", None), "batch_size", None) or int(trainer.batch_size) * 2
            self.test_loader = trainer.get_dataloader(trainer.data["test"], batch_size=batch_size, rank=-1, mode="val")

        validator_args = deepcopy(vars(trainer.validator.args))
        validator_args.update(
            {
                "split": "test",
                "plots": bool(periodic.get("plots", False)),
                "save_json": bool(periodic.get("save_json", False)),
                "save_txt": bool(periodic.get("save_txt", False)),
                "save_conf": bool(periodic.get("save_conf", False)),
                "conf": get_with_fallback(periodic, "conf", validator_args.get("conf")),
                "iou": get_with_fallback(periodic, "iou", validator_args.get("iou")),
                "max_det": get_with_fallback(periodic, "max_det", validator_args.get("max_det")),
                "verbose": bool(self.train_args.get("verbose", self.verbose)),
            }
        )
        validator_args["data"] = self.train_args["data"]
        validator_args["device"] = getattr(trainer.args, "device", self.train_args.get("device"))
        validator_args["task"] = getattr(trainer.args, "task", self.train_args.get("task", "detect"))
        validator = trainer.validator.__class__(
            self.test_loader,
            save_dir=save_dir,
            args=validator_args,
            _callbacks=trainer.callbacks,
        )
        source_model = trainer.ema.ema if getattr(trainer, "ema", None) is not None else trainer.model
        if getattr(trainer.args, "compile", False) and hasattr(source_model, "_orig_mod"):
            source_model = source_model._orig_mod
        # AutoBackend fuses PyTorch modules during standalone validation. Validate a throwaway copy so the
        # live training model/EMA keeps the same state_dict keys for subsequent EMA updates.
        model_for_test = deepcopy(source_model).eval()
        try:
            validation_result = validator(model=model_for_test)
            metrics = metrics_to_dict(validation_result)
        finally:
            del model_for_test
        copy_test_batch_aliases(save_dir)
        write_metrics(save_dir, metrics, prefix="test_metrics")
        if bool(periodic.get("classwise", False)):
            write_per_class_metrics(save_dir, getattr(validator, "metrics", None))
        dump_config_snapshot(
            save_dir,
            self.config,
            self.train_args,
            {
                "event": "periodic_test",
                "epoch": epoch_number,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "save_dir": str(save_dir),
            },
            self.source_config,
            self.generated_dataset_yaml,
        )
        self.tested_epochs.add(epoch_number)
        self.last_test_time = time.monotonic()
        blue(f"Scheduled test metrics saved to {save_dir}.", self.verbose)


def run_final_test(
    model: Any,
    config: Mapping[str, Any],
    train_args: Mapping[str, Any],
    source_config: Optional[Path],
    generated_dataset_yaml: Optional[Path],
    verbose: bool,
) -> None:
    """Run final test split evaluation with the trained model."""
    periodic = config.get("periodic_test", {})
    if not periodic.get("run_final_test", True):
        return
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        blue("Final test skipped because no trainer object was returned.", verbose)
        return
    if not getattr(trainer, "data", {}).get("test"):
        blue("Final test skipped because dataset has no test split.", verbose)
        return

    timestamp = str(config.get("runtime", {}).get("timestamp") or datetime.now().strftime(TIMESTAMP_FORMAT))
    output_name = render_output_dir_name(
        periodic.get("final_output_dir_name", "final_test"),
        config,
        timestamp,
        train_args,
        {"event": "final_test"},
    )
    output_dir = Path(trainer.save_dir) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    blue("Running final test split evaluation.", verbose)

    mode = periodic_test_mode(periodic)
    if mode != shared_modes.FULL_IMAGE_MODE:
        model_path = resolve_trainer_weight_path(trainer, prefer_best=True)
        if model_path is None:
            blue("Final image-level test skipped because no saved best.pt/last.pt checkpoint was found.", force=True)
            return
        run_yolo_image_level_test(
            config=config,
            source_config=source_config,
            train_args=train_args,
            periodic=periodic,
            output_dir=output_dir,
            model_path=model_path,
            mode=mode,
        )
        blue(f"Final {mode} test metrics saved to {output_dir}.", verbose)
        return

    val_args = {
        "data": train_args["data"],
        "split": "test",
        "imgsz": train_args.get("imgsz"),
        "batch": get_with_fallback(periodic, "batch", train_args.get("batch")),
        "device": train_args.get("device"),
        "project": str(Path(trainer.save_dir)),
        "name": output_name,
        "exist_ok": True,
        "plots": bool(periodic.get("plots", False)),
        "save_json": bool(periodic.get("save_json", False)),
        "save_txt": bool(periodic.get("save_txt", False)),
        "save_conf": bool(periodic.get("save_conf", False)),
        "conf": get_with_fallback(periodic, "conf", train_args.get("conf")),
        "iou": get_with_fallback(periodic, "iou", train_args.get("iou")),
        "max_det": get_with_fallback(periodic, "max_det", train_args.get("max_det")),
        "half": get_with_fallback(periodic, "half", train_args.get("half", False)),
        "task": train_args.get("task"),
        "verbose": bool(train_args.get("verbose", verbose)),
    }
    val_args = {key: value for key, value in val_args.items() if value is not None}
    val_metrics = model.val(**val_args)
    metrics = metrics_to_dict(val_metrics)
    copy_test_batch_aliases(output_dir)
    write_metrics(output_dir, metrics, prefix="test_metrics")
    if bool(periodic.get("classwise", False)):
        write_per_class_metrics(output_dir, val_metrics)
    dump_config_snapshot(
        output_dir,
        config,
        train_args,
        {
            "event": "final_test",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "save_dir": str(output_dir),
        },
        source_config,
        generated_dataset_yaml,
    )
    blue(f"Final test metrics saved to {output_dir}.", verbose)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    usage = """
Detailed usage:
  Configure most options in config/yolo_train.yaml, then override important values from CLI.
  The model can be any YAML under ultralytics/cfg/models by filename or relative path.
  Use --yes for non-interactive execution after you have accepted the output/resource estimate.
  Use --demo to write to the small demo output folder and clamp epochs/fraction/batch.

Example usage:
  # Minimal: use all defaults from config
  uv run python projects/yolo_trainer/train_ultralytics_model.py --yes

  # Override common training params
  uv run python projects/yolo_trainer/train_ultralytics_model.py --device 7 --epochs 1000 --batch 32 --yes

  # Use a model YAML from ultralytics/cfg/models by filename
  uv run python projects/yolo_trainer/train_ultralytics_model.py --model rtdetr-l.yaml --task auto --yes

  # Run val every 5 epochs (affects early stopping and best.pt), patience scaled to 30 val evals
  uv run python projects/yolo_trainer/train_ultralytics_model.py --val-interval 5 --patience 150 --yes

  # Val every 5 epochs + test-split check every 10 epochs
  uv run python projects/yolo_trainer/train_ultralytics_model.py --val-interval 5 --test-interval-epochs 10 --yes

  # Disable val entirely
  uv run python projects/yolo_trainer/train_ultralytics_model.py --val-interval 0 --yes

  # Dry run: validate config and estimate outputs without training
  uv run python projects/yolo_trainer/train_ultralytics_model.py --dry-run

  # Pass any Ultralytics argument not listed here
  uv run python projects/yolo_trainer/train_ultralytics_model.py --extra warmup_epochs=5 --extra nbs=64

  # Demo mode: clamp to tiny run
  uv run python projects/yolo_trainer/train_ultralytics_model.py --demo --yes
"""
    parser = argparse.ArgumentParser(
        description="Ultralytics model trainer with model-config lookup, configurable val interval, and periodic test evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=usage,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the YAML config file.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Validate config and estimate outputs only.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None, help="Enable blue wrapper logs.")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Disable blue wrapper logs.")
    parser.add_argument("--confirm-before-run", type=parse_bool, default=None, help="Ask before heavy output is created.")
    parser.add_argument("--demo", action="store_true", default=None, help="Enable small demo output mode.")
    parser.add_argument("--no-demo", dest="demo", action="store_false", help="Disable demo output mode.")

    parser.add_argument("--data-yaml", default=None, help="Existing YOLO dataset YAML.")
    parser.add_argument("--dataset-root", dest="dataset_path", default=None, help="Dataset root path.")
    parser.add_argument("--train", default=None, help="Train split path relative to dataset root.")
    parser.add_argument("--val", default=None, help="Val split path relative to dataset root.")
    parser.add_argument("--test", default=None, help="Test split path relative to dataset root.")

    parser.add_argument("--model", default=None, help="Model YAML/PT path, e.g. rtdetr-l.yaml or 26/yolo26-p2.yaml.")
    parser.add_argument(
        "--task",
        choices=sorted(SUPPORTED_TASKS),
        default=None,
        help="Task override. Use auto to infer from model YAML. Options: auto, detect, segment, classify, pose, obb.",
    )
    parser.add_argument("--pretrained", default=None, help="Pretrained weights path or true/false.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--time", type=float, default=None, help="Maximum training hours.")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size.")
    parser.add_argument("--batch", type=parse_scalar, default=None, help="Batch size, or AutoBatch fraction.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience in epochs.")
    parser.add_argument("--device", default=None, help="CUDA device ids like 0, 7, 0,1, or cpu.")
    parser.add_argument("--project", default=None, help="Ultralytics output project directory.")
    parser.add_argument("--name", default=None, help="Run name. Supports {timestamp}, {date}, and config placeholders like {train.imgsz}.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers.")
    parser.add_argument("--optimizer", default=None, help="Optimizer: auto, SGD, AdamW, etc.")

    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=None, help="Final LR fraction.")
    parser.add_argument("--momentum", type=float, default=None, help="SGD momentum or Adam beta1.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV hue augmentation.")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV saturation augmentation.")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV value augmentation.")
    parser.add_argument("--degrees", type=float, default=None, help="Rotation degrees.")
    parser.add_argument("--translate", type=float, default=None, help="Translation fraction.")
    parser.add_argument("--scale", type=float, default=None, help="Scale augmentation.")
    parser.add_argument("--shear", type=float, default=None, help="Shear degrees.")
    parser.add_argument("--perspective", type=float, default=None, help="Perspective augmentation.")
    parser.add_argument("--flipud", type=float, default=None, help="Vertical flip probability.")
    parser.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability.")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic probability.")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp probability.")
    parser.add_argument("--cutmix", type=float, default=None, help="CutMix probability.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="Disable mosaic for final N epochs.")
    parser.add_argument("--save-period", type=int, default=None, help="Save checkpoint every N epochs; -1 disables.")
    parser.add_argument("--fraction", type=float, default=None, help="Training data fraction.")
    parser.add_argument("--multi-scale", type=float, default=None, help="Multi-scale image size range fraction.")
    parser.add_argument("--freeze", type=parse_scalar, default=None, help="Freeze first N layers or list of layer ids.")
    parser.add_argument("--conf", type=float, default=None, help="Validation/test confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="Validation/test IoU threshold.")
    parser.add_argument("--max-det", type=int, default=None, help="Maximum detections per image.")

    parser.add_argument(
        "--val-interval", type=int, default=None,
        help="Run val every N epochs (1=every epoch, 5=every 5th, 0=disable). "
             "Affects early stopping and best.pt. With N>1, patience counts every training epoch; "
             "set patience=N*desired_val_evals for predictable early stopping.",
    )
    parser.add_argument("--save", type=parse_bool, default=None, help="Save checkpoints.")
    parser.add_argument("--cache", type=parse_scalar, default=None, help="Cache: false, true, ram, disk.")
    parser.add_argument("--exist-ok", type=parse_bool, default=None, help="Allow overwriting project/name.")
    parser.add_argument("--amp", type=parse_bool, default=None, help="Enable AMP training.")
    parser.add_argument("--plots", type=parse_bool, default=None, help="Save training/validation plots.")
    parser.add_argument("--val-enabled", type=parse_bool, default=None, help="Enable/disable val (true/false).")
    parser.add_argument("--rect", type=parse_bool, default=None, help="Use rectangular batching.")
    parser.add_argument("--cos-lr", type=parse_bool, default=None, help="Use cosine LR scheduler.")
    parser.add_argument("--resume", type=parse_bool, default=None, help="Resume training.")
    parser.add_argument("--deterministic", type=parse_bool, default=None, help="Use deterministic operations.")
    parser.add_argument("--half", type=parse_bool, default=None, help="Use FP16 for validation/test.")

    parser.add_argument("--periodic-test", type=parse_bool, default=None, help="Enable scheduled test-split evaluation.")
    parser.add_argument("--test-interval-epochs", type=int, default=None, help="Run test every N epochs.")
    parser.add_argument("--test-interval-minutes", type=float, default=None, help="Run test every N minutes.")
    parser.add_argument("--final-test", dest="final_test", action="store_true", default=None, help="Run final test.")
    parser.add_argument("--no-final-test", dest="final_test", action="store_false", help="Skip final test.")
    parser.add_argument(
        "--periodic-test-classwise",
        dest="periodic_test_classwise",
        action="store_true",
        default=None,
        help="Write per-class periodic/final test metrics.",
    )
    parser.add_argument(
        "--no-periodic-test-classwise",
        dest="periodic_test_classwise",
        action="store_false",
        help="Disable per-class periodic/final test metrics.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=None,
        help="Additional Ultralytics argument as key=value. Can be repeated.",
    )
    return parser


def main() -> int:
    """
    Main entry point for YOLO training with configurable val interval and periodic test evaluation.

    Overview
    --------
    This script wraps the Ultralytics YOLO Python API to provide:
      - Training from any YAML under ultralytics/cfg/models, resolved by filename or relative path.
      - Configurable validation interval (val_interval): run val every N epochs instead of every
        epoch. On non-val epochs the previous result is reused, keeping early stopping and best.pt
        tracking valid without saving non-validated weights as best.
      - Periodic test-split evaluation on a schedule (every N epochs or N minutes).
      - Final test evaluation after training.
      - Pre-run resource estimate + developer confirmation before large outputs are written.
      - Config snapshots in every output directory for full reproducibility.

    Configuration
    -------------
    Edit projects/yolo_trainer/config/yolo_train.yaml for persistent settings.
    Set model to any YAML in ultralytics/cfg/models, for example rtdetr-l.yaml,
    26/yolo26-p2.yaml, v8/yolov8-seg.yaml, or an absolute YAML path.
    Set task=auto to infer detect/segment/classify/pose/obb from the model config.
    Any YAML value can be overridden from the CLI (see --help for all flags).
    Unknown Ultralytics arguments can be passed with --extra key=value.

    Val interval behaviour
    ----------------------
    train.val_interval=1    Every epoch (default, identical to stock Ultralytics).
    train.val_interval=5    Val runs on epochs 5, 10, 15 ... On other epochs the
                            previous val fitness is returned so that:
                              - best.pt is saved only when a real val epoch improves fitness.
                              - patience counts every training epoch (val and non-val).
                            Rule of thumb: patience = desired_val_count * val_interval.
                            Example: 30 val evaluations -> patience = 30 * 5 = 150.
    train.val_interval=0    Val disabled entirely (same as val=false).

    Usage examples
    --------------
    # All settings from config/yolo_train.yaml
    uv run python projects/yolo_trainer/train_ultralytics_model.py --yes

    # Override device, epochs, batch
    uv run python projects/yolo_trainer/train_ultralytics_model.py --device 7 --epochs 1000 --batch 32 --yes

    # Use any bundled model config by filename or subfolder path
    uv run python projects/yolo_trainer/train_ultralytics_model.py --model rtdetr-l.yaml --task auto --yes
    uv run python projects/yolo_trainer/train_ultralytics_model.py --model v8/yolov8-seg.yaml --task auto --yes

    # Val every 5 epochs, patience scaled to 30 val evaluations
    uv run python projects/yolo_trainer/train_ultralytics_model.py --val-interval 5 --patience 150 --yes

    # Val every 5 epochs + test-split check every 10 epochs
    uv run python projects/yolo_trainer/train_ultralytics_model.py --val-interval 5 --test-interval-epochs 10 --yes

    # Disable val entirely
    uv run python projects/yolo_trainer/train_ultralytics_model.py --val-interval 0 --yes

    # Custom config file
    uv run python projects/yolo_trainer/train_ultralytics_model.py \\
        --config /path/to/my_experiment.yaml --yes

    # Dry run: validate config and print resource estimate, do not train
    uv run python projects/yolo_trainer/train_ultralytics_model.py --dry-run

    # Demo mode: clamp to tiny run (2 epochs, 5% data, batch 4)
    uv run python projects/yolo_trainer/train_ultralytics_model.py --demo --yes

    # Pass arbitrary Ultralytics args not exposed as first-class flags
    uv run python projects/yolo_trainer/train_ultralytics_model.py --extra warmup_epochs=5 --extra nbs=64

    # Resume interrupted training
    uv run python projects/yolo_trainer/train_ultralytics_model.py --resume true --yes
    """
    parser = build_parser()
    args = parser.parse_args()
    source_config = Path(args.config).expanduser()
    if not source_config.is_absolute():
        source_config = (Path.cwd() / source_config).resolve()
    if not source_config.exists():
        raise FileNotFoundError(f"Config file not found: {source_config}")

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    with tqdm(total=6, desc="Preparing") as bar:
        config = load_yaml(source_config)
        apply_cli_overrides(config, args)
        config.setdefault("runtime", {})["timestamp"] = timestamp
        verbose = bool(config.get("runtime", {}).get("verbose", True))
        apply_demo_mode(config, timestamp, verbose)
        bar.update(1)

        dataset = config.get("dataset", {})
        run_name = str(render_output_template(config.get("train", {}).get("name", f"run_{timestamp}"), config, timestamp))
        generated_dataset_yaml = make_generated_dataset_yaml(config, run_name)
        data_yaml = generated_dataset_yaml or resolve_existing_or_raw(
            dataset.get("data_yaml"), [Path.cwd(), PROJECT_DIR, REPO_ROOT]
        )
        if not data_yaml:
            raise ValueError("No dataset YAML could be prepared. Set dataset.data_yaml or dataset.names/path/splits.")
        bar.set_description("Building args")
        bar.update(1)

        train_args = build_train_args(config, data_yaml, timestamp)
        train_cfg = config.setdefault("train", {})
        train_cfg["resolved_project"] = train_args.get("project")
        train_cfg["resolved_name"] = train_args.get("name")
        model_source = train_args.pop("_model_source")
        task_override = str(train_args.pop("_task", "auto")).lower()
        effective_task = infer_task_from_model_source(model_source, task_override)
        val_interval = int(train_args.pop("_val_interval") or 1)

        warnings = validate_requirements(
            config,
            {
                **train_args,
                "_model_source": model_source,
                "_task": effective_task,
                "_val_interval": val_interval,
            },
            data_yaml,
        )
        for warning in warnings:
            blue(f"Requirement check warning: {warning}", verbose=verbose, force=True)
        bar.set_description("Estimating output")
        bar.update(1)

        periodic_count, periodic_note = estimate_periodic_tests(train_args, config.get("periodic_test", {}))
        estimate = estimate_outputs(config, train_args, data_yaml, periodic_count)
        estimate["periodic_test_count"] = periodic_count if periodic_count is not None else periodic_note
        estimate["model_source"] = relative_to_repo(model_source)
        estimate["task"] = effective_task
        estimate["task_override"] = task_override
        estimate["val_interval"] = val_interval
        confirm = bool(config.get("runtime", {}).get("confirm_before_run", True))
        assume_yes = bool(args.yes or not confirm)
        confirm_or_exit(estimate, verbose=verbose, assume_yes=assume_yes)
        bar.set_description("Confirmed")
        bar.update(1)

        if bool(config.get("runtime", {}).get("dry_run", False)):
            blue("Dry run complete. Training was not started.", verbose=verbose, force=True)
            bar.update(bar.total - bar.n)
            return 0

        if train_args.get("project"):
            Path(str(train_args["project"])).mkdir(parents=True, exist_ok=True)

        blue("Importing Ultralytics and preparing model.", verbose)
        model, detected_task = create_ultralytics_model(
            model_source,
            task_override,
            verbose=bool(train_args.get("verbose", verbose)),
        )
        detected_task = str(getattr(model, "task", detected_task))
        train_args["task"] = detected_task
        blue(f"Resolved model source: {relative_to_repo(model_source)}", verbose)
        blue(f"Resolved Ultralytics task: {detected_task}", verbose)

        # Resolve val_interval: 0=disable val, 1=every epoch (no custom trainer), N>1=task-specific custom trainer.
        trainer_cls = None
        if val_interval == 0:
            train_args["val"] = False
            blue("val_interval=0: validation disabled.", verbose)
        elif val_interval > 1:
            train_args["val"] = True  # ensure Ultralytics calls validate() every epoch so we intercept
            base_trainer_cls = model.task_map.get(detected_task, {}).get("trainer")
            if base_trainer_cls is None:
                raise ValueError(f"No Ultralytics trainer found for task={detected_task!r}.")
            trainer_cls = _make_interval_val_trainer(val_interval, base_trainer_cls)
            blue(f"val_interval={val_interval}: using {trainer_cls.__name__}.", verbose)

        snapshot_callback = PeriodicTestRunner(
            config=config,
            source_config=source_config,
            generated_dataset_yaml=generated_dataset_yaml,
            train_args=train_args,
            verbose=verbose,
        )
        model.add_callback("on_pretrain_routine_start", lambda trainer: dump_config_snapshot(
            Path(trainer.save_dir),
            config,
            train_args,
            {
                "event": "train_start",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "argv": sys.argv,
                "cwd": str(Path.cwd()),
                "save_dir": str(trainer.save_dir),
                "val_interval": val_interval,
            },
            source_config,
            generated_dataset_yaml,
        ))
        model.add_callback("on_fit_epoch_end", snapshot_callback)
        bar.set_description("Training")
        bar.update(1)

        blue("Starting YOLO training.", verbose)
        model.train(trainer=trainer_cls, **train_args)
        bar.set_description("Final test")
        bar.update(1)

        run_final_test(model, config, train_args, source_config, generated_dataset_yaml, verbose)
        bar.set_description("Done")
        bar.update(1)

    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        blue(f"Training output directory: {Path(trainer.save_dir)}", verbose=verbose, force=True)
    blue("Done.", verbose=verbose, force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
