"""Shared config, path, device, and network helpers for the D-FINE-seg wrapper."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from string import Formatter
from typing import Any, Mapping, MutableMapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
VENDOR_DIR = PROJECT_DIR / "vendor" / "D-FINE-seg"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "dfine_seg_train.yaml"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".npy"}
VENDORED_DFINE_COMMIT = "0a0f0a12511568857922924854a63d06e1ae0fbd"


def now_timestamp() -> str:
    """Return the compact timestamp used in output placeholders."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def parse_bool(value: Any) -> bool:
    """Parse common CLI/config boolean strings."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_scalar(value: str) -> Any:
    """Parse a CLI scalar/list/dict with YAML semantics."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def parse_extra_args(values: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeatable key=value CLI overrides."""
    parsed: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected key=value, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"Expected non-empty key in {item!r}.")
        set_by_dot_path(parsed, key, parse_scalar(value))
    return parsed


def set_by_dot_path(data: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set nested mapping values using dot notation."""
    parts = key.split(".")
    cursor: MutableMapping[str, Any] = data
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, MutableMapping):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def get_by_dot_path(data: Mapping[str, Any], key: str, default: Any = "") -> Any:
    """Read nested mapping values using dot notation."""
    cursor: Any = data
    for part in key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge a mapping into ``base``."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def save_yaml(path: Path, data: Mapping[str, Any]) -> None:
    """Save a YAML file with readable Unicode output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(data), file, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: Any) -> None:
    """Write JSON with UTF-8 and stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, indent=2, ensure_ascii=False)


def json_safe(value: Any) -> Any:
    """Convert common path/scalar values into JSON-safe structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return repr(value)
    return value


def is_abs_any_os(value: str) -> bool:
    """Return True for POSIX or Windows absolute paths on any host OS."""
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def resolve_path(value: Any, base: Path = REPO_ROOT, must_exist: bool = False) -> Path:
    """Resolve config paths from the repo root unless already absolute."""
    if value is None or str(value).strip() == "":
        raise ValueError("Path value cannot be empty.")
    text = str(value).strip()
    path = Path(text).expanduser()
    resolved = path if is_abs_any_os(text) else (base / path)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved.resolve()


def sanitize_name(value: Any) -> str:
    """Make a filesystem-safe path component."""
    text = re.sub(r"\s+", "_", str(value).strip())
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    return text.strip(" ._") or "run"


def placeholder_value(value: Any) -> str:
    """Convert a config value into a safe placeholder fragment."""
    if value is None or value == "":
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p")
    if isinstance(value, (list, tuple)):
        return "-".join(placeholder_value(item) for item in value)
    return sanitize_name(value)


def output_context(config: Mapping[str, Any], timestamp: str, region: str) -> dict[str, str]:
    """Build supported placeholders for output folder templates."""
    dataset_dir = get_by_dot_path(config, "dataset.dataset_dir", "")
    dataset_name = Path(str(dataset_dir)).name if str(dataset_dir).strip() else "dataset"
    raw: dict[str, Any] = {
        "timestamp": timestamp,
        "date": timestamp[:8],
        "time": timestamp[8:],
        "task": get_by_dot_path(config, "model.task", "segment"),
        "model_name": get_by_dot_path(config, "model.name", "s"),
        "dataset_name": dataset_name,
        "dataset_format": get_by_dot_path(config, "dataset.source_format", "auto"),
        "epochs": get_by_dot_path(config, "train.epochs", "epochs"),
        "batch_size": get_by_dot_path(config, "train.batch_size", "batch"),
        "device": get_by_dot_path(config, "train.device", "auto"),
        "gpus": get_by_dot_path(config, "train.gpus", "auto"),
        "region": region,
        "imgsz": "-".join(str(x) for x in get_by_dot_path(config, "train.img_size", [640, 640])),
        "lr": get_by_dot_path(config, "train.base_lr", "lr"),
        "workers": get_by_dot_path(config, "train.num_workers", "workers"),
    }
    return {key: placeholder_value(value) for key, value in raw.items()}


def render_template(value: Any, config: Mapping[str, Any], timestamp: str, region: str) -> Any:
    """Render output placeholders in a string."""
    if not isinstance(value, str):
        return value
    fields = {field for _, field, _, _ in Formatter().parse(value) if field}
    if not fields:
        return value
    context = output_context(config, timestamp, region)
    unknown = sorted(field for field in fields if field not in context)
    if unknown:
        available = ", ".join(sorted(context))
        raise ValueError(f"Unknown output placeholder(s): {', '.join(unknown)}. Available: {available}")
    return value.format_map(context)


def build_output_dir(config: Mapping[str, Any], timestamp: str, region: str) -> Path:
    """Resolve the final output directory."""
    output = config.get("output", {}) if isinstance(config.get("output"), Mapping) else {}
    exact = str(output.get("output_dir") or "").strip()
    if exact:
        return resolve_path(render_template(exact, config, timestamp, region), REPO_ROOT, must_exist=False)
    root = render_template(output.get("root", "runs/detect/dfine_seg/train"), config, timestamp, region)
    name = render_template(output.get("name", "dfine_{task}_{model_name}_{timestamp}"), config, timestamp, region)
    return (resolve_path(root, REPO_ROOT, must_exist=False) / sanitize_name(name)).resolve()


def detect_ip_region(timeout: float = 2.5) -> dict[str, Any]:
    """Best-effort IP country detection without raising on offline machines."""
    providers = [
        ("ipapi", "https://ipapi.co/json/"),
        ("ip-api", "http://ip-api.com/json/?fields=status,countryCode,country,query"),
    ]
    for provider, url in providers:
        try:
            request = Request(url, headers={"User-Agent": "dfine-seg-trainer/0.1"})
            with urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
            code = str(data.get("country_code") or data.get("countryCode") or "").upper()
            if code:
                return {"provider": provider, "country_code": code, "raw": data}
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            continue
    return {"provider": None, "country_code": "UNKNOWN", "raw": {}}


def resolve_region(config: Mapping[str, Any]) -> str:
    """Resolve network region from config or IP detection."""
    configured = str(get_by_dot_path(config, "network.region", "auto")).strip().lower()
    if configured in {"china", "cn", "mainland_china"}:
        return "china"
    if configured in {"global", "overseas", "international"}:
        return "global"
    detected = detect_ip_region()
    return "china" if detected.get("country_code") == "CN" else "global"


def run_command(command: Sequence[str], cwd: Path, env: Mapping[str, str] | None = None) -> int:
    """Run a subprocess while streaming output."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    process = subprocess.Popen(list(command), cwd=str(cwd), env=merged_env)
    return int(process.wait())


def command_output(command: Sequence[str], timeout: float = 8.0) -> str:
    """Return command output, or an empty string when unavailable."""
    try:
        return subprocess.check_output(list(command), stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except Exception:
        return ""


def detect_nvidia() -> list[dict[str, Any]]:
    """Return NVIDIA GPU information from nvidia-smi if available."""
    if shutil.which("nvidia-smi") is None:
        return []
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": int(float(parts[3])) if parts[3].replace(".", "", 1).isdigit() else parts[3],
            }
        )
    return gpus


def detect_cuda_banner() -> str:
    """Read the CUDA version shown by nvidia-smi, if any."""
    output = command_output(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else ""


def parse_device_ids(value: Any) -> list[int]:
    """Parse device/gpu selectors into CUDA integer ids."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value] if value >= 0 else []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value if str(item).strip().lstrip("-").isdigit() and int(item) >= 0]
    text = str(value).strip().lower()
    if text in {"", "auto", "cuda"}:
        return []
    if text in {"cpu", "mps", "-1"}:
        return []
    ids: list[int] = []
    for token in text.replace("cuda:", "").split(","):
        token = token.strip()
        if token.isdigit():
            ids.append(int(token))
    return ids


def resolve_training_device(config: MutableMapping[str, Any]) -> tuple[str, list[int]]:
    """Resolve device and selected GPUs from config."""
    train = config.setdefault("train", {})
    device = str(train.get("device", "auto")).strip().lower()
    gpus_value = train.get("gpus", "auto")
    available = detect_nvidia()

    if device in {"cpu", "mps"}:
        return device, []
    explicit = parse_device_ids(gpus_value) or parse_device_ids(device)
    if explicit:
        train["device"] = f"cuda:{explicit[0]}"
        train["gpus"] = explicit
        return train["device"], explicit
    if device == "auto":
        if available:
            train["device"] = "cuda"
            train["gpus"] = [int(available[0]["index"])]
            return "cuda", [int(available[0]["index"])]
        train["device"] = "cpu"
        train["gpus"] = []
        return "cpu", []
    if device == "cuda" and available:
        train["gpus"] = [int(available[0]["index"])]
        return "cuda", [int(available[0]["index"])]
    return device, []


def environment_snapshot(region: str) -> dict[str, Any]:
    """Collect lightweight environment metadata for reproducibility."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "region": region,
        "nvidia_gpus": detect_nvidia(),
        "nvidia_cuda_banner": detect_cuda_banner(),
        "vendored_dfine_commit": VENDORED_DFINE_COMMIT,
    }


def format_bytes(num_bytes: float | int | None) -> str:
    """Format byte counts for terminal display."""
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def copy_config_for_mutation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable deep copy of a config mapping."""
    return deepcopy(dict(config))


def wait_for_enter_confirmation(prompt: str) -> bool:
    """Prompt for yes/no confirmation."""
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def touch_with_timestamp(path: Path) -> None:
    """Create a small timestamp marker file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")
