"""Bound RF-DETR numerical thread pools before heavy libraries initialize.

The public entrypoints call :func:`bootstrap_from_argv` before importing
PyTorch, NumPy, or OpenCV. Ordinary library imports do not call this module's
bootstrap functions and therefore retain the embedding process' settings.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from rf_detr_config import load_yaml


DEFAULT_CPU_LIMIT_ENABLED = True
DEFAULT_CPU_BUDGET_PERCENT = 50.0
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class CpuRuntimePolicy:
    """Resolved per-process numerical thread policy."""

    task: str
    enabled: bool
    budget_percent: float
    logical_cpus: int
    total_thread_budget: Optional[int]
    model_processes: int
    threads_per_process: Optional[int]
    source_config: str


_ACTIVE_POLICY: Optional[CpuRuntimePolicy] = None
_ACTIVE_SUMMARY: Optional[Dict[str, Any]] = None
_RUNTIME_APPLIED = False


def parse_bool(value: Any, field_name: str) -> bool:
    """Parse a strict human-friendly boolean value."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{field_name} must be true or false, got {value!r}.")


def validate_budget_percent(value: Any) -> float:
    """Return a finite CPU budget percentage in the inclusive range 1..100."""
    if isinstance(value, bool):
        raise ValueError("runtime.cpu.budget_percent must be a number from 1 to 100, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"runtime.cpu.budget_percent must be a number from 1 to 100, got {value!r}."
        ) from exc
    if not math.isfinite(parsed) or not 1.0 <= parsed <= 100.0:
        raise ValueError(
            f"runtime.cpu.budget_percent must be between 1 and 100, got {value!r}."
        )
    return parsed


def cpu_settings(config: Mapping[str, Any]) -> tuple[bool, float]:
    """Read validated ``runtime.cpu`` settings with stable defaults."""
    runtime = config.get("runtime", {})
    if runtime is None:
        runtime = {}
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be a mapping.")
    cpu = runtime.get("cpu", {})
    if cpu is None:
        cpu = {}
    if not isinstance(cpu, Mapping):
        raise ValueError("runtime.cpu must be a mapping.")
    enabled = parse_bool(cpu.get("enabled", DEFAULT_CPU_LIMIT_ENABLED), "runtime.cpu.enabled")
    budget_percent = validate_budget_percent(cpu.get("budget_percent", DEFAULT_CPU_BUDGET_PERCENT))
    return enabled, budget_percent


def add_cpu_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared CPU-budget CLI to an entrypoint parser."""
    parser.add_argument(
        "--cpu-limit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable RF-DETR numerical thread limits (use --no-cpu-limit to disable).",
    )
    parser.add_argument(
        "--cpu-budget-percent",
        type=float,
        default=None,
        help="Logical-CPU thread budget percentage, from 1 to 100 (default: 50).",
    )


def apply_cpu_cli_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    """Persist CPU CLI overrides into the merged config snapshot."""
    cpu_limit = getattr(args, "cpu_limit", None)
    cpu_budget_percent = getattr(args, "cpu_budget_percent", None)
    if cpu_limit is None and cpu_budget_percent is None:
        return
    runtime = config.setdefault("runtime", {})
    if not isinstance(runtime, MutableMapping):
        raise ValueError("runtime must be a mapping.")
    cpu = runtime.setdefault("cpu", {})
    if not isinstance(cpu, MutableMapping):
        raise ValueError("runtime.cpu must be a mapping.")
    if cpu_limit is not None:
        cpu["enabled"] = bool(cpu_limit)
    if cpu_budget_percent is not None:
        cpu["budget_percent"] = validate_budget_percent(cpu_budget_percent)


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _device_process_count(value: Any, *, integer_is_count: bool) -> Optional[int]:
    """Infer an explicit device count without probing CUDA."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        if integer_is_count:
            return value if value > 0 else None
        return 1
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return max(1, len(value))
    text = str(value).strip().lower()
    if text in {"", "auto", "cpu", "cuda", "gpu", "mps", "-1"}:
        return None
    if "," in text:
        items = [item.strip() for item in text.split(",") if item.strip()]
        return max(1, len(items))
    if text.startswith("cuda:"):
        return 1
    if text.isdecimal():
        return int(text) if integer_is_count and int(text) > 0 else 1
    return None


def infer_model_processes(
    config: Mapping[str, Any],
    task: str,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Infer concurrently active model processes for thread-budget division."""
    environment = os.environ if env is None else env
    world_size = _positive_int(environment.get("WORLD_SIZE")) or 1
    normalized_task = str(task).strip().lower()
    configured = 1
    if normalized_task == "test":
        test = config.get("test", {})
        if isinstance(test, Mapping):
            parallel = test.get("parallel", {})
            if parallel is None:
                parallel = {}
            if not isinstance(parallel, Mapping):
                raise ValueError("test.parallel must be a mapping.")
            raw_chunks = parallel.get("chunks", 1)
            chunks = _positive_int(raw_chunks)
            if chunks is None:
                raise ValueError("test.parallel.chunks must be a positive integer.")
            configured = chunks
    elif normalized_task == "train":
        train = config.get("train", {})
        if not isinstance(train, Mapping):
            train = {}
        trainer = config.get("trainer", {})
        if not isinstance(trainer, Mapping):
            trainer = {}
        extra = trainer.get("extra_trainer_args", {})
        if not isinstance(extra, Mapping):
            extra = {}
        configured = (
            _device_process_count(extra.get("devices"), integer_is_count=True)
            or _device_process_count(train.get("devices"), integer_is_count=True)
            or _device_process_count(train.get("device"), integer_is_count=False)
            or 1
        )
        num_nodes = _positive_int(extra.get("num_nodes")) or _positive_int(train.get("num_nodes")) or 1
        configured *= num_nodes
    return max(1, world_size, configured)


def resolve_cpu_policy(
    config: Mapping[str, Any],
    task: str,
    source_config: Path,
    *,
    logical_cpus: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
) -> CpuRuntimePolicy:
    """Resolve and validate the complete CPU policy for one process."""
    enabled, budget_percent = cpu_settings(config)
    available = max(1, int(logical_cpus if logical_cpus is not None else (os.cpu_count() or 1)))
    model_processes = infer_model_processes(config, task, env)
    total_budget: Optional[int] = None
    threads_per_process: Optional[int] = None
    if enabled:
        total_budget = max(1, int(math.floor(available * budget_percent / 100.0)))
        if model_processes > total_budget:
            raise ValueError(
                "Configured model process count exceeds the CPU thread budget: "
                f"processes={model_processes}, budget={total_budget}, logical_cpus={available}, "
                f"budget_percent={budget_percent:g}. Reduce test.parallel.chunks/train devices, "
                "increase runtime.cpu.budget_percent, or use --no-cpu-limit explicitly."
            )
        threads_per_process = max(1, total_budget // model_processes)
    return CpuRuntimePolicy(
        task=str(task),
        enabled=enabled,
        budget_percent=budget_percent,
        logical_cpus=available,
        total_thread_budget=total_budget,
        model_processes=model_processes,
        threads_per_process=threads_per_process,
        source_config=str(source_config.expanduser().resolve()),
    )


def _preparse_args(
    argv: Optional[Sequence[str]],
    default_config: Path,
    task: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", default=str(default_config))
    add_cpu_cli_arguments(parser)
    normalized_task = str(task).strip().lower()
    if normalized_task == "test":
        parser.add_argument("--chunks", type=int, default=None)
    elif normalized_task == "train":
        parser.add_argument("--device", default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def apply_topology_cli_overrides(
    config: MutableMapping[str, Any],
    args: argparse.Namespace,
    task: str,
) -> None:
    """Apply process-count CLI fields needed before numerical imports."""
    normalized_task = str(task).strip().lower()
    if normalized_task == "test" and getattr(args, "chunks", None) is not None:
        test = config.setdefault("test", {})
        if not isinstance(test, MutableMapping):
            raise ValueError("test must be a mapping.")
        parallel = test.setdefault("parallel", {})
        if not isinstance(parallel, MutableMapping):
            raise ValueError("test.parallel must be a mapping.")
        parallel["chunks"] = args.chunks
    elif normalized_task == "train" and getattr(args, "device", None) is not None:
        train = config.setdefault("train", {})
        if not isinstance(train, MutableMapping):
            raise ValueError("train must be a mapping.")
        train["device"] = args.device


def bootstrap_from_argv(
    default_config: Path,
    task: str,
    argv: Optional[Sequence[str]] = None,
) -> CpuRuntimePolicy:
    """Resolve CLI/YAML settings and set thread env vars before numerical imports."""
    global _ACTIVE_POLICY, _ACTIVE_SUMMARY, _RUNTIME_APPLIED
    args = _preparse_args(argv, default_config, task)
    source_config = Path(args.config).expanduser()
    if not source_config.is_absolute():
        source_config = (Path.cwd() / source_config).resolve()
    config = load_yaml(source_config)
    apply_topology_cli_overrides(config, args, task)
    apply_cpu_cli_overrides(config, args)
    policy = resolve_cpu_policy(config, task, source_config)
    if policy.enabled:
        assert policy.threads_per_process is not None
        thread_text = str(policy.threads_per_process)
        for name in THREAD_ENVIRONMENT_VARIABLES:
            os.environ[name] = thread_text
    _ACTIVE_POLICY = policy
    _ACTIVE_SUMMARY = {
        **asdict(policy),
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
        },
    }
    _RUNTIME_APPLIED = False
    return policy


def apply_loaded_runtime(policy: Optional[CpuRuntimePolicy] = None) -> Dict[str, Any]:
    """Apply PyTorch/OpenCV runtime APIs once, after imports but before work starts."""
    global _ACTIVE_SUMMARY, _RUNTIME_APPLIED
    active = policy or _ACTIVE_POLICY
    if active is None:
        return {}
    if _RUNTIME_APPLIED:
        return dict(_ACTIVE_SUMMARY or {})
    summary = {
        **asdict(active),
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
        },
    }
    import torch

    if active.enabled:
        assert active.threads_per_process is not None
        torch.set_num_threads(active.threads_per_process)
        torch.set_num_interop_threads(1)
    summary["torch_intraop_threads"] = int(torch.get_num_threads())
    summary["torch_interop_threads"] = int(torch.get_num_interop_threads())
    try:
        import cv2
    except ImportError:
        summary["opencv_threads"] = None
    else:
        if active.enabled:
            cv2.setNumThreads(1)
        summary["opencv_threads"] = int(cv2.getNumThreads())

    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        summary["native_threadpools"] = []
    else:
        summary["native_threadpools"] = [
            {
                key: pool.get(key)
                for key in ("user_api", "internal_api", "prefix", "num_threads", "version")
            }
            for pool in threadpool_info()
        ]
    _ACTIVE_SUMMARY = summary
    _RUNTIME_APPLIED = True
    return dict(summary)


def validate_active_config(config: Mapping[str, Any], task: str, source_config: Path) -> Dict[str, Any]:
    """Ensure full parser results match the pre-import bootstrap policy."""
    resolved = resolve_cpu_policy(config, task, source_config)
    active = _ACTIVE_POLICY
    if active is None:
        return asdict(resolved)
    comparable_fields = (
        "enabled",
        "budget_percent",
        "logical_cpus",
        "total_thread_budget",
        "model_processes",
        "threads_per_process",
    )
    mismatches = [
        name for name in comparable_fields if getattr(active, name) != getattr(resolved, name)
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: bootstrap={getattr(active, name)!r}, loaded={getattr(resolved, name)!r}"
            for name in mismatches
        )
        raise RuntimeError(f"CPU runtime config changed after numerical imports ({details}).")
    return current_summary()


def current_summary() -> Dict[str, Any]:
    """Return a copy of the active CPU runtime metadata."""
    return dict(_ACTIVE_SUMMARY or {})


def format_summary(summary: Optional[Mapping[str, Any]] = None) -> str:
    """Format one concise, user-facing CPU runtime status line."""
    values = dict(current_summary() if summary is None else summary)
    if not values:
        return "CPU runtime: not configured by this entrypoint."
    if not bool(values.get("enabled", False)):
        return "CPU runtime: limits disabled explicitly; library defaults are unchanged."
    return (
        "CPU runtime: "
        f"logical={values.get('logical_cpus')}, "
        f"budget={values.get('total_thread_budget')} ({float(values.get('budget_percent', 0)):g}%), "
        f"model_processes={values.get('model_processes')}, "
        f"threads/process={values.get('threads_per_process')}, "
        f"torch_interop={values.get('torch_interop_threads', 1)}, "
        f"opencv={values.get('opencv_threads', 1)}."
    )


def reset_for_tests() -> None:
    """Reset process-local state for isolated unit tests."""
    global _ACTIVE_POLICY, _ACTIVE_SUMMARY, _RUNTIME_APPLIED
    _ACTIVE_POLICY = None
    _ACTIVE_SUMMARY = None
    _RUNTIME_APPLIED = False
