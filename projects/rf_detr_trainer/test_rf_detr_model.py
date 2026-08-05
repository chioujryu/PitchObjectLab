"""
Standalone RF-DETR test runner with full_image, sahi, and class_crop modes.

Usage:
    uv run python test_rf_detr_model.py --config config/rf_detr_test.yaml

    uv run python test_rf_detr_model.py \\
        --config config/rf_detr_test.yaml \\
        --checkpoint runs/rf_detr/my_run/checkpoint_best_total.pth \\
        --test-mode sahi \\
        --output-dir runs/rf_detr/test_debug \\
        --yes

Notes:
    - The config controls model, dataset, output, visual sample count, and error-case diagnostics.
    - test.visual_samples.max_images limits saved prediction images and, by default, error-case images.
    - test.visual_samples.render_class_names/render_class_ids and test.error_cases.render_class_names/render_class_ids
      independently control which classes are drawn in visual and error-case images.
    - test.error_cases defaults to football diagnostics and writes GT/prediction boxes with scores.
    - test.parallel.chunks or --chunks requests concurrent bbox-evaluation workers. By default workers are
      capped to one per resolved device; set test.parallel.allow_same_gpu_oversubscription=true only for
      an explicit same-device concurrency experiment.
    - Before writing images, metrics, or cache files, the script prints a resource estimate and asks for confirmation unless --yes is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, MutableMapping, Optional, Sequence

import rf_detr_cpu_runtime as cpu_runtime

_CPU_BOOTSTRAP_POLICY = (
    cpu_runtime.bootstrap_from_argv(
        Path(__file__).resolve().parent / "config" / "rf_detr_test.yaml",
        "test",
    )
    if __name__ in {"__main__", "__mp_main__"}
    else None
)

import colorama  # noqa: E402 - CPU bootstrap must run before numerical imports.
from colorama import Fore, Style  # noqa: E402
from tqdm import tqdm  # noqa: E402

import rf_detr_runtime as trainer  # noqa: E402
from projects.object_detection_common import test_modes as shared_modes  # noqa: E402
from projects.object_detection_dataset_evaluator.object_detection_dataset_evaluator import (  # noqa: E402
    expand_chunk_devices,
    parse_devices,
    run_evaluation,
)

colorama.init(autoreset=True)
if _CPU_BOOTSTRAP_POLICY is not None:
    cpu_runtime.apply_loaded_runtime(_CPU_BOOTSTRAP_POLICY)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "rf_detr_test.yaml"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"


def load_yaml(path: Path) -> Dict[str, Any]:
    return trainer.load_yaml(path)


def apply_cli_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    runtime = config.setdefault("runtime", {})
    cpu_runtime.apply_cpu_cli_overrides(config, args)
    model = config.setdefault("model", {})
    output = config.setdefault("output", {})
    test = config.setdefault("test", {})
    if getattr(args, "performance_profile", None) is not None:
        runtime["performance_profile"] = str(args.performance_profile).strip().lower()
    if args.yes:
        runtime["yes"] = True
        runtime["confirm_before_run"] = False
    if args.dry_run:
        runtime["dry_run"] = True
    if args.model_size:
        model["size"] = args.model_size
    if args.checkpoint:
        model["pretrain_weights"] = args.checkpoint
    if args.resolution is not None:
        model["resolution"] = args.resolution
    if args.num_classes is not None:
        model["num_classes"] = args.num_classes
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
        tensorrt["manifest_path"] = ""
    if getattr(args, "tensorrt_cache_dir", None) is not None:
        tensorrt["cache_dir"] = args.tensorrt_cache_dir
    if getattr(args, "tensorrt_force_rebuild", False):
        tensorrt["force_rebuild"] = True
    if args.output_dir:
        output["output_dir"] = args.output_dir
    if args.test_mode:
        config.setdefault("test_mode", {})["mode"] = args.test_mode
        test.setdefault("test_mode", {})["mode"] = args.test_mode
    if args.max_images is not None:
        test["max_images"] = args.max_images
    if args.batch_size is not None:
        test["batch_size"] = args.batch_size
    if args.sahi_batch_size is not None:
        test.setdefault("sahi", {})["batch_size"] = args.sahi_batch_size
    if getattr(args, "chunks", None) is not None:
        parallel = test.setdefault("parallel", {})
        parallel["chunks"] = args.chunks
        parallel.pop("requested_chunks", None)
    if getattr(args, "allow_same_gpu_oversubscription", None) is not None:
        test.setdefault("parallel", {})["allow_same_gpu_oversubscription"] = bool(
            args.allow_same_gpu_oversubscription
        )


def apply_test_performance_profile(config: MutableMapping[str, Any]) -> Optional[str]:
    """Apply safe/fast standalone-test backend defaults before explicit CLI overrides."""
    runtime = config.setdefault("runtime", {})
    if not isinstance(runtime, MutableMapping):
        raise ValueError("runtime must be a mapping.")
    legacy_performance = config.get("performance", {})
    legacy_profile = (
        legacy_performance.get("profile")
        if isinstance(legacy_performance, Mapping)
        else None
    )
    profile = trainer.normalize_performance_profile(
        runtime.get("performance_profile", legacy_profile)
    )
    if profile is None:
        return None
    runtime["performance_profile"] = profile
    optimization = config.setdefault("model", {}).setdefault("inference_optimization", {})
    if profile == "safe":
        optimization["backend"] = "pytorch"
        optimization.setdefault("pytorch", {})["precision"] = "bf16"
    else:
        optimization["backend"] = "tensorrt"
        optimization.setdefault("tensorrt", {})["precision"] = "fp16"
    return profile


def normalize_test_parallel_chunks(value: Any) -> int:
    """Validate test.parallel.chunks as a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("test.parallel.chunks must be a positive integer.")
    return int(value)


def _test_parallel_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return validated standalone-test parallel settings."""
    test_settings = config.get("test", {})
    if not isinstance(test_settings, Mapping):
        raise ValueError("test must be a mapping.")
    parallel = test_settings.get("parallel", {})
    if parallel is None:
        parallel = {}
    if not isinstance(parallel, Mapping):
        raise ValueError("test.parallel must be a mapping.")
    return parallel


def requested_standalone_parallel_chunks(config: Mapping[str, Any]) -> int:
    """Return the requested standalone-test chunk count (default one)."""
    parallel = _test_parallel_settings(config)
    return normalize_test_parallel_chunks(
        parallel.get("requested_chunks", parallel.get("chunks", 1))
    )


def allow_same_gpu_oversubscription(config: Mapping[str, Any]) -> bool:
    """Return whether multiple model replicas may intentionally share a device."""
    value = _test_parallel_settings(config).get("allow_same_gpu_oversubscription", False)
    if not isinstance(value, bool):
        raise ValueError("test.parallel.allow_same_gpu_oversubscription must be a boolean.")
    return value


def _standalone_test_devices(config: Mapping[str, Any]) -> list[str]:
    """Resolve model.device, including comma-separated CUDA device strings."""
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    raw_device = model_cfg.get("device", "auto")
    if isinstance(raw_device, str) and "," in raw_device:
        parts = [part.strip() for part in raw_device.split(",") if part.strip()]
        return parse_devices({"devices": parts})
    effective_model_cfg = dict(model_cfg)
    effective_model_cfg.setdefault("device", raw_device)
    return parse_devices(effective_model_cfg)


def standalone_parallel_chunks(config: Mapping[str, Any]) -> int:
    """Return the effective worker count after the one-worker-per-device safety cap."""
    requested = requested_standalone_parallel_chunks(config)
    if allow_same_gpu_oversubscription(config):
        return requested
    unique_devices = len(set(_standalone_test_devices(config)))
    return min(requested, max(1, unique_devices))


def apply_test_parallel_device_policy(config: MutableMapping[str, Any]) -> Optional[str]:
    """Persist the effective no-oversubscription worker count for all downstream consumers."""
    requested = requested_standalone_parallel_chunks(config)
    effective = standalone_parallel_chunks(config)
    parallel = config.setdefault("test", {}).setdefault("parallel", {})
    parallel.setdefault("allow_same_gpu_oversubscription", False)
    if effective >= requested:
        return None
    parallel["requested_chunks"] = requested
    parallel["chunks"] = effective
    device_count = len(set(_standalone_test_devices(config)))
    return (
        f"Capped test.parallel.chunks from {requested} to {effective}: only {device_count} unique "
        "device(s) were resolved and same-device oversubscription is disabled."
    )


def build_test_parallel_plan(config: Mapping[str, Any], image_count: Optional[int]) -> Dict[str, Any]:
    """Build deterministic chunk/device assignments for estimates and execution."""
    requested_chunks = requested_standalone_parallel_chunks(config)
    chunks = standalone_parallel_chunks(config)
    if image_count is not None and chunks > int(image_count):
        raise ValueError(
            f"test.parallel.chunks is {chunks}, but the evaluated image count is only {int(image_count)}."
        )
    devices = _standalone_test_devices(config)
    assignments = expand_chunk_devices(devices, chunks)
    device_counts = dict(Counter(assignments))
    unique_devices = max(1, len(device_counts))
    same_device_warning = None
    if chunks > unique_devices:
        same_device_warning = (
            "Multiple concurrent model replicas share one or more devices; this does not increase the "
            "runtime speedup cap and may increase VRAM usage or trigger CUDA OOM downshifts."
        )
    elif requested_chunks > chunks:
        same_device_warning = (
            f"Requested {requested_chunks} chunks but capped execution to {chunks} worker(s), one per "
            "resolved device. Set test.parallel.allow_same_gpu_oversubscription=true only to opt in."
        )
    return {
        "requested_chunks": requested_chunks,
        "chunks": chunks,
        "devices_requested": devices,
        "chunk_devices": assignments,
        "device_worker_counts": device_counts,
        "unique_device_count": unique_devices,
        "same_device_warning": same_device_warning,
    }


def configured_segmentation_head(config: Mapping[str, Any]) -> bool:
    """Return the effective RF-DETR segmentation-head setting without model loading."""
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, Mapping):
        return False
    extra_model_args = model_cfg.get("extra_model_args", {}) or {}
    if isinstance(extra_model_args, Mapping) and "segmentation_head" in extra_model_args:
        return bool(extra_model_args["segmentation_head"])
    return trainer.is_segmentation_model_size(model_cfg.get("size", "medium"))


def validate_parallel_test_compatibility(config: Mapping[str, Any]) -> None:
    """Reject mask-aware full-image segmentation evaluation in spawned workers."""
    if standalone_parallel_chunks(config) <= 1:
        return
    test_settings = config.get("test", {})
    mode = trainer.periodic_test_mode(test_settings if isinstance(test_settings, Mapping) else {})
    evaluation = config.get("evaluation", {})
    evaluation_type = str(
        evaluation.get("type", "auto") if isinstance(evaluation, Mapping) else "auto"
    ).strip().lower()
    if (
        configured_segmentation_head(config)
        and mode == "full_image"
        and evaluation_type != "bbox"
    ):
        raise ValueError(
            "Parallel full_image testing for a segmentation model does not support mask-aware evaluation. "
            "Set evaluation.type=bbox to run parallel bbox evaluation, or set test.parallel.chunks=1 "
            "(or --chunks 1) for segmentation metrics."
        )


def _model_config_default_resolution(model_cls: Any) -> int:
    """Read an RF-DETR variant's default resolution without constructing its model."""
    candidates = [getattr(model_cls, "_model_config_class", None)]
    try:
        import rfdetr.config as rfdetr_config

        candidates.append(getattr(rfdetr_config, f"{model_cls.__name__}Config", None))
    except ImportError:
        pass
    for config_cls in candidates:
        if config_cls is None:
            continue
        direct_value = getattr(config_cls, "resolution", None)
        if isinstance(direct_value, int) and not isinstance(direct_value, bool) and direct_value > 0:
            return direct_value
        fields = getattr(config_cls, "model_fields", {})
        field = fields.get("resolution") if isinstance(fields, Mapping) else None
        default_value = getattr(field, "default", None)
        if isinstance(default_value, int) and not isinstance(default_value, bool) and default_value > 0:
            return default_value
    raise ValueError(f"Could not determine the default resolution for {model_cls.__name__}.")


def build_rfdetr_evaluator_preview(
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Build the lightweight evaluator inputs without loading a model or checkpoint."""
    del output_dir  # Kept in the signature to mirror the concrete build path.
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError("model must be a mapping.")
    model_cls = trainer.get_model_class(str(model_cfg.get("size", "medium")))
    model_kwargs = trainer.build_model_kwargs(config)
    resolution_value = model_kwargs.get("resolution")
    resolution = (
        int(resolution_value)
        if resolution_value is not None
        else _model_config_default_resolution(model_cls)
    )
    model_preview = SimpleNamespace(
        resolution=resolution,
        segmentation_head=configured_segmentation_head(config),
    )

    train_cfg = config.get("train", {})
    dataset_cfg = config.get("dataset", {})
    if not isinstance(train_cfg, Mapping):
        raise ValueError("train must be a mapping.")
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError("dataset must be a mapping.")
    dataset_dir = str(train_cfg.get("dataset_dir") or dataset_cfg.get("dataset_dir") or "").strip()
    if not dataset_dir:
        raise ValueError("train.dataset_dir or dataset.dataset_dir is required for parallel testing.")
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        evaluation = {}
    train_preview = SimpleNamespace(
        dataset_dir=dataset_dir,
        eval_max_dets=int(train_cfg.get("eval_max_dets") or _last_max_det(evaluation)),
    )
    return model_preview, train_preview


def build_parallel_rfdetr_evaluator_config(
    config: Mapping[str, Any],
    output_dir: Path,
    split: str,
    *,
    prepared_tensorrt: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the spawn-worker evaluator config for standalone parallel testing."""
    chunks = standalone_parallel_chunks(config)
    if chunks <= 1:
        raise ValueError("Parallel evaluator config requires test.parallel.chunks greater than 1.")
    model_preview, train_preview = build_rfdetr_evaluator_preview(config, output_dir)
    evaluator_config = trainer.build_rfdetr_evaluator_config(
        merged_config=config,
        model_config=model_preview,
        train_config=train_preview,
        output_dir=output_dir,
        split=split,
        test_section="test",
    )
    evaluator_config.setdefault("runtime", {})["validate_devices_in_parent"] = False
    evaluator_config.setdefault("inference", {})["chunks"] = chunks
    factory_config = {
        "merged_config": deepcopy(dict(config)),
        "output_dir": str(output_dir),
    }
    if prepared_tensorrt is not None:
        factory_config["prepared_tensorrt"] = deepcopy(dict(prepared_tensorrt))
    try:
        pickle.dumps(factory_config)
    except Exception as exc:
        raise TypeError("Parallel RF-DETR factory_config must be multiprocessing-pickle-safe.") from exc
    evaluator_model = evaluator_config.setdefault("model", {})
    evaluator_model["devices"] = _standalone_test_devices(config)
    evaluator_model["factory"] = "rf_detr_runtime.build_rfdetr_evaluator_model"
    evaluator_model["factory_config"] = factory_config
    return evaluator_config


def _last_max_det(evaluation: Mapping[str, Any], default: int = 500) -> int:
    values = evaluation.get("max_detections")
    if isinstance(values, (list, tuple)) and values:
        return int(values[-1])
    if values is not None:
        return int(values)
    return int(evaluation.get("eval_max_dets", default) or default)


def normalize_test_device(value: Any) -> str:
    """Normalize standalone test device shortcuts into torch/RF-DETR device strings."""
    if value is None:
        return "auto"
    if isinstance(value, int) and not isinstance(value, bool):
        return "cpu" if value < 0 else f"cuda:{value}"
    text = str(value).strip()
    if not text:
        return "auto"
    lower = text.lower()
    if lower == "auto":
        return "auto"
    if lower == "-1":
        return "cpu"
    if lower.isdecimal():
        return f"cuda:{int(lower)}"
    if lower in {"cpu", "cuda", "mps"} or lower.startswith("cuda:"):
        return lower
    return text


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average_inference_seconds(result: Mapping[str, Any], output_dir: Path) -> Optional[float]:
    """Return average inference seconds per image from result payload or saved evaluator files."""
    summary = result.get("summary")
    if isinstance(summary, Mapping):
        value = _float_or_none(summary.get("avg_inference_seconds_per_image"))
        if value is not None:
            return value

    stats = result.get("stats")
    if isinstance(stats, list):
        elapsed = [
            value
            for row in stats
            if isinstance(row, Mapping)
            for value in [_float_or_none(row.get("elapsed_seconds"))]
            if value is not None
        ]
        if elapsed:
            return sum(elapsed) / len(elapsed)

    stage_timing = result.get("stage_timing")
    if isinstance(stage_timing, Mapping):
        total = _float_or_none(stage_timing.get("total_seconds"))
        count = _float_or_none(stage_timing.get("images_or_frames"))
        if total is not None and count is not None and count > 0:
            return total / count

    metrics_path = output_dir / "metrics_summary.json"
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = data.get("metrics") if isinstance(data, Mapping) else None
        if isinstance(metrics, Mapping):
            value = _float_or_none(metrics.get("avg_inference_seconds_per_image"))
            if value is not None:
                return value

    stats_path = output_dir / "inference_stats.csv"
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8", newline="") as file:
            elapsed = [
                value
                for row in csv.DictReader(file)
                for value in [_float_or_none(row.get("elapsed_seconds"))]
                if value is not None
            ]
        if elapsed:
            return sum(elapsed) / len(elapsed)

    return None


def print_inference_timing_summary(result: Mapping[str, Any], output_dir: Path) -> None:
    """Print standalone test timing outputs."""
    average = average_inference_seconds(result, output_dir)
    if average is not None:
        print(f"Average inference seconds per image: {average:.6f}")
    timing = result.get("stage_timing")
    if not isinstance(timing, Mapping):
        timing = result.get("summary")
    if isinstance(timing, Mapping):
        labels = (
            ("model_forward_ratio", "Model forward share"),
            ("sahi_model_forward_ratio", "SAHI model-forward share"),
            ("recheck_model_forward_ratio", "Recheck model-forward share"),
        )
        for key, label in labels:
            value = _float_or_none(timing.get(key))
            if value is not None:
                print(f"{label}: {value * 100.0:.2f}%")
    stats_path = output_dir / "inference_stats.csv"
    if stats_path.exists():
        print(f"Per-image inference stats: {stats_path}")


def observed_test_model_work(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize actual full-image, slice, and recheck model inputs from evaluator stats."""
    source_inputs = 0
    slice_inputs = 0
    secondary_inputs = 0
    recheck_inputs = 0
    observed_batches = 0
    for row in rows:
        slices = max(0, int(row.get("slice_count", 0) or 0))
        if slices:
            slice_inputs += slices
            if float(row.get("base_model_forward_seconds", 0.0) or 0.0) > 0.0:
                source_inputs += 1
            recheck = row.get("sahi_recheck", {})
            if isinstance(recheck, Mapping):
                recheck_inputs += max(0, int(recheck.get("rechecked", 0) or 0))
                observed_batches += max(0, int(recheck.get("batch_count", 0) or 0))
            batch_size = max(1, int(row.get("slice_batch_size", 1) or 1))
            observed_batches += int(math.ceil(slices / batch_size))
            continue
        source_inputs += 1
        mode = str(row.get("test_mode", row.get("inference_engine", ""))).lower()
        if "class_crop" in mode:
            secondary_inputs += 1
    total_inputs = source_inputs + slice_inputs + secondary_inputs + recheck_inputs
    return {
        "source_inputs": source_inputs,
        "slice_inputs": slice_inputs,
        "secondary_inputs": secondary_inputs,
        "recheck_inputs": recheck_inputs,
        "total_model_inputs": total_inputs,
        "observed_or_lower_bound_model_batches": observed_batches,
    }


def normalized_stage_timing(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return additive timing totals in the shared run_timing.json schema."""

    parallel_summary = result.get("parallel_summary")
    stats = result.get("stats")
    stats_rows = (
        [row for row in stats if isinstance(row, Mapping)]
        if isinstance(stats, list)
        else []
    )
    model_work = observed_test_model_work(stats_rows) if stats_rows else None

    def parallel_speedup_cap() -> int:
        if not isinstance(parallel_summary, Mapping):
            return 1
        assignments = parallel_summary.get("assignments", [])
        if not isinstance(assignments, list):
            return 1
        devices = {
            str(assignment.get("device"))
            for assignment in assignments
            if isinstance(assignment, Mapping) and assignment.get("device") is not None
        }
        return max(1, len(devices))

    def add_model_work(timing: Dict[str, Any]) -> Dict[str, Any]:
        if model_work is None:
            return timing
        speedup_cap = parallel_speedup_cap()
        timing["model_work"] = {
            **model_work,
            "parallel_speedup_cap": speedup_cap,
        }
        timing.setdefault(
            "runtime_units",
            float(model_work["total_model_inputs"]) / float(speedup_cap),
        )
        return timing

    def add_parallel_timing(timing: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parallel_summary, Mapping):
            timing.setdefault("critical_path_wall_seconds", timing.get("total_seconds", 0.0))
            timing.setdefault("aggregate_worker_seconds", timing.get("total_seconds", 0.0))
            return add_model_work(timing)
        wall_seconds = _float_or_none(parallel_summary.get("wall_seconds"))
        assignments = parallel_summary.get("assignments", [])
        aggregate_worker_seconds = 0.0
        if isinstance(assignments, list):
            for assignment in assignments:
                if not isinstance(assignment, Mapping):
                    continue
                aggregate_worker_seconds += max(
                    0.0,
                    float(_float_or_none(assignment.get("model_load_seconds")) or 0.0),
                )
                aggregate_worker_seconds += max(
                    0.0,
                    float(_float_or_none(assignment.get("inference_seconds")) or 0.0),
                )
        timing["critical_path_wall_seconds"] = max(0.0, float(wall_seconds or 0.0))
        timing["aggregate_worker_seconds"] = aggregate_worker_seconds
        timing["parallel_worker_count"] = int(parallel_summary.get("chunks", len(assignments)) or 0)
        return add_model_work(timing)

    existing = result.get("stage_timing")
    if isinstance(existing, Mapping):
        timing = dict(existing)
        forward_seconds = float(timing.get("model_forward_seconds", 0.0) or 0.0)
        total_seconds = float(timing.get("total_seconds", 0.0) or 0.0)
        timing.setdefault("base_model_forward_seconds", forward_seconds)
        timing.setdefault("sahi_model_forward_seconds", 0.0)
        timing.setdefault("recheck_model_forward_seconds", 0.0)
        timing.setdefault("model_forward_ratio", forward_seconds / total_seconds if total_seconds > 0 else 0.0)
        timing.setdefault("sahi_model_forward_ratio", 0.0)
        timing.setdefault("recheck_model_forward_ratio", 0.0)
        return add_parallel_timing(timing)
    if not isinstance(stats, list):
        return add_parallel_timing({}) if isinstance(parallel_summary, Mapping) else {}
    rows = stats_rows
    if not rows:
        return add_parallel_timing({}) if isinstance(parallel_summary, Mapping) else {}

    def total(key: str, *, fallback: str | None = None) -> float:
        return sum(
            float(row.get(key, row.get(fallback, 0.0) if fallback is not None else 0.0) or 0.0)
            for row in rows
        )

    total_seconds = total("elapsed_seconds")
    forward_seconds = total("model_forward_seconds", fallback="elapsed_seconds")
    timing = {
        "images_or_frames": len(rows),
        "total_seconds": total_seconds,
        "preprocess_seconds": total("preprocess_seconds"),
        "model_forward_seconds": forward_seconds,
        "base_model_forward_seconds": total("base_model_forward_seconds"),
        "sahi_model_forward_seconds": total("sahi_model_forward_seconds"),
        "recheck_model_forward_seconds": total("recheck_model_forward_seconds"),
        "postprocess_seconds": total("postprocess_seconds"),
    }
    timing.update(
        {
            "model_forward_ratio": forward_seconds / total_seconds if total_seconds > 0.0 else 0.0,
            "sahi_model_forward_ratio": (
                timing["sahi_model_forward_seconds"] / total_seconds if total_seconds > 0.0 else 0.0
            ),
            "recheck_model_forward_ratio": (
                timing["recheck_model_forward_seconds"] / total_seconds if total_seconds > 0.0 else 0.0
            ),
        }
    )
    return add_parallel_timing(timing)


def _non_negative_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "all", "none", "null"}:
        return None
    return max(0, int(value))


def error_case_image_cap(test_settings: Mapping[str, Any]) -> int:
    """Return the shared max image count for all configured error-case outputs."""
    visual = dict(test_settings.get("visual_samples", {}) or {})
    error_cases = dict(test_settings.get("error_cases", {}) or {})
    for key in ("max_images",):
        value = _non_negative_int_or_none(error_cases.get(key))
        if value is not None:
            return value
    value = _non_negative_int_or_none(visual.get("max_images"))
    if value is not None:
        return value
    legacy = [
        _non_negative_int_or_none(error_cases.get("max_missed_images")),
        _non_negative_int_or_none(error_cases.get("max_false_positive_images")),
    ]
    legacy = [item for item in legacy if item is not None]
    return max(legacy) if legacy else 25


TEST_ESTIMATE_FALLBACK_SIZE = (3840, 2160)


def _split_aliases(split: str) -> list[str]:
    normalized = str(split).strip().lower().replace("_", "-")
    aliases = [normalized]
    if normalized == "val":
        aliases.append("valid")
    elif normalized == "valid":
        aliases.append("val")
    elif normalized == "test-original":
        aliases.extend(["test_original", "test"])
    return list(dict.fromkeys(aliases))


def _valid_dimensions(record: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    try:
        width = int(record.get("width", 0) or 0)
        height = int(record.get("height", 0) or 0)
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _bounded_coco_image_metadata(
    annotation: Path,
    limit: Optional[int],
) -> list[Mapping[str, Any]]:
    """Incrementally decode only the COCO ``images`` rows needed by max-images."""
    if limit is not None and limit <= 0:
        return []
    decoder = json.JSONDecoder()
    rows: list[Mapping[str, Any]] = []
    buffer = ""
    position = 0
    array_started = False
    eof = False
    with annotation.open("r", encoding="utf-8") as file:
        while limit is None or len(rows) < limit:
            if not eof:
                chunk = file.read(64 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            if not array_started:
                key_at = buffer.find('"images"')
                if key_at < 0:
                    if eof:
                        break
                    buffer = buffer[-32:]
                    continue
                array_at = buffer.find("[", key_at + len('"images"'))
                if array_at < 0:
                    if eof:
                        break
                    continue
                array_started = True
                position = array_at + 1

            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    return rows
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                position = end
                if isinstance(value, Mapping):
                    rows.append(value)
                    if limit is not None and len(rows) >= limit:
                        return rows
            if eof:
                break
            if array_started and position > 0:
                buffer = buffer[position:]
                position = 0
    return rows


def test_estimate_image_dimensions(
    dataset_plan: Mapping[str, Any],
    *,
    split: str,
    image_count: Optional[int],
) -> Dict[str, Any]:
    """Read bounded split metadata for workload estimation, with a 4K upper fallback."""
    limit = None if image_count is None else max(0, int(image_count))
    dimensions: list[tuple[int, int]] = []
    source = "unavailable"
    records_by_split = dataset_plan.get("records_by_split", {})
    if isinstance(records_by_split, Mapping):
        records = None
        for alias in _split_aliases(split):
            candidate = records_by_split.get(alias)
            if isinstance(candidate, list):
                records = candidate
                break
        if records is not None:
            bounded = records if limit is None else records[:limit]
            dimensions = [size for record in bounded if isinstance(record, Mapping) if (size := _valid_dimensions(record))]
            source = "dataset_plan_records"

    if not dimensions:
        dataset_dir_value = dataset_plan.get("dataset_dir") or dataset_plan.get("cache_dir")
        if dataset_dir_value:
            dataset_dir = Path(str(dataset_dir_value))
            for alias in _split_aliases(split):
                split_dir = trainer.dataset_split_dir(dataset_dir, alias)
                annotation = split_dir / "_annotations.coco.json" if split_dir is not None else None
                if annotation is None or not annotation.exists():
                    continue
                try:
                    images = _bounded_coco_image_metadata(annotation, limit)
                except (OSError, UnicodeDecodeError):
                    continue
                dimensions = [size for image in images if (size := _valid_dimensions(image))]
                source = "bounded_coco_image_metadata"
                break

    resolved_count = int(image_count) if image_count is not None else len(dimensions)
    resolved_count = max(0, resolved_count)
    metadata_count = min(len(dimensions), resolved_count)
    dimensions = dimensions[:resolved_count]
    fallback_count = max(0, resolved_count - len(dimensions))
    if fallback_count:
        dimensions.extend([TEST_ESTIMATE_FALLBACK_SIZE] * fallback_count)
        source = f"{source}+conservative_4k_fallback" if source != "unavailable" else "conservative_4k_fallback"
    size_counts = Counter(f"{width}x{height}" for width, height in dimensions)
    return {
        "dimensions": dimensions,
        "image_count": resolved_count,
        "metadata_image_count": metadata_count,
        "fallback_image_count": fallback_count,
        "source": source,
        "size_counts": dict(sorted(size_counts.items())),
    }


def _positive_batch_size(value: Any, *, automatic_default: int) -> tuple[int, Any]:
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return automatic_default, value if value is not None else automatic_default
    if isinstance(value, bool):
        raise ValueError("Model batch size must be a positive integer or auto.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("Model batch size must be a positive integer or auto.")
    return parsed, value


def estimate_standalone_test_model_work(
    config: Mapping[str, Any],
    dataset_plan: Mapping[str, Any],
    *,
    split: str,
    image_count: Optional[int],
) -> Dict[str, Any]:
    """Estimate model inputs/batches from test mode, dimensions, slices, and recheck cap."""
    test_settings = config.get("test", {})
    if not isinstance(test_settings, Mapping):
        test_settings = {}
    mode = shared_modes.canonical_test_mode(
        {
            "test_mode": config.get("test_mode", test_settings.get("test_mode", {})),
            "inference": config.get("inference", {}),
        }
    )
    dimension_info = test_estimate_image_dimensions(
        dataset_plan,
        split=split,
        image_count=image_count,
    )
    count = int(dimension_info["image_count"])
    dimensions = list(dimension_info.pop("dimensions"))
    outer_batch, outer_setting = _positive_batch_size(
        test_settings.get("batch_size", 4),
        automatic_default=4,
    )
    source_inputs = 0
    slice_inputs = 0
    secondary_inputs = 0
    recheck_inputs = 0
    model_batches = 0
    model_batch = outer_batch
    model_batch_setting: Any = outer_setting
    recheck_basis = "disabled"

    if mode == shared_modes.SAHI_MODE:
        sahi = config.get("sahi", test_settings.get("sahi", {}))
        if not isinstance(sahi, Mapping):
            sahi = {}
        model_batch, model_batch_setting = _positive_batch_size(
            sahi.get("batch_size", 4),
            automatic_default=16,
        )
        slice_counts = [
            len(
                shared_modes.generate_slice_windows_for_size(
                    width=width,
                    height=height,
                    slice_width=int(sahi.get("slice_width", width)),
                    slice_height=int(sahi.get("slice_height", height)),
                    overlap_width_ratio=float(sahi.get("overlap_width_ratio", 0.2)),
                    overlap_height_ratio=float(sahi.get("overlap_height_ratio", 0.2)),
                )
            )
            for width, height in dimensions
        ]
        slice_inputs = sum(slice_counts)
        standard_prediction = bool(sahi.get("standard_prediction", True))
        source_inputs = count if standard_prediction else 0
        recheck = sahi.get("recheck", {})
        if not isinstance(recheck, Mapping):
            recheck = {}
        if bool(recheck.get("enabled", False)):
            max_rechecks = max(0, int(recheck.get("max_rechecks_per_image", 50) or 0))
            recheck_inputs = count * max_rechecks
            recheck_basis = "conservative_per_image_cap"
        for group_start in range(0, count, outer_batch):
            group_slices = slice_counts[group_start : group_start + outer_batch]
            group_count = len(group_slices)
            model_batches += int(math.ceil(sum(group_slices) / model_batch))
            if standard_prediction:
                model_batches += int(math.ceil(group_count / model_batch))
            if recheck_inputs:
                max_rechecks = recheck_inputs // max(1, count)
                model_batches += int(math.ceil(group_count * max_rechecks / model_batch))
    elif mode == shared_modes.CLASS_CROP_MODE:
        # The implementation always runs a source pass and then a crop or fallback pass.
        source_inputs = count
        secondary_inputs = count
        model_batches = 2 * int(math.ceil(count / model_batch)) if count else 0
    else:
        source_inputs = count
        model_batches = int(math.ceil(count / model_batch)) if count else 0

    total_inputs = source_inputs + slice_inputs + secondary_inputs + recheck_inputs
    return {
        "mode": mode,
        "images": count,
        "source_inputs": source_inputs,
        "slice_inputs": slice_inputs,
        "secondary_inputs": secondary_inputs,
        "recheck_input_cap": recheck_inputs,
        "recheck_estimate_basis": recheck_basis,
        "total_model_inputs": total_inputs,
        "outer_batch_size": outer_batch,
        "model_batch_size_setting": model_batch_setting,
        "model_batch_size_assumed": model_batch,
        "estimated_model_batches": model_batches,
        "image_dimensions": dimension_info,
    }


def estimate_standalone_test_outputs(
    config: Mapping[str, Any],
    output_dir: Path,
    dataset_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Estimate standalone test files before materializing dataset or output folders."""
    test_settings = config.get("test", {})
    evaluation = config.get("evaluation", {})
    visual = dict(test_settings.get("visual_samples", {}) or {})
    error_cases = dict(test_settings.get("error_cases", {}) or {})
    split_counts = dict(dataset_plan.get("split_counts", {}) or {})
    split = str(test_settings.get("split", "test"))
    image_count = split_counts.get(split)
    if image_count is None and split == "test-original":
        image_count = split_counts.get("test-original") or split_counts.get("test_original") or split_counts.get("test")
    test_limit = trainer.parse_limit_value(test_settings.get("max_images"), "test.max_images")
    if test_limit is not None:
        image_count = min(int(image_count), test_limit) if image_count is not None else test_limit
    model_work = estimate_standalone_test_model_work(
        config,
        dataset_plan,
        split=split,
        image_count=int(image_count) if image_count is not None else None,
    )
    if image_count is None and model_work["images"]:
        image_count = int(model_work["images"])
    parallel_plan = build_test_parallel_plan(config, int(image_count) if image_count is not None else None)
    parallel_summary_files = 1 if parallel_plan["chunks"] > 1 else 0

    config_files = 6
    metric_files = 8 if bool(evaluation.get("classwise", True)) else 6
    model_input_files = int(config.get("output", {}).get("max_model_input_batches", 3) or 3)
    plot_files = 4 if bool(test_settings.get("plots", False)) else 0
    dataset_case_files = 0
    if bool(test_settings.get("save_dataset_cases", False)):
        dataset_case_files = 7
    visual_files = 0
    if bool(visual.get("enabled", False)):
        max_images = _non_negative_int_or_none(visual.get("max_images"))
        visual_files = max_images if max_images is not None else int(image_count or 0)
        visual_files += 2
    error_case_files = 0
    if bool(error_cases.get("enabled", False)):
        error_case_files = error_case_image_cap(test_settings) + 3

    dataset_cache_files = 0
    dataset_cache_bytes = 0
    if dataset_plan.get("action") == "prepare_cache":
        dataset_cache_files = int(dataset_plan.get("cache_file_count", 0) or 0)
        dataset_cache_bytes = int(dataset_plan.get("copy_bytes", 0) or 0)
    tensorrt_artifacts = trainer.estimate_tensorrt_cache_artifacts(config)

    file_count = (
        config_files
        + metric_files
        + model_input_files
        + plot_files
        + dataset_case_files
        + visual_files
        + error_case_files
        + dataset_cache_files
        + parallel_summary_files
        + int(tensorrt_artifacts["file_count"])
    )
    image_outputs = model_input_files + max(0, visual_files - 2 if visual_files else 0) + max(0, error_case_files - 3 if error_case_files else 0) + dataset_case_files
    approx_bytes = (
        dataset_cache_bytes
        + metric_files * 200_000
        + image_outputs * 500_000
        + plot_files * 350_000
        + config_files * 50_000
        + parallel_summary_files * 50_000
        + int(tensorrt_artifacts["bytes"])
    )
    estimate = {
        "output_dir": str(output_dir),
        "dataset_source_format": dataset_plan.get("source_format"),
        "dataset_plan_action": dataset_plan.get("action"),
        "dataset_cache_dir": str(dataset_plan.get("cache_dir")) if dataset_plan.get("cache_dir") else None,
        "dataset_cache_files": dataset_cache_files,
        "dataset_cache_disk_usage": trainer.format_bytes(dataset_cache_bytes),
        "split": split,
        "split_image_count": image_count,
        "model_input_batch_images": model_input_files,
        "visual_sample_files": visual_files,
        "error_case_files": error_case_files,
        "dataset_case_files": dataset_case_files,
        "plot_files": plot_files,
        "parallel_chunks": parallel_plan["chunks"],
        "parallel_chunks_requested": parallel_plan["requested_chunks"],
        "parallel_model_replicas": parallel_plan["chunks"],
        "parallel_devices_requested": parallel_plan["devices_requested"],
        "parallel_chunk_devices": parallel_plan["chunk_devices"],
        "parallel_device_worker_counts": parallel_plan["device_worker_counts"],
        "parallel_summary_files": parallel_summary_files,
        "parallel_same_device_warning": parallel_plan["same_device_warning"],
        "model_work": model_work,
        "tensorrt_cache": tensorrt_artifacts,
        "estimated_total_files": file_count,
        "estimated_disk_usage": trainer.format_bytes(approx_bytes),
        "note": "Test outputs and first-run TensorRT artifacts in the configured cache are estimated conservatively.",
    }
    trainer.add_runtime_estimate(
        estimate=estimate,
        config=config,
        output_dir=output_dir,
        task="test",
        runtime_units=float(model_work["total_model_inputs"]) / float(parallel_plan["unique_device_count"]),
        default_rate_key="default_test_seconds_per_image",
        basis={
            "test_images": image_count,
            "split": split,
            "parallel_chunks": parallel_plan["chunks"],
            "parallel_speedup_cap": parallel_plan["unique_device_count"],
            "model_work": model_work,
        },
        extra_seconds=float(tensorrt_artifacts.get("estimated_build_seconds", 0) or 0),
    )
    return estimate


def confirm_test_or_exit(estimate: Mapping[str, Any], verbose: bool, assume_yes: bool) -> None:
    """Ask for confirmation before standalone test outputs are created."""
    if verbose:
        print(Fore.BLUE + Style.BRIGHT + "Output and resource estimate before standalone test:")
    print(json.dumps(dict(estimate), indent=2, ensure_ascii=False))
    if assume_yes:
        if verbose:
            print(Fore.BLUE + Style.BRIGHT + "Confirmation skipped because --yes or confirm_before_run=false is enabled.")
        return
    answer = input(Fore.BLUE + Style.BRIGHT + "Continue and start standalone test? [y/N]: " + Style.RESET_ALL).strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted by developer before test output was produced.")


def build_internal_test_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate a test-only YAML into the internal RF-DETR/evaluator shape."""
    internal = deepcopy(dict(config))
    test = dict(internal.get("test", {}) or {})
    periodic_source = dict(internal.get("periodic_test", {}) or {})
    dataset = internal.setdefault("dataset", {})
    model = internal.setdefault("model", {})
    evaluation = internal.setdefault("evaluation", {})
    raw_device = model.get("device")
    device = normalize_test_device(raw_device)
    if raw_device is not None:
        model["device"] = device
    periodic_mode = (
        periodic_source.get("test_mode", {}).get("mode")
        if isinstance(periodic_source.get("test_mode"), Mapping)
        else None
    )
    test_mode = test.get("test_mode", {})
    test_mode_value = test_mode.get("mode") if isinstance(test_mode, Mapping) else None
    mode = (
        internal.get("test_mode", {}).get("mode")
        if isinstance(internal.get("test_mode"), Mapping)
        else None
    ) or test_mode_value or test.get("mode") or periodic_mode or "full_image"
    split = str(test.get("split") or periodic_source.get("split") or dataset.get("split", "test") or "test")
    max_images = trainer.parse_limit_value(test.get("max_images", periodic_source.get("max_images")), "test.max_images")
    crop = test.get("crop") or internal.get("crop") or periodic_source.get("crop") or {}
    sahi = test.get("sahi") or internal.get("sahi") or periodic_source.get("sahi") or {}
    error_cases = test.get("error_cases") or periodic_source.get("error_cases") or {}
    visual_samples = dict(test.get("visual_samples") or periodic_source.get("visual_samples") or {})
    artifacts = dict(test.get("artifacts") or {})
    full_model_input_manifest = artifacts.get("full_model_input_manifest", False)
    if not isinstance(full_model_input_manifest, bool):
        raise ValueError("test.artifacts.full_model_input_manifest must be a boolean.")
    artifacts["full_model_input_manifest"] = full_model_input_manifest
    parallel_source = test.get("parallel", {}) or {}
    if not isinstance(parallel_source, Mapping):
        raise ValueError("test.parallel must be a mapping.")
    parallel = dict(parallel_source)
    parallel["chunks"] = normalize_test_parallel_chunks(parallel.get("chunks", 1))
    allow_oversubscription = parallel.get("allow_same_gpu_oversubscription", False)
    if not isinstance(allow_oversubscription, bool):
        raise ValueError("test.parallel.allow_same_gpu_oversubscription must be a boolean.")
    parallel["allow_same_gpu_oversubscription"] = allow_oversubscription
    visual_sample_cap = _non_negative_int_or_none(visual_samples.get("max_images"))
    if visual_samples.get("max_images") is not None:
        visual_samples["max_images"] = visual_sample_cap
    if visual_sample_cap is not None and isinstance(error_cases, Mapping):
        error_cases = dict(error_cases)
        error_cases.setdefault("max_images", visual_sample_cap)

    internal["test_mode"] = {"mode": mode}
    internal["crop"] = dict(crop)
    internal["sahi"] = dict(sahi)
    evaluation.setdefault("classwise", bool(test.get("classwise", periodic_source.get("classwise", True))))
    evaluation.setdefault("max_detections", [1, 10, int(test.get("max_dets", periodic_source.get("max_dets", 500)) or 500)])
    evaluation.setdefault("match_iou_threshold", float(test.get("match_iou_threshold", periodic_source.get("match_iou_threshold", 0.5))))
    test_settings = {
        **test,
        "split": split,
        "max_images": max_images,
        "parallel": parallel,
        "test_mode": {"mode": mode},
        "crop": internal["crop"],
        "sahi": internal["sahi"],
        "classwise": bool(evaluation.get("classwise", True)),
        "progress_bar": bool(test.get("progress_bar", periodic_source.get("progress_bar", True))),
        "plots": bool(test.get("plots", periodic_source.get("plots", False))),
        "save_dataset_cases": bool(test.get("save_dataset_cases", periodic_source.get("save_dataset_cases", False))),
        "visual_samples": dict(visual_samples),
        "model_input_batch_size": int(test.get("model_input_batch_size", periodic_source.get("model_input_batch_size", 9)) or 9),
        "batch_size": int(test.get("batch_size", periodic_source.get("batch_size", 4)) or 4),
        "error_cases": dict(error_cases),
        "artifacts": artifacts,
        "conf": model.get("confidence_threshold", 0.25),
        "match_iou_threshold": evaluation.get("match_iou_threshold", 0.5),
        "max_dets": _last_max_det(evaluation),
    }
    internal["test"] = test_settings
    apply_test_parallel_device_policy(internal)

    internal["periodic_test"] = {
        "enabled": False,
        **test_settings,
        "run_final_test": True,
    }

    dataset_dir = str(dataset.get("dataset_dir") or "").strip()
    internal["train"] = {
        "dataset_dir": dataset_dir,
        "dataset_file": "roboflow",
        "batch_size": int(test_settings.get("batch_size", 4) or 4),
        "grad_accum_steps": 1,
        "num_workers": int(2 if test.get("num_workers", 2) is None else test.get("num_workers", 2)),
        "device": device,
        "eval_max_dets": _last_max_det(evaluation),
        "run_test": False,
        "extra_train_args": {},
    }
    return internal


def _main_impl(timing_context: Optional[MutableMapping[str, Any]] = None) -> int:
    parser = argparse.ArgumentParser(description="RF-DETR standalone test runner.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to rf_detr_test.yaml.")
    parser.add_argument("--model-size", help="RF-DETR size override, e.g. medium, large, seg-small, seg-2xlarge.")
    parser.add_argument("--checkpoint", help="RF-DETR .pth checkpoint/pretrain_weights path.")
    parser.add_argument("--resolution", type=int, help="Input resolution override.")
    parser.add_argument("--num-classes", type=int, help="Optional class count override.")
    parser.add_argument(
        "--tracknet-focus",
        choices=("single", "all"),
        help="Override model.motion.focus.mode for temporal TrackNet evaluation.",
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
    parser.add_argument("--output-dir", help="Exact output directory override.")
    parser.add_argument("--test-mode", choices=["full_image", "sahi", "class_crop"], help="Test mode override.")
    parser.add_argument("--max-images", type=trainer.parse_scalar, help="Maximum test images to evaluate. Use all/null for all.")
    parser.add_argument("--batch-size", type=int, help="RF-DETR full-image/class-crop evaluator batch size.")
    parser.add_argument(
        "--sahi-batch-size",
        type=trainer.parse_scalar,
        help="RF-DETR SAHI slice/recheck batch size or auto.",
    )
    parser.add_argument(
        "--performance-profile",
        choices=["safe", "fast"],
        help="Apply safe PyTorch BF16 or fast TensorRT FP16 defaults before explicit backend overrides.",
    )
    parser.add_argument("--chunks", type=int, help="Concurrent standalone test chunks/model replicas.")
    parser.add_argument(
        "--allow-same-gpu-oversubscription",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow multiple concurrent model replicas on one GPU (disabled by default).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned output without inference.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    cpu_runtime.add_cpu_cli_arguments(parser)
    args = parser.parse_args()

    with tqdm(total=6, desc="Preparing RF-DETR test", unit="step") as bar:
        source_config = Path(args.config).expanduser()
        if not source_config.is_absolute():
            source_config = (Path.cwd() / source_config).resolve()
        config = load_yaml(source_config)
        if args.performance_profile is not None:
            config.setdefault("runtime", {})["performance_profile"] = args.performance_profile
        selected_performance_profile = apply_test_performance_profile(config)
        apply_cli_overrides(config, args)
        cpu_summary = cpu_runtime.validate_active_config(config, "test", source_config)
        print(cpu_runtime.format_summary(cpu_summary))
        parallel_policy_message = apply_test_parallel_device_policy(config)
        if parallel_policy_message:
            print(Fore.YELLOW + parallel_policy_message)
        if selected_performance_profile is not None:
            print(Fore.BLUE + f"Test performance profile: {selected_performance_profile}")
        internal_config = build_internal_test_config(config)
        trainer._require_custom_architecture_checkpoint(internal_config, "Standalone test")
        validate_parallel_test_compatibility(internal_config)
        acceleration_settings = trainer.validate_inference_acceleration_config(internal_config)
        parallel_chunks = standalone_parallel_chunks(internal_config)
        if timing_context is not None:
            timing_context["verbose"] = bool(internal_config.get("runtime", {}).get("verbose", True))
            timing_context["cpu_runtime"] = cpu_summary
            timing_context["execution_profile"] = trainer.inference_execution_profile(internal_config)
        bar.update(1)

        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        output_dir = trainer.build_output_dir(internal_config, timestamp)
        if timing_context is not None:
            timing_context["output_dir"] = str(output_dir)
        test_split = str(internal_config.get("test", {}).get("split", "test"))
        dataset_plan = trainer.build_dataset_plan(
            internal_config,
            output_dir,
            source_config,
            required_splits=[test_split],
            required_split_limits={
                test_split: internal_config.get("test", {}).get("max_images")
            },
        )
        bar.update(1)

        if output_dir.exists() and not bool(internal_config.get("output", {}).get("exist_ok", False)):
            raise FileExistsError(f"Output directory already exists and output.exist_ok=false: {output_dir}")
        estimate = estimate_standalone_test_outputs(internal_config, output_dir, dataset_plan)
        if timing_context is not None:
            timing_context["estimate"] = estimate
            timing_context["dry_run"] = bool(internal_config.get("runtime", {}).get("dry_run", False))
        confirm = bool(internal_config.get("runtime", {}).get("confirm_before_run", True))
        assume_yes = bool(internal_config.get("runtime", {}).get("yes", False) or not confirm)
        confirm_test_or_exit(estimate, bool(internal_config.get("runtime", {}).get("verbose", True)), assume_yes)
        preflight_devices = (
            list(dict.fromkeys(_standalone_test_devices(internal_config)))
            if parallel_chunks > 1
            else [internal_config.get("model", {}).get("device")]
        )
        for preflight_device in preflight_devices:
            trainer.preflight_rfdetr_inference_acceleration(
                internal_config,
                device=preflight_device,
            )
        bar.update(1)

        if bool(internal_config.get("runtime", {}).get("dry_run", False)):
            bar.update(bar.total - bar.n)
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        if timing_context is not None:
            timing_context["outputs_created"] = True
        trainer.start_run_log_capture(output_dir, "test", timing_context)

        dataset_metadata = trainer.materialize_dataset_plan(
            dataset_plan,
            internal_config,
            output_dir,
            bool(internal_config.get("runtime", {}).get("verbose", True)),
        )
        if dataset_plan.get("action") == "none" and dataset_plan.get("dataset_dir"):
            resolved_dataset_dir = str(dataset_plan["dataset_dir"])
            internal_config.setdefault("dataset", {})["dataset_dir"] = resolved_dataset_dir
            internal_config.setdefault("train", {})["dataset_dir"] = resolved_dataset_dir
        if not internal_config.get("train", {}).get("dataset_dir"):
            internal_config["train"]["dataset_dir"] = internal_config.get("dataset", {}).get("dataset_dir", "")
        bar.update(1)

        trainer.dump_config_snapshot(
            output_dir=output_dir,
            merged_config=internal_config,
            metadata={"event": "standalone_test_start", "dataset": dataset_metadata},
            source_config=source_config,
        )
        bar.update(1)

        rf_model = None
        train_config = None
        parallel_evaluator_config = None
        if parallel_chunks > 1:
            prepared_tensorrt = None
            if acceleration_settings.backend == "tensorrt":
                preparation = trainer.prepare_parallel_tensorrt_artifacts(
                    internal_config,
                    output_dir,
                    _standalone_test_devices(internal_config),
                )
                prepared_tensorrt = preparation.get("device_artifacts")
                if not isinstance(prepared_tensorrt, Mapping) or not prepared_tensorrt:
                    raise RuntimeError("TensorRT preparation did not return worker artifacts.")
                if timing_context is not None:
                    timing_context["acceleration"] = {
                        key: value
                        for key, value in preparation.items()
                        if key != "device_artifacts"
                    }
            parallel_evaluator_config = build_parallel_rfdetr_evaluator_config(
                internal_config,
                output_dir,
                str(internal_config.get("test", {}).get("split", "test")),
                prepared_tensorrt=prepared_tensorrt,
            )
        else:
            rf_model, train_config = trainer.build_rfdetr_evaluator_runtime(
                internal_config,
                output_dir,
            )
            if timing_context is not None:
                acceleration_handle = trainer.get_inference_acceleration_handle(rf_model)
                timing_context["acceleration"] = dict(acceleration_handle.metadata)
        bar.update(1)

    test_settings = internal_config.get("test", {})
    temporal_enabled = trainer.temporal_motion_enabled(internal_config)
    mode = trainer.periodic_test_mode(test_settings)
    evaluation_type = str(internal_config.get("evaluation", {}).get("type", "auto")).strip().lower()
    segmentation_model = bool(
        rf_model is not None and getattr(rf_model.model_config, "segmentation_head", False)
    )
    use_segmentation_eval = (
        segmentation_model
        and mode == "full_image"
        and evaluation_type in {"auto", "segm", "segment", "segmentation", "mask", "masks"}
    )
    if temporal_enabled:
        if parallel_chunks > 1:
            raise ValueError("Real temporal TrackNet standalone test currently requires test.parallel.chunks=1")
        if rf_model is None or train_config is None:
            raise RuntimeError("Temporal RF-DETR evaluator runtime was not initialized.")
        from rf_detr_temporal_runtime import run_temporal_split

        result = run_temporal_split(
            rf_model=rf_model,
            config=internal_config,
            output_dir=output_dir,
            split=str(test_settings.get("split", "test")),
            save_heatmaps=True,
        )
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    elif parallel_chunks > 1:
        if parallel_evaluator_config is None:
            raise RuntimeError("Parallel evaluator config was not initialized.")
        result = run_evaluation(
            deepcopy(parallel_evaluator_config),
            source_config,
            prebuilt_model=None,
            print_summary=True,
        )
    elif use_segmentation_eval:
        if rf_model is None or train_config is None:
            raise RuntimeError("Single-model segmentation runtime was not initialized.")
        from rfdetr.training import RFDETRDataModule

        datamodule = RFDETRDataModule(rf_model.model_config, train_config)
        inference_runtime = trainer.get_inference_acceleration_handle(rf_model)
        result = trainer.manual_test_evaluation(
            trainer=None,
            pl_module=None,
            datamodule=datamodule,
            model_config=rf_model.model_config,
            train_config=train_config,
            output_dir=output_dir,
            split=str(test_settings.get("split", "test")),
            event="standalone_test",
            metadata={"event": "standalone_test_start", "dataset": dataset_metadata},
            merged_config=internal_config,
            source_config=source_config,
            verbose=bool(internal_config.get("runtime", {}).get("verbose", True)),
            progress_bar=bool(test_settings.get("progress_bar", True)),
            inference_runtime=inference_runtime,
        )
    else:
        if rf_model is None or train_config is None:
            raise RuntimeError("Single-model evaluator runtime was not initialized.")
        evaluator_config = trainer.build_rfdetr_evaluator_config(
            merged_config=internal_config,
            model_config=rf_model.model_config,
            train_config=train_config,
            output_dir=output_dir,
            split=str(test_settings.get("split", "test")),
            test_section="test",
        )
        result = run_evaluation(deepcopy(evaluator_config), source_config, prebuilt_model=rf_model, print_summary=True)
    if not use_segmentation_eval:
        trainer.write_rfdetr_evaluator_aliases(output_dir, result)
    if timing_context is not None and isinstance(result, Mapping):
        parallel_summary = result.get("parallel_summary")
        if isinstance(parallel_summary, Mapping):
            assignments = parallel_summary.get("assignments", [])
            if isinstance(assignments, list):
                worker_acceleration = [
                    {
                        "chunk_id": assignment.get("chunk_id"),
                        "device": assignment.get("device"),
                        "model_load_seconds": assignment.get("model_load_seconds"),
                        "acceleration": dict(assignment["acceleration"]),
                    }
                    for assignment in assignments
                    if isinstance(assignment, Mapping) and isinstance(assignment.get("acceleration"), Mapping)
                ]
                acceleration_summary = dict(timing_context.get("acceleration", {}))
                acceleration_summary["workers"] = worker_acceleration
                acceleration_summary["worker_model_load_seconds"] = sum(
                    float(item.get("model_load_seconds", 0.0) or 0.0)
                    for item in assignments
                    if isinstance(item, Mapping)
                )
                timing_context["acceleration"] = acceleration_summary
        stage_timing = normalized_stage_timing(result)
        if stage_timing:
            timing_context["stage_timing"] = stage_timing
    trainer.dump_config_snapshot(
        output_dir=output_dir,
        merged_config=internal_config,
        metadata={
            "event": "standalone_test_complete",
            "dataset": dataset_metadata,
            "acceleration": dict(timing_context.get("acceleration", {})) if timing_context is not None else {},
            "stage_timing": dict(timing_context.get("stage_timing", {})) if timing_context is not None else {},
        },
        source_config=source_config,
    )
    print_inference_timing_summary(result if isinstance(result, Mapping) else {}, output_dir)
    print(f"RF-DETR test output directory: {output_dir}")
    return 0


def main() -> int:
    """Run standalone test with elapsed-time reporting."""
    timing_context = trainer.start_run_timing("test")
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
