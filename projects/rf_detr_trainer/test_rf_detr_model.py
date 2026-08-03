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
    - test.parallel.chunks or --chunks starts that many concurrent model replicas for bbox evaluation.
    - Before writing images, metrics, or cache files, the script prints a resource estimate and asks for confirmation unless --yes is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, MutableMapping, Optional

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
        test.setdefault("parallel", {})["chunks"] = args.chunks


def normalize_test_parallel_chunks(value: Any) -> int:
    """Validate test.parallel.chunks as a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("test.parallel.chunks must be a positive integer.")
    return int(value)


def standalone_parallel_chunks(config: Mapping[str, Any]) -> int:
    """Return the normalized standalone-test chunk count (default one)."""
    test_settings = config.get("test", {})
    if not isinstance(test_settings, Mapping):
        raise ValueError("test must be a mapping.")
    parallel = test_settings.get("parallel", {})
    if parallel is None:
        parallel = {}
    if not isinstance(parallel, Mapping):
        raise ValueError("test.parallel must be a mapping.")
    return normalize_test_parallel_chunks(parallel.get("chunks", 1))


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


def build_test_parallel_plan(config: Mapping[str, Any], image_count: Optional[int]) -> Dict[str, Any]:
    """Build deterministic chunk/device assignments for estimates and execution."""
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
    return {
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


def normalized_stage_timing(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return additive timing totals in the shared run_timing.json schema."""

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
        return timing
    stats = result.get("stats")
    if not isinstance(stats, list):
        return {}
    rows = [row for row in stats if isinstance(row, Mapping)]
    if not rows:
        return {}

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
    return timing


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
        "parallel_model_replicas": parallel_plan["chunks"],
        "parallel_devices_requested": parallel_plan["devices_requested"],
        "parallel_chunk_devices": parallel_plan["chunk_devices"],
        "parallel_device_worker_counts": parallel_plan["device_worker_counts"],
        "parallel_summary_files": parallel_summary_files,
        "parallel_same_device_warning": parallel_plan["same_device_warning"],
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
        runtime_units=float(image_count or 0) / float(parallel_plan["unique_device_count"]),
        default_rate_key="default_test_seconds_per_image",
        basis={
            "test_images": image_count,
            "split": split,
            "parallel_chunks": parallel_plan["chunks"],
            "parallel_speedup_cap": parallel_plan["unique_device_count"],
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
    parallel_source = test.get("parallel", {}) or {}
    if not isinstance(parallel_source, Mapping):
        raise ValueError("test.parallel must be a mapping.")
    parallel = dict(parallel_source)
    parallel["chunks"] = normalize_test_parallel_chunks(parallel.get("chunks", 1))
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
        "conf": model.get("confidence_threshold", 0.25),
        "match_iou_threshold": evaluation.get("match_iou_threshold", 0.5),
        "max_dets": _last_max_det(evaluation),
    }
    internal["test"] = test_settings

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
    parser.add_argument("--sahi-batch-size", type=int, help="RF-DETR SAHI slice/recheck batch size.")
    parser.add_argument("--chunks", type=int, help="Concurrent standalone test chunks/model replicas.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned output without inference.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    cpu_runtime.add_cpu_cli_arguments(parser)
    args = parser.parse_args()

    with tqdm(total=6, desc="Preparing RF-DETR test", unit="step") as bar:
        source_config = Path(args.config).expanduser()
        if not source_config.is_absolute():
            source_config = (Path.cwd() / source_config).resolve()
        config = load_yaml(source_config)
        apply_cli_overrides(config, args)
        cpu_summary = cpu_runtime.validate_active_config(config, "test", source_config)
        print(cpu_runtime.format_summary(cpu_summary))
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
        dataset_plan = trainer.build_dataset_plan(internal_config, output_dir, source_config)
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
