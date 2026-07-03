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
    - Before writing images, metrics, or cache files, the script prints a resource estimate and asks for confirmation unless --yes is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional

import colorama
import yaml
from colorama import Fore, Style
from tqdm import tqdm

import rf_detr_runtime as trainer
from projects.object_detection_dataset_evaluator.object_detection_dataset_evaluator import run_evaluation

colorama.init(autoreset=True)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "rf_detr_test.yaml"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def apply_cli_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    runtime = config.setdefault("runtime", {})
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
    stats_path = output_dir / "inference_stats.csv"
    if stats_path.exists():
        print(f"Per-image inference stats: {stats_path}")


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

    file_count = (
        config_files
        + metric_files
        + model_input_files
        + plot_files
        + dataset_case_files
        + visual_files
        + error_case_files
        + dataset_cache_files
    )
    image_outputs = model_input_files + max(0, visual_files - 2 if visual_files else 0) + max(0, error_case_files - 3 if error_case_files else 0) + dataset_case_files
    approx_bytes = dataset_cache_bytes + metric_files * 200_000 + image_outputs * 500_000 + plot_files * 350_000 + config_files * 50_000
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
        "estimated_total_files": file_count,
        "estimated_disk_usage": trainer.format_bytes(approx_bytes),
        "note": "Test output estimates are conservative. Prediction-filtered visuals and error cases are known exactly after inference.",
    }
    trainer.add_runtime_estimate(
        estimate=estimate,
        config=config,
        output_dir=output_dir,
        task="test",
        runtime_units=float(image_count or 0),
        default_rate_key="default_test_seconds_per_image",
        basis={"test_images": image_count, "split": split},
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
        "num_workers": int(test.get("num_workers", 2) or 2),
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
    parser.add_argument("--output-dir", help="Exact output directory override.")
    parser.add_argument("--test-mode", choices=["full_image", "sahi", "class_crop"], help="Test mode override.")
    parser.add_argument("--max-images", type=trainer.parse_scalar, help="Maximum test images to evaluate. Use all/null for all.")
    parser.add_argument("--batch-size", type=int, help="RF-DETR full-image/class-crop evaluator batch size.")
    parser.add_argument("--sahi-batch-size", type=int, help="RF-DETR SAHI slice/recheck batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned output without inference.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    args = parser.parse_args()

    with tqdm(total=6, desc="Preparing RF-DETR test", unit="step") as bar:
        source_config = Path(args.config).expanduser()
        if not source_config.is_absolute():
            source_config = (Path.cwd() / source_config).resolve()
        config = load_yaml(source_config)
        apply_cli_overrides(config, args)
        internal_config = build_internal_test_config(config)
        if timing_context is not None:
            timing_context["verbose"] = bool(internal_config.get("runtime", {}).get("verbose", True))
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

        model_cls = trainer.get_model_class(str(internal_config.get("model", {}).get("size", "medium")))
        rf_model = model_cls(**trainer.build_model_kwargs(internal_config))
        train_kwargs = trainer.build_train_kwargs(internal_config, output_dir)
        train_kwargs.pop("_device", None)
        train_config = rf_model.get_train_config(**train_kwargs)
        rf_model._align_num_classes_from_dataset(train_config.dataset_dir)
        bar.update(1)

    test_settings = internal_config.get("test", {})
    mode = trainer.periodic_test_mode(test_settings)
    evaluation_type = str(internal_config.get("evaluation", {}).get("type", "auto")).strip().lower()
    segmentation_model = bool(getattr(rf_model.model_config, "segmentation_head", False))
    use_segmentation_eval = (
        segmentation_model
        and mode == "full_image"
        and evaluation_type in {"auto", "segm", "segment", "segmentation", "mask", "masks"}
    )
    if use_segmentation_eval:
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule

        module = RFDETRModelModule(rf_model.model_config, train_config)
        datamodule = RFDETRDataModule(rf_model.model_config, train_config)
        result = trainer.manual_test_evaluation(
            trainer=None,
            pl_module=module,
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
        )
    else:
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
    print_inference_timing_summary(result if isinstance(result, Mapping) else {}, output_dir)
    print(f"RF-DETR test output directory: {output_dir}")
    return 0


def main() -> int:
    """Run standalone test with elapsed-time reporting."""
    timing_context = trainer.start_run_timing("test")
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
