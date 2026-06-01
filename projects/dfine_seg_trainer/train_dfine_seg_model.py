"""
Train D-FINE-seg with a config-first, Ultralytics-style workflow.

This wrapper adds:

1. Dataset auto-detection and conversion into D-FINE-seg COCO-style cache.
2. Configurable rich augmentation controls inspired by Ultralytics.
3. Custom output folders with placeholders.
4. Resource/file estimates and confirmation before heavy outputs.
5. Reproducibility config snapshots in every run folder.
6. CPU, single-GPU, and torchrun multi-GPU launch support on Windows and Linux.

Example usage:

    uv run python train_dfine_seg_model.py --config config/dfine_seg_train.yaml --dry-run --yes

    uv run python train_dfine_seg_model.py \\
        --dataset-dir D:/datasets/my_seg_dataset \\
        --dataset-format auto \\
        --task segment \\
        --model-name s \\
        --device 0 \\
        --epochs 100 \\
        --batch-size 4 \\
        --output-dir D:/runs/dfine_seg/my_run \\
        --yes

    uv run python train_dfine_seg_model.py \\
        --dataset-dir /data/voc_box_dataset \\
        --task segment \\
        --extra dataset.box_to_mask=true \\
        --yes

    uv run python train_dfine_seg_model.py --demo --dry-run --yes
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import colorama
from colorama import Fore, Style
from tqdm import tqdm

from dfine_seg_trainer.augmentation import build_aug_runtime
from dfine_seg_trainer.common import (
    DEFAULT_CONFIG,
    PROJECT_DIR,
    REPO_ROOT,
    VENDOR_DIR,
    VENDORED_DFINE_COMMIT,
    build_output_dir,
    copy_config_for_mutation,
    deep_update,
    environment_snapshot,
    format_bytes,
    get_by_dot_path,
    load_yaml,
    now_timestamp,
    parse_bool,
    parse_extra_args,
    parse_scalar,
    render_template,
    resolve_region,
    resolve_training_device,
    run_command,
    save_yaml,
    wait_for_enter_confirmation,
    write_json,
)
from dfine_seg_trainer.dataset_adapter import PreparedDataset, build_dataset_plan, materialize_dataset

colorama.init(autoreset=True)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

LR_TABLE = {
    "n": {"backbone_lr": 0.0004, "base_lr": 0.0008},
    "s": {"backbone_lr": 0.00006, "base_lr": 0.00025},
    "m": {"backbone_lr": 0.00002, "base_lr": 0.00015},
    "l": {"backbone_lr": 0.00001, "base_lr": 0.00016},
    "x": {"backbone_lr": 0.000002, "base_lr": 0.0002},
}


def blue(message: str, verbose: bool = True, force: bool = False) -> None:
    """Print English blue wrapper text."""
    if verbose or force:
        print(Fore.BLUE + Style.BRIGHT + message)


def apply_cli_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI arguments to config."""
    runtime = config.setdefault("runtime", {})
    model = config.setdefault("model", {})
    dataset = config.setdefault("dataset", {})
    output = config.setdefault("output", {})
    train = config.setdefault("train", {})

    if args.verbose is not None:
        runtime["verbose"] = args.verbose
    if args.dry_run:
        runtime["dry_run"] = True
    if args.confirm_before_run is not None:
        runtime["confirm_before_run"] = args.confirm_before_run
    if args.task is not None:
        model["task"] = args.task
    if args.model_name is not None:
        model["name"] = args.model_name
    if args.dataset_dir is not None:
        dataset["dataset_dir"] = args.dataset_dir
    if args.dataset_format is not None:
        dataset["source_format"] = args.dataset_format
    if args.device is not None:
        train["device"] = args.device
    if args.gpus is not None:
        train["gpus"] = parse_scalar(args.gpus)
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.batch_size is not None:
        train["batch_size"] = args.batch_size
    if args.output_dir is not None:
        output["output_dir"] = args.output_dir
    if args.project is not None:
        output["root"] = args.project
    if args.name is not None:
        output["name"] = args.name
    if args.exist_ok is not None:
        output["exist_ok"] = args.exist_ok
    if args.demo is not None:
        config.setdefault("demo", {})["enabled"] = args.demo

    extra = parse_extra_args(args.extra)
    if extra:
        deep_update(config, extra)


def apply_demo_mode(config: MutableMapping[str, Any], timestamp: str, region: str, verbose: bool) -> None:
    """Clamp config for demo mode."""
    demo = config.get("demo", {})
    if not bool(demo.get("enabled", False)):
        return
    train = config.setdefault("train", {})
    output = config.setdefault("output", {})
    output["output_dir"] = render_template(demo.get("output_dir", "demo_runs/dfine_demo_{timestamp}"), config, timestamp, region)
    train["epochs"] = min(int(train.get("epochs", 1)), int(demo.get("max_epochs", 2)))
    if isinstance(train.get("batch_size"), int) and int(train["batch_size"]) > 0:
        train["batch_size"] = min(int(train["batch_size"]), int(demo.get("max_batch_size", 2)))
    if bool(demo.get("disable_heavy_outputs", True)):
        train["debug_img_processing"] = False
        train["to_visualize_eval"] = False
    blue("Demo mode enabled: epochs, batch size, and heavy image outputs were clamped.", verbose)


def estimate_outputs(config: Mapping[str, Any], output_dir: Path, dataset_plan: Any) -> dict[str, Any]:
    """Estimate output file/resource counts before training."""
    train = config.get("train", {})
    epochs = int(train.get("epochs", 1))
    debug_enabled = bool(train.get("debug_img_processing", False))
    eval_enabled = bool(train.get("to_visualize_eval", True))
    checkpoint_files = 2
    metrics_files = 6
    config_files = 5
    debug_images = min(100, int(dataset_plan.estimate.get("source_image_files", 0))) if debug_enabled else 0
    eval_images = min(max(int(train.get("batch_size", 1) if isinstance(train.get("batch_size"), int) and train.get("batch_size") > 0 else 4) * 6, 6), int(dataset_plan.estimate.get("source_image_files", 0))) if eval_enabled else 0
    plots = 12 if eval_enabled else 0
    cache_files = int(dataset_plan.estimate.get("conversion_output_files_estimate", 0))
    source_bytes = int(dataset_plan.estimate.get("source_image_bytes", 0))
    link_mode = str(config.get("dataset", {}).get("link_mode", "auto"))
    cache_bytes = source_bytes if link_mode == "copy" else min(source_bytes, 1024 * 1024 * 20)
    return {
        "output_dir": str(output_dir),
        "epochs": epochs,
        "checkpoint_files_estimate": checkpoint_files,
        "metrics_and_config_files_estimate": metrics_files + config_files,
        "debug_images_estimate": debug_images,
        "eval_images_estimate": eval_images,
        "plot_files_estimate": plots,
        "dataset_cache_files_estimate": cache_files,
        "dataset_cache_disk_estimate": cache_bytes,
        "source_images": dataset_plan.estimate.get("source_image_files", 0),
        "source_image_bytes": source_bytes,
        "notes": [
            "Checkpoint size depends on model size and optimizer state.",
            "Dataset cache uses hardlinks/symlinks when possible unless link_mode=copy.",
        ],
    }


def print_estimate(estimate: Mapping[str, Any], verbose: bool) -> None:
    """Print a concise resource estimate."""
    blue("Planned output/resource estimate:", verbose, force=True)
    blue(f"  Output directory: {estimate['output_dir']}", verbose, force=True)
    blue(f"  Source images: {estimate['source_images']} ({format_bytes(estimate['source_image_bytes'])})", verbose, force=True)
    blue(
        "  Estimated generated files: "
        f"checkpoints={estimate['checkpoint_files_estimate']}, "
        f"metrics/config={estimate['metrics_and_config_files_estimate']}, "
        f"debug_images={estimate['debug_images_estimate']}, "
        f"eval_images={estimate['eval_images_estimate']}, "
        f"plots={estimate['plot_files_estimate']}, "
        f"cache_files={estimate['dataset_cache_files_estimate']}",
        verbose,
        force=True,
    )
    blue(f"  Dataset cache disk estimate: {format_bytes(estimate['dataset_cache_disk_estimate'])}", verbose, force=True)


def confirm_or_exit(config: Mapping[str, Any], args: argparse.Namespace, estimate: Mapping[str, Any], verbose: bool) -> None:
    """Ask confirmation before heavy outputs."""
    print_estimate(estimate, verbose)
    confirm = bool(config.get("runtime", {}).get("confirm_before_run", True))
    if args.yes or not confirm:
        return
    if not wait_for_enter_confirmation("Continue and create these outputs? [y/N] "):
        raise SystemExit("Aborted before creating training outputs.")


def class_names_or_config(prepared: PreparedDataset, config: Mapping[str, Any]) -> dict[int, str]:
    """Return D-FINE label_to_name mapping."""
    names = prepared.class_names or []
    if not names:
        names_cfg = config.get("dataset", {}).get("names") or []
        if isinstance(names_cfg, Mapping):
            return {int(k): str(v) for k, v in names_cfg.items()}
        names = [str(x) for x in names_cfg]
    if not names:
        names = ["class_0"]
    return {idx: name for idx, name in enumerate(names)}


def resolve_pretrained_path(config: Mapping[str, Any], region: str, verbose: bool) -> str:
    """Resolve pretrained weight path and set mirror environment when useful."""
    model = config.get("model", {})
    task = str(model.get("task", "segment"))
    name = str(model.get("name", "s")).lower()
    raw = model.get("pretrained_model_path", "auto")
    if raw in {None, False, "none", "None", ""}:
        return ""
    if str(raw).lower() != "auto":
        return str(raw)
    filename = f"dfine_seg_{name}_coco.pt" if task == "segment" else f"dfine_{name}_{model.get('pretrained_dataset', 'coco')}.pt"
    path = VENDOR_DIR / "pretrained" / filename
    if path.exists():
        return str(path)
    if region == "china":
        source = str(config.get("network", {}).get("model_source", "auto")).lower()
        if source in {"auto", "modelscope"}:
            repo_id = str(config.get("network", {}).get("modelscope_repo", "ArgoSA/D-FINE-seg"))
            downloaded = try_modelscope_download(repo_id, filename, path, verbose)
            if downloaded:
                return str(downloaded)
        blue("Pretrained weights are missing. China mode will use HF_ENDPOINT fallback for this run.", verbose, force=True)
    return str(path)


def try_modelscope_download(repo_id: str, filename: str, destination: Path, verbose: bool) -> Path | None:
    """Best-effort ModelScope file download for China-region runs."""
    try:
        try:
            from modelscope import snapshot_download
        except ImportError:
            from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        blue("ModelScope is not installed; skipping ModelScope pretrained download.", verbose, force=True)
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        blue(f"Downloading pretrained weights from ModelScope repo {repo_id}.", verbose, force=True)
        snapshot_dir = Path(snapshot_download(repo_id, local_dir=str(destination.parent)))
    except Exception as exc:
        blue(f"ModelScope download failed: {exc}", verbose, force=True)
        return None

    for candidate in [snapshot_dir / filename, *snapshot_dir.rglob(filename)]:
        if candidate.exists():
            if candidate.resolve() != destination.resolve():
                shutil.copy2(candidate, destination)
            return destination
    blue(f"ModelScope repo did not contain {filename}; falling back.", verbose, force=True)
    return None


def maybe_prepare_model_mirror_env(config: Mapping[str, Any], region: str) -> dict[str, str]:
    """Return environment variables for model download mirrors."""
    env: dict[str, str] = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    network = config.get("network", {})
    hf_endpoint = str(network.get("hf_endpoint") or "").strip()
    if region == "china" and not hf_endpoint:
        hf_endpoint = "https://hf-mirror.com"
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    return env


def build_dfine_runtime_config(
    config: Mapping[str, Any],
    prepared: PreparedDataset,
    output_dir: Path,
    timestamp: str,
    region: str,
    verbose: bool,
) -> dict[str, Any]:
    """Translate wrapper config into the D-FINE-seg Hydra config."""
    model = config.get("model", {})
    train = config.get("train", {})
    task = str(model.get("task", "segment")).lower()
    model_name = str(model.get("name", "s")).lower()
    lr_defaults = LR_TABLE.get(model_name, LR_TABLE["s"])
    aug_runtime = build_aug_runtime(config.get("augmentation", {}), task)
    label_to_name = class_names_or_config(prepared, config)
    pretrained_path = resolve_pretrained_path(config, region, verbose)

    runtime = {
        "project_name": f"dfine_seg_{task}",
        "exp_name": output_dir.name,
        "exp": output_dir.name,
        "model_name": model_name,
        "task": task,
        "train": {
            "root": str(output_dir),
            "pretrained_dataset": str(model.get("pretrained_dataset", "coco")),
            "pretrained_model_path": pretrained_path,
            "coco_dataset": True,
            "data_path": str(prepared.path),
            "path_to_test_data": str(prepared.path / "images" / "test"),
            "path_to_save": str(output_dir),
            "debug_img_path": str(output_dir / "debug_images"),
            "eval_preds_path": str(output_dir / "eval_preds"),
            "bench_img_path": str(output_dir / "bench_imgs"),
            "infer_path": str(output_dir / "infer"),
            "use_wandb": bool(train.get("use_wandb", False)),
            "device": str(train.get("device", "cpu")),
            "label_to_name": label_to_name,
            "use_one_class": bool(train.get("use_one_class", False)),
            "ddp": {
                "enabled": bool(get_by_dot_path(config, "train.ddp.enabled", False)),
                "n_gpus": int(get_by_dot_path(config, "train.ddp.n_gpus", 1)),
            },
            "decision_metrics": list(train.get("decision_metrics", ["f1", "mAP_50", "iou"])),
            "img_size": list(train.get("img_size", [640, 640])),
            "in_channels": int(model.get("in_channels", 3)),
            "keep_ratio": bool(train.get("keep_ratio", False)),
            "to_visualize_eval": bool(train.get("to_visualize_eval", True)),
            "debug_img_processing": bool(train.get("debug_img_processing", False)),
            "amp_enabled": bool(train.get("amp_enabled", True)),
            "clip_max_norm": float(train.get("clip_max_norm", 0.1) or 0.0),
            "batch_size": int(train.get("batch_size", -1)),
            "b_accum_steps": int(train.get("b_accum_steps", 1)),
            "epochs": int(train.get("epochs", 75)),
            "early_stopping": int(train.get("early_stopping", 0)),
            "ignore_background_epochs": int(train.get("ignore_background_epochs", 0)),
            "num_workers": int(train.get("num_workers", 4)),
            "mask_batch_size": int(train.get("mask_batch_size", 150)),
            "conf_thresh": float(train.get("conf_thresh", 0.5)),
            "iou_thresh": float(train.get("iou_thresh", 0.5)),
            "use_ema": bool(train.get("use_ema", True)),
            "ema_momentum": float(train.get("ema_momentum", 0.9998)),
            "use_scheduler": bool(train.get("use_scheduler", True)),
            "base_lr": float(train.get("base_lr") or lr_defaults["base_lr"]),
            "backbone_lr": float(train.get("backbone_lr") or lr_defaults["backbone_lr"]),
            "cycler_pct_start": float(train.get("cycler_pct_start", 0.1)),
            "weight_decay": float(train.get("weight_decay", 0.000125)),
            "betas": list(train.get("betas", [0.9, 0.999])),
            "label_smoothing": float(train.get("label_smoothing", 0.0)),
            "seed": int(train.get("seed", 42)),
            "cudnn_fixed": bool(train.get("cudnn_fixed", False)),
            "lrs": LR_TABLE,
            **aug_runtime,
        },
        "split": {
            "ignore_negatives": False,
            "shuffle": True,
            "train_split": 0.8,
            "val_split": 0.1,
        },
        "export": {
            "half": bool(get_by_dot_path(config, "export.half", True)),
            "max_batch_size": int(get_by_dot_path(config, "export.max_batch_size", 1)),
            "dynamic_input": bool(get_by_dot_path(config, "export.dynamic_input", False)),
            "ov_int8_max_drop": 0.02,
            "trt_int8_workspace_gb": 4,
            "trt_int8_validate": True,
        },
        "infer": {
            "to_crop": True,
            "to_track": False,
            "paddings": {"w": 0.05, "h": 0.05},
        },
        "defaults": [
            "_self_",
            {"override hydra/hydra_logging": "disabled"},
            {"override hydra/job_logging": "disabled"},
        ],
        "hydra": {"output_subdir": None, "run": {"dir": "."}},
        "now_dir": timestamp[:8],
        "wrapper_metadata": {
            "region": region,
            "vendored_dfine_commit": VENDORED_DFINE_COMMIT,
            "dataset_metadata": str(prepared.metadata_path),
        },
    }
    return runtime


def dump_snapshots(
    output_dir: Path,
    merged_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    source_config: Path,
    prepared: PreparedDataset | None,
    region: str,
    event: str,
) -> None:
    """Write config and environment snapshots into the output folder."""
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config_dir / "merged_config.yaml", merged_config)
    save_yaml(config_dir / "dfine_runtime_config.yaml", runtime_config)
    if source_config.exists():
        shutil.copy2(source_config, config_dir / "source_config.yaml")
    write_json(config_dir / "environment.json", environment_snapshot(region))
    write_json(
        config_dir / "run_metadata.json",
        {
            "event": event,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "output_dir": str(output_dir),
            "prepared_dataset": str(prepared.path) if prepared else None,
            "prepared_dataset_metadata": str(prepared.metadata_path) if prepared else None,
        },
    )


def build_train_command(config: Mapping[str, Any], runtime_config_path: Path, gpus: list[int]) -> list[str]:
    """Build single-process or torchrun command."""
    ddp_enabled = bool(get_by_dot_path(config, "train.ddp.enabled", False))
    n_gpus = int(get_by_dot_path(config, "train.ddp.n_gpus", len(gpus) or 1))
    config_name = runtime_config_path.stem
    if ddp_enabled and n_gpus > 1:
        return [
            "torchrun",
            f"--nproc_per_node={n_gpus}",
            f"--master_port={int(get_by_dot_path(config, 'train.ddp.master_port', 29500))}",
            "-m",
            "src.dl.train",
            "--config-name",
            config_name,
        ]
    return [sys.executable, "-m", "src.dl.train", "--config-name", config_name]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="D-FINE-seg trainer wrapper with dataset conversion, rich augmentation config, and reproducible outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to wrapper YAML config.")
    parser.add_argument("--yes", action="store_true", help="Skip output/resource confirmation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate without training.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None, help="Enable blue wrapper logs.")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Disable blue wrapper logs.")
    parser.add_argument("--confirm-before-run", type=parse_bool, default=None, help="Ask before creating heavy outputs.")
    parser.add_argument("--demo", action="store_true", default=None, help="Enable demo constraints.")
    parser.add_argument("--no-demo", dest="demo", action="store_false", help="Disable demo constraints.")
    parser.add_argument("--task", choices=["detect", "segment"], default=None, help="D-FINE task.")
    parser.add_argument("--dataset-dir", default=None, help="Source dataset root.")
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "dfine_coco", "coco_json", "roboflow_coco", "roboflow_yolo", "ultralytics_yolo", "labelme", "pascal_voc", "dota"],
        default=None,
        help="Source dataset format.",
    )
    parser.add_argument("--model-name", choices=["n", "s", "m", "l", "x"], default=None, help="D-FINE model size.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, 0, 0,1, or mps.")
    parser.add_argument("--gpus", default=None, help="GPU ids such as 0 or 0,1 or [0,1].")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Physical batch size per process, -1 for auto.")
    parser.add_argument("--output-dir", default=None, help="Exact output directory with placeholders.")
    parser.add_argument("--project", default=None, help="Output root when output-dir is empty.")
    parser.add_argument("--name", default=None, help="Output run folder name when output-dir is empty.")
    parser.add_argument("--exist-ok", type=parse_bool, default=None, help="Allow existing output directory.")
    parser.add_argument("--extra", action="append", default=None, help="Extra config override as dot.path=value. Repeatable.")
    return parser


def main() -> int:
    """
    Main entry point.

    Usage:
      1. Prepare environment:
         uv run python setup_dfine_seg_env.py
         uv sync
      2. Edit config/dfine_seg_train.yaml.
      3. Validate outputs:
         uv run python train_dfine_seg_model.py --dry-run --yes
      4. Train:
         uv run python train_dfine_seg_model.py

    Example usage:
      uv run python train_dfine_seg_model.py --config config/dfine_seg_train.yaml --dry-run --yes
      uv run python train_dfine_seg_model.py --dataset-dir D:/datasets/my_dataset --task segment --model-name s --device 0 --epochs 100 --yes
      uv run python train_dfine_seg_model.py --dataset-dir /data/voc --task segment --extra dataset.box_to_mask=true --yes
      uv run python train_dfine_seg_model.py --device cpu --demo --yes
    """
    args = build_parser().parse_args()
    source_config = Path(args.config).expanduser()
    if not source_config.is_absolute():
        source_config = (Path.cwd() / source_config).resolve()
    if not source_config.exists():
        raise FileNotFoundError(f"Config file not found: {source_config}")
    if not VENDOR_DIR.exists():
        raise FileNotFoundError(f"Vendored D-FINE-seg snapshot is missing: {VENDOR_DIR}")

    timestamp = now_timestamp()
    with tqdm(total=8, desc="Preparing", disable=False) as bar:
        config = copy_config_for_mutation(load_yaml(source_config))
        apply_cli_overrides(config, args)
        verbose = bool(config.get("runtime", {}).get("verbose", True))
        region = resolve_region(config)
        device, gpus = resolve_training_device(config)
        config.setdefault("network", {})["resolved_region"] = region
        blue(f"Resolved region={region}, device={device}, gpus={gpus or 'none'}.", verbose)
        apply_demo_mode(config, timestamp, region, verbose)
        bar.update(1)

        output_dir = build_output_dir(config, timestamp, region)
        if output_dir.exists() and not bool(config.get("output", {}).get("exist_ok", False)):
            raise FileExistsError(f"Output directory already exists and output.exist_ok=false: {output_dir}")
        bar.set_description("Dataset plan")
        dataset_plan = build_dataset_plan(config)
        bar.update(1)

        estimate = estimate_outputs(config, output_dir, dataset_plan)
        confirm_or_exit(config, args, estimate, verbose)
        if bool(config.get("runtime", {}).get("dry_run", False)) or args.dry_run:
            blue("Dry run complete. Training was not started and dataset cache was not materialized.", verbose, force=True)
            bar.update(bar.total - bar.n)
            return 0
        bar.set_description("Dataset cache")
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared = materialize_dataset(dataset_plan, config, progress=bool(config.get("runtime", {}).get("progress_bar", True)))
        config.setdefault("dataset", {})["prepared_dir"] = str(prepared.path)
        bar.update(1)

        runtime_config = build_dfine_runtime_config(config, prepared, output_dir, timestamp, region, verbose)
        dump_snapshots(output_dir, config, runtime_config, source_config, prepared, region, event="pre_train")
        runtime_config_path = VENDOR_DIR / "_dfine_wrapper_runtime.yaml"
        save_yaml(runtime_config_path, runtime_config)
        bar.update(1)

        command = build_train_command(config, runtime_config_path, gpus)
        env = maybe_prepare_model_mirror_env(config, region)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
        bar.set_description("Launching")
        blue(f"Launching D-FINE-seg: {' '.join(command)}", verbose, force=True)
        bar.update(1)

        bar.set_description("Training")
        return_code = run_command(command, cwd=VENDOR_DIR, env=env)
        bar.update(1)
        if return_code != 0:
            dump_snapshots(output_dir, config, runtime_config, source_config, prepared, region, event="train_failed")
            raise SystemExit(return_code)

        dump_snapshots(output_dir, config, runtime_config, source_config, prepared, region, event="train_complete")
        bar.update(1)
        bar.set_description("Done")
        bar.update(1)

    blue(f"Training output directory: {output_dir}", verbose=True, force=True)
    blue("Done.", verbose=True, force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
