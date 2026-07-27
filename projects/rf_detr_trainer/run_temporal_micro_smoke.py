"""Bounded one-window overfit smoke for real-temporal RF-DETR + TrackNetV5.

This is deliberately not a benchmark. It proves the smallest useful contract:
one real three-frame batch can run through RF-DETR, backpropagate through the
TrackNet heatmap branch, reduce a deterministic loss, save a compatible
checkpoint, and restore its TrackNet tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from rf_detr_temporal_runtime import TemporalCriterion

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent
            / "config"
            / "rf_detr_train_smoke_temporal_tracknet_v5.yaml"
        ),
    )
    parser.add_argument("--focus", choices=("single", "all"), default="all")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-minutes", type=float, default=20.0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "runs" / "rf_detr" / "micro_smoke"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 2:
        raise ValueError("--steps must be at least 2 so loss reduction can be measured")
    if not math.isfinite(args.max_minutes) or args.max_minutes <= 0:
        raise ValueError("--max-minutes must be a finite positive number")


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    """Terminate the worker and every descendant it may have spawned."""

    if process.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def supervise_worker(args: argparse.Namespace) -> int:
    """Run one worker under a wall-clock timeout without recursive spawning."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker",
    ]
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kwargs)
    timeout_seconds = args.max_minutes * 60.0
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        print(
            (
                "ERROR: Temporal micro-smoke exceeded the "
                f"{args.max_minutes:g}-minute wall-clock limit; "
                "the worker process tree was terminated."
            ),
            file=sys.stderr,
            flush=True,
        )
        return 124


def move_targets(
    targets: list[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    import torch

    return [
        {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def weighted_total(
    criterion: TemporalCriterion,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = criterion(outputs, targets)
    total = sum(
        value * criterion.weight_dict[key]
        for key, value in losses.items()
        if key in criterion.weight_dict
    )
    return total, losses


def sync(device: torch.device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_worker(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    import train_rf_detr_model as trainer
    from rf_detr_motion import (
        _find_lwdetr,
        attach_motion_module,
        load_motion_checkpoint_weights,
    )
    from rf_detr_temporal_runtime import (
        TemporalCriterion,
        build_temporal_datamodule,
    )

    started = time.monotonic()
    deadline = started + args.max_minutes * 60.0
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config_path = Path(args.config).expanduser().resolve()
    config = trainer.load_yaml(config_path)
    config.setdefault("model", {}).setdefault("motion", {}).setdefault("focus", {})[
        "mode"
    ] = args.focus
    config.setdefault("dataset", {}).setdefault("temporal", {})[
        "max_windows_per_split"
    ] = {"train": 1, "val": 1, "test": 1}

    output_dir = Path(args.output_dir).expanduser().resolve() / args.focus
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cls = trainer.get_model_class(str(config["model"].get("size", "small")))
    rf_model = model_cls(**trainer.build_model_kwargs(config))
    motion_config = dict(config["model"]["motion"])
    attach_motion_module(rf_model.model, motion_config)
    lwdetr = _find_lwdetr(rf_model.model)
    if lwdetr is None:
        raise RuntimeError("Could not locate the attached LWDETR instance")
    motion_module = lwdetr.motion_module
    device = next(lwdetr.parameters()).device

    train_kwargs = trainer.build_train_kwargs(config, output_dir)
    train_kwargs.pop("_device", None)
    train_config = rf_model.get_train_config(**train_kwargs)
    datamodule = build_temporal_datamodule(config, rf_model.model_config, train_config)
    datamodule.setup("fit")
    batch, targets = next(iter(datamodule.train_dataloader()))
    batch = batch.to(device)
    targets = move_targets(targets, device)

    from rfdetr.models.lwdetr import build_criterion_from_config

    base_criterion, _ = build_criterion_from_config(rf_model.model_config, train_config)
    loss_config = motion_config.get("loss", {}) or {}
    criterion = TemporalCriterion(
        base_criterion.to(device),
        heatmap_weight=float(loss_config.get("heatmap_weight", 1.0)),
        gamma=float(loss_config.get("gamma", 2.0)),
    ).to(device)

    # Keep the detector and zero-init fusion fixed. The short run then measures
    # deterministic heatmap learning instead of decoder dropout/noise.
    for parameter in lwdetr.parameters():
        parameter.requires_grad_(False)
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in motion_module.named_parameters():
        if name.startswith("fusions."):
            continue
        parameter.requires_grad_(True)
        trainable.append(parameter)
    if not trainable:
        raise RuntimeError("TrackNet branch exposed no trainable parameters")

    lwdetr.eval()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    history: list[dict[str, float]] = []

    def evaluate() -> tuple[float, float]:
        outputs = lwdetr(batch, targets)
        total, losses = weighted_total(criterion, outputs, targets)
        heatmap = losses["loss_tracknet_heatmap"]
        if not torch.isfinite(total) or not torch.isfinite(heatmap):
            raise RuntimeError("Temporal smoke produced a non-finite loss")
        return float(total.detach()), float(heatmap.detach())

    initial_total, initial_heatmap = evaluate()
    for step in range(args.steps):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Micro-smoke exceeded its {args.max_minutes:g}-minute safety limit"
            )
        optimizer.zero_grad(set_to_none=True)
        outputs = lwdetr(batch, targets)
        total, losses = weighted_total(criterion, outputs, targets)
        heatmap = losses["loss_tracknet_heatmap"]
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 5.0))
        if not np.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError(f"Invalid TrackNet gradient norm at step {step}: {gradient_norm}")
        optimizer.step()
        history.append(
            {
                "step": float(step + 1),
                "total_loss": float(total.detach()),
                "heatmap_loss": float(heatmap.detach()),
                "gradient_norm": gradient_norm,
            }
        )

    final_total, final_heatmap = evaluate()
    if not final_heatmap < initial_heatmap:
        raise RuntimeError(
            f"Heatmap loss did not decrease: {initial_heatmap:.6f} -> {final_heatmap:.6f}"
        )
    if not final_total < initial_total:
        raise RuntimeError(
            f"Total weighted loss did not decrease: {initial_total:.6f} -> {final_total:.6f}"
        )

    architecture = trainer.build_pitchobjectlab_architecture(config)
    checkpoint_path = output_dir / "checkpoint_micro_smoke.pth"
    checkpoint = {
        "model": lwdetr.state_dict(),
        "args": {
            "class_names": list(getattr(datamodule, "class_names", ["soccer_ball"])),
            "num_queries": int(rf_model.model_config.num_queries),
            "group_detr": int(rf_model.model_config.group_detr),
            trainer.PITCHOBJECTLAB_ARCHITECTURE_KEY: architecture,
        },
        trainer.PITCHOBJECTLAB_ARCHITECTURE_KEY: architecture,
    }
    torch.save(checkpoint, checkpoint_path)

    reference_key, reference_value = next(iter(motion_module.state_dict().items()))
    reference_value = reference_value.detach().clone()
    with torch.no_grad():
        dict(motion_module.state_dict())[reference_key].zero_()
    load_motion_checkpoint_weights(lwdetr, checkpoint_path)
    if not torch.equal(motion_module.state_dict()[reference_key], reference_value):
        raise RuntimeError("TrackNet checkpoint reload did not restore the saved tensor")

    sync(device)
    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "success": True,
        "focus": args.focus,
        "steps": args.steps,
        "resolution": int(rf_model.model_config.resolution),
        "batch_shape": list(batch.frames.shape),
        "initial_total_loss": initial_total,
        "final_total_loss": final_total,
        "total_loss_reduction": initial_total - final_total,
        "initial_heatmap_loss": initial_heatmap,
        "final_heatmap_loss": final_heatmap,
        "heatmap_loss_reduction": initial_heatmap - final_heatmap,
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload": True,
        "dataset_manifest_sha256": architecture.get("dataset_manifest_sha256"),
        "elapsed_seconds": elapsed,
        "history": history,
    }
    (output_dir / "micro_smoke_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not args.worker:
        return supervise_worker(args)
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
