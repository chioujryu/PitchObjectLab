"""Bounded one-window overfit smoke for real-temporal RF-DETR + TrackNetV5.

This is deliberately not a benchmark. It proves the smallest useful contract:
one real three-frame batch can run through RF-DETR, backpropagate through the
TrackNet heatmap branch, reduce a deterministic loss, save a compatible
checkpoint, and restore its TrackNet tensors.
"""

from __future__ import annotations

import argparse
import gc
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
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-minutes", type=float, default=20.0)
    parser.add_argument("--model-size", choices=("small", "medium", "large"), default="small")
    parser.add_argument(
        "--p2",
        choices=("on", "off"),
        default="on",
        help="Enable or disable the stride-4 P2 feature graph.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run Small/Medium/Large with P2 off and on.",
    )
    parser.add_argument(
        "--official-resolution",
        action="store_true",
        help="Use RF-DETR's official 512/576/704 resolution instead of the smoke resolution.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Construct a fresh model and strictly reload the saved checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "runs" / "rf_detr" / "micro_smoke"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    minimum_steps = 1 if args.matrix else 2
    if args.steps < minimum_steps:
        raise ValueError(f"--steps must be at least {minimum_steps}")
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


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    import torch

    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def _parameter_delta(
    parameters: list[torch.nn.Parameter],
    before: list[torch.Tensor],
) -> float:
    return float(
        sum(
            (parameter.detach().cpu() - reference).abs().sum().item()
            for parameter, reference in zip(parameters, before)
        )
    )


def run_case(
    args: argparse.Namespace,
    *,
    model_size: str,
    p2_enabled: bool,
    deadline: float,
) -> dict[str, Any]:
    import numpy as np
    import torch

    import train_rf_detr_model as trainer
    from rf_detr_motion import (
        _find_lwdetr,
        assert_motion_checkpoint_compatible,
        attach_motion_module,
    )
    from rf_detr_p2 import assert_p2_checkpoint_compatible
    from rf_detr_temporal_runtime import TemporalCriterion, build_temporal_datamodule

    started = time.monotonic()
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = trainer.load_yaml(Path(args.config).expanduser().resolve())
    model = config.setdefault("model", {})
    model["size"] = model_size
    model["resolution"] = None if args.official_resolution else 128
    model.setdefault("p2", {})["enabled"] = p2_enabled
    model.setdefault("motion", {}).setdefault("focus", {})["mode"] = args.focus
    config.setdefault("dataset", {}).setdefault("temporal", {})[
        "max_windows_per_split"
    ] = {"train": 1, "val": 1, "test": 1}

    case_name = f"{model_size}_{'p2_' if p2_enabled else ''}tracknet_{args.focus}"
    output_dir = Path(args.output_dir).expanduser().resolve() / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cls = trainer.get_model_class(model_size)
    rf_model = model_cls(**trainer.build_model_kwargs(config))
    motion_config = dict(model["motion"])
    attach_motion_module(rf_model.model, motion_config)
    lwdetr = _find_lwdetr(rf_model.model)
    if lwdetr is None:
        raise RuntimeError("Could not locate the attached LWDETR instance")
    requested_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lwdetr.to(requested_device)
    motion_module = lwdetr.motion_module
    device = next(lwdetr.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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

    for parameter in lwdetr.parameters():
        parameter.requires_grad_(False)
    tracknet_parameters = list(motion_module.parameters())
    detector_parameters = [
        parameter
        for name, parameter in lwdetr.named_parameters()
        if not name.startswith("motion_module.")
        and any(
            token in name
            for token in (
                "transformer.decoder",
                "class_embed",
                "bbox_embed",
                "enc_out_",
                "refpoint_embed",
                "query_embed",
            )
        )
    ]
    for parameter in (*tracknet_parameters, *detector_parameters):
        parameter.requires_grad_(True)
    if not tracknet_parameters or not detector_parameters:
        raise RuntimeError("Smoke requires trainable TrackNet and detector-head parameters")

    tracknet_before = [parameter.detach().cpu().clone() for parameter in tracknet_parameters]
    detector_before = [parameter.detach().cpu().clone() for parameter in detector_parameters]
    trainable = [*tracknet_parameters, *detector_parameters]
    lwdetr.eval()
    optimizer = torch.optim.AdamW(
        [
            {"params": tracknet_parameters, "lr": args.learning_rate},
            {
                "params": detector_parameters,
                "lr": min(args.learning_rate, 1e-3),
            },
        ],
        weight_decay=0.0,
    )
    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    history: list[dict[str, float]] = []

    def evaluate(
        model_object: torch.nn.Module,
        batch_object: Any,
        target_objects: list[dict[str, torch.Tensor]],
        criterion_object: TemporalCriterion,
    ) -> tuple[float, float, float]:
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = model_object(batch_object, target_objects)
            total, losses = weighted_total(
                criterion_object,
                outputs,
                target_objects,
            )
        heatmap = losses["loss_tracknet_heatmap"]
        best_iou = criterion_object.last_diagnostics["best_box_iou"]
        if not torch.isfinite(total) or not torch.isfinite(heatmap):
            raise RuntimeError("Temporal smoke produced a non-finite loss")
        return float(total), float(heatmap), float(best_iou)

    initial_total, initial_heatmap, initial_iou = evaluate(
        lwdetr,
        batch,
        targets,
        criterion,
    )
    for step in range(args.steps):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Micro-smoke exceeded its {args.max_minutes:g}-minute safety limit"
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = lwdetr(batch, targets)
            total, losses = weighted_total(criterion, outputs, targets)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        tracknet_gradient_norm = _gradient_norm(tracknet_parameters)
        detector_gradient_norm = _gradient_norm(detector_parameters)
        if not np.isfinite(tracknet_gradient_norm) or tracknet_gradient_norm <= 0:
            raise RuntimeError(
                f"Invalid TrackNet gradient norm at step {step}: {tracknet_gradient_norm}"
            )
        if not np.isfinite(detector_gradient_norm) or detector_gradient_norm <= 0:
            raise RuntimeError(
                f"Invalid detector gradient norm at step {step}: {detector_gradient_norm}"
            )
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        scaler.step(optimizer)
        scaler.update()
        history.append(
            {
                "step": float(step + 1),
                "total_loss": float(total.detach()),
                "heatmap_loss": float(losses["loss_tracknet_heatmap"].detach()),
                "tracknet_gradient_norm": tracknet_gradient_norm,
                "detector_gradient_norm": detector_gradient_norm,
            }
        )

    final_total, final_heatmap, final_iou = evaluate(
        lwdetr,
        batch,
        targets,
        criterion,
    )
    heatmap_reduction_ratio = (initial_heatmap - final_heatmap) / max(initial_heatmap, 1e-8)
    total_reduction_ratio = (initial_total - final_total) / max(initial_total, 1e-8)
    initial_matched = initial_iou >= 0.5
    final_matched = final_iou >= 0.5
    tracknet_parameter_delta = _parameter_delta(tracknet_parameters, tracknet_before)
    detector_parameter_delta = _parameter_delta(detector_parameters, detector_before)
    if tracknet_parameter_delta <= 0 or detector_parameter_delta <= 0:
        raise RuntimeError("TrackNet and detector parameters must both change")

    acceptance_checked = args.steps >= 25 and not args.matrix
    if acceptance_checked:
        if heatmap_reduction_ratio < 0.20:
            raise RuntimeError(
                f"Heatmap loss reduction {heatmap_reduction_ratio:.1%} is below 20%"
            )
        if total_reduction_ratio < 0.10:
            raise RuntimeError(
                f"Total loss reduction {total_reduction_ratio:.1%} is below 10%"
            )
        if final_iou - initial_iou < 0.05 and not (not initial_matched and final_matched):
            raise RuntimeError(
                "Best-box IoU neither improved by 0.05 nor crossed the 0.5 match "
                f"threshold: {initial_iou:.6f} -> {final_iou:.6f}"
            )

    architecture = trainer.build_pitchobjectlab_architecture(config, rf_model.model_config)
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

    checkpoint_reload = False
    if args.reload:
        fresh_model = model_cls(**trainer.build_model_kwargs(config))
        attach_motion_module(fresh_model.model, motion_config)
        fresh_lwdetr = _find_lwdetr(fresh_model.model)
        if fresh_lwdetr is None:
            raise RuntimeError("Fresh reload model has no LWDETR instance")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        fresh_lwdetr.load_state_dict(saved["model"], strict=True)
        assert_motion_checkpoint_compatible(
            fresh_model.model,
            checkpoint_path,
            architecture,
        )
        if p2_enabled:
            assert_p2_checkpoint_compatible(
                fresh_model.model,
                checkpoint_path,
                architecture,
            )
        checkpoint_reload = True
        del fresh_lwdetr, fresh_model, saved

    sync(device)
    peak_vram_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    result: dict[str, Any] = {
        "success": True,
        "model_size": model_size,
        "p2": p2_enabled,
        "focus": args.focus,
        "steps": args.steps,
        "resolution": int(rf_model.model_config.resolution),
        "batch_shape": list(batch.frames.shape),
        "initial_total_loss": initial_total,
        "final_total_loss": final_total,
        "total_loss_reduction_ratio": total_reduction_ratio,
        "initial_heatmap_loss": initial_heatmap,
        "final_heatmap_loss": final_heatmap,
        "heatmap_loss_reduction_ratio": heatmap_reduction_ratio,
        "initial_best_box_iou": initial_iou,
        "final_best_box_iou": final_iou,
        "tracknet_gradient_norm": history[-1]["tracknet_gradient_norm"],
        "detector_gradient_norm": history[-1]["detector_gradient_norm"],
        "tracknet_parameter_delta": tracknet_parameter_delta,
        "detector_parameter_delta": detector_parameter_delta,
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_gib": peak_vram_bytes / (1024**3),
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload": checkpoint_reload,
        "architecture_fingerprint": architecture["architecture_fingerprint"],
        "dataset_manifest_sha256": architecture.get("dataset_manifest_sha256"),
        "acceptance_checked": acceptance_checked,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    (output_dir / "micro_smoke_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    del criterion, optimizer, lwdetr, rf_model, datamodule, batch, targets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_worker(args: argparse.Namespace) -> int:
    started = time.monotonic()
    deadline = started + args.max_minutes * 60.0
    cases = (
        [(size, p2) for size in ("small", "medium", "large") for p2 in (False, True)]
        if args.matrix
        else [(args.model_size, args.p2 == "on")]
    )
    results = [
        run_case(
            args,
            model_size=model_size,
            p2_enabled=p2_enabled,
            deadline=deadline,
        )
        for model_size, p2_enabled in cases
    ]
    if args.matrix:
        matrix_path = Path(args.output_dir).expanduser().resolve() / "matrix_result.json"
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_text(
            json.dumps(
                {
                    "success": all(result["success"] for result in results),
                    "elapsed_seconds": time.monotonic() - started,
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not args.worker:
        return supervise_worker(args)
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
