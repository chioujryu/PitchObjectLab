"""Training and evaluation glue for real-temporal RF-DETR + TrackNetV5.

This module is imported only for ``model.motion.enabled=true`` with
``temporal.mode=real``.  Keeping the integration here is intentional: the
stock RF-DETR model, datamodule, criterion, and state dict are untouched when
the optional branch is disabled.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from rf_detr_motion import (
    MotionModule,
    _find_lwdetr,
    attach_motion_module,
    extract_heatmap_peaks,
    load_motion_checkpoint_weights,
    run_temporal_lwdetr,
    weighted_heatmap_bce,
)
from rf_detr_temporal_data import TemporalBatch, TemporalRFDETRDataModule


HEATMAP_LOSS_KEY = "loss_tracknet_heatmap"


def _motion_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("model config must be a mapping")
    motion = model.get("motion", {})
    if not isinstance(motion, Mapping):
        raise ValueError("model.motion must be a mapping")
    return motion


def temporal_data_yaml(config: Mapping[str, Any]) -> Path:
    """Resolve the canonical dataset descriptor used by the temporal loader."""

    dataset = config.get("dataset", {})
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset config must be a mapping")
    raw = str(dataset.get("data_yaml") or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
    else:
        dataset_dir = str(dataset.get("dataset_dir") or "").strip()
        if not dataset_dir:
            train = config.get("train", {})
            if isinstance(train, Mapping):
                dataset_dir = str(train.get("dataset_dir") or "").strip()
        if not dataset_dir:
            raise ValueError("Temporal TrackNet requires dataset.data_yaml or dataset.dataset_dir")
        candidate = Path(dataset_dir).expanduser() / "dataset.yaml"
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Temporal dataset descriptor does not exist: {candidate}")
    return candidate


class TemporalCriterion(nn.Module):
    """Add TrackNet heatmap supervision without changing RF-DETR matching."""

    def __init__(self, base: nn.Module, *, heatmap_weight: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        if heatmap_weight < 0:
            raise ValueError("heatmap_weight must be non-negative")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.base = base
        self.heatmap_weight = float(heatmap_weight)
        self.gamma = float(gamma)
        self.weight_dict = dict(getattr(base, "weight_dict", {}))
        self.weight_dict[HEATMAP_LOSS_KEY] = self.heatmap_weight
        self.supports_loss_normalizer_override = bool(getattr(base, "supports_loss_normalizer_override", False))

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
        **kwargs: Any,
    ) -> MutableMapping[str, torch.Tensor]:
        loss_dict = dict(self.base(outputs, targets, **kwargs))
        logits = outputs.get("pred_heatmap_logits")
        if logits is None:
            raise RuntimeError(
                "Temporal RF-DETR output is missing pred_heatmap_logits; "
                "the instance-bound TrackNet adapter was not used."
            )
        heatmaps = []
        for target in targets:
            value = target.get("temporal_heatmaps")
            if value is None:
                raise RuntimeError("Temporal target is missing temporal_heatmaps")
            heatmaps.append(value)
        expected = torch.stack(heatmaps).to(device=logits.device, dtype=logits.dtype)
        if expected.shape != logits.shape:
            if expected.ndim != 4 or expected.shape[:2] != logits.shape[:2]:
                raise RuntimeError(
                    f"Heatmap target/logit shape mismatch: {tuple(expected.shape)} vs {tuple(logits.shape)}"
                )
            expected = F.interpolate(expected, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        loss_dict[HEATMAP_LOSS_KEY] = weighted_heatmap_bce(
            logits,
            expected,
            gamma=self.gamma,
        )
        return loss_dict


def checkpoint_contains_motion(checkpoint_path: Any) -> bool:
    """Return whether a local RF-DETR checkpoint contains TrackNet tensors."""

    if checkpoint_path is None or not str(checkpoint_path).strip():
        return False
    path = Path(str(checkpoint_path)).expanduser()
    if not path.is_file():
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        return False
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        return False
    return any("motion_module." in str(key) for key in state)


def maybe_load_motion_checkpoint(model: nn.Module, checkpoint_path: Any, *, required: bool) -> bool:
    """Restore TrackNet tensors after the module has been attached."""

    has_motion = checkpoint_contains_motion(checkpoint_path)
    if not has_motion:
        if required:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path!r} contains no motion_module.* tensors; "
                "a stock or legacy TrackNet prototype checkpoint cannot run real temporal TrackNetV5."
            )
        return False
    load_motion_checkpoint_weights(model, checkpoint_path)
    return True


def build_temporal_model_module(model_config: Any, train_config: Any, config: Mapping[str, Any]) -> Any:
    """Build the upstream Lightning module, then attach instance-local TrackNet."""

    from rfdetr.training import RFDETRModelModule
    from rfdetr.training.module_model import compute_multi_scale_scales

    motion = dict(_motion_config(config))

    class _TemporalRFDETRModelModule(RFDETRModelModule):
        def on_train_batch_start(self, batch: tuple[Any, Any], batch_idx: int) -> None:
            samples, _ = batch
            if isinstance(samples, TemporalBatch):
                return
            super().on_train_batch_start(batch, batch_idx)

        def training_step(
            self,
            batch: tuple[Any, Any],
            batch_idx: int,
        ) -> torch.Tensor | dict[str, Any]:
            samples, targets = batch
            tc = self.train_config
            mc = self.model_config
            if isinstance(samples, TemporalBatch) and tc.multi_scale and not tc.do_random_resize_via_padding:
                scales = compute_multi_scale_scales(
                    mc.resolution,
                    tc.expanded_scales,
                    mc.patch_size,
                    mc.num_windows,
                )
                random.seed(self.trainer.global_step)
                scale = random.choice(scales)
                batch_size, num_frames, channels, height, width = samples.frames.shape
                with torch.no_grad():
                    resized_frames = F.interpolate(
                        samples.frames.reshape(
                            batch_size * num_frames,
                            channels,
                            height,
                            width,
                        ),
                        size=scale,
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(batch_size, num_frames, channels, scale, scale)
                    resized_masks = (
                        F.interpolate(
                            samples.padding_masks.reshape(
                                batch_size * num_frames,
                                1,
                                height,
                                width,
                            ).float(),
                            size=scale,
                            mode="nearest",
                        )
                        .reshape(batch_size, num_frames, scale, scale)
                        .bool()
                    )
                samples = replace(
                    samples,
                    frames=resized_frames,
                    padding_masks=resized_masks,
                )
                batch = (samples, targets)
            return super().training_step(batch, batch_idx)

    module = _TemporalRFDETRModelModule(model_config, train_config)
    attach_motion_module(module.model, motion)
    checkpoint = getattr(model_config, "pretrain_weights", None)
    maybe_load_motion_checkpoint(module.model, checkpoint, required=False)
    loss = motion.get("loss", {}) or {}
    module.criterion = TemporalCriterion(
        module.criterion,
        heatmap_weight=float(loss.get("heatmap_weight", 1.0)),
        gamma=float(loss.get("gamma", 2.0)),
    )
    return module


def build_temporal_datamodule(
    config: Mapping[str, Any],
    model_config: Any,
    train_config: Any,
) -> TemporalRFDETRDataModule:
    """Construct the project-local complete-window DataModule."""

    motion = _motion_config(config)
    temporal = motion.get("temporal", {}) or {}
    focus = motion.get("focus", {}) or {}
    tracknet = motion.get("tracknet_v5", {}) or {}
    heatmap = tracknet.get("heatmap", {}) or {}
    dataset = config.get("dataset", {})
    augment = dataset.get("temporal_augmentation", {}) if isinstance(dataset, Mapping) else {}
    if not isinstance(augment, Mapping):
        augment = {}
    temporal_dataset = dataset.get("temporal", {}) if isinstance(dataset, Mapping) else {}
    if not isinstance(temporal_dataset, Mapping):
        temporal_dataset = {}
    max_windows = temporal_dataset.get("max_windows_per_split", {})
    if isinstance(max_windows, int):
        max_windows = {"train": max_windows, "val": max_windows, "test": max_windows}
    if not isinstance(max_windows, Mapping):
        raise ValueError("dataset.temporal.max_windows_per_split must be an integer or mapping")
    block_size = int(getattr(model_config, "patch_size")) * int(getattr(model_config, "num_windows"))
    pin_memory = getattr(train_config, "pin_memory", None)
    return TemporalRFDETRDataModule(
        temporal_data_yaml(config),
        image_size=int(getattr(model_config, "resolution")),
        batch_size=int(getattr(train_config, "batch_size")),
        num_workers=int(getattr(train_config, "num_workers", 2)),
        num_frames=int(temporal.get("num_frames", 3)),
        frame_stride=int(temporal.get("frame_stride", 1)),
        focus_mode=str(focus.get("mode", "all")),
        primary_field=str(focus.get("primary_field", "primary_label_index")),
        min_sigma=float(heatmap.get("min_sigma", 1.0)),
        block_size=block_size,
        train_horizontal_flip=bool(augment.get("horizontal_flip", False)),
        max_windows_per_split={str(key): int(value) for key, value in max_windows.items()},
        pin_memory=True if pin_memory is None else bool(pin_memory),
        persistent_workers=getattr(train_config, "persistent_workers", None),
    )


def _tensor_rows(value: torch.Tensor | None) -> list[Any]:
    if value is None:
        return []
    return value.detach().cpu().tolist()


def _save_heatmap(path: Path, heatmap: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = heatmap.detach().float().clamp(0, 1).mul(255).byte().cpu().numpy()
    Image.fromarray(array, mode="L").save(path)


def _postprocess_context(rf_model: Any) -> Any:
    context = getattr(rf_model, "model", None)
    callback = getattr(context, "postprocess", None)
    if not callable(callback):
        raise RuntimeError("RF-DETR model context does not expose postprocess")
    return callback


def _window_target_centres(target: Mapping[str, torch.Tensor], height: int, width: int) -> torch.Tensor:
    boxes = target["boxes"]
    indices = target.get("tracknet_box_indices")
    if indices is not None:
        boxes = boxes[indices]
    if boxes.numel() == 0:
        return boxes.new_empty((0, 2))
    return boxes[:, :2] * boxes.new_tensor([width, height])


def run_temporal_split(
    *,
    rf_model: Any,
    config: Mapping[str, Any],
    output_dir: str | Path,
    split: str = "test",
    save_heatmaps: bool = True,
) -> dict[str, Any]:
    """Run every complete temporal window and write smoke diagnostics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_config = getattr(rf_model, "model_config")
    test_settings = config.get("test", {}) or {}
    inference_settings = config.get("inference", {}) or {}
    raw_batch_size = test_settings.get("batch_size", inference_settings.get("batch_size", 1))
    raw_num_workers = test_settings.get("num_workers", inference_settings.get("num_workers", 2))
    train_stub = type(
        "TemporalEvalConfig",
        (),
        {
            "batch_size": int(1 if raw_batch_size is None else raw_batch_size),
            # Zero is intentional for micro-smoke: avoid Windows worker spawn/import cost.
            "num_workers": int(2 if raw_num_workers is None else raw_num_workers),
            "pin_memory": True,
            "persistent_workers": None,
        },
    )()
    datamodule = build_temporal_datamodule(config, model_config, train_stub)
    datamodule.setup("test")
    loader = (
        datamodule.test_dataloader()
        if str(split).lower() == "test"
        else datamodule._loader(str(split).lower(), shuffle=False)
    )

    context = getattr(rf_model, "model", None)
    lwdetr = _find_lwdetr(context)
    if lwdetr is None:
        raise RuntimeError("Could not locate attached LWDETR for temporal inference")
    motion_module = getattr(lwdetr, "motion_module", None)
    if not isinstance(motion_module, MotionModule):
        raise RuntimeError("Temporal inference requires an attached MotionModule")
    device = next(lwdetr.parameters()).device
    lwdetr.eval()
    postprocess = _postprocess_context(rf_model)

    rows: list[dict[str, Any]] = []
    centre_errors: list[float] = []
    empty_frames = 0
    empty_false_positive_frames = 0
    forward_seconds = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (batch, targets) in enumerate(loader):
            if not isinstance(batch, TemporalBatch):
                raise TypeError(f"Temporal loader returned {type(batch).__name__}, expected TemporalBatch")
            batch = batch.to(device, non_blocking=True)
            moved_targets = [
                {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
                for target in targets
            ]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            outputs = run_temporal_lwdetr(lwdetr, batch, targets=None)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - forward_started
            orig_sizes = torch.stack([target["orig_size"] for target in moved_targets])
            try:
                detections = postprocess(outputs, target_sizes=orig_sizes)
            except TypeError:
                detections = postprocess(outputs, orig_sizes)
            heatmaps = outputs["pred_heatmaps"]
            peaks = extract_heatmap_peaks(
                heatmaps,
                focus_mode=motion_module.focus_mode,
                threshold=motion_module.peak_threshold,
                nms_kernel=motion_module.peak_nms_kernel,
                max_peaks=motion_module.max_peaks,
            )
            for sample_index, (metadata, detection, sample_peaks) in enumerate(zip(batch.metadata, detections, peaks)):
                global_index = len(rows)
                frame_targets = batch.frame_targets[sample_index]
                frame_diagnostics = []
                for frame_offset, (frame_target, frame_peaks) in enumerate(zip(frame_targets, sample_peaks)):
                    height, width = heatmaps.shape[-2:]
                    centres = _window_target_centres(frame_target, height, width)
                    if centres.numel() == 0:
                        empty_frames += 1
                        if frame_peaks.numel() > 0:
                            empty_false_positive_frames += 1
                    elif frame_peaks.numel() > 0:
                        distances = torch.cdist(centres, frame_peaks[:, :2])
                        centre_errors.extend(distances.min(dim=1).values.detach().cpu().tolist())
                    heatmap_file = None
                    if save_heatmaps:
                        heatmap_file = output_path / "heatmaps" / f"window_{global_index:05d}_t{frame_offset}.png"
                        _save_heatmap(heatmap_file, heatmaps[sample_index, frame_offset])
                    frame_diagnostics.append(
                        {
                            "frame_index": int(metadata["frame_indices"][frame_offset]),
                            "peaks": _tensor_rows(frame_peaks),
                            "target_centres": _tensor_rows(centres),
                            "heatmap": str(heatmap_file) if heatmap_file else None,
                        }
                    )
                rows.append(
                    {
                        "window_index": global_index,
                        "split": metadata["split"],
                        "sequence_id": metadata["sequence_id"],
                        "anchor_frame_index": int(metadata["anchor_frame_index"]),
                        "boundary_padding": bool(metadata.get("boundary_padding", False)),
                        "detections": {
                            "boxes": _tensor_rows(detection.get("boxes")),
                            "scores": _tensor_rows(detection.get("scores")),
                            "labels": _tensor_rows(detection.get("labels")),
                        },
                        "tracknet": frame_diagnostics,
                    }
                )

    elapsed = time.perf_counter() - started
    summary = {
        "split": str(split),
        "windows": len(rows),
        "detections": sum(len(row["detections"]["boxes"]) for row in rows),
        "focus_mode": motion_module.focus_mode,
        "heatmap_shape": [
            len(rows),
            motion_module.num_frames,
            int(model_config.resolution),
            int(model_config.resolution),
        ],
        "centre_error_mean_pixels": (sum(centre_errors) / len(centre_errors)) if centre_errors else None,
        "centre_error_count": len(centre_errors),
        "empty_frames": empty_frames,
        "empty_false_positive_frames": empty_false_positive_frames,
        "total_seconds": elapsed,
        "model_forward_seconds": forward_seconds,
        "seconds_per_window": elapsed / len(rows) if rows else None,
        "finite_outputs": all(math.isfinite(float(value)) for row in rows for value in row["detections"]["scores"]),
    }
    predictions_path = output_path / "temporal_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_path / "temporal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"summary": summary, "rows": rows, "output_dir": str(output_path)}


__all__ = [
    "HEATMAP_LOSS_KEY",
    "TemporalCriterion",
    "build_temporal_datamodule",
    "build_temporal_model_module",
    "checkpoint_contains_motion",
    "maybe_load_motion_checkpoint",
    "run_temporal_split",
    "temporal_data_yaml",
]
