"""Shared RF-DETR inference acceleration primitives.

The module deliberately imports TensorRT and ONNX only when the TensorRT
backend is selected.  PyTorch-only installations can therefore import and use
the normal inference/test entrypoints without either optional dependency.

TensorRT engines are GPU-, TensorRT-, model-, precision-, and profile-specific.
Engines created here always carry a sidecar manifest and are rejected when the
current runtime identity differs.  There is no implicit backend fallback.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch


_MANIFEST_SCHEMA_VERSION = 2
_ONNX_CACHE_SCHEMA_VERSION = 1
_TENSORRT_EXPORT_ABI_VERSION = 4
_TENSORRT_EXPORT_SHAPE_CONTRACT = "dynamic-batch-static-nchw"
_ACCELERATION_MARKER = "_pitch_object_lab_inference_acceleration"
_FORWARD_RECORDER_MARKER = "_pitch_object_lab_forward_timing_recorder"
_POSTPROCESS_RECORDER_MARKER = "_pitch_object_lab_postprocess_timing_recorder"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TENSORRT_CACHE_DIR = PROJECT_DIR / "runs" / "rf_detr" / "tensorrt_cache"
_OUTPUT_NAME_MAP = {
    "dets": "pred_boxes",
    "labels": "pred_logits",
    "masks": "pred_masks",
    "pred_boxes": "pred_boxes",
    "pred_logits": "pred_logits",
    "pred_masks": "pred_masks",
}


@dataclass(frozen=True)
class TensorRTProfile:
    """Resolved TensorRT dynamic-batch optimization profile."""

    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1


@dataclass(frozen=True)
class TensorRTSettings:
    """Validated TensorRT settings."""

    precision: str = "fp16"
    engine_path: Path | None = None
    manifest_path: Path | None = None
    cache_dir: Path | None = None
    workspace_gib: float = 4.0
    force_rebuild: bool = False
    reuse_output_buffers: bool = False
    profile: TensorRTProfile = field(default_factory=TensorRTProfile)


def tensorrt_optimization_profiles(profile: TensorRTProfile) -> tuple[TensorRTProfile, ...]:
    """Expand one dynamic range into common batch-tuned TensorRT profiles.

    Every profile preserves the configured min/max range so real tail batches
    remain valid. Profiles are tuned for batch 1/4/8/16 plus an explicitly
    configured optimization batch when it is different.
    """

    candidates = {
        profile.opt_batch_size,
        *(batch for batch in (1, 4, 8, 16) if profile.min_batch_size <= batch <= profile.max_batch_size),
    }
    return tuple(
        TensorRTProfile(profile.min_batch_size, batch, profile.max_batch_size)
        for batch in sorted(candidates)
    )


@dataclass(frozen=True)
class InferenceOptimizationConfig:
    """Validated acceleration configuration shared by inference and test."""

    backend: str = "pytorch"
    pytorch_precision: str = "fp32"
    tensorrt: TensorRTSettings = field(default_factory=TensorRTSettings)
    resolution: int | None = None

    @property
    def precision(self) -> str:
        return self.pytorch_precision if self.backend == "pytorch" else self.tensorrt.precision


@dataclass(frozen=True)
class TensorRTArtifact:
    """A validated TensorRT engine and its provenance manifest."""

    engine_path: Path
    manifest_path: Path
    onnx_path: Path | None
    cache_hit: bool
    manifest: Mapping[str, Any]
    timing_cache_path: Path | None = None
    export_seconds: float = 0.0
    build_seconds: float = 0.0


class PreprocessTiming:
    """Deferred, idempotent host/H2D/device preprocessing telemetry."""

    def __init__(
        self,
        *,
        host_seconds: float,
        prepare_wall_seconds: float,
        h2d: ForwardTimingRecorder | None = None,
        resize_normalize: ForwardTimingRecorder | None = None,
    ) -> None:
        self.host_seconds = max(0.0, float(host_seconds))
        self.prepare_wall_seconds = max(0.0, float(prepare_wall_seconds))
        self.h2d = h2d
        self.resize_normalize = resize_normalize
        self._lock = threading.Lock()
        self._consumed: dict[str, float] | None = None

    def consume(self) -> dict[str, float]:
        with self._lock:
            if self._consumed is not None:
                return dict(self._consumed)
            h2d_seconds = self.h2d.consume_seconds() if self.h2d is not None else 0.0
            resize_seconds = (
                self.resize_normalize.consume_seconds()
                if self.resize_normalize is not None
                else 0.0
            )
            device_seconds = h2d_seconds + resize_seconds
            self._consumed = {
                "host_preprocess_seconds": self.host_seconds,
                "h2d_seconds": h2d_seconds,
                "resize_normalize_seconds": resize_seconds,
                "device_preprocess_seconds": device_seconds,
                "prepare_wall_seconds": self.prepare_wall_seconds,
                "total_seconds": self.host_seconds + device_seconds,
            }
            return dict(self._consumed)


@dataclass(frozen=True)
class PreparedInferenceBatch:
    """A normalized NCHW batch and its original ``(height, width)`` sizes."""

    tensor: torch.Tensor
    target_sizes: torch.Tensor
    _timing: PreprocessTiming = field(repr=False, compare=False)

    def consume_timing(self) -> dict[str, float]:
        """Resolve only this batch's deferred CUDA events and return stage timing."""

        return self._timing.consume()


@dataclass
class AccelerationHandle:
    """Stable interface consumed by inference and standalone segmentation test."""

    model: Any
    settings: InferenceOptimizationConfig
    metadata: dict[str, Any]
    artifact: TensorRTArtifact | None = None
    _infer_raw: Callable[[torch.Tensor], Mapping[str, torch.Tensor]] | None = None
    _forward_recorder: ForwardTimingRecorder | None = None
    _postprocess_recorder: ForwardTimingRecorder | None = None
    _infer_into: Callable[[torch.Tensor, Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]] | None = None
    _allocate_output_buffers: Callable[[int], dict[str, torch.Tensor]] | None = None
    _reuse_output_buffers_by_default: bool = False
    _reusable_output_buffers: dict[tuple[int, int], dict[str, torch.Tensor]] = field(
        default_factory=dict,
        repr=False,
    )
    _reusable_output_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _preprocess_timings: list[PreprocessTiming] = field(default_factory=list, repr=False)
    _preprocess_timing_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def backend(self) -> str:
        """Effective backend name (exactly ``pytorch`` or ``tensorrt``)."""

        return self.settings.backend

    @property
    def precision(self) -> str:
        return self.settings.precision

    @property
    def device(self) -> torch.device:
        value = getattr(self.model, "device", None)
        if value is None:
            value = getattr(getattr(self.model, "model", None), "device", None)
        if value is None:
            raise RuntimeError("Accelerated RF-DETR model does not expose its device.")
        return value if isinstance(value, torch.device) else torch.device(value)

    def infer_raw(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run raw model inference and return RF-DETR postprocessor keys."""

        if self._reuse_output_buffers_by_default:
            return self.infer_raw_reusing_buffers(tensor)
        if self._infer_raw is None:
            raise RuntimeError("This acceleration handle does not expose raw inference.")
        return normalize_raw_outputs(self._infer_raw(tensor))

    def infer_raw_reusing_buffers(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Infer into a per-thread/batch output pool for sequential postprocessing.

        Returned tensors are overwritten by the next call with the same batch
        size on the same thread. Callers must finish postprocessing before that
        next call; use :meth:`infer_raw` with reuse disabled when retaining raw
        outputs across calls.
        """

        if self._infer_into is None or self._allocate_output_buffers is None:
            if self._infer_raw is None:
                raise RuntimeError("This acceleration handle does not expose raw inference.")
            return normalize_raw_outputs(self._infer_raw(tensor))
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[0] <= 0:
            raise ValueError("Reusable RF-DETR inference requires a non-empty rank-4 tensor.")
        key = (threading.get_ident(), int(tensor.shape[0]))
        with self._reusable_output_lock:
            buffers = self._reusable_output_buffers.get(key)
            if buffers is None:
                buffers = self._allocate_output_buffers(key[1])
                self._reusable_output_buffers[key] = buffers
        return normalize_raw_outputs(self._infer_into(tensor, buffers))

    def clear_reusable_output_buffers(self) -> None:
        """Release handle-owned output pools after pending inference is consumed."""

        with self._reusable_output_lock:
            self._reusable_output_buffers.clear()

    def prepare_batch(
        self,
        images: Any,
        *,
        shape: int | tuple[int, int] | None = None,
        non_blocking: bool = True,
    ) -> PreparedInferenceBatch:
        """Prepare images once as a contiguous batch for :meth:`infer_raw`.

        Unlike upstream ``predict()``, this path avoids a source-image copy,
        per-image CUDA transfers, and full-image min/max reductions.  Integer
        images are scaled from bytes while floating-point tensor inputs retain
        RF-DETR's documented ``[0, 1]`` contract.
        """

        resolution = (
            shape
            if shape is not None
            else (self.settings.resolution or _model_resolution(self.model, self.settings))
        )
        mean = getattr(self.model, "means", (0.485, 0.456, 0.406))
        std = getattr(self.model, "stds", (0.229, 0.224, 0.225))
        prepared = prepare_inference_batch(
            images,
            shape=resolution,
            device=self.device,
            mean=mean,
            std=std,
            num_channels=_model_num_channels(self.model),
            non_blocking=non_blocking,
        )
        with self._preprocess_timing_lock:
            self._preprocess_timings.append(prepared._timing)
        return prepared

    def consume_preprocess_timing(self) -> dict[str, float | int]:
        """Consume and aggregate preprocessing telemetry since the previous call."""

        with self._preprocess_timing_lock:
            timings = self._preprocess_timings
            self._preprocess_timings = []
        keys = (
            "host_preprocess_seconds",
            "h2d_seconds",
            "resize_normalize_seconds",
            "device_preprocess_seconds",
            "prepare_wall_seconds",
            "total_seconds",
        )
        result: dict[str, float | int] = {key: 0.0 for key in keys}
        for timing in timings:
            report = timing.consume()
            for key in keys:
                result[key] = float(result[key]) + float(report[key])
        result["batches"] = len(timings)
        return result

    def postprocess(self, outputs: Mapping[str, torch.Tensor], target_sizes: torch.Tensor) -> Any:
        """Use the original RF-DETR postprocessor for bbox/mask decoding."""

        callback = getattr(self.model, "postprocess", None)
        if callable(callback):
            return callback(outputs, target_sizes)
        model_context = getattr(self.model, "model", None)
        callback = getattr(model_context, "postprocess", None)
        if not callable(callback):
            raise RuntimeError("Accelerated RF-DETR model does not expose postprocess().")
        try:
            return callback(outputs, target_sizes=target_sizes)
        except TypeError:
            return callback(outputs, target_sizes)

    def consume_forward_seconds(self) -> float:
        """Return and reset true model-forward time accumulated since the last call."""

        if self._forward_recorder is None:
            return 0.0
        return self._forward_recorder.consume_seconds()

    def consume_postprocess_seconds(self) -> float:
        """Return and reset RF-DETR postprocessor time accumulated since the last call."""

        if self._postprocess_recorder is None:
            return 0.0
        return self._postprocess_recorder.consume_seconds()


class ForwardTimingRecorder:
    """Accumulate CPU wall time or CUDA-event time without synchronizing each forward."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wall_seconds = 0.0
        self._cuda_events: list[tuple[Any, Any, Callable[[], None] | None]] = []

    def add_wall_seconds(self, seconds: float) -> None:
        with self._lock:
            self._wall_seconds += max(0.0, float(seconds))

    def add_cuda_events(
        self,
        start: Any,
        end: Any,
        *,
        release: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            self._cuda_events.append((start, end, release))

    def consume_seconds(self) -> float:
        with self._lock:
            wall_seconds = self._wall_seconds
            events = self._cuda_events
            self._wall_seconds = 0.0
            self._cuda_events = []
        cuda_seconds = 0.0
        for start, end, release in events:
            try:
                end.synchronize()
                cuda_seconds += float(start.elapsed_time(end)) / 1000.0
            finally:
                if release is not None:
                    release()
        return wall_seconds + cuda_seconds


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    tensors = getattr(value, "tensors", None)
    return tensors if isinstance(tensors, torch.Tensor) else None


class _TimedCallable:
    """Fallback timing proxy for callables that do not support module hooks."""

    def __init__(self, callback: Any, recorder: ForwardTimingRecorder) -> None:
        self._callback = callback
        self._recorder = recorder

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tensor = _first_tensor((args, kwargs))
        if tensor is not None and tensor.device.type == "cuda":
            with torch.cuda.device(tensor.device):
                stream = torch.cuda.current_stream(tensor.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                result = self._callback(*args, **kwargs)
                end.record(stream)
            self._recorder.add_cuda_events(start, end)
            return result
        started = time.perf_counter()
        result = self._callback(*args, **kwargs)
        self._recorder.add_wall_seconds(time.perf_counter() - started)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback, name)


def _install_forward_timing(callback: Any, recorder: ForwardTimingRecorder) -> Any:
    register_pre = getattr(callback, "register_forward_pre_hook", None)
    register_post = getattr(callback, "register_forward_hook", None)
    if not callable(register_pre) or not callable(register_post):
        return _TimedCallable(callback, recorder)
    local = threading.local()

    def pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
        stack = getattr(local, "stack", None)
        if stack is None:
            stack = []
            local.stack = stack
        tensor = _first_tensor(args)
        if tensor is not None and tensor.device.type == "cuda":
            with torch.cuda.device(tensor.device):
                stream = torch.cuda.current_stream(tensor.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
            stack.append(("cuda", start, end, stream))
        else:
            stack.append(("cpu", time.perf_counter()))

    def post_hook(_module: Any, _args: tuple[Any, ...], _output: Any) -> None:
        stack = getattr(local, "stack", None)
        if not stack:
            return
        timing = stack.pop()
        if timing[0] == "cuda":
            _, start, end, stream = timing
            end.record(stream)
            recorder.add_cuda_events(start, end)
        else:
            recorder.add_wall_seconds(time.perf_counter() - timing[1])

    register_pre(pre_hook)
    register_post(post_hook)
    return callback


def _ensure_postprocess_timing(model: Any) -> ForwardTimingRecorder | None:
    instance_state = getattr(model, "__dict__", None)
    recorder = instance_state.get(_POSTPROCESS_RECORDER_MARKER) if isinstance(instance_state, dict) else None
    if isinstance(recorder, ForwardTimingRecorder):
        return recorder
    context = getattr(model, "model", None)
    callback = getattr(context, "postprocess", None)
    if not callable(callback):
        return None
    recorder = ForwardTimingRecorder()
    context.postprocess = _install_forward_timing(callback, recorder)
    setattr(model, _POSTPROCESS_RECORDER_MARKER, recorder)
    return recorder


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _choice(value: Any, name: str, choices: set[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        valid = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {valid}; got {value!r}.")
    return normalized


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.") from None
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return result


def _optional_path(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(str(value)).expanduser()


def resolve_tensorrt_cache_dir(value: Any = None) -> Path:
    """Resolve an optional TensorRT cache path from the trainer project root."""
    text = str(value or "").strip()
    if not text:
        return DEFAULT_TENSORRT_CACHE_DIR.resolve()
    path = Path(text).expanduser()
    if PureWindowsPath(text).is_absolute() or PurePosixPath(text).is_absolute():
        return path
    windows_path = PureWindowsPath(text)
    if windows_path.drive or (windows_path.root and not PurePosixPath(text).is_absolute()):
        raise ValueError(
            "TensorRT cache paths must be project-relative, fully absolute, or start with '../'; "
            f"ambiguous Windows path is not supported: {text}"
        )
    return (PROJECT_DIR / path).resolve()


def _optimization_source(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return ``(optimization block, model block)`` for root/model/direct inputs."""

    model_source = config.get("model")
    if isinstance(model_source, Mapping):
        optimization = model_source.get("inference_optimization", {})
        return _mapping(optimization, "model.inference_optimization"), model_source
    nested = config.get("inference_optimization")
    if nested is not None:
        return _mapping(nested, "inference_optimization"), config
    return config, config


def resolve_acceleration_config(
    config: Mapping[str, Any],
    *,
    batch_sizes: Iterable[int] = (),
    resolution: int | None = None,
) -> InferenceOptimizationConfig:
    """Validate and resolve the shared RF-DETR acceleration configuration.

    ``config`` may be the complete execution configuration, the ``model``
    mapping, or the ``model.inference_optimization`` mapping itself.
    """

    if not isinstance(config, Mapping):
        raise TypeError("Acceleration config must be a mapping.")
    source, model_source = _optimization_source(config)
    backend = _choice(source.get("backend", "pytorch"), "inference_optimization.backend", {"pytorch", "tensorrt"})

    pytorch_source = _mapping(source.get("pytorch", {}), "inference_optimization.pytorch")
    pytorch_precision = _choice(
        pytorch_source.get("precision", "fp32"),
        "inference_optimization.pytorch.precision",
        {"fp32", "bf16"},
    )

    trt_source = _mapping(source.get("tensorrt", {}), "inference_optimization.tensorrt")
    trt_precision = _choice(
        trt_source.get("precision", "fp16"),
        "inference_optimization.tensorrt.precision",
        {"fp16", "bf16"},
    )
    engine_path = _optional_path(trt_source.get("engine_path"))
    manifest_path = _optional_path(trt_source.get("manifest_path"))
    if engine_path is not None and manifest_path is None:
        # The CLI intentionally needs only --tensorrt-engine.  Project-created
        # external engines use this deterministic adjacent sidecar name.
        manifest_path = engine_path.with_suffix(engine_path.suffix + ".manifest.json")
    if engine_path is None and manifest_path is not None:
        raise ValueError("TensorRT manifest_path cannot be provided without engine_path.")
    force_rebuild = bool(trt_source.get("force_rebuild", False))
    if engine_path is not None and force_rebuild:
        raise ValueError("TensorRT force_rebuild cannot be used with an explicit engine_path.")
    reuse_output_buffers = trt_source.get("reuse_output_buffers", False)
    if not isinstance(reuse_output_buffers, bool):
        raise ValueError("TensorRT reuse_output_buffers must be true or false.")

    workspace = trt_source.get("workspace_gib", 4)
    if isinstance(workspace, bool):
        raise ValueError("TensorRT workspace_gib must be a finite positive number.")
    try:
        workspace_gib = float(workspace)
    except (TypeError, ValueError):
        raise ValueError("TensorRT workspace_gib must be a finite positive number.") from None
    if not math.isfinite(workspace_gib) or workspace_gib <= 0:
        raise ValueError("TensorRT workspace_gib must be a finite positive number.")

    discovered_batches: list[int] = []
    for index, value in enumerate(batch_sizes):
        discovered_batches.append(_positive_int(value, f"batch_sizes[{index}]"))
    # Callers provide the primary workload first, followed by any additional
    # enabled paths (for example video).  Tune tactics for the primary batch
    # while keeping the profile wide enough for every path.
    automatic_opt_batch = discovered_batches[0] if discovered_batches else 1
    automatic_max_batch = max(discovered_batches, default=1)
    profile_source = _mapping(trt_source.get("profile", {}), "inference_optimization.tensorrt.profile")

    def profile_value(key: str, default: int) -> int:
        raw = profile_source.get(key, default)
        if raw is None or (isinstance(raw, str) and raw.strip().lower() == "auto"):
            return default
        return _positive_int(raw, f"inference_optimization.tensorrt.profile.{key}")

    min_batch = profile_value("min_batch_size", 1)
    opt_batch = profile_value("opt_batch_size", automatic_opt_batch)
    max_batch = profile_value("max_batch_size", automatic_max_batch)
    if min_batch != 1:
        raise ValueError(
            "TensorRT min_batch_size must be 1 so real tail batches and the batch-1 warmup remain valid."
        )
    if not min_batch <= opt_batch <= max_batch:
        raise ValueError(
            "TensorRT profile must satisfy min_batch_size <= opt_batch_size <= max_batch_size; "
            f"got {min_batch} <= {opt_batch} <= {max_batch}."
        )

    resolved_resolution = resolution
    if resolved_resolution is None:
        resolved_resolution = model_source.get("resolution")
    if resolved_resolution is not None:
        resolved_resolution = _positive_int(resolved_resolution, "model.resolution")

    return InferenceOptimizationConfig(
        backend=backend,
        pytorch_precision=pytorch_precision,
        tensorrt=TensorRTSettings(
            precision=trt_precision,
            engine_path=engine_path,
            manifest_path=manifest_path,
            cache_dir=resolve_tensorrt_cache_dir(trt_source.get("cache_dir")),
            workspace_gib=workspace_gib,
            force_rebuild=force_rebuild,
            reuse_output_buffers=reuse_output_buffers,
            profile=TensorRTProfile(min_batch, opt_batch, max_batch),
        ),
        resolution=resolved_resolution,
    )


def _model_device(model: Any) -> torch.device:
    candidate = getattr(model, "device", None)
    if candidate is None:
        candidate = getattr(getattr(model, "model", None), "device", None)
    if candidate is None:
        raise RuntimeError("RF-DETR model does not expose model.device.")
    return candidate if isinstance(candidate, torch.device) else torch.device(candidate)


def _require_cuda(device: torch.device, *, bf16: bool = False) -> None:
    if device.type != "cuda":
        suffix = " BF16" if bf16 else ""
        raise RuntimeError(f"{suffix.strip() or 'TensorRT'} inference requires a CUDA device; got {device}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA acceleration was requested, but torch.cuda.is_available() is false.")
    with torch.cuda.device(device):
        capability = tuple(int(v) for v in torch.cuda.get_device_capability(device))
        if bf16:
            supported = capability[0] >= 8
            checker = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(checker):
                supported = supported and bool(checker())
            if not supported:
                raise RuntimeError(
                    f"BF16 inference requires an Ampere-or-newer CUDA GPU with BF16 support; "
                    f"device {device} has compute capability {capability[0]}.{capability[1]}."
                )


def normalize_raw_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    """Normalize RF-DETR dict/tuple/TensorRT output names to raw model keys."""

    if isinstance(outputs, (tuple, list)):
        if len(outputs) not in {2, 3}:
            raise ValueError(f"Expected two or three RF-DETR outputs, got {len(outputs)}.")
        result = {"pred_boxes": outputs[0], "pred_logits": outputs[1]}
        if len(outputs) == 3:
            result["pred_masks"] = outputs[2]
        return result
    if not isinstance(outputs, Mapping):
        raise TypeError(f"RF-DETR raw outputs must be a mapping or tuple, got {type(outputs).__name__}.")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in outputs.items():
        key = _OUTPUT_NAME_MAP.get(str(name))
        if key is not None:
            normalized[key] = value
    missing = {"pred_boxes", "pred_logits"} - normalized.keys()
    if missing:
        raise ValueError(f"RF-DETR raw outputs are missing: {', '.join(sorted(missing))}.")
    return normalized


def _resolved_image_shape(shape: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(shape, bool):
        raise ValueError("Inference image shape must be a positive integer or (height, width) pair.")
    if isinstance(shape, int):
        dimensions = (shape, shape)
    else:
        try:
            dimensions = tuple(shape)
        except TypeError:
            raise ValueError("Inference image shape must be a positive integer or (height, width) pair.") from None
        if len(dimensions) != 2:
            raise ValueError("Inference image shape must contain exactly height and width.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError(f"Inference image shape values must be positive integers; got {dimensions!r}.")
    return int(dimensions[0]), int(dimensions[1])


def _image_to_chw_float_tensor(image: Any, num_channels: int) -> torch.Tensor:
    """Convert one path/PIL/NumPy/CHW Tensor image without value reductions."""

    if isinstance(image, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - RF-DETR requires Pillow
            raise ImportError("Pillow is required to prepare path-based RF-DETR images.") from exc
        with Image.open(image) as opened:
            return _image_to_chw_float_tensor(opened, num_channels)

    if isinstance(image, torch.Tensor):
        if image.ndim != 3 or int(image.shape[0]) != num_channels:
            raise ValueError(
                f"Tensor images must be CHW with {num_channels} channels; got {tuple(image.shape)}."
            )
        if image.dtype == torch.uint8:
            return image.to(dtype=torch.float32).div_(255.0)
        if not image.is_floating_point():
            raise ValueError(f"Tensor images must use uint8 or floating dtype; got {image.dtype}.")
        return image.to(dtype=torch.float32)

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - RF-DETR requires NumPy
        raise ImportError("NumPy is required to prepare PIL/array RF-DETR images.") from exc
    try:
        # One owned contiguous copy is deliberate: PIL-backed arrays can be
        # read-only, and the final batch otherwise triggers another hidden copy.
        array = np.array(image, copy=True, order="C")
    except Exception as exc:
        raise TypeError(
            "RF-DETR batch images must be paths, PIL images, NumPy arrays, or CHW torch tensors."
        ) from exc
    if array.ndim == 2 and num_channels == 1:
        array = array[:, :, None]
    if array.ndim != 3 or int(array.shape[2]) != num_channels:
        raise ValueError(
            f"Array/PIL images must be HWC with {num_channels} channels; got {tuple(array.shape)}."
        )
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if tensor.dtype == torch.uint8:
        return tensor.to(dtype=torch.float32).div_(255.0)
    if not tensor.is_floating_point():
        raise ValueError(f"Array images must use uint8 or floating dtype; got {tensor.dtype}.")
    return tensor.to(dtype=torch.float32)


def _stack_uniform_array_images(
    images: list[Any],
    num_channels: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]] | None:
    """Stack the common PIL/NumPy SAHI case before converting to Torch."""

    if any(isinstance(image, (str, Path, torch.Tensor)) for image in images):
        return None
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - RF-DETR requires NumPy
        raise ImportError("NumPy is required to prepare PIL/array RF-DETR images.") from exc
    arrays = [np.asarray(image) for image in images]
    for array in arrays:
        if array.ndim == 2 and num_channels == 1:
            continue
        if array.ndim != 3 or int(array.shape[2]) != num_channels:
            raise ValueError(
                f"Array/PIL images must be HWC with {num_channels} channels; got {tuple(array.shape)}."
            )
    normalized_arrays = [array[:, :, None] if array.ndim == 2 else array for array in arrays]
    if len({(tuple(array.shape), str(array.dtype)) for array in normalized_arrays}) != 1:
        return None
    # np.stack makes one owned contiguous allocation, avoiding one NumPy copy,
    # float conversion, and Torch allocation for every SAHI crop.
    stacked = np.stack(normalized_arrays, axis=0)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
    elif tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    else:
        raise ValueError(f"Array images must use uint8 or floating dtype; got {tensor.dtype}.")
    sizes = [(int(array.shape[0]), int(array.shape[1])) for array in normalized_arrays]
    return tensor, sizes


def _record_cuda_call(
    recorder: ForwardTimingRecorder,
    device: torch.device,
    callback: Callable[[], Any],
) -> Any:
    """Record one current-stream CUDA stage without synchronizing it."""

    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        result = callback()
        end.record(stream)
    recorder.add_cuda_events(start, end)
    return result


def prepare_inference_batch(
    images: Any,
    *,
    shape: int | tuple[int, int],
    device: str | torch.device,
    mean: Iterable[float] = (0.485, 0.456, 0.406),
    std: Iterable[float] = (0.229, 0.224, 0.225),
    num_channels: int = 3,
    non_blocking: bool = True,
) -> PreparedInferenceBatch:
    """Build one normalized RF-DETR batch with grouped resize and one H2D per shape.

    Inputs with a common source size (the normal SAHI case) are stacked before
    transfer and resized in one CUDA operation.  Mixed source sizes are grouped
    so the path remains lossless without falling back to one transfer per image.
    The function intentionally does not scan floating images for values outside
    ``[0, 1]``; callers of this fast path own that documented input contract.
    """

    prepare_started = time.perf_counter()
    target_shape = _resolved_image_shape(shape)
    target_device = device if isinstance(device, torch.device) else torch.device(device)
    channels = _positive_int(num_channels, "inference image channels")
    image_list = images if isinstance(images, list) else [images]
    if not image_list:
        raise ValueError("RF-DETR inference batch cannot be empty.")

    mean_values = tuple(float(value) for value in mean)
    std_values = tuple(float(value) for value in std)
    if len(mean_values) != channels or len(std_values) != channels or any(value <= 0 for value in std_values):
        raise ValueError(
            f"RF-DETR normalization requires {channels} means and positive stds; "
            f"got mean={mean_values!r}, std={std_values!r}."
        )

    uniform_array_batch = _stack_uniform_array_images(image_list, channels)
    if uniform_array_batch is not None:
        host_batch, original_sizes = uniform_array_batch
        group_batches = [(tuple(range(len(image_list))), host_batch)]
    else:
        tensors = [_image_to_chw_float_tensor(image, channels) for image in image_list]
        original_sizes = [(int(tensor.shape[1]), int(tensor.shape[2])) for tensor in tensors]
        groups: dict[tuple[str, tuple[int, ...]], list[tuple[int, torch.Tensor]]] = {}
        for index, tensor in enumerate(tensors):
            key = (str(tensor.device), tuple(int(value) for value in tensor.shape))
            groups.setdefault(key, []).append((index, tensor))
        group_batches = []
        for members in groups.values():
            indices, group_tensors = zip(*members)
            group_batches.append((tuple(indices), torch.stack(group_tensors, dim=0)))

    target_sizes_cpu = torch.tensor(original_sizes, dtype=torch.int64)
    host_seconds = time.perf_counter() - prepare_started
    h2d_timing = ForwardTimingRecorder() if target_device.type == "cuda" else None
    resize_timing = ForwardTimingRecorder() if target_device.type == "cuda" else None

    resized_by_index: list[torch.Tensor | None] = [None] * len(image_list)
    single_group_batch: torch.Tensor | None = None
    for indices, source_batch in group_batches:
        def move() -> torch.Tensor:
            return source_batch.to(
                device=target_device,
                dtype=torch.float32,
                non_blocking=bool(non_blocking),
            )

        batch = (
            _record_cuda_call(h2d_timing, target_device, move)
            if h2d_timing is not None
            else move()
        )
        if tuple(batch.shape[-2:]) != target_shape:
            def resize() -> torch.Tensor:
                return torch.nn.functional.interpolate(
                    batch,
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

            batch = (
                _record_cuda_call(resize_timing, target_device, resize)
                if resize_timing is not None
                else resize()
            )
        if len(group_batches) == 1:
            single_group_batch = batch
        else:
            for batch_index, image_index in enumerate(indices):
                resized_by_index[image_index] = batch[batch_index]

    if single_group_batch is not None:
        batch_tensor = single_group_batch
    else:
        if any(value is None for value in resized_by_index):  # pragma: no cover - defensive invariant
            raise RuntimeError("RF-DETR batch grouping lost one or more images.")
        def stack_resized() -> torch.Tensor:
            return torch.stack(
                [value for value in resized_by_index if value is not None],
                dim=0,
            )

        batch_tensor = (
            _record_cuda_call(resize_timing, target_device, stack_resized)
            if resize_timing is not None
            else stack_resized()
        )

    def normalize_batch() -> torch.Tensor:
        mean_tensor = batch_tensor.new_tensor(mean_values).view(1, channels, 1, 1)
        std_tensor = batch_tensor.new_tensor(std_values).view(1, channels, 1, 1)
        return batch_tensor.sub_(mean_tensor).div_(std_tensor).contiguous()

    batch_tensor = (
        _record_cuda_call(resize_timing, target_device, normalize_batch)
        if resize_timing is not None
        else normalize_batch()
    )

    def move_sizes() -> torch.Tensor:
        return target_sizes_cpu.to(device=target_device, non_blocking=bool(non_blocking))

    target_sizes = (
        _record_cuda_call(h2d_timing, target_device, move_sizes)
        if h2d_timing is not None
        else move_sizes()
    )
    prepare_wall_seconds = time.perf_counter() - prepare_started
    if target_device.type != "cuda":
        host_seconds = prepare_wall_seconds
    timing = PreprocessTiming(
        host_seconds=host_seconds,
        prepare_wall_seconds=prepare_wall_seconds,
        h2d=h2d_timing,
        resize_normalize=resize_timing,
    )
    return PreparedInferenceBatch(
        tensor=batch_tensor,
        target_sizes=target_sizes,
        _timing=timing,
    )


def _pytorch_raw_infer(model: Any, precision: str) -> Callable[[torch.Tensor], dict[str, torch.Tensor]]:
    context = getattr(model, "model", None)
    if context is None:
        raise RuntimeError("RF-DETR model does not expose its model context.")

    def infer(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        if precision == "bf16":
            callback = getattr(context, "inference_model", None)
            if not callable(callback):
                raise RuntimeError("RF-DETR BF16 inference model is not installed.")
            outputs = callback(tensor.to(dtype=torch.bfloat16))
        else:
            callback = getattr(context, "model", None)
            if not callable(callback):
                raise RuntimeError("RF-DETR PyTorch model is not callable.")
            first_parameter = next(callback.parameters(), None) if hasattr(callback, "parameters") else None
            target_device = _model_device(model)
            if first_parameter is not None and first_parameter.device != target_device:
                callback = callback.to(target_device)
                context.model = callback
            if tensor.device != target_device:
                tensor = tensor.to(target_device)
            outputs = callback(tensor)
        return normalize_raw_outputs(outputs)

    return infer


def _base_metadata(settings: InferenceOptimizationConfig) -> dict[str, Any]:
    profile = settings.tensorrt.profile
    return {
        "requested_backend": settings.backend,
        "effective_backend": settings.backend,
        "requested_precision": settings.precision,
        "effective_precision": settings.precision,
        "cache_hit": None,
        "export_seconds": 0.0,
        "build_seconds": 0.0,
        "load_seconds": 0.0,
        "warmup_seconds": 0.0,
        "engine_path": None,
        "manifest_path": None,
        "engine_sha256": None,
        "onnx_path": None,
        "onnx_cache_key": None,
        "onnx_cache_hit": None,
        "timing_cache_path": None,
        "batch_profile": asdict(profile),
        "reuse_output_buffers": False,
    }


def apply_pytorch_optimization(model: Any, settings: InferenceOptimizationConfig) -> AccelerationHandle:
    """Apply the validated PyTorch backend, with FP32 preserving legacy behavior."""

    if settings.backend != "pytorch":
        raise ValueError("apply_pytorch_optimization() requires backend='pytorch'.")
    metadata = _base_metadata(settings)
    signature = f"pytorch:{settings.pytorch_precision}"
    # Mock/proxy objects can synthesize arbitrary attributes from __getattr__;
    # only a marker that was actually written on this instance is authoritative.
    instance_state = getattr(model, "__dict__", None)
    existing = instance_state.get(_ACCELERATION_MARKER) if isinstance(instance_state, dict) else None
    if existing is not None and existing != signature:
        raise RuntimeError(f"RF-DETR model already has incompatible acceleration {existing!r}.")

    if settings.pytorch_precision == "bf16" and existing is None:
        device = _model_device(model)
        _require_cuda(device, bf16=True)
        callback = getattr(model, "optimize_for_inference", None)
        if not callable(callback):
            raise RuntimeError("RF-DETR model does not provide optimize_for_inference().")
        callback(compile=False, dtype=torch.bfloat16)
        setattr(model, _ACCELERATION_MARKER, signature)
    elif existing is None:
        setattr(model, _ACCELERATION_MARKER, signature)

    recorder = instance_state.get(_FORWARD_RECORDER_MARKER) if isinstance(instance_state, dict) else None
    if not isinstance(recorder, ForwardTimingRecorder):
        context = getattr(model, "model", None)
        target_name = "inference_model" if settings.pytorch_precision == "bf16" else "model"
        target = getattr(context, target_name, None)
        if callable(target):
            recorder = ForwardTimingRecorder()
            setattr(context, target_name, _install_forward_timing(target, recorder))
            setattr(model, _FORWARD_RECORDER_MARKER, recorder)
        else:
            # Integration tests and lightweight planning fakes may intentionally
            # omit executable weights. Acceleration remains usable; only optional
            # stage instrumentation is absent. Real RF-DETR contexts are callable.
            recorder = None
    postprocess_recorder = _ensure_postprocess_timing(model)

    metadata["already_applied"] = existing == signature
    return AccelerationHandle(
        model=model,
        settings=settings,
        metadata=metadata,
        _infer_raw=_pytorch_raw_infer(model, settings.pytorch_precision),
        _forward_recorder=recorder,
        _postprocess_recorder=postprocess_recorder,
    )


def _import_optional(name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(f"{name} is required for TensorRT inference. Install {install_hint}.") from exc


def _major_minor_version(value: Any, dependency: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", str(value))
    if match is None:
        raise RuntimeError(f"Unable to determine the installed {dependency} version: {value!r}.")
    return int(match.group(1)), int(match.group(2))


def _import_tensorrt() -> Any:
    trt = _import_optional("tensorrt", "the project's TensorRT optional dependencies")
    version = str(getattr(trt, "__version__", ""))
    parsed = _major_minor_version(version, "TensorRT")
    if parsed < (10, 16) or parsed >= (11, 0):
        raise RuntimeError(f"This runtime requires TensorRT >=10.16,<11; found {version!r}.")
    return trt


def _import_onnx() -> Any:
    onnx = _import_optional("onnx", "onnx>=1.16,<2")
    version = str(getattr(onnx, "__version__", ""))
    parsed = _major_minor_version(version, "ONNX")
    if parsed < (1, 16) or parsed >= (2, 0):
        raise RuntimeError(f"This runtime requires ONNX >=1.16,<2; found {version!r}.")
    return onnx


def _preflight_device(value: str | int | torch.device | None) -> torch.device:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "auto"}):
        return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(value, bool):
        raise ValueError(f"Inference acceleration device must not be boolean; got {value!r}.")
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        return torch.device("cuda", int(value))
    return value if isinstance(value, torch.device) else torch.device(str(value).strip())


def preflight_inference_acceleration(
    settings_or_config: Mapping[str, Any] | InferenceOptimizationConfig,
    *,
    device: str | int | torch.device | None = None,
) -> dict[str, Any]:
    """Fail before model construction when acceleration dependencies are unavailable.

    FP32 remains a zero-dependency no-op. BF16 validates CUDA hardware support.
    TensorRT validates CUDA, ONNX, TensorRT 10, its requested precision flag,
    and the existence of explicitly supplied artifacts. It never builds or
    writes an engine during preflight.
    """

    settings = (
        settings_or_config
        if isinstance(settings_or_config, InferenceOptimizationConfig)
        else resolve_acceleration_config(settings_or_config)
    )
    resolved_device = _preflight_device(device)
    result = {
        "backend": settings.backend,
        "precision": settings.precision,
        "device": str(resolved_device),
        "tensorrt_version": None,
        "onnx_version": None,
    }
    if settings.backend == "pytorch":
        if settings.pytorch_precision == "bf16":
            _require_cuda(resolved_device, bf16=True)
        return result

    _require_cuda(resolved_device, bf16=settings.tensorrt.precision == "bf16")
    trt = _import_tensorrt()
    onnx = _import_onnx()
    flag_name = "FP16" if settings.tensorrt.precision == "fp16" else "BF16"
    if getattr(getattr(trt, "BuilderFlag", None), flag_name, None) is None:
        raise RuntimeError(f"Installed TensorRT does not expose the {flag_name} builder precision flag.")
    engine_path = settings.tensorrt.engine_path
    manifest_path = settings.tensorrt.manifest_path
    if engine_path is not None:
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine does not exist: {engine_path}")
        if manifest_path is None or not manifest_path.is_file():
            raise FileNotFoundError(f"TensorRT manifest does not exist: {manifest_path}")
    result["tensorrt_version"] = str(getattr(trt, "__version__", "unknown"))
    result["onnx_version"] = str(getattr(onnx, "__version__", "unknown"))
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    return repr(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(name)
    return None


def _default_cache_dir() -> Path:
    return DEFAULT_TENSORRT_CACHE_DIR.resolve()


def _model_resolution(model: Any, settings: InferenceOptimizationConfig) -> int:
    if settings.resolution is not None:
        return settings.resolution
    context = getattr(model, "model", None)
    value = getattr(context, "resolution", None)
    if value is None:
        value = getattr(getattr(model, "model_config", None), "resolution", None)
    if value is None:
        raise ValueError("TensorRT requires a fixed model resolution.")
    return _positive_int(value, "model resolution")


def _model_num_channels(model: Any) -> int:
    value = getattr(getattr(model, "model_config", None), "num_channels", 3)
    return _positive_int(value, "model num_channels")


def _model_num_classes(model: Any) -> int:
    value = getattr(getattr(model, "model_config", None), "num_classes", None)
    if value is None:
        class_names = getattr(model, "class_names", None)
        if isinstance(class_names, (list, tuple)):
            value = len(class_names)
    if value is None:
        raise ValueError("TensorRT requires the RF-DETR class count in model_config or class_names.")
    return _positive_int(value, "model num_classes")


def _model_num_queries(model: Any) -> int:
    value = getattr(getattr(model, "model_config", None), "num_queries", None)
    if value is None:
        module = getattr(getattr(model, "model", None), "model", None)
        value = getattr(module, "num_queries", None)
    if value is None:
        raise ValueError("TensorRT requires the RF-DETR query count in model_config or loaded model.")
    return _positive_int(value, "model num_queries")


def _model_num_logit_slots(model: Any) -> int:
    module = getattr(getattr(model, "model", None), "model", None)
    class_embed = getattr(module, "class_embed", None)
    candidates: list[Any] = [class_embed]
    if isinstance(class_embed, (list, tuple, torch.nn.ModuleList)):
        candidates = list(class_embed)
    for candidate in reversed(candidates):
        value = getattr(candidate, "out_features", None)
        if value is not None:
            return _positive_int(value, "model logit slots")
    # RF-DETR 1.8.x reserves one final class-logit slot for background.
    return _model_num_classes(model) + 1


def _qualified_type(value: Any) -> str | None:
    if value is None:
        return None
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _module_state_shapes(module: Any) -> dict[str, list[int]]:
    """Return a weight-free structural signature for cache provenance."""

    callback = getattr(module, "state_dict", None)
    if not callable(callback):
        return {}
    return {
        str(name): [int(dimension) for dimension in value.shape]
        for name, value in sorted(callback().items())
        if isinstance(value, torch.Tensor)
    }


def _model_architecture_identity(model: Any) -> dict[str, Any]:
    """Fingerprint the graph-affecting P2/TrackNet structure actually attached."""

    context = getattr(model, "model", None)
    module = getattr(context, "model", None)
    if module is None:
        module = context if context is not None else model

    backbone_collection = getattr(module, "backbone", None)
    if isinstance(backbone_collection, (list, tuple, torch.nn.ModuleList, torch.nn.Sequential)):
        backbone = backbone_collection[0] if len(backbone_collection) else None
    else:
        backbone = backbone_collection
    encoder = getattr(backbone, "encoder", None)
    projector = getattr(backbone, "projector", None)
    cross_attn_projector = getattr(backbone, "cross_attn_projector", None)
    projector_scale = list(getattr(backbone, "projector_scale", []) or [])

    motion = getattr(module, "motion_module", None)
    motion_identity: dict[str, Any] = {"attached": motion is not None}
    if motion is not None:
        rstr_heads = getattr(motion, "rstr_heads", None)
        motion_identity.update(
            {
                "class": _qualified_type(motion),
                "num_frames": getattr(motion, "num_frames", None),
                "fallback_mode": getattr(motion, "fallback_mode", None),
                "noise_std": getattr(motion, "noise_std", None),
                "mdd_enabled": getattr(motion, "mdd_enabled", None),
                "rstr_enabled": getattr(motion, "rstr_enabled", None),
                "gate_count": len(getattr(motion, "gates", []) or []),
                "rstr_head_count": len(rstr_heads) if rstr_heads is not None else 0,
                "state_shapes": _module_state_shapes(motion),
            }
        )

    return {
        "module_class": _qualified_type(module),
        "backbone_class": _qualified_type(backbone),
        "p2_enabled": "P2" in projector_scale,
        "projector_scale": projector_scale,
        "encoder": {
            "class": _qualified_type(encoder),
            "out_feature_channels": list(getattr(encoder, "_out_feature_channels", []) or []),
            "shape": getattr(encoder, "shape", None),
            "patch_size": getattr(encoder, "patch_size", None),
            "num_windows": getattr(encoder, "num_windows", None),
        },
        "projector": {
            "class": _qualified_type(projector),
            "scale_factors": list(getattr(projector, "scale_factors", []) or []),
            "survival_prob": getattr(projector, "survival_prob", None),
            "force_drop_last_n_features": getattr(projector, "force_drop_last_n_features", None),
            "use_extra_pool": getattr(projector, "use_extra_pool", None),
            "stage_count": len(getattr(projector, "stages", []) or []),
            "state_shapes": _module_state_shapes(projector),
        },
        "cross_attn_projector": {
            "attached": cross_attn_projector is not None,
            "class": _qualified_type(cross_attn_projector),
            "state_shapes": _module_state_shapes(cross_attn_projector),
        },
        "motion": motion_identity,
    }


def _model_manifest_identity(model: Any, supplied: Mapping[str, Any] | None) -> Any:
    config = getattr(model, "model_config", None)
    if hasattr(config, "model_dump"):
        result = dict(config.model_dump())
        # Runtime paths and device placement are covered by dedicated identity fields.
        result.pop("pretrain_weights", None)
        result.pop("device", None)
        resolved_config: Any = _jsonable(result)
    else:
        resolved_config = {"class": type(model).__qualname__}
    if supplied is None:
        return resolved_config
    return {
        "resolved_model_config": resolved_config,
        "project_model_identity": _jsonable(supplied),
    }


def _checkpoint_identity(model: Any, checkpoint_path: str | Path | None) -> dict[str, Any]:
    candidate = _optional_path(checkpoint_path)
    if candidate is None:
        candidate = _optional_path(getattr(getattr(model, "model_config", None), "pretrain_weights", None))
    if candidate is None:
        return {"name": None, "sha256": _loaded_model_state_sha256(model), "source": "loaded_model_state"}
    source_token = str(candidate)
    candidate = candidate.resolve()
    if candidate.is_file():
        return {"name": candidate.name, "sha256": sha256_file(candidate), "source": "checkpoint_file"}
    # RF-DETR configs also accept hosted/default weight identifiers.  At this
    # point the model is already loaded, so hashing its real tensors is both
    # deterministic and safer than treating the user-facing token as content.
    return {
        "name": source_token,
        "sha256": _loaded_model_state_sha256(model),
        "source": "loaded_model_state",
    }


def _loaded_model_state_sha256(model: Any) -> str:
    context = getattr(model, "model", None)
    module = getattr(context, "model", None)
    state_dict_callback = getattr(module, "state_dict", None)
    if not callable(state_dict_callback):
        raise RuntimeError("RF-DETR loaded model does not expose state_dict() for TensorRT cache identity.")
    digest = hashlib.sha256()
    for name, value in sorted(state_dict_callback().items()):
        digest.update(str(name).encode("utf-8"))
        if not isinstance(value, torch.Tensor):
            digest.update(_canonical_json(value))
            continue
        tensor = value.detach()
        if tensor.is_sparse:
            tensor = tensor.to_dense()
        tensor = tensor.to(device="cpu").contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)))
        # NumPy lacks a native bfloat16 dtype. Reinterpret all tensors as raw
        # bytes so every Torch dtype is hashed without numeric conversion.
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _normalized_gpu_uuid(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "n/a", "unknown"}:
        return None
    return text.removeprefix("gpu-")


def _nvidia_smi_gpu_details(cuda_uuid: str | None) -> dict[str, str]:
    """Return immutable device/driver identifiers without requiring NVML bindings."""

    executable = shutil.which("nvidia-smi")
    normalized_uuid = _normalized_gpu_uuid(cuda_uuid)
    if executable is None or normalized_uuid is None:
        return {}
    fields = (
        "uuid",
        "pci.bus_id",
        "pci.device_id",
        "pci.sub_device_id",
        "vbios_version",
        "driver_version",
    )
    try:
        completed = subprocess.run(
            [
                executable,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(row) != len(fields) or _normalized_gpu_uuid(row[0]) != normalized_uuid:
            continue
        values = [str(value).strip() for value in row]
        return {
            "uuid": values[0],
            "pci_bus_id": values[1].lower(),
            "pci_device_id": values[2].lower(),
            "pci_sub_device_id": values[3].lower(),
            "vbios_version": values[4],
            "driver_version": values[5],
        }
    return {}


def _gpu_runtime_identity(device: torch.device) -> dict[str, Any]:
    """Fingerprint the physical CUDA device that will deserialize the engine."""

    with torch.cuda.device(device):
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        gpu_name = torch.cuda.get_device_name(device)
    uuid = _normalized_gpu_uuid(getattr(properties, "uuid", None))
    domain = getattr(properties, "pci_domain_id", None)
    bus = getattr(properties, "pci_bus_id", None)
    slot = getattr(properties, "pci_device_id", None)
    torch_pci_bus_id = None
    if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (domain, bus, slot)):
        torch_pci_bus_id = f"{int(domain):08x}:{int(bus):02x}:{int(slot):02x}.0"
    smi = _nvidia_smi_gpu_details(uuid)
    return {
        "name": str(gpu_name),
        "uuid": smi.get("uuid") or (f"GPU-{uuid}" if uuid is not None else None),
        "pci_bus_id": smi.get("pci_bus_id") or torch_pci_bus_id,
        "pci_device_id": smi.get("pci_device_id"),
        "pci_sub_device_id": smi.get("pci_sub_device_id"),
        "vbios_version": smi.get("vbios_version"),
        "driver_version": smi.get("driver_version"),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
        "multiprocessor_count": int(getattr(properties, "multi_processor_count", 0)),
        "l2_cache_size_bytes": int(getattr(properties, "L2_cache_size", 0)),
        "memory_bus_width_bits": int(getattr(properties, "memory_bus_width", 0)),
        "is_multi_gpu_board": bool(getattr(properties, "is_multi_gpu_board", False)),
    }


def gpu_runtime_identity(device: str | torch.device | int | None = None) -> dict[str, Any]:
    """Return the public physical-GPU identity used by TensorRT cache keys."""

    resolved = _preflight_device(device)
    _require_cuda(resolved)
    return dict(_gpu_runtime_identity(resolved))


def _build_identity(
    model: Any,
    settings: InferenceOptimizationConfig,
    trt: Any,
    *,
    checkpoint_path: str | Path | None,
    model_identity: Mapping[str, Any] | None,
    segmentation: bool,
) -> dict[str, Any]:
    device = _model_device(model)
    _require_cuda(device, bf16=settings.tensorrt.precision == "bf16")
    gpu_identity = _gpu_runtime_identity(device)
    resolution = _model_resolution(model, settings)
    output_names = ["dets", "labels", "masks"] if segmentation else ["dets", "labels"]
    return {
        "export_abi": {
            "version": _TENSORRT_EXPORT_ABI_VERSION,
            "shape_contract": _TENSORRT_EXPORT_SHAPE_CONTRACT,
            "dynamic_axes": ["batch"],
        },
        "checkpoint": _checkpoint_identity(model, checkpoint_path),
        "model": _model_manifest_identity(model, model_identity),
        "architecture": _model_architecture_identity(model),
        "resolution": [resolution, resolution],
        "num_channels": _model_num_channels(model),
        "num_classes": _model_num_classes(model),
        "num_logit_slots": _model_num_logit_slots(model),
        "num_queries": _model_num_queries(model),
        "segmentation": bool(segmentation),
        "mask_downsample_ratio": (
            _positive_int(
                getattr(getattr(model, "model_config", None), "mask_downsample_ratio", 4),
                "model mask_downsample_ratio",
            )
            if segmentation
            else None
        ),
        "outputs": output_names,
        "io_contract": {
            "input_dtype": "float32",
            "input_rank": 4,
            "output_dtypes": {name: "float32" for name in output_names},
            "output_ranks": {"dets": 3, "labels": 3, **({"masks": 4} if segmentation else {})},
        },
        "precision": settings.tensorrt.precision,
        "workspace_gib": settings.tensorrt.workspace_gib,
        "profile": asdict(settings.tensorrt.profile),
        "optimization_profiles": [
            asdict(profile) for profile in tensorrt_optimization_profiles(settings.tensorrt.profile)
        ],
        "runtime": {
            "rfdetr": _package_version("rfdetr"),
            "onnx": _package_version("onnx"),
            "tensorrt": str(getattr(trt, "__version__", "unknown")),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "nvidia_driver": gpu_identity.get("driver_version"),
            "python": sys.version.split()[0],
        },
        "gpu": gpu_identity,
    }


def _manifest_template(identity: Mapping[str, Any]) -> dict[str, Any]:
    cache_key = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "cache_key": cache_key,
        "identity": _jsonable(identity),
    }


def _onnx_cache_identity(engine_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Remove TensorRT/physical-GPU fields that cannot affect exported ONNX."""

    identity = dict(_jsonable(engine_identity))
    identity.pop("gpu", None)
    identity.pop("precision", None)
    identity.pop("workspace_gib", None)
    identity.pop("profile", None)
    identity.pop("optimization_profiles", None)
    runtime = dict(identity.get("runtime", {}))
    runtime.pop("tensorrt", None)
    runtime.pop("nvidia_driver", None)
    identity["runtime"] = runtime
    return identity


def _onnx_manifest_template(engine_identity: Mapping[str, Any]) -> dict[str, Any]:
    identity = _onnx_cache_identity(engine_identity)
    return {
        "schema_version": _ONNX_CACHE_SCHEMA_VERSION,
        "cache_key": hashlib.sha256(_canonical_json(identity)).hexdigest(),
        "identity": identity,
    }


def _validate_cached_onnx(
    onnx_path: Path,
    manifest_path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Cached ONNX model does not exist: {onnx_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cached ONNX manifest does not exist: {manifest_path}")
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema_version") != _ONNX_CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported ONNX cache manifest schema: {manifest.get('schema_version')!r}.")
    for key in ("cache_key", "identity"):
        if _jsonable(manifest.get(key)) != _jsonable(expected.get(key)):
            raise RuntimeError(f"Cached ONNX manifest {key} does not match the requested model/export runtime.")
    if manifest.get("onnx_sha256") != sha256_file(onnx_path):
        raise RuntimeError(f"Cached ONNX hash does not match its manifest: {onnx_path}")
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read TensorRT manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"TensorRT manifest must contain a JSON object: {path}")
    return value


def validate_engine_manifest(
    engine_path: str | Path,
    manifest_path: str | Path,
    expected: Mapping[str, Any] | None = None,
    *,
    validate_timing_cache: bool = False,
) -> dict[str, Any]:
    """Validate provenance plus engine content hash and return the manifest.

    A timing cache accelerates future builds but is not required to execute an
    engine. Callers considering timing-cache reuse opt into validating it.
    """

    engine = Path(engine_path)
    manifest_file = Path(manifest_path)
    if not engine.is_file():
        raise FileNotFoundError(f"TensorRT engine does not exist: {engine}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"TensorRT manifest does not exist: {manifest_file}")
    manifest = _read_manifest(manifest_file)
    if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported TensorRT manifest schema {manifest.get('schema_version')!r}; "
            f"expected {_MANIFEST_SCHEMA_VERSION}."
        )
    if expected is not None:
        for key in ("cache_key", "identity"):
            if _jsonable(manifest.get(key)) != _jsonable(expected.get(key)):
                raise RuntimeError(f"TensorRT manifest {key} does not match the requested model/runtime.")
    actual_hash = sha256_file(engine)
    declared_hash = manifest.get("engine_sha256")
    if not isinstance(declared_hash, str) or declared_hash != actual_hash:
        raise RuntimeError(f"TensorRT engine hash does not match its manifest: {engine}")
    declared_timing_hash = manifest.get("timing_cache_sha256")
    if validate_timing_cache and declared_timing_hash is not None:
        timing_name = str(manifest.get("timing_cache_file") or "timing.cache")
        timing_path = manifest_file.parent / timing_name
        if not timing_path.is_file() or sha256_file(timing_path) != declared_timing_hash:
            raise RuntimeError(f"TensorRT timing cache does not match its manifest: {timing_path}")
    return manifest


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, json.dumps(_jsonable(value), indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


class _ArtifactLock:
    """Small cross-platform exclusive-create lock with stale-lock recovery."""

    def __init__(self, path: Path, timeout: float = 600.0, stale_after: float = 3600.0) -> None:
        self.path = path
        self.timeout = timeout
        self.stale_after = stale_after
        self._owned = False

    def __enter__(self) -> _ArtifactLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    json.dump({"pid": os.getpid(), "created": time.time()}, file)
                self._owned = True
                return self
            except FileExistsError:
                with contextlib.suppress(OSError):
                    if time.time() - self.path.stat().st_mtime > self.stale_after:
                        self.path.unlink()
                        continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for TensorRT cache lock: {self.path}") from None
                time.sleep(0.1)

    def __exit__(self, *_args: Any) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)


def _trt_logger(trt: Any) -> Any:
    severity = getattr(getattr(trt, "Logger", None), "WARNING", None)
    return trt.Logger(severity) if severity is not None else trt.Logger()


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    settings: InferenceOptimizationConfig,
    *,
    trt_module: Any | None = None,
    timing_cache_path: str | Path | None = None,
) -> Path:
    """Build a TensorRT 10 engine with dynamic batch and fixed spatial shape."""

    if settings.backend != "tensorrt":
        raise ValueError("build_tensorrt_engine() requires backend='tensorrt'.")
    trt = trt_module if trt_module is not None else _import_tensorrt()
    source = Path(onnx_path)
    destination = Path(engine_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {source}")

    logger = _trt_logger(trt)
    builder = trt.Builder(logger)
    network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(source.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(int(parser.num_errors))]
        raise RuntimeError("TensorRT ONNX parsing failed:\n" + "\n".join(errors))
    if int(network.num_inputs) != 1:
        raise RuntimeError(f"TensorRT RF-DETR engine requires exactly one input; ONNX has {network.num_inputs}.")

    input_tensor = network.get_input(0)
    input_shape = tuple(int(value) for value in input_tensor.shape)
    if len(input_shape) != 4:
        raise RuntimeError(f"TensorRT RF-DETR input must be NCHW rank 4; got {input_shape}.")
    if input_shape[0] != -1:
        raise RuntimeError(f"ONNX input batch dimension must be dynamic (-1); got {input_shape}.")
    resolution = settings.resolution
    if resolution is None:
        if input_shape[2] <= 0 or input_shape[3] <= 0 or input_shape[2] != input_shape[3]:
            raise RuntimeError(f"ONNX input must have fixed square spatial dimensions; got {input_shape}.")
        resolution = input_shape[2]
    if input_shape[2:] != (resolution, resolution):
        raise RuntimeError(
            f"ONNX spatial shape {input_shape[2:]} does not match configured resolution {(resolution, resolution)}."
        )

    build_config = builder.create_builder_config()
    workspace_bytes = int(settings.tensorrt.workspace_gib * (1024**3))
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    flag_name = "FP16" if settings.tensorrt.precision == "fp16" else "BF16"
    flag = getattr(trt.BuilderFlag, flag_name, None)
    if flag is None:
        raise RuntimeError(f"Installed TensorRT does not expose the {flag_name} builder flag.")
    support_attr = "platform_has_fast_fp16" if flag_name == "FP16" else "platform_has_fast_bf16"
    if hasattr(builder, support_attr) and not bool(getattr(builder, support_attr)):
        raise RuntimeError(f"The current TensorRT platform does not support fast {flag_name} inference.")
    build_config.set_flag(flag)
    timing_path = Path(timing_cache_path) if timing_cache_path is not None else None
    timing_cache = None
    create_timing_cache = getattr(build_config, "create_timing_cache", None)
    set_timing_cache = getattr(build_config, "set_timing_cache", None)
    if callable(create_timing_cache) and callable(set_timing_cache):
        timing_data = timing_path.read_bytes() if timing_path is not None and timing_path.is_file() else b""
        timing_cache = create_timing_cache(timing_data)
        if timing_cache is None:
            raise RuntimeError("TensorRT could not create a timing cache.")
        if set_timing_cache(timing_cache, ignore_mismatch=False) is False:
            raise RuntimeError("TensorRT rejected the timing cache for this builder configuration.")

    channels = input_shape[1]
    if channels <= 0:
        raise RuntimeError(f"ONNX input channel dimension must be fixed; got {input_shape}.")
    for resolved in tensorrt_optimization_profiles(settings.tensorrt.profile):
        profile = builder.create_optimization_profile()
        accepted = profile.set_shape(
            input_tensor.name,
            (resolved.min_batch_size, channels, resolution, resolution),
            (resolved.opt_batch_size, channels, resolution, resolution),
            (resolved.max_batch_size, channels, resolution, resolution),
        )
        if accepted is False:
            raise RuntimeError(
                "TensorRT rejected dynamic-batch optimization profile "
                f"{resolved.min_batch_size}/{resolved.opt_batch_size}/{resolved.max_batch_size}."
            )
        profile_index = build_config.add_optimization_profile(profile)
        if profile_index is False or (
            isinstance(profile_index, int) and not isinstance(profile_index, bool) and profile_index < 0
        ):
            raise RuntimeError(
                "TensorRT could not add dynamic-batch optimization profile "
                f"{resolved.min_batch_size}/{resolved.opt_batch_size}/{resolved.max_batch_size}."
            )

    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the serialized engine.")
    _atomic_write_bytes(destination, bytes(serialized))
    if timing_path is not None and timing_cache is not None:
        serialized_timing = timing_cache.serialize()
        if serialized_timing is None:
            raise RuntimeError("TensorRT failed to serialize its timing cache.")
        _atomic_write_bytes(timing_path, bytes(serialized_timing))
    return destination


def _onnx_dimension_value(dimension: Any) -> int | None:
    '''Return a positive static ONNX dimension, or None when symbolic/unknown.'''

    has_field = getattr(dimension, 'HasField', None)
    if callable(has_field):
        with contextlib.suppress(ValueError):
            if not has_field('dim_value'):
                return None
    value = getattr(dimension, 'dim_value', None)
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _onnx_dimension_text(dimension: Any) -> str:
    value = _onnx_dimension_value(dimension)
    if value is not None:
        return str(value)
    symbolic = str(getattr(dimension, 'dim_param', '') or '').strip()
    return symbolic or 'unknown'


def _onnx_value_dimensions(value_info: Any) -> list[Any]:
    tensor_type = getattr(getattr(value_info, 'type', None), 'tensor_type', None)
    shape = getattr(tensor_type, 'shape', None)
    dimensions = getattr(shape, 'dim', None)
    return list(dimensions) if dimensions is not None else []


def _onnx_io_contract_issues(
    graph: Any,
    *,
    resolution: int,
    num_channels: int,
    segmentation: bool,
    num_queries: int | None = None,
    num_logit_slots: int | None = None,
    mask_downsample_ratio: int | None = None,
) -> list[str]:
    initializer_names = {str(initializer.name) for initializer in graph.initializer}
    graph_inputs = [value for value in graph.input if str(value.name) not in initializer_names]
    graph_outputs = list(graph.output)
    expected_outputs = ['dets', 'labels', *(['masks'] if segmentation else [])]
    issues: list[str] = []

    actual_inputs = [str(value.name) for value in graph_inputs]
    actual_outputs = [str(value.name) for value in graph_outputs]
    if actual_inputs != ['input']:
        issues.append(f'graph inputs must be [input], got {actual_inputs}')
    if actual_outputs != expected_outputs:
        issues.append(f'graph outputs must be {expected_outputs}, got {actual_outputs}')

    expected_ranks = {'input': 4, 'dets': 3, 'labels': 3, 'masks': 4}
    for value_info in [*graph_inputs, *graph_outputs]:
        name = str(value_info.name)
        tensor_type = getattr(getattr(value_info, 'type', None), 'tensor_type', None)
        element_type = getattr(tensor_type, 'elem_type', None)
        if element_type is not None and int(element_type) != 1:
            issues.append(f'{name!r} must use ONNX FLOAT tensors, got elem_type={element_type!r}')
        dimensions = _onnx_value_dimensions(value_info)
        expected_rank = expected_ranks.get(name)
        if expected_rank is not None and len(dimensions) != expected_rank:
            issues.append(f'{name!r} must be rank {expected_rank}, got rank {len(dimensions)}')
            continue
        for axis, dimension in enumerate(dimensions):
            static_value = _onnx_dimension_value(dimension)
            if axis == 0:
                if static_value is not None:
                    issues.append(
                        f'{name!r} batch axis must be dynamic, got {_onnx_dimension_text(dimension)}'
                    )
                elif not str(getattr(dimension, 'dim_param', '') or '').strip():
                    issues.append(f'{name!r} batch axis must be symbolic, got unknown')
            elif static_value is None:
                issues.append(
                    f'{name!r} axis {axis} must be static, got {_onnx_dimension_text(dimension)}'
                )
    expected_output_shapes: dict[str, list[int | None]] = {
        'dets': [None, num_queries, 4],
        'labels': [None, num_queries, num_logit_slots],
    }
    if segmentation:
        mask_size = None
        if mask_downsample_ratio is not None:
            mask_size = int(resolution) // int(mask_downsample_ratio)
        expected_output_shapes['masks'] = [None, num_queries, mask_size, mask_size]
    for value_info in graph_outputs:
        name = str(value_info.name)
        expected_shape = expected_output_shapes.get(name)
        dimensions = _onnx_value_dimensions(value_info)
        if expected_shape is None or len(dimensions) != len(expected_shape):
            continue
        for axis, expected_value in enumerate(expected_shape):
            if expected_value is None:
                continue
            actual_value = _onnx_dimension_value(dimensions[axis])
            if actual_value != int(expected_value):
                issues.append(
                    f'{name!r} axis {axis} must be {expected_value}, '
                    f'got {_onnx_dimension_text(dimensions[axis])}'
                )

    input_info = next((value for value in graph_inputs if str(value.name) == 'input'), None)
    if input_info is not None:
        dimensions = _onnx_value_dimensions(input_info)
        if len(dimensions) == 4:
            expected_nchw = [None, int(num_channels), int(resolution), int(resolution)]
            for axis, expected_value in enumerate(expected_nchw[1:], start=1):
                actual_value = _onnx_dimension_value(dimensions[axis])
                if actual_value != expected_value:
                    issues.append(
                        f'input axis {axis} must be {expected_value}, '
                        f'got {_onnx_dimension_text(dimensions[axis])}'
                    )
    return issues


def _onnx_convolution_contract_issues(graph: Any) -> list[str]:
    value_infos = {
        str(value.name): value
        for value in [*graph.input, *graph.output, *graph.value_info]
    }
    issues: list[str] = []
    for node in graph.node:
        operation = str(node.op_type)
        if operation not in {'Conv', 'ConvTranspose'} or not node.input:
            continue
        tensor_name = str(node.input[0])
        value_info = value_infos.get(tensor_name)
        node_name = str(node.name or '<unnamed>')
        if value_info is None:
            issues.append(
                f'{operation} node {node_name!r} input tensor {tensor_name!r} has no inferred shape'
            )
            continue
        dimensions = _onnx_value_dimensions(value_info)
        if len(dimensions) != 4:
            issues.append(
                f'{operation} node {node_name!r} input tensor {tensor_name!r} '
                f'must be rank 4 NCHW, got rank {len(dimensions)}'
            )
            continue
        for axis, label in ((1, 'channel'), (2, 'height'), (3, 'width')):
            if _onnx_dimension_value(dimensions[axis]) is None:
                issues.append(
                    f'{operation} node {node_name!r} input tensor {tensor_name!r} {label} axis '
                    f'must be static, got {_onnx_dimension_text(dimensions[axis])}'
                )
    return issues


def _onnx_shape_tensor_contract_issues(graph: Any) -> list[str]:
    """Reject the pre-1.8.3 dynamic shape rewrite unsupported by TensorRT."""

    issues: list[str] = []
    for node in graph.node:
        if str(node.op_type) != 'ScatterND':
            continue
        node_name = str(node.name or '<unnamed>')
        inputs = [str(value) for value in node.input]
        outputs = [str(value) for value in node.output]
        issues.append(
            f'ScatterND node {node_name!r} is not allowed in the RF-DETR TensorRT '
            f'shape contract; inputs={inputs}, outputs={outputs}. '
            'Use the rfdetr==1.8.3 export path.'
        )
    return issues


def _validate_onnx_export_contract(
    path: str | Path,
    *,
    resolution: int,
    num_channels: int,
    segmentation: bool,
    num_queries: int | None = None,
    num_logit_slots: int | None = None,
    mask_downsample_ratio: int | None = None,
    onnx_module: Any | None = None,
) -> Path:
    '''Validate and persist TensorRT-safe dynamic-batch/static-NCHW shapes.'''

    onnx = onnx_module or _import_onnx()
    source = Path(path)
    try:
        model_proto = onnx.load(str(source))
        onnx.checker.check_model(model_proto)
    except Exception as exc:
        raise RuntimeError(f'Unable to load/check exported ONNX model {source}: {exc}') from exc
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model_proto,
            check_type=True,
            strict_mode=True,
            data_prop=True,
        )
        onnx.checker.check_model(inferred)
    except Exception as exc:
        raise RuntimeError(f'Unable to infer/check exported ONNX shapes for {source}: {exc}') from exc

    issues = _onnx_io_contract_issues(
        inferred.graph,
        resolution=resolution,
        num_channels=num_channels,
        segmentation=segmentation,
        num_queries=num_queries,
        num_logit_slots=num_logit_slots,
        mask_downsample_ratio=mask_downsample_ratio,
    )
    issues.extend(_onnx_convolution_contract_issues(inferred.graph))
    issues.extend(_onnx_shape_tensor_contract_issues(inferred.graph))
    if issues:
        raise RuntimeError(
            'TensorRT ONNX export contract validation failed:\n- ' + '\n- '.join(issues)
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{source.name}.',
        suffix='.shape-inferred.tmp',
        dir=source.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        onnx.save_model(inferred, str(temporary))
        os.replace(temporary, source)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'Unable to persist inferred ONNX shapes for {source}: {exc}') from exc
    return source


def _export_dynamic_onnx(model: Any, output_dir: Path, settings: InferenceOptimizationConfig) -> Path:
    onnx = _import_onnx()
    from rf_detr_motion import assert_motion_export_ready

    assert_motion_export_ready(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = _model_resolution(model, settings)
    callback = getattr(model, "export", None)
    if not callable(callback):
        raise RuntimeError("RF-DETR model does not provide export().")
    exported = callback(
        output_dir=str(output_dir),
        shape=(resolution, resolution),
        # A single sample is sufficient to trace the dynamic batch axis.  Using
        # the optimization batch here can allocate hundreds of samples for
        # SAHI/recheck profiles before TensorRT has even started building.
        batch_size=1,
        dynamic_batch=True,
        format="onnx",
        verbose=False,
    )
    path = Path(exported)
    if not path.is_file():
        raise RuntimeError(f"RF-DETR export did not produce an ONNX file: {path}")
    return _validate_onnx_export_contract(
        path,
        resolution=resolution,
        num_channels=_model_num_channels(model),
        segmentation=_infer_segmentation(model),
        num_queries=_model_num_queries(model),
        num_logit_slots=_model_num_logit_slots(model),
        mask_downsample_ratio=(
            _positive_int(
                getattr(getattr(model, 'model_config', None), 'mask_downsample_ratio', 4),
                'model mask_downsample_ratio',
            )
            if _infer_segmentation(model)
            else None
        ),
        onnx_module=onnx,
    )


def _infer_segmentation(model: Any) -> bool:
    config = getattr(model, "model_config", None)
    return bool(getattr(config, "segmentation_head", False))


def _release_pytorch_cuda_weights(model: Any) -> None:
    """Move live RF-DETR weights off CUDA before TensorRT build/load.

    RF-DETR's exporter restores the source model to its configured device.
    Keeping that copy resident while the builder reserves its workspace is a
    common first-build OOM on laptop GPUs.  The TensorRT adapter retains this
    CPU copy only so callers can explicitly restore the PyTorch backend later.
    """

    context = getattr(model, "model", None)
    module = getattr(context, "model", None)
    if module is None or not hasattr(module, "to"):
        return
    context.model = module.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _combined_p2_motion_diagnostic(model: Any, resolution: int) -> str | None:
    """Describe the quadratic attention pressure of a P2+TrackNet graph."""

    architecture = _model_architecture_identity(model)
    levels = list(architecture.get("projector_scale", []) or [])
    motion = architecture.get("motion", {}) or {}
    if "P2" not in levels or not bool(motion.get("attached", False)):
        return None
    stride_by_level = {"P2": 4, "P3": 8, "P4": 16, "P5": 32, "P6": 64}
    tokens_by_level = {
        level: max(1, int(resolution) // stride_by_level[level]) ** 2
        for level in levels
        if level in stride_by_level
    }
    total_tokens = sum(tokens_by_level.values())
    return (
        "P2+TrackNet TensorRT diagnostic: "
        f"resolution={resolution}x{resolution}, tokens_by_level={tokens_by_level}, "
        f"total_feature_tokens={total_tokens}. TrackNet R-STR uses global feature attention, "
        "whose memory grows quadratically with each level's token count. Retry with P2 disabled "
        "or a lower model resolution."
    )


def prepare_tensorrt_engine(
    model: Any,
    settings: InferenceOptimizationConfig,
    *,
    checkpoint_path: str | Path | None = None,
    model_identity: Mapping[str, Any] | None = None,
    segmentation: bool | None = None,
) -> TensorRTArtifact:
    """Validate a supplied engine or atomically export/build a cached engine."""

    if settings.backend != "tensorrt":
        raise ValueError("prepare_tensorrt_engine() requires backend='tensorrt'.")
    trt = _import_tensorrt()
    if segmentation is None:
        segmentation = _infer_segmentation(model)
    identity = _build_identity(
        model,
        settings,
        trt,
        checkpoint_path=checkpoint_path,
        model_identity=model_identity,
        segmentation=bool(segmentation),
    )
    expected = _manifest_template(identity)
    onnx_expected = _onnx_manifest_template(identity)
    supplied_engine = settings.tensorrt.engine_path
    supplied_manifest = settings.tensorrt.manifest_path
    if supplied_engine is not None and supplied_manifest is not None:
        manifest = validate_engine_manifest(supplied_engine, supplied_manifest, expected)
        return TensorRTArtifact(
            engine_path=supplied_engine.resolve(),
            manifest_path=supplied_manifest.resolve(),
            onnx_path=None,
            cache_hit=True,
            manifest=manifest,
            timing_cache_path=None,
        )

    cache_root = (settings.tensorrt.cache_dir or _default_cache_dir()).resolve()
    artifact_dir = cache_root / expected["cache_key"]
    engine_path = artifact_dir / "model.engine"
    manifest_path = artifact_dir / "model.engine.manifest.json"
    timing_cache_path = artifact_dir / "timing.cache"
    onnx_cache_root = cache_root / "onnx"
    onnx_artifact_dir = onnx_cache_root / onnx_expected["cache_key"]
    onnx_path = onnx_artifact_dir / "model.onnx"
    onnx_manifest_path = onnx_artifact_dir / "model.onnx.manifest.json"

    def validated_onnx_path() -> Path | None:
        try:
            _validate_cached_onnx(onnx_path, onnx_manifest_path, onnx_expected)
        except (FileNotFoundError, RuntimeError):
            return None
        return onnx_path

    def cached_artifact() -> TensorRTArtifact | None:
        if settings.tensorrt.force_rebuild:
            return None
        try:
            manifest = validate_engine_manifest(engine_path, manifest_path, expected)
        except (FileNotFoundError, RuntimeError):
            return None
        return TensorRTArtifact(
            engine_path=engine_path,
            manifest_path=manifest_path,
            onnx_path=validated_onnx_path(),
            cache_hit=True,
            manifest=manifest,
            timing_cache_path=timing_cache_path if timing_cache_path.is_file() else None,
        )

    def validated_timing_cache() -> Path | None:
        """Reuse timing data only when its published artifact validates fully."""
        try:
            manifest = validate_engine_manifest(
                engine_path,
                manifest_path,
                expected,
                validate_timing_cache=True,
            )
        except (FileNotFoundError, RuntimeError):
            return None
        declared_hash = manifest.get("timing_cache_sha256")
        if not declared_hash or not timing_cache_path.is_file():
            return None
        return timing_cache_path

    def shared_onnx_artifact() -> tuple[Path, float, bool]:
        cached_onnx = validated_onnx_path()
        if cached_onnx is not None:
            return cached_onnx, 0.0, True
        onnx_cache_root.mkdir(parents=True, exist_ok=True)
        onnx_lock_path = onnx_cache_root / f".{onnx_expected['cache_key']}.lock"
        with _ArtifactLock(onnx_lock_path):
            cached_onnx = validated_onnx_path()
            if cached_onnx is not None:
                return cached_onnx, 0.0, True
            temporary_onnx_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{onnx_expected['cache_key']}.",
                    suffix=".tmp",
                    dir=onnx_cache_root,
                )
            )
            try:
                export_started = time.perf_counter()
                exported_path = _export_dynamic_onnx(model, temporary_onnx_dir, settings)
                export_seconds = time.perf_counter() - export_started
                temporary_onnx = temporary_onnx_dir / "model.onnx"
                if exported_path.resolve() != temporary_onnx.resolve():
                    shutil.copy2(exported_path, temporary_onnx)
                onnx_manifest = {
                    **onnx_expected,
                    "onnx_sha256": sha256_file(temporary_onnx),
                    "created_unix_seconds": time.time(),
                }
                _atomic_write_json(temporary_onnx_dir / "manifest.json", onnx_manifest)
                onnx_artifact_dir.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_onnx, onnx_path)
                # Publish the manifest last so readers reject partial exports.
                os.replace(temporary_onnx_dir / "manifest.json", onnx_manifest_path)
                return onnx_path, export_seconds, False
            finally:
                shutil.rmtree(temporary_onnx_dir, ignore_errors=True)

    cached = cached_artifact()
    if cached is not None:
        return cached

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / f".{expected['cache_key']}.lock"
    with _ArtifactLock(lock_path):
        cached = cached_artifact()
        if cached is not None:
            return cached
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{expected['cache_key']}.", suffix=".tmp", dir=cache_root))
        try:
            build_device = _model_device(model)
            shared_onnx, export_seconds, onnx_cache_hit = shared_onnx_artifact()
            temporary_onnx = temporary_dir / "model.onnx"
            shutil.copy2(shared_onnx, temporary_onnx)

            # Export moves the original RF-DETR model back to CUDA.  Release it
            # before the TensorRT builder allocates tactic/workspace memory.
            _release_pytorch_cuda_weights(model)

            temporary_engine = temporary_dir / "model.engine"
            temporary_timing_cache = temporary_dir / "timing.cache"
            reusable_timing_cache = validated_timing_cache()
            if reusable_timing_cache is not None:
                shutil.copy2(reusable_timing_cache, temporary_timing_cache)
            build_started = time.perf_counter()
            with torch.cuda.device(build_device):
                build_tensorrt_engine(
                    temporary_onnx,
                    temporary_engine,
                    settings,
                    trt_module=trt,
                    timing_cache_path=temporary_timing_cache,
                )
            build_seconds = time.perf_counter() - build_started
            manifest = {
                **expected,
                "engine_sha256": sha256_file(temporary_engine),
                "onnx_sha256": sha256_file(shared_onnx),
                "onnx_cache_key": onnx_expected["cache_key"],
                "onnx_cache_hit": onnx_cache_hit,
                "onnx_manifest_path": str(onnx_manifest_path),
                "created_unix_seconds": time.time(),
            }
            if temporary_timing_cache.is_file():
                manifest.update(
                    {
                        "timing_cache_file": "timing.cache",
                        "timing_cache_sha256": sha256_file(temporary_timing_cache),
                    }
                )
            _atomic_write_json(temporary_dir / "manifest.json", manifest)

            artifact_dir.mkdir(parents=True, exist_ok=True)
            # Publish the manifest last so readers never accept a partial artifact.
            os.replace(temporary_engine, engine_path)
            if temporary_timing_cache.is_file():
                os.replace(temporary_timing_cache, timing_cache_path)
            os.replace(temporary_dir / "manifest.json", manifest_path)
            return TensorRTArtifact(
                engine_path=engine_path,
                manifest_path=manifest_path,
                onnx_path=shared_onnx,
                cache_hit=False,
                manifest=manifest,
                timing_cache_path=timing_cache_path if timing_cache_path.is_file() else None,
                export_seconds=export_seconds,
                build_seconds=build_seconds,
            )
        except Exception as exc:
            diagnostic = _combined_p2_motion_diagnostic(model, _model_resolution(model, settings))
            if diagnostic is not None:
                raise RuntimeError(f"{exc}\n{diagnostic}") from exc
            raise
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)


def _torch_dtype_for_trt(trt: Any, dtype: Any) -> torch.dtype:
    mapping = {
        getattr(getattr(trt, "DataType", object()), "FLOAT", object()): torch.float32,
        getattr(getattr(trt, "DataType", object()), "HALF", object()): torch.float16,
        getattr(getattr(trt, "DataType", object()), "BF16", object()): torch.bfloat16,
        getattr(getattr(trt, "DataType", object()), "INT8", object()): torch.int8,
        getattr(getattr(trt, "DataType", object()), "INT32", object()): torch.int32,
        getattr(getattr(trt, "DataType", object()), "INT64", object()): torch.int64,
        getattr(getattr(trt, "DataType", object()), "BOOL", object()): torch.bool,
    }
    result = mapping.get(dtype)
    if result is None:
        raise RuntimeError(f"Unsupported TensorRT tensor dtype: {dtype!r}.")
    return result


def chunk_batch_ranges(total: int, maximum: int) -> list[tuple[int, int]]:
    """Return lossless contiguous batch chunks no larger than ``maximum``."""

    total = _positive_int(total, "batch size")
    maximum = _positive_int(maximum, "maximum batch size")
    return [(start, min(start + maximum, total)) for start in range(0, total, maximum)]


class _CudaEventPool:
    """Reuse CUDA events after their recorded work has completed."""

    def __init__(self, device: torch.device, *, enable_timing: bool) -> None:
        self.device = device
        self.enable_timing = enable_timing
        self._events: list[Any] = []
        self._lock = threading.Lock()

    def acquire(self) -> Any:
        with self._lock:
            if self._events:
                return self._events.pop()
        with torch.cuda.device(self.device):
            return torch.cuda.Event(enable_timing=self.enable_timing)

    def release(self, event: Any) -> None:
        with self._lock:
            self._events.append(event)


class TensorRTRunner:
    """TensorRT 10 runner using Torch CUDA tensors and ``execute_async_v3``."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        manifest_path: str | Path | None = None,
        device: str | torch.device | None = None,
        expected_manifest: Mapping[str, Any] | None = None,
        trt_module: Any | None = None,
    ) -> None:
        self.engine_path = Path(engine_path).resolve()
        self.forward_timing = ForwardTimingRecorder()
        self.manifest_path = Path(manifest_path).resolve() if manifest_path is not None else None
        self.manifest: dict[str, Any] | None = None
        if self.manifest_path is not None:
            self.manifest = validate_engine_manifest(self.engine_path, self.manifest_path, expected_manifest)
        elif expected_manifest is not None:
            raise ValueError("expected_manifest requires manifest_path.")

        self.device = torch.device(device or "cuda:0")
        _require_cuda(self.device)
        self._trt = trt_module if trt_module is not None else _import_tensorrt()
        self._validate_physical_device()
        logger = _trt_logger(self._trt)
        with torch.cuda.device(self.device):
            self._runtime = self._trt.Runtime(logger)
            self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
            if self._engine is None:
                raise RuntimeError(f"TensorRT could not deserialize engine: {self.engine_path}")
            self._context = self._engine.create_execution_context()
            self._stream = torch.cuda.Stream(device=self.device)
        if self._context is None:
            raise RuntimeError("TensorRT could not create an execution context.")
        self._execution_lock = threading.Lock()
        self._timing_events = _CudaEventPool(self.device, enable_timing=True)
        self._completion_events = _CudaEventPool(self.device, enable_timing=False)
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for index in range(int(self._engine.num_io_tensors)):
            name = str(self._engine.get_tensor_name(index))
            mode = self._engine.get_tensor_mode(name)
            if mode == self._trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)
        if len(self._input_names) != 1:
            raise RuntimeError(f"TensorRT RF-DETR engine must have one input; found {self._input_names}.")
        self.input_name = self._input_names[0]
        self.input_dtype = _torch_dtype_for_trt(self._trt, self._engine.get_tensor_dtype(self.input_name))
        if self.input_dtype is not torch.float32:
            raise RuntimeError(
                f"TensorRT RF-DETR project engines require FP32 input I/O; {self.input_name!r} is "
                f"{self.input_dtype}."
            )
        profile_count = int(getattr(self._engine, "num_optimization_profiles", 1))
        if profile_count <= 0:
            raise RuntimeError("TensorRT engine does not contain an optimization profile.")
        resolved_profile_shapes: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
        for profile_index in range(profile_count):
            profile_shapes = self._engine.get_tensor_profile_shape(self.input_name, profile_index)
            if len(profile_shapes) != 3:
                raise RuntimeError(
                    f"TensorRT profile {profile_index} did not expose min/opt/max input shapes."
                )
            resolved_shapes = tuple(tuple(int(value) for value in shape) for shape in profile_shapes)
            minimum, optimum, maximum = resolved_shapes
            if len(maximum) != 4 or maximum[2] != maximum[3]:
                raise RuntimeError(
                    "TensorRT RF-DETR engine must use fixed square NCHW input; "
                    f"profile {profile_index} has {maximum}."
                )
            if minimum[1:] != maximum[1:] or optimum[1:] != maximum[1:]:
                raise RuntimeError(
                    "TensorRT RF-DETR optimization profiles may only vary the batch dimension."
                )
            resolved_profile_shapes.append(resolved_shapes)
        self.profile_shapes = tuple(resolved_profile_shapes)
        self.min_shape, self.opt_shape, self.max_shape = self.profile_shapes[0]
        if any(shapes[2][1:] != self.max_shape[1:] for shapes in self.profile_shapes[1:]):
            raise RuntimeError("TensorRT optimization profiles disagree on channel/spatial input shape.")
        self.max_batch_size = max(shapes[2][0] for shapes in self.profile_shapes)
        self.resolution = self.max_shape[2]
        self._selected_profile_index = 0
        active_profile = getattr(self._context, "active_optimization_profile", 0)
        self._active_profile_index = int(active_profile) if isinstance(active_profile, int) else 0
        self._semantic_output_names: dict[str, str] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self._output_dtypes: dict[str, torch.dtype] = {}
        for name in self._output_names:
            semantic_name = _OUTPUT_NAME_MAP.get(name)
            if semantic_name is None or semantic_name in self._semantic_output_names:
                raise RuntimeError(f"TensorRT RF-DETR engine has unsupported or duplicate output {name!r}.")
            dtype = _torch_dtype_for_trt(self._trt, self._engine.get_tensor_dtype(name))
            if dtype is not torch.float32:
                raise RuntimeError(
                    f"TensorRT RF-DETR project engines require FP32 output I/O; {name!r} is {dtype}."
                )
            self._semantic_output_names[semantic_name] = name
            self._output_shapes[name] = tuple(int(value) for value in self._engine.get_tensor_shape(name))
            self._output_dtypes[name] = dtype
        missing_outputs = {"pred_boxes", "pred_logits"} - set(self._semantic_output_names)
        if missing_outputs:
            raise RuntimeError(
                f"TensorRT engine outputs {self._output_names} cannot be mapped to {sorted(missing_outputs)}."
            )
        self._validate_output_shapes()
        self._pending_bindings: list[tuple[Any, dict[str, torch.Tensor]]] = []
        self._validate_engine_contract()

    def _validate_physical_device(self) -> None:
        """Reject a new-style engine before deserializing it on another GPU."""

        if self.manifest is None:
            return
        identity = self.manifest.get("identity")
        gpu_identity = identity.get("gpu") if isinstance(identity, Mapping) else None
        expected_uuid = (
            _normalized_gpu_uuid(gpu_identity.get("uuid"))
            if isinstance(gpu_identity, Mapping)
            else None
        )
        # Older manifests did not record a physical UUID. They retain their
        # historical behavior, while all engines produced by this version are
        # protected before TensorRT has a chance to deserialize an alien plan.
        if expected_uuid is None:
            return
        with torch.cuda.device(self.device):
            properties = torch.cuda.get_device_properties(self.device)
        actual_uuid = _normalized_gpu_uuid(getattr(properties, "uuid", None))
        if actual_uuid != expected_uuid:
            raise RuntimeError(
                "TensorRT engine physical GPU UUID does not match the selected CUDA device: "
                f"manifest=GPU-{expected_uuid}, device=GPU-{actual_uuid or 'unknown'}."
            )

    def _validate_output_shapes(self) -> None:
        boxes_name = self._semantic_output_names["pred_boxes"]
        logits_name = self._semantic_output_names["pred_logits"]
        boxes_shape = self._output_shapes[boxes_name]
        logits_shape = self._output_shapes[logits_name]
        if len(boxes_shape) != 3 or boxes_shape[-1] != 4 or boxes_shape[1] <= 0:
            raise RuntimeError(f"TensorRT boxes output must be [N, queries, 4]; got {boxes_shape}.")
        if len(logits_shape) != 3 or logits_shape[1] != boxes_shape[1] or logits_shape[2] <= 0:
            raise RuntimeError(
                f"TensorRT logits output must be [N, {boxes_shape[1]}, classes]; got {logits_shape}."
            )
        masks_name = self._semantic_output_names.get("pred_masks")
        if masks_name is not None:
            masks_shape = self._output_shapes[masks_name]
            if (
                len(masks_shape) != 4
                or masks_shape[1] != boxes_shape[1]
                or masks_shape[2] <= 0
                or masks_shape[2] != masks_shape[3]
            ):
                raise RuntimeError(
                    "TensorRT masks output must be [N, queries, fixed-height, fixed-width]; "
                    f"got {masks_shape}."
                )

    def _validate_engine_contract(self) -> None:
        if self.manifest is None:
            return
        identity = self.manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise RuntimeError("TensorRT manifest identity must be a mapping.")
        expected_resolution = identity.get("resolution")
        if expected_resolution != [self.resolution, self.resolution]:
            raise RuntimeError(
                f"TensorRT engine resolution {self.resolution} does not match manifest {expected_resolution!r}."
            )
        expected_channels = identity.get("num_channels")
        if expected_channels != self.max_shape[1]:
            raise RuntimeError(
                f"TensorRT engine channels {self.max_shape[1]} do not match manifest {expected_channels!r}."
            )
        profile = identity.get("profile")
        if not isinstance(profile, Mapping):
            raise RuntimeError("TensorRT manifest does not contain a batch profile.")
        profile_shapes = getattr(self, "profile_shapes", ((self.min_shape, self.opt_shape, self.max_shape),))
        actual_profiles = [
            {
                "min_batch_size": shapes[0][0],
                "opt_batch_size": shapes[1][0],
                "max_batch_size": shapes[2][0],
            }
            for shapes in profile_shapes
        ]
        expected_profiles = identity.get("optimization_profiles")
        if expected_profiles is not None:
            if _jsonable(actual_profiles) != _jsonable(expected_profiles):
                raise RuntimeError(
                    f"TensorRT engine optimization profiles {actual_profiles} do not match manifest "
                    f"{expected_profiles}."
                )
        else:
            # Schema-v2 manifests created before export ABI v4 contain one
            # profile only and remain directly loadable for compatibility.
            expected_batches = {
                "min_batch_size": profile.get("min_batch_size"),
                "opt_batch_size": profile.get("opt_batch_size"),
                "max_batch_size": profile.get("max_batch_size"),
            }
            if len(actual_profiles) != 1 or actual_profiles[0] != expected_batches:
                raise RuntimeError(
                    f"TensorRT engine batch profile {actual_profiles} does not match manifest "
                    f"{expected_batches}."
                )
        expected_outputs = identity.get("outputs")
        if not isinstance(expected_outputs, list):
            raise RuntimeError("TensorRT manifest does not contain its output names.")
        if set(self._output_names) != set(str(name) for name in expected_outputs):
            raise RuntimeError(
                f"TensorRT engine outputs {self._output_names} do not match manifest {expected_outputs}."
            )
        io_contract = identity.get("io_contract")
        if not isinstance(io_contract, Mapping):
            raise RuntimeError("TensorRT manifest does not contain its project I/O contract.")
        if io_contract.get("input_dtype") != "float32" or io_contract.get("input_rank") != 4:
            raise RuntimeError(f"TensorRT manifest has an incompatible input I/O contract: {io_contract!r}.")
        expected_num_queries = identity.get("num_queries")
        expected_num_logit_slots = identity.get("num_logit_slots")
        boxes_shape = self._output_shapes[self._semantic_output_names["pred_boxes"]]
        logits_shape = self._output_shapes[self._semantic_output_names["pred_logits"]]
        if boxes_shape[1] != expected_num_queries:
            raise RuntimeError(
                f"TensorRT engine query count {boxes_shape[1]} does not match manifest {expected_num_queries!r}."
            )
        if logits_shape[2] != expected_num_logit_slots:
            raise RuntimeError(
                f"TensorRT engine logit-slot count {logits_shape[2]} does not match manifest "
                f"{expected_num_logit_slots!r}."
            )
        expects_masks = bool(identity.get("segmentation"))
        has_masks = "pred_masks" in self._semantic_output_names
        if has_masks != expects_masks:
            raise RuntimeError(
                f"TensorRT engine mask output presence ({has_masks}) does not match manifest ({expects_masks})."
            )
        if has_masks:
            ratio = identity.get("mask_downsample_ratio")
            if not isinstance(ratio, int) or isinstance(ratio, bool) or ratio <= 0:
                raise RuntimeError(f"TensorRT manifest has invalid mask_downsample_ratio {ratio!r}.")
            expected_mask_size = self.resolution // ratio
            masks_shape = self._output_shapes[self._semantic_output_names["pred_masks"]]
            if masks_shape[2:] != (expected_mask_size, expected_mask_size):
                raise RuntimeError(
                    f"TensorRT engine mask size {masks_shape[2:]} does not match manifest "
                    f"resolution/downsample ratio {(expected_mask_size, expected_mask_size)}."
                )

    def _release_completed_bindings(self) -> None:
        pending: list[tuple[Any, dict[str, torch.Tensor]]] = []
        for event, bindings in self._pending_bindings:
            if bool(event.query()):
                pool = getattr(self, "_completion_events", None)
                if pool is not None:
                    pool.release(event)
            else:
                pending.append((event, bindings))
        self._pending_bindings = pending

    def _profile_index_for_batch(self, batch_size: int) -> int:
        profiles = [
            (index, shapes)
            for index, shapes in enumerate(self.profile_shapes)
            if shapes[0][0] <= batch_size <= shapes[2][0]
        ]
        if not profiles:
            raise ValueError(f"No TensorRT optimization profile supports batch size {batch_size}.")
        if any(index == self._selected_profile_index for index, _shapes in profiles):
            return self._selected_profile_index
        return min(profiles, key=lambda item: abs(item[1][1][0] - batch_size))[0]

    def _activate_profile(self, profile_index: int) -> None:
        if profile_index == self._active_profile_index:
            return
        callback = getattr(self._context, "set_optimization_profile_async", None)
        if not callable(callback):
            raise RuntimeError("TensorRT execution context cannot switch optimization profiles.")
        accepted = callback(profile_index, self._stream.cuda_stream)
        if accepted is False:
            raise RuntimeError(f"TensorRT rejected optimization profile {profile_index}.")
        self._active_profile_index = profile_index

    def output_shapes(self, batch_size: int) -> dict[str, tuple[int, ...]]:
        """Return semantic output shapes for one supported, unchunked batch."""

        batch = _positive_int(batch_size, "TensorRT output batch size")
        self._profile_index_for_batch(batch)
        result: dict[str, tuple[int, ...]] = {}
        for semantic_name, engine_name in self._semantic_output_names.items():
            dimensions = list(self._output_shapes[engine_name])
            if dimensions[0] < 0:
                dimensions[0] = batch
            if dimensions[0] != batch or any(value <= 0 for value in dimensions):
                raise RuntimeError(
                    f"TensorRT output {engine_name!r} cannot resolve batch {batch} from "
                    f"{self._output_shapes[engine_name]}."
                )
            result[semantic_name] = tuple(dimensions)
        return result

    def allocate_output_buffers(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Allocate caller-owned outputs that may be reused with :meth:`infer_into`."""

        shapes = self.output_shapes(batch_size)
        return {
            semantic_name: torch.empty(
                shape,
                dtype=self._output_dtypes[engine_name],
                device=self.device,
            )
            for semantic_name, engine_name in self._semantic_output_names.items()
            for shape in (shapes[semantic_name],)
        }

    def _validate_output_buffers(
        self,
        batch_size: int,
        output_buffers: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        expected_shapes = self.output_shapes(batch_size)
        if set(output_buffers) != set(expected_shapes):
            raise ValueError(
                f"TensorRT reusable outputs must contain {sorted(expected_shapes)}; "
                f"got {sorted(str(name) for name in output_buffers)}."
            )
        engine_buffers: dict[str, torch.Tensor] = {}
        for semantic_name, expected_shape in expected_shapes.items():
            value = output_buffers[semantic_name]
            engine_name = self._semantic_output_names[semantic_name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TensorRT reusable output {semantic_name!r} must be a torch.Tensor.")
            if value.device != self.device or value.dtype != self._output_dtypes[engine_name]:
                raise ValueError(
                    f"TensorRT reusable output {semantic_name!r} must use "
                    f"{self.device}/{self._output_dtypes[engine_name]}; got {value.device}/{value.dtype}."
                )
            if tuple(value.shape) != expected_shape or not value.is_contiguous():
                raise ValueError(
                    f"TensorRT reusable output {semantic_name!r} must be contiguous with shape "
                    f"{expected_shape}; got {tuple(value.shape)}."
                )
            engine_buffers[engine_name] = value
        return engine_buffers

    def _infer_chunk(
        self,
        tensor: torch.Tensor,
        *,
        output_bindings: Mapping[str, torch.Tensor] | None = None,
        profile_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.cuda.device(self.device):
            if tensor.device != self.device:
                tensor = tensor.to(self.device, non_blocking=True)
            tensor = tensor.to(dtype=self.input_dtype).contiguous()
            caller_stream = torch.cuda.current_stream(self.device)
            with self._execution_lock:
                self._release_completed_bindings()
                self._stream.wait_stream(caller_stream)
                with torch.cuda.stream(self._stream):
                    selected_profile = (
                        profile_index
                        if profile_index is not None
                        else self._profile_index_for_batch(int(tensor.shape[0]))
                    )
                    self._activate_profile(selected_profile)
                    accepted = self._context.set_input_shape(self.input_name, tuple(tensor.shape))
                    if accepted is False:
                        raise RuntimeError(f"TensorRT rejected input shape {tuple(tensor.shape)}.")

                    bindings: dict[str, torch.Tensor] = {self.input_name: tensor}
                    for name in self._output_names:
                        shape = tuple(int(value) for value in self._context.get_tensor_shape(name))
                        if any(value < 0 for value in shape):
                            raise RuntimeError(f"TensorRT output {name!r} still has unresolved shape {shape}.")
                        reusable = output_bindings.get(name) if output_bindings is not None else None
                        if reusable is not None:
                            if tuple(reusable.shape) != shape:
                                raise RuntimeError(
                                    f"TensorRT reusable output {name!r} shape changed from "
                                    f"{tuple(reusable.shape)} to {shape}."
                                )
                            bindings[name] = reusable
                        else:
                            bindings[name] = torch.empty(
                                shape,
                                dtype=self._output_dtypes[name],
                                device=self.device,
                            )
                    for name, value in bindings.items():
                        accepted = self._context.set_tensor_address(name, int(value.data_ptr()))
                        if accepted is False:
                            raise RuntimeError(f"TensorRT rejected tensor address for {name!r}.")

                    start_event = self._timing_events.acquire()
                    end_event = self._timing_events.acquire()
                    completion_event = self._completion_events.acquire()
                    start_event.record(self._stream)
                    executed = self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
                    end_event.record(self._stream)
                    completion_event.record(self._stream)
                if not executed:
                    self._stream.synchronize()
                    self._timing_events.release(start_event)
                    self._timing_events.release(end_event)
                    self._completion_events.release(completion_event)
                    raise RuntimeError("TensorRT execute_async_v3() failed.")

                def release_timing_events() -> None:
                    self._timing_events.release(start_event)
                    self._timing_events.release(end_event)

                self.forward_timing.add_cuda_events(
                    start_event,
                    end_event,
                    release=release_timing_events,
                )
                # TensorRT executes outside PyTorch's allocator awareness. Keep
                # all bindings alive until the private execution stream finishes.
                self._pending_bindings.append((completion_event, bindings))
                caller_stream.wait_event(completion_event)
                for name in self._output_names:
                    bindings[name].record_stream(caller_stream)
        return normalize_raw_outputs({name: bindings[name] for name in self._output_names})

    def _validate_input(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError("TensorRT RF-DETR input must be a rank-4 NCHW torch.Tensor.")
        if tensor.shape[0] <= 0:
            raise ValueError("TensorRT RF-DETR input batch cannot be empty.")
        if tuple(tensor.shape[1:]) != tuple(self.max_shape[1:]):
            raise ValueError(
                f"TensorRT RF-DETR input must have shape "
                f"[N, {', '.join(str(value) for value in self.max_shape[1:])}]; "
                f"got {tuple(tensor.shape)}."
            )

    def infer(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(tensor)
        chunks = [self._infer_chunk(tensor[start:end]) for start, end in chunk_batch_ranges(tensor.shape[0], self.max_batch_size)]
        if len(chunks) == 1:
            return chunks[0]
        keys = set(chunks[0])
        if any(set(chunk) != keys for chunk in chunks[1:]):
            raise RuntimeError("TensorRT output keys changed between dynamic-batch chunks.")
        return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in sorted(keys)}

    def infer_into(
        self,
        tensor: torch.Tensor,
        output_buffers: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Infer into caller-owned buffers, avoiding per-call output allocation.

        The batch must fit one optimization profile. The caller may reuse the
        returned buffers after consuming them on the same CUDA stream; the next
        invocation waits for that stream before overwriting the buffers.
        """

        self._validate_input(tensor)
        batch_size = int(tensor.shape[0])
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"TensorRT infer_into batch {batch_size} exceeds unchunked maximum {self.max_batch_size}."
            )
        engine_buffers = self._validate_output_buffers(batch_size, output_buffers)
        return self._infer_chunk(tensor, output_bindings=engine_buffers)

    def autotune_profiles(
        self,
        batch_size: int,
        *,
        warmup_iterations: int = 1,
        measure_iterations: int = 3,
    ) -> dict[str, Any]:
        """Benchmark compatible profiles on the private stream and select the fastest."""

        batch = _positive_int(batch_size, "TensorRT profile autotune batch size")
        warmups = int(warmup_iterations)
        measurements = _positive_int(measure_iterations, "TensorRT profile autotune iterations")
        if warmups < 0:
            raise ValueError("TensorRT profile autotune warmup iterations cannot be negative.")
        candidates = [
            index
            for index, shapes in enumerate(self.profile_shapes)
            if shapes[0][0] <= batch <= shapes[2][0]
        ]
        if not candidates:
            raise ValueError(f"No TensorRT optimization profile supports autotune batch size {batch}.")
        if len(candidates) == 1:
            self._selected_profile_index = candidates[0]
            return {
                "batch_size": batch,
                "selected_profile_index": candidates[0],
                "benchmarked": False,
                "profiles": [],
            }

        dummy = torch.zeros((batch, *self.max_shape[1:]), dtype=self.input_dtype, device=self.device)
        output_buffers = self.allocate_output_buffers(batch)
        reports: list[dict[str, Any]] = []
        for profile_index in candidates:
            self._selected_profile_index = profile_index
            for _ in range(warmups):
                self.infer_into(dummy, output_buffers)
                self.synchronize()
                self.consume_forward_seconds()
            durations: list[float] = []
            for _ in range(measurements):
                self.infer_into(dummy, output_buffers)
                self.synchronize()
                durations.append(self.consume_forward_seconds())
            reports.append(
                {
                    "profile_index": profile_index,
                    "opt_batch_size": self.profile_shapes[profile_index][1][0],
                    "median_forward_seconds": statistics.median(durations),
                    "measurements": durations,
                }
            )
        selected = min(reports, key=lambda report: report["median_forward_seconds"])
        self._selected_profile_index = int(selected["profile_index"])
        return {
            "batch_size": batch,
            "selected_profile_index": self._selected_profile_index,
            "benchmarked": True,
            "profiles": reports,
        }

    __call__ = infer

    def consume_forward_seconds(self) -> float:
        with self._execution_lock:
            seconds = self.forward_timing.consume_seconds()
            self._release_completed_bindings()
        return seconds

    def synchronize(self) -> None:
        """Wait only for this runner's private stream, never the whole device."""

        with self._execution_lock:
            self._stream.synchronize()
            self._release_completed_bindings()

    def warmup(self, batch_size: int = 1) -> float:
        batch_size = min(_positive_int(batch_size, "warmup batch size"), self.max_batch_size)
        started = time.perf_counter()
        dummy = torch.zeros((batch_size, *self.max_shape[1:]), dtype=self.input_dtype, device=self.device)
        self.infer(dummy)
        self.synchronize()
        return time.perf_counter() - started


class _TensorRTModelPlaceholder(torch.nn.Module):
    """Parameter-free placeholder preventing RF-DETR's device decorator reloading PyTorch weights."""

    def forward(self, _tensor: torch.Tensor) -> Any:  # pragma: no cover - defensive only
        raise RuntimeError("TensorRT adapter must run the installed inference_model, not the PyTorch placeholder.")


class TensorRTPredictAdapter:
    """Predict-compatible wrapper which reuses RF-DETR preprocessing/postprocessing."""

    def __init__(self, model: Any, runner: TensorRTRunner, *, release_pytorch_cuda: bool = True) -> None:
        self._model = model
        self.runner = runner
        context = getattr(model, "model", None)
        if context is None or not hasattr(context, "postprocess"):
            raise RuntimeError("RF-DETR model context with postprocess() is required.")
        if _ensure_postprocess_timing(model) is None:
            raise RuntimeError("RF-DETR model context with callable postprocess() is required.")
        original = getattr(context, "model", None)
        if original is None:
            raise RuntimeError("RF-DETR model context does not contain model weights.")
        if release_pytorch_cuda and hasattr(original, "to"):
            original = original.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._original_pytorch_model = original
        context.model = _TensorRTModelPlaceholder()
        context.inference_model = runner
        model._is_optimized_for_inference = True
        model._optimized_has_been_compiled = False
        model._optimized_batch_size = None
        model._optimized_resolution = runner.resolution
        model._optimized_dtype = runner.input_dtype
        setattr(model, _ACCELERATION_MARKER, f"tensorrt:{runner.engine_path}")

    @property
    def device(self) -> torch.device:
        return self.runner.device

    @property
    def model_config(self) -> Any:
        return self._model.model_config

    @property
    def class_names(self) -> Any:
        return self._model.class_names

    @property
    def model(self) -> Any:
        return self._model.model

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(*args, **kwargs)

    def infer_raw(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.runner.infer(tensor)

    def postprocess(self, outputs: Mapping[str, torch.Tensor], target_sizes: torch.Tensor) -> Any:
        callback = self._model.model.postprocess
        try:
            return callback(outputs, target_sizes=target_sizes)
        except TypeError:
            return callback(outputs, target_sizes)

    def restore_pytorch_model(self) -> Any:
        context = self._model.model
        context.model = self._original_pytorch_model
        context.inference_model = None
        self._model._is_optimized_for_inference = False
        self._model._optimized_has_been_compiled = False
        self._model._optimized_batch_size = None
        self._model._optimized_resolution = None
        self._model._optimized_dtype = None
        with contextlib.suppress(AttributeError):
            delattr(self._model, _ACCELERATION_MARKER)
        return self._model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def install_tensorrt_backend(
    model: Any,
    runner: TensorRTRunner,
    *,
    release_pytorch_cuda: bool = True,
) -> TensorRTPredictAdapter:
    """Install a TensorRT runner behind RF-DETR's existing ``predict`` path."""

    return TensorRTPredictAdapter(model, runner, release_pytorch_cuda=release_pytorch_cuda)


def run_inference_accuracy_parity_check(
    handle: AccelerationHandle,
    callback: Callable[[AccelerationHandle], bool | Mapping[str, Any]],
) -> dict[str, Any]:
    """Run a caller-supplied BF16-reference accuracy gate for a candidate backend.

    The acceleration layer cannot infer dataset mAP, class recall, or tracking
    tolerances from raw tensors. The callback owns those metrics and returns a
    bool or a mapping containing boolean ``accepted``/``passed`` plus details.
    Callback exceptions are represented as a failed gate so callers may safely
    fall back instead of deploying an unverified FP16 engine.
    """

    started = time.perf_counter()
    try:
        result = callback(handle)
        if isinstance(result, bool):
            accepted = result
            details: dict[str, Any] = {}
        elif isinstance(result, Mapping):
            details = dict(result)
            raw_accepted = details.get("accepted", details.get("passed"))
            if not isinstance(raw_accepted, bool):
                raise ValueError("Accuracy parity result must contain boolean 'accepted' or 'passed'.")
            accepted = raw_accepted
        else:
            raise TypeError("Accuracy parity callback must return bool or a mapping.")
        return {
            "performed": True,
            "reference_backend": "pytorch",
            "reference_precision": "bf16",
            "accepted": accepted,
            "seconds": time.perf_counter() - started,
            "details": _jsonable(details),
            "error": None,
        }
    except Exception as exc:
        return {
            "performed": True,
            "reference_backend": "pytorch",
            "reference_precision": "bf16",
            "accepted": False,
            "seconds": time.perf_counter() - started,
            "details": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def configure_inference_acceleration(
    model: Any,
    config_or_settings: Mapping[str, Any] | InferenceOptimizationConfig,
    *,
    batch_sizes: Iterable[int] = (),
    checkpoint_path: str | Path | None = None,
    model_identity: Mapping[str, Any] | None = None,
    segmentation: bool | None = None,
    device: str | torch.device | None = None,
    parity_check: Callable[[AccelerationHandle], bool | Mapping[str, Any]] | None = None,
    fallback_on_parity_failure: bool = True,
    reuse_output_buffers: bool | None = None,
) -> AccelerationHandle:
    """Fail-fast high-level configuration for PyTorch or TensorRT inference."""

    settings = (
        config_or_settings
        if isinstance(config_or_settings, InferenceOptimizationConfig)
        else resolve_acceleration_config(config_or_settings, batch_sizes=batch_sizes)
    )
    if settings.backend == "pytorch":
        return apply_pytorch_optimization(model, settings)
    if reuse_output_buffers is not None and not isinstance(reuse_output_buffers, bool):
        raise ValueError("reuse_output_buffers override must be true or false.")
    effective_reuse_output_buffers = (
        settings.tensorrt.reuse_output_buffers
        if reuse_output_buffers is None
        else reuse_output_buffers
    )

    runner_device = _preflight_device(device) if device is not None else _model_device(model)
    artifact = prepare_tensorrt_engine(
        model,
        settings,
        checkpoint_path=checkpoint_path,
        model_identity=model_identity,
        segmentation=segmentation,
    )
    # Cache hits skip export/build, so explicitly release the PyTorch CUDA copy
    # before deserializing the TensorRT engine as well.
    _release_pytorch_cuda_weights(model)
    load_started = time.perf_counter()
    runner = TensorRTRunner(
        artifact.engine_path,
        manifest_path=artifact.manifest_path,
        device=runner_device,
        expected_manifest=artifact.manifest,
    )
    load_seconds = time.perf_counter() - load_started
    autotune_started = time.perf_counter()
    autotune_batch = min(settings.tensorrt.profile.opt_batch_size, 16)
    profile_autotune = runner.autotune_profiles(autotune_batch)
    autotune_seconds = time.perf_counter() - autotune_started
    adapter = install_tensorrt_backend(model, runner)
    warmup_seconds = runner.warmup(1)
    runner.consume_forward_seconds()  # Warm-up is reported separately, never charged to the first image batch.
    metadata = _base_metadata(settings)
    metadata.update(
        {
            "cache_hit": artifact.cache_hit,
            "export_seconds": artifact.export_seconds,
            "build_seconds": artifact.build_seconds,
            "load_seconds": load_seconds,
            "profile_autotune_seconds": autotune_seconds,
            "profile_autotune": profile_autotune,
            "warmup_seconds": warmup_seconds,
            "engine_path": str(artifact.engine_path),
            "manifest_path": str(artifact.manifest_path),
            "engine_sha256": artifact.manifest.get("engine_sha256"),
            "onnx_path": str(artifact.onnx_path) if artifact.onnx_path is not None else None,
            "onnx_cache_key": artifact.manifest.get("onnx_cache_key"),
            "onnx_cache_hit": artifact.manifest.get("onnx_cache_hit"),
            "timing_cache_path": str(artifact.timing_cache_path) if artifact.timing_cache_path is not None else None,
            "gpu_identity": dict(artifact.manifest.get("identity", {}).get("gpu", {})),
            "accuracy_parity": {
                "performed": False,
                "reference_backend": "pytorch",
                "reference_precision": "bf16",
                "accepted": None,
            },
            "reuse_output_buffers": effective_reuse_output_buffers,
        }
    )
    handle = AccelerationHandle(
        model=adapter,
        settings=settings,
        metadata=metadata,
        artifact=artifact,
        _infer_raw=runner.infer,
        _forward_recorder=runner.forward_timing,
        _postprocess_recorder=getattr(model, "__dict__", {}).get(_POSTPROCESS_RECORDER_MARKER),
        _infer_into=runner.infer_into,
        _allocate_output_buffers=runner.allocate_output_buffers,
        _reuse_output_buffers_by_default=effective_reuse_output_buffers,
    )
    if parity_check is None:
        return handle

    parity = run_inference_accuracy_parity_check(handle, parity_check)
    handle.metadata["accuracy_parity"] = parity
    if bool(parity["accepted"]):
        return handle
    if not fallback_on_parity_failure:
        reason = parity.get("error") or parity.get("details") or "accuracy gate rejected TensorRT"
        raise RuntimeError(f"TensorRT FP16 accuracy parity failed: {reason}")

    restored = adapter.restore_pytorch_model()
    fallback_settings = InferenceOptimizationConfig(
        backend="pytorch",
        pytorch_precision="bf16",
        tensorrt=settings.tensorrt,
        resolution=settings.resolution,
    )
    fallback = apply_pytorch_optimization(restored, fallback_settings)
    fallback.metadata.update(
        {
            "requested_backend": "tensorrt",
            "requested_precision": settings.tensorrt.precision,
            "effective_backend": "pytorch",
            "effective_precision": "bf16",
            "fallback_reason": "TensorRT accuracy parity failed",
            "accuracy_parity": parity,
            "rejected_tensorrt": metadata,
        }
    )
    return fallback


__all__ = [
    "AccelerationHandle",
    "ForwardTimingRecorder",
    "InferenceOptimizationConfig",
    "PreprocessTiming",
    "PreparedInferenceBatch",
    "TensorRTArtifact",
    "TensorRTPredictAdapter",
    "TensorRTProfile",
    "TensorRTRunner",
    "TensorRTSettings",
    "apply_pytorch_optimization",
    "build_tensorrt_engine",
    "chunk_batch_ranges",
    "configure_inference_acceleration",
    "gpu_runtime_identity",
    "install_tensorrt_backend",
    "normalize_raw_outputs",
    "prepare_inference_batch",
    "preflight_inference_acceleration",
    "prepare_tensorrt_engine",
    "resolve_acceleration_config",
    "run_inference_accuracy_parity_check",
    "sha256_file",
    "tensorrt_optimization_profiles",
    "validate_engine_manifest",
]
