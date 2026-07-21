"""Shared RF-DETR trainer/test/inference runtime helpers.

This module is the stable import target for standalone entrypoints. The current
implementation re-exports the mature helper functions from the training module
so train, test, and inference do not import each other's entrypoint files.
"""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import rf_detr_acceleration as _acceleration
import train_rf_detr_model as _trainer_runtime
from train_rf_detr_model import *  # noqa: F401,F403

get_model_class = _trainer_runtime.get_model_class
build_model_kwargs = _trainer_runtime.build_model_kwargs
build_train_kwargs = _trainer_runtime.build_train_kwargs
build_pitchobjectlab_architecture = _trainer_runtime.build_pitchobjectlab_architecture


def _require_custom_architecture_checkpoint(config: Mapping[str, Any], operation: str) -> None:
    """Reject P2/TrackNetV5 test or inference without an explicit checkpoint."""
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        return
    p2 = model.get("p2", {})
    motion = model.get("motion", {})
    p2_enabled = bool(p2.get("enabled", False)) if isinstance(p2, Mapping) else False
    motion_enabled = bool(motion.get("enabled", False)) if isinstance(motion, Mapping) else False
    if not p2_enabled and not motion_enabled:
        return
    should_pass, checkpoint = _trainer_runtime.normalize_pretrain_weights(
        model.get("pretrain_weights", "default")
    )
    if should_pass and checkpoint not in {None, "default"}:
        return
    architecture = " + ".join(
        name for name, enabled in (("P2", p2_enabled), ("TrackNetV5", motion_enabled)) if enabled
    )
    raise ValueError(
        f"{operation} with the custom {architecture} architecture requires an explicit matching "
        "checkpoint via model.pretrain_weights or --checkpoint."
    )


def inference_acceleration_batch_sizes(config: Mapping[str, Any]) -> list[int]:
    """Collect active *model-call* batches for the current inference/test mode."""
    values: list[int] = []

    def add(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return
        if parsed > 0:
            values.append(parsed)

    test = config.get("test")
    if isinstance(test, Mapping):
        test_mode = test.get("test_mode", config.get("test_mode", {}))
        mode = str(test_mode.get("mode", "full_image") if isinstance(test_mode, Mapping) else "full_image").lower()
        if mode == "sahi":
            sahi = test.get("sahi", config.get("sahi", {}))
            if isinstance(sahi, Mapping):
                # Slice, optional standard prediction, and recheck all use this
                # actual RF-DETR batch; test.batch_size is only the outer group.
                add(sahi.get("batch_size") or test.get("batch_size"))
        else:
            add(test.get("batch_size"))
    else:
        inference = config.get("inference", {})
        if isinstance(inference, Mapping):
            mode = str(inference.get("mode", "full_image")).strip().lower()
            if mode == "sahi":
                sahi = config.get("sahi", {})
                if isinstance(sahi, Mapping):
                    add(sahi.get("batch_size") or inference.get("batch_size"))
            else:
                add(inference.get("batch_size"))
                video = inference.get("video", {})
                if isinstance(video, Mapping):
                    add(video.get("batch_size"))
    return list(dict.fromkeys(values)) or [1]


def inference_acceleration_identity(model: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the stable model identity used by TensorRT manifests/cache keys."""
    model_settings = config.get("model", {})
    if not isinstance(model_settings, Mapping):
        model_settings = {}
    model_config = getattr(model, "model_config", None)
    return {
        "size": model_settings.get("size", "medium"),
        "resolution": getattr(model_config, "resolution", model_settings.get("resolution")),
        "num_classes": getattr(model_config, "num_classes", model_settings.get("num_classes")),
        "segmentation_head": bool(getattr(model_config, "segmentation_head", False)),
        "p2": deepcopy(dict(model_settings.get("p2", {}) or {})),
        "motion": deepcopy(dict(model_settings.get("motion", {}) or {})),
        "extra_model_args": deepcopy(dict(model_settings.get("extra_model_args", {}) or {})),
    }


def validate_inference_acceleration_config(
    config: Mapping[str, Any],
    *,
    resolution: int | None = None,
) -> Any:
    """Resolve and validate inference optimization settings without loading a model."""
    if resolution is None:
        model_settings = config.get("model", {})
        if isinstance(model_settings, Mapping):
            configured_resolution = model_settings.get("resolution")
            if isinstance(configured_resolution, int) and not isinstance(configured_resolution, bool):
                resolution = configured_resolution
    return _acceleration.resolve_acceleration_config(
        config,
        batch_sizes=inference_acceleration_batch_sizes(config),
        resolution=resolution,
    )


def preflight_rfdetr_inference_acceleration(
    config: Mapping[str, Any],
    *,
    device: Any = None,
) -> dict[str, Any]:
    """Validate acceleration dependencies, GPU support, and supplied artifacts."""
    settings = validate_inference_acceleration_config(config)
    normalized_device = _trainer_runtime.normalize_model_constructor_device(device)
    return _acceleration.preflight_inference_acceleration(settings, device=normalized_device)


def configure_rfdetr_inference_acceleration(
    model: Any,
    config: Mapping[str, Any],
    *,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Apply the configured backend and attach its handle to the returned predictor."""
    model_settings = config.get("model", {})
    if not isinstance(model_settings, Mapping):
        model_settings = {}
    model_config = getattr(model, "model_config", None)
    runtime_device = _trainer_runtime.normalize_model_constructor_device(
        device if device is not None else model_settings.get("device")
    )
    handle = _acceleration.configure_inference_acceleration(
        model,
        config,
        batch_sizes=inference_acceleration_batch_sizes(config),
        checkpoint_path=model_settings.get("pretrain_weights"),
        model_identity=inference_acceleration_identity(model, config),
        segmentation=bool(getattr(model_config, "segmentation_head", False)),
        # Resolve integer/"auto" shortcuts before torch.device() sees them.
        # None intentionally lets the runtime inspect the already-built model.
        device=runtime_device,
    )
    accelerated_model = handle.model
    setattr(accelerated_model, "_rf_detr_acceleration_handle", handle)
    return accelerated_model, handle


def get_inference_acceleration_handle(model: Any) -> Any:
    """Return the acceleration handle installed by the shared runtime."""
    handle = getattr(model, "_rf_detr_acceleration_handle", None)
    if handle is None:
        raise RuntimeError("RF-DETR model does not have an installed inference acceleration handle.")
    return handle


def estimate_tensorrt_cache_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate first-run TensorRT cache writes for the pre-run confirmation."""
    model_settings = config.get("model", {})
    if not isinstance(model_settings, Mapping):
        model_settings = {}
    optimization = model_settings.get("inference_optimization", {})
    if not isinstance(optimization, Mapping) or str(optimization.get("backend", "pytorch")).lower() != "tensorrt":
        return {
            "file_count": 0,
            "bytes": 0,
            "cache_dir": None,
            "build_required": False,
            "estimated_build_seconds": 0,
            "estimated_build_hms": "00:00:00",
        }
    tensorrt = optimization.get("tensorrt", {})
    if not isinstance(tensorrt, Mapping):
        tensorrt = {}
    engine_path = str(tensorrt.get("engine_path") or "").strip()
    if engine_path:
        return {
            "file_count": 0,
            "bytes": 0,
            "cache_dir": None,
            "build_required": False,
            "provided_engine": engine_path,
            "estimated_build_seconds": 0,
            "estimated_build_hms": "00:00:00",
        }
    checkpoint = model_settings.get("pretrain_weights")
    checkpoint_bytes = 0
    if checkpoint:
        candidate = Path(str(checkpoint)).expanduser()
        if candidate.is_file():
            checkpoint_bytes = int(candidate.stat().st_size)
    # ONNX + engine + manifest + serialized TensorRT timing cache.  A 1 GiB
    # floor keeps the confirmation conservative for hosted/default weights.
    estimated_bytes = max(1_000_000_000, checkpoint_bytes * 3)
    # TensorRT tactic selection varies substantially by model/GPU. Ten minutes
    # is an intentionally rough first-run allowance shown before confirmation;
    # actual export/build time is recorded separately in run_timing.json.
    estimated_build_seconds = 600
    return {
        "file_count": 4,
        "bytes": estimated_bytes,
        "cache_dir": str(_acceleration.resolve_tensorrt_cache_dir(tensorrt.get("cache_dir"))),
        "build_required": True,
        "estimated_build_seconds": estimated_build_seconds,
        "estimated_build_hms": _trainer_runtime.format_duration_hms(estimated_build_seconds),
    }


def _prepare_parallel_tensorrt_profiles(
    config: Mapping[str, Any],
    output_dir: str,
    devices: list[str],
) -> dict[str, Any]:
    """Spawn-target which prepares one engine per unique GPU compatibility profile."""
    import torch

    settings = validate_inference_acceleration_config(config)
    if settings.backend != "tensorrt":
        raise ValueError("Parallel TensorRT preparation requires backend=tensorrt.")

    profile_groups: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}
    for requested_device in devices:
        preflight = preflight_rfdetr_inference_acceleration(config, device=requested_device)
        resolved_device = torch.device(preflight["device"])
        if resolved_device.type != "cuda":
            raise RuntimeError(
                f"TensorRT preparation requires a CUDA device; got {resolved_device} "
                f"for requested device {requested_device!r}."
            )
        properties = torch.cuda.get_device_properties(resolved_device)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(resolved_device))
        key = (str(properties.name), capability)
        group = profile_groups.setdefault(
            key,
            {
                "representative_device": str(resolved_device),
                "requested_devices": [],
                "resolved_devices": [],
                "gpu_name": str(properties.name),
                "compute_capability": list(capability),
                "preflight": preflight,
            },
        )
        group["requested_devices"].append(str(requested_device))
        group["resolved_devices"].append(str(resolved_device))

    profiles: list[dict[str, Any]] = []
    device_artifacts: dict[str, dict[str, Any]] = {}
    for group in profile_groups.values():
        rf_model, _ = build_rfdetr_evaluator_runtime(
            config,
            Path(output_dir),
            device=str(group["representative_device"]),
        )
        handle = get_inference_acceleration_handle(rf_model)
        metadata = dict(handle.metadata)
        engine_path = metadata.get("engine_path")
        manifest_path = metadata.get("manifest_path")
        if not engine_path or not manifest_path:
            raise RuntimeError("Prepared TensorRT runtime did not publish engine and manifest paths.")
        artifact = {
            "engine_path": str(engine_path),
            "manifest_path": str(manifest_path),
            "force_rebuild": False,
        }
        aliases = set(group["requested_devices"]) | set(group["resolved_devices"])
        for alias in aliases:
            device_artifacts[str(alias)] = dict(artifact)
            normalized_alias = _trainer_runtime.normalize_model_constructor_device(alias)
            if normalized_alias is not None:
                device_artifacts[str(normalized_alias)] = dict(artifact)
        profiles.append(
            {
                **group,
                "metadata": metadata,
            }
        )

    return {
        "requested_backend": "tensorrt",
        "effective_backend": "tensorrt",
        "requested_precision": settings.precision,
        "effective_precision": settings.precision,
        "cache_hit": all(bool(item["metadata"].get("cache_hit")) for item in profiles),
        "build_seconds": sum(float(item["metadata"].get("build_seconds", 0.0) or 0.0) for item in profiles),
        "export_seconds": sum(float(item["metadata"].get("export_seconds", 0.0) or 0.0) for item in profiles),
        "load_seconds": sum(float(item["metadata"].get("load_seconds", 0.0) or 0.0) for item in profiles),
        "warmup_seconds": sum(float(item["metadata"].get("warmup_seconds", 0.0) or 0.0) for item in profiles),
        "profiles": profiles,
        "device_artifacts": device_artifacts,
    }


def prepare_parallel_tensorrt_artifacts(
    config: Mapping[str, Any],
    output_dir: Path,
    devices: list[str],
) -> dict[str, Any]:
    """Prepare TensorRT artifacts in one dedicated spawn subprocess."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    settings = validate_inference_acceleration_config(config)
    if settings.backend != "tensorrt":
        raise ValueError("Parallel TensorRT preparation requires backend=tensorrt.")
    requested_devices = list(dict.fromkeys(str(device) for device in devices))
    if not requested_devices:
        requested_devices = ["auto"]
    spawn_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn_context) as executor:
        future = executor.submit(
            _prepare_parallel_tensorrt_profiles,
            deepcopy(dict(config)),
            str(Path(output_dir)),
            requested_devices,
        )
        return dict(future.result())


def build_rfdetr_evaluator_runtime(
    merged_source: Mapping[str, Any],
    output_dir: Path,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Construct an RF-DETR model and train config for standalone evaluation."""
    if not isinstance(merged_source, Mapping):
        raise TypeError("RF-DETR evaluator config must be a mapping.")
    merged_config = deepcopy(dict(merged_source))
    model_source = merged_config.get("model", {})
    train_source = merged_config.get("train", {})
    if not isinstance(model_source, Mapping) or not isinstance(train_source, Mapping):
        raise ValueError("RF-DETR evaluator model/train sections must be mappings.")
    model_settings = dict(model_source)
    train_settings = dict(train_source)
    merged_config["model"] = model_settings
    merged_config["train"] = train_settings
    if device is not None:
        assigned_device = str(device).strip()
        if not assigned_device:
            raise ValueError("Parallel RF-DETR worker device must be non-empty.")
        model_settings["device"] = assigned_device
        train_settings["device"] = assigned_device

    output_dir = Path(output_dir).expanduser()
    model_cls = get_model_class(str(model_settings.get("size", "medium")))
    rf_model = model_cls(**build_model_kwargs(merged_config))
    p2_config = model_settings.get('p2', {}) or {}
    if bool(p2_config.get('enabled', False)):
        from rf_detr_p2 import assert_p2_checkpoint_compatible

        assert_p2_checkpoint_compatible(
            rf_model.model,
            getattr(rf_model.model_config, 'pretrain_weights', None),
            build_pitchobjectlab_architecture(merged_config),
        )

    motion_config = model_settings.get("motion", {}) or {}
    if bool(motion_config.get("enabled", False)):
        from rf_detr_motion import attach_motion_module

        attach_motion_module(rf_model.model, motion_config)
        model_config = getattr(rf_model, "model_config", None)
        checkpoint_path = getattr(model_config, "pretrain_weights", None)
        from rf_detr_motion import assert_motion_checkpoint_compatible

        assert_motion_checkpoint_compatible(
            rf_model.model,
            checkpoint_path,
            build_pitchobjectlab_architecture(merged_config),
        )

    train_kwargs = build_train_kwargs(merged_config, output_dir)
    train_kwargs.pop("_device", None)
    train_config = rf_model.get_train_config(**train_kwargs)
    rf_model._align_num_classes_from_dataset(train_config.dataset_dir)
    rf_model, _ = configure_rfdetr_inference_acceleration(
        rf_model,
        merged_config,
        device=device if device is not None else model_settings.get("device"),
    )
    return rf_model, train_config


def build_rfdetr_evaluator_model(model_cfg: Mapping[str, Any], device: str) -> Any:
    """Construct one RF-DETR replica inside a spawned evaluator worker."""
    if not isinstance(model_cfg, Mapping):
        raise TypeError("Parallel RF-DETR model config must be a mapping.")
    factory_config = model_cfg.get("factory_config")
    if not isinstance(factory_config, Mapping):
        raise ValueError("model.factory_config must contain merged_config and output_dir.")
    merged_source = factory_config.get("merged_config")
    if not isinstance(merged_source, Mapping):
        raise ValueError("model.factory_config.merged_config must be a mapping.")
    merged_source = deepcopy(dict(merged_source))
    prepared_artifacts = factory_config.get("prepared_tensorrt")
    if prepared_artifacts is not None:
        if not isinstance(prepared_artifacts, Mapping):
            raise ValueError("model.factory_config.prepared_tensorrt must be a mapping.")
        normalized_device = _trainer_runtime.normalize_model_constructor_device(device)
        candidates = [str(device)]
        if normalized_device is not None:
            candidates.append(str(normalized_device))
        prepared = next(
            (prepared_artifacts[key] for key in candidates if key in prepared_artifacts),
            None,
        )
        if not isinstance(prepared, Mapping):
            raise RuntimeError(
                f"No prepared TensorRT artifact is available for worker device {device!r}; "
                "parallel workers are not allowed to rebuild engines."
            )
        optimization = merged_source.setdefault("model", {}).setdefault("inference_optimization", {})
        if str(optimization.get("backend", "pytorch")).strip().lower() != "tensorrt":
            raise RuntimeError("Prepared TensorRT artifacts require backend=tensorrt in worker config.")
        tensorrt = optimization.setdefault("tensorrt", {})
        tensorrt["engine_path"] = str(prepared.get("engine_path") or "")
        tensorrt["manifest_path"] = str(prepared.get("manifest_path") or "")
        tensorrt["force_rebuild"] = False
    output_dir_value = factory_config.get("output_dir")
    if output_dir_value is None or not str(output_dir_value).strip():
        raise ValueError("model.factory_config.output_dir must be a non-empty path.")
    rf_model, _ = build_rfdetr_evaluator_runtime(
        merged_source,
        Path(str(output_dir_value)),
        device=device,
    )
    return rf_model
