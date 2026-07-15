"""Shared RF-DETR trainer/test/inference runtime helpers.

This module is the stable import target for standalone entrypoints. The current
implementation re-exports the mature helper functions from the training module
so train, test, and inference do not import each other's entrypoint files.
"""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import train_rf_detr_model as _trainer_runtime
from train_rf_detr_model import *  # noqa: F401,F403

get_model_class = _trainer_runtime.get_model_class
build_model_kwargs = _trainer_runtime.build_model_kwargs
build_train_kwargs = _trainer_runtime.build_train_kwargs


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

    motion_config = model_settings.get("motion", {}) or {}
    if bool(motion_config.get("enabled", False)):
        from rf_detr_motion import attach_motion_module

        attach_motion_module(rf_model.model, motion_config)

    train_kwargs = build_train_kwargs(merged_config, output_dir)
    train_kwargs.pop("_device", None)
    train_config = rf_model.get_train_config(**train_kwargs)
    rf_model._align_num_classes_from_dataset(train_config.dataset_dir)
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
    output_dir_value = factory_config.get("output_dir")
    if output_dir_value is None or not str(output_dir_value).strip():
        raise ValueError("model.factory_config.output_dir must be a non-empty path.")
    rf_model, _ = build_rfdetr_evaluator_runtime(
        merged_source,
        Path(str(output_dir_value)),
        device=device,
    )
    return rf_model
