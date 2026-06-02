"""Translate wrapper augmentation settings into D-FINE-seg runtime config values."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def _float(section: Mapping[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    if value is None:
        return default
    return float(value)


def _preset_defaults(preset: str) -> dict[str, float]:
    """Return advanced Albumentations probabilities for a preset."""
    preset = str(preset or "default").lower()
    presets = {
        "none": {
            "rotation_p": 0.0,
            "rotate_90": 0.0,
            "to_gray": 0.0,
            "blur": 0.0,
            "motion_blur": 0.0,
            "gamma": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "noise": 0.0,
            "iso_noise": 0.0,
            "clahe": 0.0,
            "sharpen": 0.0,
            "compression": 0.0,
            "coarse_dropout": 0.0,
            "grid_dropout": 0.0,
            "random_shadow": 0.0,
            "random_weather": 0.0,
            "downscale": 0.0,
        },
        "light": {
            "rotation_p": 0.03,
            "rotate_90": 0.01,
            "to_gray": 0.005,
            "blur": 0.005,
            "motion_blur": 0.003,
            "gamma": 0.02,
            "brightness": 0.03,
            "contrast": 0.03,
            "noise": 0.005,
            "iso_noise": 0.003,
            "clahe": 0.005,
            "sharpen": 0.005,
            "compression": 0.005,
            "coarse_dropout": 0.0,
            "grid_dropout": 0.0,
            "random_shadow": 0.0,
            "random_weather": 0.0,
            "downscale": 0.0,
        },
        "default": {
            "rotation_p": 0.05,
            "rotate_90": 0.03,
            "to_gray": 0.01,
            "blur": 0.01,
            "motion_blur": 0.01,
            "gamma": 0.03,
            "brightness": 0.04,
            "contrast": 0.04,
            "noise": 0.01,
            "iso_noise": 0.01,
            "clahe": 0.01,
            "sharpen": 0.01,
            "compression": 0.01,
            "coarse_dropout": 0.02,
            "grid_dropout": 0.01,
            "random_shadow": 0.01,
            "random_weather": 0.005,
            "downscale": 0.005,
        },
        "heavy": {
            "rotation_p": 0.1,
            "rotate_90": 0.08,
            "to_gray": 0.03,
            "blur": 0.03,
            "motion_blur": 0.03,
            "gamma": 0.08,
            "brightness": 0.08,
            "contrast": 0.08,
            "noise": 0.03,
            "iso_noise": 0.03,
            "clahe": 0.03,
            "sharpen": 0.03,
            "compression": 0.03,
            "coarse_dropout": 0.08,
            "grid_dropout": 0.04,
            "random_shadow": 0.03,
            "random_weather": 0.02,
            "downscale": 0.02,
        },
        "ultralytics_like": {
            "rotation_p": 0.05,
            "rotate_90": 0.0,
            "to_gray": 0.01,
            "blur": 0.01,
            "motion_blur": 0.01,
            "gamma": 0.02,
            "brightness": 0.04,
            "contrast": 0.04,
            "noise": 0.01,
            "iso_noise": 0.01,
            "clahe": 0.01,
            "sharpen": 0.01,
            "compression": 0.01,
            "coarse_dropout": 0.02,
            "grid_dropout": 0.0,
            "random_shadow": 0.01,
            "random_weather": 0.0,
            "downscale": 0.0,
        },
        "custom": {},
    }
    if preset not in presets:
        raise ValueError("augmentation.preset must be one of none, light, default, heavy, ultralytics_like, custom.")
    return deepcopy(presets[preset])


def build_aug_runtime(augmentation: Mapping[str, Any], task: str) -> dict[str, Any]:
    """Build the D-FINE ``train.mosaic_augs`` and ``train.augs`` sections.

    D-FINE-seg natively consumes the classic mosaic and Albumentations keys. This wrapper also persists
    Ultralytics-style blend augmentation knobs in the runtime config; the vendored snapshot reads the extra keys when
    patched and otherwise ignores them without breaking older upstream code.
    """
    augmentation = augmentation or {}
    preset_values = _preset_defaults(str(augmentation.get("preset", "default")))
    advanced = dict(preset_values)
    advanced.update(augmentation.get("advanced", {}) or {})

    hsv_h = _float(augmentation, "hsv_h", 0.015)
    hsv_s = _float(augmentation, "hsv_s", 0.7)
    hsv_v = _float(augmentation, "hsv_v", 0.4)

    mosaic_augs = {
        "mosaic_prob": _float(augmentation, "mosaic", 0.8),
        "no_mosaic_epochs": int(augmentation.get("close_mosaic", 5) or 0),
        "mosaic_scale": augmentation.get(
            "mosaic_scale", [1.0 - _float(augmentation, "scale", 0.5), 1.0 + _float(augmentation, "scale", 0.5)]
        ),
        "degrees": _float(augmentation, "degrees", 0.0),
        "translate": _float(augmentation, "translate", 0.1),
        "shear": _float(augmentation, "shear", 0.0),
    }

    augs = {
        "rotation_degree": abs(_float(augmentation, "degrees", 10.0)),
        "rotation_p": _float(advanced, "rotation_p", 0.05),
        "multiscale_prob": _float(augmentation, "multi_scale", 0.0),
        "rotate_90": _float(advanced, "rotate_90", 0.03),
        "left_right_flip": _float(augmentation, "fliplr", 0.5),
        "up_down_flip": _float(augmentation, "flipud", 0.0),
        "to_gray": _float(advanced, "to_gray", 0.01),
        "blur": _float(advanced, "blur", 0.01),
        "motion_blur": _float(advanced, "motion_blur", 0.01),
        "gamma": _float(advanced, "gamma", 0.03),
        "brightness": max(_float(advanced, "brightness", 0.04), min(hsv_v, 1.0) * 0.05),
        "contrast": _float(advanced, "contrast", 0.04),
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "noise": _float(advanced, "noise", 0.01),
        "iso_noise": _float(advanced, "iso_noise", 0.01),
        "clahe": _float(advanced, "clahe", 0.01),
        "sharpen": _float(advanced, "sharpen", 0.01),
        "compression": _float(advanced, "compression", 0.01),
        "coarse_dropout": _float(advanced, "coarse_dropout", 0.02),
        "grid_dropout": _float(advanced, "grid_dropout", 0.01),
        "random_shadow": _float(advanced, "random_shadow", 0.01),
        "random_weather": _float(advanced, "random_weather", 0.005),
        "downscale": _float(advanced, "downscale", 0.005),
        "bgr": _float(augmentation, "bgr", 0.0),
        "mixup": _float(augmentation, "mixup", 0.0),
        "cutmix": _float(augmentation, "cutmix", 0.0),
        "copy_paste": _float(augmentation, "copy_paste", 0.0) if task == "segment" else 0.0,
        "copy_paste_mode": str(augmentation.get("copy_paste_mode", "mixup")),
        "perspective": _float(augmentation, "perspective", 0.0),
        "scale": _float(augmentation, "scale", 0.5),
    }
    return {"mosaic_augs": mosaic_augs, "augs": augs}
