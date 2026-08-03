"""Lightweight YAML configuration helpers shared by RF-DETR entrypoints.

This module intentionally avoids importing numerical libraries so it can be
used during process bootstrap, before PyTorch/OpenMP thread pools initialize.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


LEGACY_TRACKNET_CONFIG_ALIASES = {
    "rf_detr_train_motion_v5_medium.yaml": "rf_detr_train_medium_p2_tracknet_v5.yaml",
    "rf_detr_train_motion_v5_medium_no-p2.yaml": "rf_detr_train_medium_tracknet_v5.yaml",
    "rf_detr_train_motion_v5_large.yaml": "rf_detr_train_large_p2_tracknet_v5.yaml",
    "rf_detr_train_motion_v5_large_no-p2.yaml": "rf_detr_train_large_tracknet_v5.yaml",
    "rf_detr_train_motion_v5_large_v2.yaml": "rf_detr_train_large_p2_tracknet_v5.yaml",
    "rf_detr_train_smoke_motion_v5_medium.yaml": "rf_detr_train_smoke_temporal_tracknet_v5.yaml",
    "rf_detr_train_smoke_motion_v5_p2_medium.yaml": "rf_detr_train_smoke_temporal_tracknet_v5.yaml",
}


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge dictionaries in-place."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path, _seen: Optional[set[Path]] = None) -> Dict[str, Any]:
    """Load a YAML mapping, resolving an optional relative ``extends`` chain."""
    path = path.expanduser().resolve()
    if _seen is None and path.name in LEGACY_TRACKNET_CONFIG_ALIASES:
        warnings.warn(
            f"{path.name} is deprecated; use "
            f"{LEGACY_TRACKNET_CONFIG_ALIASES[path.name]} instead.",
            FutureWarning,
            stacklevel=2,
        )
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"Config extends cycle detected: {chain}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    parent = data.pop("extends", None)
    if parent not in (None, ""):
        parent_path = Path(str(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = (path.parent / parent_path).resolve()
        base = load_yaml(parent_path, seen)
        return dict(deep_update(base, data))
    return data
