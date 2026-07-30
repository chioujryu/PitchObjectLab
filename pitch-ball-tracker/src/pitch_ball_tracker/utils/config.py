from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: dict | None = None) -> DictConfig:
    """Load a YAML config, optionally merging a second defaults block first.

    If the file contains a `defaults` key listing another config name, that base config is loaded from the same
    directory and merged first.
    """
    path = Path(config_path)
    cfg: DictConfig = OmegaConf.load(path)

    # Resolve `defaults` inheritance (single level, like Hydra lite)
    if "defaults" in cfg:
        base_name = cfg.defaults[0] if isinstance(cfg.defaults, list) else cfg.defaults
        base_path = path.parent / f"{base_name}.yaml"
        if base_path.exists():
            base_cfg = OmegaConf.load(base_path)
            cfg = OmegaConf.merge(base_cfg, cfg)
        OmegaConf.update(cfg, "defaults", None, merge=False)

    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.set_readonly(cfg, True)
    return cfg
