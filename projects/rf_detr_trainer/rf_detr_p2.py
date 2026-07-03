"""Pluggable, config-toggled "P2" (stride-4) feature level for RF-DETR.

RF-DETR has no native P2 level: its multi-scale pyramid bottoms out at P3/P4/P5
(stride 8/16/32). For extremely small objects (e.g. footballs a few pixels wide) a
P2 level (stride 4, scale-factor x4) is the most on-point structural fix.

The *installed* ``rfdetr`` package is already 95% ready for P2:

* ``MultiScaleProjector`` already implements the ``scale == 4.0`` branch
  (two stride-2 ``ConvTranspose2d`` layers) -- see
  ``rfdetr/models/backbone/projector.py``.
* The deformable-attention decoder sizes itself from
  ``num_feature_levels = len(projector_scale)``; ``MSDeformAttn`` Linear shapes,
  ``level_embed``, two-stage top-K query selection, ``spatial_shapes`` and the
  per-level sine positional encoding all derive dynamically from the level count
  and per-map size. An extra level "just works" structurally.
* Pretrained weights load with ``strict=False`` + PE interpolation. But the stock
  detection checkpoints are single-scale (``projector_scale=["P4"]`` for every size,
  nano..large), so enabling P2 changes ``num_feature_levels`` and *resizes* existing
  deformable-attention ``sampling_offsets``/``attention_weights`` Linear layers plus the
  projector's first stage. ``torch``'s ``strict=False`` only skips missing/extra *keys*,
  not size-mismatched ones, so the loader would raise ``RuntimeError``. We therefore drop
  the mismatched checkpoint tensors so they start randomly initialised -- exactly the
  documented fine-tuning behavior.

Three spots block P2, and all live in ``.venv`` (not version-controlled):

1. ``rfdetr.config.ModelConfig.projector_scale`` is
   ``List[Literal["P3", "P4", "P5"]]`` (each variant re-declares its own, even
   narrower, Literal -- e.g. ``RFDETRLargeConfig`` is ``List[Literal["P4",]]``).
   Pydantic rejects ``"P2"``.
2. ``Backbone.__init__`` builds a *local* dict
   ``level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)`` with no ``P2`` key
   -> ``KeyError`` when ``"P2"`` is requested.
3. ``rfdetr.models.weights.load_pretrain_weights`` ends in
   ``nn_model.load_state_dict(ckpt, strict=False)``; ``strict=False`` does NOT skip
   size-mismatched tensors, so the P2 feature-level-count change makes it raise. We
   override ``LWDETR.load_state_dict`` to drop those tensors on non-strict loads.

Rather than edit ``.venv`` (lost on ``uv sync``), this module applies an in-process
monkey-patch, version-controlled here and applied *only* when ``model.p2.enabled``
is true. It is the single place that reaches into ``rfdetr`` internals.

Public surface:
    ensure_p2_support(p2_config)        -- idempotent; relax Literal + patch backbone.
    resolve_p2_projector_scale(p2_cfg)  -- validated scale list (default [P2,P3,P4]).
    apply_p2_overrides(kwargs, p2_cfg)  -- merge p2.overrides into model kwargs.
    P2_SETTINGS                         -- projector internals read by patched backbone.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Mapping, MutableMapping, Optional

__all__ = [
    "P2_SETTINGS",
    "ensure_p2_support",
    "resolve_p2_projector_scale",
    "apply_p2_overrides",
    "is_patched",
]

# Version of rfdetr the Backbone.__init__ copy below was written against. A mismatch
# only warns (the copy may have drifted from upstream); P2 still attempts to apply.
_BASELINE_RFDETR_VERSION = "1.8.1"

# Valid projector levels in ascending P-name (= descending scale-factor / resolution) order.
# name -> (stride, scale-factor): P2->(4, 4.0) P3->(8, 2.0) P4->(16, 1.0) P5->(32, 0.5) P6->(64, 0.25)
_VALID_SCALES = ("P2", "P3", "P4", "P5", "P6")
_DEFAULT_PROJECTOR_SCALE = ["P2", "P3", "P4"]

# Real ModelConfig fields that p2.overrides may set, with their coercion.
_OVERRIDE_CASTS = {
    "resolution": int,
    "num_queries": int,
    "num_select": int,
    "dec_n_points": int,
    "gradient_checkpointing": bool,
}

# Projector internals threaded to the patched Backbone (NOT ModelConfig fields, because
# rfdetr.config.BaseConfig rejects unknown constructor kwargs). None => use upstream default.
P2_SETTINGS: Dict[str, Any] = {
    "num_blocks": None,
    "survival_prob": None,
    "force_drop_last_n_features": None,
    "layer_norm": None,
    "rms_norm": None,
}

# Upstream MultiScaleProjector defaults (rfdetr 1.8.1) used when a setting is None.
_PROJECTOR_DEFAULTS = {
    "num_blocks": 3,
    "survival_prob": 1.0,
    "force_drop_last_n_features": 0,
}

_LITERAL_PATCHED = False
_BACKBONE_PATCHED = False
_PRETRAIN_FILTER_PATCHED = False
_WARN_FILTER_INSTALLED = False
_PATCHED_INIT: Any = None  # the installed _p2_backbone_init, for idempotency checks/tests


def is_patched() -> bool:
    """Return True once the rfdetr Literal + backbone + loader patches have been applied."""
    return _LITERAL_PATCHED and _BACKBONE_PATCHED and _PRETRAIN_FILTER_PATCHED


def _first(value: Any, fallback: Any) -> Any:
    """Return value unless it is None, in which case the fallback."""
    return fallback if value is None else value


# --------------------------------------------------------------------------------------
# Config helpers (pure; safe to call whether or not P2 is enabled).
# --------------------------------------------------------------------------------------
def resolve_p2_projector_scale(p2_config: Optional[Mapping[str, Any]]) -> List[str]:
    """Validate and normalize model.p2.projector_scale (default [P2, P3, P4])."""
    raw = (p2_config or {}).get("projector_scale") or _DEFAULT_PROJECTOR_SCALE
    if isinstance(raw, str):
        raw = [raw]
    scales = [str(item).strip().upper() for item in raw]
    if not scales:
        raise ValueError("model.p2.projector_scale must be a non-empty list of level names.")
    for level in scales:
        if level not in _VALID_SCALES:
            raise ValueError(
                f"model.p2.projector_scale has invalid level {level!r}; allowed: {list(_VALID_SCALES)}."
            )
    if sorted(scales) != scales:
        raise ValueError(
            f"model.p2.projector_scale must be in ascending P-name order, got {scales}. "
            "Example: [P2, P3, P4]."
        )
    if "P2" not in scales:
        warnings.warn(
            f"[rf_detr_p2] projector_scale={scales} does not include 'P2'; the P2 level will "
            "not be used. Set model.p2.enabled: false or add 'P2' to projector_scale.",
            stacklevel=2,
        )
    return scales


def apply_p2_overrides(
    kwargs: MutableMapping[str, Any],
    p2_config: Optional[Mapping[str, Any]],
) -> MutableMapping[str, Any]:
    """Merge model.p2.overrides (real ModelConfig fields) into model kwargs in place.

    Only non-null overrides are applied; they take precedence over values already
    placed in kwargs from the model.* section.
    """
    overrides = (p2_config or {}).get("overrides", {}) or {}
    for key, cast in _OVERRIDE_CASTS.items():
        value = overrides.get(key)
        if value is not None:
            kwargs[key] = cast(value)
    return kwargs


def _record_settings(p2_config: Optional[Mapping[str, Any]]) -> None:
    """Capture projector internals from model.p2.projector for the patched backbone."""
    projector = (p2_config or {}).get("projector", {}) or {}
    for key in P2_SETTINGS:
        P2_SETTINGS[key] = projector.get(key)


def _build_projector_kwargs(layer_norm_arg: bool, rms_norm_arg: bool) -> Dict[str, Any]:
    """Resolve MultiScaleProjector kwargs from P2_SETTINGS + backbone-provided norms."""
    return {
        "num_blocks": _first(P2_SETTINGS.get("num_blocks"), _PROJECTOR_DEFAULTS["num_blocks"]),
        "survival_prob": _first(P2_SETTINGS.get("survival_prob"), _PROJECTOR_DEFAULTS["survival_prob"]),
        "force_drop_last_n_features": _first(
            P2_SETTINGS.get("force_drop_last_n_features"),
            _PROJECTOR_DEFAULTS["force_drop_last_n_features"],
        ),
        "layer_norm": _first(P2_SETTINGS.get("layer_norm"), layer_norm_arg),
        "rms_norm": _first(P2_SETTINGS.get("rms_norm"), rms_norm_arg),
    }


# --------------------------------------------------------------------------------------
# Patches.
# --------------------------------------------------------------------------------------
def _all_subclasses(cls: type) -> List[type]:
    """Return every (transitive) subclass of cls."""
    seen: set = set()
    out: List[type] = []
    stack = [cls]
    while stack:
        current = stack.pop()
        for sub in current.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                out.append(sub)
                stack.append(sub)
    return out


def _check_version() -> None:
    """Warn (do not fail) when the installed rfdetr differs from the patched baseline."""
    try:
        import importlib.metadata as metadata

        version = metadata.version("rfdetr")
    except Exception:  # pragma: no cover - metadata lookup is best-effort
        version = "unknown"
    if version != _BASELINE_RFDETR_VERSION:
        warnings.warn(
            f"[rf_detr_p2] P2 patch was written against rfdetr=={_BASELINE_RFDETR_VERSION} but "
            f"found {version}. The Backbone.__init__ copy may be stale; re-check "
            "rfdetr/models/backbone/backbone.py and projector.py if model building fails.",
            stacklevel=2,
        )


def _relax_projector_scale_literal() -> None:
    """Widen projector_scale's Literal to include P2/P6 on ModelConfig + every subclass."""
    global _LITERAL_PATCHED
    import rfdetr  # noqa: F401  (ensures variant/config subclasses are imported)
    import rfdetr.config as rf_config

    new_annotation = List[Literal["P2", "P3", "P4", "P5", "P6"]]
    targets = [rf_config.ModelConfig, *_all_subclasses(rf_config.ModelConfig)]
    failures: List[str] = []
    for cls in targets:
        field = cls.model_fields.get("projector_scale")
        if field is None:
            continue
        try:
            field.annotation = new_annotation
            cls.model_rebuild(force=True)
        except Exception as exc:  # pragma: no cover - defensive
            failures.append(f"{cls.__name__}: {exc}")
    if failures:
        warnings.warn(
            "[rf_detr_p2] Could not relax projector_scale Literal on: " + "; ".join(failures),
            stacklevel=2,
        )
    _LITERAL_PATCHED = True


def _patch_backbone_init() -> None:
    """Replace Backbone.__init__ with a P2-aware copy (adds P2=4.0 + projector knobs).

    Faithful copy of rfdetr/models/backbone/backbone.py Backbone.__init__ (v1.8.1).
    The only differences from upstream are marked with ``# P2:`` comments.
    """
    global _BACKBONE_PATCHED, _PATCHED_INIT
    import rfdetr.models.backbone.backbone as backbone_module
    from rfdetr.models.backbone.base import BackboneBase
    from rfdetr.models.backbone.dinov2 import DinoV2
    from rfdetr.models.backbone.projector import MultiScaleProjector

    def _p2_backbone_init(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,
        drop_path=0.0,
        out_channels=256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 0,
        dual_projector: bool = False,
    ):
        # P2: explicit parent init (no __class__ cell in a module-level function).
        BackboneBase.__init__(self)
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"
        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")
        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")
        assert len(name_parts) == 2, (
            "name should be dinov2, then either registers, windowed, both, or none, then the size"
        )
        self.encoder = DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=use_registers,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            drop_path_rate=drop_path,
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        assert sorted(self.projector_scale) == self.projector_scale, (
            "only support projector scale P2/P3/P4/P5/P6 in ascending order."
        )
        # P2: added P2=4.0 (stride-4, scale-factor x4) -- upstream lacks this key.
        level2scalefactor = dict(P2=4.0, P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        # P2: wire projector internals from config (defaults reproduce upstream behavior).
        projector_kwargs = _build_projector_kwargs(layer_norm, rms_norm)
        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            **projector_kwargs,
        )
        self.cross_attn_projector = (
            MultiScaleProjector(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                scale_factors=scale_factors,
                **projector_kwargs,
            )
            if dual_projector
            else None
        )

        self._export = False

    backbone_module.Backbone.__init__ = _p2_backbone_init
    _PATCHED_INIT = _p2_backbone_init
    _BACKBONE_PATCHED = True


def _patch_pretrain_weight_filter() -> None:
    """Drop size-mismatched checkpoint tensors on non-strict ``LWDETR.load_state_dict``.

    Stock detection checkpoints are single-scale (``projector_scale=["P4"]``); enabling
    P2 changes ``num_feature_levels`` and resizes existing deformable-attention Linear
    layers + the projector's first stage. ``torch``'s ``strict=False`` skips only
    missing/extra keys, not size-mismatched ones, so ``load_pretrain_weights`` would
    raise. We drop those tensors so they start randomly initialised (the documented P2
    behavior). Applied only under P2, only on ``strict=False`` loads; a matching P2
    checkpoint (test / inference / resume) mismatches nothing and loads unchanged.
    """
    global _PRETRAIN_FILTER_PATCHED
    import rfdetr.models.lwdetr as lwdetr_module

    model_cls = lwdetr_module.LWDETR
    existing = model_cls.__dict__.get("load_state_dict")
    if existing is not None and getattr(existing, "_p2_drops_mismatch", False):
        _PRETRAIN_FILTER_PATCHED = True
        return
    base_load_state_dict = model_cls.load_state_dict  # inherited nn.Module.load_state_dict

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        if not strict:
            model_state = self.state_dict()
            mismatched = [
                key
                for key, value in state_dict.items()
                if key in model_state
                and hasattr(value, "shape")
                and tuple(value.shape) != tuple(model_state[key].shape)
            ]
            if mismatched:
                drop = set(mismatched)
                state_dict = {k: v for k, v in state_dict.items() if k not in drop}
                warnings.warn(
                    f"[rf_detr_p2] Dropped {len(mismatched)} size-mismatched pretrained "
                    f"tensor(s) so they start randomly initialised (P2 changes the "
                    f"feature-level count); e.g. {mismatched[0]}.",
                    stacklevel=2,
                )
        return base_load_state_dict(self, state_dict, strict, *args, **kwargs)

    load_state_dict._p2_drops_mismatch = True
    model_cls.load_state_dict = load_state_dict
    _PRETRAIN_FILTER_PATCHED = True


def _suppress_compat_warning() -> None:
    """Silence the expected projector_scale PretrainWeightsCompatibilityWarning."""
    try:
        from rfdetr.config import PretrainWeightsCompatibilityWarning

        warnings.filterwarnings("ignore", category=PretrainWeightsCompatibilityWarning)
    except Exception:  # pragma: no cover - fall back to message match
        warnings.filterwarnings("ignore", message=r".*pretrained weights.*")


def ensure_p2_support(p2_config: Optional[Mapping[str, Any]] = None) -> None:
    """Apply the in-process P2 patches and record projector settings (idempotent).

    Safe to call once per process before model construction (training, test,
    inference, tracking). The class patches run once; P2_SETTINGS is refreshed every
    call so the most recent config wins, which matters for re-launched DDP ranks.
    """
    global _WARN_FILTER_INSTALLED
    p2_config = p2_config or {}
    _record_settings(p2_config)

    weights = p2_config.get("weights", {}) or {}
    if weights.get("suppress_compat_warning", True) and not _WARN_FILTER_INSTALLED:
        _suppress_compat_warning()
        _WARN_FILTER_INSTALLED = True

    if is_patched():
        return

    _check_version()
    _relax_projector_scale_literal()
    _patch_backbone_init()
    _patch_pretrain_weight_filter()
