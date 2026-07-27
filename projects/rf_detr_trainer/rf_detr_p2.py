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
monkey-patch, version-controlled here. The shared model builder installs it for all
architectures so the static ONNX projector boundary also covers stock/TrackNet export;
P2 scales, overrides, and mismatch filtering remain conditional on a real P2 model.
It is the single place that reaches into ``rfdetr`` internals.

Public surface:
    ensure_p2_support(p2_config)        -- idempotent; relax Literal + patch backbone.
    resolve_p2_projector_scale(p2_cfg)  -- validated scale list (default [P2,P3,P4]).
    apply_p2_overrides(kwargs, p2_cfg)  -- merge p2.overrides into model kwargs.
    P2_SETTINGS                         -- projector internals read by patched backbone.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence

import torch

__all__ = [
    'assert_p2_checkpoint_compatible',
    'assert_p2_training_checkpoint_compatible',
    "P2_SETTINGS",
    "ensure_p2_support",
    "resolve_p2_projector_scale",
    "apply_p2_overrides",
    "is_patched",
]

# Version of rfdetr the Backbone.__init__ copy below was written against. A mismatch
# only warns (the copy may have drifted from upstream); P2 still attempts to apply.
_BASELINE_RFDETR_VERSION = "1.8.3"

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

# Upstream MultiScaleProjector defaults (rfdetr 1.8.3) used when a setting is None.
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


def _stabilize_raw_features_for_export(
    raw_features: Sequence[torch.Tensor],
    *,
    channels: Sequence[int],
    height: int,
    width: int,
) -> List[torch.Tensor]:
    """Give TensorRT a static C/H/W contract at the projector boundary.

    DINO's ONNX graph reconstructs its feature maps through shape-tensor
    operations. With a dynamic input batch, ONNX/TensorRT can consequently
    lose the otherwise-known channel dimension before a P2/P3
    ``ConvTranspose``. TensorRT requires deconvolution input channels to be a
    build-time constant.

    ``reshape(-1, C, H, W)`` deliberately leaves only batch inferred while
    materialising the encoder metadata as constants in the exported graph.
    This helper is used only by ``Backbone.forward_export`` and owns no state,
    so normal training/inference and existing P2 checkpoint keys are unchanged.
    """

    if len(raw_features) != len(channels):
        raise RuntimeError(
            "P2 ONNX export expected one channel declaration per DINO feature, "
            f"got {len(raw_features)} feature(s) and {len(channels)} declaration(s)."
        )
    if height <= 0 or width <= 0:
        raise RuntimeError(f"P2 ONNX export requires positive fixed feature dimensions, got {(height, width)}.")

    stabilized: List[torch.Tensor] = []
    for index, (feature, channel_count) in enumerate(zip(raw_features, channels)):
        if feature.ndim != 4:
            raise RuntimeError(
                f"P2 ONNX export expected DINO feature {index} to be rank-4 NCHW, "
                f"got shape {tuple(feature.shape)}."
            )
        channel_count = int(channel_count)
        if channel_count <= 0:
            raise RuntimeError(
                f"P2 ONNX export requires a positive fixed channel count for DINO feature {index}, "
                f"got {channel_count}."
            )
        actual_chw = tuple(int(dimension) for dimension in feature.shape[1:])
        expected_chw = (channel_count, int(height), int(width))
        if actual_chw != expected_chw:
            raise RuntimeError(
                f'P2 ONNX export feature {index} does not match encoder metadata: '
                f'actual C/H/W={actual_chw}, expected C/H/W={expected_chw}. '
                'Export shape must match the model resolution.'
            )
        # Keep the shape tuple fully constant. ``-1`` is inferred from the
        # dynamic input batch; every dimension required by Conv/ConvTranspose
        # is therefore known to the TensorRT parser.
        stabilized.append(feature.reshape(-1, channel_count, int(height), int(width)))
    return stabilized


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

    Faithful copy of rfdetr/models/backbone/backbone.py Backbone.__init__ (v1.8.3).
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

    def _p2_backbone_forward_export(self, tensors: torch.Tensor):
        """Tensor-only backbone forward with a static projector shape boundary."""

        raw_feats = self.encoder(tensors)
        encoder_shape = getattr(self.encoder, "shape", None)
        patch_size = int(getattr(self.encoder, "patch_size", 0) or 0)
        if not isinstance(encoder_shape, (tuple, list)) or len(encoder_shape) != 2:
            raise RuntimeError(
                "P2 ONNX export could not determine the fixed DINO target shape; "
                f"encoder.shape={encoder_shape!r}."
            )
        target_height, target_width = (int(value) for value in encoder_shape)
        actual_input_hw = tuple(int(dimension) for dimension in tensors.shape[-2:])
        if actual_input_hw != (target_height, target_width):
            raise RuntimeError(
                'P2 ONNX export shape must match the model resolution exactly: '
                f'input H/W={actual_input_hw}, encoder.shape={(target_height, target_width)}.'
            )
        if patch_size <= 0 or target_height % patch_size != 0 or target_width % patch_size != 0:
            raise RuntimeError(
                "P2 ONNX export requires encoder.shape to be divisible by encoder.patch_size; "
                f"got shape={(target_height, target_width)}, patch_size={patch_size}."
            )
        raw_feats = _stabilize_raw_features_for_export(
            raw_feats,
            channels=self.encoder._out_feature_channels,
            height=target_height // patch_size,
            width=target_width // patch_size,
        )
        feats = self.projector(raw_feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            batch, _, height, width = feat.shape
            out_masks.append(torch.zeros((batch, height, width), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)

        cross_attn_feats = None
        if self.cross_attn_projector is not None:
            cross_attn_feats = list(self.cross_attn_projector(raw_feats))

        return out_feats, out_masks, cross_attn_feats

    _p2_backbone_forward_export._p2_static_export_shapes = True
    backbone_module.Backbone.__init__ = _p2_backbone_init
    backbone_module.Backbone.forward_export = _p2_backbone_forward_export
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
        backbone = getattr(self, 'backbone', None)
        try:
            backbone = backbone[0]
        except (IndexError, KeyError, TypeError):
            pass
        projector_scale = list(getattr(backbone, 'projector_scale', []) or [])
        if not strict and 'P2' in projector_scale:
            model_state = self.state_dict()
            mismatched = [
                key
                for key, value in state_dict.items()
                if key in model_state
                and _is_p2_architecture_tensor(key)
                and hasattr(value, "shape")
                and tuple(value.shape) != tuple(model_state[key].shape)
            ]
            if mismatched:
                drop = set(mismatched)
                state_dict = {k: v for k, v in state_dict.items() if k not in drop}
                warnings.warn(
                    f"[rf_detr_p2] Dropped {len(mismatched)} known P2 size-mismatched pretrained "
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


def _find_p2_lwdetr(model: Any) -> Any:
    current = model
    for _ in range(8):
        optimized = getattr(current, '_orig_mod', None)
        if optimized is not None:
            current = optimized
            continue
        if hasattr(current, 'backbone') and hasattr(current, 'transformer'):
            return current
        current = getattr(current, 'model', None)
        if current is None:
            break
    return None


def _p2_checkpoint_state(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint.get('model'), Mapping):
        state = checkpoint['model']
    elif isinstance(checkpoint.get('state_dict'), Mapping):
        state = checkpoint['state_dict']
    elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state = checkpoint
    else:
        raise RuntimeError(
            'P2 checkpoint must contain an RF-DETR model mapping or Lightning state_dict mapping.'
        )
    normalized: Dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ('module.', 'model.', '_orig_mod.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        normalized[key] = value
    return normalized


def _checkpoint_architecture_metadata(checkpoint: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    metadata = checkpoint.get('pitchobjectlab_architecture')
    if metadata is None and isinstance(checkpoint.get('args'), Mapping):
        metadata = checkpoint['args'].get('pitchobjectlab_architecture')
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise RuntimeError('Checkpoint pitchobjectlab_architecture metadata must be a mapping.')
    return metadata


def _assert_p2_metadata_compatible(
    checkpoint: Mapping[str, Any],
    expected_architecture: Optional[Mapping[str, Any]],
) -> None:
    metadata = _checkpoint_architecture_metadata(checkpoint)
    if metadata is None or expected_architecture is None:
        return
    if metadata.get('schema_version') != expected_architecture.get('schema_version'):
        raise RuntimeError(
            'Checkpoint architecture metadata schema does not match this runtime: '
            f"checkpoint={metadata.get('schema_version')!r}, "
            f"runtime={expected_architecture.get('schema_version')!r}."
        )
    saved_size = str(metadata.get('model_size', '')).strip().lower()
    expected_size = str(expected_architecture.get('model_size', '')).strip().lower()
    if saved_size != expected_size:
        raise RuntimeError(
            'Checkpoint model size does not match the configured runtime: '
            f'checkpoint={saved_size!r}, runtime={expected_size!r}.'
        )
    saved_p2 = metadata.get('p2')
    expected_p2 = expected_architecture.get('p2')
    if saved_p2 != expected_p2:
        raise RuntimeError(
            'Checkpoint P2 architecture metadata does not match model.p2 in the runtime config: '
            f'checkpoint={saved_p2!r}, runtime={expected_p2!r}.'
        )


def _is_p2_architecture_tensor(name: str) -> bool:
    return (
        name.startswith('backbone.0.projector.')
        or name.startswith('backbone.0.cross_attn_projector.')
        or 'cross_attn.sampling_offsets.' in name
        or 'cross_attn.attention_weights.' in name
        or name == 'transformer.level_embed'
    )


def assert_p2_checkpoint_compatible(
    model: Any,
    checkpoint_path: Any,
    expected_architecture: Optional[Mapping[str, Any]] = None,
) -> None:
    '''Require shape-exact P2 architecture tensors for test/inference.'''

    lwdetr = _find_p2_lwdetr(model)
    if lwdetr is None:
        raise RuntimeError(
            'P2-enabled RF-DETR runtime has no LWDETR model to validate.'
        )
    backbone = getattr(lwdetr, 'backbone', None)
    try:
        backbone = backbone[0]
    except (IndexError, KeyError, TypeError):
        pass
    projector_scale = list(getattr(backbone, 'projector_scale', []) or [])
    if 'P2' not in projector_scale:
        raise RuntimeError(
            f'P2 validation expected projector_scale containing P2, got {projector_scale}.'
        )
    if checkpoint_path is None or not str(checkpoint_path).strip():
        raise RuntimeError(
            'P2-enabled RF-DETR test/inference requires a trained P2 checkpoint.'
        )
    path = Path(str(checkpoint_path)).expanduser()
    if not path.is_file():
        raise RuntimeError(f'P2 checkpoint does not exist: {path}')
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f'P2 checkpoint must be a mapping: {path}')

    _assert_p2_metadata_compatible(checkpoint, expected_architecture)

    checkpoint_state = _p2_checkpoint_state(checkpoint)
    # Runtime P2 checkpoints must be an exact architecture/state layout match.
    # The constructor's non-strict loader intentionally permits stock -> P2
    # initialization for training, but test/inference must detect every key or
    # shape that loader may have dropped, interpolated, or reinitialized
    # (including positional embeddings, query embeddings, and class heads).
    expected_state = {
        key: value
        for key, value in lwdetr.state_dict().items()
        if torch.is_tensor(value)
    }
    checkpoint_architecture = dict(checkpoint_state)
    missing = sorted(set(expected_state) - set(checkpoint_architecture))
    unexpected = sorted(set(checkpoint_architecture) - set(expected_state))
    mismatched = sorted(
        key
        for key in set(expected_state) & set(checkpoint_architecture)
        if tuple(expected_state[key].shape) != tuple(checkpoint_architecture[key].shape)
    )
    if missing or unexpected or mismatched:
        details: List[str] = []
        if missing:
            details.append(f'missing={missing[:5]}')
        if unexpected:
            details.append(f'unexpected={unexpected[:5]}')
        if mismatched:
            shape_details = [
                f'{key}: checkpoint={tuple(checkpoint_architecture[key].shape)}, '
                f'model={tuple(expected_state[key].shape)}'
                for key in mismatched[:5]
            ]
            details.append(f'shape_mismatch={shape_details}')
        raise RuntimeError(
            f'Checkpoint {path} is incompatible with configured P2 architecture '
            f'{projector_scale}: ' + '; '.join(details)
        )


def assert_p2_training_checkpoint_compatible(
    model: Any,
    checkpoint_path: Any,
    expected_architecture: Optional[Mapping[str, Any]] = None,
    *,
    allow_stock_initialization: bool,
) -> None:
    '''Permit mismatch filtering only for an official single-level stock checkpoint.'''

    if not allow_stock_initialization:
        assert_p2_checkpoint_compatible(model, checkpoint_path, expected_architecture)
        return
    path = Path(str(checkpoint_path)).expanduser()
    if not path.is_file():
        raise RuntimeError(f'P2 training checkpoint does not exist: {path}')
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f'P2 training checkpoint must be a mapping: {path}')

    lwdetr = _find_p2_lwdetr(model)
    if lwdetr is None:
        raise RuntimeError('P2 training runtime has no LWDETR model to validate.')
    checkpoint_state = _p2_checkpoint_state(checkpoint)
    model_state = lwdetr.state_dict()
    metadata = _checkpoint_architecture_metadata(checkpoint)
    if metadata is not None:
        saved_p2 = metadata.get('p2')
        if not isinstance(saved_p2, Mapping):
            raise RuntimeError('P2 training checkpoint metadata has no valid p2 mapping.')
        if bool(saved_p2.get('enabled', False)):
            assert_p2_checkpoint_compatible(model, checkpoint_path, expected_architecture)
            return
        if expected_architecture is not None:
            if metadata.get('schema_version') != expected_architecture.get('schema_version'):
                raise RuntimeError(
                    'P2 stock checkpoint architecture metadata schema does not match this runtime.'
                )
            saved_size = str(metadata.get('model_size', '')).strip().lower()
            expected_size = str(expected_architecture.get('model_size', '')).strip().lower()
            if saved_size != expected_size:
                raise RuntimeError(
                    'P2 stock checkpoint model size does not match the configured runtime: '
                    f'checkpoint={saved_size!r}, runtime={expected_size!r}.'
                )
    sampling_key = next(
        (
            key
            for key in model_state
            if key.endswith('cross_attn.sampling_offsets.weight') and key in checkpoint_state
        ),
        None,
    )
    backbone = getattr(lwdetr, 'backbone', None)
    try:
        backbone = backbone[0]
    except (IndexError, KeyError, TypeError):
        pass
    level_count = len(list(getattr(backbone, 'projector_scale', []) or []))
    is_single_level_stock = False
    if sampling_key is not None and level_count > 1:
        model_shape = tuple(model_state[sampling_key].shape)
        checkpoint_shape = tuple(checkpoint_state[sampling_key].shape)
        is_single_level_stock = (
            len(model_shape) == len(checkpoint_shape)
            and checkpoint_shape[0] * level_count == model_shape[0]
            and checkpoint_shape[1:] == model_shape[1:]
        )
    if not is_single_level_stock:
        assert_p2_checkpoint_compatible(model, checkpoint_path, expected_architecture)
        return

    # Only the known feature-level tensors may differ for stock -> P2
    # initialization. Resolution, class heads, queries, encoder width, and all
    # other architecture tensors must remain shape-exact so the patched
    # non-strict loader cannot silently discard an unrelated incompatibility.
    # RF-DETR 1.8.3 derives an empty detection-only keypoint mask at runtime;
    # older official stock checkpoints predate that persistent buffer.
    missing = sorted(
        key
        for key in set(model_state) - set(checkpoint_state)
        if not _is_p2_architecture_tensor(key)
        and not (
            key == '_kp_active_mask'
            and torch.is_tensor(model_state[key])
            and model_state[key].dtype == torch.bool
            and model_state[key].numel() == 0
        )
    )
    unexpected = sorted(
        key
        for key in set(checkpoint_state) - set(model_state)
        if not _is_p2_architecture_tensor(key)
    )
    mismatched = sorted(
        key
        for key in set(model_state) & set(checkpoint_state)
        if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape)
        and not _is_p2_architecture_tensor(key)
    )
    if missing or unexpected or mismatched:
        details: List[str] = []
        if missing:
            details.append(f'missing_non_p2={missing[:5]}')
        if unexpected:
            details.append(f'unexpected_non_p2={unexpected[:5]}')
        if mismatched:
            details.append(
                'shape_mismatch_non_p2='
                + repr(
                    [
                        f'{key}: checkpoint={tuple(checkpoint_state[key].shape)}, '
                        f'model={tuple(model_state[key].shape)}'
                        for key in mismatched[:5]
                    ]
                )
            )
        raise RuntimeError(
            f'Checkpoint {path} is not a compatible stock initialization for P2: '
            + '; '.join(details)
        )


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
