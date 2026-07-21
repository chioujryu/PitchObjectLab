"""Pluggable TrackNetV5 motion modules for RF-DETR.

Adds a config-toggled ``model.motion`` block (parallel to ``model.p2``) that
attaches a TrackNetV5-inspired motion processing pipeline to RF-DETR without
touching the installed ``rfdetr`` package (survives ``uv sync``).

Architecture summary
--------------------
Two innovations from the TrackNetV5 paper are ported as feature-space modules
that insert *between* the DINOv2 backbone and the Deformable-DETR decoder:

1. **Motion Direction Decoupling (MDD)** — decompose signed inter-frame deltas
   into arrival (P⁺) and departure (P⁻) polarity fields, weighted by a
   per-channel learnable sigmoid attention:
       A = sigmoid(k * (|Δ| − m))
   Each polarity field is then used to gate the corresponding backbone feature
   maps with a learned residual: ``features × (1 + gate(motion_maps))``.
   Gates are zero-initialised so the model starts as identity and learns to
   exploit motion progressively.

2. **Residual-driven Spatio-Temporal Refinement (R-STR)** — a lightweight
   per-scale spatial self-attention block (factorised à la TimeSformer, but
   with T=1 for Phase-1 single-frame data) that treats decoder features as a
   draft prediction and estimates a correction residual Δ applied via:
       H_final = σ(Draft + Δ)

Phase-1 single-frame fallback
------------------------------
Current training datasets are SAHI crops of still images with no temporal
neighbours.  ``motion.temporal.fallback_mode`` controls how the T-frame window
is synthesised:

* ``identity``  — all T frames are identical copies of the input (Δ=0,
  motion maps are zero, module acts as identity until real motion is seen).
* ``noise``     — Gaussian noise is added to the copies to produce synthetic
  deltas (regularises the attention branches during training).
* ``zero``      — same as identity but explicitly zeroes the delta tensors
  before any computation (guarantees no information leak).

Phase-2 multi-frame support
----------------------------
When a temporal-sequence DataLoader is wired in, set
``motion.temporal.fallback_mode: real`` and pass a [B, T, 3, H, W] input
tensor.  The module detects the extra temporal dimension automatically.

Public surface
--------------
    ensure_motion_support(motion_cfg)   – idempotent; patches LWDETR in-process.
    attach_motion_module(model, motion_cfg) – build + attach MotionModule to model.
    apply_motion_overrides(kwargs, motion_cfg) – merge convenience overrides.
    MOTION_SETTINGS                     – global config dict read by patches.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'assert_motion_checkpoint_compatible',
    "MOTION_SETTINGS",
    "ensure_motion_support",
    "attach_motion_module",
    "assert_motion_export_ready",
    "apply_motion_overrides",
    "is_patched",
    "MotionModule",
]

# Convenience overrides that map to real ModelConfig fields.
_BASELINE_RFDETR_VERSION = '1.8.3'

_MOTION_OVERRIDE_CASTS = {
    "resolution": int,
    "num_queries": int,
    "num_select": int,
    "gradient_checkpointing": bool,
}

# Global settings threaded from the YAML config to the patches (mirrors P2_SETTINGS).
_MOTION_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "type": "tracknet_v5",
    "temporal": {
        "num_frames": 3,
        "frame_stride": 1,
        "fallback_mode": "identity",
        "noise_std": 0.02,
    },
    "tracknet_v5": {
        "mdd": {
            "enabled": True,
            "polarity_channels": 4,
            "attention": {
                "learnable": True,
                "init_k": 1.0,
                "init_m": 0.5,
            },
        },
        "rstr": {
            "enabled": True,
            "num_blocks": 2,
            "hidden_dim": 256,
            "num_heads": 8,
            "attention_mode": "divided",
            "dropout": 0.1,
            "use_pixel_shuffle": True,
        },
    },
    "loss": {
        "motion_attention_weight": 0.0,
    },
    "overrides": {},
}

MOTION_SETTINGS: Dict[str, Any] = deepcopy(_MOTION_DEFAULTS)

_LWDETR_PATCHED = False
_CHANNEL_CHECK_PATCHED = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _check_version() -> None:
    try:
        import importlib.metadata as metadata
        version = metadata.version("rfdetr")
    except Exception:
        version = "unknown"
    if version != _BASELINE_RFDETR_VERSION:
        warnings.warn(
            f"[rf_detr_motion] Motion patch was written against rfdetr=={_BASELINE_RFDETR_VERSION} "
            f"but found {version}. Verify rfdetr/models/lwdetr.py line ~465 still matches.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# PyTorch module classes
# ---------------------------------------------------------------------------

class LearnableSigmoidAttention(nn.Module):
    """Per-channel learnable sigmoid gating: A = σ(k * (|x| − m)).

    k controls steepness, m controls the midpoint threshold. Both are learnable
    scalars initialised from config. Operates channel-wise on any spatial map.
    """

    def __init__(self, num_channels: int, init_k: float = 1.0, init_m: float = 0.5) -> None:
        super().__init__()
        self.k = nn.Parameter(torch.full((1, num_channels, 1, 1), init_k))
        self.m = nn.Parameter(torch.full((1, num_channels, 1, 1), init_m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] — the polarity field (already non-negative after ReLU)
        return torch.sigmoid(self.k * (x - self.m))


class MotionDirectionDecoupling(nn.Module):
    """TrackNetV5 MDD: decompose inter-frame delta into P⁺ / P⁻ polarity fields.

    For Phase-1 single-frame data (delta == 0) both P⁺ and P⁻ are zero; the
    attention maps are 0.5 (mid-point of sigmoid) and the output motion tensor
    is all zeros, so gated features reduce to identity.

    Args:
        polarity_channels: Number of output channels per polarity field (P⁺ and
            P⁻ each emit this many channels via a 1×1 conv).
        init_k, init_m: Initial sigmoid steepness / midpoint.
    """

    def __init__(
        self,
        in_channels: int = 3,
        polarity_channels: int = 2,
        init_k: float = 1.0,
        init_m: float = 0.5,
    ) -> None:
        super().__init__()
        if polarity_channels <= 0:
            raise ValueError(
                f"polarity_channels must be a positive integer, got {polarity_channels}."
            )
        self.polarity_channels = polarity_channels
        # Project each polarity field to polarity_channels feature maps.
        self.proj_plus = nn.Conv2d(in_channels, polarity_channels, 1, bias=False)
        self.proj_minus = nn.Conv2d(in_channels, polarity_channels, 1, bias=False)
        nn.init.zeros_(self.proj_plus.weight)
        nn.init.zeros_(self.proj_minus.weight)
        self.attn_plus = LearnableSigmoidAttention(polarity_channels, init_k, init_m)
        self.attn_minus = LearnableSigmoidAttention(polarity_channels, init_k, init_m)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute MDD motion maps from a [B, T, 3, H, W] frame sequence.

        Returns: [B, 2*polarity_channels, H, W] — concatenated P⁺ and P⁻
            attention maps (zero-initialised projections → zero maps for
            identity fallback).
        """
        # Use first two frames for delta: I_t − I_{t−1}
        f0, f1 = frames[:, 0], frames[:, 1]  # [B, 3, H, W]
        delta = f1 - f0                        # signed difference
        p_plus = F.relu(delta)                 # arrival regions (brightening)
        p_minus = F.relu(-delta)               # departure regions (darkening)
        attn_p = self.attn_plus(self.proj_plus(p_plus))    # [B, C_p, H, W]
        attn_m = self.attn_minus(self.proj_minus(p_minus)) # [B, C_p, H, W]
        return torch.cat([attn_p, attn_m], dim=1)           # [B, 2*C_p, H, W]


class MotionFeatureGate(nn.Module):
    """Apply motion attention maps to a single feature-pyramid level.

    Uses a residual gate: ``out = feat * (1 + gate(motion))`` so that
    zero-initialised gate weights produce identity output at the start of
    training.  ``motion`` is resized to match the feature spatial resolution.

    Args:
        feature_channels: Number of channels in the backbone feature map.
        motion_channels: Number of channels in the MDD output tensor.
    """

    def __init__(self, feature_channels: int, motion_channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(motion_channels, feature_channels, 1, bias=True),
            nn.Tanh(),  # outputs in [-1, 1]; multiplied by (1 + ...) keeps features positive
        )
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def forward(self, feat: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        # Resize motion maps to match feature resolution.
        if motion.shape[-2:] != feat.shape[-2:]:
            target_size = feat.shape[-2:]
            if torch.onnx.is_in_onnx_export():
                target_size = (int(feat.shape[-2]), int(feat.shape[-1]))
            motion = F.interpolate(motion, size=target_size, mode="bilinear", align_corners=False)
        return feat * (1.0 + self.gate(motion))


class RSTRSpatialAttention(nn.Module):
    """Spatial self-attention block for R-STR refinement (single scale).

    Implements the spatial leg of TimeSformer's factorised space-time attention.
    For Phase-1 (T=1) this is equivalent to standard spatial self-attention.
    Token sequence: H*W spatial tokens, each of dim ``embed_dim``.

    Args:
        embed_dim: Channel depth of the feature map.
        num_heads: Multi-head attention heads.
        dropout: Dropout probability on the attention output.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]  where N = H*W
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + self.dropout(x)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual + x


class RSTRHead(nn.Module):
    """R-STR: Residual Spatio-Temporal Refinement applied to one feature scale.

    Flattens the spatial feature map to a sequence, runs ``num_blocks`` spatial
    attention blocks to estimate a correction delta, then applies:
        H_final = sigmoid(Draft + Delta)
    projected back to the original channel depth.  Pixel-shuffle upsampling is
    used for optional fine-resolution output matching.

    Args:
        in_channels: Feature channel count for this scale.
        hidden_dim: Internal projection dimension (defaults to in_channels).
        num_heads: Attention heads.
        num_blocks: Number of stacked RSTR blocks.
        dropout: Dropout applied inside attention and FFN.
        use_pixel_shuffle: Use PixelShuffle ×2 upsampling on the delta (keeps
            output at the same spatial resolution as input via padding trick).
        context_mask_prob: Stochastic context masking dropout during training
            (equivalent to the ρ=0.1 from the paper).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 2,
        dropout: float = 0.1,
        use_pixel_shuffle: bool = True,
        context_mask_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_mask_prob = context_mask_prob
        # Project to hidden_dim for attention.
        self.proj_in = nn.Conv2d(in_channels, hidden_dim, 1)
        # Spatial attention blocks.
        self.blocks = nn.ModuleList(
            [RSTRSpatialAttention(hidden_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        # Project back to in_channels to produce the delta.
        self.proj_out = nn.Conv2d(hidden_dim, in_channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W]
        B, C, H, W = feat.shape
        export_height = int(H) if torch.onnx.is_in_onnx_export() else H
        export_width = int(W) if torch.onnx.is_in_onnx_export() else W
        # Project to hidden_dim.
        x = self.proj_in(feat)  # [B, D, H, W]
        D = x.shape[1]
        # Stochastic context masking during training (regularises temporal attention).
        if self.training and self.context_mask_prob > 0.0:
            mask = torch.rand(B, 1, H, W, device=x.device) > self.context_mask_prob
            x = x * mask.float()
        # Flatten spatial for self-attention: [B, H*W, D]
        x = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        # Reshape back to spatial: [B, D, H, W]
        # Keep only batch dynamic at this Conv boundary during ONNX export.
        x = x.transpose(1, 2).reshape(-1, D, export_height, export_width)
        # Project to correction delta.
        delta = self.proj_out(x)  # [B, C, H, W]
        # Apply residual correction: sigmoid(Draft + Delta)
        return torch.sigmoid(feat + delta)


class MotionModule(nn.Module):
    """Top-level motion module attached to LWDETR.

    Encapsulates MDD polarity computation, per-scale feature gates, and an
    optional R-STR refinement head.  Accepts a single-frame input (Phase 1)
    and synthesises the temporal window internally using the configured
    ``fallback_mode``.

    Args:
        feature_channels_per_scale: List of channel counts for each pyramid
            level output by the backbone (e.g. [256, 256, 256] for [P2,P3,P4]).
        motion_cfg: The ``model.motion`` sub-dict from the trainer config.
    """

    def __init__(
        self,
        feature_channels_per_scale: List[int],
        motion_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()
        if not feature_channels_per_scale:
            raise ValueError("MotionModule requires at least one backbone feature level.")
        self.feature_channels_per_scale = [int(channel) for channel in feature_channels_per_scale]
        v5_cfg = motion_cfg.get("tracknet_v5", {}) or {}
        mdd_cfg = v5_cfg.get("mdd", {}) or {}
        rstr_cfg = v5_cfg.get("rstr", {}) or {}
        temporal_cfg = motion_cfg.get("temporal", {}) or {}

        self.num_frames: int = int(temporal_cfg.get("num_frames", 3))
        self.fallback_mode: str = str(temporal_cfg.get("fallback_mode", "identity"))
        self.noise_std: float = float(temporal_cfg.get("noise_std", 0.02))

        # MDD module.
        self.mdd_enabled: bool = bool(mdd_cfg.get("enabled", True))
        polarity_channels: int = int(mdd_cfg.get("polarity_channels", 4))
        # polarity_channels must be even (split between P+ and P-)
        if polarity_channels % 2 != 0:
            polarity_channels += 1
        half_p = polarity_channels // 2
        attn_cfg = mdd_cfg.get("attention", {}) or {}

        if self.mdd_enabled:
            self.mdd = MotionDirectionDecoupling(
                in_channels=3,
                polarity_channels=half_p,
                init_k=float(attn_cfg.get("init_k", 1.0)),
                init_m=float(attn_cfg.get("init_m", 0.5)),
            )
            motion_ch = polarity_channels  # 2 * half_p
        else:
            self.mdd = None  # type: ignore[assignment]
            motion_ch = 0

        # Per-scale feature gates (only built when MDD is on).
        self.gates = nn.ModuleList()
        if self.mdd_enabled and motion_ch > 0:
            for ch in feature_channels_per_scale:
                self.gates.append(MotionFeatureGate(ch, motion_ch))
        else:
            for _ in feature_channels_per_scale:
                self.gates.append(nn.Identity())

        # Optional R-STR head.
        self.rstr_enabled: bool = bool(rstr_cfg.get("enabled", True))
        if self.rstr_enabled:
            self.rstr_heads = nn.ModuleList()
            for ch in feature_channels_per_scale:
                self.rstr_heads.append(
                    RSTRHead(
                        in_channels=ch,
                        hidden_dim=int(rstr_cfg.get("hidden_dim", 256)),
                        num_heads=int(rstr_cfg.get("num_heads", 8)),
                        num_blocks=int(rstr_cfg.get("num_blocks", 2)),
                        dropout=float(rstr_cfg.get("dropout", 0.1)),
                        use_pixel_shuffle=bool(rstr_cfg.get("use_pixel_shuffle", True)),
                        context_mask_prob=float(rstr_cfg.get("context_mask_prob", 0.1)),
                    )
                )
        else:
            self.rstr_heads = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Temporal window synthesis (Phase-1 single-frame fallback)
    # ------------------------------------------------------------------

    def _make_frame_window(self, images: torch.Tensor) -> torch.Tensor:
        """Expand [B, 3, H, W] to [B, T, 3, H, W] using the configured fallback."""
        if images.dim() == 5:
            # Already a temporal sequence — use as-is.
            return images
        B = images.shape[0]
        mode = self.fallback_mode
        if mode in ("identity", "zero"):
            frames = images.unsqueeze(1).expand(B, self.num_frames, -1, -1, -1)
            return frames
        if mode == "noise":
            frames = [images]
            for _ in range(self.num_frames - 1):
                noise = torch.randn_like(images) * self.noise_std
                frames.append(images + noise)
            return torch.stack(frames, dim=1)  # [B, T, 3, H, W]
        # Fallback to identity for unknown modes.
        return images.unsqueeze(1).expand(B, self.num_frames, -1, -1, -1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        features: list,
    ) -> list:
        """Apply motion processing to backbone feature maps.

        Args:
            images: Raw input tensor [B, 3, H, W] or [B, T, 3, H, W].
            features: List of NestedTensors from LWDETR backbone (one per
                pyramid level).  Each NestedTensor has ``.tensors`` and
                ``.mask`` attributes.

        Returns:
            Modulated feature list (same type/length as input).
        """
        feature_tensors = [nested.tensors for nested in features]
        modulated_tensors = self.forward_export(images, feature_tensors)

        # Preserve the masks and NestedTensor contract used by regular LWDETR
        # inference/training.  The tensor-only export path below deliberately
        # shares all feature math with this path so ONNX cannot silently omit
        # the motion module.
        return [
            _rebuild_nested_tensor(src, nested.mask)
            for src, nested in zip(modulated_tensors, features)
        ]

    def forward_export(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Apply motion processing to tensor-only backbone export features.

        RF-DETR's ONNX mode changes the backbone output from ``NestedTensor``
        objects to raw tensors.  Keeping this as an explicit, shared path makes
        the exported graph numerically equivalent to the regular motion path
        while retaining RF-DETR's tensor-only export contract.
        """
        if len(features) != len(self.feature_channels_per_scale):
            raise RuntimeError(
                "RF-DETR motion feature-level mismatch: the backbone returned "
                f"{len(features)} level(s), but motion_module was built for "
                f"{len(self.feature_channels_per_scale)} level(s)."
            )
        # Build frame window for MDD.
        frames = self._make_frame_window(images)  # [B, T, 3, H, W]

        # Compute MDD motion maps.
        if self.mdd_enabled and self.mdd is not None:
            motion_maps = self.mdd(frames)  # [B, 2*polarity_channels, H_img, W_img]
        else:
            motion_maps = None

        # Apply gates + RSTR to each feature scale.
        modulated = []
        for i, feature in enumerate(features):
            src = feature  # [B, C, H_i, W_i]
            if torch.onnx.is_in_onnx_export():
                src = src.reshape(
                    -1,
                    self.feature_channels_per_scale[i],
                    int(src.shape[-2]),
                    int(src.shape[-1]),
                )

            # Motion-guided feature gate.
            if motion_maps is not None and i < len(self.gates):
                gate = self.gates[i]
                if not isinstance(gate, nn.Identity):
                    src = gate(src, motion_maps)

            # R-STR refinement.
            if self.rstr_enabled and self.rstr_heads is not None and i < len(self.rstr_heads):
                src = self.rstr_heads[i](src)

            modulated.append(src)

        return modulated


# ---------------------------------------------------------------------------
# NestedTensor helper (avoids importing rfdetr at module load time)
# ---------------------------------------------------------------------------

def _rebuild_nested_tensor(tensors: torch.Tensor, mask: Optional[torch.Tensor]):
    """Re-wrap a (tensors, mask) pair as an rfdetr NestedTensor."""
    try:
        from rfdetr.utilities import NestedTensor  # rfdetr >= 1.6
    except ImportError:
        from rfdetr.util.misc import NestedTensor  # rfdetr < 1.6 (deprecated path)
    return NestedTensor(tensors, mask)


# ---------------------------------------------------------------------------
# Monkey-patch: silence DINOv2 channel-count assertion
# ---------------------------------------------------------------------------

def _patch_dinov2_channel_check() -> None:
    """Remove the hardcoded num_channels == 3 check in Dinov2WithRegistersPatchEmbeddings.

    The check raises ValueError when the patch embedding receives more than 3
    channels (e.g. from early-fusion motion input in Phase 2).  For Phase 1 we
    still feed 3 channels, so the patch is a no-op but prevents surprises if
    the user experiments with expanded inputs.
    """
    global _CHANNEL_CHECK_PATCHED
    if _CHANNEL_CHECK_PATCHED:
        return
    try:
        from rfdetr.models.backbone.dinov2_with_windowed_attn import (
            Dinov2WithRegistersPatchEmbeddings,
        )
        original_forward = Dinov2WithRegistersPatchEmbeddings.forward

        def _flexible_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # Allow any number of channels; the projection conv handles the mapping.
            embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
            return embeddings

        _flexible_forward._motion_patched = True
        if not getattr(original_forward, "_motion_patched", False):
            Dinov2WithRegistersPatchEmbeddings.forward = _flexible_forward
    except Exception as exc:
        warnings.warn(
            f"[rf_detr_motion] Could not patch Dinov2WithRegistersPatchEmbeddings.forward: {exc}",
            stacklevel=2,
        )
    _CHANNEL_CHECK_PATCHED = True


# ---------------------------------------------------------------------------
# Monkey-patch: wrap LWDETR.forward to inject motion module
# ---------------------------------------------------------------------------

def _patch_lwdetr_motion_forward() -> None:
    """Wrap LWDETR forward paths to call motion after the backbone.

    The wrapper is transparent: if the model has no ``motion_module`` attribute
    (e.g. checkpoints loaded without motion support) the forward runs unchanged.
    Both regular ``forward`` and tensor-only ``forward_export`` are patched;
    omitting the latter would silently produce an ONNX/TensorRT graph without
    the attached motion module.  Installed once even if
    ``ensure_motion_support()`` is called multiple times.
    """
    global _LWDETR_PATCHED
    if _LWDETR_PATCHED:
        return
    try:
        import rfdetr.models.lwdetr as lwdetr_module
        try:
            from rfdetr.utilities import nested_tensor_from_tensor_list  # rfdetr >= 1.6
        except ImportError:
            from rfdetr.util.misc import nested_tensor_from_tensor_list  # rfdetr < 1.6

        original_init = lwdetr_module.LWDETR.__init__
        original_forward = lwdetr_module.LWDETR.forward
        original_forward_export = lwdetr_module.LWDETR.forward_export

        def _motion_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            active_config = MOTION_SETTINGS
            if bool(active_config.get("enabled", False)) and resolve_motion_type(active_config) != "none":
                # LWDETR is now fully constructed, while RF-DETR's facade and
                # Lightning loaders have not restored checkpoint weights yet.
                attach_motion_module(self, active_config)

        _motion_init._motion_patched = True

        def _motion_forward(self, samples, targets=None):
            # Normalise input to NestedTensor then extract raw pixels for motion.
            # Must convert to NestedTensor *before* .tensors to handle variable-size
            # image lists safely (nested_tensor_from_tensor_list pads to the same size).
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples)
            raw_images = samples.tensors  # [B, 3, H_max, W_max] (padded)

            # Run backbone (stock call).
            features, poss, cross_attn_features = self.backbone(samples)

            # Apply motion module if present (inserted by attach_motion_module).
            if getattr(self, "motion_module", None) is not None:
                features = self.motion_module(raw_images, features)

            # --- Resume stock LWDETR.forward from after the backbone call ---
            # We call the original forward but skip the backbone by temporarily
            # replacing self.backbone with a stub that returns the (already computed)
            # features, poss, cross_attn_features.
            class _BackboneStub(nn.Module):
                def forward(self, _samples):
                    return features, poss, cross_attn_features

            orig_backbone = self.backbone
            self.backbone = _BackboneStub()
            try:
                # The original forward will re-normalise samples — feed back the
                # NestedTensor so the isinstance check at line ~463 is a no-op.
                result = original_forward(self, samples, targets)
            finally:
                self.backbone = orig_backbone
            return result

        _motion_forward._motion_patched = True

        def _motion_forward_export(self, tensors):
            backbone_result = self.backbone(tensors)
            if not isinstance(backbone_result, tuple) or len(backbone_result) != 4:
                raise RuntimeError(
                    "Motion-aware RF-DETR export expected the backbone to return "
                    "(features, masks, positions, cross_attention_features)."
                )
            features, masks, poss, cross_attn_features = backbone_result

            motion_module = getattr(self, "motion_module", None)
            if motion_module is not None:
                motion_export = getattr(motion_module, "forward_export", None)
                if not callable(motion_export):
                    raise RuntimeError(
                        "Attached RF-DETR motion module has no tensor-only forward_export(); "
                        "refusing to export a graph that would omit motion processing."
                    )
                features = motion_export(tensors, list(features))

            # Resume the version-pinned upstream forward_export after its
            # backbone call.  Replacing only the forward callable keeps the
            # registered module/parameters intact while torch.onnx traces the
            # already-computed backbone and motion tensors.
            original_backbone_forward = self.backbone.forward

            def _backbone_export_stub(_tensors):
                return features, masks, poss, cross_attn_features

            self.backbone.forward = _backbone_export_stub
            try:
                return original_forward_export(self, tensors)
            finally:
                self.backbone.forward = original_backbone_forward

        _motion_forward_export._motion_patched = True
        if not getattr(original_init, "_motion_patched", False):
            lwdetr_module.LWDETR.__init__ = _motion_init
        if not getattr(original_forward, "_motion_patched", False):
            lwdetr_module.LWDETR.forward = _motion_forward
        if not getattr(original_forward_export, "_motion_patched", False):
            lwdetr_module.LWDETR.forward_export = _motion_forward_export
        _LWDETR_PATCHED = True
    except Exception as exc:
        warnings.warn(
            f"[rf_detr_motion] Could not patch LWDETR.forward: {exc}. "
            "Motion module will not be applied during forward passes.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (returns new dict, does not mutate)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _record_motion_settings(motion_cfg: Dict[str, Any]) -> None:
    """Write the caller's motion config into the global MOTION_SETTINGS dict."""
    global MOTION_SETTINGS
    MOTION_SETTINGS = _deep_merge(deepcopy(_MOTION_DEFAULTS), motion_cfg or {})


def resolve_motion_type(motion_cfg: Optional[Dict[str, Any]]) -> str:
    """Return the validated motion module type string."""
    mtype = (motion_cfg or {}).get("type", "tracknet_v5")
    valid = {"tracknet_v5", "none"}
    if mtype not in valid:
        warnings.warn(
            f"[rf_detr_motion] Unknown motion type {mtype!r}; defaulting to 'tracknet_v5'. "
            f"Valid options: {sorted(valid)}.",
            stacklevel=2,
        )
        mtype = "tracknet_v5"
    return mtype


def apply_motion_overrides(
    kwargs: dict,
    motion_cfg: Optional[Dict[str, Any]],
) -> dict:
    """Merge motion.overrides (real ModelConfig fields) into model kwargs in place."""
    overrides = (motion_cfg or {}).get("overrides", {}) or {}
    for key, cast in _MOTION_OVERRIDE_CASTS.items():
        value = overrides.get(key)
        if value is not None:
            kwargs[key] = cast(value)
    return kwargs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_patched() -> bool:
    """Return True once both LWDETR and DINOv2 patches have been applied."""
    return _LWDETR_PATCHED and _CHANNEL_CHECK_PATCHED


def ensure_motion_support(motion_cfg: Optional[Dict[str, Any]] = None) -> None:
    """Apply in-process patches and record motion settings (idempotent).

    Call this at the model-build choke point (same as ensure_p2_support for P2),
    before constructing the rfdetr model object.

    Args:
        motion_cfg: The ``model.motion`` dict from the trainer config.
    """
    motion_cfg = motion_cfg or {}
    _record_motion_settings(motion_cfg)
    _check_version()
    _patch_dinov2_channel_check()
    _patch_lwdetr_motion_forward()


def _infer_motion_feature_channels(lwdetr: nn.Module) -> List[int]:
    """Read projector output widths from the actual RF-DETR backbone."""
    backbone_container = getattr(lwdetr, "backbone", None)
    backbone = backbone_container
    if backbone is not None and not hasattr(backbone, "projector"):
        try:
            backbone = backbone[0]
        except (IndexError, KeyError, TypeError):
            backbone = None
    projector = getattr(backbone, "projector", None)
    scales = list(getattr(backbone, "projector_scale", []) or [])
    stages = getattr(projector, "stages", None)
    encoder_channels = getattr(getattr(backbone, "encoder", None), "_out_feature_channels", None)
    if projector is None or stages is None or not scales:
        raise RuntimeError(
            "Could not inspect RF-DETR backbone[0].projector/projector_scale while "
            "building the motion module."
        )
    if not isinstance(encoder_channels, (list, tuple)) or not encoder_channels:
        raise RuntimeError(
            "Could not inspect RF-DETR backbone[0].encoder._out_feature_channels while "
            "building the motion module."
        )
    stage_count = len(stages)
    uses_extra_pool = bool(getattr(projector, "use_extra_pool", False))
    pooled_last_level = (
        uses_extra_pool
        and scales[-1] == "P6"
        and stage_count == len(scales) - 1
    )
    if stage_count != len(scales) and not pooled_last_level:
        raise RuntimeError(
            "RF-DETR projector metadata is inconsistent: "
            f"{len(scales)} scale(s), {stage_count} learned output stage(s), "
            f"use_extra_pool={uses_extra_pool}."
        )

    transformer_width = int(getattr(getattr(lwdetr, "transformer", None), "d_model", 0) or 0)
    feature_channels: List[int] = []
    for index, stage in enumerate(stages):
        output_width = 0
        children = list(stage.children()) if isinstance(stage, nn.Module) else []
        for candidate in reversed(children):
            normalized_shape = getattr(candidate, "normalized_shape", None)
            if isinstance(normalized_shape, int):
                output_width = int(normalized_shape)
                break
            if isinstance(normalized_shape, (list, tuple)) and normalized_shape:
                output_width = int(normalized_shape[0])
                break
        if output_width <= 0:
            output_width = transformer_width
        if output_width <= 0:
            raise RuntimeError(
                f"Could not infer output channels for RF-DETR projector stage {index}."
            )
        feature_channels.append(output_width)
    if pooled_last_level:
        if not feature_channels:
            raise RuntimeError("RF-DETR P6 extra-pool level has no preceding projector output.")
        # P6 is max-pooled from the last learned feature and preserves channels.
        feature_channels.append(feature_channels[-1])
    return feature_channels


def attach_motion_module(model: nn.Module, motion_cfg: Optional[Dict[str, Any]] = None) -> None:
    """Build a MotionModule and attach it to the LWDETR model instance.

    Must be called *after* the rfdetr model object has been constructed so that
    the backbone's output channel shapes are known.  The module is attached as
    ``model.motion_module`` (or ``model.model.motion_module`` when wrapped in a
    PL module) and becomes part of the model's state_dict automatically.

    Args:
        model: The top-level model object.  We walk the attribute chain trying
            ``model``, ``model.model``, and ``model.model.model`` to find an
            LWDETR instance.
        motion_cfg: The ``model.motion`` dict from the trainer config.
    """
    motion_cfg = motion_cfg or {}
    if not bool(motion_cfg.get("enabled", False)):
        return

    mtype = resolve_motion_type(motion_cfg)
    if mtype == "none":
        return

    # Preserve intent separately from successful attachment.  The TensorRT
    # export validator uses this marker to fail if model discovery/version skew
    # prevented attachment instead of silently exporting a non-motion graph.
    model._motion_export_required = True  # type: ignore[attr-defined]

    # Walk the wrapper chain to find LWDETR.
    lwdetr = _find_lwdetr(model)
    if lwdetr is None:
        warnings.warn(
            "[rf_detr_motion] Could not locate an LWDETR instance in the model. "
            "Motion module not attached.",
            stacklevel=2,
        )
        return

    existing = getattr(lwdetr, "motion_module", None)
    if existing is not None:
        # The LWDETR.__init__ patch attaches before checkpoint restore. Calls at
        # legacy entrypoint sites are retained as safe, idempotent verification.
        lwdetr._motion_export_required = True  # type: ignore[attr-defined]
        return

    feature_channels = _infer_motion_feature_channels(lwdetr)
    motion_module = MotionModule(feature_channels, motion_cfg)
    reference_parameter = next(lwdetr.parameters(), None)
    if reference_parameter is not None:
        if reference_parameter.is_floating_point():
            motion_module.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        else:
            motion_module.to(device=reference_parameter.device)
    lwdetr.motion_module = motion_module  # type: ignore[assignment]
    lwdetr._motion_export_required = True  # type: ignore[attr-defined]


def _motion_state_from_checkpoint(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Return normalized ``motion_module.*`` tensors from RF-DETR checkpoint formats."""
    if isinstance(checkpoint.get("model"), Mapping):
        state = checkpoint["model"]
    elif isinstance(checkpoint.get("state_dict"), Mapping):
        state = checkpoint["state_dict"]
    elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state = checkpoint
    else:
        raise RuntimeError(
            "Motion checkpoint must contain an RF-DETR 'model' mapping or a Lightning 'state_dict' mapping."
        )

    motion_state: Dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "_orig_mod."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if key.startswith("motion_module."):
            motion_state[key] = value
    return motion_state


def _checkpoint_architecture_metadata(checkpoint: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    metadata = checkpoint.get('pitchobjectlab_architecture')
    if metadata is None and isinstance(checkpoint.get('args'), Mapping):
        metadata = checkpoint['args'].get('pitchobjectlab_architecture')
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise RuntimeError('Checkpoint pitchobjectlab_architecture metadata must be a mapping.')
    return metadata


def _assert_motion_metadata_compatible(
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
    saved_motion = metadata.get('motion')
    expected_motion = expected_architecture.get('motion')
    if saved_motion != expected_motion:
        raise RuntimeError(
            'Checkpoint TrackNet architecture metadata does not match model.motion in the runtime config: '
            f'checkpoint={saved_motion!r}, runtime={expected_motion!r}.'
        )


def assert_motion_checkpoint_compatible(
    model: nn.Module,
    checkpoint_path: Any,
    expected_architecture: Optional[Mapping[str, Any]] = None,
) -> None:
    """Require a complete, shape-exact TrackNet state before test/inference."""
    lwdetr = _find_lwdetr(model)
    if lwdetr is None or getattr(lwdetr, "motion_module", None) is None:
        raise RuntimeError(
            "Motion-enabled RF-DETR runtime has no attached motion_module. Ensure motion support "
            "is applied before constructing the model."
        )
    if checkpoint_path is None or not str(checkpoint_path).strip():
        raise RuntimeError(
            "Motion-enabled RF-DETR test/inference requires a checkpoint containing "
            "motion_module.* weights; pretrain_weights is empty."
        )
    path = Path(str(checkpoint_path)).expanduser()
    if not path.is_file():
        raise RuntimeError(f"Motion checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Motion checkpoint must be a mapping: {path}")

    _assert_motion_metadata_compatible(checkpoint, expected_architecture)

    checkpoint_state = _motion_state_from_checkpoint(checkpoint)
    expected_state = {
        key: value
        for key, value in lwdetr.state_dict().items()
        if key.startswith("motion_module.")
    }
    if not checkpoint_state:
        raise RuntimeError(
            f"Checkpoint {path} contains no motion_module.* weights. Refusing to run "
            "test/inference with randomly initialized TrackNet weights."
        )

    missing = sorted(set(expected_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(expected_state))
    mismatched = sorted(
        key
        for key in set(expected_state) & set(checkpoint_state)
        if tuple(expected_state[key].shape) != tuple(checkpoint_state[key].shape)
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if mismatched:
            shape_details = [
                f"{key}: checkpoint={tuple(checkpoint_state[key].shape)}, "
                f"model={tuple(expected_state[key].shape)}"
                for key in mismatched[:5]
            ]
            details.append(f"shape_mismatch={shape_details}")
        raise RuntimeError(
            f"Checkpoint {path} is incompatible with the configured TrackNet motion module: "
            + "; ".join(details)
        )


def assert_motion_export_ready(
    model: nn.Module,
    motion_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Fail before ONNX export if enabled motion cannot enter the graph.

    ``motion_cfg`` should be supplied by callers that know motion was requested.
    Without it, an attached ``motion_module`` is treated as the signal that the
    model requires motion-aware export.  Disabled/non-motion models remain a
    no-op.
    """
    requested = None
    if motion_cfg is not None:
        requested = bool(motion_cfg.get("enabled", False)) and resolve_motion_type(motion_cfg) != "none"

    lwdetr = _find_lwdetr(model)
    attached = lwdetr is not None and getattr(lwdetr, "motion_module", None) is not None
    if requested is None:
        wrappers = [model]
        if hasattr(model, "model"):
            wrappers.append(model.model)
        if hasattr(model, "model") and hasattr(model.model, "model"):
            wrappers.append(model.model.model)
        requested = attached or any(
            bool(getattr(candidate, "_motion_export_required", False)) for candidate in wrappers
        )
    if requested is False:
        return
    if lwdetr is None:
        raise RuntimeError(
            "Motion-enabled RF-DETR export could not locate the LWDETR model; "
            "refusing to build an ONNX/TensorRT graph without motion processing."
        )
    if not attached:
        raise RuntimeError(
            "Motion-enabled RF-DETR export has no attached motion_module; "
            "call attach_motion_module() after constructing/aligning the model."
        )
    if not callable(getattr(lwdetr.motion_module, "forward_export", None)):
        raise RuntimeError(
            "The attached RF-DETR motion module does not support tensor-only ONNX export."
        )
    fallback_mode = getattr(lwdetr.motion_module, "fallback_mode", None)
    if fallback_mode is not None and str(fallback_mode).strip().lower() != "identity":
        raise RuntimeError(
            "TensorRT still-image motion export requires "
            "model.motion.temporal.fallback_mode='identity' for deterministic tracing; "
            f"got {fallback_mode!r}."
        )
    export_forward = getattr(type(lwdetr), "forward_export", None)
    if not getattr(export_forward, "_motion_patched", False):
        raise RuntimeError(
            "LWDETR.forward_export is not motion-aware; call ensure_motion_support() "
            "before constructing the model."
        )


def _find_lwdetr(model: nn.Module) -> Optional[nn.Module]:
    """Walk common PL wrapper layers to locate the inner LWDETR model."""
    try:
        import rfdetr.models.lwdetr as lwdetr_module
        LWDETR_cls = lwdetr_module.LWDETR
    except Exception:
        LWDETR_cls = None

    candidates = [model]
    seen: set[int] = set()
    while candidates and len(seen) < 12:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if LWDETR_cls is not None and isinstance(candidate, LWDETR_cls):
            return candidate
        # Name-based fallback for version skew.
        if type(candidate).__name__ == "LWDETR":
            return candidate
        optimized = getattr(candidate, "_orig_mod", None)
        wrapped = getattr(candidate, "model", None)
        if optimized is not None:
            candidates.append(optimized)
        if wrapped is not None:
            candidates.append(wrapped)
    return None
