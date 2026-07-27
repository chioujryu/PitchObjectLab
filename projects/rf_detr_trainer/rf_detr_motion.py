"""Optional three-frame TrackNetV5 branch for RF-DETR.

The enabled path runs the shared RF-DETR backbone over ``B*T`` real RGB frames,
keeps all three temporal feature maps for TrackNet, and sends only the centre
frame through the RF-DETR transformer/decoder. TrackNet provides:

* four luminance Motion Direction Decoupling maps for both adjacent pairs;
* per-frame draft heatmaps and factorised spatial/temporal R-STR refinement;
* bbox-Gaussian targets, weighted focal BCE, and single/all peak extraction;
* a zero-initialised heatmap residual fused into the highest-resolution centre
  feature, preserving stock detector behaviour at initialisation.

Integration is instance-bound. ``attach_motion_module`` adds the module and a
5-D/TemporalBatch dispatcher only to the requested LWDETR instance; disabled
models import and execute stock RF-DETR without process-global patches or
settings. The implementation is hard-pinned to ``rfdetr==1.8.3``.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'assert_motion_checkpoint_compatible',
    "ensure_motion_support",
    "attach_motion_module",
    "load_motion_checkpoint_weights",
    "assert_motion_export_ready",
    "apply_motion_overrides",
    "is_patched",
    "MotionModule",
    "MotionOutput",
    "MotionDirectionDecoupling",
    "TemporalRSTRHead",
    "build_gaussian_heatmap_targets",
    "weighted_heatmap_bce",
    "extract_heatmap_peaks",
    "run_temporal_lwdetr",
]

# Convenience overrides that map to real ModelConfig fields.
_BASELINE_RFDETR_VERSION = '1.8.3'
_TRACKNET_ARCHITECTURE_SCHEMA_VERSION = 2

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
        "mode": "real",
        "num_frames": 3,
        "frame_stride": 1,
        "anchor": "center",
        "boundary_policy": "drop",
        "allow_single_frame_fallback": False,
        # Kept for old configs; real temporal input is the default.
        "fallback_mode": "real",
        "noise_std": 0.02,
    },
    "focus": {
        "mode": "all",
        "primary_field": "primary_label_index",
    },
    "tracknet_v5": {
        "feature_source": "all_frames",
        "feature_level": "highest_resolution",
        "mdd": {
            "enabled": True,
            "polarity_channels": 4,
            "attention": {
                "learnable": True,
                "init_k": 1.0,
                "init_m": 0.5,
            },
        },
        "heatmap": {
            "target": "bbox_gaussian",
            "min_sigma": 1.0,
            "peak_threshold": 0.5,
            "peak_nms_kernel": 3,
            "max_peaks": 20,
        },
        "rstr": {
            "enabled": True,
            "num_blocks": 2,
            "hidden_dim": 256,
            "num_heads": 8,
            "attention_mode": "divided",
            "dropout": 0.1,
            "use_pixel_shuffle": True,
            "context_mask_prob": 0.1,
        },
        "fusion": {"mode": "zero_init_residual"},
    },
    "loss": {
        "heatmap_weight": 1.0,
        "gamma": 2.0,
    },
    "overrides": {},
}


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
        raise RuntimeError(
            "TrackNet temporal integration is version-pinned to "
            f"rfdetr=={_BASELINE_RFDETR_VERSION}, but found {version}. "
            "Refusing to run against an unverified LWDETR forward contract."
        )


def _validated_motion_config(motion_cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge defaults and reject temporal graph options not implemented here."""

    merged = _deep_merge(deepcopy(_MOTION_DEFAULTS), dict(motion_cfg or {}))
    temporal = merged["temporal"]
    tracknet = merged["tracknet_v5"]
    rstr = tracknet["rstr"]
    supported = (
        (
            "model.motion.temporal.anchor",
            str(temporal.get("anchor", "")).lower(),
            "center",
        ),
        (
            "model.motion.temporal.boundary_policy",
            str(temporal.get("boundary_policy", "")).lower(),
            "drop",
        ),
        (
            "model.motion.tracknet_v5.feature_source",
            str(tracknet.get("feature_source", "")).lower(),
            "all_frames",
        ),
        (
            "model.motion.tracknet_v5.feature_level",
            str(tracknet.get("feature_level", "")).lower(),
            "highest_resolution",
        ),
        (
            "model.motion.tracknet_v5.rstr.attention_mode",
            str(rstr.get("attention_mode", "")).lower(),
            "divided",
        ),
    )
    for field, value, expected in supported:
        if value != expected:
            raise ValueError(
                f"{field} must be {expected!r} for the current TrackNetV5 "
                f"temporal graph, got {value!r}."
            )
    return merged


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
    """TrackNetV5 luminance MDD for a three-frame window.

    The four output planes are, in order, positive and negative luminance
    changes for ``previous -> centre`` followed by positive and negative
    changes for ``centre -> next``. Multiplicative learnable attention keeps
    exact zero motion at zero instead of turning it into a sigmoid constant.
    """

    LUMA_WEIGHTS: Tuple[float, float, float] = (0.299, 0.587, 0.114)

    def __init__(
        self,
        in_channels: int = 3,
        polarity_channels: int = 4,
        init_k: float = 1.0,
        init_m: float = 0.5,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"MDD requires RGB input (3 channels), got {in_channels}.")
        if polarity_channels != 4:
            raise ValueError(
                "TrackNetV5 MDD emits exactly four polarity maps "
                f"(two adjacent pairs x two polarities), got {polarity_channels}."
            )
        self.polarity_channels = 4
        self.learnable = bool(learnable)
        self.register_buffer(
            "luma_weights",
            torch.tensor(self.LUMA_WEIGHTS, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.attention = (
            LearnableSigmoidAttention(4, init_k, init_m)
            if self.learnable
            else nn.Identity()
        )

    def raw_polarities(self, frames: torch.Tensor) -> torch.Tensor:
        """Return unnormalised ``[B, 4, H, W]`` luminance polarity fields."""
        if frames.ndim != 5:
            raise ValueError(
                "MDD expects [B, 3, 3, H, W] frames, "
                f"received shape {tuple(frames.shape)}."
            )
        if frames.shape[1] != 3 or frames.shape[2] != 3:
            raise ValueError(
                "MDD requires exactly three RGB frames, "
                f"received shape {tuple(frames.shape)}."
            )
        weights = self.luma_weights.to(device=frames.device, dtype=frames.dtype)
        luminance = (frames * weights).sum(dim=2)
        previous_to_centre = luminance[:, 1] - luminance[:, 0]
        centre_to_next = luminance[:, 2] - luminance[:, 1]
        return torch.stack(
            (
                F.relu(previous_to_centre),
                F.relu(-previous_to_centre),
                F.relu(centre_to_next),
                F.relu(-centre_to_next),
            ),
            dim=1,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        raw = self.raw_polarities(frames)
        if not self.learnable:
            return raw
        return raw * self.attention(raw)


class RSTRSpatialAttention(nn.Module):
    """Pre-norm transformer block used for both spatial and temporal tokens."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"R-STR hidden_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
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
        normalized = self.norm(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + self.dropout(attended)
        return x + self.ffn(self.norm2(x))


class TemporalDraftHeatmapHead(nn.Module):
    """Produce one draft heatmap logit plane per temporal feature map."""

    def __init__(self, in_channels: int, num_frames: int, hidden_dim: int) -> None:
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.num_frames = int(num_frames)
        self.feature_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.motion_to_time = nn.Conv2d(4, self.num_frames, 3, padding=1)

    def forward(
        self,
        temporal_feature: torch.Tensor,
        motion_maps: torch.Tensor,
    ) -> torch.Tensor:
        if temporal_feature.ndim != 5:
            raise ValueError(
                "Draft heatmap head expects [B, T, C, H, W], "
                f"got {tuple(temporal_feature.shape)}."
            )
        batch, frames, channels, height, width = temporal_feature.shape
        if frames != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got T={frames}.")
        feature_logits = self.feature_head(
            temporal_feature.reshape(batch * frames, channels, height, width)
        ).reshape(batch, frames, height, width)
        resized_motion = F.interpolate(
            motion_maps, size=(height, width), mode="bilinear", align_corners=False
        )
        return feature_logits + self.motion_to_time(resized_motion)


class TemporalRSTRHead(nn.Module):
    """Factorised spatial/temporal refinement of three draft heatmaps.

    The final correction projection is zero-initialised, so the module begins
    as the exact identity in logit space. With PixelShuffle enabled the patch
    embedding downsamples by two and the residual projection restores the
    original resolution.
    """

    def __init__(
        self,
        num_frames: int = 3,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 2,
        dropout: float = 0.1,
        use_pixel_shuffle: bool = True,
        context_mask_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if num_frames != 3:
            raise ValueError(f"TrackNetV5 R-STR requires exactly three frames, got {num_frames}.")
        if not 0.0 <= context_mask_prob < 1.0:
            raise ValueError("context_mask_prob must be in [0, 1).")
        self.num_frames = num_frames
        self.context_mask_prob = float(context_mask_prob)
        self.use_pixel_shuffle = bool(use_pixel_shuffle)
        self.patch_scale = 2 if self.use_pixel_shuffle else 1
        self.draft_embed = nn.Conv2d(
            1,
            hidden_dim,
            kernel_size=self.patch_scale,
            stride=self.patch_scale,
        )
        self.motion_embed = nn.Conv2d(4, hidden_dim, 3, padding=1)
        self.spatial_blocks = nn.ModuleList(
            [RSTRSpatialAttention(hidden_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.temporal_blocks = nn.ModuleList(
            [RSTRSpatialAttention(hidden_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        output_channels = self.patch_scale * self.patch_scale
        self.residual_projection = nn.Conv2d(
            hidden_dim, output_channels, 3, padding=1
        )
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(
        self,
        draft_logits: torch.Tensor,
        motion_maps: torch.Tensor,
    ) -> torch.Tensor:
        if draft_logits.ndim != 4:
            raise ValueError(
                "R-STR expects [B, T, H, W] draft logits, "
                f"got {tuple(draft_logits.shape)}."
            )
        batch, frames, height, width = draft_logits.shape
        if frames != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got T={frames}.")
        pad_height = (-height) % self.patch_scale
        pad_width = (-width) % self.patch_scale
        padded = F.pad(draft_logits, (0, pad_width, 0, pad_height))
        padded_height, padded_width = padded.shape[-2:]
        embedded = self.draft_embed(
            padded.reshape(batch * frames, 1, padded_height, padded_width)
        )
        token_height, token_width = embedded.shape[-2:]
        embedded = embedded.reshape(batch, frames, -1, token_height, token_width)

        motion_context = self.motion_embed(
            F.interpolate(
                motion_maps,
                size=(padded_height, padded_width),
                mode="bilinear",
                align_corners=False,
            )
        )
        motion_context = F.interpolate(
            motion_context,
            size=(token_height, token_width),
            mode="bilinear",
            align_corners=False,
        )
        embedded = embedded + motion_context.unsqueeze(1)

        if self.training and self.context_mask_prob > 0.0:
            keep = (
                torch.rand(batch, frames, 1, 1, 1, device=embedded.device)
                >= self.context_mask_prob
            )
            # Always retain the centre draft; only temporal context is dropped.
            keep[:, frames // 2] = True
            embedded = embedded * keep.to(embedded.dtype)

        hidden_dim = embedded.shape[2]
        for spatial, temporal in zip(self.spatial_blocks, self.temporal_blocks):
            spatial_tokens = embedded.reshape(
                batch * frames, hidden_dim, token_height * token_width
            ).transpose(1, 2)
            spatial_tokens = spatial(spatial_tokens)
            embedded = spatial_tokens.transpose(1, 2).reshape(
                batch, frames, hidden_dim, token_height, token_width
            )

            temporal_tokens = embedded.permute(0, 3, 4, 1, 2).reshape(
                batch * token_height * token_width, frames, hidden_dim
            )
            temporal_tokens = temporal(temporal_tokens)
            embedded = temporal_tokens.reshape(
                batch, token_height, token_width, frames, hidden_dim
            ).permute(0, 3, 4, 1, 2)

        residual = self.residual_projection(
            embedded.reshape(batch * frames, hidden_dim, token_height, token_width)
        )
        if self.use_pixel_shuffle:
            residual = F.pixel_shuffle(residual, self.patch_scale)
        elif residual.shape[-2:] != (padded_height, padded_width):
            residual = F.interpolate(
                residual,
                size=(padded_height, padded_width),
                mode="bilinear",
                align_corners=False,
            )
        residual = residual.reshape(batch, frames, padded_height, padded_width)
        residual = residual[..., :height, :width]
        return draft_logits + residual


class RSTRHead(TemporalRSTRHead):
    """Compatibility wrapper treating input channels as temporal drafts.

    New TrackNet code uses :class:`TemporalRSTRHead` directly. This wrapper is
    retained for imports from older integrations and returns probabilities as
    the previous class did.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 2,
        dropout: float = 0.1,
        use_pixel_shuffle: bool = True,
        context_mask_prob: float = 0.1,
    ) -> None:
        if in_channels != 3:
            raise ValueError(
                "RSTRHead now refines exactly three temporal draft heatmaps; "
                f"use in_channels=3, got {in_channels}."
            )
        super().__init__(
            num_frames=3,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
            use_pixel_shuffle=use_pixel_shuffle,
            context_mask_prob=context_mask_prob,
        )

    def forward(
        self,
        feat: torch.Tensor,
        motion_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if motion_maps is None:
            motion_maps = feat.new_zeros((feat.shape[0], 4, *feat.shape[-2:]))
        return torch.sigmoid(super().forward(feat, motion_maps))


class HeatmapResidualFusion(nn.Module):
    """Add a heatmap-conditioned residual to one RF-DETR feature level."""

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, feature_channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, feature: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        if heatmap.shape[-2:] != feature.shape[-2:]:
            heatmap = F.interpolate(
                heatmap, size=feature.shape[-2:], mode="bilinear", align_corners=False
            )
        return feature + self.projection(heatmap)


@dataclass
class MotionOutput:
    """TrackNet outputs consumed by the temporal LWDETR adapter and losses."""

    features: List[torch.Tensor]
    draft_heatmap_logits: torch.Tensor
    heatmap_logits: torch.Tensor
    heatmaps: torch.Tensor
    motion_maps: torch.Tensor
    feature_level: int


def _normalise_temporal_targets(
    targets: Sequence[Any] | Mapping[str, Any],
) -> List[List[Mapping[str, Any]]]:
    if isinstance(targets, Mapping):
        return [[targets]]
    targets_list = list(targets)
    if not targets_list:
        return []
    if isinstance(targets_list[0], Mapping):
        return [[item for item in targets_list]]
    normalised: List[List[Mapping[str, Any]]] = []
    for sample in targets_list:
        frames = list(sample)
        if not all(isinstance(item, Mapping) for item in frames):
            raise TypeError("Every temporal target must be a mapping.")
        normalised.append(frames)
    return normalised


def _target_image_size(
    target: Mapping[str, Any],
    image_size: Optional[Tuple[int, int]],
) -> Tuple[float, float]:
    size = image_size if image_size is not None else target.get("size")
    if size is None:
        raise ValueError("Pixel-coordinate boxes require image_size or target['size'].")
    if torch.is_tensor(size):
        size = size.detach().cpu().tolist()
    if len(size) != 2:
        raise ValueError(f"Expected image size (height, width), got {size!r}.")
    return float(size[0]), float(size[1])


def build_gaussian_heatmap_targets(
    targets: Sequence[Any] | Mapping[str, Any],
    output_size: Tuple[int, int],
    *,
    image_size: Optional[Tuple[int, int]] = None,
    focus_mode: str = "all",
    primary_field: str = "primary_label_index",
    min_sigma: float = 1.0,
) -> torch.Tensor:
    """Rasterise RF-DETR boxes into soft Gaussian TrackNet targets.

    ``targets`` is normally ``[batch][time]`` mappings. Boxes default to
    normalized ``cxcywh``; set ``box_format`` to ``xyxy_normalized``, ``cxcywh``
    or ``xyxy`` for the other supported representations.
    """
    focus_mode = str(focus_mode).lower()
    if focus_mode not in {"single", "all"}:
        raise ValueError("focus_mode must be 'single' or 'all'.")
    height, width = (int(output_size[0]), int(output_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"output_size must be positive, got {output_size!r}.")
    if min_sigma <= 0:
        raise ValueError("min_sigma must be positive.")

    nested_targets = _normalise_temporal_targets(targets)
    if not nested_targets:
        return torch.empty((0, 0, height, width), dtype=torch.float32)
    frame_count = len(nested_targets[0])
    if frame_count == 0 or any(len(sample) != frame_count for sample in nested_targets):
        raise ValueError("Every sample must provide the same non-zero number of frames.")

    reference_boxes = next(
        (
            item.get("boxes")
            for sample in nested_targets
            for item in sample
            if torch.is_tensor(item.get("boxes"))
        ),
        None,
    )
    device = reference_boxes.device if reference_boxes is not None else torch.device("cpu")
    dtype = (
        reference_boxes.dtype
        if reference_boxes is not None and reference_boxes.is_floating_point()
        else torch.float32
    )
    heatmaps = torch.zeros(
        (len(nested_targets), frame_count, height, width), device=device, dtype=dtype
    )
    grid_y = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    grid_x = torch.arange(width, device=device, dtype=dtype).view(1, width)

    for batch_index, sample in enumerate(nested_targets):
        for frame_index, target in enumerate(sample):
            boxes = target.get("boxes")
            if boxes is None:
                boxes = torch.empty((0, 4), device=device, dtype=dtype)
            boxes = torch.as_tensor(boxes, device=device, dtype=dtype).reshape(-1, 4)
            if boxes.numel() == 0:
                continue
            selected = torch.arange(boxes.shape[0], device=device)
            if focus_mode == "single":
                if boxes.shape[0] == 1:
                    selected = selected[:1]
                else:
                    primary = target.get(primary_field)
                    if primary is None and isinstance(target.get("metadata"), Mapping):
                        primary = target["metadata"].get(primary_field)
                    if torch.is_tensor(primary):
                        primary = int(primary.item())
                    if primary is None:
                        raise ValueError(
                            f"single focus found {boxes.shape[0]} boxes without {primary_field!r}."
                        )
                    primary = int(primary)
                    if primary < 0 or primary >= boxes.shape[0]:
                        raise ValueError(
                            f"{primary_field}={primary} is outside [0, {boxes.shape[0] - 1}]."
                        )
                    selected = selected.new_tensor([primary])
            boxes = boxes[selected]

            box_format = str(target.get("box_format", "cxcywh_normalized")).lower()
            if box_format in {"cxcywh_normalized", "cxcywhn"}:
                centres_x = boxes[:, 0] * width
                centres_y = boxes[:, 1] * height
                box_widths = boxes[:, 2] * width
                box_heights = boxes[:, 3] * height
            elif box_format in {"xyxy_normalized", "xyxyn"}:
                centres_x = (boxes[:, 0] + boxes[:, 2]) * 0.5 * width
                centres_y = (boxes[:, 1] + boxes[:, 3]) * 0.5 * height
                box_widths = (boxes[:, 2] - boxes[:, 0]).abs() * width
                box_heights = (boxes[:, 3] - boxes[:, 1]).abs() * height
            elif box_format in {"cxcywh", "xyxy"}:
                source_height, source_width = _target_image_size(target, image_size)
                if box_format == "cxcywh":
                    centres_x = boxes[:, 0] * width / source_width
                    centres_y = boxes[:, 1] * height / source_height
                    box_widths = boxes[:, 2].abs() * width / source_width
                    box_heights = boxes[:, 3].abs() * height / source_height
                else:
                    centres_x = (boxes[:, 0] + boxes[:, 2]) * 0.5 * width / source_width
                    centres_y = (boxes[:, 1] + boxes[:, 3]) * 0.5 * height / source_height
                    box_widths = (boxes[:, 2] - boxes[:, 0]).abs() * width / source_width
                    box_heights = (boxes[:, 3] - boxes[:, 1]).abs() * height / source_height
            else:
                raise ValueError(f"Unsupported box_format {box_format!r}.")

            sigma_x = (box_widths / 6.0).clamp_min(float(min_sigma))
            sigma_y = (box_heights / 6.0).clamp_min(float(min_sigma))
            frame_heatmap = heatmaps[batch_index, frame_index]
            for centre_x, centre_y, sx, sy in zip(
                centres_x, centres_y, sigma_x, sigma_y
            ):
                gaussian = torch.exp(
                    -0.5
                    * (
                        ((grid_x - centre_x) / sx).square()
                        + ((grid_y - centre_y) / sy).square()
                    )
                )
                frame_heatmap.copy_(torch.maximum(frame_heatmap, gaussian))
    return heatmaps


def weighted_heatmap_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    positive_weight: Optional[float] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Class-balanced focal BCE over soft Gaussian heatmap targets."""
    if logits.shape != targets.shape:
        raise ValueError(
            f"Heatmap logits/targets must have equal shapes, got {logits.shape} and {targets.shape}."
        )
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    probabilities = logits.sigmoid()
    if positive_weight is None:
        positive_mass = targets.detach().sum().clamp_min(1.0)
        negative_mass = (1.0 - targets.detach()).sum()
        positive_weight_tensor = (negative_mass / positive_mass).clamp(1.0, 100.0)
    else:
        if positive_weight <= 0:
            raise ValueError("positive_weight must be positive.")
        positive_weight_tensor = logits.new_tensor(float(positive_weight))
    focal_weights = (
        targets * (1.0 - probabilities).pow(gamma) * positive_weight_tensor
        + (1.0 - targets) * probabilities.pow(gamma)
    )
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    losses = losses * focal_weights
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'sum', or 'mean'.")
    return losses.sum() / focal_weights.sum().clamp_min(1.0)


def extract_heatmap_peaks(
    heatmaps: torch.Tensor,
    *,
    focus_mode: str = "all",
    threshold: float = 0.5,
    nms_kernel: int = 3,
    max_peaks: int = 20,
    from_logits: bool = False,
) -> List[List[torch.Tensor]]:
    """Extract local maxima as per-frame ``[x, y, score]`` tensors."""
    if heatmaps.ndim == 3:
        heatmaps = heatmaps.unsqueeze(1)
    if heatmaps.ndim != 4:
        raise ValueError(f"Expected [B, T, H, W], got {tuple(heatmaps.shape)}.")
    focus_mode = str(focus_mode).lower()
    if focus_mode not in {"single", "all"}:
        raise ValueError("focus_mode must be 'single' or 'all'.")
    if nms_kernel < 1 or nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer.")
    if max_peaks <= 0:
        raise ValueError("max_peaks must be positive.")
    probabilities = heatmaps.sigmoid() if from_logits else heatmaps
    pooled = F.max_pool2d(
        probabilities.reshape(-1, 1, *probabilities.shape[-2:]),
        kernel_size=nms_kernel,
        stride=1,
        padding=nms_kernel // 2,
    ).reshape_as(probabilities)
    local = (probabilities >= threshold) & (probabilities == pooled)
    result: List[List[torch.Tensor]] = []
    limit = 1 if focus_mode == "single" else max_peaks
    for sample_scores, sample_local in zip(probabilities, local):
        sample_result: List[torch.Tensor] = []
        for frame_scores, frame_local in zip(sample_scores, sample_local):
            coordinates = frame_local.nonzero(as_tuple=False)
            if coordinates.numel() == 0:
                sample_result.append(frame_scores.new_empty((0, 3)))
                continue
            scores = frame_scores[coordinates[:, 0], coordinates[:, 1]]
            count = min(limit, scores.numel())
            top_scores, order = scores.topk(count, largest=True, sorted=True)
            top_coordinates = coordinates[order]
            sample_result.append(
                torch.stack(
                    (
                        top_coordinates[:, 1].to(top_scores.dtype),
                        top_coordinates[:, 0].to(top_scores.dtype),
                        top_scores,
                    ),
                    dim=1,
                )
            )
        result.append(sample_result)
    return result


class MotionModule(nn.Module):
    """Three-frame TrackNetV5 branch attached to one LWDETR instance."""

    def __init__(
        self,
        feature_channels_per_scale: List[int],
        motion_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()
        if not feature_channels_per_scale:
            raise ValueError("MotionModule requires at least one backbone feature level.")
        self.feature_channels_per_scale = [int(channel) for channel in feature_channels_per_scale]
        merged_cfg = _validated_motion_config(motion_cfg)
        temporal_cfg = merged_cfg["temporal"]
        focus_cfg = merged_cfg["focus"]
        v5_cfg = merged_cfg["tracknet_v5"]
        mdd_cfg = v5_cfg["mdd"]
        heatmap_cfg = v5_cfg["heatmap"]
        rstr_cfg = v5_cfg["rstr"]
        loss_cfg = merged_cfg["loss"]

        self.num_frames = int(temporal_cfg.get("num_frames", 3))
        if self.num_frames != 3:
            raise ValueError(
                f"TrackNetV5 temporal integration requires num_frames=3, got {self.num_frames}."
            )
        self.frame_stride = int(temporal_cfg.get("frame_stride", 1))
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive.")
        self.anchor_index = 1
        anchor = str(temporal_cfg.get("anchor", "center")).lower()
        if anchor != "center":
            raise ValueError("Only centre-frame TrackNet/RF-DETR alignment is supported.")
        self.fallback_mode = str(temporal_cfg.get("fallback_mode", "real")).lower()
        self.noise_std = float(temporal_cfg.get("noise_std", 0.02))
        self.allow_single_frame_fallback = bool(
            temporal_cfg.get("allow_single_frame_fallback", False)
            or self.fallback_mode in {"identity", "zero", "noise"}
        )

        self.focus_mode = str(focus_cfg.get("mode", "all")).lower()
        if self.focus_mode not in {"single", "all"}:
            raise ValueError("model.motion.focus.mode must be 'single' or 'all'.")
        self.primary_field = str(
            focus_cfg.get("primary_field", "primary_label_index")
        )
        self.min_sigma = float(heatmap_cfg.get("min_sigma", 1.0))
        self.peak_threshold = float(heatmap_cfg.get("peak_threshold", 0.5))
        self.peak_nms_kernel = int(heatmap_cfg.get("peak_nms_kernel", 3))
        self.max_peaks = int(heatmap_cfg.get("max_peaks", 20))
        self.heatmap_weight = float(loss_cfg.get("heatmap_weight", 1.0))
        self.heatmap_gamma = float(loss_cfg.get("gamma", 2.0))

        self.mdd_enabled = bool(mdd_cfg.get("enabled", True))
        attention_cfg = mdd_cfg.get("attention", {}) or {}
        self.mdd = (
            MotionDirectionDecoupling(
                in_channels=3,
                polarity_channels=int(mdd_cfg.get("polarity_channels", 4)),
                init_k=float(attention_cfg.get("init_k", 1.0)),
                init_m=float(attention_cfg.get("init_m", 0.5)),
                learnable=bool(attention_cfg.get("learnable", True)),
            )
            if self.mdd_enabled
            else None
        )

        rstr_hidden = int(rstr_cfg.get("hidden_dim", 256))
        self.draft_heads = nn.ModuleList(
            [
                TemporalDraftHeatmapHead(
                    channel,
                    self.num_frames,
                    min(rstr_hidden, max(16, channel // 2)),
                )
                for channel in self.feature_channels_per_scale
            ]
        )
        self.rstr_enabled = bool(rstr_cfg.get("enabled", True))
        self.rstr = (
            TemporalRSTRHead(
                num_frames=self.num_frames,
                hidden_dim=rstr_hidden,
                num_heads=int(rstr_cfg.get("num_heads", 8)),
                num_blocks=int(rstr_cfg.get("num_blocks", 2)),
                dropout=float(rstr_cfg.get("dropout", 0.1)),
                use_pixel_shuffle=bool(rstr_cfg.get("use_pixel_shuffle", True)),
                context_mask_prob=float(rstr_cfg.get("context_mask_prob", 0.1)),
            )
            if self.rstr_enabled
            else None
        )
        fusion_mode = str((v5_cfg.get("fusion", {}) or {}).get("mode", "zero_init_residual"))
        if fusion_mode != "zero_init_residual":
            raise ValueError(
                "Only tracknet_v5.fusion.mode='zero_init_residual' is supported."
            )
        self.fusions = nn.ModuleList(
            [HeatmapResidualFusion(channel) for channel in self.feature_channels_per_scale]
        )
        self.last_output: Optional[MotionOutput] = None

    def _make_frame_window(self, images: torch.Tensor) -> torch.Tensor:
        """Validate real input or explicitly synthesize an allowed still-image window."""
        if images.ndim == 5:
            if images.shape[1] != self.num_frames or images.shape[2] != 3:
                raise ValueError(
                    f"Expected [B, {self.num_frames}, 3, H, W], got {tuple(images.shape)}."
                )
            return images
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB images or temporal frames, got {tuple(images.shape)}.")
        if not self.allow_single_frame_fallback:
            raise RuntimeError(
                "TrackNet is enabled but received one frame. Set "
                "temporal.allow_single_frame_fallback=true explicitly to replicate it."
            )
        batch = images.shape[0]
        if self.fallback_mode == "noise":
            frames = [images]
            for _ in range(self.num_frames - 1):
                frames.append(images + torch.randn_like(images) * self.noise_std)
            return torch.stack(frames, dim=1)
        return images.unsqueeze(1).expand(batch, self.num_frames, -1, -1, -1)

    @staticmethod
    def _validate_temporal_features(
        frames: torch.Tensor,
        features: Sequence[torch.Tensor],
        expected_channels: Sequence[int],
    ) -> None:
        if len(features) != len(expected_channels):
            raise RuntimeError(
                f"Backbone returned {len(features)} feature levels; expected {len(expected_channels)}."
            )
        for index, (feature, channels) in enumerate(zip(features, expected_channels)):
            if feature.ndim != 5:
                raise ValueError(
                    f"Temporal feature {index} must be [B, T, C, H, W], got {tuple(feature.shape)}."
                )
            if feature.shape[:2] != frames.shape[:2] or feature.shape[2] != channels:
                raise ValueError(
                    f"Temporal feature {index} shape {tuple(feature.shape)} does not match "
                    f"frames {tuple(frames.shape)} and C={channels}."
                )

    def forward_temporal(
        self,
        frames: torch.Tensor,
        temporal_features: Sequence[torch.Tensor],
    ) -> MotionOutput:
        """Run MDD/heatmap/R-STR and return centre-frame RF-DETR features."""
        frames = self._make_frame_window(frames)
        self._validate_temporal_features(
            frames, temporal_features, self.feature_channels_per_scale
        )
        if self.mdd is None:
            motion_maps = frames.new_zeros(
                (frames.shape[0], 4, frames.shape[-2], frames.shape[-1])
            )
        else:
            motion_maps = self.mdd(frames)

        feature_level = max(
            range(len(temporal_features)),
            key=lambda index: int(temporal_features[index].shape[-2])
            * int(temporal_features[index].shape[-1]),
        )
        draft_low_resolution = self.draft_heads[feature_level](
            temporal_features[feature_level], motion_maps
        )
        heatmap_low_resolution = (
            self.rstr(draft_low_resolution, motion_maps)
            if self.rstr is not None
            else draft_low_resolution
        )
        output_size = (int(frames.shape[-2]), int(frames.shape[-1]))
        draft_logits = F.interpolate(
            draft_low_resolution,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        heatmap_logits = F.interpolate(
            heatmap_low_resolution,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        heatmaps = heatmap_logits.sigmoid()

        centre_features = [
            feature[:, self.anchor_index].contiguous() for feature in temporal_features
        ]
        centre_features[feature_level] = self.fusions[feature_level](
            centre_features[feature_level],
            heatmaps[:, self.anchor_index : self.anchor_index + 1],
        )
        output = MotionOutput(
            features=centre_features,
            draft_heatmap_logits=draft_logits,
            heatmap_logits=heatmap_logits,
            heatmaps=heatmaps,
            motion_maps=motion_maps,
            feature_level=feature_level,
        )
        self.last_output = output
        return output

    def build_heatmap_targets(
        self,
        targets: Sequence[Any] | Mapping[str, Any],
        output_size: Tuple[int, int],
        *,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        return build_gaussian_heatmap_targets(
            targets,
            output_size,
            image_size=image_size,
            focus_mode=self.focus_mode,
            primary_field=self.primary_field,
            min_sigma=self.min_sigma,
        )

    def heatmap_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.heatmap_weight * weighted_heatmap_bce(
            logits, targets, gamma=self.heatmap_gamma
        )

    def extract_peaks(self, heatmaps: torch.Tensor) -> List[List[torch.Tensor]]:
        return extract_heatmap_peaks(
            heatmaps,
            focus_mode=self.focus_mode,
            threshold=self.peak_threshold,
            nms_kernel=self.peak_nms_kernel,
            max_peaks=self.max_peaks,
        )

    def forward(self, images: torch.Tensor, features: list) -> list:
        feature_tensors = [nested.tensors for nested in features]
        modulated_tensors = self.forward_export(images, feature_tensors)
        return [
            _rebuild_nested_tensor(src, nested.mask)
            for src, nested in zip(modulated_tensors, features)
        ]

    def forward_export(
        self,
        images: torch.Tensor,
        features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Compatibility path for still-image export and legacy callers.

        Real temporal execution must pass one feature tensor per frame through
        :meth:`forward_temporal`; a 4-D feature is repeated only when explicit
        single-frame fallback is enabled.
        """
        frames = self._make_frame_window(images)
        temporal_features: List[torch.Tensor] = []
        for feature, channels in zip(features, self.feature_channels_per_scale):
            if feature.ndim == 5:
                temporal_features.append(feature)
                continue
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f"Expected feature [B, {channels}, H, W], got {tuple(feature.shape)}."
                )
            temporal_features.append(
                feature.unsqueeze(1).expand(-1, self.num_frames, -1, -1, -1)
            )
        return self.forward_temporal(frames, temporal_features).features


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


def _select_temporal_tensor(
    tensor: torch.Tensor,
    batch_size: int,
    num_frames: int,
    frame_index: int,
) -> torch.Tensor:
    if tensor.shape[0] != batch_size * num_frames:
        raise RuntimeError(
            "RF-DETR backbone temporal batch mismatch: "
            f"first dimension is {tensor.shape[0]}, expected {batch_size * num_frames}."
        )
    return tensor.reshape(batch_size, num_frames, *tensor.shape[1:])[:, frame_index]


def run_temporal_lwdetr(
    lwdetr: nn.Module,
    frames: Any,
    padding_masks: Optional[torch.Tensor] = None,
    targets: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    """Run a three-frame batch through one attached LWDETR instance.

    ``frames`` may be a tensor or a batch object exposing ``.frames`` and
    optionally ``.padding_masks``. The backbone sees all ``B*T`` frames in one
    call, the TrackNet branch receives reshaped temporal features, and stock
    LWDETR continuation receives only centre-frame features. No class-level or
    process-global method is patched.
    """
    mdd_frames = frames if torch.is_tensor(frames) else None
    if not torch.is_tensor(frames):
        batch_object = frames
        frames = getattr(batch_object, "frames", None)
        if frames is None:
            raise TypeError("Temporal input must be a tensor or expose a .frames tensor.")
        mdd_frames = getattr(batch_object, "mdd_frames", frames)
        if padding_masks is None:
            padding_masks = getattr(batch_object, "padding_masks", None)
        if targets is None:
            targets = getattr(batch_object, "detection_targets", None)
        if targets is None:
            targets = getattr(batch_object, "anchor_targets", None)
    if frames.ndim != 5 or frames.shape[1:3] != (3, 3):
        raise ValueError(
            "Temporal LWDETR expects [B, 3, 3, H, W], "
            f"received {tuple(frames.shape)}."
        )
    if not torch.is_tensor(mdd_frames) or mdd_frames.shape != frames.shape:
        shape = getattr(mdd_frames, "shape", None)
        raise ValueError(
            "TemporalBatch.mdd_frames must be a tensor matching .frames, "
            f"got {shape!r} vs {tuple(frames.shape)}."
        )
    mdd_frames = mdd_frames.to(device=frames.device, dtype=frames.dtype)
    motion_module = getattr(lwdetr, "motion_module", None)
    if not isinstance(motion_module, MotionModule):
        raise RuntimeError(
            "LWDETR has no attached MotionModule; call attach_motion_module() first."
        )
    batch_size, num_frames, channels, height, width = frames.shape
    if padding_masks is None:
        padding_masks = torch.zeros(
            (batch_size, num_frames, height, width),
            dtype=torch.bool,
            device=frames.device,
        )
    if padding_masks.shape != (batch_size, num_frames, height, width):
        raise ValueError(
            "padding_masks must be [B, T, H, W], "
            f"got {tuple(padding_masks.shape)}."
        )
    padding_masks = padding_masks.to(device=frames.device, dtype=torch.bool)
    flattened_samples = _rebuild_nested_tensor(
        frames.reshape(batch_size * num_frames, channels, height, width),
        padding_masks.reshape(batch_size * num_frames, height, width),
    )
    backbone_result = lwdetr.backbone(flattened_samples)
    if not isinstance(backbone_result, tuple) or len(backbone_result) != 3:
        raise RuntimeError(
            "rfdetr==1.8.3 backbone must return (features, positions, cross_attn_features)."
        )
    features, positions, cross_attn_features = backbone_result
    temporal_features = [
        feature.tensors.reshape(
            batch_size, num_frames, *feature.tensors.shape[1:]
        )
        for feature in features
    ]
    motion_output = motion_module.forward_temporal(mdd_frames, temporal_features)
    anchor = motion_module.anchor_index

    centre_features = []
    for modulated, original in zip(motion_output.features, features):
        centre_mask = None
        if original.mask is not None:
            centre_mask = _select_temporal_tensor(
                original.mask, batch_size, num_frames, anchor
            )
        centre_features.append(_rebuild_nested_tensor(modulated, centre_mask))
    centre_positions = [
        _select_temporal_tensor(position, batch_size, num_frames, anchor)
        for position in positions
    ]
    centre_cross_attn_features = None
    if cross_attn_features is not None:
        centre_cross_attn_features = []
        for feature in cross_attn_features:
            centre_mask = None
            if feature.mask is not None:
                centre_mask = _select_temporal_tensor(
                    feature.mask, batch_size, num_frames, anchor
                )
            centre_cross_attn_features.append(
                _rebuild_nested_tensor(
                    _select_temporal_tensor(
                        feature.tensors, batch_size, num_frames, anchor
                    ),
                    centre_mask,
                )
            )

    centre_samples = _rebuild_nested_tensor(
        frames[:, anchor], padding_masks[:, anchor]
    )
    original_backbone_forward = lwdetr.backbone.forward

    def _centre_backbone_forward(_samples):
        return centre_features, centre_positions, centre_cross_attn_features

    # Call the version-pinned stock continuation without mutating its class.
    stock_forward = getattr(type(lwdetr).forward, "_motion_original", type(lwdetr).forward)
    lwdetr.backbone.forward = _centre_backbone_forward
    try:
        output = stock_forward(lwdetr, centre_samples, targets)
    finally:
        lwdetr.backbone.forward = original_backbone_forward
    if not isinstance(output, dict):
        raise RuntimeError("LWDETR forward must return a prediction dictionary.")
    output["pred_heatmaps"] = motion_output.heatmaps
    output["pred_heatmap_logits"] = motion_output.heatmap_logits
    output["motion_maps"] = motion_output.motion_maps
    return output


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
    """Return False; TrackNet integration is instance-bound and never patched."""
    return False


def ensure_motion_support(motion_cfg: Optional[Dict[str, Any]] = None) -> None:
    """Validate an enabled TrackNet request without process-global mutation."""
    motion_cfg = motion_cfg or {}
    if not bool(motion_cfg.get("enabled", False)):
        return
    if resolve_motion_type(motion_cfg) == "none":
        return
    _validated_motion_config(motion_cfg)
    _check_version()


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


def _instance_motion_forward(
    lwdetr: nn.Module,
    samples: Any,
    targets: Optional[Any] = None,
):
    """Instance-bound dispatch: temporal batches use TrackNet, images stay stock."""
    is_temporal_tensor = torch.is_tensor(samples) and samples.ndim == 5
    is_temporal_batch = hasattr(samples, "frames")
    if is_temporal_tensor or is_temporal_batch:
        return run_temporal_lwdetr(lwdetr, samples, targets=targets)
    return type(lwdetr).forward(lwdetr, samples, targets)


def _instance_temporal_forward(
    lwdetr: nn.Module,
    frames: Any,
    padding_masks: Optional[torch.Tensor] = None,
    targets: Optional[Any] = None,
):
    return run_temporal_lwdetr(lwdetr, frames, padding_masks, targets)


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
    _validated_motion_config(motion_cfg)
    _check_version()

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
    lwdetr.forward = MethodType(_instance_motion_forward, lwdetr)  # type: ignore[method-assign]
    lwdetr.forward_temporal = MethodType(  # type: ignore[attr-defined]
        _instance_temporal_forward, lwdetr
    )
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
        if key.startswith("temporal_adapter.motion_module."):
            key = key[len("temporal_adapter."):]
        if key.startswith("motion_module."):
            motion_state[key] = value
    return motion_state


def load_motion_checkpoint_weights(
    model: nn.Module,
    checkpoint_path: Any,
) -> None:
    """Load only attached ``motion_module`` tensors from a training checkpoint."""
    lwdetr = _find_lwdetr(model)
    if lwdetr is None or not isinstance(getattr(lwdetr, "motion_module", None), MotionModule):
        raise RuntimeError(
            "Cannot load TrackNet weights before attach_motion_module() has succeeded."
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
    motion_state = _motion_state_from_checkpoint(checkpoint)
    if not motion_state:
        raise RuntimeError(f"Checkpoint {path} contains no motion_module.* weights.")
    _assert_new_motion_architecture_metadata(checkpoint, motion_state)
    stripped_state = {
        key[len("motion_module."):]: value for key, value in motion_state.items()
    }
    expected_state = lwdetr.motion_module.state_dict()
    missing = sorted(set(expected_state) - set(stripped_state))
    unexpected = sorted(set(stripped_state) - set(expected_state))
    mismatched = sorted(
        key
        for key in set(expected_state) & set(stripped_state)
        if tuple(expected_state[key].shape) != tuple(stripped_state[key].shape)
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if mismatched:
            details.append(f"shape_mismatch={mismatched[:5]}")
        raise RuntimeError(
            f"Checkpoint {path} is incompatible with the attached TrackNet module: "
            + "; ".join(details)
        )
    lwdetr.motion_module.load_state_dict(stripped_state, strict=True)


def _checkpoint_architecture_metadata(checkpoint: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    metadata = checkpoint.get('pitchobjectlab_architecture')
    if metadata is None and isinstance(checkpoint.get('args'), Mapping):
        metadata = checkpoint['args'].get('pitchobjectlab_architecture')
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise RuntimeError('Checkpoint pitchobjectlab_architecture metadata must be a mapping.')
    return metadata


def _assert_new_motion_architecture_metadata(
    checkpoint: Mapping[str, Any],
    motion_state: Mapping[str, torch.Tensor],
) -> None:
    """Reject legacy TrackNet tensors while leaving stock checkpoints untouched."""

    if not motion_state:
        return
    metadata = _checkpoint_architecture_metadata(checkpoint)
    if metadata is None:
        raise RuntimeError(
            "Checkpoint contains motion_module.* tensors but has no "
            "pitchobjectlab_architecture metadata. Refusing to load a legacy "
            "TrackNet prototype checkpoint."
        )
    schema_version = metadata.get("schema_version")
    if schema_version != _TRACKNET_ARCHITECTURE_SCHEMA_VERSION:
        raise RuntimeError(
            "TrackNet checkpoint architecture metadata must use schema_version="
            f"{_TRACKNET_ARCHITECTURE_SCHEMA_VERSION}, got {schema_version!r}."
        )
    saved_motion = metadata.get("motion")
    if not isinstance(saved_motion, Mapping):
        raise RuntimeError(
            "TrackNet checkpoint architecture metadata must contain a model.motion mapping."
        )
    if not bool(saved_motion.get("enabled", False)) or str(
        saved_motion.get("type", "")
    ).lower() != "tracknet_v5":
        raise RuntimeError(
            "TrackNet checkpoint architecture metadata must identify an enabled "
            "motion.type='tracknet_v5' graph."
        )


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

    checkpoint_state = _motion_state_from_checkpoint(checkpoint)
    _assert_new_motion_architecture_metadata(checkpoint, checkpoint_state)
    _assert_motion_metadata_compatible(checkpoint, expected_architecture)
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
    """Reject ONNX/TensorRT export for the true temporal TrackNet graph."""
    if motion_cfg is not None:
        requested = bool(motion_cfg.get("enabled", False)) and (
            resolve_motion_type(motion_cfg) != "none"
        )
    else:
        lwdetr = _find_lwdetr(model)
        requested = lwdetr is not None and getattr(lwdetr, "motion_module", None) is not None
    if not requested:
        return
    raise RuntimeError(
        "True temporal TrackNetV5 ONNX/TensorRT export is not supported. "
        "Use the PyTorch train/test/inference path or disable model.motion."
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
