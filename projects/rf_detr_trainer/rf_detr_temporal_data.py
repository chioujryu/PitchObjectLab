"""Temporal YOLO data support for the optional RF-DETR TrackNetV5 branch.

The stock RF-DETR data path remains untouched.  This module is used only when
the motion branch is enabled and a dataset provides
``metadata/temporal_index.jsonl``.  It builds complete, sequence-local temporal
windows and returns an RF-DETR-compatible ``(samples, targets)`` batch where
``samples.tensors`` and ``samples.mask`` expose the center/anchor frame.

YOLO boxes are read as normalized ``cx, cy, width, height`` values.  Every
spatial transform is replayed across all frames in a window, and target boxes
remain normalized ``cxcywh`` as expected by RF-DETR's criterion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def _canonical_split(value: str) -> str:
    split = str(value).strip().lower()
    try:
        return _SPLIT_ALIASES[split]
    except KeyError as exc:
        supported = ", ".join(sorted(_SPLIT_ALIASES))
        raise ValueError(f"Unsupported split {value!r}; expected one of: {supported}") from exc


def _normalise_size(size: int | Sequence[int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"image_size must be positive, got {size}")
        return size, size
    values = tuple(int(value) for value in size)
    if len(values) != 2 or min(values) <= 0:
        raise ValueError(f"image_size must be an int or (height, width), got {size!r}")
    return values


def _resolve_dataset_root(data_yaml: Path, descriptor: Mapping[str, Any]) -> Path:
    raw_root = descriptor.get("path", ".")
    if raw_root in (None, ""):
        raw_root = "."
    candidate = Path(str(raw_root))
    if candidate.is_absolute():
        return candidate.resolve()

    relative_to_yaml = (data_yaml.parent / candidate).resolve()
    if relative_to_yaml.exists():
        return relative_to_yaml

    # Ultralytics descriptors outside the dataset sometimes use paths relative
    # to the invocation directory.  Keep this compatibility fallback explicit.
    relative_to_cwd = (Path.cwd() / candidate).resolve()
    if relative_to_cwd.exists():
        return relative_to_cwd
    return relative_to_yaml


def _resolve_indexed_path(dataset_root: Path, value: str, field: str, line_number: int) -> Path:
    cleaned = str(value).strip().replace("\\", "/")
    if not cleaned:
        raise ValueError(f"temporal index line {line_number}: {field} must not be empty")
    path = Path(cleaned)
    resolved = path.resolve() if path.is_absolute() else (dataset_root / path).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"temporal index line {line_number}: {field} escapes dataset root: {value!r}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"temporal index line {line_number}: {field} does not exist: {resolved}")
    return resolved


def _class_names(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    names = descriptor.get("names", ())
    if isinstance(names, Mapping):
        try:
            ordered = sorted(((int(key), str(value)) for key, value in names.items()), key=lambda item: item[0])
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset.yaml names mapping keys must be integer-like") from exc
        return tuple(value for _, value in ordered)
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return tuple(str(value) for value in names)
    raise ValueError("dataset.yaml names must be a list or integer-keyed mapping")


@dataclass(frozen=True)
class TemporalFrameRecord:
    """One frame from ``metadata/temporal_index.jsonl``."""

    split: str
    sequence_id: str
    frame_index: int
    image_path: Path
    label_path: Path
    image_id: int
    primary_label_index: int | None
    box_count: int | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TemporalIndex:
    """Resolved dataset descriptor and temporal frame records."""

    data_yaml: Path
    dataset_root: Path
    temporal_index_path: Path
    class_names: tuple[str, ...]
    records: tuple[TemporalFrameRecord, ...]


@dataclass(frozen=True)
class TemporalWindow:
    """A complete frame window contained in one split and sequence."""

    records: tuple[TemporalFrameRecord, ...]
    anchor_index: int

    @property
    def anchor(self) -> TemporalFrameRecord:
        return self.records[self.anchor_index]

    @property
    def split(self) -> str:
        return self.anchor.split

    @property
    def sequence_id(self) -> str:
        return self.anchor.sequence_id


def load_temporal_index(
    data_yaml: str | Path,
    temporal_index_path: str | Path | None = None,
) -> TemporalIndex:
    """Load and validate a temporal dataset index.

    Paths in the JSONL index must remain under the dataset root. Frame identity is unique within ``(split,
    sequence_id)`` and is retained in ``metadata``.
    """
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset.yaml does not exist: {yaml_path}")
    descriptor = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"dataset descriptor must be a mapping: {yaml_path}")
    dataset_root = _resolve_dataset_root(yaml_path, descriptor)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    if temporal_index_path is None:
        index_path = dataset_root / "metadata" / "temporal_index.jsonl"
    else:
        candidate = Path(temporal_index_path).expanduser()
        index_path = candidate if candidate.is_absolute() else dataset_root / candidate
    index_path = index_path.resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"temporal index does not exist: {index_path}")

    records: list[TemporalFrameRecord] = []
    seen: set[tuple[str, str, int]] = set()
    for line_number, raw_line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"temporal index line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"temporal index line {line_number}: row must be an object")

        try:
            split = _canonical_split(str(row["split"]))
            sequence_id = str(row.get("sequence_id") or row.get("group") or "").strip()
            frame_index = int(row["frame_index"])
            image_path = _resolve_indexed_path(dataset_root, str(row["image"]), "image", line_number)
            label_path = _resolve_indexed_path(dataset_root, str(row["label"]), "label", line_number)
        except KeyError as exc:
            raise ValueError(f"temporal index line {line_number}: missing field {exc.args[0]!r}") from exc
        if not sequence_id:
            raise ValueError(f"temporal index line {line_number}: sequence_id/group must not be empty")
        if frame_index < 0:
            raise ValueError(f"temporal index line {line_number}: frame_index must be non-negative")

        primary_raw = row.get("primary_label_index")
        primary = None if primary_raw is None else int(primary_raw)
        if primary is not None and primary < 0:
            raise ValueError(f"temporal index line {line_number}: primary_label_index must be non-negative or null")
        box_count_raw = row.get("box_count")
        box_count = None if box_count_raw is None else int(box_count_raw)
        if box_count is not None and box_count < 0:
            raise ValueError(f"temporal index line {line_number}: box_count must be non-negative")

        identity = (split, sequence_id, frame_index)
        if identity in seen:
            raise ValueError(
                "temporal index contains duplicate frame identity "
                f"(split={split!r}, sequence_id={sequence_id!r}, frame_index={frame_index})"
            )
        seen.add(identity)
        records.append(
            TemporalFrameRecord(
                split=split,
                sequence_id=sequence_id,
                frame_index=frame_index,
                image_path=image_path,
                label_path=label_path,
                image_id=len(records),
                primary_label_index=primary,
                box_count=box_count,
                metadata=dict(row),
            )
        )
    if not records:
        raise ValueError(f"temporal index has no records: {index_path}")

    return TemporalIndex(
        data_yaml=yaml_path,
        dataset_root=dataset_root,
        temporal_index_path=index_path,
        class_names=_class_names(descriptor),
        records=tuple(records),
    )


def build_temporal_windows(
    records: Iterable[TemporalFrameRecord],
    *,
    num_frames: int = 3,
    frame_stride: int = 1,
    anchor: str = "center",
    boundary_policy: str = "drop",
) -> tuple[TemporalWindow, ...]:
    """Build complete sequence-local temporal windows.

    Only ``anchor='center'`` and ``boundary_policy='drop'`` are intentionally supported for training/evaluation. Missing
    frame indices produce no window; frames are never borrowed from a neighboring sequence or split.
    """
    if num_frames < 1 or num_frames % 2 == 0:
        raise ValueError(f"num_frames must be a positive odd integer, got {num_frames}")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    if str(anchor).lower() != "center":
        raise ValueError("Only anchor='center' is supported")
    if str(boundary_policy).lower() != "drop":
        raise ValueError("Only boundary_policy='drop' is supported for temporal datasets")

    grouped: dict[tuple[str, str], dict[int, TemporalFrameRecord]] = {}
    for record in records:
        group = grouped.setdefault((record.split, record.sequence_id), {})
        if record.frame_index in group:
            raise ValueError(f"duplicate frame_index {record.frame_index} in {record.split}/{record.sequence_id}")
        group[record.frame_index] = record

    radius = num_frames // 2
    offsets = tuple((position - radius) * frame_stride for position in range(num_frames))
    windows: list[TemporalWindow] = []
    for group_key in sorted(grouped):
        by_index = grouped[group_key]
        for anchor_index in sorted(by_index):
            wanted = tuple(anchor_index + offset for offset in offsets)
            if not all(frame_index in by_index for frame_index in wanted):
                continue
            window_records = tuple(by_index[frame_index] for frame_index in wanted)
            windows.append(TemporalWindow(records=window_records, anchor_index=radius))
    return tuple(windows)


def temporal_split_window_counts(
    dataset_root: str | Path,
    *,
    num_frames: int = 3,
    stride: int = 1,
) -> dict[str, int]:
    """Return complete-window counts per split without materializing images.

    ``dataset_root`` may be either the temporal dataset directory or its ``dataset.yaml`` path. The helper is
    metadata-only so train entrypoints can produce a reliable pre-run estimate cheaply.
    """
    supplied = Path(dataset_root).expanduser()
    data_yaml = supplied if supplied.suffix.lower() in {".yaml", ".yml"} else supplied / "dataset.yaml"
    index = load_temporal_index(data_yaml)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        records = tuple(record for record in index.records if record.split == split)
        counts[split] = (
            len(
                build_temporal_windows(
                    records,
                    num_frames=num_frames,
                    frame_stride=stride,
                    anchor="center",
                    boundary_policy="drop",
                )
            )
            if records
            else 0
        )
    return counts


def _clone_target(target: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.clone() for key, value in target.items()}


def resize_temporal_frames(
    images: Sequence[Image.Image],
    targets: Sequence[Mapping[str, Tensor]],
    size: int | Sequence[int],
) -> tuple[list[Image.Image], list[dict[str, Tensor]]]:
    """Resize every frame to one exact ``(height, width)`` using one operation."""
    target_size = _normalise_size(size)
    assert target_size is not None
    height, width = target_size
    if len(images) != len(targets):
        raise ValueError("images and targets must have the same length")
    resized_images = [image.resize((width, height), resample=Image.Resampling.BILINEAR) for image in images]
    resized_targets: list[dict[str, Tensor]] = []
    for target in targets:
        transformed = _clone_target(target)
        transformed["size"] = torch.tensor([height, width], dtype=torch.int64)
        boxes = transformed["boxes"]
        if boxes.numel():
            transformed["area"] = boxes[:, 2] * width * boxes[:, 3] * height
        else:
            transformed["area"] = torch.zeros((0,), dtype=torch.float32)
        resized_targets.append(transformed)
    return resized_images, resized_targets


def horizontal_flip_temporal_frames(
    images: Sequence[Image.Image],
    targets: Sequence[Mapping[str, Tensor]],
) -> tuple[list[Image.Image], list[dict[str, Tensor]]]:
    """Horizontally flip all frames and normalized ``cxcywh`` targets together."""
    if len(images) != len(targets):
        raise ValueError("images and targets must have the same length")
    flipped_images = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images]
    flipped_targets: list[dict[str, Tensor]] = []
    for target in targets:
        transformed = _clone_target(target)
        boxes = transformed["boxes"]
        if boxes.numel():
            boxes[:, 0] = 1.0 - boxes[:, 0]
        flipped_targets.append(transformed)
    return flipped_images, flipped_targets


@dataclass(frozen=True)
class TemporalTransformConfig:
    """Deterministic transform replayed across a full temporal window."""

    image_size: int | tuple[int, int] | None = (512, 512)
    horizontal_flip: bool = False
    normalize: bool = True
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        _normalise_size(self.image_size)
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must each contain three channel values")
        if any(value <= 0 for value in self.std):
            raise ValueError("std values must be positive")


class SynchronizedTemporalTransform:
    """Apply exact resize/flip/normalization decisions to every frame."""

    def __init__(self, config: TemporalTransformConfig | None = None) -> None:
        self.config = config or TemporalTransformConfig()

    def __call__(
        self,
        images: Sequence[Image.Image],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> tuple[Tensor, tuple[dict[str, Tensor], ...]]:
        if not images:
            raise ValueError("a temporal window must contain at least one image")
        if len(images) != len(targets):
            raise ValueError("images and targets must have the same length")

        transformed_images = [image.copy() for image in images]
        transformed_targets = [_clone_target(target) for target in targets]
        size = _normalise_size(self.config.image_size)
        if size is not None:
            transformed_images, transformed_targets = resize_temporal_frames(
                transformed_images, transformed_targets, size
            )
        if self.config.horizontal_flip:
            transformed_images, transformed_targets = horizontal_flip_temporal_frames(
                transformed_images, transformed_targets
            )

        tensors: list[Tensor] = []
        mean = torch.tensor(self.config.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.config.std, dtype=torch.float32).view(3, 1, 1)
        for image in transformed_images:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
            if self.config.normalize:
                tensor = (tensor - mean) / std
            tensors.append(tensor)

        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError(
                "all frames in a temporal window must share one shape; configure image_size for mixed inputs"
            )
        return torch.stack(tensors), tuple(transformed_targets)


def _parse_yolo_label(path: Path) -> tuple[Tensor, Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO detection fields, got {len(fields)}")
        try:
            class_value, cx, cy, width, height = (float(value) for value in fields)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: YOLO fields must be numeric") from exc
        values = (class_value, cx, cy, width, height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: YOLO fields must be finite")
        class_id = int(class_value)
        if class_value != class_id or class_id < 0:
            raise ValueError(f"{path}:{line_number}: class id must be a non-negative integer")
        if not 0.0 <= cx <= 1.0 or not 0.0 <= cy <= 1.0:
            raise ValueError(f"{path}:{line_number}: box center must be within [0, 1]")
        if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
            raise ValueError(f"{path}:{line_number}: box width/height must be within (0, 1]")
        boxes.append([cx, cy, width, height])
        labels.append(class_id)
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.int64),
    )


def _focus_indices(
    box_count: int,
    *,
    focus_mode: str,
    primary_label_index: int | None,
    record: TemporalFrameRecord,
) -> Tensor:
    mode = str(focus_mode).strip().lower()
    if mode not in {"single", "all"}:
        raise ValueError(f"focus_mode must be 'single' or 'all', got {focus_mode!r}")
    if mode == "all":
        return torch.arange(box_count, dtype=torch.int64)
    if box_count == 0:
        return torch.zeros((0,), dtype=torch.int64)
    if box_count == 1:
        if primary_label_index not in (None, 0):
            raise ValueError(f"{record.label_path}: primary_label_index {primary_label_index} is invalid for one box")
        return torch.tensor([0], dtype=torch.int64)
    if primary_label_index is None:
        raise ValueError(
            "single focus requires primary_label_index for multi-ball frame "
            f"{record.split}/{record.sequence_id}/{record.frame_index}"
        )
    if primary_label_index >= box_count:
        raise ValueError(
            f"{record.label_path}: primary_label_index {primary_label_index} is outside [0, {box_count - 1}]"
        )
    return torch.tensor([primary_label_index], dtype=torch.int64)


def generate_gaussian_heatmap(
    boxes: Tensor,
    size: int | Sequence[int],
    *,
    box_indices: Tensor | Sequence[int] | None = None,
    min_sigma: float = 1.0,
    truncate: float = 3.0,
) -> Tensor:
    """Render max-composited bbox-centred Gaussian targets.

    ``boxes`` must be normalized ``cxcywh``. Standard deviation is bbox width/height divided by six, with ``min_sigma``
    measured in output pixels.
    """
    output_size = _normalise_size(size)
    if output_size is None:
        raise ValueError("heatmap size must not be None")
    height, width = output_size
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {tuple(boxes.shape)}")
    if min_sigma <= 0 or truncate <= 0:
        raise ValueError("min_sigma and truncate must be positive")
    heatmap = torch.zeros((height, width), dtype=torch.float32, device=boxes.device)
    if box_indices is None:
        selected = torch.arange(boxes.shape[0], device=boxes.device)
    else:
        selected = torch.as_tensor(box_indices, dtype=torch.int64, device=boxes.device).flatten()
    if selected.numel() == 0:
        return heatmap
    if int(selected.min()) < 0 or int(selected.max()) >= boxes.shape[0]:
        raise IndexError("box_indices contains an out-of-range index")

    for box in boxes[selected]:
        center_x = float(box[0]) * max(width - 1, 1)
        center_y = float(box[1]) * max(height - 1, 1)
        sigma_x = max(float(box[2]) * width / 6.0, float(min_sigma))
        sigma_y = max(float(box[3]) * height / 6.0, float(min_sigma))
        x0 = max(0, math.floor(center_x - truncate * sigma_x))
        x1 = min(width, math.ceil(center_x + truncate * sigma_x) + 1)
        y0 = max(0, math.floor(center_y - truncate * sigma_y))
        y1 = min(height, math.ceil(center_y + truncate * sigma_y) + 1)
        xs = torch.arange(x0, x1, dtype=torch.float32, device=boxes.device)
        ys = torch.arange(y0, y1, dtype=torch.float32, device=boxes.device)
        gaussian = torch.exp(
            -0.5
            * (((xs.unsqueeze(0) - center_x) / sigma_x).square() + ((ys.unsqueeze(1) - center_y) / sigma_y).square())
        )
        heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], gaussian)
    return heatmap


@dataclass(frozen=True)
class TemporalSample:
    """One transformed temporal window before batching."""

    frames: Tensor
    padding_masks: Tensor
    frame_targets: tuple[dict[str, Tensor], ...]
    metadata: Mapping[str, Any]
    anchor_index: int
    normalization_mean: tuple[float, float, float] | None = None
    normalization_std: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class TemporalBatch:
    """Padded temporal frames with NestedTensor-compatible anchor properties."""

    frames: Tensor
    padding_masks: Tensor
    frame_targets: tuple[tuple[dict[str, Tensor], ...], ...]
    metadata: tuple[Mapping[str, Any], ...]
    anchor_index: int
    normalization_mean: tuple[float, float, float] | None = None
    normalization_std: tuple[float, float, float] | None = None

    @property
    def mdd_frames(self) -> Tensor:
        """Return RGB frames in ``[0, 1]`` for luminance motion differences.

        RF-DETR's shared backbone consumes ``frames`` exactly as transformed
        (normally ImageNet-normalized).  MDD instead needs physical RGB
        differences, so normalized batches are inverted here and clamped.
        Manually constructed/raw batches leave the normalization fields unset
        and therefore keep their existing raw-tensor semantics.
        """
        frames = self.frames
        if self.normalization_mean is not None:
            if self.normalization_std is None:
                raise RuntimeError("TemporalBatch normalization_std is required when normalization_mean is set")
            mean = frames.new_tensor(self.normalization_mean).view(1, 1, 3, 1, 1)
            std = frames.new_tensor(self.normalization_std).view(1, 1, 3, 1, 1)
            frames = frames * std + mean
        elif self.normalization_std is not None:
            raise RuntimeError("TemporalBatch normalization_mean is required when normalization_std is set")
        frames = frames.clamp(0.0, 1.0)
        return frames.masked_fill(self.padding_masks.unsqueeze(2), 0.0)

    @property
    def tensors(self) -> Tensor:
        """Anchor images, compatible with ``NestedTensor.tensors`` consumers."""
        return self.frames[:, self.anchor_index]

    @property
    def mask(self) -> Tensor:
        """Anchor padding mask, compatible with ``NestedTensor.mask`` consumers."""
        return self.padding_masks[:, self.anchor_index]

    @property
    def flattened_tensors(self) -> Tensor:
        """All temporal images as ``[B*T, C, H, W]`` for the shared backbone."""
        batch, frames, channels, height, width = self.frames.shape
        return self.frames.reshape(batch * frames, channels, height, width)

    @property
    def flattened_mask(self) -> Tensor:
        """All temporal padding masks as ``[B*T, H, W]``."""
        batch, frames, height, width = self.padding_masks.shape
        return self.padding_masks.reshape(batch * frames, height, width)

    def decompose(self) -> tuple[Tensor, Tensor]:
        """Return anchor tensors/mask like RF-DETR's ``NestedTensor``."""
        return self.tensors, self.mask

    def to(self, device: torch.device | str, **kwargs: Any) -> TemporalBatch:
        """Move temporal tensors and tensor-valued frame targets to a device."""
        frames = self.frames.to(device, **kwargs)
        non_blocking = bool(kwargs.get("non_blocking", False))
        masks = self.padding_masks.to(device=device, non_blocking=non_blocking)
        moved_targets = tuple(
            tuple({key: value.to(device, **kwargs) for key, value in target.items()} for target in sample_targets)
            for sample_targets in self.frame_targets
        )
        return replace(self, frames=frames, padding_masks=masks, frame_targets=moved_targets)

    def pin_memory(self) -> TemporalBatch:
        """Pin batch tensors for asynchronous host-to-device transfer."""
        pinned_targets = tuple(
            tuple({key: value.pin_memory() for key, value in target.items()} for target in sample_targets)
            for sample_targets in self.frame_targets
        )
        return replace(
            self,
            frames=self.frames.pin_memory(),
            padding_masks=self.padding_masks.pin_memory(),
            frame_targets=pinned_targets,
        )


class TemporalDataset(Dataset[TemporalSample]):
    """YOLO temporal dataset backed by the canonical JSONL frame index."""

    def __init__(
        self,
        data_yaml: str | Path,
        *,
        split: str = "train",
        temporal_index_path: str | Path | None = None,
        num_frames: int = 3,
        frame_stride: int = 1,
        anchor: str = "center",
        boundary_policy: str = "drop",
        focus_mode: str = "all",
        primary_field: str = "primary_label_index",
        image_size: int | Sequence[int] | None = (512, 512),
        horizontal_flip: bool = False,
        normalize: bool = True,
        min_sigma: float = 1.0,
        max_windows: int | None = None,
        transform: SynchronizedTemporalTransform | None = None,
    ) -> None:
        super().__init__()
        self.index = load_temporal_index(data_yaml, temporal_index_path)
        self.split = _canonical_split(split)
        self.focus_mode = str(focus_mode).strip().lower()
        if self.focus_mode not in {"single", "all"}:
            raise ValueError(f"focus_mode must be 'single' or 'all', got {focus_mode!r}")
        self.primary_field = str(primary_field)
        if not self.primary_field:
            raise ValueError("primary_field must not be empty")
        self.min_sigma = float(min_sigma)
        if self.min_sigma <= 0:
            raise ValueError("min_sigma must be positive")

        records = tuple(record for record in self.index.records if record.split == self.split)
        if not records:
            raise ValueError(f"temporal index has no records for split {self.split!r}")
        self.windows = build_temporal_windows(
            records,
            num_frames=num_frames,
            frame_stride=frame_stride,
            anchor=anchor,
            boundary_policy=boundary_policy,
        )
        if max_windows is not None:
            max_windows = int(max_windows)
            if max_windows < 1:
                raise ValueError("max_windows must be positive when provided")
            self.windows = self.windows[:max_windows]
        if not self.windows:
            raise ValueError(
                f"split {self.split!r} has no complete temporal windows "
                f"(num_frames={num_frames}, frame_stride={frame_stride})"
            )
        self.num_frames = num_frames
        self.anchor_index = num_frames // 2
        self.transform = transform or SynchronizedTemporalTransform(
            TemporalTransformConfig(
                image_size=_normalise_size(image_size),
                horizontal_flip=bool(horizontal_flip),
                normalize=bool(normalize),
            )
        )

    def __len__(self) -> int:
        return len(self.windows)

    def _target_for_record(
        self,
        record: TemporalFrameRecord,
        *,
        image_height: int,
        image_width: int,
    ) -> dict[str, Tensor]:
        boxes, labels = _parse_yolo_label(record.label_path)
        if record.box_count is not None and record.box_count != boxes.shape[0]:
            raise ValueError(
                f"{record.label_path}: temporal index box_count={record.box_count}, "
                f"but label contains {boxes.shape[0]} boxes"
            )
        primary_raw = record.metadata.get(self.primary_field, record.primary_label_index)
        primary = None if primary_raw is None else int(primary_raw)
        focus_indices = _focus_indices(
            boxes.shape[0],
            focus_mode=self.focus_mode,
            primary_label_index=primary,
            record=record,
        )
        area = boxes[:, 2] * image_width * boxes[:, 3] * image_height
        primary_tensor = torch.tensor(-1 if primary is None else primary, dtype=torch.int64)
        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([record.image_id], dtype=torch.int64),
            "area": area.to(torch.float32),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "orig_size": torch.tensor([image_height, image_width], dtype=torch.int64),
            "size": torch.tensor([image_height, image_width], dtype=torch.int64),
            "tracknet_box_indices": focus_indices,
            "primary_label_index": primary_tensor,
        }

    def __getitem__(self, index: int) -> TemporalSample:
        window = self.windows[index]
        images: list[Image.Image] = []
        targets: list[dict[str, Tensor]] = []
        try:
            for record in window.records:
                with Image.open(record.image_path) as source:
                    image = source.convert("RGB")
                width, height = image.size
                images.append(image)
                targets.append(self._target_for_record(record, image_height=height, image_width=width))
            frames, transformed_targets = self.transform(images, targets)
        finally:
            for image in images:
                image.close()

        height, width = frames.shape[-2:]
        targets_with_heatmaps: list[dict[str, Tensor]] = []
        for target in transformed_targets:
            enriched = dict(target)
            enriched["tracknet_heatmap"] = generate_gaussian_heatmap(
                enriched["boxes"],
                (height, width),
                box_indices=enriched["tracknet_box_indices"],
                min_sigma=self.min_sigma,
            )
            targets_with_heatmaps.append(enriched)

        metadata = {
            "split": window.split,
            "sequence_id": window.sequence_id,
            "anchor_index": window.anchor_index,
            "anchor_frame_index": window.anchor.frame_index,
            "frame_indices": tuple(record.frame_index for record in window.records),
            "image_paths": tuple(str(record.image_path) for record in window.records),
            "label_paths": tuple(str(record.label_path) for record in window.records),
            "records": tuple(dict(record.metadata) for record in window.records),
            "focus_mode": self.focus_mode,
            "boundary_padding": False,
        }
        return TemporalSample(
            frames=frames,
            padding_masks=torch.zeros((self.num_frames, height, width), dtype=torch.bool),
            frame_targets=tuple(targets_with_heatmaps),
            metadata=metadata,
            anchor_index=window.anchor_index,
            normalization_mean=(
                tuple(float(value) for value in self.transform.config.mean) if self.transform.config.normalize else None
            ),
            normalization_std=(
                tuple(float(value) for value in self.transform.config.std) if self.transform.config.normalize else None
            ),
        )


def _round_up(value: int, block_size: int | None) -> int:
    if block_size is None:
        return value
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return ((value + block_size - 1) // block_size) * block_size


def temporal_collate_fn(
    samples: Sequence[TemporalSample],
    *,
    block_size: int | None = None,
) -> tuple[TemporalBatch, list[dict[str, Tensor]]]:
    """Pad temporal samples and return RF-DETR's ``(samples, targets)`` shape."""
    if not samples:
        raise ValueError("cannot collate an empty temporal batch")
    frame_counts = {sample.frames.shape[0] for sample in samples}
    anchor_indices = {sample.anchor_index for sample in samples}
    if len(frame_counts) != 1 or len(anchor_indices) != 1:
        raise ValueError("all samples in a temporal batch must share num_frames and anchor_index")
    if any(sample.frames.ndim != 4 for sample in samples):
        raise ValueError("each TemporalSample.frames must have shape [T, C, H, W]")
    channels = {sample.frames.shape[1] for sample in samples}
    if channels != {3}:
        raise ValueError(f"temporal RGB batches require three channels, got {sorted(channels)}")
    normalizations = {(sample.normalization_mean, sample.normalization_std) for sample in samples}
    if len(normalizations) != 1:
        raise ValueError("all samples in a temporal batch must share one normalization")
    normalization_mean, normalization_std = next(iter(normalizations))

    num_frames = next(iter(frame_counts))
    anchor_index = next(iter(anchor_indices))
    max_height = _round_up(max(sample.frames.shape[-2] for sample in samples), block_size)
    max_width = _round_up(max(sample.frames.shape[-1] for sample in samples), block_size)
    frames = samples[0].frames.new_zeros((len(samples), num_frames, 3, max_height, max_width))
    masks = torch.ones(
        (len(samples), num_frames, max_height, max_width),
        dtype=torch.bool,
        device=samples[0].frames.device,
    )

    batched_targets: list[tuple[dict[str, Tensor], ...]] = []
    anchor_targets: list[dict[str, Tensor]] = []
    for batch_index, sample in enumerate(samples):
        height, width = sample.frames.shape[-2:]
        frames[batch_index, :, :, :height, :width].copy_(sample.frames)
        masks[batch_index, :, :height, :width].copy_(sample.padding_masks)

        padded_frame_targets: list[dict[str, Tensor]] = []
        for target in sample.frame_targets:
            padded_target = dict(target)
            heatmap = target["tracknet_heatmap"]
            if heatmap.shape != (height, width):
                raise ValueError(
                    "tracknet_heatmap shape must match transformed frame size, "
                    f"got {tuple(heatmap.shape)} vs {(height, width)}"
                )
            padded_target["tracknet_heatmap"] = F.pad(heatmap, (0, max_width - width, 0, max_height - height))
            padded_frame_targets.append(padded_target)
        batched_targets.append(tuple(padded_frame_targets))

        anchor_target = dict(padded_frame_targets[anchor_index])
        anchor_target["temporal_heatmaps"] = torch.stack(
            [target["tracknet_heatmap"] for target in padded_frame_targets]
        )
        anchor_target["temporal_primary_label_indices"] = torch.stack(
            [target["primary_label_index"] for target in padded_frame_targets]
        )
        anchor_target["temporal_image_ids"] = torch.cat([target["image_id"] for target in padded_frame_targets])
        anchor_targets.append(anchor_target)

    temporal_batch = TemporalBatch(
        frames=frames,
        padding_masks=masks,
        frame_targets=tuple(batched_targets),
        metadata=tuple(sample.metadata for sample in samples),
        anchor_index=anchor_index,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )
    return temporal_batch, anchor_targets


class TemporalRFDETRDataModule(LightningDataModule):
    """Small Lightning-compatible facade around temporal PyTorch DataLoaders.

    It deliberately has no dependency on Lightning itself. RF-DETR entrypoints may pass the returned loaders to their
    training module or wrap this object in the installed Lightning version.
    """

    def __init__(
        self,
        data_yaml: str | Path,
        *,
        image_size: int | Sequence[int] = (512, 512),
        batch_size: int = 1,
        num_workers: int = 2,
        num_frames: int = 3,
        frame_stride: int = 1,
        focus_mode: str = "all",
        primary_field: str = "primary_label_index",
        min_sigma: float = 1.0,
        block_size: int | None = None,
        train_horizontal_flip: bool = False,
        max_windows_per_split: Mapping[str, int] | None = None,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.data_yaml = Path(data_yaml)
        self.image_size = _normalise_size(image_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.focus_mode = focus_mode
        self.primary_field = primary_field
        self.min_sigma = float(min_sigma)
        self.block_size = block_size
        self.train_horizontal_flip = bool(train_horizontal_flip)
        self.max_windows_per_split = {
            _canonical_split(split): int(limit) for split, limit in dict(max_windows_per_split or {}).items()
        }
        if any(limit < 1 for limit in self.max_windows_per_split.values()):
            raise ValueError("max_windows_per_split values must be positive")
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = self.num_workers > 0 if persistent_workers is None else bool(persistent_workers)
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")
        self.datasets: dict[str, TemporalDataset] = {}
        index = load_temporal_index(self.data_yaml)
        self.class_names = list(index.class_names)

    def prepare_data(self) -> None:
        """Validate that the descriptor and temporal index can be read."""
        load_temporal_index(self.data_yaml)

    def setup(self, stage: str | None = None) -> None:
        """Construct only datasets relevant to the requested stage."""
        requested: tuple[str, ...]
        if stage in (None, "fit"):
            requested = ("train", "val")
        elif stage in ("validate",):
            requested = ("val",)
        elif stage in ("test", "predict"):
            requested = ("test",)
        else:
            raise ValueError(f"unsupported data module stage: {stage!r}")
        for split in requested:
            if split not in self.datasets:
                self.datasets[split] = TemporalDataset(
                    self.data_yaml,
                    split=split,
                    num_frames=self.num_frames,
                    frame_stride=self.frame_stride,
                    focus_mode=self.focus_mode,
                    primary_field=self.primary_field,
                    image_size=self.image_size,
                    horizontal_flip=self.train_horizontal_flip and split == "train",
                    min_sigma=self.min_sigma,
                    max_windows=self.max_windows_per_split.get(split),
                )

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        if split not in self.datasets:
            self.setup("fit" if split in {"train", "val"} else "test")
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=partial(temporal_collate_fn, block_size=self.block_size),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def transfer_batch_to_device(
        self,
        batch: tuple[TemporalBatch, list[dict[str, Tensor]]],
        device: torch.device,
        dataloader_idx: int,
    ) -> tuple[TemporalBatch, list[dict[str, Tensor]]]:
        """Move both temporal frames and all detection/heatmap targets together."""
        del dataloader_idx
        samples, targets = batch
        moved_targets = [
            {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        return samples.to(device, non_blocking=True), moved_targets


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SynchronizedTemporalTransform",
    "TemporalBatch",
    "TemporalDataset",
    "TemporalFrameRecord",
    "TemporalIndex",
    "TemporalRFDETRDataModule",
    "TemporalSample",
    "TemporalTransformConfig",
    "TemporalWindow",
    "build_temporal_windows",
    "generate_gaussian_heatmap",
    "horizontal_flip_temporal_frames",
    "load_temporal_index",
    "resize_temporal_frames",
    "temporal_collate_fn",
    "temporal_split_window_counts",
]
