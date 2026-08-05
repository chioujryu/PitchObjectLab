"""Shared object-detection test mode helpers.

The functions in this module intentionally avoid importing model frameworks.
They operate on COCO-style records and PIL images so YOLO, RF-DETR, and the
standalone evaluator can share mode normalization, crop geometry, NMS, and
model-input visual artifacts without coupling their inference stacks together.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

SUPPORTED_TEST_MODES = {"full_image", "sahi", "class_crop"}
FULL_IMAGE_MODE = "full_image"
SAHI_MODE = "sahi"
CLASS_CROP_MODE = "class_crop"


def get_value(record: Any, key: str, default: Any = None) -> Any:
    """Read a field from a mapping or an object."""
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def canonical_test_mode(config: Mapping[str, Any]) -> str:
    """Return full_image, sahi, or class_crop from new or legacy config fields."""
    test_mode = config.get("test_mode", {})
    inference = config.get("inference", {})
    raw = None
    if isinstance(test_mode, Mapping):
        raw = test_mode.get("mode")
    if raw is None and isinstance(inference, Mapping):
        raw = inference.get("mode")
    if raw is None and isinstance(inference, Mapping) and "use_sahi" in inference:
        raw = SAHI_MODE if bool(inference.get("use_sahi")) else FULL_IMAGE_MODE
    if raw is None:
        raw = FULL_IMAGE_MODE
    mode = str(raw).strip().lower().replace("-", "_")
    if mode in {"full", "direct", "original", "whole_image"}:
        mode = FULL_IMAGE_MODE
    if mode in {"crop", "class_cropped", "prediction_crop"}:
        mode = CLASS_CROP_MODE
    if mode not in SUPPORTED_TEST_MODES:
        raise ValueError(f"Unsupported test mode {raw!r}. Options: {', '.join(sorted(SUPPORTED_TEST_MODES))}.")
    return mode


def is_sahi_mode(config: Mapping[str, Any]) -> bool:
    """Return True when configured test mode is SAHI sliced inference."""
    return canonical_test_mode(config) == SAHI_MODE


def default_crop_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return crop config with defaults applied."""
    crop = dict(config.get("crop", {}) or {})
    crop.setdefault("class_ids", [])
    crop.setdefault("class_names", [])
    crop.setdefault("source", "prediction")
    crop.setdefault("source_conf", config.get("model", {}).get("confidence_threshold", 0.25))
    crop.setdefault("padding_pixels", 0)
    crop.setdefault("padding_ratio", 0.05)
    crop.setdefault("fallback", FULL_IMAGE_MODE)
    return crop


def coco_xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert COCO xywh to xyxy."""
    x, y, w, h = [float(value) for value in bbox[:4]]
    return x, y, x + w, y + h


def xyxy_to_coco_bbox(
    box: Sequence[float],
    width: int,
    height: int,
) -> Optional[Tuple[List[float], float]]:
    """Clip xyxy to an image and return COCO xywh plus area."""
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    bbox = [x1, y1, x2 - x1, y2 - y1]
    return bbox, bbox[2] * bbox[3]


def bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def bbox_ios_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute intersection over smaller-box area for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = min(area_a, area_b)
    return inter / denom if denom > 0.0 else 0.0


def bbox_match_score_xyxy(a: Sequence[float], b: Sequence[float], metric: str) -> float:
    """Return a normalized box-overlap score for IOU or IOS matching."""
    normalized = str(metric or "IOU").strip().upper()
    if normalized == "IOS":
        return bbox_ios_xyxy(a, b)
    if normalized != "IOU":
        raise ValueError("SAHI postprocess_match_metric must be IOU or IOS.")
    return bbox_iou_xyxy(a, b)


def merge_coco_prediction_cluster(cluster: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge one SAHI duplicate cluster using the union box and best score/class."""
    if not cluster:
        raise ValueError("Cannot merge an empty prediction cluster.")
    ordered = sorted(cluster, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    best = dict(ordered[0])
    boxes = [coco_xywh_to_xyxy(item.get("bbox", [0, 0, 0, 0])) for item in ordered]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    best["bbox"] = [float(x1), float(y1), float(width), float(height)]
    best["area"] = float(width * height)
    best["score"] = float(best.get("score", 0.0))
    best["merged_prediction_count"] = len(ordered)
    return best


def coco_box_matches_any_cluster_box(
    candidate_box: Sequence[float],
    cluster_boxes: Sequence[Sequence[float]],
    match_metric: str,
    match_threshold: float,
) -> bool:
    """Return True when a candidate overlaps any box already assigned to a cluster."""
    return any(bbox_match_score_xyxy(candidate_box, cluster_box, match_metric) >= match_threshold for cluster_box in cluster_boxes)


def nms_coco_predictions(
    predictions: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> List[Dict[str, Any]]:
    """Apply simple score-descending NMS to COCO prediction rows."""
    threshold = max(0.0, min(1.0, float(iou_threshold)))
    groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    for prediction in predictions:
        image_id = int(prediction.get("image_id", 0))
        class_key = -1 if class_agnostic else int(prediction.get("category_id", 0))
        groups.setdefault((image_id, class_key), []).append(prediction)

    kept: List[Dict[str, Any]] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        selected: List[Mapping[str, Any]] = []
        for prediction in ordered:
            box = coco_xywh_to_xyxy(prediction.get("bbox", [0, 0, 0, 0]))
            if all(bbox_iou_xyxy(box, coco_xywh_to_xyxy(other.get("bbox", [0, 0, 0, 0]))) <= threshold for other in selected):
                selected.append(prediction)
        kept.extend(dict(item) for item in selected)
    kept.sort(key=lambda item: (int(item.get("image_id", 0)), int(item.get("category_id", 0)), -float(item.get("score", 0.0))))
    return kept


def postprocess_sahi_coco_predictions(
    predictions: Sequence[Mapping[str, Any]],
    postprocess_type: str = "GREEDYNMM",
    match_metric: str = "IOS",
    match_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> List[Dict[str, Any]]:
    """Postprocess SAHI-style slice predictions with NMS or greedy non-max merging."""
    normalized_type = str(postprocess_type or "GREEDYNMM").strip().upper()
    if normalized_type in {"NMS", "LSNMS"}:
        if str(match_metric or "IOU").strip().upper() == "IOU":
            return nms_coco_predictions(predictions, match_threshold, class_agnostic)
        merge_matching = False
    elif normalized_type in {"NMM", "GREEDYNMM"}:
        merge_matching = True
    else:
        raise ValueError("SAHI postprocess_type must be GREEDYNMM, NMM, NMS, or LSNMS.")

    threshold = max(0.0, min(1.0, float(match_threshold)))
    groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    for prediction in predictions:
        image_id = int(prediction.get("image_id", 0))
        class_key = -1 if class_agnostic else int(prediction.get("category_id", 0))
        groups.setdefault((image_id, class_key), []).append(prediction)

    kept: List[Dict[str, Any]] = []
    for group in groups.values():
        pending = sorted(group, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        while pending:
            seed = pending.pop(0)
            matched: List[Mapping[str, Any]] = [seed]
            cluster_boxes: List[Tuple[float, float, float, float]] = [coco_xywh_to_xyxy(seed.get("bbox", [0, 0, 0, 0]))]
            expanded = True
            while expanded:
                expanded = False
                remaining: List[Mapping[str, Any]] = []
                for candidate in pending:
                    candidate_box = coco_xywh_to_xyxy(candidate.get("bbox", [0, 0, 0, 0]))
                    if coco_box_matches_any_cluster_box(candidate_box, cluster_boxes, match_metric, threshold):
                        cluster_boxes.append(candidate_box)
                        expanded = True
                        if merge_matching:
                            matched.append(candidate)
                    else:
                        remaining.append(candidate)
                pending = remaining
            if merge_matching:
                kept.append(merge_coco_prediction_cluster(matched))
            else:
                kept.append(dict(seed))
    kept.sort(key=lambda item: (int(item.get("image_id", 0)), int(item.get("category_id", 0)), -float(item.get("score", 0.0))))
    return kept


def category_maps(categories: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Build category id/name maps."""
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    for category in categories:
        category_id = int(category["id"])
        name = str(category.get("name", category_id))
        id_to_name[category_id] = name
        name_to_id[name.casefold()] = category_id
    return id_to_name, name_to_id


def config_list(value: Any) -> List[Any]:
    """Normalize scalar/list config values to a list."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None and item != ""]
    return [value]


def resolve_crop_class_ids(categories: Sequence[Mapping[str, Any]], crop_cfg: Mapping[str, Any]) -> List[int]:
    """Resolve crop class ids. Empty means all predicted classes are eligible."""
    id_to_name, name_to_id = category_maps(categories)
    selected: List[int] = []
    seen = set()
    for value in config_list(crop_cfg.get("class_ids")):
        category_id = int(value)
        if category_id not in id_to_name:
            raise ValueError(f"Unknown crop category id: {category_id}")
        if category_id not in seen:
            selected.append(category_id)
            seen.add(category_id)
    for value in config_list(crop_cfg.get("class_names")):
        name = str(value).strip()
        category_id = name_to_id.get(name.casefold())
        if category_id is None:
            available = ", ".join(id_to_name.values())
            raise ValueError(f"Unknown crop category name: {name!r}. Available names: {available}")
        if category_id not in seen:
            selected.append(category_id)
            seen.add(category_id)
    return selected


def select_crop_window_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
    crop_cfg: Mapping[str, Any],
    categories: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int, int]]:
    """Return padded crop window from prediction boxes, or None when no class matches."""
    selected_ids = set(resolve_crop_class_ids(categories, crop_cfg))
    source_conf = crop_cfg.get("source_conf")
    source_conf = float(source_conf) if source_conf is not None else 0.0
    eligible = []
    for prediction in predictions:
        category_id = int(prediction.get("category_id", 0))
        if selected_ids and category_id not in selected_ids:
            continue
        if float(prediction.get("score", 0.0)) < source_conf:
            continue
        eligible.append(prediction)
    if not eligible:
        return None

    boxes = [coco_xywh_to_xyxy(item.get("bbox", [0, 0, 0, 0])) for item in eligible]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    crop_w = max(1.0, x2 - x1)
    crop_h = max(1.0, y2 - y1)
    pad_pixels = float(crop_cfg.get("padding_pixels", 0) or 0)
    pad_ratio = float(crop_cfg.get("padding_ratio", 0.0) or 0.0)
    pad_x = pad_pixels + crop_w * pad_ratio
    pad_y = pad_pixels + crop_h * pad_ratio
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(width), x2 + pad_x)
    y2 = min(float(height), y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    ix1 = int(math.floor(x1))
    iy1 = int(math.floor(y1))
    ix2 = int(math.ceil(x2))
    iy2 = int(math.ceil(y2))
    ix2 = max(ix1 + 1, min(int(width), ix2))
    iy2 = max(iy1 + 1, min(int(height), iy2))
    return ix1, iy1, ix2 - ix1, iy2 - iy1, len(eligible)


def project_predictions_to_original(
    predictions: Sequence[Mapping[str, Any]],
    offset_x: int,
    offset_y: int,
    original_width: int,
    original_height: int,
) -> List[Dict[str, Any]]:
    """Project crop-local COCO predictions back to original image coordinates."""
    projected: List[Dict[str, Any]] = []
    for prediction in predictions:
        x, y, w, h = [float(value) for value in prediction.get("bbox", [0, 0, 0, 0])[:4]]
        converted = xyxy_to_coco_bbox(
            [x + offset_x, y + offset_y, x + offset_x + w, y + offset_y + h],
            original_width,
            original_height,
        )
        if converted is None:
            continue
        bbox, area = converted
        row = dict(prediction)
        row["bbox"] = bbox
        row["area"] = float(area)
        row["crop_offset_x"] = int(offset_x)
        row["crop_offset_y"] = int(offset_y)
        projected.append(row)
    return projected


def slice_axis_starts(length: int, window: int, overlap_ratio: float) -> List[int]:
    """Return deterministic starts that cover one image axis."""
    length = max(1, int(length))
    window = max(1, min(int(window), length))
    if window >= length:
        return [0]
    overlap_ratio = max(0.0, min(0.95, float(overlap_ratio)))
    step = max(1, int(round(window * (1.0 - overlap_ratio))))
    last_start = length - window
    starts = list(range(0, last_start + 1, step))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def generate_slice_windows_for_size(
    width: int,
    height: int,
    slice_width: int,
    slice_height: int,
    overlap_width_ratio: float = 0.2,
    overlap_height_ratio: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """Generate SAHI-style slice windows as x, y, width, height."""
    y_starts = slice_axis_starts(height, slice_height, overlap_height_ratio)
    x_starts = slice_axis_starts(width, slice_width, overlap_width_ratio)
    clipped_width = min(int(slice_width), int(width))
    clipped_height = min(int(slice_height), int(height))
    return [
        (x, y, min(clipped_width, int(width) - x), min(clipped_height, int(height) - y))
        for y in y_starts
        for x in x_starts
    ]


def bbox_intersects_window(bbox: Sequence[float], window: Tuple[int, int, int, int]) -> bool:
    """Return whether a COCO xywh bbox intersects a window."""
    x, y, w, h = [float(value) for value in bbox[:4]]
    wx, wy, ww, wh = [float(value) for value in window]
    return x < wx + ww and x + w > wx and y < wy + wh and y + h > wy


def bbox_to_window_local(
    bbox: Sequence[float],
    window: Tuple[int, int, int, int],
) -> Optional[List[float]]:
    """Clip a COCO bbox to a window and return local xywh."""
    x, y, w, h = [float(value) for value in bbox[:4]]
    wx, wy, ww, wh = [float(value) for value in window]
    x1 = max(x, wx)
    y1 = max(y, wy)
    x2 = min(x + w, wx + ww)
    y2 = min(y + h, wy + wh)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 - wx, y1 - wy, x2 - x1, y2 - y1]


def build_model_input_cases(
    images: Sequence[Any],
    config: Mapping[str, Any],
    stats_rows: Sequence[Mapping[str, Any]],
    *,
    max_cases: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build manifest rows for images actually sent to the model."""
    mode = canonical_test_mode(config)
    sahi_cfg = config.get("sahi", {})
    stats_by_image = {int(row.get("image_id", 0)): row for row in stats_rows}
    cases: List[Dict[str, Any]] = []
    case_limit = None if max_cases is None else max(0, int(max_cases))
    if case_limit == 0:
        return cases
    case_index = 1
    for image in images:
        image_id = int(get_value(image, "image_id"))
        width = int(get_value(image, "width"))
        height = int(get_value(image, "height"))
        stat = stats_by_image.get(image_id, {})
        if mode == SAHI_MODE:
            windows = generate_slice_windows_for_size(
                width=width,
                height=height,
                slice_width=int(sahi_cfg.get("slice_width", width)),
                slice_height=int(sahi_cfg.get("slice_height", height)),
                overlap_width_ratio=float(sahi_cfg.get("overlap_width_ratio", 0.2)),
                overlap_height_ratio=float(sahi_cfg.get("overlap_height_ratio", 0.2)),
            )
            for slice_index, (x, y, w, h) in enumerate(windows, start=1):
                cases.append(
                    {
                        "case_index": case_index,
                        "image_id": image_id,
                        "file_name": get_value(image, "file_name", ""),
                        "source_path": get_value(image, "path", ""),
                        "input_type": "sahi_slice",
                        "slice_index": slice_index,
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "fallback": False,
                    }
                )
                case_index += 1
                if case_limit is not None and len(cases) >= case_limit:
                    return cases
        elif mode == CLASS_CROP_MODE and stat.get("crop_x") is not None:
            cases.append(
                {
                    "case_index": case_index,
                    "image_id": image_id,
                    "file_name": get_value(image, "file_name", ""),
                    "source_path": get_value(image, "path", ""),
                    "input_type": stat.get("model_input_type", "class_crop"),
                    "slice_index": 0,
                    "x": int(stat.get("crop_x", 0)),
                    "y": int(stat.get("crop_y", 0)),
                    "width": int(stat.get("crop_width", width)),
                    "height": int(stat.get("crop_height", height)),
                    "fallback": bool(stat.get("crop_fallback", False)),
                    "crop_source_matches": int(stat.get("crop_source_matches", 0) or 0),
                }
            )
            case_index += 1
            if case_limit is not None and len(cases) >= case_limit:
                return cases
        else:
            cases.append(
                {
                    "case_index": case_index,
                    "image_id": image_id,
                    "file_name": get_value(image, "file_name", ""),
                    "source_path": get_value(image, "path", ""),
                    "input_type": "full_image",
                    "slice_index": 0,
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "fallback": bool(stat.get("crop_fallback", False)),
                }
            )
            case_index += 1
            if case_limit is not None and len(cases) >= case_limit:
                return cases
    return cases


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 13)
    except Exception:
        return ImageFont.load_default()


def _draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: Sequence[Mapping[str, Any]],
    window: Tuple[int, int, int, int],
    scale: float,
    color: Tuple[int, int, int],
    names: Mapping[int, str],
    with_scores: bool,
) -> None:
    font = _font()
    for item in boxes:
        local = bbox_to_window_local(item.get("bbox", [0, 0, 0, 0]), window)
        if local is None:
            continue
        x, y, w, h = local
        xyxy = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
        draw.rectangle(xyxy, outline=color, width=2)
        category_id = int(item.get("category_id", item.get("class_id", 0)))
        label = names.get(category_id, str(category_id))
        if with_scores and item.get("score") is not None:
            label = f"{label} {float(item['score']):.2f}"
        text_bbox = draw.textbbox((xyxy[0] + 2, xyxy[1] + 2), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xyxy[0] + 2, xyxy[1] + 2), label, fill=(255, 255, 255), font=font)


def _render_batch_grid(
    batch_cases: Sequence[Mapping[str, Any]],
    annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    predictions_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    names: Mapping[int, str],
    output_path: Path,
    draw_predictions: bool,
) -> None:
    tile_size = 320
    label_h = 28
    cols = 3
    rows = 3
    canvas = Image.new("RGB", (cols * tile_size, rows * (tile_size + label_h)), color=(245, 245, 245))
    font = _font()
    for index, case in enumerate(batch_cases[: cols * rows]):
        col = index % cols
        row = index // cols
        x0 = col * tile_size
        y0 = row * (tile_size + label_h)
        source_path = Path(str(case["source_path"]))
        window = (int(case["x"]), int(case["y"]), int(case["width"]), int(case["height"]))
        try:
            with Image.open(source_path) as source:
                crop = source.convert("RGB").crop((window[0], window[1], window[0] + window[2], window[1] + window[3]))
        except Exception:
            crop = Image.new("RGB", (max(1, window[2]), max(1, window[3])), color=(40, 40, 40))
        scale = min(tile_size / max(1, crop.width), tile_size / max(1, crop.height))
        resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        tile = Image.new("RGB", (tile_size, tile_size), color=(20, 20, 20))
        paste_x = (tile_size - resized.width) // 2
        paste_y = (tile_size - resized.height) // 2
        tile.paste(resized, (paste_x, paste_y))

        overlay = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        shifted = ImageDraw.Draw(overlay)
        draw_offset_window = (window[0] - paste_x / scale, window[1] - paste_y / scale, tile_size / scale, tile_size / scale)
        image_id = int(case["image_id"])
        if draw_predictions:
            _draw_boxes(
                shifted,
                predictions_by_image.get(image_id, []),
                draw_offset_window,
                scale,
                (239, 68, 68),
                names,
                with_scores=True,
            )
        else:
            _draw_boxes(
                shifted,
                annotations_by_image.get(image_id, []),
                draw_offset_window,
                scale,
                (34, 197, 94),
                names,
                with_scores=False,
            )
        tile = Image.alpha_composite(tile.convert("RGBA"), overlay).convert("RGB")
        canvas.paste(tile, (x0, y0))
        label = f"{case['case_index']} img={image_id} {case.get('input_type', '')}"
        ImageDraw.Draw(canvas).text((x0 + 6, y0 + tile_size + 7), label[:48], fill=(20, 20, 20), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def write_model_input_artifacts(
    output_dir: Path,
    images: Sequence[Any],
    categories: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    stats_rows: Sequence[Mapping[str, Any]],
    prefix: str = "test",
) -> List[Dict[str, Any]]:
    """Write model-input manifest and Ultralytics-style batch grids."""
    output_cfg = config.get("output", {})
    max_batches = max(0, int(output_cfg.get("max_model_input_batches", 3)))
    batch_size = max(1, int(output_cfg.get("model_input_batch_size", 9)))
    full_manifest = bool(output_cfg.get("full_model_input_manifest", False))
    case_limit = None if full_manifest else max_batches * batch_size
    cases = build_model_input_cases(images, config, stats_rows, max_cases=case_limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_index",
        "image_id",
        "file_name",
        "source_path",
        "input_type",
        "slice_index",
        "x",
        "y",
        "width",
        "height",
        "fallback",
        "crop_source_matches",
    ]
    csv_path = output_dir / "model_inputs_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in cases:
            writer.writerow(row)
    json_path = output_dir / "model_inputs_manifest.json"
    json_path.write_text(
        json.dumps(
            {
                "full_manifest": full_manifest,
                "case_limit": case_limit,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    annotations_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for annotation in annotations:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    predictions_by_image: Dict[int, List[Mapping[str, Any]]] = {}
    for prediction in predictions:
        predictions_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)
    names, _ = category_maps(categories)

    manifest_description = "Full model input manifest" if full_manifest else "Sampled model input manifest"
    manifest_rows = [
        {"path": str(csv_path), "kind": "model_inputs", "description": f"{manifest_description} CSV."},
        {"path": str(json_path), "kind": "model_inputs", "description": f"{manifest_description} JSON."},
    ]
    for batch_index in range(max_batches):
        start = batch_index * batch_size
        batch_cases = cases[start : start + batch_size]
        if not batch_cases:
            break
        labels_path = output_dir / f"{prefix}_batch{batch_index}_labels.jpg"
        preds_path = output_dir / f"{prefix}_batch{batch_index}_pred.jpg"
        _render_batch_grid(batch_cases, annotations_by_image, predictions_by_image, names, labels_path, draw_predictions=False)
        _render_batch_grid(batch_cases, annotations_by_image, predictions_by_image, names, preds_path, draw_predictions=True)
        manifest_rows.append({"path": str(labels_path), "kind": "model_inputs", "description": "Model input label batch grid."})
        manifest_rows.append({"path": str(preds_path), "kind": "model_inputs", "description": "Model input prediction batch grid."})
    return manifest_rows
