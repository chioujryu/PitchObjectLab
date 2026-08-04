"""Label Studio sequence parsing, cache fingerprints, and tracking proxy metrics.

The supplied football annotations contain boxes and semantic roles but no identity
labels. Therefore metrics from this module are explicitly marked as proxies; an
official HOTA/IDF1 claim requires human-reviewed track identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


SELECTED_SEGMENT_IDS = (
    "49523ed0-3740-47e1-a206-8c40ae0335af",
    "519816ce-d533-403b-a8e9-12213e7fa785",
    "e196962d-3d24-4218-a8ef-7fd24fa48e49",
)


@dataclass(frozen=True)
class GTBox:
    bbox: Tuple[float, float, float, float]
    role: str
    label: str


@dataclass(frozen=True)
class FrameGT:
    segment_id: str
    frame_index: int
    timestamp: float
    image_name: str
    width: int
    height: int
    boxes: Tuple[GTBox, ...]


def _annotation_results(task: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    annotations = task.get("annotations") or task.get("predictions") or []
    preferred = [item for item in annotations if item.get("ground_truth")]
    selected = preferred or list(annotations[:1])
    for annotation in selected:
        for result in annotation.get("result", []):
            if result.get("type") == "rectanglelabels":
                yield result


def _frame_metadata(task: Mapping[str, Any]) -> Tuple[str, int, float, str]:
    data = task.get("data", {}) or {}
    split = data.get("frameSplit", {}) or {}
    frame = split.get("frame", {}) or {}
    segment_id = str(data.get("segment_id") or split.get("segment", {}).get("id") or "unknown")
    image_name = str(data.get("image_name") or Path(str(data.get("image", "unknown.jpg"))).name)
    match = re.search(r"_t(\d+)", image_name)
    fallback_ms = int(match.group(1)) if match else 0
    frame_index = int(frame.get("index", 0))
    timestamp = float(frame.get("timestamp_seconds", fallback_ms / 1000.0))
    return segment_id, frame_index, timestamp, image_name


def load_label_studio_sequences(root: Path) -> Dict[str, List[FrameGT]]:
    """Load tasks.json, converting percent boxes to pixels and sorting each segment."""
    root = Path(root)
    with (root / "tasks.json").open("r", encoding="utf-8") as file:
        tasks = json.load(file)
    grouped: Dict[str, List[FrameGT]] = {}
    for task in tasks:
        segment_id, frame_index, timestamp, image_name = _frame_metadata(task)
        boxes: List[GTBox] = []
        width = height = 0
        for result in _annotation_results(task):
            width = int(result.get("original_width") or width)
            height = int(result.get("original_height") or height)
            value = result.get("value", {}) or {}
            labels = value.get("rectanglelabels") or []
            if not labels or width <= 0 or height <= 0:
                continue
            box_width = float(value.get("width", 0.0)) * width / 100.0
            box_height = float(value.get("height", 0.0)) * height / 100.0
            if box_width <= 0.0 or box_height <= 0.0:
                continue
            label = str(labels[0])
            role = "key" if label.casefold() == "key_soccer_ball" else "side"
            boxes.append(
                GTBox(
                    (
                        float(value.get("x", 0.0)) * width / 100.0,
                        float(value.get("y", 0.0)) * height / 100.0,
                        box_width,
                        box_height,
                    ),
                    role,
                    label,
                )
            )
        if width <= 0 or height <= 0:
            split = (task.get("data", {}) or {}).get("frameSplit", {}) or {}
            resolution = split.get("dataset", {}).get("SHOOTING_INFO")
            width, height = 1920, 1080
            if resolution:
                try:
                    info = json.loads(resolution)
                    numbers = re.findall(r"\d+", str(info.get("resolution", "")))
                    if len(numbers) >= 2:
                        width, height = int(numbers[0]), int(numbers[1])
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
        grouped.setdefault(segment_id, []).append(
            FrameGT(segment_id, frame_index, timestamp, image_name, width, height, tuple(boxes))
        )
    for frames in grouped.values():
        frames.sort(key=lambda item: (item.frame_index, item.timestamp, item.image_name))
    return grouped


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0


def link_pseudo_gt_tracks(
    frames: Sequence[FrameGT], max_distance_pixels: float = 120.0
) -> List[List[Dict[str, Any]]]:
    """Deterministically link role-consistent boxes; result is review aid, not official identity GT."""
    next_id = 1
    previous: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {"key": [], "side": []}
    output: List[List[Dict[str, Any]]] = []
    for frame in frames:
        linked: List[Dict[str, Any]] = []
        current_by_role: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {"key": [], "side": []}
        for role in ("key", "side"):
            role_boxes = [(index, box) for index, box in enumerate(frame.boxes) if box.role == role]
            role_previous = previous[role]
            assignments: Dict[int, int] = {}
            if role_boxes and role_previous:
                costs = np.full((len(role_previous), len(role_boxes)), 1.0e6, dtype=np.float64)
                for old_index, (_track_id, old_center) in enumerate(role_previous):
                    for new_index, (_box_index, box) in enumerate(role_boxes):
                        center = _bbox_center(box.bbox)
                        distance = math.hypot(center[0] - old_center[0], center[1] - old_center[1])
                        if distance <= max_distance_pixels:
                            costs[old_index, new_index] = distance
                rows, columns = linear_sum_assignment(costs)
                for old_index, new_index in zip(rows.tolist(), columns.tolist()):
                    if costs[old_index, new_index] < 1.0e6:
                        assignments[new_index] = role_previous[old_index][0]
            for role_index, (_box_index, box) in enumerate(role_boxes):
                track_id = assignments.get(role_index)
                if track_id is None:
                    track_id = next_id
                    next_id += 1
                center = _bbox_center(box.bbox)
                current_by_role[role].append((track_id, center))
                linked.append(
                    {
                        "segment_id": frame.segment_id,
                        "frame_index": frame.frame_index,
                        "gt_track_id": track_id,
                        "bbox": list(box.bbox),
                        "role": box.role,
                        "label": box.label,
                    }
                )
        previous = current_by_role
        linked.sort(key=lambda row: int(row["gt_track_id"]))
        output.append(linked)
    return output


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lw, lh = (float(value) for value in left[:4])
    rx1, ry1, rw, rh = (float(value) for value in right[:4])
    lx2, ly2, rx2, ry2 = lx1 + lw, ly1 + lh, rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = max(0.0, lw * lh) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0.0 else 0.0


def match_tracking_rows(
    pseudo_gt_by_frame: Sequence[Sequence[Mapping[str, Any]]],
    prediction_rows: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.30,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Framewise one-to-one IoU matching used before proxy association metrics."""
    predictions: Dict[int, List[Mapping[str, Any]]] = {}
    for row in prediction_rows:
        if row.get("track_id") is not None:
            predictions.setdefault(int(row["frame_index"]), []).append(row)
    matches: List[Dict[str, Any]] = []
    gt_count = sum(len(rows) for rows in pseudo_gt_by_frame)
    prediction_count = sum(len(rows) for rows in predictions.values())
    for gt_rows in pseudo_gt_by_frame:
        if not gt_rows:
            continue
        frame_index = int(gt_rows[0]["frame_index"])
        predicted = predictions.get(frame_index, [])
        if not predicted:
            continue
        costs = np.ones((len(gt_rows), len(predicted)), dtype=np.float64)
        for gt_index, gt in enumerate(gt_rows):
            for pred_index, pred in enumerate(predicted):
                costs[gt_index, pred_index] = 1.0 - bbox_iou(gt["bbox"], pred.get("bbox", (0, 0, 0, 0)))
        rows, columns = linear_sum_assignment(costs)
        for gt_index, pred_index in zip(rows.tolist(), columns.tolist()):
            iou = 1.0 - float(costs[gt_index, pred_index])
            if iou >= iou_threshold:
                matches.append(
                    {
                        "frame_index": frame_index,
                        "gt_track_id": int(gt_rows[gt_index]["gt_track_id"]),
                        "track_id": int(predicted[pred_index]["track_id"]),
                        "iou": iou,
                        "role": gt_rows[gt_index].get("role"),
                    }
                )
    return matches, gt_count, prediction_count


def association_proxy_metrics(
    matches: Sequence[Mapping[str, Any]], gt_count: int, prediction_count: int
) -> Dict[str, Any]:
    """Calculate transparent identity proxies from pseudo-linked GT boxes."""
    ordered = sorted(matches, key=lambda row: (int(row["frame_index"]), int(row["gt_track_id"])))
    last_prediction_for_gt: Dict[int, int] = {}
    gt_ids_for_prediction: Dict[int, set] = {}
    id_switches = 0
    for row in ordered:
        gt_id, prediction_id = int(row["gt_track_id"]), int(row["track_id"])
        previous = last_prediction_for_gt.get(gt_id)
        if previous is not None and previous != prediction_id:
            id_switches += 1
        last_prediction_for_gt[gt_id] = prediction_id
        gt_ids_for_prediction.setdefault(prediction_id, set()).add(gt_id)
    false_merges = sum(max(0, len(gt_ids) - 1) for gt_ids in gt_ids_for_prediction.values())
    true_matches = len(matches)
    detection_accuracy = true_matches / max(1, gt_count + prediction_count - true_matches)
    association_errors = id_switches + false_merges
    association_accuracy = true_matches / max(1, true_matches + association_errors)
    hota_proxy = math.sqrt(detection_accuracy * association_accuracy)
    idf1_proxy = 2.0 * true_matches / max(1, gt_count + prediction_count)
    return {
        "official_hota": False,
        "official_metrics_reason": "source annotations have no reviewed track identities",
        "matches": true_matches,
        "gt_boxes": int(gt_count),
        "predicted_tracked_boxes": int(prediction_count),
        "id_switches": id_switches,
        "false_merges": false_merges,
        "detection_accuracy_proxy": detection_accuracy,
        "association_accuracy_proxy": association_accuracy,
        "hota_proxy": hota_proxy,
        "idf1_proxy": idf1_proxy,
    }


def cache_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def detection_cache_identity(config_path: Path, checkpoint_path: Path, source_root: Path) -> Dict[str, Any]:
    """Fingerprint config/checkpoint/tasks/source listing so stale caches are rejected."""
    source_root = Path(source_root)
    images = sorted(source_root.glob("*.jpg")) + sorted(source_root.glob("*.jpeg"))
    payload = {
        "config_sha256": file_sha256(config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "tasks_sha256": file_sha256(source_root / "tasks.json"),
        "source": [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in images],
    }
    return {**payload, "fingerprint": cache_fingerprint(payload), "image_count": len(images)}
