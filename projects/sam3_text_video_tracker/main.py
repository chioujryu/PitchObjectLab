#!/usr/bin/env python3
"""SAM3 text-guided single-target video tracking demo project."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics.models.sam import SAM3VideoSemanticPredictor


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter_area / (area_a + area_b - inter_area + 1e-6)


def box_center(box: np.ndarray) -> tuple[float, float]:
    """Return center (x, y) from xyxy box."""
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def box_area(box: np.ndarray) -> float:
    """Return area for xyxy box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def clip_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clip xyxy box to image bounds."""
    x1 = np.clip(box[0], 0, w - 1)
    y1 = np.clip(box[1], 0, h - 1)
    x2 = np.clip(box[2], 0, w - 1)
    y2 = np.clip(box[3], 0, h - 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def roi_hist_hsv(frame: np.ndarray, box: np.ndarray, bins: tuple[int, int] = (30, 32)) -> np.ndarray | None:
    """Build normalized HSV histogram for the box ROI."""
    x1, y1, x2, y2 = box.astype(int)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins[0], bins[1]], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


@dataclass
class TrackConfig:
    model: str
    source: str
    text_prompt: str
    output_video: str
    output_csv: str
    imgsz: int
    conf: float
    iou: float
    half: bool
    vid_stride: int
    init_pick: str
    target_index: int
    min_score: float
    reid_max_lost: int
    reid_min_iou: float
    reid_hist_weight: float
    reid_area_weight: float
    use_ema_smoothing: bool
    ema_alpha: float
    use_kalman: bool
    draw_mask: bool
    mask_alpha: float
    draw_all_candidates: bool
    save_debug_frame: bool
    output_full_video: bool


class TargetState:
    """Keep single-target tracking states."""

    def __init__(self, cfg: TrackConfig):
        self.cfg = cfg
        self.target_id: int | None = None
        self.last_box: np.ndarray | None = None
        self.smoothed_box: np.ndarray | None = None
        self.last_hist: np.ndarray | None = None
        self.lost_frames = 0
        self.kf = self._init_kalman() if cfg.use_kalman else None

    @staticmethod
    def _init_kalman() -> cv2.KalmanFilter:
        # State = [cx, cy, vx, vy, w, h]
        kf = cv2.KalmanFilter(6, 4)
        kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # cx
                [0, 1, 0, 0, 0, 0],  # cy
                [0, 0, 0, 0, 1, 0],  # w
                [0, 0, 0, 0, 0, 1],  # h
            ],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.2
        return kf

    def _kalman_predict_box(self) -> np.ndarray | None:
        if self.kf is None:
            return None
        pred = self.kf.predict()
        cx, cy, _, _, w, h = pred.flatten().tolist()
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    def _kalman_update(self, box: np.ndarray) -> None:
        if self.kf is None:
            return
        cx, cy = box_center(box)
        w = box[2] - box[0]
        h = box[3] - box[1]
        m = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        if self.last_box is None:
            self.kf.statePost = np.array([[cx], [cy], [0], [0], [w], [h]], dtype=np.float32)
        self.kf.correct(m)

    def _ema(self, box: np.ndarray) -> np.ndarray:
        if not self.cfg.use_ema_smoothing or self.smoothed_box is None:
            self.smoothed_box = box.copy()
            return box
        alpha = self.cfg.ema_alpha
        self.smoothed_box = alpha * box + (1.0 - alpha) * self.smoothed_box
        return self.smoothed_box.copy()

    def _select_initial_target(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
        if self.cfg.init_pick == "largest":
            ordered = sorted(candidates, key=lambda c: c["area"], reverse=True)
        idx = max(0, self.cfg.target_index)
        idx = min(idx, len(ordered) - 1)
        return ordered[idx]

    def _pick_same_id(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.target_id is None:
            return None
        same = [c for c in candidates if c["obj_id"] == self.target_id]
        if not same:
            return None
        return max(same, key=lambda c: c["score"])

    def _pick_reid(self, frame: np.ndarray, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.last_box is None or not candidates or self.lost_frames > self.cfg.reid_max_lost:
            return None

        best, best_score = None, -math.inf
        ref_area = box_area(self.last_box) + 1e-6
        for c in candidates:
            iou_s = iou_xyxy(self.last_box, c["box"])
            if iou_s < self.cfg.reid_min_iou and self.lost_frames <= 3:
                continue
            area_ratio = min(c["area"] / ref_area, ref_area / (c["area"] + 1e-6))
            area_s = max(0.0, min(1.0, area_ratio))
            hist_s = 0.0
            if self.last_hist is not None and c["hist"] is not None:
                hist_s = float(cv2.compareHist(self.last_hist, c["hist"], cv2.HISTCMP_CORREL))
                hist_s = (hist_s + 1.0) / 2.0  # map [-1,1] -> [0,1]
            score = (
                0.45 * c["score"]
                + 0.25 * iou_s
                + self.cfg.reid_area_weight * area_s
                + self.cfg.reid_hist_weight * hist_s
            )
            if score > best_score:
                best_score = score
                best = c
        return best

    def update(self, frame: np.ndarray, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        chosen = None
        if self.target_id is None:
            chosen = self._select_initial_target(candidates)
            if chosen is not None:
                self.target_id = chosen["obj_id"]
        else:
            chosen = self._pick_same_id(candidates)
            if chosen is None:
                chosen = self._pick_reid(frame, candidates)
                if chosen is not None:
                    self.target_id = chosen["obj_id"]

        if chosen is None:
            self.lost_frames += 1
            pred_box = self._kalman_predict_box()
            if pred_box is not None and self.smoothed_box is not None:
                self.smoothed_box = self._ema(pred_box)
            return None

        self.lost_frames = 0
        self.last_box = chosen["box"].copy()
        self.last_hist = chosen["hist"]
        self._kalman_update(chosen["box"])
        chosen["draw_box"] = self._ema(chosen["box"])
        return chosen


def parse_result(frame: np.ndarray, result: Any) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """Convert Ultralytics result object to candidate dict list."""
    candidates: list[dict[str, Any]] = []
    mask_data = None

    if result.masks is not None and hasattr(result.masks, "data") and result.masks.data is not None:
        mask_data = result.masks.data.detach().cpu().numpy().astype(np.uint8)

    if result.boxes is None or len(result.boxes) == 0:
        return candidates, mask_data

    data = result.boxes.data.detach().cpu().numpy()
    for i, row in enumerate(data):
        if len(row) < 6:
            continue
        x1, y1, x2, y2 = row[:4].astype(np.float32)
        obj_id = int(row[4]) if len(row) >= 7 else int(i)
        score = float(row[5]) if len(row) >= 7 else float(row[4])
        cls_id = int(row[6]) if len(row) >= 7 else 0
        box = clip_box(np.array([x1, y1, x2, y2], dtype=np.float32), frame.shape[1], frame.shape[0])
        candidates.append(
            {
                "box": box,
                "obj_id": obj_id,
                "score": score,
                "cls_id": cls_id,
                "area": box_area(box),
                "hist": roi_hist_hsv(frame, box),
                "mask": mask_data[i] if mask_data is not None and i < len(mask_data) else None,
            }
        )
    return candidates, mask_data


def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay binary mask on frame."""
    overlay = frame.copy()
    color = np.zeros_like(frame, dtype=np.uint8)
    color[:, :, 1] = 255
    m = mask.astype(bool)
    overlay[m] = cv2.addWeighted(frame[m], 1.0 - alpha, color[m], alpha, 0)
    return overlay


def load_cfg(path: str) -> TrackConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrackConfig(**raw)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run(cfg: TrackConfig) -> None:
    ensure_parent(cfg.output_video)
    ensure_parent(cfg.output_csv)

    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    vid_stride = max(1, int(cfg.vid_stride))
    output_fps = fps if cfg.output_full_video else max(fps / vid_stride, 1e-3)

    writer = cv2.VideoWriter(
        cfg.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    predictor = SAM3VideoSemanticPredictor(
        overrides={
            "conf": cfg.conf,
            "iou": cfg.iou,
            "imgsz": cfg.imgsz,
            "task": "segment",
            "mode": "predict",
            "model": cfg.model,
            "half": cfg.half,
            "vid_stride": vid_stride,
            "verbose": False,
        }
    )

    state = TargetState(cfg)
    stream_iter = iter(predictor(source=cfg.source, text=[cfg.text_prompt], stream=True))
    debug_dir = Path(cfg.output_video).parent / "debug_frames"
    if cfg.save_debug_frame:
        debug_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.output_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["frame_idx", "target_id", "x1", "y1", "x2", "y2", "score", "lost_frames", "status", "text_prompt"]
        )

        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {cfg.source}")

        frame_idx = 0
        det_stream_ended = False
        while True:
            ok, frame_raw = cap.read()
            if not ok:
                break
            frame = frame_raw.copy()
            is_detection_frame = (frame_idx % vid_stride) == 0
            chosen = None
            candidates: list[dict[str, Any]] = []

            if is_detection_frame and not det_stream_ended:
                try:
                    r = next(stream_iter)
                    frame = r.orig_img.copy()
                    candidates, _ = parse_result(frame, r)
                    candidates = [c for c in candidates if c["score"] >= cfg.min_score]
                    chosen = state.update(frame, candidates)
                except StopIteration:
                    det_stream_ended = True

            if cfg.draw_all_candidates and is_detection_frame:
                for c in candidates:
                    x1, y1, x2, y2 = c["box"].astype(int).tolist()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)

            if chosen is not None:
                draw_box = chosen["draw_box"] if "draw_box" in chosen else chosen["box"]
                x1, y1, x2, y2 = draw_box.astype(int).tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                status = "tracked"

                if cfg.draw_mask and chosen["mask"] is not None:
                    frame = draw_mask_overlay(frame, chosen["mask"], cfg.mask_alpha)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"id={state.target_id} score={chosen['score']:.2f} lost={state.lost_frames}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                csv_writer.writerow(
                    [
                        frame_idx,
                        state.target_id,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"{chosen['score']:.5f}",
                        state.lost_frames,
                        status,
                        cfg.text_prompt,
                    ]
                )
            else:
                status = "lost" if is_detection_frame else "skipped"
                if not is_detection_frame and state.smoothed_box is not None:
                    x1, y1, x2, y2 = state.smoothed_box.astype(int).tolist()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                    cv2.putText(
                        frame,
                        f"id={state.target_id} propagated",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 220, 255),
                        2,
                    )
                cv2.putText(frame, f"target {status} ({state.lost_frames})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                csv_writer.writerow([frame_idx, state.target_id, "", "", "", "", "", state.lost_frames, status, cfg.text_prompt])

            if cfg.output_full_video or is_detection_frame:
                writer.write(frame)
                if cfg.save_debug_frame and frame_idx % 30 == 0:
                    cv2.imwrite(str(debug_dir / f"frame_{frame_idx:06d}.jpg"), frame)

            frame_idx += 1

        cap.release()

    writer.release()
    print(f"Saved video: {cfg.output_video}")
    print(f"Saved csv: {cfg.output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 text-guided single-target video tracker")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--source", type=str, default=None, help="Override source video")
    parser.add_argument("--text", type=str, default=None, help="Override text prompt")
    parser.add_argument("--output", type=str, default=None, help="Override output video path")
    parser.add_argument("--csv", type=str, default=None, help="Override output csv path")
    parser.add_argument("--vid-stride", type=int, default=None, help="Run detection every N frames")
    parser.add_argument(
        "--output-full-video",
        dest="output_full_video",
        action="store_true",
        default=None,
        help="Write all source frames to output video",
    )
    parser.add_argument(
        "--det-only-video",
        dest="output_full_video",
        action="store_false",
        help="Write only detection frames to output video",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.source is not None:
        cfg.source = args.source
    if args.text is not None:
        cfg.text_prompt = args.text
    if args.output is not None:
        cfg.output_video = args.output
    if args.csv is not None:
        cfg.output_csv = args.csv
    if args.vid_stride is not None:
        cfg.vid_stride = max(1, args.vid_stride)
    if args.output_full_video is not None:
        cfg.output_full_video = args.output_full_video
    run(cfg)


if __name__ == "__main__":
    main()
