from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@dataclass
class SegmentResult:
    """Output of one segmentation pass on a single frame."""

    masks: np.ndarray  # (N, H, W) bool
    boxes: np.ndarray  # (N, 4) xyxy pixel coords
    scores: np.ndarray  # (N,) float32


class SAM3Segmentor:
    """Wraps Sam3Processor for per-frame automatic segmentation driven by text prompts (e.g. "ball", "soccer ball").

    Half-precision and GPU selection are handled here; the processor is kept in eval mode throughout inference.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self.device = self._resolve_device(cfg.model.gpus)
        self.half = cfg.model.half_precision and self.device.type == "cuda"
        self._prompts: list[str] = list(cfg.inference.text_prompts)
        self._confidence = cfg.inference.confidence_threshold
        self._processor = self._build_processor()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _resolve_device(self, gpus: list[int]) -> torch.device:
        if torch.cuda.is_available() and gpus:
            return torch.device(f"cuda:{gpus[0]}")
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, running on CPU (will be slow).")
        return torch.device("cpu")

    def _build_processor(self) -> Sam3Processor:
        logger.info(
            f"Loading SAM3 image model (type={self._cfg.model.sam3_type}, device={self.device}, half={self.half})"
        )
        checkpoint = self._cfg.model.sam3_checkpoint  # None → HuggingFace auto-download
        model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            device=str(self.device),
            eval_mode=True,
        )
        model = model.to(self.device)
        processor = Sam3Processor(
            model=model,
            device=str(self.device),
            confidence_threshold=self._confidence,
        )
        logger.info("SAM3 model ready.")
        return processor

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def segment(self, frame_bgr: np.ndarray) -> SegmentResult:
        """Segment a single BGR frame (numpy HxWx3 uint8). Runs all configured text prompts and merges results before
        returning.
        """
        import cv2
        from PIL import Image as PILImage

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = PILImage.fromarray(frame_rgb)

        all_masks: list[np.ndarray] = []
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.half,
        )
        for prompt in self._prompts:
            state: dict = {}
            with autocast_ctx:
                state = self._processor.set_image(frame_pil, state)
                state = self._processor.set_text_prompt(prompt, state)

            if "boxes" not in state or len(state["boxes"]) == 0:
                continue

            boxes = state["boxes"].cpu().float().numpy()  # (N, 4) xyxy
            masks = state["masks"].squeeze(1).cpu().numpy()  # (N, H, W) bool
            scores = state["scores"].cpu().float().numpy()  # (N,)

            all_masks.append(masks)
            all_boxes.append(boxes)
            all_scores.append(scores)

        if not all_boxes:
            h, w = frame_bgr.shape[:2]
            return SegmentResult(
                masks=np.empty((0, h, w), dtype=bool),
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        masks_all = np.concatenate(all_masks, axis=0)
        boxes_all = np.concatenate(all_boxes, axis=0)
        scores_all = np.concatenate(all_scores, axis=0)

        # Cross-prompt NMS to remove duplicates
        keep = _nms_numpy(boxes_all, scores_all, iou_threshold=0.5)
        return SegmentResult(
            masks=masks_all[keep],
            boxes=boxes_all[keep],
            scores=scores_all[keep],
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Simple greedy NMS over xyxy boxes."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        ix1 = np.maximum(x1[i], x1)
        iy1 = np.maximum(y1[i], y1)
        ix2 = np.minimum(x2[i], x2)
        iy2 = np.minimum(y2[i], y2)
        inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
        iou = inter / (areas[i] + areas - inter + 1e-6)
        suppressed |= iou > iou_threshold
        suppressed[i] = False
    return keep
