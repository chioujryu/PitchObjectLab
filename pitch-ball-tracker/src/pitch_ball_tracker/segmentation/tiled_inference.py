from __future__ import annotations

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from pitch_ball_tracker.segmentation.sam3_segmentor import SAM3Segmentor, SegmentResult, _nms_numpy


class TiledInference:
    """
    Splits a high-resolution frame into overlapping tiles, runs SAM3 on each
    tile independently, then merges all detections back into frame coordinates
    using NMS.

    Tile layout (stride = tile_size - overlap):

        |<--- tile_size --->|
        |<-- stride -->|
        +------------------+
        |   tile [0,0]     |---+
        +------------------+  |
                    |          |
                    +------------------+
                    |   tile [0,1]     |
                    +------------------+
    """

    def __init__(self, segmentor: SAM3Segmentor, cfg: DictConfig) -> None:
        self._seg = segmentor
        self._tile_size = cfg.inference.tile_size
        self._overlap = cfg.inference.tile_overlap
        self._merge_iou = cfg.inference.tile_merge_iou
        self._stride = self._tile_size - self._overlap

    def segment(self, frame_bgr: np.ndarray) -> SegmentResult:
        H, W = frame_bgr.shape[:2]
        tiles = list(self._generate_tiles(H, W))
        logger.debug(f"Tiled inference: {len(tiles)} tiles on {W}x{H} frame")

        all_masks: list[np.ndarray] = []
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        for (r0, r1, c0, c1) in tiles:
            tile = frame_bgr[r0:r1, c0:c1]
            result = self._seg.segment(tile)
            if len(result.boxes) == 0:
                continue

            # Translate boxes from tile coords → frame coords
            offset = np.array([c0, r0, c0, r0], dtype=np.float32)
            boxes_frame = result.boxes + offset

            # Expand masks from tile size → full frame size (sparse placement)
            fh, fw = H, W
            frame_masks = np.zeros((len(result.masks), fh, fw), dtype=bool)
            for k, m in enumerate(result.masks):
                th, tw = m.shape
                frame_masks[k, r0 : r0 + th, c0 : c0 + tw] = m

            all_masks.append(frame_masks)
            all_boxes.append(boxes_frame)
            all_scores.append(result.scores)

        if not all_boxes:
            return SegmentResult(
                masks=np.empty((0, H, W), dtype=bool),
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        masks_all = np.concatenate(all_masks, axis=0)
        boxes_all = np.concatenate(all_boxes, axis=0)
        scores_all = np.concatenate(all_scores, axis=0)

        keep = _nms_numpy(boxes_all, scores_all, iou_threshold=self._merge_iou)
        return SegmentResult(
            masks=masks_all[keep],
            boxes=boxes_all[keep],
            scores=scores_all[keep],
        )

    def _generate_tiles(self, H: int, W: int):
        """Yield (r0, r1, c0, c1) for each tile, ensuring full coverage."""
        r0 = 0
        while r0 < H:
            r1 = min(r0 + self._tile_size, H)
            c0 = 0
            while c0 < W:
                c1 = min(c0 + self._tile_size, W)
                yield (r0, r1, c0, c1)
                if c1 == W:
                    break
                c0 += self._stride
            if r1 == H:
                break
            r0 += self._stride
