from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from pitch_ball_tracker.detection.ball_filter import BallFilter
from pitch_ball_tracker.filtering.field_filter import FieldFilter
from pitch_ball_tracker.filtering.motion_filter import MotionFilter
from pitch_ball_tracker.segmentation.sam3_segmentor import SAM3Segmentor
from pitch_ball_tracker.segmentation.tiled_inference import TiledInference
from pitch_ball_tracker.tracking.reid import BallReIDExtractor
from pitch_ball_tracker.tracking.tracker import BotSortTracker
from pitch_ball_tracker.tracking.tracklet import Tracklet
from pitch_ball_tracker.utils.video_io import VideoReader, VideoWriter
from pitch_ball_tracker.utils.visualization import Visualizer


class BallTrackerPipeline:
    """End-to-end ball-tracking pipeline.

    Frame-level data flow: VideoReader
        → FieldFilter.update()          (refresh field mask every 30 frames)
        → MotionFilter.update_frame()   (push frame into optical-flow buffer)
        → [SAM3Segmentor | TiledInference].segment()
        → BallFilter.filter()           (geometric shape filtering)
        → MotionFilter.filter()         (motion + field filtering)
        → BotSortTracker.update()       (Kalman + Hungarian + ReID)
        → Visualizer.draw()
        → VideoWriter.write()
        → track records accumulated
    → save_tracks()
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._device = self._resolve_device()

        logger.info("Initializing pipeline components …")
        self._segmentor = SAM3Segmentor(cfg)
        self._tiled = TiledInference(self._segmentor, cfg) if cfg.inference.tiled else None
        self._ball_filter = BallFilter(cfg)
        self._motion_filter = MotionFilter(cfg)
        self._field_filter = FieldFilter(cfg)
        self._reid = self._build_reid()
        self._tracker = BotSortTracker(cfg, self._reid)
        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, video_path: str | Path, output_dir: str | Path) -> None:
        """Process an entire video file and write outputs to `output_dir`."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        reader = VideoReader(video_path)
        stem = video_path.stem
        fps = reader.fps
        w, h = reader.width, reader.height

        writer: VideoWriter | None = None
        if self.cfg.output.save_video:
            out_video = output_dir / f"{stem}_tracked.mp4"
            writer = VideoWriter(out_video, fps, w, h, codec=self.cfg.output.video_codec)

        viz = Visualizer(
            draw_masks=self.cfg.output.draw_masks,
            draw_trails=self.cfg.output.draw_trails,
            trail_length=self.cfg.output.trail_length,
        )

        track_records: list[dict] = []
        t0 = time.perf_counter()

        try:
            for frame_idx, frame in tqdm(reader, total=reader.total_frames, desc="Tracking"):
                confirmed = self._process_frame(frame, frame_idx)

                # Collect track records
                for t in confirmed:
                    box = t.get_bbox()
                    track_records.append(
                        {
                            "frame": frame_idx,
                            "track_id": t.track_id,
                            "bbox_xyxy": box.tolist(),
                            "score": float(t.score),
                        }
                    )

                if writer is not None:
                    field_mask = self._field_filter._mask
                    vis_frame = viz.draw(frame, confirmed, field_mask, frame_idx)
                    writer.write(vis_frame)

                viz.cleanup_lost_trails({t.track_id for t in confirmed})

        finally:
            reader.release()
            if writer is not None:
                writer.release()

        elapsed = time.perf_counter() - t0
        logger.info(f"Done: {frame_idx + 1} frames in {elapsed:.1f}s ({(frame_idx + 1) / elapsed:.1f} fps)")

        if self.cfg.output.save_tracks:
            self._save_tracks(track_records, output_dir, stem)

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int = 0) -> list[Tracklet]:
        """Public single-frame entry point (useful for streaming / testing)."""
        return self._process_frame(frame_bgr, frame_idx)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> list[Tracklet]:
        # 1. Field boundary mask
        field_mask = self._field_filter.update(frame)

        # 2. Motion buffer update
        self._motion_filter.update_frame(frame)

        # 3. Segmentation (tiled or standard)
        seg_result = self._tiled.segment(frame) if self._tiled is not None else self._segmentor.segment(frame)

        if frame_idx < 10 or frame_idx % 30 == 0:
            logger.debug(
                f"[frame {frame_idx}] SAM3: {len(seg_result.masks)} detections"
                + (f", scores={seg_result.scores.tolist()}" if len(seg_result.scores) else "")
            )

        # 4. Geometric ball filtering
        candidates = self._ball_filter.filter(seg_result, frame_idx)

        if frame_idx < 10 or frame_idx % 30 == 0:
            logger.debug(f"[frame {frame_idx}] BallFilter: {len(candidates)} candidates")

        # 5. Motion + field filtering
        candidates = self._motion_filter.filter(candidates, field_mask)

        if frame_idx < 10 or frame_idx % 30 == 0:
            logger.debug(f"[frame {frame_idx}] MotionFilter: {len(candidates)} candidates")

        # 6. Tracking
        confirmed = self._tracker.update(candidates, frame)

        if frame_idx < 10 or frame_idx % 30 == 0:
            logger.debug(f"[frame {frame_idx}] Tracker confirmed: {len(confirmed)}")

        return confirmed

    def _build_reid(self) -> BallReIDExtractor | None:
        if not self.cfg.tracking.use_reid:
            return None
        return BallReIDExtractor(
            device=self._device,
            embedding_dim=self.cfg.reid.embedding_dim,
            half=self.cfg.model.half_precision,
        )

    def _resolve_device(self) -> torch.device:
        gpus: list[int] = list(self.cfg.model.gpus)
        if torch.cuda.is_available() and gpus:
            return torch.device(f"cuda:{gpus[0]}")
        return torch.device("cpu")

    @staticmethod
    def _save_tracks(
        records: list[dict],
        output_dir: Path,
        stem: str,
    ) -> None:
        out_path = output_dir / f"{stem}_tracks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        logger.info(f"Track data saved → {out_path} ({len(records)} records)")
