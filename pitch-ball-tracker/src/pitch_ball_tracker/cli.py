from __future__ import annotations

import click
from loguru import logger

from pitch_ball_tracker.pipeline import BallTrackerPipeline
from pitch_ball_tracker.utils.config import load_config


@click.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Input video file.")
@click.option(
    "--output-dir",
    "-o",
    default="results",
    show_default=True,
    help="Output directory for annotated video and track data.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    default="configs/default.yaml",
    show_default=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
# --- model overrides ---
@click.option("--half-precision/--no-half-precision", default=None, help="Enable FP16 half-precision inference.")
@click.option("--gpus", default=None, help="Comma-separated GPU IDs to use (e.g. '0' or '0,1').")
@click.option(
    "--sam3-checkpoint", default=None, type=click.Path(), help="Path to SAM3 checkpoint (default: auto-download)."
)
# --- inference overrides ---
@click.option("--tiled/--no-tiled", default=None, help="Enable tiled inference for high-resolution frames.")
@click.option("--tile-size", default=None, type=int, help="Tile size in pixels when tiled inference is enabled.")
@click.option("--confidence", default=None, type=float, help="SAM3 detection confidence threshold.")
# --- tracking overrides ---
@click.option("--use-reid/--no-reid", default=None, help="Enable ReID-based re-identification.")
@click.option("--max-age", default=None, type=int, help="Max frames a track survives without a detection.")
def main(
    input_path: str,
    output_dir: str,
    config_path: str,
    half_precision: bool | None,
    gpus: str | None,
    sam3_checkpoint: str | None,
    tiled: bool | None,
    tile_size: int | None,
    confidence: float | None,
    use_reid: bool | None,
    max_age: int | None,
) -> None:
    """Pitch Ball Tracker — track in-play balls using SAM3 + BoT-SORT ReID."""
    # Build CLI override dict (only non-None values)
    overrides: dict = {}
    if half_precision is not None:
        overrides.setdefault("model", {})["half_precision"] = half_precision
    if gpus is not None:
        overrides.setdefault("model", {})["gpus"] = [int(g) for g in gpus.split(",")]
    if sam3_checkpoint is not None:
        overrides.setdefault("model", {})["sam3_checkpoint"] = sam3_checkpoint
    if tiled is not None:
        overrides.setdefault("inference", {})["tiled"] = tiled
    if tile_size is not None:
        overrides.setdefault("inference", {})["tile_size"] = tile_size
    if confidence is not None:
        overrides.setdefault("inference", {})["confidence_threshold"] = confidence
    if use_reid is not None:
        overrides.setdefault("tracking", {})["use_reid"] = use_reid
    if max_age is not None:
        overrides.setdefault("tracking", {})["max_age"] = max_age

    cfg = load_config(config_path, overrides or None)

    logger.info(f"Input  : {input_path}")
    logger.info(f"Output : {output_dir}")
    logger.info(f"Config : {config_path}")

    pipeline = BallTrackerPipeline(cfg)
    pipeline.run(input_path, output_dir)
