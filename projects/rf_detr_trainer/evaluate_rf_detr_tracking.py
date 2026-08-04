"""Prepare Label Studio GT, cache RF-DETR detections, and compare football trackers.

Examples:
  python evaluate_rf_detr_tracking.py prepare_gt --source-root <label-studio-run> --output-dir runs/tracking_gt --yes
  python evaluate_rf_detr_tracking.py cache --source-root <label-studio-run> --config config/rf_detr_inference_medium_p2_sahi320_recheck640_hybrid.yaml --checkpoint <checkpoint> --output-dir runs/tracking_eval --yes
  python evaluate_rf_detr_tracking.py evaluate --source-root <label-studio-run> --config config/rf_detr_inference_medium_p2_sahi320_recheck640_hybrid.yaml --checkpoint <checkpoint> --output-dir runs/tracking_eval --all-segments --yes
  python evaluate_rf_detr_tracking.py sweep --source-root <label-studio-run> --config <preset> --checkpoint <checkpoint> --grid tracking_grids/hybrid_tracking_sweep.yaml --output-dir runs/tracking_eval --yes

Every action estimates its outputs and asks before writing unless --yes is passed.
Metrics are marked as proxies because the supplied Label Studio boxes have no reviewed IDs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm
import yaml

import inference_rf_detr_model as inference_runner
from rf_detr_hybrid_tracker import HybridTrackingConfig
import rf_detr_tracking_evaluation as tracking_eval
import rf_detr_video_tracking as video_tracking



def require_local_reid_weights(config: Any, algorithms: Sequence[str]) -> None:
    """Prevent BoxMOT evaluation from implicitly downloading a ReID checkpoint.

    BoxMOT 13 constructs its ReID backend for DeepOCSORT and BoT-SORT even when
    appearance matching is disabled, so evaluation must receive an existing local file.
    """
    requested = sorted(set(algorithms) & {"deepocsort", "botsort"})
    if not requested:
        return
    configured = getattr(config, "reid_weights", None)
    if configured and Path(str(configured)).expanduser().is_file():
        return
    raise ValueError(
        f"{','.join(requested)} evaluation requires existing local ReID weights; "
        "pass --reid-weights PATH (no automatic download is allowed)."
    )
ALGORITHMS = ("circle", "ocsort", "deepocsort", "botsort", "bytetrack", "hybrid")
DEFAULT_ALGORITHMS = ("circle", "ocsort", "bytetrack", "hybrid")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def selected_sequences(args: argparse.Namespace) -> Dict[str, List[tracking_eval.FrameGT]]:
    sequences = tracking_eval.load_label_studio_sequences(args.source_root)
    selected = set(sequences) if args.all_segments else set(args.segment_id or tracking_eval.SELECTED_SEGMENT_IDS)
    missing = sorted(selected - set(sequences))
    if missing:
        raise ValueError(f"selected segment IDs not found: {missing}")
    return {segment_id: sequences[segment_id] for segment_id in sorted(selected)}


def confirm(args: argparse.Namespace, description: str, files: int, frames: int) -> None:
    print(Fore.CYAN + Style.BRIGHT + f"Estimate: action={args.action}, frames={frames}, output_files?{files}")
    print(Fore.CYAN + f"Output directory: {args.output_dir}")
    if args.yes:
        return
    answer = input(f"{description} Continue? [y/N] ").strip().casefold()
    if answer not in {"y", "yes"}:
        raise SystemExit("Cancelled before writing outputs.")


def _pseudo_rows(sequences: Mapping[str, Sequence[tracking_eval.FrameGT]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for segment_id, frames in sequences.items():
        for frame, linked in zip(frames, tracking_eval.link_pseudo_gt_tracks(frames)):
            for row in linked:
                rows.append({**row, "image_name": frame.image_name, "timestamp": frame.timestamp})
    return rows


def _label_studio_video_tasks(sequences: Mapping[str, Sequence[tracking_eval.FrameGT]]) -> List[Dict[str, Any]]:
    tasks = []
    for segment_id, frames in sequences.items():
        linked = tracking_eval.link_pseudo_gt_tracks(frames)
        tracks: Dict[int, Dict[str, Any]] = {}
        for frame, frame_rows in zip(frames, linked):
            for row in frame_rows:
                gt_id = int(row["gt_track_id"])
                x, y, width, height = row["bbox"]
                track = tracks.setdefault(
                    gt_id,
                    {
                        "id": f"pseudo-{segment_id}-{gt_id}",
                        "from_name": "box",
                        "to_name": "video",
                        "type": "videorectangle",
                        "value": {"labels": [row["label"]], "sequence": []},
                    },
                )
                track["value"]["sequence"].append(
                    {
                        "frame": frame.frame_index + 1,
                        "time": frame.timestamp - frames[0].timestamp,
                        "x": 100.0 * x / frame.width,
                        "y": 100.0 * y / frame.height,
                        "width": 100.0 * width / frame.width,
                        "height": 100.0 * height / frame.height,
                        "rotation": 0,
                        "enabled": True,
                    }
                )
        tasks.append(
            {
                "data": {"video": f"{segment_id}.mp4", "segment_id": segment_id},
                "predictions": [{"model_version": "pseudo-link-review-required", "result": list(tracks.values())}],
            }
        )
    return tasks


def _write_segment_video(root: Path, segment_id: str, frames: Sequence[tracking_eval.FrameGT], output: Path) -> None:
    """Encode the ordered JPEG sequence as constant-frame-rate H.264 for Label Studio review."""
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        for parent in (root, *list(root.parents)[:6]):
            candidates = sorted(parent.glob("*/_internal/imageio_ffmpeg/binaries/ffmpeg*.exe"))
            if candidates:
                ffmpeg = str(candidates[0])
                break
    if ffmpeg is None:
        raise RuntimeError("H.264 export requires ffmpeg on PATH or bundled under the source application")

    output.parent.mkdir(parents=True, exist_ok=True)
    concat_path = output.with_suffix(".concat.txt")
    lines = []
    for frame in frames:
        path = (root / frame.image_name).resolve()
        escaped = str(path).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        lines.append("duration 0.0333333333333333")
    lines.append(lines[-2])
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-vf", "fps=30", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        concat_path.unlink(missing_ok=True)


def prepare_gt(args: argparse.Namespace) -> int:
    sequences = selected_sequences(args)
    frame_count = sum(len(frames) for frames in sequences.values())
    confirm(args, "Prepare pseudo-linked GT and Label Studio review tasks?", 4 + len(sequences), frame_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pseudo = _pseudo_rows(sequences)
    write_jsonl(args.output_dir / "gt_boxes_with_pseudo_ids.jsonl", pseudo)
    write_json(args.output_dir / "label_studio_video_tasks.json", _label_studio_video_tasks(sequences))
    write_json(
        args.output_dir / "segments.json",
        {
            "segments": [
                {"segment_id": key, "frames": len(value), "start": value[0].timestamp, "end": value[-1].timestamp}
                for key, value in sequences.items()
            ]
        },
    )
    write_json(
        args.output_dir / "review_status.json",
        {
            "human_identity_review_complete": False,
            "official_hota_allowed": False,
            "reason": "pseudo IDs are role-aware nearest-neighbor links; review them in Label Studio before official scoring",
        },
    )
    if args.write_videos:
        for segment_id, frames in sequences.items():
            _write_segment_video(args.source_root, segment_id, frames, args.output_dir / f"{segment_id}.mp4")
    return 0


def _cache_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    return args.output_dir / "detection_cache.jsonl", args.output_dir / "detection_cache_manifest.json"


def _cache_identity(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config is None or args.checkpoint is None:
        raise ValueError("cache/evaluate/sweep require --config and --checkpoint")
    return tracking_eval.detection_cache_identity(args.config, args.checkpoint, args.source_root)


def ensure_cache_valid(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cache_path, manifest_path = _cache_paths(args)
    identity = _cache_identity(args)
    if not cache_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("detection cache is missing; run the cache action first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != identity["fingerprint"]:
        raise RuntimeError("detection cache fingerprint is stale; rerun the cache action")
    return read_jsonl(cache_path), manifest


def cache_detections(args: argparse.Namespace) -> int:
    identity = _cache_identity(args)
    cache_path, manifest_path = _cache_paths(args)
    if manifest_path.exists() and cache_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("fingerprint") == identity["fingerprint"]:
            print(Fore.GREEN + "Detection cache is current; inference was skipped.")
            return 0
    confirm(args, "Run RF-DETR over the complete frame cache?", 8, identity["image_count"])
    inference_dir = args.output_dir / "cache_runs" / identity["fingerprint"][:16]
    predictions_path = inference_dir / "predictions.jsonl"
    if not predictions_path.exists():
        command = [
            sys.executable,
            str(Path(__file__).with_name("inference_rf_detr_model.py")),
            "--config", str(args.config),
            "--checkpoint", str(args.checkpoint),
            "--source", str(args.source_root),
            "--output-dir", str(inference_dir),
            "--no-track",
            "--yes",
        ]
        subprocess.run(command, cwd=Path(__file__).parent, check=True)
    rows = read_jsonl(predictions_path)
    normalized = []
    for row in rows:
        source = str(row.get("source") or row.get("file_name") or "")
        normalized.append({**row, "image_name": Path(source).name})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache_path, normalized)
    manifest = {
        **identity,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_count": len(normalized),
        "cache_file": cache_path.name,
    }
    write_json(manifest_path, manifest)
    return 0


def _detections_by_name(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["image_name"]), []).append(dict(row))
    return result


def _algorithm_config(base: Any, algorithm: str) -> Any:
    return replace(
        base,
        enabled=True,
        algorithm=algorithm,
        target_class_ids={0},
        deepocsort_embedding_off=True,
        deepocsort_cmc_off=False,
        botsort_with_reid=False,
    )


def _run_segment(
    algorithm: str,
    base_config: Any,
    frames: Sequence[tracking_eval.FrameGT],
    detections: Mapping[str, Sequence[Mapping[str, Any]]],
    source_root: Path,
    hybrid_override: Optional[HybridTrackingConfig] = None,
) -> List[Dict[str, Any]]:
    config = _algorithm_config(base_config, algorithm)
    if algorithm == "hybrid" and hybrid_override is not None:
        from rf_detr_hybrid_tracker import HybridFootballTracker

        tracker = HybridFootballTracker(hybrid_override, {0})
    else:
        tracker = inference_runner.create_tracker(config, tracker_device="cpu", frame_size=(frames[0].width, frames[0].height))
    output: List[Dict[str, Any]] = []
    needs_image = algorithm in {"deepocsort", "botsort"} or (
        algorithm == "hybrid" and bool(getattr(tracker.cfg, "cmc_enabled", False))
    )
    for frame_gt in frames:
        frame_detections = [dict(row) for row in detections.get(frame_gt.image_name, [])]
        image = cv2.imread(str(source_root / frame_gt.image_name)) if needs_image else None
        if algorithm == "hybrid":
            packets = tracker.step(frame_gt.frame_index, frame_gt.timestamp, image, frame_detections)
            for packet in packets:
                output.extend({**row, "frame_index": packet["frame_index"]} for row in packet["detections"])
        else:
            rows = tracker.update(frame_gt.frame_index, frame_detections, frame=image)
            output.extend({**row, "frame_index": frame_gt.frame_index} for row in rows)
    if algorithm == "hybrid":
        for packet in tracker.flush():
            output.extend({**row, "frame_index": packet["frame_index"]} for row in packet["detections"])
    return output


def _aggregate_metrics(segment_metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    matches = sum(int(row["matches"]) for row in segment_metrics)
    gt = sum(int(row["gt_boxes"]) for row in segment_metrics)
    predicted = sum(int(row["predicted_tracked_boxes"]) for row in segment_metrics)
    switches = sum(int(row["id_switches"]) for row in segment_metrics)
    merges = sum(int(row["false_merges"]) for row in segment_metrics)
    det_a = matches / max(1, gt + predicted - matches)
    ass_a = matches / max(1, matches + switches + merges)
    return {
        "official_hota": False,
        "official_metrics_reason": "source annotations have no reviewed track identities",
        "matches": matches,
        "gt_boxes": gt,
        "predicted_tracked_boxes": predicted,
        "id_switches": switches,
        "false_merges": merges,
        "hota_proxy": (det_a * ass_a) ** 0.5,
        "idf1_proxy": 2.0 * matches / max(1, gt + predicted),
    }


def evaluate_algorithm(
    algorithm: str,
    base_config: Any,
    sequences: Mapping[str, Sequence[tracking_eval.FrameGT]],
    detections: Mapping[str, Sequence[Mapping[str, Any]]],
    source_root: Path,
    hybrid_override: Optional[HybridTrackingConfig] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    started = time.perf_counter()
    metrics = []
    all_rows = []
    for segment_id, frames in tqdm(sequences.items(), desc=f"Evaluate {algorithm}", unit="segment"):
        rows = _run_segment(algorithm, base_config, frames, detections, source_root, hybrid_override)
        all_rows.extend({**row, "segment_id": segment_id} for row in rows)
        pseudo = tracking_eval.link_pseudo_gt_tracks(frames)
        matches, gt_count, prediction_count = tracking_eval.match_tracking_rows(pseudo, rows)
        segment_metric = tracking_eval.association_proxy_metrics(matches, gt_count, prediction_count)
        metrics.append({"segment_id": segment_id, **segment_metric})
    aggregate = _aggregate_metrics(metrics)
    return {"algorithm": algorithm, "elapsed_seconds": time.perf_counter() - started, "aggregate": aggregate, "segments": metrics}, all_rows


def _acceptance(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    hybrid = metrics.get("hybrid", {}).get("aggregate")
    baselines = [row["aggregate"] for name, row in metrics.items() if name != "hybrid"]
    if not hybrid or not baselines:
        return {"passed": False, "reason": "hybrid and at least one baseline are required"}
    best = min(
        baselines,
        key=lambda row: (row["false_merges"], row["id_switches"], -row["hota_proxy"], -row["idf1_proxy"]),
    )
    def reduced(hybrid_value: int, baseline_value: int) -> bool:
        return hybrid_value == 0 if baseline_value == 0 else hybrid_value <= baseline_value * 0.5
    checks = {
        "false_merges_reduced_50_percent": reduced(hybrid["false_merges"], best["false_merges"]),
        "id_switches_reduced_50_percent": reduced(hybrid["id_switches"], best["id_switches"]),
        "hota_proxy_not_lower": hybrid["hota_proxy"] >= best["hota_proxy"],
        "idf1_proxy_not_lower": hybrid["idf1_proxy"] >= best["idf1_proxy"],
    }
    return {"passed": all(checks.values()), "checks": checks, "best_baseline": best, "official": False}


def evaluate(args: argparse.Namespace) -> int:
    cache_rows, manifest = ensure_cache_valid(args)
    sequences = selected_sequences(args)
    algorithms = tuple(args.algorithms.split(","))
    unknown = sorted(set(algorithms) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"unknown algorithms: {unknown}")
    frame_count = sum(len(frames) for frames in sequences.values())
    confirm(args, "Evaluate cached detections with the selected trackers?", len(algorithms) * 2 + 2, frame_count)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = video_tracking.parse_tracking_config(config, inference_runner.build_categories(config))
    if getattr(args, "reid_weights", None) is not None:
        base = replace(base, reid_weights=str(args.reid_weights))
    require_local_reid_weights(base, algorithms)
    detections = _detections_by_name(cache_rows)
    reports = {}
    for algorithm in algorithms:
        report, rows = evaluate_algorithm(algorithm, base, sequences, detections, args.source_root)
        reports[algorithm] = report
        write_jsonl(args.output_dir / "tracking_rows" / f"{algorithm}.jsonl", rows)
    result = {"cache_fingerprint": manifest["fingerprint"], "algorithms": reports, "acceptance": _acceptance(reports)}
    write_json(args.output_dir / "tracking_metrics.json", result)
    return 0


def _set_nested(raw: Dict[str, Any], dotted: str, value: Any) -> None:
    cursor = raw
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def sweep(args: argparse.Namespace) -> int:
    if args.grid is None:
        raise ValueError("sweep requires --grid")
    cache_rows, manifest = ensure_cache_valid(args)
    sequences = selected_sequences(args)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = video_tracking.parse_tracking_config(config, inference_runner.build_categories(config))
    grid = yaml.safe_load(args.grid.read_text(encoding="utf-8")) or {}
    keys = sorted(grid)
    combinations = list(itertools.product(*(grid[key] for key in keys)))
    confirm(args, "Run the hybrid parameter grid on cached detections?", len(combinations) + 2, sum(len(v) for v in sequences.values()))
    detections = _detections_by_name(cache_rows)
    results = []
    base_options = base.hybrid_options
    for values in tqdm(combinations, desc="Hybrid sweep", unit="config"):
        options = json.loads(json.dumps(base_options))
        for key, value in zip(keys, values):
            _set_nested(options, key, value)
        hybrid = HybridTrackingConfig.from_mapping(options)
        report, _rows = evaluate_algorithm("hybrid", base, sequences, detections, args.source_root, hybrid)
        results.append({"parameters": dict(zip(keys, values)), **report})
    results.sort(key=lambda row: (
        row["aggregate"]["false_merges"], row["aggregate"]["id_switches"],
        -row["aggregate"]["hota_proxy"], -row["aggregate"]["idf1_proxy"], row["elapsed_seconds"],
    ))
    write_json(args.output_dir / "hybrid_sweep.json", {"cache_fingerprint": manifest["fingerprint"], "results": results, "best": results[0] if results else None})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare_gt", "cache", "evaluate", "sweep"))
    parser.add_argument("--source-root", type=Path, required=True, help="Label Studio run containing tasks.json and JPEG frames.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Evaluation artifact directory.")
    parser.add_argument("--config", type=Path, help="Inference YAML used to fingerprint/cache detections.")
    parser.add_argument("--checkpoint", type=Path, help="RF-DETR checkpoint used to fingerprint/cache detections.")
    parser.add_argument("--segment-id", action="append", help="Segment to process; repeatable. Defaults to the reviewed three-segment subset.")
    parser.add_argument("--reid-weights", type=Path, help="Existing local ReID checkpoint required by DeepOCSORT/BoT-SORT evaluation.")
    parser.add_argument("--all-segments", action="store_true", help="Process all 15 segments and reset tracker identity per segment.")
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS), help="Comma-separated trackers; DeepOCSORT/BoT-SORT additionally require --reid-weights.")
    parser.add_argument("--grid", type=Path, help="YAML mapping of hybrid dotted parameter names to value lists.")
    parser.add_argument("--write-videos", action="store_true", help="Also reconstruct per-segment review MP4 files.")
    parser.add_argument("--yes", action="store_true", help="Skip the output confirmation prompt.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    colorama_init(autoreset=True)
    args = build_parser().parse_args(argv)
    args.source_root = args.source_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.config is not None:
        args.config = args.config.expanduser().resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.reid_weights is not None:
        args.reid_weights = args.reid_weights.expanduser().resolve()
    if args.grid is not None:
        args.grid = args.grid.expanduser().resolve()
    handlers = {"prepare_gt": prepare_gt, "cache": cache_detections, "evaluate": evaluate, "sweep": sweep}
    return handlers[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
