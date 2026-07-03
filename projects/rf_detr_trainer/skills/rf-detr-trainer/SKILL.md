---
name: rf-detr-trainer
description: Work on this repo's RF-DETR training/testing workflow, including config-first YAMLs, output safety checks, UV/PyTorch setup, dataset cache handling, and football-focused test diagnostics.
---

# RF-DETR Trainer

Use this skill when editing `projects/rf_detr_trainer` or related object-detection evaluator code.

## Workflow

1. Inspect the request for contradictions, missing details, risky assumptions, and safer alternatives before editing. Then do broad web/source research when network access is available, summarize findings and the proposed change, and get developer confirmation before implementation; if the task is strictly local/offline or network is forbidden, state that clearly before continuing. If a high-impact choice cannot be resolved from the repo, ask for confirmation.
2. When creating a new project, major tool, or standalone module, choose a clear project name that fits the purpose, domain, and expected users. Do not rename this project or existing public entrypoints without explicit confirmation.
3. Read current entrypoints/configs first: `train_rf_detr_model.py`, `test_rf_detr_model.py`, `inference_rf_detr_model.py`, `config/*.yaml`, `README.md`, and `AGENTS.md`.
4. Keep execution config parameters grouped as `runtime`, `model`, `dataset`, `output`, task-specific settings, and `evaluation`. Every execution-config key needs a nearby `#` comment with purpose and valid options. Output folders must be configurable by parameters, placeholders, or plain string templates.
5. Preserve framework-native dataset YAMLs. Do not add trainer-only keys to Ultralytics data YAMLs.
6. Any output-producing code must:
   - print file-count and disk-usage estimates before writing;
   - include a rough HH:MM:SS runtime estimate before writing;
   - ask for confirmation unless `--yes` or config bypass is enabled;
   - provide a no-confirm execution path such as `--yes`;
   - create missing output directories;
   - save source/resolved config metadata inside the output folder;
   - print elapsed HH:MM:SS at process exit and write `run_timing.json` when outputs were created;
   - write JSON/YAML with UTF-8 Chinese text preserved.
7. Python entrypoints should import `colorama`, call `colorama.init(autoreset=True)`, include a triple-quoted `"""..."""` usage docstring, and show progress bars for long loops.
8. CUDA must be configurable as `auto`, `cpu`, `cuda`, `cuda:N`, or specific IDs. Keep Windows and Linux path handling.
9. For package/model/dataset downloads, check the observed public IP region first. Use China/HK/MO/TW mirrors when detected, otherwise official sources, and allow mirror/source overrides so the workflow works with or without VPN.
10. Keep DataLoader/worker-style parallelism uniformly set to `2` in configs and defaults unless the developer explicitly asks for a different value.
11. Maintain a project-local `uv` environment. If `.venv` is missing, create it through `uv`; if it exists, add new dependencies with `uv add` and keep `pyproject.toml`/`uv.lock` in sync so future setup only needs `uv sync`.
12. GPU PyTorch setup must auto-detect hardware and choose the appropriate PyTorch build/index. Reuse `scripts/setup_pytorch_uv.py` when possible and follow the region-aware download rule.
13. Update `README.md` whenever entrypoint usage, config options, setup steps, or output behavior changes.
14. Every runnable script must produce useful console logs. For output-producing runs, write logs or a log summary inside the output folder when practical.

## Test Diagnostics

- `test_rf_detr_model.py` supports `full_image`, `sahi`, and `class_crop`.
- Use `test.max_images` for the number of test images evaluated. `all`/`null` means the full split; a positive integer means first N images.
- Use real RF-DETR list-input batching for test speedups. `test.batch_size` controls full-image/class-crop image batches, and `test.sahi.batch_size` controls SAHI slice/recheck batches. Balanced defaults are 4; downshift on CUDA OOM before failing.
- Use `test.visual_samples.max_images` to limit saved prediction images.
- Use `test.visual_samples.class_ids/class_names` only to choose visual candidate images; use `test.visual_samples.render_class_ids/render_class_names` to limit which GT/prediction classes are drawn.
- Football diagnostics default to `test.error_cases.target_class_names: [football]`.
- Use `test.error_cases.target_class_ids/target_class_names` to choose missed/misclassified/false-positive target classes; use `test.error_cases.render_class_ids/render_class_names` only to limit rendered GT/prediction classes. Empty render lists mean draw all classes.
- Error-case images should include GT boxes, predicted boxes, prediction class names, and scores.
- Diagnose three cases: target missed, target misclassified, and target false positive.
- For RF-DETR SAHI tests, small slices can create duplicate boxes for the same object. Keep defaults at `postprocess_type: GREEDYNMM`, `postprocess_match_metric: IOS`, and `postprocess_class_agnostic: false` unless the user explicitly wants another tradeoff.
- If `sahi.recheck.enabled` is true, verify only the configured target classes, defaulting to football when no target is set. Keep the first-stage box geometry, run a centered second-pass crop, and fuse confidence scores by the configured weights.
- Test metrics should include per-class mAP50/mAP50-95/P/R/F1 and per-class small/medium/large results in CSV/JSON outputs.
- Visual and error-case rendered prediction boxes should default to the model confidence threshold unless the config explicitly overrides the draw/match threshold.

## Inference

- `inference_rf_detr_model.py` is RF-DETR only and supports image files, video files, folders, mixed image/video folders, and HTTP(S) media URLs.
- Use `inference.max_sources`, `inference.max_images`, and `inference.max_videos` to limit first-N inference sources. `all`/`null` means no limit.
- Use `inference.batch_size` for batched image-source inference and `inference.video.batch_size` for video detection-frame batches. Default image/video detection batch is 8, and `inference.video.batch_size: null` inherits `inference.batch_size`.
- Use `inference.video.start_time`/`end_time` for video segment inference and `inference.video.max_seconds` to cap selected duration. Time values accept seconds, `MM:SS`, and `HH:MM:SS`; `all`/`null` end/max means no cap.
- Inference should save rendered media, prediction records, class color metadata, and config snapshots inside the output folder.
- Class colors must be stable within a run: the same class ID uses the same bounding-box color across every image and every video frame.
- Video tracking is selected by `inference.tracking.algorithm`: `circle` (built-in stdlib tracker in `rf_detr_video_tracking.py`, the default) or `ocsort`/`deepocsort`/`botsort`/`bytetrack` from `boxmot` (pinned `boxmot==13.0.0`). The adapter lives in `rf_detr_boxmot_tracker.py` and imports boxmot lazily; `algorithm: circle` must stay byte-compatible.
- Each boxmot tracker's params live in its own nested sub-block (`inference.tracking.ocsort`/`deepocsort`/`botsort`/`bytetrack`); shared ReID/CMC and circle/rendering keys stay at the `tracking` top level. The parser maps the nested blocks onto flat `TrackingConfig` fields, so the adapter is unaffected.
- The tracked class is configurable for all algorithms via `inference.tracking.target_class_ids`/`target_class_names` (default football). `predictions.jsonl` track fields and `tracking_summary.json` are identical across algorithms.
- For `deepocsort`/`botsort`, prefer a local `inference.tracking.reid_weights` path; the default ReID weights auto-download from Google Drive (unreliable in CN/HK/MO/TW, rule 8). `ocsort`/`bytetrack` need no weights, and `botsort.with_reid: false` disables BoT-SORT appearance. ReID device follows `model.device`; `reid_half` is GPU-only.

## Dataset Limits

- Train-time limits belong in `dataset.max_images`, `dataset.max_train_images`, `dataset.max_val_images`, and `dataset.max_test_images`.
- For non-RF-DETR sources, apply train limits after deterministic split assignment and before cache materialization.
- For existing RF-DETR datasets, first-N limits should build a limited RF-DETR cache rather than mutating the source dataset.

## Runtime Timing

- Runtime estimates should be fast and should not load models or run calibration inference before confirmation.
- Use `runtime.time_estimate` settings and latest sibling `run_timing.json` when available; otherwise use default-rate estimates.
- Format estimated and elapsed durations as `HH:MM:SS`.

## Validation

Run focused checks after changes:

```bash
uv run python -m unittest discover -s tests
uv run python test_rf_detr_model.py --config config/rf_detr_test.yaml --dry-run --yes
uv run python inference_rf_detr_model.py --config config/rf_detr_inference.yaml --dry-run --yes
```
