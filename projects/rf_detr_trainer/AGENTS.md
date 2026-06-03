# RF-DETR Trainer Agent Rules

Apply these rules when changing this project.

1. Before implementation, inspect the request for contradictions, missing details, or safer alternatives. If a material tradeoff remains, ask for confirmation before editing.
2. Keep execution configs in `config/` grouped by category: `runtime`, `model`, `dataset`, `output`, task section, and `evaluation`. Execution configs must include all supported parameters with `#` comments explaining purpose and valid options. Dataset YAML files should remain in their framework-native format.
3. Any code path that writes outputs must save the config that created those outputs inside the output folder. Use UTF-8 and preserve simplified/traditional Chinese text.
4. Before writing images, videos, documents, metrics, caches, or checkpoints, print an output estimate with file count and disk usage, then require developer confirmation unless `--yes` or config confirmation bypass is enabled.
5. If a path for saved photos or other outputs does not exist, create it with `mkdir(parents=True, exist_ok=True)`.
6. Python entrypoints must import `colorama` and call `colorama.init(autoreset=True)`. Long-running loops must use progress bars.
7. CUDA usage must be configurable as CPU-only, auto GPU, or specific GPU IDs. Code must work on Linux and Windows paths.
8. When adding packages or model/dataset downloads, first check the public IP region. Use China-friendly mirrors for China/HK/MO/TW regions and official sources elsewhere.
9. Prefer existing project helpers and vectorized data structures for speed. Avoid unrelated refactors and preserve user-created files.
10. For SAHI sliced RF-DETR tests, prevent duplicate boxes from small slices with same-class `GREEDYNMM` + `IOS` postprocessing by default. Do not use class-agnostic merging unless the config explicitly asks for it.
11. Standalone and scheduled RF-DETR test diagnostics must keep visual sample candidate filters separate from rendered-class filters. `visual_samples.render_class_ids/render_class_names` controls rendered classes for sampled visuals, while `error_cases.render_class_ids/render_class_names` controls rendered classes for error-case images.
12. Error-case diagnostics must default to football when no target class is configured, and must render both ground-truth and prediction boxes with prediction scores for missed, misclassified, and false-positive target-class cases.
13. Test metrics must include COCO-style overall metrics plus per-class mAP50, mAP50-95, precision, recall, F1, and per-class small/medium/large detection metrics.
14. SAHI target-class recheck must keep the first-stage box geometry, verify the object with a centered B*B second pass, fuse first/second confidence scores by config weights, and keep the box only when the fused score passes the configured threshold.
15. Train, test, and inference entrypoints must remain separate Python files. Shared behavior should live behind shared helper modules rather than making standalone entrypoints import each other.
16. Inference outputs must support RF-DETR image/video file, folder, and HTTP(S) media sources. Mixed image/video folders are allowed, and the same class ID must use the same bounding-box color across all rendered images and video frames in a run.
17. Train, test, and inference configs must support running all data or only the first N records. Use `all`/`null` for full data and positive integers for first-N limits. Keep train limits in `dataset`, standalone/scheduled test limits in the task test section, and inference source limits in `inference`.
18. Inference video configs must support `inference.video.start_time`/`end_time` for segment inference and `inference.video.max_seconds` with `all`/`null` for full selected ranges or a positive seconds value for partial video inference. Time values must accept seconds, `MM:SS`, and `HH:MM:SS`.
19. Train, test, and inference entrypoints must include a rough HH:MM:SS runtime estimate in the pre-run estimate, print elapsed HH:MM:SS when the process exits, and write `run_timing.json` inside the output folder when outputs were created.
20. Keep large datasets, caches, checkpoints, generated media, and run outputs out of git via `.gitignore`.
