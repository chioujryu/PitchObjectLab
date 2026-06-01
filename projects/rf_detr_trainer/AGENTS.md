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
13. Keep large datasets, caches, checkpoints, generated media, and run outputs out of git via `.gitignore`.
