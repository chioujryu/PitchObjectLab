---
name: rf-detr-trainer
description: Work on this repo's RF-DETR training/testing workflow, including config-first YAMLs, output safety checks, UV/PyTorch setup, dataset cache handling, and football-focused test diagnostics.
---

# RF-DETR Trainer

Use this skill when editing `projects/rf_detr_trainer` or related object-detection evaluator code.

## Workflow

1. Inspect the request for contradictions, missing details, and risky assumptions before editing. If a high-impact choice cannot be resolved from the repo, ask for confirmation.
2. Read current entrypoints/configs first: `train_rf_detr_model.py`, `test_rf_detr_model.py`, `config/*.yaml`, `README.md`, and `AGENTS.md`.
3. Keep execution config parameters grouped as `runtime`, `model`, `dataset`, `output`, task-specific settings, and `evaluation`. Every execution-config key needs a nearby `#` comment with purpose and valid options.
4. Preserve framework-native dataset YAMLs. Do not add trainer-only keys to Ultralytics data YAMLs.
5. Any output-producing code must:
   - print file-count and disk-usage estimates before writing;
   - ask for confirmation unless `--yes` or config bypass is enabled;
   - create missing output directories;
   - save source/resolved config metadata inside the output folder;
   - write JSON/YAML with UTF-8 Chinese text preserved.
6. Python entrypoints should import `colorama`, call `colorama.init(autoreset=True)`, include a triple-quoted usage docstring, and show progress bars for long loops.
7. CUDA must be configurable as `auto`, `cpu`, `cuda`, `cuda:N`, or specific IDs. Keep Windows and Linux path handling.
8. For package/model/dataset downloads, check public IP region first. Use China/HK/MO/TW mirrors when detected, otherwise official sources.

## Test Diagnostics

- `test_rf_detr_model.py` supports `full_image`, `sahi`, and `class_crop`.
- Use `test.visual_samples.max_images` to limit saved prediction images.
- Football diagnostics default to `test.error_cases.target_class_names: [football]`.
- Error-case images should include GT boxes, predicted boxes, prediction class names, and scores.
- Diagnose three cases: target missed, target misclassified, and target false positive.
- For RF-DETR SAHI tests, small slices can create duplicate boxes for the same object. Keep defaults at `postprocess_type: GREEDYNMM`, `postprocess_match_metric: IOS`, and `postprocess_class_agnostic: false` unless the user explicitly wants another tradeoff.

## Validation

Run focused checks after changes:

```bash
uv run python -m unittest discover -s tests
uv run python test_rf_detr_model.py --config config/rf_detr_test.yaml --dry-run --yes
```
