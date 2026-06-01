# Object Detection Dataset Evaluator

This subproject evaluates YOLO-format or COCO-format object-detection datasets with COCO-style metrics. It can run either:

- SAHI sliced inference for large-image or small-object evaluation.
- Direct Ultralytics full-image inference when SAHI is disabled.

The evaluator exports metrics, plots, prediction JSON, optional annotated visuals, and raw evaluated dataset cases under `outputs/<run>/datasets`.

## Files

```text
projects/object_detection_dataset_evaluator/
|-- README.md
|-- .gitignore
|-- pyproject.toml
|-- object_detection_dataset_evaluator.py
`-- config/
    `-- object_detection_dataset_evaluate.yaml
```

## Environment

From this project folder:

```bash
cd projects/object_detection_dataset_evaluator
uv sync
```

If your machine blocks the default uv cache folder, use a local cache:

```bash
# PowerShell
$env:UV_CACHE_DIR="$PWD/.uv-cache"; uv sync

# Linux/macOS
UV_CACHE_DIR="$PWD/.uv-cache" uv sync
```

Run:

```bash
uv run python object_detection_dataset_evaluator.py --config config/object_detection_dataset_evaluate.yaml
```

After checking the printed resource estimate, skip the prompt with:

```bash
uv run python object_detection_dataset_evaluator.py --config config/object_detection_dataset_evaluate.yaml --yes
```

## Quick Examples

Evaluate a YOLO dataset with SAHI:

```bash
uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py \
  --config projects/object_detection_dataset_evaluator/config/object_detection_dataset_evaluate.yaml \
  --use-sahi \
  --data-yaml ultralytics/cfg/datasets/coco8.yaml \
  --split val \
  --model-path yolo26n.pt \
  --device cuda:0 \
  --slice-height 640 \
  --slice-width 640 \
  --yes
```

Evaluate without SAHI using direct Ultralytics inference:

```bash
uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py \
  --config projects/object_detection_dataset_evaluator/config/object_detection_dataset_evaluate.yaml \
  --no-sahi \
  --data-yaml ultralytics/cfg/datasets/coco8.yaml \
  --split val \
  --model-path yolo26n.pt \
  --device cpu \
  --yes
```

Evaluate a COCO dataset:

```bash
uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py \
  --dataset-format coco \
  --coco-json /datasets/coco/annotations/instances_val.json \
  --image-dir /datasets/coco/val2017 \
  --model-path runs/detect/train/weights/best.pt \
  --device cpu \
  --yes
```

Run a tiny demo output:

```bash
uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py --demo --yes
```

Dry-run only, with no inference:

```bash
uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py --dry-run
```

## Config

Edit:

```text
projects/object_detection_dataset_evaluator/config/object_detection_dataset_evaluate.yaml
```

Important sections:

```yaml
inference:
  mode: sahi          # full_image, sahi, class_crop
  use_sahi: true

test_mode:
  mode: sahi

crop:
  class_names: []     # empty means all predicted classes can define the crop
  source_conf: 0.25
  padding_pixels: 0
  padding_ratio: 0.05
  fallback: full_image

dataset:
  format: yolo
  data_yaml: ultralytics/cfg/datasets/coco8.yaml
  split: val

model:
  type: ultralytics
  path: yolo26n.pt
  device: cpu

sahi:
  enabled: true
  sliced_prediction: true
  slice_height: 640
  slice_width: 640
  overlap_height_ratio: 0.2
  overlap_width_ratio: 0.2

output:
  save_dataset_cases: true
  save_model_input_batches: true
  max_dataset_case_images: 5
  max_slices_per_image: 12
  save_visuals: false
  max_visuals: 25
```

Every parameter in the YAML has a `#` comment describing what it means and common replacement options.

All modes can write `test_batch*_labels.jpg`, `test_batch*_pred.jpg`, and
`model_inputs_manifest.*`. In `class_crop`, the first model pass chooses the
padded crop window from predicted classes, the second pass tests that crop, and
predictions are projected back to original-image coordinates.

## Dataset Cases And Visuals

`output.save_dataset_cases: true` writes raw evaluation examples under:

```text
outputs/<run>/datasets/
```

When `inference.use_sahi: true`, cases are slice crops:

```text
datasets/sliced_cases/
datasets/dataset_cases_manifest.csv
datasets/dataset_cases_metadata.json
```

When `inference.use_sahi: false`, cases are original full images:

```text
datasets/original_cases/
datasets/dataset_cases_manifest.csv
datasets/dataset_cases_metadata.json
```

The original `output.save_visuals` behavior is still separate. It writes annotated GT/prediction images under `visuals/`, plus `visuals_metadata.json` and `visuals_manifest.csv`.

## Metrics

The evaluator exports:

- `mAP50`
- `mAP50-95`
- `mAP75`
- `Precision`
- `Recall`
- `F1`
- `TP`, `FP`, `FN`
- AP small/medium/large
- AR at configurable max detections
- per-class AP50-95/AP50/AP75/AR
- per-class Precision/Recall/F1
- per-image TP/FP/FN
- confidence-threshold sweep
- PR curve, P curve, R curve, F1 curve
- confusion matrix and normalized confusion matrix
- inference speed and prediction count per image

For best confidence curves, set `model.confidence_threshold` low, such as `0.001` or `0.01`.

## Output

Default output:

```text
projects/object_detection_dataset_evaluator/outputs/<run_name>/
```

`output.dir`, `output.name`, `output.dataset_cases_subdir`, and `output.visual_output_subdir` can be plain strings or templates. Built-in placeholders include `{timestamp}`, `{date}`, `{engine}`, `{model}`, and `{split}`; config dot-paths such as `{dataset.split}`, `{model.image_size}`, or `{sahi.slice_width}` are also supported.

Typical files:

```text
config/resolved_config.yaml
config/source_config.yaml
config/run_metadata.json
datasets/
ground_truth_coco.json
predictions_coco.json
predictions_coco.metadata.json
metrics_summary.json
metrics_summary.csv
per_class_metrics.csv
per_image_metrics.csv
threshold_sweep.csv
inference_stats.csv
confusion_matrix.csv
PR_curve.png
P_curve.png
R_curve.png
F1_curve.png
confusion_matrix.png
confusion_matrix_normalized.png
visuals/
visuals_manifest.csv
output_manifest.json
```

All metric tables include `run_id` and `config_hash`. JSON sidecars include metadata/config information. Plot PNGs include the config hash in the title and PNG metadata. Dataset cases and visual images have manifests plus metadata JSON files containing the full resolved config.

## Output Confirmation

Before inference starts, the script prints:

- output directory
- inference engine
- images to evaluate
- ground-truth annotations
- estimated output file count
- estimated disk usage
- dataset case image count
- visual image count
- plot count

Unless `--yes` is used, type `YES` before inference and output writing continue.

## GPU / CPU

Single device:

```yaml
model:
  device: cpu
```

or:

```yaml
model:
  device: cuda:0
```

Multiple devices:

```yaml
model:
  devices: [cuda:0, cuda:1]
```

On Linux, PyTorch device IDs are relative to `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=6 uv run python object_detection_dataset_evaluator.py --config config/object_detection_dataset_evaluate.yaml
```

In that command, use `model.device: cuda:0` because physical GPU 6 is exposed as visible device `cuda:0`.

## Demo Mode

Demo mode reduces output volume:

```yaml
demo:
  enabled: true
  output_dir: projects/object_detection_dataset_evaluator/demo_outputs
  max_images: 8
  max_visuals: 4
  max_dataset_case_images: 3
  max_slices_per_image: 4
  save_visuals: true
  save_dataset_cases: true
```

CLI:

```bash
uv run python projects/object_detection_dataset_evaluator/object_detection_dataset_evaluator.py --demo --yes
```
