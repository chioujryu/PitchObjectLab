# RF-DETR Trainer

Config-first RF-DETR training project with an Ultralytics-style workflow:

1. Custom output directory using `output.output_dir` or `output.root` + `output.name`.
2. Validation every N epochs through RF-DETR `eval_interval`.
3. Scheduled test-set evaluation every N epochs or minutes.
4. Final test-set evaluation after training.
5. Overall and per-class test metrics saved as JSON and CSV.
6. Config snapshots inside every training/test output folder.
7. Augmented train/validation dataset sample grids saved before training.
8. A file/resource estimate plus confirmation before large outputs are written.
9. Auto-detection and cache conversion for RF-DETR/Roboflow, Ultralytics YOLO,
   COCO JSON, Pascal VOC, DOTA, and LabelMe JSON datasets.

## Environment

This subproject has its own `pyproject.toml`, `uv.lock`, and `.venv`.

```bash
cd projects/rf_detr_trainer
uv sync
```

Auto-select CPU/GPU PyTorch wheels by checking CUDA and the current public IP
region first:

```bash
uv run python scripts/setup_pytorch_uv.py --dry-run
uv run python scripts/setup_pytorch_uv.py --yes
```

The setup script writes the selected PyTorch index into `pyproject.toml`; after
that, `uv sync` is enough for future environment setup. China/HK/MO/TW IP
regions use mirror indexes by default, while other regions use official PyTorch
indexes.

On Windows and Linux, `uv sync` installs the CUDA 12.8 PyTorch wheels from
the official PyTorch index configured in `pyproject.toml`. The CUDA build can
still run on CPU when no CUDA GPU is available; use `train.device: cpu` or
`--device cpu` to force CPU, and use `auto`, `cuda`, `0`, or `cuda:0` to use
GPU when available.

Check the active PyTorch build:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

Then run commands with:

```bash
uv run python train_rf_detr_model.py --help
```

## Quick Start

Edit:

```text
projects/rf_detr_trainer/config/rf_detr_train.yaml
```

Run a dry run first:

```bash
cd projects/rf_detr_trainer
uv run python train_rf_detr_model.py --dry-run --yes
```

Start training:

```bash
uv run python train_rf_detr_model.py
```

Skip the confirmation prompt after you accept the estimate:

```bash
uv run python train_rf_detr_model.py --yes
```

## Common CLI Examples

Custom dataset and output folder:

```bash
uv run python train_rf_detr_model.py \
  --dataset-dir /data/football_dataset \
  --model-size medium \
  --device 0 \
  --epochs 300 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --eval-interval 5 \
  --test-interval-epochs 30 \
  --output-dir /runs/rf_detr/football_medium \
  --yes
```

Windows path example:

```powershell
uv run python train_rf_detr_model.py `
  --dataset-dir D:\datasets\football_dataset `
  --output-dir D:\runs\rf_detr\football_medium `
  --device 0 `
  --yes
```

CPU-only demo:

```bash
uv run python train_rf_detr_model.py --device cpu --demo --yes
```

Auto-detect and cache a source dataset:

```bash
uv run python train_rf_detr_model.py \
  --dataset-source-format auto \
  --dataset-dir D:/datasets/football_dataset \
  --dataset-link-mode auto \
  --yes
```

## Config Highlights

`model.size` options include:

```text
Detection: base, nano, small, medium, large
Segmentation: seg-preview, seg-nano, seg-small, seg-medium, seg-large, seg-xlarge, seg-2xlarge
```

`train.device` options:

```text
auto, cpu, cuda, 0, 1, 0,1, cuda:0, mps, -1
```

Validation interval:

```yaml
train:
  eval_interval: 5
```

Scheduled test interval:

```yaml
periodic_test:
  enabled: true
  split: test
  test_mode:
    mode: full_image   # full_image, sahi, class_crop
  test_interval_epochs: 30
run_final_test: true
classwise: true
```

`sahi` and `class_crop` use the shared image-level evaluator. In `class_crop`,
the same RF-DETR model first predicts the configured `periodic_test.crop`
classes, the padded union crop is tested, and crop predictions are projected
back to original-image coordinates. Missing crop-class predictions fall back to
full-image test per image.

Standalone test:

```bash
uv run python test_rf_detr_model.py --config config/rf_detr_test.yaml --yes

uv run python test_rf_detr_model.py \
  --config config/rf_detr_test.yaml \
  --model-size seg-large \
  --checkpoint runs/rf_detr/my_seg_run/checkpoint_best_total.pth \
  --yes
```

`config/rf_detr_test.yaml` is test-only. Use `test.split`,
`test_mode.mode`, `crop`, `sahi`, `periodic_test`, and `evaluation` there; the
script builds any RF-DETR runtime adapter settings internally. Full-image
segmentation tests use RF-DETR's mask-aware evaluation path; `sahi` and
`class_crop` tests evaluate boxes.

Prediction visuals and football diagnostics are controlled in the test config:

```yaml
test:
  visual_samples:
    enabled: true
    max_images: 25
  error_cases:
    enabled: true
    target_class_names: [football]
    max_images: 25
```

The saved diagnostic images include ground-truth boxes, predicted boxes, class
labels, and prediction scores. Error cases cover missed football, football
misclassified as another class, and false-positive football predictions.

For SAHI tests, small slice sizes can produce multiple boxes on the same object.
The RF-DETR direct SAHI path uses same-class `GREEDYNMM` with `IOS` by default:

```yaml
test:
  sahi:
    postprocess_type: GREEDYNMM
    postprocess_match_metric: IOS
    postprocess_match_threshold: 0.5
    postprocess_class_agnostic: false
```

Every standalone test prints a file/disk estimate before creating output folders
or cache files. Use `--dry-run --yes` to inspect the estimate without running
inference, and use `--yes` only after accepting the output size.

Dataset sample grids:

```yaml
train:
  save_dataset_grids: true
```

When enabled, the trainer saves RF-DETR dataloader samples after resize,
normalization reversal, and augmentation to `<output_dir>/dataset_grids/`.
Use `--no-save-dataset-grids` to skip this for a run.

Custom output address:

```yaml
output:
  output_dir: "/runs/rf_detr/{dataset_name}/{model_size}_e{epochs}_b{batch_size}_{timestamp}"
```

Or use Ultralytics-style root/name:

```yaml
output:
  output_dir: ""
  root: runs/rf_detr/{dataset_name}
  name: "rfdetr_{model_size}_e{epochs}_val{eval_interval}_{timestamp}"
```

Output folder fields can be pure strings or placeholder templates. Supported placeholders:

```text
{timestamp}, {date}, {time}, {model_size}, {resolution}, {pretrain}, {num_classes},
{dataset_name}, {dataset_file}, {source_format}, {device}, {epochs}, {batch_size}, {grad_accum_steps},
{effective_batch}, {lr}, {lr_encoder}, {weight_decay}, {workers}, {checkpoint_interval},
{eval_interval}, {test_interval_epochs}, {test_interval_minutes}, {test_split},
{logger_project}, {logger_run}
```

The same placeholders also work in `periodic_test.output_dir_name`,
`periodic_test.final_output_dir_name`, and `demo.output_dir`.

## Dataset Layout

RF-DETR `dataset_file: roboflow` auto-detects Roboflow COCO and YOLO layouts.

For other annotation formats, set `dataset.source_format: auto` or pass
`--dataset-source-format auto`. The trainer detects and converts these formats
into a reusable Roboflow COCO cache under `projects/rf_detr_trainer/dataset_cache/`:

```text
auto, rfdetr, roboflow, ultralytics_yolo, coco_json, pascal_voc, dota, labelme_json
```

Converted cache layout:

```text
projects/rf_detr_trainer/dataset_cache/
`-- <format>_<dataset>_<fingerprint>/
    |-- source_fingerprint.json
    |-- adapter_metadata.json
    |-- train/
    |   |-- _annotations.coco.json
    |   `-- *.jpg
    |-- valid/
    |   |-- _annotations.coco.json
    |   `-- *.jpg
    `-- test/
        |-- _annotations.coco.json
        `-- *.jpg
```

Config:

```yaml
dataset:
  source_format: auto
  dataset_dir: D:/datasets/football_dataset
  cache_root: dataset_cache
  refresh_cache: false
  split_ratio: [8, 1, 1]
  link_mode: auto
```

`link_mode: auto` tries file hardlinks and then symlinks. It does not silently
copy images; use `link_mode: copy` only when links are unavailable and you have
reviewed the dry-run disk estimate. If a source has no explicit split, the cache
uses deterministic train/valid/test splitting with `split_ratio: [8, 1, 1]`.
DOTA oriented boxes are converted to axis-aligned enclosing boxes for RF-DETR
detection training, with the original polygon recorded in the COCO segmentation
field and adapter metadata.

Ultralytics YOLO can be provided by `data_yaml`; single-file COCO can be
provided by `coco_json` plus optional `image_dir`.

Roboflow COCO example:

```text
dataset/
|-- train/
|   |-- _annotations.coco.json
|   `-- *.jpg
|-- valid/
|   |-- _annotations.coco.json
|   `-- *.jpg
`-- test/
    |-- _annotations.coco.json
    `-- *.jpg
```

Roboflow YOLO example:

```text
dataset/
|-- data.yaml
|-- train/
|   |-- images/
|   `-- labels/
|-- valid/
|   |-- images/
|   `-- labels/
`-- test/
    |-- images/
    `-- labels/
```

## Outputs

Training outputs are written to `<output_dir>/`:

```text
<output_dir>/
|-- checkpoint_best_regular.pth
|-- checkpoint_best_ema.pth
|-- checkpoint_best_total.pth
|-- checkpoint_*.pth
|-- metrics.csv
|-- dataset_grids/
|   |-- train_batch0_grid.jpg
|   |-- train_batch1_grid.jpg
|   |-- train_batch2_grid.jpg
|   |-- val_batch0_grid.jpg
|   |-- val_batch1_grid.jpg
|   `-- val_batch2_grid.jpg
|-- config/
|   |-- merged_config.yaml
|   |-- source_config.yaml
|   |-- rfdetr_train_config.yaml
|   |-- rfdetr_model_config.yaml
|   `-- run_metadata.json
|-- periodic_tests/
|   `-- epoch_0030/
|       |-- test_metrics.json
|       |-- test_metrics.csv
|       |-- test_per_class_metrics.json
|       |-- test_per_class_metrics.csv
|       `-- config/
`-- final_test/
    |-- test_metrics.json
    |-- test_metrics.csv
    |-- test_per_class_metrics.json
    |-- test_per_class_metrics.csv
    `-- config/
```

`test_metrics.json` contains the overall metrics and raw torchmetrics output.  
`test_per_class_metrics.csv/json` contains per-class `ap`, `ar`, `f1`, `precision`, and `recall`.

## Notes

- Scheduled in-training test is designed for single-process training. If you use `train.device: "0,1"` or another multi-process strategy, scheduled test is skipped during fit to avoid distributed synchronization issues; final test still runs after training.
- Every output folder includes the config snapshot that created it.
- Demo mode writes to `projects/rf_detr_trainer/demo_runs/` and clamps epochs, batch size, logging, and checkpoint interval.
