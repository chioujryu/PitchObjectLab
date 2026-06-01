# D-FINE-seg Trainer

Config-first D-FINE-seg training subproject for Windows and Linux.

It vendors D-FINE-seg at commit `0a0f0a12511568857922924854a63d06e1ae0fbd`, adds automatic dataset conversion, richer augmentation config, output estimates, and config snapshots for reproducibility.

## Environment

From this folder:

```bash
cd projects/dfine_seg_trainer
uv run python setup_dfine_seg_env.py
uv sync
```

`setup_dfine_seg_env.py` detects the IP region, OS, NVIDIA driver/GPU, and writes a suitable `pyproject.toml`. On this machine it defaults to PyTorch `2.9.0` / torchvision `0.24.0` with the `cu130` wheel index. If the IP region is China, setup enables a Tsinghua PyPI index and the trainer uses a Hugging Face mirror unless you provide local or ModelScope weights.

## Quick Start

Edit:

```text
projects/dfine_seg_trainer/config/dfine_seg_train.yaml
```

Dry run:

```bash
uv run python train_dfine_seg_model.py --dry-run --yes
```

Train:

```bash
uv run python train_dfine_seg_model.py --yes
```

Common override:

```bash
uv run python train_dfine_seg_model.py \
  --dataset-dir /data/my_dataset \
  --dataset-format auto \
  --task segment \
  --model-name s \
  --device 0 \
  --epochs 100 \
  --batch-size 4 \
  --output-dir /runs/dfine_seg/my_run \
  --yes
```

Windows path example:

```powershell
uv run python train_dfine_seg_model.py `
  --dataset-dir D:\datasets\my_dataset `
  --task segment `
  --device 0 `
  --epochs 100 `
  --yes
```

## Dataset Conversion

The trainer auto-detects and converts these sources into a reusable D-FINE COCO-style cache:

```text
dfine_coco, coco_json, roboflow_coco, roboflow_yolo,
ultralytics_yolo, labelme, pascal_voc, dota
```

For `model.task: segment`, box-only datasets are rejected by default. Set:

```yaml
dataset:
  box_to_mask: true
```

to train with rectangular masks.

The cache is written under `projects/dfine_seg_trainer/dataset_cache/` by default and includes:

```text
train.json
val.json
test.json
images/
class_map.json
categories.json
source_fingerprint.json
dataset_adapter_metadata.json
```

## Outputs

Default output root:

```text
runs/detect/dfine_seg/train/
```

Each run writes:

```text
model.pt
last.pt
metrics.csv
extended_metrics.csv
config/merged_config.yaml
config/source_config.yaml
config/dfine_runtime_config.yaml
config/environment.json
config/run_metadata.json
debug_images/
eval_preds/
plots/
```

Before training, the wrapper prints an estimate of checkpoint, metric, plot, debug image, eval image, and dataset cache files. Without `--yes`, it asks for confirmation before creating heavy outputs.

## Augmentation

The config exposes Ultralytics-like controls:

```text
hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective,
flipud, fliplr, bgr, mosaic, mixup, cutmix, copy_paste, copy_paste_mode,
close_mosaic, multi_scale
```

It also exposes advanced Albumentations probabilities for CLAHE, brightness/contrast, gamma, blur, motion blur, noise, ISO noise, grayscale, sharpen, compression, coarse/grid dropout, random shadow, weather-like rain, and downscale.

## Vendored Upstream

The upstream source is under:

```text
projects/dfine_seg_trainer/vendor/D-FINE-seg
```

License: Apache-2.0, preserved in `vendor/D-FINE-seg/LICENSE`.
