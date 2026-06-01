# Ultralytics Model Trainer

This subproject trains Ultralytics models from local repo configs with these additions over plain `yolo train`:

1. Any YAML under `ultralytics/cfg/models` can be used by filename or relative path.
2. `task: auto` lets Ultralytics infer `detect`, `segment`, `classify`, `pose`, or `obb` from the model.
3. `val_interval` can run validation every N epochs while keeping early stopping and `best.pt` tied to real val results.
4. Periodic test-split evaluation and final test evaluation can write metrics during/after training.
5. Every training/test output folder includes the config used to create that output.
6. A resource/file-count estimate is printed before heavy output is produced, with confirmation unless `--yes` is used.

## Files

```text
projects/yolo_trainer/
|-- README.md
|-- .gitignore
|-- train_ultralytics_model.py    # main script
|-- train_yolo.py                 # compatibility wrapper
|-- config/
|   |-- yolo_train.yaml           # default config with inline comments
|   `-- *.yaml                    # experiment configs
`-- generated_configs/            # auto-generated dataset YAMLs, git-ignored
```

## Environment

The repo root already has `pyproject.toml`, `uv.lock`, and `.venv`.

```bash
uv sync
```

No new dependency was added for this trainer. The required packages are already in the repo config: `pyyaml`, `colorama`, and `tqdm`.

## Quick Start

```bash
# Edit config first
nano projects/yolo_trainer/config/yolo_train.yaml

# Dry run: validate config and print resource estimate without training
uv run python projects/yolo_trainer/train_ultralytics_model.py --dry-run --yes

# Run with confirmation prompt
uv run python projects/yolo_trainer/train_ultralytics_model.py

# Skip confirmation prompt
uv run python projects/yolo_trainer/train_ultralytics_model.py --yes
```

Older commands that call `train_yolo.py` still work because it forwards to `train_ultralytics_model.py`.

## Model Configs

`model` can point to any YAML under `ultralytics/cfg/models`:

```yaml
model: rtdetr-l.yaml
task: auto
```

Supported forms:

```yaml
model: rtdetr-l.yaml
model: rt-detr/rtdetr-l.yaml
model: 26/yolo26-p2.yaml
model: v8/yolov8-seg.yaml
model: /absolute/path/to/custom_model.yaml
model: yolo26n.pt
```

CLI examples:

```bash
uv run python projects/yolo_trainer/train_ultralytics_model.py --model rtdetr-l.yaml --task auto --yes
uv run python projects/yolo_trainer/train_ultralytics_model.py --model v8/yolov8-seg.yaml --task auto --yes
uv run python projects/yolo_trainer/train_ultralytics_model.py --model 26/yolo26-p2.yaml --task detect --yes
```

The wrapper can load the model config, but the dataset still must match the task:

```text
detect/segment/pose/obb: YOLO YAML dataset with labels in that task format
classify: classification dataset directory layout
```

## Configure Dataset

Edit `projects/yolo_trainer/config/yolo_train.yaml`.

Point to an existing YOLO data YAML:

```yaml
dataset:
  data_yaml: /path/to/your/data.yaml
```

Or define the dataset inline so the script generates a data YAML automatically:

```yaml
dataset:
  path: /path/to/your/dataset
  train: images/train
  val: images/val
  test: images/test
  names:
    0: player
    1: goalkeeper
    2: ball
```

## Validation Interval

```yaml
train:
  val_interval: 1    # every epoch
  val_interval: 5    # every 5th epoch
  val_interval: 0    # disabled, same as val: false
  patience: 150      # with val_interval=5, about 30 val evaluations before early stop
```

On skipped epochs the previous validation fitness is reused, and `best.pt` is only updated from epochs where validation actually ran. Patience still counts training epochs, so use:

```text
patience = desired_validation_count * val_interval
```

CLI override:

```bash
uv run python projects/yolo_trainer/train_ultralytics_model.py --val-interval 5 --patience 150 --yes
```

## Periodic Test

Run the `test` split on a schedule during training:

```yaml
periodic_test:
  enabled: true
  test_mode:
    mode: full_image   # full_image, sahi, class_crop
  test_interval_epochs: 10
  test_interval_minutes: 0
  run_final_test: true
  classwise: true
```

`class_crop` runs a first prediction pass, unions the selected predicted classes,
adds `periodic_test.crop` padding, tests the crop, and projects predictions back
to the original image. If no crop class is predicted, it falls back to full-image
test for that image. Non-full-image modes write `test_batch*_labels.jpg`,
`test_batch*_pred.jpg`, and `model_inputs_manifest.*` in each test output folder.

CLI override:

```bash
uv run python projects/yolo_trainer/train_ultralytics_model.py --test-interval-epochs 10 --periodic-test-classwise --final-test --yes
```

## CLI Reference

Common overrides:

```bash
uv run python projects/yolo_trainer/train_ultralytics_model.py \
  --model rtdetr-l.yaml \
  --task auto \
  --device 7 \
  --epochs 1000 \
  --imgsz 640 \
  --batch 32 \
  --patience 150 \
  --val-interval 5 \
  --test-interval-epochs 10 \
  --name "rtdetr-l_val5_{timestamp}" \
  --yes
```

Pass any Ultralytics argument not listed as a first-class flag:

```bash
uv run python projects/yolo_trainer/train_ultralytics_model.py \
  --extra warmup_epochs=5 \
  --extra nbs=64
```

Important flags:

```text
--config PATH                 YAML config file
--yes                         Skip confirmation prompt
--dry-run                     Estimate outputs and exit without training
--verbose / --quiet           Toggle wrapper logs
--demo / --no-demo            Enable/disable demo mode

Dataset:
  --data-yaml PATH            Existing YOLO data YAML
  --dataset-root PATH         Dataset root directory
  --train PATH                Train split folder
  --val PATH                  Val split folder
  --test PATH                 Test split folder

Model:
  --model PATH                YAML/PT path or ultralytics/cfg/models alias
  --task TASK                 auto, detect, segment, classify, pose, obb
  --pretrained PATH/BOOL      Pretrained weights

Training:
  --epochs INT                Number of epochs
  --time FLOAT                Max training hours
  --imgsz INT                 Image size
  --batch INT/FLOAT           Batch size or AutoBatch fraction
  --patience INT              Early stopping patience in training epochs
  --val-interval INT          Val every N epochs, 1=every, 0=disable
  --device STR                CUDA device(s), cpu, mps, or -1
  --project PATH              Output project directory
  --name STR                  Run name, supports {timestamp}, {date}, and config placeholders
  --workers INT               Dataloader workers
  --optimizer STR             Optimizer name
  --save-period INT           Checkpoint every N epochs
  --fraction FLOAT            Training data fraction
  --resume BOOL               Resume from last checkpoint
  --amp BOOL                  Automatic mixed precision
  --plots BOOL                Save plots
  --val-enabled BOOL          Enable/disable validation
  --conf FLOAT                Val/test confidence threshold
  --iou FLOAT                 Val/test IoU threshold
  --max-det INT               Max detections per image

Periodic test:
  --periodic-test BOOL        Enable/disable test-split evaluation
  --test-interval-epochs INT  Test every N epochs
  --test-interval-minutes FLOAT
  --final-test                Run final test after training
  --no-final-test             Skip final test

Pass-through:
  --extra KEY=VALUE           Any Ultralytics argument, repeatable
```

## Demo Mode

Demo mode clamps epochs, fraction, and batch to a tiny run and writes to a separate folder:

```bash
uv run python projects/yolo_trainer/train_ultralytics_model.py --demo --yes
```

Config:

```yaml
demo:
  enabled: false
  project: projects/yolo_trainer/demo_runs
  max_epochs: 2
  max_fraction: 0.05
  max_batch: 4
  plots: false
  test_interval_epochs: 1
```

## Outputs

Training outputs go to `<project>/<name>/`:

```text
<project>/<name>/
|-- weights/
|   |-- best.pt
|   `-- last.pt
|-- results.csv
|-- config/
|   |-- merged_config.yaml
|   |-- source_config.yaml
|   |-- dataset.yaml
|   |-- ultralytics_train_args.yaml
|   `-- run_metadata.json
|-- periodic_tests/
|   `-- epoch_0010/
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

`train.project`, `train.name`, `periodic_test.output_dir_name`, and `periodic_test.final_output_dir_name` can be plain strings or templates. Built-in placeholders include `{timestamp}`, `{date}`, `{model}`, `{dataset}`, `{task}`, `{split}`, `{run_name}`, and periodic `{epoch}` / `{epoch4}`; config dot-paths such as `{train.imgsz}`, `{train.batch}`, `{dataset.path}`, or `{periodic_test.test_interval_epochs}` are also supported.

Every output folder includes a `config/` snapshot so the result can be reproduced.

## Useful Commands

```bash
# Dry run
uv run python projects/yolo_trainer/train_ultralytics_model.py --dry-run --yes

# CPU-only demo
uv run python projects/yolo_trainer/train_ultralytics_model.py --device cpu --demo --yes

# RT-DETR config from ultralytics/cfg/models/rt-detr
uv run python projects/yolo_trainer/train_ultralytics_model.py --model rtdetr-l.yaml --task auto --dry-run --yes

# Resume interrupted run
uv run python projects/yolo_trainer/train_ultralytics_model.py --resume true --yes
```

## Notes

- The script prints an output/resource estimate before training and asks for confirmation unless `--yes` or `runtime.confirm_before_run: false` is set.
- Generated dataset YAML files are written to `generated_configs/`, which is git-ignored.
- Large outputs, checkpoints, datasets, and demo runs are ignored by `.gitignore`.
- For periodic tests during training, prefer a single GPU or CPU. Ultralytics DDP multi-GPU training runs in subprocesses where Python callbacks may not fire in the parent process.
- Config values that use curly braces must be quoted in YAML, for example `name: "run_{timestamp}"`.
