# RF-DETR Trainer

Config-first RF-DETR training project with an Ultralytics-style workflow:

1. Project-anchored output directory using `output.output_dir` or
   `output.root` + `output.name`, with explicit paths still able to leave the
   project when needed.
2. Validation every epoch by default through RF-DETR `eval_interval`.
3. Scheduled test-set evaluation every N epochs or minutes.
4. Final test-set evaluation after training.
5. Per-epoch validation metrics plus overall/per-class test metrics saved as JSON and CSV.
6. Config snapshots inside every training/test output folder.
7. Train batch grids plus validation label/prediction grids for label checks.
8. A file/resource estimate plus confirmation before large outputs are written.
9. Auto-detection and cache conversion for RF-DETR/Roboflow, Ultralytics YOLO,
   COCO JSON, Pascal VOC, DOTA, and LabelMe JSON datasets.

## Environment

This subproject has its own `pyproject.toml`, `uv.lock`, and `.venv`.
RF-DETR is pinned to `rfdetr==1.8.3`; keep `pyproject.toml` and the local
`uv.lock` together when reproducing TensorRT exports.

```bash
cd projects/rf_detr_trainer
uv sync
```

PyTorch inference is installed by default. Install the optional TensorRT 10
CUDA 12 runtime only on a supported NVIDIA CUDA machine:

```bash
uv sync --extra tensorrt
```

The extra intentionally uses `onnx>=1.16,<2` and
`tensorrt-cu12>=10.16,<11`; it does not install PyCUDA, onnxsim, or depend on
the external `trtexec` command.

Auto-select CPU/GPU PyTorch wheels by checking CUDA and the current public IP
region first:

```bash
uv run python scripts/setup_pytorch_uv.py --dry-run
uv run python scripts/setup_pytorch_uv.py --yes
```

The setup script writes the selected PyTorch index into `pyproject.toml`; after
that, `uv sync` is enough for future environment setup. China/HK/MO/TW IP
regions use mirror indexes by default, while other regions use official PyTorch
indexes. Use `--region official` or `--region china` when a VPN makes the public
IP different from the download route you want.

On Windows and Linux, `uv sync` installs the CUDA 12.8 PyTorch wheels from
the PyTorch index configured in `pyproject.toml`. The CUDA build can
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

## Tests and Continuous Integration

Run the complete RF-DETR unit-test suite from this subproject:

```bash
uv run --frozen python -u -m unittest discover -s tests -p "test_*.py" -v
```

The PitchObjectLab CI workflow runs this suite with Python 3.12 on both Linux
and Windows for every pull request and every push to `main`. CI installs
CPU-only PyTorch wheels from the dependencies declared in `pyproject.toml`; it
does not change the CUDA 12.8 development lockfile, build a TensorRT engine, or
claim real GPU/TensorRT execution coverage. TensorRT presets are still parsed
and checked as configuration contracts by the unit tests.

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

Resume an interrupted run from its latest complete Lightning checkpoint:

```bash
uv run python train_rf_detr_model.py \
  --resume runs/rf_detr/train/EXISTING_RUN/checkpoint_N.ckpt \
  --output-dir runs/rf_detr/train/EXISTING_RUN \
  --exist-ok true
```

`--resume` restores the model, optimizer, scheduler, EMA, early-stopping, and
checkpoint callback state. Use a `checkpoint_N.ckpt` archive rather than a
`checkpoint_best_*.pth` inference checkpoint. Resuming starts after the last
completed epoch stored in the checkpoint; work from a partially completed
epoch is repeated. The normal output estimate and confirmation still apply,
or pass `--yes` after reviewing them.

Training and evaluation DataLoader workers default to `2` in configs for stable
Windows/Linux behavior. An explicit `num_workers: 0` is preserved, which is
useful when debugging or when process creation is undesirable.

### CPU thread budget

All train, test, and inference presets enable a balanced numerical-library
thread budget by default:

```yaml
runtime:
  cpu:
    enabled: true
    budget_percent: 50
    torch_intraop_threads: auto
```

The total budget is `floor(logical CPUs * budget_percent / 100)`. Training
shares it across `WORLD_SIZE` or explicit GPU devices, standalone testing
shares it across `test.parallel.chunks`, and inference uses one model process.
For example, 32 logical CPUs at 50% produce 16 total threads; six test chunks
receive two threads each. A run fails before model work starts when the model
process count is larger than the total thread budget. With
`torch_intraop_threads: auto`, single-GPU test/inference is capped at 16
threads even on 64/128-thread hosts. When both 8 and 16 threads fit the CPU
budget, startup runs a short batched preprocessing benchmark and keeps the
faster choice in the timing metadata. The RF-DETR per-image preprocessing path
becomes slower when it fans small operations across dozens of CPU threads.
Training keeps the full per-process budget. Set a positive integer to override
the task-aware default and skip this benchmark.

CLI values override YAML, and YAML overrides the defaults:

```bash
# Use 25% of logical CPUs as the total thread budget.
uv run python train_rf_detr_model.py --cpu-budget-percent 25 --dry-run --yes

# Explicitly disable project-managed limits and retain library/environment defaults.
uv run python test_rf_detr_model.py --no-cpu-limit --dry-run --yes

# Re-enable a limit disabled by YAML.
uv run python inference_rf_detr_model.py --cpu-limit --cpu-budget-percent 50 --yes

# Explicitly use eight PyTorch preprocessing threads.
uv run python test_rf_detr_model.py --torch-intraop-threads 8 --yes
```

The supported flags on all three entrypoints are `--cpu-budget-percent 1..100`,
`--torch-intraop-threads {auto|N}`, `--cpu-limit`, and `--no-cpu-limit`. The percentage is a cross-platform thread
budget for PyTorch, OpenMP, BLAS, NumExpr, and OpenCV; it is not a precise CPU
utilization cap. The startup log, config snapshot metadata, and
`run_timing.json` record the resolved process split and observed library thread
pool values.

## RF-DETR + TrackNetV5-inspired temporal training

This project implements a TrackNetV5-inspired temporal head around RF-DETR; it
is not the original TrackNet U-Net. RF-DETR remains the detector and may use
the optional `[P2, P3, P4]` feature graph. The temporal branch adds
paper-derived MDD attention, divided spatial/temporal R-STR attention,
factorized positional encoding, PixelShuffle residuals, three-frame Gaussian
heatmaps, context dropout, and class-balanced focal BCE.

The six supported training presets use official RF-DETR resolutions
Small/Medium/Large = 512/576/704:

```bash
uv run python train_rf_detr_model.py --config config/rf_detr_train_small_tracknet_v5.yaml --yes
uv run python train_rf_detr_model.py --config config/rf_detr_train_small_p2_tracknet_v5.yaml --yes
uv run python train_rf_detr_model.py --config config/rf_detr_train_medium_tracknet_v5.yaml --yes
uv run python train_rf_detr_model.py --config config/rf_detr_train_medium_p2_tracknet_v5.yaml --yes
uv run python train_rf_detr_model.py --config config/rf_detr_train_large_tracknet_v5.yaml --yes
uv run python train_rf_detr_model.py --config config/rf_detr_train_large_p2_tracknet_v5.yaml --yes
```

The defaults target an 8 GB GPU: BF16/AMP and gradient checkpointing are on,
`model.motion.temporal.backbone_grad_mode: center_only` retains backbone
activations only for the centre frame, Small uses batch 4, and Medium/Large
use batch 1. Context frames still use the same updated backbone weights, but
their feature extraction runs sequentially under `no_grad`.

All six presets use `grad_accum_steps: 1` and disable early stopping. Before
training, the estimate prints micro-batches/epoch, optimizer steps/epoch, total
optimizer steps, and effective batch size. Fewer than five updates per epoch or
100 total updates produces a warning. With 20 windows and batch 4, for example,
`grad_accum_steps: 8` yields only one optimizer update per epoch; `Epoch 5:
0/5` then means the first slow batch of the next epoch has not finished, not
that training is deadlocked.

Each epoch summary records `global_step`, detector and heatmap loss, best-box
IoU, top-query score, and first-batch duration. Early mAP can remain zero while
the newly initialized single-class/P2 heads calibrate, so use optimizer-step
count and these diagnostics to distinguish slow learning from a stalled run.

The old `rf_detr_train_motion_v5_*` filenames are deprecated aliases. Custom
checkpoints now use architecture metadata schema v3 with a stable fingerprint
covering resolved model size, resolution, queries/classes, P2 graph, TrackNet
graph, and RF-DETR version. Resume/test/inference require exact graph and state
shapes. Legacy custom TrackNet checkpoints must be retrained; official stock
RF-DETR checkpoints remain valid training initialization.

### Temporal smoke checks

The bounded smoke uses one real three-frame window. At 25 steps it requires
both detector and TrackNet gradients/parameter changes, heatmap loss reduction
of at least 20%, total loss reduction of at least 10%, and either a 0.05
best-box IoU gain or a failed-to-successful match transition:

```bash
uv run --frozen python run_temporal_micro_smoke.py --model-size small --p2 on --steps 25 --reload
uv run --frozen python run_temporal_micro_smoke.py --matrix --steps 1 --official-resolution --reload --max-minutes 30
```

`--matrix` covers Small/Medium/Large with P2 off/on and reports loss, separate
gradient norms, parameter deltas, peak VRAM, checkpoint path, and fresh-model
reload status. The parent watchdog terminates the complete worker process tree
with exit code 124 if `--max-minutes` is exceeded.

The formal Small P2 smoke checkpoint can exercise matching test and inference
graphs:

```bash
uv run python train_rf_detr_model.py --config config/rf_detr_train_smoke_temporal_tracknet_v5.yaml --yes
uv run python test_rf_detr_model.py --config config/rf_detr_test_smoke_temporal_tracknet_v5.yaml --checkpoint PATH/TO/checkpoint_best_regular.pth --yes
uv run python inference_rf_detr_model.py --config config/rf_detr_inference_smoke_temporal_tracknet_v5.yaml --checkpoint PATH/TO/checkpoint_best_regular.pth --yes
```

The inference smoke writes detections and TrackNet peaks to
`temporal_predictions.jsonl` plus one heatmap PNG per frame. Real-temporal
TrackNet (`temporal.mode: real`) is PyTorch-only; ONNX/TensorRT requests fail
instead of silently switching to a single-frame graph.

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
  --eval-interval 1 \
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

### Small architecture presets

Small detection keeps the official RF-DETR 512 x 512 default and is provided
as a complete Stock/P2/TrackNetV5 matrix for training, standalone test, and
inference:

| Architecture | Train | Test | Inference |
| --- | --- | --- | --- |
| Stock | `config/rf_detr_train_small.yaml` | `config/rf_detr_test_small.yaml` | `config/rf_detr_inference_small.yaml` |
| P2 only | `config/rf_detr_train_small_p2.yaml` | `config/rf_detr_test_small_p2.yaml` | `config/rf_detr_inference_small_p2.yaml` |
| TrackNetV5 only | `config/rf_detr_train_small_tracknet_v5.yaml` | `config/rf_detr_test_small_tracknet_v5.yaml` | `config/rf_detr_inference_small_tracknet_v5.yaml` |
| P2 + TrackNetV5 | `config/rf_detr_train_small_p2_tracknet_v5.yaml` | `config/rf_detr_test_small_p2_tracknet_v5.yaml` | `config/rf_detr_inference_small_p2_tracknet_v5.yaml` |

Stock uses the official Small initialization. P2 presets use
`projector_scale: [P2, P3, P4]`; TrackNetV5 presets use
`model.motion.type: tracknet_v5`. TrackNetV5 here is the model motion module,
not `inference.tracking`. Test and inference with P2, TrackNetV5, or their
combination require a checkpoint trained with that exact architecture.

```bash
# Stock Small uses the checkpoint/pretrain configured by the preset.
uv run python inference_rf_detr_model.py --config config/rf_detr_inference_small.yaml --source PATH/TO/media --yes

# Custom Small architectures require a matching checkpoint.
uv run python test_rf_detr_model.py --config config/rf_detr_test_small_p2.yaml --checkpoint PATH/TO/small_p2_checkpoint.pth --yes
uv run python inference_rf_detr_model.py --config config/rf_detr_inference_small_p2_tracknet_v5.yaml --checkpoint PATH/TO/small_p2_tracknet_v5_checkpoint.pth --source PATH/TO/media --yes
```

`train.device` options:

```text
auto, cpu, cuda, 0, 1, 0,1, cuda:0, mps, -1
```

Validation and checkpoint defaults:

```yaml
train:
  # Save archive checkpoint_<epoch>.pth every epoch.
  checkpoint_interval: 1
  # Save resumable last.ckpt every epoch; fast profiles may use 2.
  save_last_interval: 1
  # Run RF-DETR validation loader and metrics every epoch.
  eval_interval: 1
```

`train.eval_interval` is also forwarded to Lightning's
`check_val_every_n_epoch` unless you explicitly override that key under
`trainer.extra_trainer_args`.

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

uv run python test_rf_detr_model.py \
  --config config/rf_detr_test.yaml \
  --chunks 6 \
  --yes
```

`config/rf_detr_test.yaml` is test-only. Use `test.split`,
`test_mode.mode`, `crop`, `sahi`, `periodic_test`, and `evaluation` there; the
script builds any RF-DETR runtime adapter settings internally. Full-image
segmentation tests use RF-DETR's mask-aware evaluation path; `sahi` and
`class_crop` tests evaluate boxes.

Standalone bounding-box tests can split the selected dataset across concurrent
model replicas:

```yaml
model:
  # One device or a comma-separated round-robin list.
  device: "0,1"

test:
  parallel:
    chunks: 6
```

Each chunk loads the same configured model and checkpoint. By default the
runner clamps `chunks` to the number of unique GPUs, so `chunks: 6` with
`device: 0` runs one persistent model worker instead of six contending replicas.
Set `test.parallel.allow_same_gpu_oversubscription: true` only for an explicit
experiment. With `device: "0,1"`,
the six workers are assigned to GPUs `0,1,0,1,0,1`. The chunk count must be a
positive integer no larger than the number of selected test images. Multiple
replicas on one GPU multiply model-memory demand and do not imply a linear
speedup, so use `--dry-run --chunks 6 --yes` to inspect the assignment and
estimate first.

Parallel workers produce one combined set of predictions, metrics, diagnostics,
and aliases. `parallel_summary.json` records every chunk's device, image count,
timings, effective batch sizes, and terminal status; any failed chunk fails the
whole evaluation instead of publishing partial metrics. `chunks: 1` is the
default and preserves the original single-model path. This setting applies only
to `test_rf_detr_model.py`, not scheduled/final tests inside training.

Mask-aware full-image segmentation (`evaluation.type: auto` or `segm`) requires
`chunks: 1`. To run a segmentation checkpoint through parallel box evaluation,
set `evaluation.type: bbox` explicitly; parallel `sahi` and `class_crop` modes
also evaluate bounding boxes.

### Inference acceleration (inference and standalone test)

`inference_rf_detr_model.py` and `test_rf_detr_model.py` share the same optional
inference acceleration settings. The default is PyTorch FP32 and preserves the
legacy behavior. PyTorch BF16, TensorRT FP16, and TensorRT BF16 are the other
supported modes:

```yaml
model:
  # model.amp remains a separate RF-DETR construction/training setting.
  inference_optimization:
    backend: pytorch       # pytorch, tensorrt
    pytorch:
      precision: fp32      # fp32, bf16
    tensorrt:
      precision: fp16      # fp16, bf16
      engine_path: ""      # empty => build/load the automatic cache
      manifest_path: ""    # explicit engine: empty derives <engine>.manifest.json
      cache_dir: "runs/rf_detr/tensorrt_cache"  # relative => projects/rf_detr_trainer/...
      workspace_gib: 4
      force_rebuild: false
      profile:
        min_batch_size: 1  # fixed: real tail batches are not padded
        opt_batch_size: auto
        max_batch_size: auto
```

Only these backend/precision pairs are valid: `pytorch/fp32`,
`pytorch/bf16`, `tensorrt/fp16`, and `tensorrt/bf16`. TensorRT is CUDA-only;
BF16 additionally requires a compatible Ampere-or-newer GPU and TensorRT 10.
Unsupported hardware, missing optional packages, corrupt or incompatible
artifacts, and invalid precision combinations fail immediately. An application
may explicitly supply a BF16-reference accuracy callback; a rejected callback
restores PyTorch BF16 and records the rejected TensorRT metadata. The normal
CLI does not invent an accuracy result: without such a callback,
`accuracy_parity.performed` is false and ordinary runtime failures do not
silently change backend or precision.

Without `engine_path`, the runner exports dynamic-batch ONNX, builds an engine,
and reuses a cache keyed by the checkpoint/model, TensorRT/CUDA versions,
physical GPU UUID/PCI subdevice/VBIOS/driver, precision, output type, and batch
profile. A model/config-matched ONNX export may be reused across compatible
devices, but the serialized engine is always keyed to the physical GPU. New
engines contain batch 1/4/8/16 optimization profiles within the configured
range and benchmark the usable profiles at startup. The shipped presets use the
project-local `runs/rf_detr/tensorrt_cache/`; an empty `cache_dir` has the same
fallback, and every relative cache path is resolved from
`projects/rf_detr_trainer`. An absolute path or a path
beginning with `../` can intentionally place it elsewhere. An explicit trusted
engine must be paired with its project-generated JSON manifest (an empty `manifest_path`
derives `<engine>.manifest.json`); a checkpoint, GPU, precision,
I/O, class-count, or profile mismatch is rejected. Engine build time is reported
separately from steady-state inference time.

Equivalent CLI overrides are available on both entrypoints:

```bash
# PyTorch BF16
uv run python test_rf_detr_model.py --config config/rf_detr_test.yaml \
  --inference-backend pytorch --inference-precision bf16 --yes

# TensorRT with automatic cache/build
uv run python inference_rf_detr_model.py --config config/rf_detr_inference.yaml \
  --inference-backend tensorrt --inference-precision fp16 \
  --tensorrt-cache-dir D:/model-cache/rfdetr --yes

# Reuse a trusted prebuilt engine and its adjacent/project manifest
uv run python test_rf_detr_model.py --config config/rf_detr_test.yaml \
  --inference-backend tensorrt --inference-precision bf16 \
  --tensorrt-engine D:/engines/rfdetr.plan --yes
```

The full CLI set is `--inference-backend {pytorch,tensorrt}`,
`--inference-precision {fp32,fp16,bf16}`, `--tensorrt-engine PATH`,
`--tensorrt-cache-dir PATH`, and `--tensorrt-force-rebuild`.

The evaluator's accelerated path prepares a complete image batch once, performs
one grouped resize/normalization/H2D transfer, then calls raw PyTorch or
TensorRT inference. It does not use RF-DETR's legacy per-PIL-image range scans,
source-image copies, or one-H2D-per-image path. TensorRT execution uses a
private CUDA stream and reusable events. The sequential evaluator explicitly
uses reusable output buffers after postprocess has copied detections to CPU;
arbitrary raw-output callers keep buffer reuse disabled unless they opt in.

### Single-GPU safe and fast profiles

Medium-P2 train, test, and inference presets are provided as matched pairs:

```text
config/rf_detr_train_medium_p2_performance_{safe,fast}.yaml
config/rf_detr_test_medium_p2_performance_{safe,fast}.yaml
config/rf_detr_inference_medium_p2_performance_{safe,fast}.yaml
```

Replace their checkpoint/dataset/media placeholders first. Safe inference/test
uses PyTorch BF16, SAHI 320 with 20% overlap, recheck 640, automatic SAHI batch
4/8/16 selection, and full-scale CMC. Fast uses TensorRT FP16 with the same
detection geometry and half-scale CMC. The inference pair explicitly maps its
one checkpoint class to `football`; change `model.num_classes` and
`dataset.class_names` together for a different checkpoint. Fast training uses fixed shapes,
`torch.compile`, EMA every two optimizer steps, validation every five epochs,
and a resumable checkpoint every two epochs. Both training profiles can measure
microbatches 4/8/16 (50 warmup plus 200 measured backward steps), preserve an
effective batch of 32, enforce peak VRAM below 90%, and only test DataLoader
workers 4/8 when the workers-2 wait exceeds 5%. Results are written to
`train_autotune.json`; Lightning phase timing is written to
`phase_profile.txt`. Fast training reloads the best checkpoint for one complete
validation before final test. Use `--performance-profile safe|fast` for the
equivalent runtime overrides; slice geometry remains owned by the selected
config.

The word `fast` identifies the candidate policy, not a fabricated accuracy
certification. A production promotion still needs the fixed-set mAP/football
recall/tracking gate. When a caller supplies that gate through the acceleration
API, rejection automatically restores safe PyTorch BF16; otherwise the run
metadata explicitly says that parity was not performed.

Acceleration covers image/video inference, full-image batches, SAHI, target
class recheck, and class crop. Standalone full-image segmentation keeps true
bbox+mask evaluation and requires `test.parallel.chunks: 1`; SAHI and class-crop
segmentation modes continue to evaluate bounding boxes only. Training-time
periodic/final tests are unchanged.

#### P2 and TrackNetV5 TensorRT

TensorRT FP16 is the required compatibility baseline for custom P2 and TrackNetV5 models.
BF16 is optional and should be attempted only after FP16 passes and the
installed TensorRT/GPU report BF16 support. Keep the complete `model.p2` and
`model.motion` blocks identical to training; the fragments below show only the
relevant switches.

For P2-only test and inference, start from
`config/rf_detr_test_sahi_medium.yaml` and
`config/rf_detr_inference_medium_p2_video_1984090152231178242_003.yaml`:
update their dataset/source and device fields for the target machine.

For a complete Medium P2 inference example combining TensorRT FP16, 160 x 160
SAHI slices, centered 320 x 320 football rechecks, OC-SORT tracking, and six
explicit video placeholders, use
`config/rf_detr_inference_medium_p2_tensorrt_sahi160_recheck320_ocsort_6videos.yaml`.
Replace its checkpoint and six source paths before running it; keep the P2
architecture identical to the checkpoint.

```yaml
model:
  inference_optimization:
    backend: tensorrt
    tensorrt:
      precision: fp16
  p2:
    enabled: true
    projector_scale: [P2, P3, P4]
  motion:
    enabled: false
```

```bash
uv run python test_rf_detr_model.py --config config/rf_detr_test_sahi_medium.yaml --checkpoint PATH/TO/p2_checkpoint.pth --inference-backend tensorrt --inference-precision fp16 --tensorrt-force-rebuild --yes
uv run python inference_rf_detr_model.py --config config/rf_detr_inference_medium_p2_video_1984090152231178242_003.yaml --checkpoint PATH/TO/p2_checkpoint.pth --source PATH/TO/media --inference-backend tensorrt --inference-precision fp16 --tensorrt-force-rebuild --yes
```

For TrackNetV5-only runs, start from the ready-to-edit example configs
config/rf_detr_test_tracknet_tensorrt_fp16_example.yaml and
config/rf_detr_inference_tracknet_tensorrt_fp16_example.yaml. Update the
dataset path and keep their complete motion block identical to the checkpoint:

```yaml
model:
  inference_optimization:
    backend: tensorrt
    tensorrt:
      precision: fp16
  p2:
    enabled: false
  motion:
    enabled: true
    type: tracknet_v5
    temporal:
      fallback_mode: identity
```

```bash
uv run python test_rf_detr_model.py --config config/rf_detr_test_tracknet_tensorrt_fp16_example.yaml --checkpoint PATH/TO/tracknet_checkpoint.pth --tensorrt-force-rebuild --yes
uv run python inference_rf_detr_model.py --config config/rf_detr_inference_tracknet_tensorrt_fp16_example.yaml --checkpoint PATH/TO/tracknet_checkpoint.pth --source PATH/TO/media --tensorrt-force-rebuild --yes
```

Legacy P2 checkpoints need no conversion. They must use the same model size,
resolution, `projector_scale`, and projector options used for training;
test/inference rejects incompatible tensor shapes instead of silently replacing
weights. Checkpoints without architecture metadata use the YAML architecture
plus state-dict shapes for validation.

P2+TrackNet uses patch-16 R-STR attention rather than global attention over
stride-4 pixels. The provided Medium/Large presets start at batch 1 and use
centre-only backbone gradients for an 8 GB GPU. Keep the checkpoint fingerprint
and state graph exact when enabling both blocks.

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

#### Large-P2 FP16 smoke test

`config/rf_detr_inference_large_p2_tensorrt_fp16_smoke.yaml` is the fixed
compatibility preset for a Large detection checkpoint trained at 704 x 704
with one class, P2 `[P2, P3, P4]`, gradient checkpointing enabled, and no
TrackNetV5 module. It uses TensorRT FP16, a fixed batch-1 profile, a 4 GiB
workspace, full-image inference, and disabled video tracking.

The preset validates and exports the checkpoint; it does not convert a Medium,
Stock, TrackNetV5, or otherwise incompatible checkpoint into this architecture.
Use placeholders or machine-local CLI arguments for inputs rather than storing
workstation paths in the config:

```bash
# First run: build a fresh engine and infer a short video segment.
uv run python inference_rf_detr_model.py \
  --config config/rf_detr_inference_large_p2_tensorrt_fp16_smoke.yaml \
  --checkpoint PATH/TO/checkpoint_best_ema.pth \
  --source PATH/TO/input.mp4 \
  --max-seconds 0.2 \
  --tensorrt-force-rebuild \
  --yes

# Second run: omit force-rebuild and confirm acceleration.cache_hit: true in run_timing.json.
uv run python inference_rf_detr_model.py \
  --config config/rf_detr_inference_large_p2_tensorrt_fp16_smoke.yaml \
  --checkpoint PATH/TO/checkpoint_best_ema.pth \
  --source PATH/TO/input.mp4 \
  --max-seconds 0.2 \
  --yes
```

TensorRT engines are tied to their build-time GPU and TensorRT/CUDA/runtime
compatibility envelope. Rebuild on the target machine when the generated
manifest rejects a mismatch; do not copy an unverified engine between systems.
If the 4 GiB workspace cannot be allocated, retry with `workspace_gib: 2` in a
local copy of the preset without changing Large, 704, P2, or FP16.

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

Batch grids are written directly in the training output folder:

- `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg` are captured from
  the first training batches in a run to inspect labels and augmentations.
  Boxes are rendered against the actual model-input tensor/mask size, so they
  stay aligned after RF-DETR's in-step multi-scale resize.
- `val_batch*_labels.jpg` and `val_batch*_pred.jpg` are refreshed after every
  formal RF-DETR validation. Prediction grids overwrite the previous images
  with the latest model predictions whose confidence score is at least `0.25`,
  so low-confidence RF-DETR top-K candidates do not obscure useful diagnostics.

The provided `rf_detr_train*.yaml` configs enable the same RF-DETR
`train.aug_config` preset: horizontal flip, brightness/contrast jitter, and
Gaussian blur. Set `train.aug_config: null` in a copied config to use RF-DETR's
library default instead. Detection training keeps horizontal flip enabled; only
keypoint training without configured `keypoint_flip_pairs` disables horizontal
flip to avoid left/right joint label errors.

Custom output address:

```yaml
output:
  output_dir: "runs/rf_detr/{dataset_name}/{model_size}_e{epochs}_b{batch_size}_{timestamp}"
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

All relative training, standalone-test, inference, demo, and TensorRT cache
output paths are resolved from `projects/rf_detr_trainer`, independent of the
shell's current working directory and whether a same-named directory already
exists. For example, `runs/rf_detr/example` stays inside this project, while
`../shared-runs/example` intentionally escapes to the parent directory.
Absolute paths are honored unchanged. Scheduled and final tests remain nested
under the main training output directory.

## Inference

`inference_rf_detr_model.py` runs RF-DETR over image files, video files, folders,
mixed image/video folders, and HTTP(S) media URLs. Edit `config/rf_detr_inference.yaml`
and run:

```bash
uv run python inference_rf_detr_model.py --config config/rf_detr_inference.yaml --dry-run --yes
uv run python inference_rf_detr_model.py --config config/rf_detr_inference.yaml --source D:/clips/match.mp4 --yes
```

Each run keeps writing the legacy rendered media, `predictions.jsonl`,
`football_predictions.jsonl`, `class_colors.json`, `inference_summary.json`, and
a config snapshot into the output folder. Video sources additionally produce
Canonical Video Output V2 bundles by default; this is an additive interface and
does not change the legacy JSONL schemas.

Video inference defaults to a bounded, single-decode streaming pipeline:

```yaml
inference:
  video:
    streaming: true
    queue_size: 32
    save_video: true
```

Decoded frames are passed directly to the batched evaluator in memory, then
tracked, rendered, and encoded in source order. This removes the former JPEG
frame cache and second complete video decode. `save_video: false` or
`--no-save-video` disables only the annotated video; the Canonical V2 clean clip
is controlled separately. For a truly JSON-only run, use both
`--no-save-video --no-canonical-output`. When `detection_fps` skips model frames,
enabled trackers are still advanced on every source frame so their age and
motion state remain in video-frame units.

`inference_summary.json` and `run_timing.json` separate decode, frame
conversion, crop, host preprocessing, H2D, resize/normalize, forward,
postprocess, tracker/CMC, render, encode, serialization, and orchestration.
They also record source/detection frames, slice/recheck inputs, actual model
batches (including one-time autotune work), requested/effective SAHI batches,
queue peak, OOM retries, and peak allocated/reserved/device VRAM. Runtime
history uses critical-path wall time rather than summed parallel-worker time.

`football_predictions.jsonl` is the concise downstream interface for ball
coordinates. It keeps one row per detected football (multiple footballs in one
frame produce multiple rows) and uses absolute source-image pixels with
center-point `xywh = [center_x, center_y, width, height]`. It does not repeat the
last detection on video frames skipped by `inference.video.detection_fps`, and
frames with no football detection produce no row. The full `predictions.jsonl`
remains unchanged and continues to use COCO top-left
`bbox = [x, y, width, height]`.

Every concise row has the same keys:

```json
{"kind":"video","source":"match.mp4","image_id":42,"frame_index":123,"timestamp_seconds":4.1,"category_id":0,"category_name":"football","score":0.91,"xywh":[100.5,200.25,8.0,7.5],"track_id":3}
```

`track_id` is always present. It is an integer after a tracker assigns the
detection, and `null` when tracking is disabled or the active tracker has not
assigned that detection. Image rows use `null` for frame/time fields. Temporal
TrackNetV5 rows use the anchor image path and frame index, with
`kind: temporal`, `image_id: null`, `timestamp_seconds: null`, and
`track_id: null`.

The exported classes are independent of rendering and tracking filters:

```yaml
inference:
  football_output:
    enabled: true
    target_class_ids: []           # non-empty IDs take precedence
    target_class_names: [football] # case-insensitive; empty defaults to football
```

When enabled, every configured ID/name must resolve against the dataset
categories before the confirmation prompt or any output is created. For
numeric-only class metadata, set `target_class_ids` explicitly. The summaries
report `football_prediction_count` and `football_predictions_file`; the
pre-run estimate includes the additional JSONL file.

### Canonical Video Output V2

Canonical V2 is the frame-complete, VLM-facing video interface. It is enabled by
default for every video inference pipeline (one-pass, batched, and streaming)
and every tracker. Image and temporal inference do not create Canonical
bundles. Configure the exporter under `inference.canonical_output`:

```yaml
inference:
  canonical_output:
    enabled: true
    directory: canonical_v2
    ffmpeg_path: auto
    ffprobe_path: auto
    media:
      video_codec: libx264
      crf: 18
      preset: medium
      pixel_format: yuv420p
      audio_codec: aac
      audio_bitrate: 192k
```

The equivalent CLI controls are `--canonical-output` /
`--no-canonical-output`, `--canonical-output-dir DIR`,
`--ffmpeg-path PATH`, and `--ffprobe-path PATH`. CLI values override YAML.
`inference.video.save_video` and `--save-video` control only the annotated
rendered video; disabling them does not disable `media.mp4`.
Use `--no-save-video --no-canonical-output` together for a truly JSON-only run.

Every selected source video gets an independent bundle:

```text
canonical_v2/
  manifest.json
  <video_id>/
    metadata.json
    frames.jsonl
    tracks.jsonl
    media.mp4
```

All documents and JSONL rows carry `schema_version: "2.0.0"`. `video_id` is a
stable hash of the source and requested selected range; track IDs are scoped to
that `video_id`, not to the full inference run. The run-level `manifest.json`
lists only fully published bundles using relative paths. Each bundle is first
written under a `.partial` directory, validated, and atomically renamed; the
manifest is updated last, so an interrupted export is not presented as valid.

`metadata.json` is the authoritative video-level record. Its top-level objects
are `source`, `selected_segment`, `video`, `processing`, `audio`, `timestamps`,
`coordinate_convention`, `categories`, `producer`, `tracker`, `artifacts`,
`annotated_output`, and `clean_media`, together with `schema_version` and
`video_id`. They record:

- decoded and display width/height, rational and float FPS, time base, CFR/VFR
  status, codec, rotation, pixel format, and probe provenance;
- the requested and actual selected range, source/decoded/detection/clean-media
  frame counts, start/end frame and timestamp, and durations;
- detection frame interval, requested/effective detection FPS, annotated output
  FPS, timestamp source/fallback counts, detector checkpoint/config, and
  category mapping;
- audio presence and selected stream, tracker algorithm, offline
  lookahead/backfill settings, and the reliable observed/predicted/CMC
  capabilities exposed by that tracker; and
- relative artifact paths plus post-encode clean-media validation results.

`frames.jsonl` contains exactly one row per actually decoded frame in the
selected segment, including empty frames and frames on which the detector did
not run. The row contract is:

```text
schema_version, video_id,
segment_frame_index, source_frame_index,
source_timestamp_seconds, segment_timestamp_seconds, timestamp_source,
segment_progress, detection_ran,
detections[], track_states[], camera_motion
```

`segment_frame_index` is the primary zero-based index. Decoder timestamps are
preferred; a missing, invalid, or non-monotonic value falls back to nominal
`source_frame_index / FPS`, and `timestamp_source` identifies the choice on
each row. `detections` contains every final detector class. Hybrid rows preserve
the committed detector candidates before the optional legacy
`export_confirmed_only` filter. A track state refers to an observed detection by
`detection_index`; predicted-only states use `null` and never invent a detector
score. `camera_motion` is `null` for trackers without a reliable transform.
Hybrid may provide the previous-to-current affine, pixel translation, scale,
clockwise rotation, success/inlier diagnostics, method, processing scale, and
failure reason.

Each detection has:

```text
detection_index, category_id, category_name, score,
bbox_xyxy_pixels, bbox_xyxy_normalized,
center_pixels, center_normalized,
area_pixels, area_normalized,
was_clipped, in_frame, attributes
```

Each track state adds `track_id`, `provenance` (`observed` or `predicted`),
`status`, `hits`, `age_frames`, `seconds_since_observed`, the same geometry
fields, and a `motion` object. Motion is calculated in image space with the
actual timestamp delta and includes pixel/normalized velocity and speed,
clockwise direction plus an eight-way direction label, acceleration, and bbox
log-area change. The first usable point, non-positive time deltas, and unknown
frame gaps use `null` motion components rather than a fabricated zero.

Canonical boxes use continuous edge coordinates
`[x1, y1, x2, y2]`. Width and height are `x2 - x1` and `y2 - y1`; there is no
inclusive-pixel `+1`. Pixel coordinates retain their original out-of-frame
values so clipping is auditable. Normalized bbox edges and centers divide by
the decoded frame width/height and are clamped to `[0, 1]`; normalized area is
relative to frame area. Signed normalized motion values are scale conversions,
not probabilities, and are not clamped.

`tracks.jsonl` has one row per video-scoped track. The row stores category
IDs/names, start/end frame and time, duration, point/observed/predicted counts,
maximum frame/time gap, detector-score statistics, pixel/normalized path
length, and pixel/normalized net displacement. Its ordered `points` repeat
center, provenance, lifecycle, score, and motion information, but deliberately
do not duplicate the full bbox history already available in `frames.jsonl`.

Canonical output is an offline contract. In particular, Hybrid fixed-delay
lookahead and tentative-frame backfill are committed in original frame order,
so a row can reflect evidence from a later source frame. Do not treat these
rows as causal online predictions. Generic trackers export observed states
universally and predicted geometry only when their stable adapter exposes it;
unsupported values stay `null` and are declared in the metadata capabilities.
All speed and displacement fields are image-space values, not pitch/world
coordinates or physical ball speed.

Canonical video export requires FFmpeg and the configured video/audio encoders.
`auto` resolves executables from `PATH`; an explicit executable path is also
accepted. FFprobe enriches rational-FPS, VFR, container, rotation, and audio
metadata when available, with OpenCV/FFmpeg fallback provenance otherwise.
Before creating a bundle, the runner validates the required executable and
encoders. `media.mp4` is an exact re-encode of the actually processed selected
range with reset timestamps, no boxes or overlays, and the source's
default/first audio stream when present. If that audio stream ends before the
selected video range, the exporter pads only the missing tail with silence so
the clean clip remains synchronized. Sources without audio are recorded as
`has_audio: false`. Dimensions, decoded frame count, duration, and audio/video
synchronization are checked after encoding.

The dry-run and confirmation estimate includes the manifest plus four files per
video and a content-dependent clean-media disk/time estimate.
`inference_summary.json` reports the Canonical manifest path and
video/frame/track/media counts.

The existing `predictions.jsonl`, `football_predictions.jsonl`, and
`track_states.jsonl` retain their schemas and coordinate semantics:
`predictions.jsonl` keeps COCO top-left `bbox = [x, y, width, height]`, and
`football_predictions.jsonl` keeps center-based `xywh`. Consumers should opt
into Canonical V2 rather than reinterpreting those legacy fields.

When `inference.mode: sahi` and `sahi.recheck.enabled: true`,
`sahi.recheck.fused_confidence_threshold` is the final minimum score for every
prediction class. The runner applies it before tracking, JSONL/summary output,
and image/video rendering so every inference artifact uses the same filtered
predictions. Scores equal to the threshold are retained.
The recheck keeps first-stage geometry, selects a same-class second-stage box by
center residual and configurable size-ratio gates, and records `first_stage_score`,
`second_stage_score`, `second_stage_bbox`, `recheck_match_distance_pixels`, and
`recheck_passed`. Configure `match_center_tolerance_pixels`,
`match_min_size_ratio`, and `match_max_size_ratio` under `sahi.recheck`.

### Video tracking

Video inference can attach a multi-object tracker that adds stable `track_id`s, draws
trajectory trails, and writes a `tracking_summary.json`. The tracker is selected by
`inference.tracking.algorithm`:

```yaml
inference:
  tracking:
    enabled: true
    algorithm: circle     # default; or hybrid / ocsort / deepocsort / botsort / bytetrack
    target_class_names: [football]   # which class(es) to track (default football)
```

- `circle` (**default**) — the built-in, dependency-free centroid/search-circle ball tracker
  (`rf_detr_video_tracking.py`). Best for a single fast tiny ball where bounding boxes
  barely overlap between frames.
- `ocsort`, `deepocsort`, `botsort`, `bytetrack` — provided by
  [`boxmot`](https://github.com/mikel-brostrom/boxmot). OC-SORT and ByteTrack are
  motion-only and need no extra weights; Deep OC-SORT and BoT-SORT add appearance ReID.
- `hybrid` combines ByteTrack-style confidence tiers, OC-SORT observation correction,
  BoT-SORT-style camera-motion compensation, adaptive CV/CA motion, and delayed
  ambiguity handling. It tracks every visible football; it does not assume a regulation
  pitch, choose a single "main" ball, or use player/field/ReID semantics.

Each boxmot tracker's parameters live in its **own nested block** under `inference.tracking`
(`ocsort:`, `deepocsort:`, `botsort:`, `bytetrack:`); the shared ReID/camera-motion keys
(`reid_weights`, `reid_device`, `reid_half`, `cmc_method`, `per_class`) and the `circle` /
rendering keys sit at the `tracking` top level:

```yaml
inference:
  tracking:
    algorithm: ocsort
    draw_trajectory: true
    draw_predicted_trajectory: false
    draw_current_center: true
    draw_predicted_center: false
    render_confirmed_only: false  # Hybrid-only
    export_confirmed_only: false  # Hybrid-only
    ocsort:
      det_thresh: 0.2
      max_age: 30
      asso_threshold: 0.3
    bytetrack:
      track_thresh: 0.45
```

`draw_trajectory` remains the master switch for trajectory polylines.
`draw_predicted_trajectory` defaults to `false`; when disabled, the observed trail remains
visible but is not extended from its last observed point to the live predicted center.
`draw_current_center` controls only the current frame's real observed center. Its filled-dot
radius is fixed at 3 pixels and is independent of `trajectory_width`; the latter affects only
the trajectory line (including the existing old-to-new taper when `trajectory_taper: true`).
`draw_predicted_center` separately controls the live predicted dot during detection gaps for
every tracker. Hybrid never draws that dot beyond
`hybrid.lifecycle.predicted_output_seconds`, even when the switch is enabled. Search circles,
internal motion prediction, reassociation, and diagnostic tracker output are unaffected by
these rendering switches.

Hybrid can suppress false-positive display and export independently:

```yaml
inference:
  tracking:
    algorithm: hybrid
    render_confirmed_only: true
    export_confirmed_only: false
    trajectory_width: 4
```

`render_confirmed_only` hides target-class boxes, labels, centers, and trails for tracks that
never satisfy Hybrid's confirmation rule, as well as target-class detections with no track ID.
The shipped Hybrid presets use the existing two-hits-in-three-frames rule, backfill a track's
tentative first frame after final confirmation, and show its stable ID from that first frame.
A confirmed track resumes immediately after a lost/reacquired match. Non-target classes are
not filtered.

`export_confirmed_only` applies the same final-confirmation filter only to
`predictions.jsonl` and `football_predictions.jsonl`. It never filters
`track_states.jsonl` or `tracking_summary.json`, which retain tentative, lost, and predicted
diagnostics. `inference_summary.json` reports the published `prediction_count` plus
`raw_prediction_count` and `suppressed_unconfirmed_count`. Both confirmation switches default
to `false` for older external YAML files, and setting either to `true` with a non-Hybrid
algorithm is an error.

The tracked class is configurable for every algorithm via
`inference.tracking.target_class_ids` / `target_class_names`; with both empty it defaults
to `football`. The `predictions.jsonl` track fields and `tracking_summary.json` are the
same regardless of algorithm, so downstream tooling does not change when you switch.
Hybrid additionally writes `track_states.jsonl`. `predictions.jsonl` contains only
committed observed detections; predicted-only lifecycle rows, CMC diagnostics, covariance,
and ambiguity metadata live in `track_states.jsonl`. Hybrid uses fixed-delay ambiguity
handling through `hybrid.hypothesis.lookahead_seconds` and `ambiguity_margin`; `beam_width`
is unsupported and is rejected instead of being silently ignored.

The boxmot trackers require the dependency (already pinned in `pyproject.toml`):

```bash
uv add 'boxmot==13.0.0'
```

ReID notes for `deepocsort` / `botsort`: appearance weights (default
`osnet_x0_25_msmt17.pt`) auto-download from Google Drive on first use, which is
unreliable in China/HK/MO/TW. Set a local file to skip the download, disable appearance,
or use a motion-only tracker:

```yaml
inference:
  tracking:
    reid_weights: D:/weights/osnet_x0_25_msmt17.pt   # local file => no download
    # or: algorithm: ocsort        # motion-only, no ReID
    botsort:
      with_reid: false             # BoT-SORT without appearance (no download)
```

The ReID device follows `model.device` (override with `inference.tracking.reid_device`);
FP16 ReID (`reid_half`) is GPU-only and is forced off on CPU/MPS. CLI overrides:
`--tracker {circle,hybrid,ocsort,deepocsort,botsort,bytetrack}` and `--reid-weights PATH`, plus
the existing `--track` / `--no-track`. Every `inference.tracking.*` key is documented inline
in `config/rf_detr_inference.yaml`.


The ready-to-run football profile is
`config/rf_detr_inference_medium_p2_sahi320_recheck640_hybrid.yaml`: RF-DETR Medium at
resolution 576 with P2/P3/P4, SAHI 320 (20% overlap, GREEDYNMM+IOS), 640 recheck, 0.5/0.5
score fusion, final confidence 0.25, inference batch 8, and SAHI batch 4. Supply the
checkpoint explicitly so an accidental weight substitution cannot go unnoticed:

```bash
uv run --frozen python inference_rf_detr_model.py \
  --config config/rf_detr_inference_medium_p2_sahi320_recheck640_hybrid.yaml \
  --checkpoint C:/weights/checkpoint_best_ema.pth --source C:/data/video15 --yes
```

Prepare Label Studio review data, cache detections once, compare trackers, and sweep only
tracking parameters with `evaluate_rf_detr_tracking.py`. Pseudo identities are explicitly
non-official until a human completes identity review:

```bash
uv run --frozen python evaluate_rf_detr_tracking.py prepare_gt \
  --source-root LABEL_STUDIO_RUN --output-dir EVAL_DIR --yes
uv run --frozen python evaluate_rf_detr_tracking.py cache \
  --source-root LABEL_STUDIO_RUN --output-dir EVAL_DIR --config CONFIG --checkpoint MODEL --yes
uv run --frozen python evaluate_rf_detr_tracking.py evaluate \
  --source-root LABEL_STUDIO_RUN --output-dir EVAL_DIR --config CONFIG --checkpoint MODEL --yes
uv run --frozen python evaluate_rf_detr_tracking.py sweep \
  --source-root LABEL_STUDIO_RUN --output-dir EVAL_DIR --config CONFIG --checkpoint MODEL \
  --grid tracking_grids/hybrid_tracking_sweep.yaml --yes
```

The default comparison is offline-safe (`circle,ocsort,bytetrack,hybrid`). To add
`deepocsort,botsort`, pass them in `--algorithms` together with an existing local
`--reid-weights` file; evaluation never auto-downloads ReID weights.

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
|-- epoch_results/
|   |-- epoch_metrics.csv
|   |-- latest_val_metrics.json
|   `-- epoch_0001/
|       |-- val_metrics.json
|       `-- val_metrics.csv
|-- train_batch0.jpg
|-- train_batch1.jpg
|-- train_batch2.jpg
|-- val_batch0_labels.jpg
|-- val_batch0_pred.jpg
|-- val_batch1_labels.jpg
|-- val_batch1_pred.jpg
|-- val_batch2_labels.jpg
|-- val_batch2_pred.jpg
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

Training configs default to `checkpoint_interval: 1` and `eval_interval: 1`,
so every epoch keeps an archive checkpoint and writes scalar `val/*` metrics
under `epoch_results/`.
`test_metrics.json` contains the overall metrics and raw torchmetrics output.  
`test_per_class_metrics.csv/json` contains per-class `ap`, `ar`, `f1`, `precision`, and `recall`.

## Notes

- Scheduled in-training test is designed for single-process training. If you use `train.device: "0,1"` or another multi-process strategy, scheduled test is skipped during fit to avoid distributed synchronization issues; final test still runs after training.
- Every output folder includes the config snapshot that created it.
- Output-producing runs also write `run.log` in the output folder by mirroring console/stderr output after confirmation. Dry runs only print estimates and do not create output folders.
- Demo mode defaults to `demo_runs/` under `projects/rf_detr_trainer` and clamps epochs, batch size, logging, and checkpoint interval; an absolute or `../` output override may place it elsewhere.
