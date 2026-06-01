"""
Compatibility entry point for the Ultralytics model trainer.

Use this file when older scripts still call train_yolo.py. New work should call
train_ultralytics_model.py directly because it supports every model YAML under
ultralytics/cfg/models and can infer the correct Ultralytics task.

Example usage:
    uv run python projects/yolo_trainer/train_yolo.py --config projects/yolo_trainer/config/yolo_train.yaml --yes
    uv run python projects/yolo_trainer/train_yolo.py --model rtdetr-l.yaml --task auto --dry-run --yes
"""

from __future__ import annotations

import colorama

colorama.init(autoreset=True)

try:
    from .train_ultralytics_model import main
except ImportError:
    from train_ultralytics_model import main


if __name__ == "__main__":
    raise SystemExit(main())
