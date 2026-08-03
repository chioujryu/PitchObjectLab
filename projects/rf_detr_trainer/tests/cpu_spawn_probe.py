"""Subprocess fixture for verifying RF-DETR CPU limits under spawn."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import rf_detr_cpu_runtime as cpu_runtime  # noqa: E402


def child(queue: multiprocessing.Queue, config_path: str) -> None:
    cpu_runtime.reset_for_tests()
    cpu_runtime.os.cpu_count = lambda: 32
    os.environ["WORLD_SIZE"] = "6"
    policy = cpu_runtime.bootstrap_from_argv(
        Path(config_path),
        "train",
        ["--config", config_path],
    )
    queue.put(cpu_runtime.apply_loaded_runtime(policy))


def main() -> int:
    config_path = str(Path(sys.argv[1]).resolve())
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=child, args=(queue, config_path))
    process.start()
    summary = queue.get(timeout=60)
    process.join(timeout=60)
    if process.exitcode != 0:
        return int(process.exitcode or 1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())