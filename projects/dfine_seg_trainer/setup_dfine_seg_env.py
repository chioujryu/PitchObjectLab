"""
Prepare the D-FINE-seg trainer uv environment.

Example usage:
    uv run python setup_dfine_seg_env.py --dry-run
    uv run python setup_dfine_seg_env.py
    uv sync
"""

from __future__ import annotations

import argparse
import platform
import re
from typing import Any

import colorama
from colorama import Fore, Style
from dfine_seg_trainer.common import PROJECT_DIR, detect_cuda_banner, detect_ip_region, detect_nvidia, save_yaml

colorama.init(autoreset=True)

PYPROJECT = PROJECT_DIR / "pyproject.toml"

BASE_DEPENDENCIES = [
    "albumentations==2.0.8",
    "colorama>=0.4.6",
    "faster-coco-eval==1.6.5",
    "huggingface-hub==1.13.0",
    "hydra-core==1.3.2",
    "loguru==0.7.2",
    "matplotlib==3.10.3",
    "numpy==2.1.1",
    "omegaconf==2.3.0",
    "opencv-python-headless==4.10.0.84",
    "pandas==2.2.3",
    "pillow==12.2.0",
    "pycocotools==2.0.8; sys_platform != 'win32'",
    "pyyaml==6.0.2",
    "requests>=2.32.0",
    "scikit-learn==1.7.0",
    "scipy==1.15.1",
    "tabulate==0.9.0",
    "torch==2.9.0",
    "torchmetrics==1.7.4",
    "torchvision==0.24.0",
    "tqdm==4.66.5",
    "wandb==0.26.1",
]


def blue(message: str) -> None:
    """Print an English blue setup message."""
    print(Fore.BLUE + Style.BRIGHT + message)


def parse_driver_major(driver: str) -> int:
    """Return NVIDIA driver major version."""
    match = re.match(r"(\d+)", str(driver))
    return int(match.group(1)) if match else 0


def choose_torch_index(force: str | None = None) -> dict[str, str]:
    """Choose a PyTorch wheel index from local hardware."""
    if force:
        tag = force.lower().replace("cuda", "cu")
        if tag == "cpu":
            return {"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "tag": "cpu"}
        return {"name": f"pytorch-{tag}", "url": f"https://download.pytorch.org/whl/{tag}", "tag": tag}

    gpus = detect_nvidia()
    cuda_banner = detect_cuda_banner()
    if not gpus:
        return {"name": "pytorch-cpu", "url": "https://download.pytorch.org/whl/cpu", "tag": "cpu"}

    driver_major = max(parse_driver_major(str(gpu.get("driver_version", ""))) for gpu in gpus)
    if cuda_banner.startswith("13") or driver_major >= 580:
        tag = "cu130"
    elif cuda_banner >= "12.8" or driver_major >= 570:
        tag = "cu128"
    else:
        tag = "cu126"
    return {"name": f"pytorch-{tag}", "url": f"https://download.pytorch.org/whl/{tag}", "tag": tag}


def build_pyproject(torch_index: dict[str, str], region: str) -> str:
    """Render pyproject.toml."""
    dependencies = list(BASE_DEPENDENCIES)
    if region == "china":
        dependencies.append("modelscope>=1.22.0")
    deps = "\n".join(f'    "{dep}",' for dep in dependencies)
    default_index = ""
    if region == "china":
        default_index = """
[[tool.uv.index]]
name = "tsinghua-pypi"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
"""

    return f"""[project]
name = "dfine-seg-trainer"
version = "0.1.0"
description = "Config-first D-FINE-seg trainer with dataset conversion, rich augmentation controls, and reproducible outputs."
readme = "README.md"
requires-python = ">=3.11,<3.14"
dependencies = [
{deps}
]

[tool.uv]
package = false

[tool.uv.sources]
torch = [
    {{ index = "{torch_index["name"]}", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }},
]
torchvision = [
    {{ index = "{torch_index["name"]}", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }},
]

[[tool.uv.index]]
name = "{torch_index["name"]}"
url = "{torch_index["url"]}"
explicit = true
{default_index}
[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "ruff>=0.8.0",
]
"""


def write_setup_metadata(torch_index: dict[str, str], region: str, detected_region: dict[str, Any]) -> None:
    """Write environment selection metadata."""
    save_yaml(
        PROJECT_DIR / "config" / "environment_selection.yaml",
        {
            "region": region,
            "detected_ip": detected_region,
            "torch_index": torch_index,
            "platform": platform.platform(),
            "nvidia_gpus": detect_nvidia(),
            "nvidia_cuda_banner": detect_cuda_banner(),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Detect region/GPU and update pyproject.toml for the D-FINE-seg trainer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  uv run python setup_dfine_seg_env.py --dry-run
  uv run python setup_dfine_seg_env.py
  uv sync

  # Force a wheel family:
  uv run python setup_dfine_seg_env.py --torch-index cu130
  uv run python setup_dfine_seg_env.py --torch-index cpu
""",
    )
    parser.add_argument(
        "--region", choices=["auto", "china", "global"], default="auto", help="Package/model mirror region."
    )
    parser.add_argument("--torch-index", default=None, help="Force PyTorch wheel tag: cpu, cu126, cu128, cu130.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the selected environment without writing pyproject.toml."
    )
    return parser


def main() -> int:
    """Run environment setup."""
    args = build_parser().parse_args()
    detected = detect_ip_region()
    region = args.region
    if region == "auto":
        region = "china" if detected.get("country_code") == "CN" else "global"
    torch_index = choose_torch_index(args.torch_index)

    blue(f"Detected region: {region} (IP country={detected.get('country_code')}).")
    blue(f"Selected PyTorch wheel index: {torch_index['tag']} -> {torch_index['url']}.")
    for gpu in detect_nvidia():
        blue(
            f"GPU {gpu.get('index')}: {gpu.get('name')} ({gpu.get('memory_total_mib')} MiB, driver {gpu.get('driver_version')})."
        )

    if args.dry_run:
        blue("Dry run complete. pyproject.toml was not changed.")
        return 0

    PYPROJECT.write_text(build_pyproject(torch_index, region), encoding="utf-8")
    write_setup_metadata(torch_index, region, detected)
    blue(f"Updated {PYPROJECT}.")
    if region == "china":
        blue("China mirror mode enabled for PyPI. Model download will prefer ModelScope when configured.")
    blue("Next step: run `uv sync` from projects/dfine_seg_trainer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
