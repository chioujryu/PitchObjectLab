"""
Configure this project's uv environment for CPU or GPU PyTorch.

Usage:
    uv run python scripts/setup_pytorch_uv.py --dry-run
    uv run python scripts/setup_pytorch_uv.py --yes
    uv run python scripts/setup_pytorch_uv.py --device cpu --yes
    uv run python scripts/setup_pytorch_uv.py --cuda-tag cu128 --region cn --yes

The script checks the public IP region before choosing package indexes. China,
Hong Kong, Macau, and Taiwan default to PyTorch/PyPI mirrors; other regions use
the official PyTorch wheel index. It updates pyproject.toml so future runs only
need `uv sync`.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

PROJECT_DIR = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_DIR / "pyproject.toml"
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
GREATER_CHINA_REGIONS = {"CN", "HK", "MO", "TW"}
PYTORCH_INDEXES = {
    "official": {
        "cpu": "https://download.pytorch.org/whl/cpu",
        "cu118": "https://download.pytorch.org/whl/cu118",
        "cu124": "https://download.pytorch.org/whl/cu124",
        "cu126": "https://download.pytorch.org/whl/cu126",
        "cu128": "https://download.pytorch.org/whl/cu128",
    },
    "china": {
        "cpu": "https://mirrors.aliyun.com/pytorch-wheels/cpu/",
        "cu118": "https://mirrors.aliyun.com/pytorch-wheels/cu118/",
        "cu124": "https://mirrors.aliyun.com/pytorch-wheels/cu124/",
        "cu126": "https://mirrors.aliyun.com/pytorch-wheels/cu126/",
        "cu128": "https://mirrors.aliyun.com/pytorch-wheels/cu128/",
    },
}
PYPI_INDEXES = {
    "official": "https://pypi.org/simple",
    "china": "https://pypi.tuna.tsinghua.edu.cn/simple/",
}


def blue(message: str) -> None:
    """Print a blue status message."""
    print(Fore.BLUE + Style.BRIGHT + message)


def fetch_json(url: str, timeout: float = 6.0) -> Dict[str, Any]:
    """Fetch JSON from a public IP location API."""
    request = urllib.request.Request(url, headers={"User-Agent": "rf-detr-trainer-setup/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def detect_region() -> Tuple[str, str]:
    """Return country/region code and provider label from public IP services."""
    providers = (
        ("ipinfo", "https://ipinfo.io/json", "country"),
        ("ip-api", "http://ip-api.com/json/?fields=status,countryCode,query", "countryCode"),
    )
    for provider, url, key in providers:
        try:
            data = fetch_json(url)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            continue
        value = str(data.get(key) or "").upper()
        if value:
            return value, provider
    return "UNKNOWN", "unavailable"


def detect_cuda_version() -> Optional[str]:
    """Return CUDA version reported by nvidia-smi, or None when unavailable."""
    try:
        result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stdout + result.stderr
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return match.group(1) if match else None


def choose_cuda_tag(cuda_version: Optional[str], forced: Optional[str], device: str) -> str:
    """Choose the PyTorch wheel CUDA tag from hardware and user preference."""
    if forced:
        return forced
    if device == "cpu" or platform.system().lower() == "darwin":
        return "cpu"
    if not cuda_version:
        return "cpu"
    major, minor = (int(part) for part in cuda_version.split(".", 1))
    version = major + minor / 10.0
    if version >= 12.8:
        return "cu128"
    if version >= 12.6:
        return "cu126"
    if version >= 12.4:
        return "cu124"
    if version >= 11.8:
        return "cu118"
    return "cpu"


def index_name(cuda_tag: str) -> str:
    """Return the uv index name for a PyTorch wheel tag."""
    return f"pytorch-{cuda_tag}"


def update_dependency_line(text: str, package: str, version: str) -> str:
    """Replace or insert a dependency line in pyproject.toml dependencies."""
    pattern = rf'(?m)^    "{re.escape(package)}[^"]*",\s*$'
    replacement = f'    "{package}=={version}",'
    if re.search(pattern, text):
        return re.sub(pattern, replacement, text)
    marker = "dependencies = [\n"
    if marker not in text:
        raise ValueError("Could not find [project] dependencies list in pyproject.toml.")
    return text.replace(marker, marker + replacement + "\n", 1)


def update_uv_sources(text: str, cuda_tag: str, url: str) -> str:
    """Rewrite the uv source/index block for torch and torchvision."""
    name = index_name(cuda_tag)
    marker = "sys_platform == 'linux' or sys_platform == 'win32'"
    block = f"""
[tool.uv.sources]
torch = [
    {{ index = "{name}", marker = "{marker}" }},
]
torchvision = [
    {{ index = "{name}", marker = "{marker}" }},
]

[[tool.uv.index]]
name = "{name}"
url = "{url}"
explicit = true
"""
    pattern = r"\n\[tool\.uv\.sources\][\s\S]*?(?=\n\[dependency-groups\])"
    if re.search(pattern, text):
        return re.sub(pattern, "\n" + block.strip() + "\n", text)
    dependency_groups = "\n[dependency-groups]"
    if dependency_groups not in text:
        return text.rstrip() + "\n\n" + block.strip() + "\n"
    return text.replace(dependency_groups, "\n" + block.strip() + "\n" + dependency_groups, 1)


def update_pyproject(cuda_tag: str, index_url: str) -> None:
    """Update pyproject.toml so future uv sync uses the selected PyTorch wheels."""
    text = PYPROJECT.read_text(encoding="utf-8")
    text = update_dependency_line(text, "torch", TORCH_VERSION)
    text = update_dependency_line(text, "torchvision", TORCHVISION_VERSION)
    text = update_uv_sources(text, cuda_tag, index_url)
    PYPROJECT.write_text(text, encoding="utf-8")


def run_uv_sync(index_url: str, default_index_url: str) -> None:
    """Run uv sync with the selected PyTorch index available during resolution."""
    command = [
        "uv",
        "sync",
        "--default-index",
        default_index_url,
        "--index",
        index_url,
        "--index-strategy",
        "unsafe-best-match",
    ]
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-configure uv PyTorch wheels for this RF-DETR project.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Force CPU, CUDA, or auto detection.")
    parser.add_argument("--cuda-tag", choices=["cpu", "cu118", "cu124", "cu126", "cu128"], help="Override the selected PyTorch wheel tag.")
    parser.add_argument("--region", choices=["auto", "official", "china", "cn"], default="auto", help="Force official or China mirror indexes.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected setup without changing files.")
    parser.add_argument("--no-sync", action="store_true", help="Update pyproject.toml but do not run uv sync.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    args = parser.parse_args()

    country, provider = detect_region()
    region = "china" if country in GREATER_CHINA_REGIONS else "official"
    if args.region in {"china", "cn"}:
        region = "china"
    elif args.region == "official":
        region = "official"

    cuda_version = detect_cuda_version()
    cuda_tag = choose_cuda_tag(cuda_version, args.cuda_tag, args.device)
    index_url = PYTORCH_INDEXES[region][cuda_tag]
    default_index_url = PYPI_INDEXES[region]
    plan = {
        "project_dir": str(PROJECT_DIR),
        "pyproject": str(PYPROJECT),
        "ip_country": country,
        "ip_provider": provider,
        "download_region": region,
        "detected_cuda": cuda_version,
        "selected_cuda_tag": cuda_tag,
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "pypi_index": default_index_url,
        "pytorch_index": index_url,
        "will_run_uv_sync": not args.no_sync,
    }
    blue("PyTorch uv setup plan:")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.dry_run:
        return 0
    if not args.yes:
        answer = input(Fore.BLUE + Style.BRIGHT + "Update pyproject.toml and install with uv? [y/N]: " + Style.RESET_ALL).strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Cancelled before changing pyproject.toml.")

    update_pyproject(cuda_tag, index_url)
    if not args.no_sync:
        run_uv_sync(index_url, default_index_url)
    blue("PyTorch uv setup complete. Future installs can use: uv sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
