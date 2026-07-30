"""Dataset detection and conversion into a D-FINE-seg COCO-style cache."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from PIL import Image
from tqdm import tqdm

from .common import IMAGE_EXTENSIONS, PROJECT_DIR, json_safe, resolve_path, sanitize_name, write_json

SUPPORTED_FORMATS = {
    "auto",
    "dfine_coco",
    "coco",
    "coco_json",
    "roboflow_coco",
    "roboflow_yolo",
    "ultralytics_yolo",
    "yolo",
    "labelme",
    "labelme_json",
    "pascal_voc",
    "voc",
    "dota",
}


@dataclass
class DatasetPlan:
    """Prepared dataset conversion plan."""

    source_format: str
    source_dir: Path
    cache_dir: Path
    fingerprint: str
    direct_usable: bool
    needs_conversion: bool
    source_config: dict[str, Any]
    estimate: dict[str, Any]


@dataclass
class PreparedDataset:
    """Materialized dataset cache metadata."""

    path: Path
    source_format: str
    class_names: list[str]
    stats: dict[str, Any]
    metadata_path: Path


def normalize_format(value: Any) -> str:
    """Normalize dataset format aliases."""
    text = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "cocojson": "coco_json",
        "coco_instance": "coco_json",
        "roboflow": "roboflow_coco",
        "yolo_seg": "ultralytics_yolo",
        "yolo_detect": "ultralytics_yolo",
        "yolo": "ultralytics_yolo",
        "voc": "pascal_voc",
        "pascal": "pascal_voc",
        "labelme_json": "labelme",
    }
    return aliases.get(text, text)


def image_files(root: Path) -> list[Path]:
    """Return image files under a root, sorted for deterministic conversion."""
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def stat_digest(paths: Iterable[Path], root: Path) -> dict[str, Any]:
    """Build an O(n) stable file metadata digest for cache invalidation."""
    hasher = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(paths):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.resolve().relative_to(root.resolve()) if path.resolve().is_relative_to(root.resolve()) else path.name
        payload = f"{rel.as_posix()}|{int(stat.st_mtime_ns)}|{stat.st_size}\n"
        hasher.update(payload.encode("utf-8", errors="surrogateescape"))
        count += 1
        total += int(stat.st_size)
    return {"hash": hasher.hexdigest(), "count": count, "bytes": total}


def dataset_fingerprint(source_dir: Path, source_format: str, config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Create a reusable cache fingerprint."""
    relevant = []
    for suffix in ("*.yaml", "*.yml", "*.json", "*.txt", "*.xml"):
        relevant.extend(source_dir.rglob(suffix))
    relevant.extend(path for path in image_files(source_dir) if path.suffix.lower() == ".npy")
    digest = stat_digest(relevant, source_dir)
    payload = {
        "version": "dfine-seg-adapter-2026-05-25.1",
        "source_dir": str(source_dir.resolve()),
        "source_format": source_format,
        "digest": digest,
        "task": config.get("model", {}).get("task", "segment"),
        "box_to_mask": config.get("dataset", {}).get("box_to_mask", False),
        "split_ratio": config.get("dataset", {}).get("split_ratio", [0.8, 0.1, 0.1]),
        "split_seed": config.get("dataset", {}).get("split_seed", 0),
    }
    text = json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], payload


def find_yaml(source_dir: Path, configured: Any = "") -> Path | None:
    """Find a dataset YAML file."""
    if configured:
        path = resolve_path(configured, source_dir, must_exist=True)
        return path
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        candidate = source_dir / name
        if candidate.exists():
            return candidate.resolve()
    return None


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    """Load an Ultralytics/Roboflow YOLO data YAML."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must be a mapping: {path}")
    return data


def names_from_value(value: Any) -> list[str]:
    """Normalize class names from dict/list YAML forms."""
    if isinstance(value, Mapping):
        return [str(value[key]) for key in sorted(value, key=lambda x: int(x) if str(x).isdigit() else str(x))]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def configured_class_names(config: Mapping[str, Any]) -> list[str]:
    """Read class names from wrapper config."""
    names = config.get("dataset", {}).get("names")
    return names_from_value(names)


def detect_dataset_format(config: Mapping[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    """Detect the source dataset format."""
    dataset_cfg = config.get("dataset", {})
    source_dir = resolve_path(dataset_cfg.get("dataset_dir", ""), PROJECT_DIR, must_exist=True)
    requested = normalize_format(dataset_cfg.get("source_format", "auto"))
    if requested != "auto":
        if requested not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported dataset.source_format={requested!r}.")
        return requested, source_dir, {}

    if (source_dir / "train.json").exists() and (source_dir / "val.json").exists() and (source_dir / "images").exists():
        return "dfine_coco", source_dir, {}

    if any((source_dir / split / "_annotations.coco.json").exists() for split in ("train", "valid", "val", "test")):
        return "roboflow_coco", source_dir, {}

    data_yaml = find_yaml(source_dir, dataset_cfg.get("data_yaml", ""))
    if data_yaml:
        yaml_data = load_dataset_yaml(data_yaml)
        if any((source_dir / split / "labels").exists() for split in ("train", "valid", "val", "test")):
            return "roboflow_yolo", source_dir, {"data_yaml": str(data_yaml), "yaml": yaml_data}
        return "ultralytics_yolo", source_dir, {"data_yaml": str(data_yaml), "yaml": yaml_data}

    if dataset_cfg.get("coco_json") or any(
        (source_dir / name).exists() for name in ("annotations.json", "instances_train.json")
    ):
        return "coco_json", source_dir, {}

    if list(source_dir.rglob("*.xml")):
        return "pascal_voc", source_dir, {}

    if (source_dir / "labelTxt").exists() or any(path.name.lower() == "labeltxt" for path in source_dir.rglob("*")):
        return "dota", source_dir, {}

    labelme_jsons = [p for p in source_dir.rglob("*.json") if p.name not in {"train.json", "val.json", "test.json"}]
    if labelme_jsons:
        try:
            with labelme_jsons[0].open("r", encoding="utf-8") as file:
                sample = json.load(file)
            if isinstance(sample, dict) and "shapes" in sample and "imagePath" in sample:
                return "labelme", source_dir, {}
        except Exception:
            pass

    if (source_dir / "images").exists() and (source_dir / "labels").exists():
        return "ultralytics_yolo", source_dir, {}

    raise ValueError(f"Could not auto-detect dataset format under {source_dir}.")


def build_dataset_plan(config: Mapping[str, Any]) -> DatasetPlan:
    """Build a conversion plan without mutating the dataset cache."""
    source_format, source_dir, source_config = detect_dataset_format(config)
    fingerprint, payload = dataset_fingerprint(source_dir, source_format, config)
    cache_root_value = config.get("dataset", {}).get("cache_root", "dataset_cache")
    cache_root = resolve_path(cache_root_value, PROJECT_DIR, must_exist=False)
    cache_name = f"{source_format}_{sanitize_name(source_dir.name)}_{fingerprint}"
    cache_dir = cache_root / cache_name
    images = image_files(source_dir)
    total_bytes = sum(path.stat().st_size for path in images if path.exists())
    direct_usable = source_format == "dfine_coco"
    estimate = {
        "source_image_files": len(images),
        "source_image_bytes": total_bytes,
        "cache_dir": str(cache_dir),
        "fingerprint_payload": payload,
        "conversion_output_files_estimate": max(3, len(images) + 7),
    }
    return DatasetPlan(
        source_format=source_format,
        source_dir=source_dir,
        cache_dir=cache_dir,
        fingerprint=fingerprint,
        direct_usable=direct_usable,
        needs_conversion=not direct_usable or bool(config.get("dataset", {}).get("refresh_cache", False)),
        source_config=source_config,
        estimate=estimate,
    )


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """Place an image in the cache by hardlink, symlink, or copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"
    mode = str(mode or "auto").lower()
    if mode in {"auto", "hardlink"}:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    if mode in {"auto", "symlink"}:
        try:
            dst.symlink_to(src)
            return "symlink"
        except OSError:
            if mode == "symlink":
                raise
    shutil.copy2(src, dst)
    return "copy"


def image_size(path: Path) -> tuple[int, int]:
    """Read image dimensions as width, height."""
    with Image.open(path) as img:
        return img.size


def split_items(items: list[Any], ratios: list[float], seed: int) -> dict[str, list[Any]]:
    """Split items deterministically into train/val/test."""
    if len(ratios) != 3:
        raise ValueError("dataset.split_ratio must contain three values: train, val, test.")
    total = sum(float(x) for x in ratios)
    if total <= 0:
        raise ValueError("dataset.split_ratio must sum to a positive value.")
    normalized = [float(x) / total for x in ratios]
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * normalized[0])
    n_val = round(n * normalized[1])
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    if n >= 2 and not val:
        val = train[-1:]
        train = train[:-1]
    if not train and val:
        train = val[:1]
        val = val[1:]
    if not train and test:
        train = test[:1]
        test = test[1:]
    return {"train": train, "val": val, "test": test}


def yolo_split_image_dirs(source_dir: Path, yaml_data: Mapping[str, Any] | None) -> dict[str, list[Path]]:
    """Resolve YOLO split image paths from data YAML or common folders."""
    yaml_data = yaml_data or {}
    base = source_dir
    if yaml_data.get("path"):
        path_text = str(yaml_data["path"])
        path = Path(path_text)
        base = path if path.is_absolute() else (source_dir / path)

    def resolve_split(value: Any, default_names: list[str]) -> list[Path]:
        candidates: list[Path] = []
        raw_items = value if isinstance(value, list) else [value] if value else []
        for item in raw_items:
            p = Path(str(item))
            candidates.append(p if p.is_absolute() else (base / p))
        for name in default_names:
            candidates.append(source_dir / name)
            candidates.append(source_dir / "images" / name)
        found: list[Path] = []
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() == ".txt":
                with candidate.open("r", encoding="utf-8") as file:
                    found.extend(Path(line.strip()) for line in file if line.strip())
            elif candidate.exists():
                found.extend(image_files(candidate))
        return sorted(path.resolve() for path in found if path.exists())

    return {
        "train": resolve_split(yaml_data.get("train"), ["train"]),
        "val": resolve_split(yaml_data.get("val") or yaml_data.get("valid"), ["val", "valid"]),
        "test": resolve_split(yaml_data.get("test"), ["test"]),
    }


def yolo_label_for_image(image_path: Path, source_dir: Path) -> Path | None:
    """Find the YOLO label file for an image."""
    candidates = []
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        rel_after_images = Path(*parts[idx + 1 :])
        candidates.append(source_dir / "labels" / rel_after_images.with_suffix(".txt"))
        parent = image_path.parent
        if parent.name in {"train", "val", "valid", "test"}:
            candidates.append(parent.parent / "labels" / parent.name / f"{image_path.stem}.txt")
            candidates.append(parent.parent.parent / "labels" / parent.name / f"{image_path.stem}.txt")
    candidates.append(source_dir / "labels" / f"{image_path.stem}.txt")
    candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def bbox_to_segmentation(x: float, y: float, w: float, h: float) -> list[float]:
    """Create a rectangular COCO polygon from a bbox."""
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def validate_box_to_mask(task: str, has_polygon: bool, box_to_mask: bool, label_path: Path) -> None:
    """Prevent accidental fake segmentation training."""
    if task == "segment" and not has_polygon and not box_to_mask:
        raise ValueError(
            "The dataset contains box-only annotations but model.task=segment. "
            f"Set dataset.box_to_mask=true to train with rectangular masks, or use task=detect. First file: {label_path}"
        )


def convert_yolo(
    plan: DatasetPlan,
    config: Mapping[str, Any],
    yaml_data: Mapping[str, Any] | None,
    progress: bool,
) -> PreparedDataset:
    """Convert YOLO detect/seg into D-FINE COCO-style cache."""
    dataset_cfg = config.get("dataset", {})
    task = str(config.get("model", {}).get("task", "segment")).lower()
    box_to_mask = bool(dataset_cfg.get("box_to_mask", False))
    link_mode = str(dataset_cfg.get("link_mode", "auto"))
    names = configured_class_names(config) or names_from_value((yaml_data or {}).get("names"))
    split_ratio = list(dataset_cfg.get("split_ratio", [0.8, 0.1, 0.1]))
    split_seed = int(dataset_cfg.get("split_seed", 0))
    split_images = yolo_split_image_dirs(plan.source_dir, yaml_data)
    if not split_images["train"]:
        all_images = image_files(plan.source_dir / "images") or image_files(plan.source_dir)
        split_images = split_items(all_images, split_ratio, split_seed)

    plan.cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {"splits": {}, "link_methods": {}, "annotations": 0, "box_to_mask_annotations": 0}
    category_ids: set[int] = set()

    for split_name, images in split_images.items():
        coco = {"images": [], "annotations": [], "categories": []}
        ann_id = 1
        iterator = tqdm(images, desc=f"Converting {split_name}", unit="img", disable=not progress)
        for img_id, image_path in enumerate(iterator, start=1):
            if not image_path.exists():
                continue
            width, height = image_size(image_path)
            rel_name = f"{split_name}/{sanitize_name(image_path.stem)}{image_path.suffix.lower()}"
            placed = link_or_copy(image_path, plan.cache_dir / "images" / rel_name, link_mode)
            stats["link_methods"][placed] = stats["link_methods"].get(placed, 0) + 1
            coco["images"].append({"id": img_id, "file_name": rel_name, "width": width, "height": height})

            label_path = yolo_label_for_image(image_path, plan.source_dir)
            if label_path is None or not label_path.exists() or label_path.stat().st_size == 0:
                continue
            with label_path.open("r", encoding="utf-8") as file:
                for line_no, raw in enumerate(file, start=1):
                    parts = raw.strip().split()
                    if not parts:
                        continue
                    values = [float(x) for x in parts]
                    class_id = int(values[0])
                    coords = values[1:]
                    has_polygon = len(coords) >= 6
                    validate_box_to_mask(task, has_polygon, box_to_mask, label_path)
                    if has_polygon:
                        if len(coords) % 2:
                            coords = coords[:-1]
                        xs = coords[0::2]
                        ys = coords[1::2]
                        x1 = max(min(xs) * width, 0.0)
                        y1 = max(min(ys) * height, 0.0)
                        x2 = min(max(xs) * width, float(width))
                        y2 = min(max(ys) * height, float(height))
                        segmentation = [[v * (width if i % 2 == 0 else height) for i, v in enumerate(coords)]]
                    elif len(coords) == 4:
                        xc, yc, bw, bh = coords
                        x1 = (xc - bw / 2.0) * width
                        y1 = (yc - bh / 2.0) * height
                        x2 = (xc + bw / 2.0) * width
                        y2 = (yc + bh / 2.0) * height
                        segmentation = [bbox_to_segmentation(x1, y1, x2 - x1, y2 - y1)] if box_to_mask else []
                        if box_to_mask:
                            stats["box_to_mask_annotations"] += 1
                    else:
                        raise ValueError(f"Invalid YOLO line at {label_path}:{line_no}: {raw.strip()}")

                    x1 = max(0.0, min(x1, float(width)))
                    y1 = max(0.0, min(y1, float(height)))
                    x2 = max(0.0, min(x2, float(width)))
                    y2 = max(0.0, min(y2, float(height)))
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    if bw <= 0 or bh <= 0:
                        continue
                    category_ids.add(class_id)
                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [x1, y1, bw, bh],
                            "area": bw * bh,
                            "segmentation": segmentation,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
                    stats["annotations"] += 1
        stats["splits"][split_name] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}
        write_json(plan.cache_dir / f"{split_name}.json", coco)

    if not names:
        max_id = max(category_ids) if category_ids else 0
        names = [f"class_{idx}" for idx in range(max_id + 1)]
    write_categories(plan.cache_dir, names)
    inject_categories(plan.cache_dir, names)
    return finish_prepared(plan, names, stats)


def write_categories(cache_dir: Path, names: list[str]) -> None:
    """Write class map files."""
    categories = [{"id": idx, "name": name, "supercategory": "object"} for idx, name in enumerate(names)]
    write_json(cache_dir / "categories.json", categories)
    write_json(cache_dir / "class_map.json", {idx: name for idx, name in enumerate(names)})


def inject_categories(cache_dir: Path, names: list[str]) -> None:
    """Inject category lists into every split JSON."""
    categories = [{"id": idx, "name": name, "supercategory": "object"} for idx, name in enumerate(names)]
    for split in ("train", "val", "test"):
        path = cache_dir / f"{split}.json"
        if not path.exists():
            write_json(path, {"images": [], "annotations": [], "categories": categories})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["categories"] = categories
        write_json(path, data)


def finish_prepared(plan: DatasetPlan, names: list[str], stats: dict[str, Any]) -> PreparedDataset:
    """Write cache metadata and return prepared dataset info."""
    metadata = {
        "source_format": plan.source_format,
        "source_dir": str(plan.source_dir),
        "cache_dir": str(plan.cache_dir),
        "fingerprint": plan.fingerprint,
        "class_names": names,
        "stats": stats,
    }
    write_json(plan.cache_dir / "dataset_adapter_metadata.json", metadata)
    write_json(plan.cache_dir / "source_fingerprint.json", plan.estimate.get("fingerprint_payload", {}))
    return PreparedDataset(
        path=plan.cache_dir,
        source_format=plan.source_format,
        class_names=names,
        stats=stats,
        metadata_path=plan.cache_dir / "dataset_adapter_metadata.json",
    )


def convert_roboflow_coco(plan: DatasetPlan, config: Mapping[str, Any], progress: bool) -> PreparedDataset:
    """Convert Roboflow COCO split folders into flat D-FINE COCO layout."""
    dataset_cfg = config.get("dataset", {})
    link_mode = str(dataset_cfg.get("link_mode", "auto"))
    split_aliases = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}
    stats = {"splits": {}, "link_methods": {}, "annotations": 0}
    names: list[str] = configured_class_names(config)
    plan.cache_dir.mkdir(parents=True, exist_ok=True)
    for out_split, aliases in split_aliases.items():
        src_split = next((plan.source_dir / alias for alias in aliases if (plan.source_dir / alias).exists()), None)
        if src_split is None:
            write_json(plan.cache_dir / f"{out_split}.json", {"images": [], "annotations": [], "categories": []})
            continue
        ann_path = src_split / "_annotations.coco.json"
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if not names:
            names = [cat["name"] for cat in sorted(data.get("categories", []), key=lambda c: c["id"])]
        for image in tqdm(data.get("images", []), desc=f"Converting {out_split}", unit="img", disable=not progress):
            src_img = src_split / image["file_name"]
            if not src_img.exists():
                src_img = src_split / "images" / image["file_name"]
            rel_name = (
                f"{out_split}/{sanitize_name(Path(image['file_name']).stem)}{Path(image['file_name']).suffix.lower()}"
            )
            image["file_name"] = rel_name
            if src_img.exists():
                placed = link_or_copy(src_img, plan.cache_dir / "images" / rel_name, link_mode)
                stats["link_methods"][placed] = stats["link_methods"].get(placed, 0) + 1
        stats["splits"][out_split] = {
            "images": len(data.get("images", [])),
            "annotations": len(data.get("annotations", [])),
        }
        stats["annotations"] += len(data.get("annotations", []))
        write_json(plan.cache_dir / f"{out_split}.json", data)
    if not names:
        names = ["class_0"]
    inject_categories(plan.cache_dir, names)
    write_categories(plan.cache_dir, names)
    return finish_prepared(plan, names, stats)


def convert_coco_json(plan: DatasetPlan, config: Mapping[str, Any], progress: bool) -> PreparedDataset:
    """Convert already-COCO datasets into D-FINE split JSON files."""
    dataset_cfg = config.get("dataset", {})
    link_mode = str(dataset_cfg.get("link_mode", "auto"))
    image_root = (
        resolve_path(dataset_cfg.get("image_dir"), PROJECT_DIR, must_exist=True)
        if dataset_cfg.get("image_dir")
        else plan.source_dir / "images"
    )
    split_paths = {
        "train": dataset_cfg.get("coco_train_json") or dataset_cfg.get("coco_json") or plan.source_dir / "train.json",
        "val": dataset_cfg.get("coco_val_json") or plan.source_dir / "val.json",
        "test": dataset_cfg.get("coco_test_json") or plan.source_dir / "test.json",
    }
    stats = {"splits": {}, "link_methods": {}, "annotations": 0}
    names = configured_class_names(config)
    plan.cache_dir.mkdir(parents=True, exist_ok=True)
    for split, value in split_paths.items():
        path = Path(value) if isinstance(value, Path) else resolve_path(value, plan.source_dir, must_exist=False)
        if not path.exists():
            write_json(plan.cache_dir / f"{split}.json", {"images": [], "annotations": [], "categories": []})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not names:
            names = [cat["name"] for cat in sorted(data.get("categories", []), key=lambda c: c["id"])]
        for image in tqdm(data.get("images", []), desc=f"Converting {split}", unit="img", disable=not progress):
            src_img = image_root / image["file_name"]
            rel_name = f"{split}/{Path(image['file_name']).name}"
            image["file_name"] = rel_name
            if src_img.exists():
                placed = link_or_copy(src_img, plan.cache_dir / "images" / rel_name, link_mode)
                stats["link_methods"][placed] = stats["link_methods"].get(placed, 0) + 1
        stats["splits"][split] = {
            "images": len(data.get("images", [])),
            "annotations": len(data.get("annotations", [])),
        }
        stats["annotations"] += len(data.get("annotations", []))
        write_json(plan.cache_dir / f"{split}.json", data)
    if not names:
        names = ["class_0"]
    inject_categories(plan.cache_dir, names)
    write_categories(plan.cache_dir, names)
    return finish_prepared(plan, names, stats)


def convert_labelme(plan: DatasetPlan, config: Mapping[str, Any], progress: bool) -> PreparedDataset:
    """Convert LabelMe polygon JSON files."""
    json_files = sorted(
        path for path in plan.source_dir.rglob("*.json") if path.name not in {"train.json", "val.json", "test.json"}
    )
    dataset_cfg = config.get("dataset", {})
    split_ratio = list(dataset_cfg.get("split_ratio", [0.8, 0.1, 0.1]))
    split_seed = int(dataset_cfg.get("split_seed", 0))
    link_mode = str(dataset_cfg.get("link_mode", "auto"))
    splits = split_items(json_files, split_ratio, split_seed)
    name_to_id: dict[str, int] = {}
    stats = {"splits": {}, "link_methods": {}, "annotations": 0}
    plan.cache_dir.mkdir(parents=True, exist_ok=True)
    for split, files in splits.items():
        coco = {"images": [], "annotations": [], "categories": []}
        ann_id = 1
        for img_id, label_path in enumerate(
            tqdm(files, desc=f"Converting {split}", unit="json", disable=not progress), start=1
        ):
            data = json.loads(label_path.read_text(encoding="utf-8"))
            image_path = (label_path.parent / data["imagePath"]).resolve()
            if not image_path.exists():
                continue
            width, height = image_size(image_path)
            rel_name = f"{split}/{sanitize_name(image_path.stem)}{image_path.suffix.lower()}"
            placed = link_or_copy(image_path, plan.cache_dir / "images" / rel_name, link_mode)
            stats["link_methods"][placed] = stats["link_methods"].get(placed, 0) + 1
            coco["images"].append({"id": img_id, "file_name": rel_name, "width": width, "height": height})
            for shape in data.get("shapes", []):
                points = shape.get("points") or []
                if len(points) < 2:
                    continue
                label = str(shape.get("label", "object"))
                class_id = name_to_id.setdefault(label, len(name_to_id))
                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]
                x1, y1, x2, y2 = max(min(xs), 0.0), max(min(ys), 0.0), min(max(xs), width), min(max(ys), height)
                if x2 <= x1 or y2 <= y1:
                    continue
                segmentation = [[coord for point in points for coord in (float(point[0]), float(point[1]))]]
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "segmentation": segmentation,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
                stats["annotations"] += 1
        stats["splits"][split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}
        write_json(plan.cache_dir / f"{split}.json", coco)
    names = (
        configured_class_names(config)
        or [name for name, _ in sorted(name_to_id.items(), key=lambda x: x[1])]
        or ["class_0"]
    )
    inject_categories(plan.cache_dir, names)
    write_categories(plan.cache_dir, names)
    return finish_prepared(plan, names, stats)


def convert_voc(plan: DatasetPlan, config: Mapping[str, Any], progress: bool) -> PreparedDataset:
    """Convert Pascal VOC XML files."""
    dataset_cfg = config.get("dataset", {})
    task = str(config.get("model", {}).get("task", "segment")).lower()
    box_to_mask = bool(dataset_cfg.get("box_to_mask", False))
    if task == "segment" and not box_to_mask:
        raise ValueError("Pascal VOC is box-only. Set dataset.box_to_mask=true or use model.task=detect.")
    xml_files = sorted(plan.source_dir.rglob("*.xml"))
    splits = split_items(
        xml_files, list(dataset_cfg.get("split_ratio", [0.8, 0.1, 0.1])), int(dataset_cfg.get("split_seed", 0))
    )
    name_to_id: dict[str, int] = {}
    stats = {"splits": {}, "link_methods": {}, "annotations": 0, "box_to_mask_annotations": 0}
    plan.cache_dir.mkdir(parents=True, exist_ok=True)
    for split, files in splits.items():
        coco = {"images": [], "annotations": [], "categories": []}
        ann_id = 1
        for img_id, xml_path in enumerate(
            tqdm(files, desc=f"Converting {split}", unit="xml", disable=not progress), start=1
        ):
            root = ET.parse(xml_path).getroot()
            filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
            image_path = next(
                (
                    p
                    for p in [
                        xml_path.parent / filename,
                        plan.source_dir / "JPEGImages" / filename,
                        plan.source_dir / "images" / filename,
                    ]
                    if p.exists()
                ),
                None,
            )
            if image_path is None:
                continue
            width, height = image_size(image_path)
            rel_name = f"{split}/{sanitize_name(image_path.stem)}{image_path.suffix.lower()}"
            placed = link_or_copy(
                image_path, plan.cache_dir / "images" / rel_name, str(dataset_cfg.get("link_mode", "auto"))
            )
            stats["link_methods"][placed] = stats["link_methods"].get(placed, 0) + 1
            coco["images"].append({"id": img_id, "file_name": rel_name, "width": width, "height": height})
            for obj in root.findall("object"):
                label = obj.findtext("name") or "object"
                class_id = name_to_id.setdefault(label, len(name_to_id))
                box = obj.find("bndbox")
                if box is None:
                    continue
                x1 = float(box.findtext("xmin", "0"))
                y1 = float(box.findtext("ymin", "0"))
                x2 = float(box.findtext("xmax", "0"))
                y2 = float(box.findtext("ymax", "0"))
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "segmentation": [bbox_to_segmentation(x1, y1, x2 - x1, y2 - y1)] if box_to_mask else [],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
                stats["annotations"] += 1
                stats["box_to_mask_annotations"] += int(box_to_mask)
        stats["splits"][split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}
        write_json(plan.cache_dir / f"{split}.json", coco)
    names = (
        configured_class_names(config)
        or [name for name, _ in sorted(name_to_id.items(), key=lambda x: x[1])]
        or ["class_0"]
    )
    inject_categories(plan.cache_dir, names)
    write_categories(plan.cache_dir, names)
    return finish_prepared(plan, names, stats)


def convert_dota(plan: DatasetPlan, config: Mapping[str, Any], progress: bool) -> PreparedDataset:
    """Convert DOTA oriented boxes to axis-aligned COCO boxes with polygon segmentation."""
    dataset_cfg = config.get("dataset", {})
    label_root = next(
        (p for p in [plan.source_dir / "labelTxt", plan.source_dir / "labels"] if p.exists()), plan.source_dir
    )
    label_files = sorted(label_root.rglob("*.txt"))
    splits = split_items(
        label_files, list(dataset_cfg.get("split_ratio", [0.8, 0.1, 0.1])), int(dataset_cfg.get("split_seed", 0))
    )
    name_to_id: dict[str, int] = {}
    stats = {"splits": {}, "link_methods": {}, "annotations": 0}
    plan.cache_dir.mkdir(parents=True, exist_ok=True)
    for split, files in splits.items():
        coco = {"images": [], "annotations": [], "categories": []}
        ann_id = 1
        for img_id, label_path in enumerate(
            tqdm(files, desc=f"Converting {split}", unit="txt", disable=not progress), start=1
        ):
            image_path = next(
                (
                    p
                    for ext in IMAGE_EXTENSIONS
                    for p in [
                        plan.source_dir / "images" / f"{label_path.stem}{ext}",
                        plan.source_dir / f"{label_path.stem}{ext}",
                    ]
                    if p.exists()
                ),
                None,
            )
            if image_path is None:
                continue
            width, height = image_size(image_path)
            rel_name = f"{split}/{sanitize_name(image_path.stem)}{image_path.suffix.lower()}"
            placed = link_or_copy(
                image_path, plan.cache_dir / "images" / rel_name, str(dataset_cfg.get("link_mode", "auto"))
            )
            stats["link_methods"][placed] = stats["link_methods"].get(placed, 0) + 1
            coco["images"].append({"id": img_id, "file_name": rel_name, "width": width, "height": height})
            for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = raw.strip().split()
                if len(parts) < 9:
                    continue
                coords = [float(x) for x in parts[:8]]
                label = parts[8]
                class_id = name_to_id.setdefault(label, len(name_to_id))
                xs, ys = coords[0::2], coords[1::2]
                x1, y1, x2, y2 = max(min(xs), 0.0), max(min(ys), 0.0), min(max(xs), width), min(max(ys), height)
                if x2 <= x1 or y2 <= y1:
                    continue
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "segmentation": [coords],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
                stats["annotations"] += 1
        stats["splits"][split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}
        write_json(plan.cache_dir / f"{split}.json", coco)
    names = (
        configured_class_names(config)
        or [name for name, _ in sorted(name_to_id.items(), key=lambda x: x[1])]
        or ["class_0"]
    )
    inject_categories(plan.cache_dir, names)
    write_categories(plan.cache_dir, names)
    return finish_prepared(plan, names, stats)


def materialize_dataset(plan: DatasetPlan, config: Mapping[str, Any], progress: bool = True) -> PreparedDataset:
    """Create or reuse the D-FINE COCO-style cache."""
    dataset_cfg = config.get("dataset", {})
    refresh = bool(dataset_cfg.get("refresh_cache", False))
    metadata_path = plan.cache_dir / "dataset_adapter_metadata.json"
    if metadata_path.exists() and not refresh:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return PreparedDataset(
            path=plan.cache_dir,
            source_format=metadata.get("source_format", plan.source_format),
            class_names=list(metadata.get("class_names", [])),
            stats=dict(metadata.get("stats", {})),
            metadata_path=metadata_path,
        )
    if plan.cache_dir.exists() and refresh:
        shutil.rmtree(plan.cache_dir)

    fmt = normalize_format(plan.source_format)
    if fmt == "dfine_coco":
        copied_plan = DatasetPlan(
            source_format="coco_json",
            source_dir=plan.source_dir,
            cache_dir=plan.cache_dir,
            fingerprint=plan.fingerprint,
            direct_usable=False,
            needs_conversion=True,
            source_config=plan.source_config,
            estimate=plan.estimate,
        )
        return convert_coco_json(copied_plan, config, progress)
    if fmt in {"ultralytics_yolo", "roboflow_yolo"}:
        yaml_data = plan.source_config.get("yaml") if plan.source_config else None
        if yaml_data is None:
            data_yaml = find_yaml(plan.source_dir, dataset_cfg.get("data_yaml", ""))
            yaml_data = load_dataset_yaml(data_yaml) if data_yaml else {}
        return convert_yolo(plan, config, yaml_data, progress)
    if fmt == "roboflow_coco":
        return convert_roboflow_coco(plan, config, progress)
    if fmt in {"coco", "coco_json"}:
        return convert_coco_json(plan, config, progress)
    if fmt == "labelme":
        return convert_labelme(plan, config, progress)
    if fmt == "pascal_voc":
        return convert_voc(plan, config, progress)
    if fmt == "dota":
        return convert_dota(plan, config, progress)
    raise ValueError(f"Unsupported normalized dataset format: {fmt}")
