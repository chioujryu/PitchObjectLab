from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from dfine_seg_trainer.dataset_adapter import build_dataset_plan, materialize_dataset


def write_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 80, 40)).save(path)


def base_config(dataset_dir: Path, cache_root: Path, task: str = "segment") -> dict:
    return {
        "model": {"task": task, "name": "n"},
        "dataset": {
            "source_format": "auto",
            "dataset_dir": str(dataset_dir),
            "cache_root": str(cache_root),
            "split_ratio": [1, 1, 0],
            "split_seed": 0,
            "link_mode": "copy",
            "box_to_mask": False,
            "names": ["object"],
        },
    }


def test_yolo_seg_converts_to_coco_cache(tmp_path: Path) -> None:
    dataset = tmp_path / "yolo_seg"
    write_image(dataset / "images" / "train" / "a.jpg")
    write_image(dataset / "images" / "val" / "b.jpg")
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    (dataset / "labels" / "train" / "a.txt").write_text("0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n", encoding="utf-8")
    (dataset / "labels" / "val" / "b.txt").write_text("0 0.2 0.2 0.7 0.2 0.7 0.7 0.2 0.7\n", encoding="utf-8")
    (dataset / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [object]\n", encoding="utf-8")

    config = base_config(dataset, tmp_path / "cache")
    plan = build_dataset_plan(config)
    prepared = materialize_dataset(plan, config, progress=False)

    train = json.loads((prepared.path / "train.json").read_text(encoding="utf-8"))
    assert prepared.class_names == ["object"]
    assert train["images"]
    assert train["annotations"][0]["segmentation"]
    assert (prepared.path / "dataset_adapter_metadata.json").exists()


def test_box_only_segment_requires_box_to_mask(tmp_path: Path) -> None:
    dataset = tmp_path / "yolo_box"
    write_image(dataset / "images" / "train" / "a.jpg")
    write_image(dataset / "images" / "val" / "b.jpg")
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    (dataset / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (dataset / "labels" / "val" / "b.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (dataset / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [object]\n", encoding="utf-8")

    config = base_config(dataset, tmp_path / "cache")
    plan = build_dataset_plan(config)
    with pytest.raises(ValueError, match="box-only"):
        materialize_dataset(plan, config, progress=False)

    config["dataset"]["box_to_mask"] = True
    prepared = materialize_dataset(build_dataset_plan(config), config, progress=False)
    train = json.loads((prepared.path / "train.json").read_text(encoding="utf-8"))
    assert train["annotations"][0]["segmentation"]
