from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from dfine_seg_trainer.dataset_adapter import build_dataset_plan, materialize_dataset


def write_image(path: Path, size: tuple[int, int] = (80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 120, 200)).save(path)


def make_config(dataset_dir: Path, cache_root: Path, source_format: str, box_to_mask: bool = False) -> dict:
    return {
        "model": {"task": "segment", "name": "n"},
        "dataset": {
            "source_format": source_format,
            "dataset_dir": str(dataset_dir),
            "cache_root": str(cache_root),
            "split_ratio": [1, 1, 0],
            "split_seed": 0,
            "link_mode": "copy",
            "box_to_mask": box_to_mask,
            "names": [],
        },
    }


def test_labelme_conversion(tmp_path: Path) -> None:
    dataset = tmp_path / "labelme"
    write_image(dataset / "img1.jpg")
    (dataset / "img1.json").write_text(
        json.dumps(
            {
                "imagePath": "img1.jpg",
                "imageWidth": 80,
                "imageHeight": 60,
                "shapes": [{"label": "part", "points": [[10, 10], [50, 10], [50, 40], [10, 40]]}],
            }
        ),
        encoding="utf-8",
    )
    config = make_config(dataset, tmp_path / "cache", "labelme")
    prepared = materialize_dataset(build_dataset_plan(config), config, progress=False)
    train = json.loads((prepared.path / "train.json").read_text(encoding="utf-8"))
    val = json.loads((prepared.path / "val.json").read_text(encoding="utf-8"))
    assert len(train["images"]) + len(val["images"]) == 1
    assert prepared.class_names == ["part"]


def test_voc_conversion_with_rectangular_masks(tmp_path: Path) -> None:
    dataset = tmp_path / "voc"
    write_image(dataset / "JPEGImages" / "a.jpg")
    (dataset / "Annotations").mkdir(parents=True)
    (dataset / "Annotations" / "a.xml").write_text(
        """
<annotation>
  <filename>a.jpg</filename>
  <object>
    <name>box</name>
    <bndbox><xmin>5</xmin><ymin>6</ymin><xmax>40</xmax><ymax>45</ymax></bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )
    config = make_config(dataset, tmp_path / "cache", "pascal_voc", box_to_mask=True)
    prepared = materialize_dataset(build_dataset_plan(config), config, progress=False)
    train = json.loads((prepared.path / "train.json").read_text(encoding="utf-8"))
    val = json.loads((prepared.path / "val.json").read_text(encoding="utf-8"))
    annotations = train["annotations"] + val["annotations"]
    assert annotations[0]["segmentation"]
    assert prepared.class_names == ["box"]
