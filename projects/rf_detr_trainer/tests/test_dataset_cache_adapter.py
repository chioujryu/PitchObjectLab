from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import train_rf_detr_model as trainer

TEMP_ROOT = Path(os.environ.get("RF_DETR_TEST_TMP", r"C:\tmp" if os.name == "nt" else "/tmp/rf_detr_trainer_tests"))
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def make_image(path: Path, size=(100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 64, 32)).save(path)


def base_config(dataset_dir: Path, cache_root: Path, source_format: str = "auto") -> dict:
    return {
        "dataset": {
            "source_format": source_format,
            "dataset_dir": str(dataset_dir),
            "cache_root": str(cache_root),
            "link_mode": "copy",
            "split_ratio": [8, 1, 1],
            "split_seed": 0,
            "refresh_cache": False,
        },
        "train": {"dataset_file": "roboflow"},
    }


def materialize(config: dict, workdir: Path) -> tuple[dict, Path]:
    plan = trainer.build_dataset_plan(config, workdir / "run", None)
    metadata = trainer.materialize_dataset_plan(plan, config, workdir / "run", verbose=False)
    return metadata, Path(config["train"]["dataset_dir"])


def split_count(cache_dir: Path, split: str) -> int:
    data = json.loads((cache_dir / split / "_annotations.coco.json").read_text(encoding="utf-8"))
    return len(data["images"])


def annotation_count(cache_dir: Path, split: str) -> int:
    data = json.loads((cache_dir / split / "_annotations.coco.json").read_text(encoding="utf-8"))
    return len(data["annotations"])


class DatasetCacheAdapterTest(unittest.TestCase):
    def test_temporal_plan_and_estimate_apply_micro_window_limits(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp) / "temporal"
            root.mkdir(parents=True)
            data_yaml = root / "dataset.yaml"
            data_yaml.write_text(
                "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [ball]\n",
                encoding="utf-8",
            )
            config = {
                "model": {
                    "size": "small",
                    "motion": {
                        "enabled": True,
                        "type": "tracknet_v5",
                        "temporal": {
                            "mode": "real",
                            "num_frames": 3,
                            "frame_stride": 1,
                        },
                    },
                },
                "dataset": {
                    "source_format": "spatiotemporal_yolo",
                    "dataset_dir": str(root),
                    "data_yaml": str(data_yaml),
                    "temporal": {
                        "max_windows_per_split": {
                            "train": 1,
                            "val": 1,
                            "test": 1,
                        }
                    },
                },
                "train": {
                    "dataset_dir": str(root),
                    "epochs": 1,
                    "batch_size": 1,
                    "checkpoint_interval": 1,
                },
                "periodic_test": {"enabled": False, "run_final_test": False},
            }

            with patch(
                "rf_detr_temporal_data.temporal_split_window_counts",
                return_value={"train": 20, "val": 10, "test": 10},
            ):
                plan = trainer.build_dataset_plan(config, Path(temp) / "run", None)
            estimate = trainer.estimate_outputs(
                config,
                Path(temp) / "run",
                periodic_count=0,
                dataset_plan=plan,
            )

            self.assertEqual(plan["split_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(
                plan["complete_split_counts"],
                {"train": 20, "val": 10, "test": 10},
            )
            self.assertEqual(
                estimate["split_window_counts"],
                {"train": 1, "val": 1, "test": 1},
            )
            self.assertEqual(
                estimate["complete_temporal_window_counts"],
                {"train": 20, "val": 10, "test": 10},
            )
            self.assertEqual(estimate["runtime_units"], 1.0)

    def test_ultralytics_yolo_auto_detects_and_caches(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp) / "yolo"
            cache_root = Path(temp) / "cache"
            for split in ("train", "val", "test"):
                image = root / "images" / split / f"{split}.jpg"
                make_image(image)
                label_dir = root / "labels" / split
                label_dir.mkdir(parents=True, exist_ok=True)
                (label_dir / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (root / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [ball]\n",
                encoding="utf-8",
            )
            config = base_config(root, cache_root, source_format="auto")
            config["dataset"]["data_yaml"] = str(root / "data.yaml")

            metadata, cache_dir = materialize(config, Path(temp))

            self.assertEqual(metadata["source_format"], "ultralytics_yolo")
            self.assertEqual(split_count(cache_dir, "train"), 1)
            train_json = json.loads((cache_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
            self.assertEqual(train_json["categories"][0]["name"], "ball")
            self.assertEqual(train_json["annotations"][0]["bbox"], [40.0, 24.0, 20.0, 32.0])
            self.assertEqual(config["train"]["dataset_file"], "roboflow")

    def test_ultralytics_yolo_symlink_images_preserve_label_lookup(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            temp_path = Path(temp)
            source_root = temp_path / "source_images"
            root = temp_path / "yolo_symlink_export"
            cache_root = temp_path / "cache"
            for split in ("train", "val", "test"):
                source_image = source_root / split / "images" / f"{split}.jpg"
                make_image(source_image)
                image_dir = root / "images" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(source_image, image_dir / f"{split}.jpg")
                except (OSError, NotImplementedError) as exc:
                    self.skipTest(f"Symlinks are not available: {exc}")
                label_dir = root / "labels" / split
                label_dir.mkdir(parents=True, exist_ok=True)
                (label_dir / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (root / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [ball]\n",
                encoding="utf-8",
            )
            config = base_config(root, cache_root, source_format="ultralytics_yolo")
            config["dataset"]["data_yaml"] = str(root / "data.yaml")

            _, cache_dir = materialize(config, temp_path)

            self.assertEqual([split_count(cache_dir, split) for split in ("train", "valid", "test")], [1, 1, 1])
            self.assertEqual([annotation_count(cache_dir, split) for split in ("train", "valid", "test")], [1, 1, 1])

    def test_yolo_yaml_missing_host_path_falls_back_to_dataset_dir(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp) / "yolo_windows_path"
            cache_root = Path(temp) / "cache"
            for split in ("train", "val", "test"):
                image = root / "images" / split / f"{split}.jpg"
                make_image(image)
                label_dir = root / "labels" / split
                label_dir.mkdir(parents=True, exist_ok=True)
                (label_dir / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (root / "dataset.yaml").write_text(
                "path: D:\\datasets\\exported_on_windows\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "names: [ball]\n",
                encoding="utf-8",
            )
            config = base_config(root, cache_root, source_format="ultralytics_yolo")
            config["dataset"]["data_yaml"] = str(root / "dataset.yaml")

            metadata, cache_dir = materialize(config, Path(temp))

            self.assertEqual(metadata["source_format"], "ultralytics_yolo")
            self.assertIn("does not exist on this host", metadata["warnings"][0])
            self.assertEqual([split_count(cache_dir, split) for split in ("train", "valid", "test")], [1, 1, 1])

    def test_yolo_test_original_split_is_preserved_for_evaluation(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp) / "yolo_test_original"
            cache_root = Path(temp) / "cache"
            for split in ("train", "val", "test", "test-original"):
                image = root / "images" / split / f"{split}.jpg"
                make_image(image)
                label_dir = root / "labels" / split
                label_dir.mkdir(parents=True, exist_ok=True)
                (label_dir / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (root / "dataset.yaml").write_text(
                "path: .\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "test_original: images/test-original\n"
                "names: [ball]\n",
                encoding="utf-8",
            )
            config = base_config(root, cache_root, source_format="ultralytics_yolo")
            config["dataset"]["data_yaml"] = str(root / "dataset.yaml")

            metadata, cache_dir = materialize(config, Path(temp))

            self.assertEqual(metadata["split_counts"]["test-original"], 1)
            self.assertEqual(split_count(cache_dir, "test-original"), 1)

    def test_source_conversion_respects_dataset_split_limits(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp) / "yolo_limited"
            cache_root = Path(temp) / "cache"
            for split, count in (("train", 3), ("val", 2), ("test", 2)):
                for index in range(count):
                    image = root / "images" / split / f"{split}_{index}.jpg"
                    make_image(image)
                    label_dir = root / "labels" / split
                    label_dir.mkdir(parents=True, exist_ok=True)
                    (label_dir / f"{split}_{index}.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (root / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [ball]\n",
                encoding="utf-8",
            )
            config = base_config(root, cache_root, source_format="ultralytics_yolo")
            config["dataset"].update(
                {
                    "data_yaml": str(root / "data.yaml"),
                    "max_train_images": 2,
                    "max_val_images": 1,
                    "max_test_images": 1,
                }
            )

            metadata, cache_dir = materialize(config, Path(temp))

            self.assertEqual(metadata["split_counts"], {"train": 2, "valid": 1, "test": 1})
            self.assertEqual([split_count(cache_dir, split) for split in ("train", "valid", "test")], [2, 1, 1])

    def test_rfdetr_coco_split_limits_build_limited_cache(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp) / "rfdetr_coco"
            cache_root = Path(temp) / "cache"
            for split, count in (("train", 3), ("valid", 2), ("test", 2)):
                split_dir = root / split
                images = []
                annotations = []
                for index in range(count):
                    name = f"{split}_{index}.jpg"
                    make_image(split_dir / name)
                    image_id = index + 1
                    images.append({"id": image_id, "file_name": name, "width": 100, "height": 80})
                    annotations.append(
                        {"id": image_id, "image_id": image_id, "category_id": 1, "bbox": [10, 12, 20, 30]}
                    )
                (split_dir / "_annotations.coco.json").write_text(
                    json.dumps(
                        {
                            "images": images,
                            "annotations": annotations,
                            "categories": [{"id": 1, "name": "ball"}],
                        }
                    ),
                    encoding="utf-8",
                )
            config = base_config(root, cache_root, source_format="rfdetr")
            config["dataset"].update({"max_train_images": 2, "max_val_images": 1, "max_test_images": 1})

            plan = trainer.build_dataset_plan(config, Path(temp) / "run", None)
            metadata = trainer.materialize_dataset_plan(plan, config, Path(temp) / "run", verbose=False)
            cache_dir = Path(config["train"]["dataset_dir"])

            self.assertEqual(plan["action"], "prepare_cache")
            self.assertEqual(metadata["source_format"], "rfdetr")
            self.assertIn("rfdetr_limited", cache_dir.name)
            self.assertEqual([split_count(cache_dir, split) for split in ("train", "valid", "test")], [2, 1, 1])

    def test_coco_json_unsplit_uses_8_1_1_and_reuses_cache(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp) / "coco"
            image_dir = root / "images"
            images = []
            annotations = []
            for index in range(10):
                name = f"img_{index}.jpg"
                make_image(image_dir / name)
                images.append({"id": index + 1, "file_name": name, "width": 100, "height": 80})
                annotations.append({"id": index + 1, "image_id": index + 1, "category_id": 5, "bbox": [10, 12, 20, 30]})
            coco_json = root / "annotations.json"
            coco_json.write_text(
                json.dumps(
                    {
                        "images": images,
                        "annotations": annotations,
                        "categories": [{"id": 5, "name": "widget"}],
                    }
                ),
                encoding="utf-8",
            )
            config = base_config(root, Path(temp) / "cache", source_format="coco_json")
            config["dataset"]["coco_json"] = str(coco_json)
            config["dataset"]["image_dir"] = str(image_dir)

            _, cache_dir = materialize(config, Path(temp))
            self.assertEqual([split_count(cache_dir, split) for split in ("train", "valid", "test")], [8, 1, 1])

            plan = trainer.build_dataset_plan(config, Path(temp) / "run", None)
            metadata = trainer.materialize_dataset_plan(plan, config, Path(temp) / "run", verbose=False)
            self.assertTrue(metadata["cache_reused"])

    def test_pascal_voc_dota_and_labelme_convert(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            temp_path = Path(temp)
            cases = [
                ("pascal_voc", self._make_voc_fixture(temp_path / "voc")),
                ("dota", self._make_dota_fixture(temp_path / "dota")),
                ("labelme_json", self._make_labelme_fixture(temp_path / "labelme")),
            ]
            for source_format, root in cases:
                with self.subTest(source_format=source_format):
                    config = base_config(root, temp_path / f"cache_{source_format}", source_format=source_format)
                    metadata, cache_dir = materialize(config, temp_path)
                    self.assertEqual(metadata["source_format"], source_format)
                    self.assertEqual([split_count(cache_dir, split) for split in ("train", "valid", "test")], [8, 1, 1])

    def test_unsplit_dataset_requires_enough_images(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp) / "labelme_small"
            self._make_labelme_fixture(root, count=2)
            config = base_config(root, Path(temp) / "cache", source_format="labelme_json")
            with self.assertRaisesRegex(ValueError, "At least 3 images"):
                trainer.build_dataset_plan(config, Path(temp) / "run", None)

    def _make_voc_fixture(self, root: Path, count: int = 10) -> Path:
        for index in range(count):
            stem = f"voc_{index}"
            make_image(root / "JPEGImages" / f"{stem}.jpg")
            xml = f"""
<annotation>
  <filename>{stem}.jpg</filename>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <object><name>person</name><difficult>0</difficult><bndbox><xmin>10</xmin><ymin>12</ymin><xmax>40</xmax><ymax>42</ymax></bndbox></object>
</annotation>
"""
            annotation_dir = root / "Annotations"
            annotation_dir.mkdir(parents=True, exist_ok=True)
            (annotation_dir / f"{stem}.xml").write_text(xml, encoding="utf-8")
        return root

    def _make_dota_fixture(self, root: Path, count: int = 10) -> Path:
        for index in range(count):
            stem = f"dota_{index}"
            make_image(root / "images" / f"{stem}.png")
            label_dir = root / "labelTxt"
            label_dir.mkdir(parents=True, exist_ok=True)
            (label_dir / f"{stem}.txt").write_text("10 10 30 10 30 30 10 30 plane 0\n", encoding="utf-8")
        return root

    def _make_labelme_fixture(self, root: Path, count: int = 10) -> Path:
        for index in range(count):
            name = f"labelme_{index}.jpg"
            make_image(root / name)
            data = {
                "imagePath": name,
                "imageWidth": 100,
                "imageHeight": 80,
                "shapes": [{"label": "box", "shape_type": "rectangle", "points": [[10, 12], [40, 42]]}],
            }
            (root / f"labelme_{index}.json").write_text(json.dumps(data), encoding="utf-8")
        return root


if __name__ == "__main__":
    unittest.main()
