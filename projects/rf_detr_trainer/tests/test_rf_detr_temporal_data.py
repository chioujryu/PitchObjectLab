import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from rf_detr_temporal_data import (  # noqa: E402
    TemporalDataset,
    TemporalRFDETRDataModule,
    TemporalSample,
    generate_gaussian_heatmap,
    load_temporal_index,
    temporal_collate_fn,
    temporal_split_window_counts,
)

TEMP_ROOT = Path(
    os.environ.get(
        "RF_DETR_TEST_TMP",
        r"C:\tmp" if os.name == "nt" else "/tmp/rf_detr_trainer_tests",
    )
)
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def _write_image(path: Path, *, frame_index: int, size: tuple[int, int] = (6, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    image = Image.new("RGB", size, color=(10 + frame_index, 20, 30))
    for y in range(height):
        for x in range(width // 2):
            image.putpixel((x, y), (200, frame_index, 0))
    image.save(path)


def _write_fixture(root: Path, *, missing_primary: bool = False) -> Path:
    (root / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names: [ball]\n",
        encoding="utf-8",
    )
    rows = []
    specifications = (
        ("train", "cut-a/clip-a", 5),
        ("train", "cut-b/clip-b", 4),
        ("val", "cut-c/clip-c", 3),
        ("test", "cut-d/clip-d", 3),
    )
    for split, sequence_id, count in specifications:
        cut, clip = sequence_id.split("/")
        for frame_index in range(count):
            stem = f"{frame_index:05d}"
            image_rel = Path("images") / split / cut / clip / f"{stem}.jpg"
            label_rel = Path("labels") / split / cut / clip / f"{stem}.txt"
            _write_image(root / image_rel, frame_index=frame_index)
            (root / label_rel).parent.mkdir(parents=True, exist_ok=True)

            if split == "train" and sequence_id == "cut-a/clip-a" and frame_index == 1:
                labels = "0 0.25 0.5 0.20 0.40\n0 0.75 0.5 0.10 0.20\n"
                box_count = 2
                primary = None if missing_primary else 1
            elif frame_index == count - 1:
                labels = ""
                box_count = 0
                primary = None
            else:
                labels = f"0 {0.2 + frame_index * 0.1:.2f} 0.5 0.20 0.40\n"
                box_count = 1
                primary = 0
            (root / label_rel).write_text(labels, encoding="utf-8")
            rows.append(
                {
                    "split": split,
                    "sequence_id": sequence_id,
                    "group": sequence_id,
                    "cut": cut,
                    "clip": clip,
                    "frame_index": frame_index,
                    "frame": stem,
                    "image": image_rel.as_posix(),
                    "label": label_rel.as_posix(),
                    "primary_label_index": primary,
                    "box_count": box_count,
                    "is_empty": box_count == 0,
                    "source_frame_index": frame_index + 100,
                }
            )
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "temporal_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root / "dataset.yaml"


class TemporalIndexTest(unittest.TestCase):
    def test_load_index_and_build_windows_never_cross_sequences(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            yaml_path = _write_fixture(Path(temp))

            index = load_temporal_index(yaml_path)
            dataset = TemporalDataset(
                yaml_path,
                split="train",
                image_size=(4, 6),
                normalize=False,
            )

            self.assertEqual(len(index.records), 15)
            self.assertEqual(index.class_names, ("ball",))
            self.assertEqual(len(dataset), 5)
            self.assertEqual(
                temporal_split_window_counts(Path(temp)),
                {"train": 5, "val": 1, "test": 1},
            )
            for window in dataset.windows:
                self.assertEqual(len({record.split for record in window.records}), 1)
                self.assertEqual(len({record.sequence_id for record in window.records}), 1)
                frame_indices = [record.frame_index for record in window.records]
                self.assertEqual(frame_indices, list(range(frame_indices[0], frame_indices[0] + 3)))

    def test_missing_frame_produces_no_cross_gap_window(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp)
            yaml_path = _write_fixture(root)
            rows = [
                json.loads(line)
                for line in (root / "metadata" / "temporal_index.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            rows = [
                row
                for row in rows
                if not (
                    row["split"] == "train"
                    and row["sequence_id"] == "cut-a/clip-a"
                    and row["frame_index"] == 2
                )
            ]
            (root / "metadata" / "temporal_index.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            dataset = TemporalDataset(yaml_path, split="train", image_size=8)

            # cut-a no longer has a complete 3-frame run; cut-b still has two.
            self.assertEqual(len(dataset), 2)
            self.assertTrue(
                all(window.sequence_id == "cut-b/clip-b" for window in dataset.windows)
            )


class TemporalDatasetTest(unittest.TestCase):
    def test_outputs_normalized_targets_heatmaps_and_anchor_metadata(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            yaml_path = _write_fixture(Path(temp))
            dataset = TemporalDataset(
                yaml_path,
                split="train",
                focus_mode="single",
                image_size=(8, 12),
                normalize=True,
            )

            sample = dataset[0]

            self.assertEqual(sample.frames.shape, (3, 3, 8, 12))
            self.assertEqual(sample.padding_masks.shape, (3, 8, 12))
            self.assertFalse(sample.padding_masks.any())
            self.assertEqual(sample.anchor_index, 1)
            self.assertEqual(sample.metadata["frame_indices"], (0, 1, 2))
            anchor = sample.frame_targets[1]
            self.assertEqual(anchor["boxes"].shape, (2, 4))
            torch.testing.assert_close(anchor["boxes"][1], torch.tensor([0.75, 0.5, 0.1, 0.2]))
            self.assertEqual(anchor["tracknet_box_indices"].tolist(), [1])
            self.assertEqual(anchor["primary_label_index"].item(), 1)
            self.assertEqual(anchor["orig_size"].tolist(), [4, 6])
            self.assertEqual(anchor["size"].tolist(), [8, 12])
            self.assertEqual(anchor["tracknet_heatmap"].shape, (8, 12))
            self.assertGreater(float(anchor["tracknet_heatmap"].max()), 0.8)
            self.assertTrue(torch.isfinite(sample.frames).all())

    def test_resize_and_flip_are_replayed_identically_across_window(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            yaml_path = _write_fixture(Path(temp))
            unflipped = TemporalDataset(
                yaml_path,
                split="train",
                image_size=(4, 6),
                horizontal_flip=False,
                normalize=False,
            )[0]
            flipped = TemporalDataset(
                yaml_path,
                split="train",
                image_size=(4, 6),
                horizontal_flip=True,
                normalize=False,
            )[0]

            torch.testing.assert_close(flipped.frames, unflipped.frames.flip(-1))
            for original, transformed in zip(
                unflipped.frame_targets, flipped.frame_targets
            ):
                if original["boxes"].numel():
                    torch.testing.assert_close(
                        transformed["boxes"][:, 0], 1.0 - original["boxes"][:, 0]
                    )
                    torch.testing.assert_close(
                        transformed["boxes"][:, 1:], original["boxes"][:, 1:]
                    )
                torch.testing.assert_close(
                    transformed["tracknet_heatmap"],
                    original["tracknet_heatmap"].flip(-1),
                    atol=0.16,
                    rtol=0,
                )

    def test_all_focus_keeps_every_box_and_single_requires_primary_for_multi_ball(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp)
            yaml_path = _write_fixture(root)
            all_sample = TemporalDataset(
                yaml_path,
                split="train",
                focus_mode="all",
                image_size=8,
                normalize=False,
            )[0]
            self.assertEqual(all_sample.frame_targets[1]["tracknet_box_indices"].tolist(), [0, 1])

        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            yaml_path = _write_fixture(Path(temp), missing_primary=True)
            dataset = TemporalDataset(
                yaml_path,
                split="train",
                focus_mode="single",
                image_size=8,
            )
            with self.assertRaisesRegex(ValueError, "requires primary_label_index"):
                _ = dataset[0]

    def test_empty_frame_has_zero_heatmap(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            yaml_path = _write_fixture(Path(temp))
            dataset = TemporalDataset(
                yaml_path,
                split="val",
                focus_mode="single",
                image_size=8,
                normalize=False,
            )

            sample = dataset[0]

            self.assertEqual(sample.frame_targets[-1]["boxes"].shape, (0, 4))
            self.assertEqual(sample.frame_targets[-1]["tracknet_box_indices"].numel(), 0)
            self.assertEqual(float(sample.frame_targets[-1]["tracknet_heatmap"].sum()), 0.0)


class TemporalCollateTest(unittest.TestCase):
    def test_mdd_frames_invert_normalization_without_changing_backbone_frames(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            yaml_path = _write_fixture(Path(temp))
            normalized_sample = TemporalDataset(
                yaml_path,
                split="train",
                image_size=(4, 6),
                normalize=True,
            )[0]
            raw_sample = TemporalDataset(
                yaml_path,
                split="train",
                image_size=(4, 6),
                normalize=False,
            )[0]

            normalized_batch, _ = temporal_collate_fn([normalized_sample])
            raw_batch, _ = temporal_collate_fn([raw_sample])

            self.assertIsNotNone(normalized_batch.normalization_mean)
            self.assertIsNone(raw_batch.normalization_mean)
            self.assertFalse(torch.allclose(normalized_batch.frames, raw_batch.frames))
            torch.testing.assert_close(
                normalized_batch.mdd_frames,
                raw_batch.frames,
                atol=1e-6,
                rtol=0.0,
            )
            torch.testing.assert_close(raw_batch.mdd_frames, raw_batch.frames)

    def test_collate_exposes_anchor_nested_tensor_api_and_temporal_targets(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            yaml_path = _write_fixture(Path(temp))
            dataset = TemporalDataset(
                yaml_path,
                split="train",
                image_size=(4, 6),
                normalize=False,
            )
            first = dataset[0]
            second_original = dataset[-1]
            second_targets = tuple(
                {
                    **target,
                    "tracknet_heatmap": target["tracknet_heatmap"][:3, :5],
                    "size": torch.tensor([3, 5], dtype=torch.int64),
                }
                for target in second_original.frame_targets
            )
            second = TemporalSample(
                frames=second_original.frames[:, :, :3, :5],
                padding_masks=second_original.padding_masks[:, :3, :5],
                frame_targets=second_targets,
                metadata=second_original.metadata,
                anchor_index=second_original.anchor_index,
            )

            batch, targets = temporal_collate_fn([first, second], block_size=4)

            self.assertEqual(batch.frames.shape, (2, 3, 3, 4, 8))
            self.assertEqual(batch.padding_masks.shape, (2, 3, 4, 8))
            self.assertEqual(batch.tensors.shape, (2, 3, 4, 8))
            self.assertEqual(batch.mask.shape, (2, 4, 8))
            self.assertEqual(batch.flattened_tensors.shape, (6, 3, 4, 8))
            self.assertEqual(batch.flattened_mask.shape, (6, 4, 8))
            self.assertTrue(batch.mask[0, :, 6:].all())
            self.assertTrue(batch.mask[1, 3:, :].all())
            self.assertTrue(
                torch.equal(
                    batch.mdd_frames.masked_select(batch.padding_masks.unsqueeze(2)),
                    torch.zeros_like(
                        batch.mdd_frames.masked_select(batch.padding_masks.unsqueeze(2))
                    ),
                )
            )
            self.assertEqual(targets[0]["temporal_heatmaps"].shape, (3, 4, 8))
            self.assertEqual(targets[0]["temporal_image_ids"].shape, (3,))
            moved = batch.to("cpu")
            self.assertEqual(moved.frames.device.type, "cpu")
            torch.testing.assert_close(moved.tensors, batch.tensors)

    def test_data_module_returns_temporal_batch_and_anchor_targets(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            yaml_path = _write_fixture(Path(temp))
            data_module = TemporalRFDETRDataModule(
                yaml_path,
                image_size=8,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
            )
            data_module.setup("fit")

            samples, targets = next(iter(data_module.train_dataloader()))

            self.assertEqual(samples.frames.shape[:2], (2, 3))
            self.assertEqual(len(targets), 2)
            self.assertIn("temporal_heatmaps", targets[0])

    def test_data_module_can_bound_each_split_to_one_micro_window(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            yaml_path = _write_fixture(Path(temp))
            data_module = TemporalRFDETRDataModule(
                yaml_path,
                image_size=8,
                batch_size=1,
                num_workers=0,
                pin_memory=False,
                max_windows_per_split={"train": 1, "val": 1, "test": 1},
            )
            data_module.setup()

            self.assertEqual(len(data_module.datasets["train"]), 1)
            self.assertEqual(len(data_module.datasets["val"]), 1)
            data_module.setup("test")
            self.assertEqual(len(data_module.datasets["test"]), 1)


class HeatmapTest(unittest.TestCase):
    def test_gaussian_heatmap_selects_requested_boxes_and_max_composites(self):
        boxes = torch.tensor(
            [
                [0.25, 0.5, 0.2, 0.2],
                [0.75, 0.5, 0.2, 0.2],
            ],
            dtype=torch.float32,
        )

        single = generate_gaussian_heatmap(boxes, (9, 17), box_indices=[1])
        all_balls = generate_gaussian_heatmap(boxes, (9, 17))

        self.assertLess(float(single[:, :8].max()), float(single[:, 8:].max()))
        self.assertGreater(float(all_balls[:, :8].max()), 0.8)
        self.assertGreater(float(all_balls[:, 8:].max()), 0.8)
        self.assertLessEqual(float(all_balls.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
