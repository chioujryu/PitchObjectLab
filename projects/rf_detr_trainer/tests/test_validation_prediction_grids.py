from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_rf_detr_model as trainer


def fake_validation_batch() -> tuple[SimpleNamespace, list[dict[str, torch.Tensor]]]:
    samples = SimpleNamespace(tensors=torch.zeros((1, 3, 32, 32), dtype=torch.float32))
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "orig_size": torch.tensor([32, 32]),
            "size": torch.tensor([32, 32]),
        }
    ]
    return samples, targets


def fake_multiscale_resized_batch() -> tuple[SimpleNamespace, list[dict[str, torch.Tensor]]]:
    samples = SimpleNamespace(
        tensors=torch.zeros((1, 3, 608, 608), dtype=torch.float32),
        mask=torch.zeros((1, 608, 608), dtype=torch.bool),
    )
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "orig_size": torch.tensor([960, 960]),
            "size": torch.tensor([736, 736]),
        }
    ]
    return samples, targets


def fake_validation_outputs(score: float = 0.9) -> dict[str, list[dict[str, torch.Tensor]]]:
    return {
        "results": [
            {
                "boxes": torch.tensor([[5.0, 6.0, 20.0, 22.0]]),
                "scores": torch.tensor([score]),
                "labels": torch.tensor([1]),
            }
        ]
    }


class ValidationPredictionGridTest(unittest.TestCase):
    def test_target_boxes_are_converted_from_normalized_cxcywh(self):
        boxes = trainer.target_boxes_xyxy(
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.25]], dtype=torch.float32),
                "size": torch.tensor([40, 80]),
            }
        )

        self.assertEqual(boxes, [[20.0, 15.0, 60.0, 25.0]])

    def test_batch_grid_valid_size_prefers_current_mask_after_multiscale_resize(self):
        samples, targets = fake_multiscale_resized_batch()

        valid_size = trainer.batch_item_valid_size_hw(samples.tensors[0], samples.mask[0], targets[0])

        self.assertEqual(valid_size, (608.0, 608.0))

    def test_batch_grid_valid_size_falls_back_to_tensor_size_without_mask(self):
        samples, targets = fake_multiscale_resized_batch()

        valid_size = trainer.batch_item_valid_size_hw(samples.tensors[0], None, targets[0])

        self.assertEqual(valid_size, (608.0, 608.0))

    def test_batch_grid_mask_valid_size_ignores_padding(self):
        mask = torch.ones((608, 640), dtype=torch.bool)
        mask[:608, :512] = False

        self.assertEqual(trainer.mask_valid_size_hw(mask), (608.0, 512.0))

    def test_draw_target_labels_uses_supplied_current_valid_size_for_normalized_boxes(self):
        image = Image.new("RGB", (608, 608))
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "size": torch.tensor([736, 736]),
        }
        captured: dict[str, object] = {}

        def capture_draw(**kwargs):
            captured.update(kwargs)

        with patch.object(trainer, "draw_boxes_on_tile", side_effect=capture_draw):
            trainer.draw_target_labels_on_tile(image, target, valid_size=(608.0, 608.0))

        self.assertEqual(captured["boxes"], [[228.0, 228.0, 380.0, 380.0]])

    def test_train_grid_passes_current_valid_size_after_multiscale_resize(self):
        samples, targets = fake_multiscale_resized_batch()
        captured: list[tuple[float, float] | None] = []

        def fake_draw(image, target, valid_size=None, color=(37, 99, 235)):
            del image, target, color
            captured.append(valid_size)
            return Image.new("RGB", (trainer.BATCH_GRID_TILE_SIZE, trainer.BATCH_GRID_TILE_SIZE))

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(trainer, "draw_target_labels_on_tile", side_effect=fake_draw):
                trainer.save_batch_label_grid((samples, targets), Path(temp), "train_batch0.jpg", "train batch=0")

        self.assertEqual(captured, [(608.0, 608.0)])

    def test_train_callback_writes_initial_training_batch_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            callback = trainer.TrainBatchGridCallback(output_dir, verbose=False)
            fake_trainer = SimpleNamespace(is_global_zero=True)

            callback.on_train_batch_end(fake_trainer, None, None, fake_validation_batch(), 0)

            path = output_dir / "train_batch0.jpg"
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes()[:2], b"\xff\xd8")

    def test_train_callback_does_not_overwrite_after_initial_batch_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            path = output_dir / "train_batch0.jpg"
            callback = trainer.TrainBatchGridCallback(output_dir, verbose=False)
            fake_trainer = SimpleNamespace(is_global_zero=True)

            callback.on_train_batch_end(fake_trainer, None, None, fake_validation_batch(), 0)
            path.write_bytes(b"stale")
            callback.on_train_batch_end(fake_trainer, None, None, fake_validation_batch(), 0)

            self.assertEqual(path.read_bytes(), b"stale")

    def test_callback_writes_validation_label_and_prediction_grids(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            callback = trainer.ValidationPredictionGridCallback(output_dir, verbose=False)
            fake_trainer = SimpleNamespace(sanity_checking=False, is_global_zero=True)

            callback.on_validation_epoch_start(fake_trainer, None)
            callback.on_validation_batch_end(fake_trainer, None, fake_validation_outputs(), fake_validation_batch(), 0)

            for path in (output_dir / "val_batch0_labels.jpg", output_dir / "val_batch0_pred.jpg"):
                self.assertTrue(path.exists())
                self.assertEqual(path.read_bytes()[:2], b"\xff\xd8")

    def test_validation_prediction_grid_filters_low_score_boxes(self):
        image = Image.new("RGB", (100, 100))
        result = {
            "boxes": torch.tensor(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [20.0, 20.0, 30.0, 30.0],
                    [40.0, 40.0, 50.0, 50.0],
                ],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.249, 0.25, 0.9], dtype=torch.float32),
            "labels": torch.tensor([1, 2, 3], dtype=torch.int64),
        }
        captured: dict[str, object] = {}

        def capture_draw(**kwargs):
            captured.update(kwargs)

        with patch.object(trainer, "draw_boxes_on_tile", side_effect=capture_draw):
            trainer.draw_validation_predictions_on_tile(
                image,
                result,
                source_size=(100.0, 100.0),
                valid_size=(100.0, 100.0),
            )

        self.assertEqual(captured["boxes"], [[20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]])
        self.assertEqual(captured["labels"], [2, 3])
        self.assertEqual([round(score, 3) for score in captured["scores"]], [0.25, 0.9])

    def test_callback_overwrites_existing_validation_prediction_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            path = output_dir / "val_batch0_pred.jpg"
            callback = trainer.ValidationPredictionGridCallback(output_dir, verbose=False)
            fake_trainer = SimpleNamespace(sanity_checking=False, is_global_zero=True)

            callback.on_validation_epoch_start(fake_trainer, None)
            callback.on_validation_batch_end(fake_trainer, None, fake_validation_outputs(), fake_validation_batch(), 0)
            path.write_bytes(b"stale")

            callback.on_validation_epoch_start(fake_trainer, None)
            callback.on_validation_batch_end(
                fake_trainer, None, fake_validation_outputs(0.5), fake_validation_batch(), 0
            )

            self.assertNotEqual(path.read_bytes(), b"stale")
            self.assertEqual(path.read_bytes()[:2], b"\xff\xd8")

    def test_callback_skips_sanity_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            callback = trainer.ValidationPredictionGridCallback(output_dir, verbose=False)
            fake_trainer = SimpleNamespace(sanity_checking=True, is_global_zero=True)

            callback.on_validation_batch_end(fake_trainer, None, fake_validation_outputs(), fake_validation_batch(), 0)

            self.assertFalse((output_dir / "val_batch0_pred.jpg").exists())
            self.assertFalse((output_dir / "val_batch0_labels.jpg").exists())

    def test_callback_skips_non_global_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            callback = trainer.ValidationPredictionGridCallback(output_dir, verbose=False)
            fake_trainer = SimpleNamespace(sanity_checking=False, is_global_zero=False)

            callback.on_validation_batch_end(fake_trainer, None, fake_validation_outputs(), fake_validation_batch(), 0)

            self.assertFalse((output_dir / "val_batch0_pred.jpg").exists())
            self.assertFalse((output_dir / "val_batch0_labels.jpg").exists())

    def test_callback_clears_stale_validation_grids_at_validation_start(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            stale_pred_path = output_dir / "val_batch2_pred.jpg"
            stale_label_path = output_dir / "val_batch2_labels.jpg"
            stale_pred_path.write_bytes(b"stale")
            stale_label_path.write_bytes(b"stale")
            callback = trainer.ValidationPredictionGridCallback(output_dir, verbose=False)
            fake_trainer = SimpleNamespace(sanity_checking=False, is_global_zero=True)

            callback.on_validation_epoch_start(fake_trainer, None)

            self.assertFalse(stale_pred_path.exists())
            self.assertFalse(stale_label_path.exists())

    def test_estimate_includes_validation_prediction_grids_without_sample_grid_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            estimate = trainer.estimate_outputs(
                {
                    "train": {"epochs": 2, "batch_size": 2, "checkpoint_interval": 1},
                    "periodic_test": {"run_final_test": False, "classwise": False},
                },
                Path(temp) / "run",
                periodic_count=0,
                dataset_plan={"split_counts": {"train": 4, "valid": 1}},
            )

            self.assertEqual(estimate["batch_grid_files"], 9)
            self.assertEqual(estimate["train_batch_grid_files"], trainer.TRAIN_BATCH_GRID_MAX_BATCHES)
            self.assertEqual(estimate["validation_label_grid_files"], trainer.VALIDATION_PREDICTION_GRID_MAX_BATCHES)
            self.assertEqual(
                estimate["validation_prediction_grid_files"], trainer.VALIDATION_PREDICTION_GRID_MAX_BATCHES
            )
            self.assertNotIn("dataset_grid_files", estimate)
            self.assertNotIn("dataset_grid_dir", estimate)

    def test_sample_grid_cli_flags_are_removed(self):
        parser = trainer.build_parser()
        options = {option for action in parser._actions for option in action.option_strings}

        self.assertNotIn("--save-dataset-grids", options)
        self.assertNotIn("--no-save-dataset-grids", options)


if __name__ == "__main__":
    unittest.main()
