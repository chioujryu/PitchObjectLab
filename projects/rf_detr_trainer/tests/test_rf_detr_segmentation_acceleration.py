"""Standalone segmentation acceptance tests for accelerated RF-DETR inference."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

import train_rf_detr_model as trainer_runtime


class _Samples:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        self.tensors = self.tensors.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        return self


class _FakeMeanAveragePrecision:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = None
        self.updates = []
        type(self).instances.append(self)

    def to(self, device):
        self.device = torch.device(device)
        return self

    def update(self, predictions, targets):
        self.updates.append((predictions, targets))

    def compute(self):
        return {
            "bbox_map": torch.tensor(0.61),
            "bbox_map_50": torch.tensor(0.81),
            "bbox_map_75": torch.tensor(0.55),
            "bbox_mar_500": torch.tensor(0.72),
            "segm_map": torch.tensor(0.58),
            "segm_map_50": torch.tensor(0.78),
            "classes": torch.tensor([0]),
            "bbox_map_per_class": torch.tensor([0.61]),
            "bbox_mar_500_per_class": torch.tensor([0.72]),
        }


class _FakeCOCOConverter:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.predictions = None
        self.targets = None
        type(self).instances.append(self)

    def _convert_preds(self, predictions):
        self.predictions = predictions
        return predictions

    def _convert_targets(self, targets):
        self.targets = targets
        return targets


class _FakeSegmentationRuntime:
    def __init__(self, *, backend="tensorrt"):
        self.backend = backend
        self.device = torch.device("cpu")
        self.raw_outputs = {
            "pred_boxes": torch.tensor(
                [
                    [[0.5, 0.5, 0.25, 0.25]],
                    [[0.4, 0.4, 0.2, 0.2]],
                ]
            ),
            "pred_logits": torch.tensor([[[8.0]], [[7.0]]]),
            "pred_masks": torch.ones(2, 1, 8, 8),
        }
        self.infer_raw = MagicMock(return_value=self.raw_outputs)
        self.postprocess = MagicMock(side_effect=self._postprocess)

    @staticmethod
    def _postprocess(outputs, original_sizes):
        results = []
        for index, size in enumerate(original_sizes.tolist()):
            height, width = (int(size[0]), int(size[1]))
            results.append(
                {
                    "boxes": torch.tensor([[1.0, 2.0, float(width - 1), float(height - 1)]]),
                    "scores": torch.tensor([0.9 - index * 0.1]),
                    "labels": torch.tensor([0]),
                    "masks": torch.ones(1, height, width, dtype=torch.bool),
                    "raw_mask_source": outputs["pred_masks"][index],
                }
            )
        return results


def _targets():
    return [
        {
            "orig_size": torch.tensor([13, 19]),
            "boxes": torch.tensor([[1.0, 2.0, 18.0, 12.0]]),
            "labels": torch.tensor([0]),
            "masks": torch.ones(1, 13, 19, dtype=torch.bool),
        },
        {
            "orig_size": torch.tensor([11, 17]),
            "boxes": torch.tensor([[1.0, 2.0, 16.0, 10.0]]),
            "labels": torch.tensor([0]),
            "masks": torch.ones(1, 11, 17, dtype=torch.bool),
        },
    ]


class StandaloneSegmentationAccelerationTest(unittest.TestCase):
    def setUp(self):
        _FakeMeanAveragePrecision.instances.clear()
        _FakeCOCOConverter.instances.clear()

    @contextmanager
    def _manual_environment(self, batch):
        dataset = SimpleNamespace()
        datamodule = SimpleNamespace(class_names=["football"])
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    trainer_runtime,
                    "build_eval_dataloader",
                    return_value=(dataset, [batch]),
                )
            )
            stack.enter_context(
                patch(
                    "torchmetrics.detection.MeanAveragePrecision",
                    _FakeMeanAveragePrecision,
                )
            )
            stack.enter_context(
                patch(
                    "rfdetr.training.callbacks.coco_eval.COCOEvalCallback",
                    _FakeCOCOConverter,
                )
            )
            stack.enter_context(
                patch(
                    "rfdetr.evaluation.matching.init_matching_accumulator",
                    return_value={},
                )
            )
            build_matching = stack.enter_context(
                patch(
                    "rfdetr.evaluation.matching.build_matching_data",
                    return_value={"matched": True},
                )
            )
            merge_matching = stack.enter_context(
                patch("rfdetr.evaluation.matching.merge_matching_data")
            )
            stack.enter_context(
                patch.object(
                    trainer_runtime,
                    "compute_f1_by_class",
                    return_value=(
                        {"F1": 0.7, "Precision": 0.75, "Recall": 0.66},
                        {0: {"f1": 0.7, "precision": 0.75, "recall": 0.66}},
                    ),
                )
            )
            stack.enter_context(patch.object(trainer_runtime, "write_json"))
            stack.enter_context(patch.object(trainer_runtime, "write_rows"))
            stack.enter_context(patch.object(trainer_runtime, "dump_config_snapshot"))
            stack.enter_context(patch.object(trainer_runtime, "blue"))
            yield datamodule, build_matching, merge_matching

    @staticmethod
    def _evaluate(output_dir, datamodule, runtime):
        return trainer_runtime.manual_test_evaluation(
            trainer=None,
            pl_module=None,
            datamodule=datamodule,
            model_config=SimpleNamespace(segmentation_head=True, resolution=32),
            train_config=SimpleNamespace(eval_max_dets=500),
            output_dir=Path(output_dir),
            split="test",
            event="standalone_test",
            metadata={"event": "acceptance"},
            merged_config={
                "periodic_test": {
                    "test_mode": {"mode": "full_image"},
                    "classwise": False,
                },
                "evaluation": {"type": "segm"},
            },
            source_config=None,
            verbose=False,
            progress_bar=False,
            inference_runtime=runtime,
        )

    def test_raw_masks_use_original_size_postprocess_and_route_bbox_and_segm_metrics(self):
        samples = _Samples(
            torch.zeros(2, 3, 32, 32),
            torch.zeros(2, 32, 32, dtype=torch.bool),
        )
        targets = _targets()
        runtime = _FakeSegmentationRuntime()

        with tempfile.TemporaryDirectory() as temporary, self._manual_environment(
            (samples, targets)
        ) as (datamodule, build_matching, merge_matching):
            payload = self._evaluate(temporary, datamodule, runtime)

        runtime.infer_raw.assert_called_once()
        self.assertIs(runtime.infer_raw.call_args.args[0], samples.tensors)
        runtime.postprocess.assert_called_once()
        self.assertIs(runtime.postprocess.call_args.args[0], runtime.raw_outputs)
        self.assertTrue(
            torch.equal(
                runtime.postprocess.call_args.args[1],
                torch.tensor([[13, 19], [11, 17]]),
            )
        )

        metric = _FakeMeanAveragePrecision.instances[0]
        self.assertEqual(metric.kwargs["iou_type"], ["bbox", "segm"])
        self.assertEqual(metric.device, torch.device("cpu"))
        self.assertEqual(len(metric.updates), 1)
        predictions, metric_targets = metric.updates[0]
        self.assertEqual(predictions[0]["masks"].shape, (1, 13, 19))
        self.assertEqual(predictions[1]["masks"].shape, (1, 11, 17))
        self.assertEqual(metric_targets[0]["masks"].shape, (1, 13, 19))
        self.assertEqual(metric_targets[1]["masks"].shape, (1, 11, 17))
        self.assertEqual(_FakeCOCOConverter.instances[0].kwargs["segmentation"], True)
        build_matching.assert_called_once()
        self.assertEqual(build_matching.call_args.kwargs["iou_type"], "segm")
        merge_matching.assert_called_once_with({}, {"matched": True})

        self.assertAlmostEqual(payload["overall"]["mAP_50_95"], 0.61, places=5)
        self.assertAlmostEqual(payload["overall"]["mAP_50"], 0.81, places=5)
        self.assertAlmostEqual(payload["overall"]["segm_mAP_50_95"], 0.58, places=5)
        self.assertAlmostEqual(payload["overall"]["segm_mAP_50"], 0.78, places=5)
        stage_timing = payload["stage_timing"]
        self.assertEqual(stage_timing["images_or_frames"], 2)
        self.assertEqual(stage_timing["base_model_forward_seconds"], stage_timing["model_forward_seconds"])
        self.assertEqual(stage_timing["sahi_model_forward_seconds"], 0.0)
        self.assertEqual(stage_timing["recheck_model_forward_seconds"], 0.0)
        self.assertEqual(stage_timing["sahi_model_forward_ratio"], 0.0)
        self.assertEqual(stage_timing["recheck_model_forward_ratio"], 0.0)

    def test_tensorrt_segmentation_rejects_non_square_or_padded_batches_before_forward(self):
        invalid_batches = [
            (
                _Samples(
                    torch.zeros(1, 3, 32, 24),
                    torch.zeros(1, 32, 24, dtype=torch.bool),
                ),
                "fixed square inputs",
            ),
            (
                _Samples(
                    torch.zeros(1, 3, 32, 32),
                    torch.ones(1, 32, 32, dtype=torch.bool),
                ),
                "padded/non-square samples",
            ),
        ]
        target = [_targets()[0]]

        for samples, message in invalid_batches:
            with self.subTest(message=message):
                runtime = _FakeSegmentationRuntime()
                with tempfile.TemporaryDirectory() as temporary, self._manual_environment(
                    (samples, target)
                ) as (datamodule, _build_matching, _merge_matching):
                    with self.assertRaisesRegex(ValueError, message):
                        self._evaluate(temporary, datamodule, runtime)

                runtime.infer_raw.assert_not_called()
                runtime.postprocess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
