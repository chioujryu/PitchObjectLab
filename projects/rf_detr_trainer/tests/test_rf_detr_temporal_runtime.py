"""Runtime integration tests for real-temporal RF-DETR training."""

from __future__ import annotations

import random
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

import rf_detr_temporal_runtime as temporal_runtime
import torch
from pytorch_lightning import LightningModule
from rf_detr_temporal_data import TemporalBatch
from torch import nn


class _BaseCriterion(nn.Module):
    """Small criterion stub sufficient for constructing the temporal wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.weight_dict: dict[str, float] = {}


def _build_temporal_lightning_module(
    *,
    multi_scale: bool = True,
    do_random_resize_via_padding: bool = False,
):
    from rfdetr.training import RFDETRModelModule

    model_config = SimpleNamespace(
        pretrain_weights=None,
        resolution=128,
        patch_size=16,
        num_windows=2,
    )
    train_config = SimpleNamespace(
        multi_scale=multi_scale,
        expanded_scales=True,
        do_random_resize_via_padding=do_random_resize_via_padding,
    )

    def fake_init(self, received_model_config, received_train_config) -> None:
        LightningModule.__init__(self)
        self.model_config = received_model_config
        self.train_config = received_train_config
        self.model = nn.Identity()
        self.criterion = _BaseCriterion()

    with patch.object(RFDETRModelModule, "__init__", fake_init):
        with patch.object(temporal_runtime, "attach_motion_module"):
            module = temporal_runtime.build_temporal_model_module(
                model_config,
                train_config,
                {
                    "model": {
                        "motion": {
                            "enabled": True,
                            "type": "tracknet_v5",
                            "loss": {"heatmap_weight": 1.0, "gamma": 2.0},
                        }
                    }
                },
            )
    module.trainer = SimpleNamespace(global_step=0)
    return module


def _temporal_batch() -> TemporalBatch:
    batch_size, num_frames, channels = 2, 3, 1
    frames = torch.empty(batch_size, num_frames, channels, 2, 2)
    for batch_index in range(batch_size):
        for frame_index in range(num_frames):
            frames[batch_index, frame_index].fill_(float(batch_index * num_frames + frame_index))
    mask_pattern = torch.tensor(
        [[False, True], [True, False]],
        dtype=torch.bool,
    )
    padding_masks = mask_pattern.expand(batch_size, num_frames, 2, 2).clone()
    frame_targets = tuple(tuple({} for _ in range(num_frames)) for _ in range(batch_size))
    metadata = tuple({"sample": index} for index in range(batch_size))
    return TemporalBatch(
        frames=frames,
        padding_masks=padding_masks,
        frame_targets=frame_targets,
        metadata=metadata,
        anchor_index=1,
        normalization_mean=(0.1, 0.2, 0.3),
        normalization_std=(0.4, 0.5, 0.6),
    )


class TemporalMultiScaleTrainingTest(unittest.TestCase):
    def test_frozen_temporal_batch_is_replaced_and_all_frames_are_resized(self):
        from rfdetr.training import RFDETRModelModule

        module = _build_temporal_lightning_module()
        samples = _temporal_batch()
        targets = [{"sample": 0}, {"sample": 1}]
        original_frames = samples.frames.clone()
        original_masks = samples.padding_masks.clone()

        with self.assertRaises(FrozenInstanceError):
            samples.frames = samples.frames

        # Lightning invokes this hook before training_step. The temporal adapter
        # must not delegate its frozen batch to RF-DETR's in-place NestedTensor path.
        module.on_train_batch_start((samples, targets), batch_idx=0)

        captured: dict[str, object] = {}

        def capture_training_step(self, batch, batch_idx):
            captured["batch"] = batch
            captured["batch_idx"] = batch_idx
            return "delegated"

        random.seed(999)
        with patch.object(
            RFDETRModelModule,
            "training_step",
            capture_training_step,
        ):
            result = module.training_step((samples, targets), batch_idx=0)

        self.assertEqual(result, "delegated")
        self.assertEqual(captured["batch_idx"], 0)
        resized, captured_targets = captured["batch"]
        self.assertIsInstance(resized, TemporalBatch)
        self.assertIsNot(resized, samples)
        self.assertIs(captured_targets, targets)

        # RF-DETR 1.8.3 expanded multi-scale selection at resolution=128,
        # patch_size=16, num_windows=2 chooses 256 for global_step=0.
        self.assertEqual(resized.frames.shape, (2, 3, 1, 256, 256))
        self.assertEqual(resized.padding_masks.shape, (2, 3, 256, 256))
        self.assertEqual(resized.padding_masks.dtype, torch.bool)
        self.assertEqual(resized.tensors.shape, (2, 1, 256, 256))
        self.assertEqual(resized.mask.shape, (2, 256, 256))
        torch.testing.assert_close(resized.tensors, resized.frames[:, 1])
        torch.testing.assert_close(
            resized.frames[:, :, :, 128, 128],
            torch.tensor(
                [
                    [[0.0], [1.0], [2.0]],
                    [[3.0], [4.0], [5.0]],
                ]
            ),
        )
        self.assertFalse(bool(resized.padding_masks[0, 0, 32, 32]))
        self.assertTrue(bool(resized.padding_masks[0, 0, 32, 224]))
        self.assertTrue(bool(resized.padding_masks[0, 0, 224, 32]))
        self.assertFalse(bool(resized.padding_masks[0, 0, 224, 224]))

        self.assertIs(resized.frame_targets, samples.frame_targets)
        self.assertIs(resized.metadata, samples.metadata)
        self.assertEqual(resized.normalization_mean, samples.normalization_mean)
        self.assertEqual(resized.normalization_std, samples.normalization_std)
        torch.testing.assert_close(samples.frames, original_frames)
        torch.testing.assert_close(samples.padding_masks, original_masks)

    def test_temporal_batch_is_unchanged_when_multiscale_resize_is_disabled(self):
        from rfdetr.training import RFDETRModelModule

        samples = _temporal_batch()
        targets = [{"sample": 0}, {"sample": 1}]

        for module in (
            _build_temporal_lightning_module(multi_scale=False),
            _build_temporal_lightning_module(do_random_resize_via_padding=True),
        ):
            captured: dict[str, object] = {}

            def capture_training_step(self, batch, batch_idx):
                captured["batch"] = batch
                return "delegated"

            module.on_train_batch_start((samples, targets), batch_idx=0)
            with patch.object(
                RFDETRModelModule,
                "training_step",
                capture_training_step,
            ):
                result = module.training_step((samples, targets), batch_idx=0)

            self.assertEqual(result, "delegated")
            captured_samples, captured_targets = captured["batch"]
            self.assertIs(captured_samples, samples)
            self.assertIs(captured_targets, targets)

    def test_non_temporal_batch_delegates_to_upstream_batch_start(self):
        from rfdetr.training import RFDETRModelModule

        module = _build_temporal_lightning_module()
        samples = SimpleNamespace(delegated=False)
        targets: list[dict[str, object]] = []

        def upstream_batch_start(self, batch, batch_idx):
            batch[0].delegated = True

        with patch.object(
            RFDETRModelModule,
            "on_train_batch_start",
            upstream_batch_start,
        ):
            module.on_train_batch_start((samples, targets), batch_idx=7)

        self.assertTrue(samples.delegated)


if __name__ == "__main__":
    unittest.main()
