from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_rf_detr_model as trainer


def _fake_detection_args(dataset_dir: Path, *, include_keypoints: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=str(dataset_dir),
        square_resize_div_64=True,
        segmentation_head=False,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        patch_size=16,
        num_windows=2,
        aug_config={"HorizontalFlip": {"p": 1.0}},
        augmentation_backend="cpu",
        use_grouppose_keypoints=include_keypoints,
        num_keypoints_per_class=[17] if include_keypoints else [],
        keypoint_flip_pairs=[],
    )


def _transform_names(transform_pipeline) -> list[str]:
    transforms = getattr(transform_pipeline, "transforms", [])
    return [repr(transform) for transform in transforms]


class TrainingAugmentationBBoxAlignmentTest(unittest.TestCase):
    def test_detection_training_keeps_horizontal_flip_augmentation_enabled(self):
        trainer.ensure_rfdetr_detection_hflip_support()
        import rfdetr.datasets.coco as coco_module

        captured: dict[str, object] = {}

        class FakeCocoDetection:
            def __init__(self, img_folder, ann_file, transforms, **kwargs):
                del img_folder, ann_file
                captured["transforms"] = transforms
                captured.update(kwargs)

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(coco_module, "CocoDetection", FakeCocoDetection):
                coco_module.build_roboflow_from_coco("train", _fake_detection_args(Path(temp)), 64)

        self.assertFalse(captured["include_keypoints"])
        self.assertTrue(any("HorizontalFlip" in name for name in _transform_names(captured["transforms"])))

    def test_keypoint_training_without_flip_pairs_still_disables_horizontal_flip(self):
        trainer.ensure_rfdetr_detection_hflip_support()
        import rfdetr.datasets.coco as coco_module

        captured: dict[str, object] = {}

        class FakeCocoDetection:
            def __init__(self, img_folder, ann_file, transforms, **kwargs):
                del img_folder, ann_file
                captured["transforms"] = transforms
                captured.update(kwargs)

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(coco_module, "CocoDetection", FakeCocoDetection):
                coco_module.build_roboflow_from_coco(
                    "train",
                    _fake_detection_args(Path(temp), include_keypoints=True),
                    64,
                )

        self.assertTrue(captured["include_keypoints"])
        self.assertFalse(any("HorizontalFlip" in name for name in _transform_names(captured["transforms"])))

    def test_geometric_training_augmentations_keep_box_on_transformed_object(self):
        from rfdetr.datasets.transforms import AlbumentationsWrapper

        image = Image.new("RGB", (100, 100), color=(0, 0, 0))
        ImageDraw.Draw(image).rectangle([20, 30, 59, 69], fill=(255, 0, 0))
        target = {
            "boxes": torch.tensor([[20.0, 30.0, 60.0, 70.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "size": torch.tensor([100, 100]),
            "orig_size": torch.tensor([100, 100]),
        }
        wrappers = AlbumentationsWrapper.from_config(
            [
                {"HorizontalFlip": {"p": 1.0}},
                {
                    "Affine": {
                        "translate_px": {"x": [5, 5], "y": [3, 3]},
                        "rotate": 0,
                        "shear": 0,
                        "scale": 1.0,
                        "interpolation": 0,
                        "border_mode": 0,
                        "fill": [0, 0, 0],
                        "p": 1.0,
                    }
                },
            ],
            keypoint_flip_pairs=None,
        )

        for wrapper in wrappers:
            image, target = wrapper(image, target)

        pixels = np.asarray(image)
        red = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] < 50) & (pixels[:, :, 2] < 50)
        ys, xs = np.where(red)
        pixel_box = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        target_box = target["boxes"][0].detach().cpu().numpy()

        np.testing.assert_allclose(target_box, pixel_box, atol=1.0)

    def test_pixel_only_training_augmentations_do_not_move_boxes(self):
        from rfdetr.datasets.transforms import AlbumentationsWrapper

        image = Image.new("RGB", (100, 100), color=(32, 64, 96))
        target = {
            "boxes": torch.tensor([[20.0, 30.0, 60.0, 70.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "size": torch.tensor([100, 100]),
        }
        original_boxes = target["boxes"].clone()
        wrappers = AlbumentationsWrapper.from_config(
            [
                {"RandomBrightnessContrast": {"brightness_limit": 0.1, "contrast_limit": 0.1, "p": 1.0}},
                {"GaussianBlur": {"blur_limit": 3, "p": 1.0}},
            ],
            keypoint_flip_pairs=None,
        )

        for wrapper in wrappers:
            image, target = wrapper(image, target)

        torch.testing.assert_close(target["boxes"], original_boxes)


if __name__ == "__main__":
    unittest.main()
