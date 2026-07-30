from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_rf_detr_model as trainer  # noqa: E402


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
    def test_subpixel_degenerate_box_is_filtered_with_instance_fields_aligned(self):
        trainer.ensure_rfdetr_invalid_bbox_filter_support()
        # Calling the installer repeatedly must not wrap the constructor twice.
        trainer.ensure_rfdetr_invalid_bbox_filter_support()

        from albumentations import HorizontalFlip
        from rfdetr.datasets.transforms import AlbumentationsWrapper

        image_size = 480
        y_min = np.float32(0.12399024)
        y_max = np.nextafter(y_min, np.float32(np.inf))
        self.assertGreater(y_max, y_min)
        self.assertEqual(np.float32(y_min / image_size), np.float32(y_max / image_size))

        masks = torch.zeros((2, image_size, image_size), dtype=torch.bool)
        masks[1, 100:120, 100:120] = True
        target = {
            "boxes": torch.tensor(
                [
                    [10.0, y_min, 20.0, y_max],
                    [100.0, 100.0, 120.0, 120.0],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([3, 7], dtype=torch.int64),
            "area": torch.tensor([float(y_max - y_min) * 10.0, 400.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0, 0], dtype=torch.int64),
            "masks": masks,
            "size": torch.tensor([image_size, image_size]),
            "orig_size": torch.tensor([image_size, image_size]),
        }

        wrapper = AlbumentationsWrapper(HorizontalFlip(p=1.0))
        bbox_params = wrapper.transform.processors["bboxes"].params
        self.assertTrue(bbox_params.filter_invalid_bboxes)

        _, transformed = wrapper(Image.new("RGB", (image_size, image_size)), target)

        self.assertEqual(transformed["boxes"].shape, (1, 4))
        torch.testing.assert_close(
            transformed["boxes"][0],
            torch.tensor([360.0, 100.0, 380.0, 120.0]),
            atol=1e-4,
            rtol=0.0,
        )
        torch.testing.assert_close(transformed["labels"], torch.tensor([7]))
        torch.testing.assert_close(transformed["area"], torch.tensor([400.0]))
        torch.testing.assert_close(transformed["iscrowd"], torch.tensor([0]))
        self.assertEqual(transformed["masks"].shape, (1, image_size, image_size))
        self.assertEqual(int(transformed["masks"][0].sum()), 400)
        self.assertTrue(transformed["masks"][0, 100:120, 360:380].all())

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
