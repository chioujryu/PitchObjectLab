from __future__ import annotations

from dfine_seg_trainer.augmentation import build_aug_runtime
from dfine_seg_trainer.common import build_output_dir, parse_device_ids, parse_extra_args


def test_output_placeholders_render_with_safe_names() -> None:
    config = {
        "model": {"task": "segment", "name": "s"},
        "dataset": {"dataset_dir": "D:/datasets/my data", "source_format": "auto"},
        "train": {"epochs": 5, "batch_size": 2, "device": "cuda:0", "gpus": [0], "img_size": [640, 640]},
        "output": {
            "root": "runs/detect/dfine_seg/train",
            "name": "{task}_{model_name}_{dataset_name}_{epochs}_{timestamp}",
        },
    }
    out = build_output_dir(config, "20260525112233", "global")
    assert out.name == "segment_s_my_data_5_20260525112233"


def test_parse_extra_args_builds_nested_mapping() -> None:
    parsed = parse_extra_args(["dataset.box_to_mask=true", "augmentation.mixup=0.2"])
    assert parsed["dataset"]["box_to_mask"] is True
    assert parsed["augmentation"]["mixup"] == 0.2


def test_device_parser_accepts_cuda_forms() -> None:
    assert parse_device_ids("0,1") == [0, 1]
    assert parse_device_ids("cuda:0") == [0]
    assert parse_device_ids("cpu") == []


def test_augmentation_builder_maps_ultralytics_like_values() -> None:
    runtime = build_aug_runtime(
        {
            "preset": "light",
            "hsv_h": 0.02,
            "hsv_s": 0.5,
            "hsv_v": 0.5,
            "degrees": 12,
            "translate": 0.15,
            "scale": 0.25,
            "mosaic": 0.7,
            "copy_paste": 0.3,
        },
        task="segment",
    )
    assert runtime["mosaic_augs"]["mosaic_prob"] == 0.7
    assert runtime["mosaic_augs"]["degrees"] == 12.0
    assert runtime["augs"]["hsv_h"] == 0.02
    assert runtime["augs"]["copy_paste"] == 0.3
