"""Regression tests for temporal dataset class-count alignment."""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import train_rf_detr_model as trainer  # noqa: E402


class TemporalNumClassesAlignmentTest(unittest.TestCase):
    @staticmethod
    def _rf_model(*, num_classes: int, explicit: bool, context_num_classes: int):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                num_classes=num_classes,
                model_fields_set={"num_classes"} if explicit else set(),
            ),
            model=SimpleNamespace(
                args=SimpleNamespace(num_classes=context_num_classes),
            ),
        )

    def test_unset_num_classes_aligns_model_config_and_context_to_temporal_names(self):
        rf_model = self._rf_model(
            num_classes=90,
            explicit=False,
            context_num_classes=90,
        )

        trainer._align_temporal_num_classes(rf_model, ["soccer_ball"])

        self.assertEqual(rf_model.model_config.num_classes, 1)
        self.assertEqual(rf_model.model.args.num_classes, 1)

    def test_matching_num_classes_still_synchronizes_model_context(self):
        rf_model = self._rf_model(
            num_classes=1,
            explicit=False,
            context_num_classes=90,
        )

        trainer._align_temporal_num_classes(rf_model, ["soccer_ball"])

        self.assertEqual(rf_model.model_config.num_classes, 1)
        self.assertEqual(rf_model.model.args.num_classes, 1)

    def test_explicit_mismatch_warns_and_preserves_model_setting(self):
        rf_model = self._rf_model(
            num_classes=3,
            explicit=True,
            context_num_classes=3,
        )

        with self.assertWarnsRegex(UserWarning, "explicitly set to 3"):
            trainer._align_temporal_num_classes(rf_model, ["soccer_ball"])

        self.assertEqual(rf_model.model_config.num_classes, 3)
        self.assertEqual(rf_model.model.args.num_classes, 3)

    def test_temporal_dataset_requires_at_least_one_class_name(self):
        rf_model = self._rf_model(
            num_classes=90,
            explicit=False,
            context_num_classes=90,
        )

        with self.assertRaisesRegex(ValueError, "at least one dataset class name"):
            trainer._align_temporal_num_classes(rf_model, [])


class TemporalTrainingOrchestrationTest(unittest.TestCase):
    def test_temporal_classes_align_after_stock_guard_and_before_module_build(self):
        class StopAfterTemporalModuleBuild(Exception):
            pass

        events: list[str] = []
        motion_config = {
            "enabled": True,
            "type": "tracknet_v5",
            "temporal": {"mode": "real"},
        }
        config = {
            "runtime": {"verbose": False},
            "model": {
                "size": "small",
                "p2": {"enabled": True},
                "motion": motion_config,
            },
            "train": {},
            "trainer": {"extra_trainer_args": {}},
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint = directory_path / "stock.pth"
            checkpoint.touch()
            model_config = SimpleNamespace(
                pretrain_weights=checkpoint,
                num_classes=90,
                model_fields_set=set(),
            )
            core_model = SimpleNamespace(args=SimpleNamespace(num_classes=90))
            train_config = SimpleNamespace(
                batch_size=1,
                dataset_dir=str(directory_path),
                resume=None,
            )
            rf_model = SimpleNamespace(
                model=core_model,
                model_config=model_config,
                get_train_config=MagicMock(return_value=train_config),
            )
            model_cls = MagicMock(return_value=rf_model)
            model_cls.__name__ = "FakeRFDETR"

            p2_module = ModuleType("rf_detr_p2")

            def guard(*_args, **_kwargs):
                events.append("p2_guard")

            p2_module.assert_p2_training_checkpoint_compatible = guard
            motion_module = ModuleType("rf_detr_motion")

            def attach(*_args, **_kwargs):
                events.append("tracknet_attach")

            motion_module.attach_motion_module = attach
            temporal_runtime = ModuleType("rf_detr_temporal_runtime")

            def build_datamodule(*_args, **_kwargs):
                events.append("datamodule")
                return SimpleNamespace(class_names=["soccer_ball"])

            def build_module(received_model_config, *_args, **_kwargs):
                events.append("module")
                self.assertEqual(received_model_config.num_classes, 1)
                self.assertEqual(core_model.args.num_classes, 1)
                raise StopAfterTemporalModuleBuild

            temporal_runtime.build_temporal_datamodule = build_datamodule
            temporal_runtime.build_temporal_model_module = build_module
            training_module = ModuleType("rfdetr.training")
            training_module.RFDETRDataModule = object
            training_module.RFDETRModelModule = object
            training_module.build_trainer = MagicMock()
            auto_batch_module = ModuleType("rfdetr.training.auto_batch")
            auto_batch_module.resolve_auto_batch_config = MagicMock()
            original_align = trainer._align_temporal_num_classes

            def align(*args, **kwargs):
                events.append("align")
                return original_align(*args, **kwargs)

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        sys,
                        "argv",
                        ["train_rf_detr_model.py", "--config", str(Path(__file__).resolve())],
                    )
                )
                stack.enter_context(
                    patch.dict(
                        sys.modules,
                        {
                            "rf_detr_p2": p2_module,
                            "rf_detr_motion": motion_module,
                            "rf_detr_temporal_runtime": temporal_runtime,
                            "rfdetr.training": training_module,
                            "rfdetr.training.auto_batch": auto_batch_module,
                        },
                    )
                )
                stack.enter_context(patch.object(trainer, "is_nonzero_distributed_process", return_value=True))
                stack.enter_context(patch.object(trainer, "load_yaml", return_value=config))
                stack.enter_context(
                    patch.object(
                        trainer,
                        "apply_distributed_child_runtime_overrides",
                        return_value={},
                    )
                )
                stack.enter_context(
                    patch.object(
                        trainer,
                        "build_output_dir",
                        return_value=directory_path / "out",
                    )
                )
                stack.enter_context(patch.object(trainer, "export_distributed_child_runtime"))
                stack.enter_context(patch.object(trainer, "ensure_rfdetr_detection_hflip_support"))
                stack.enter_context(patch.object(trainer, "get_model_class", return_value=model_cls))
                stack.enter_context(patch.object(trainer, "build_model_kwargs", return_value={}))
                stack.enter_context(
                    patch.object(
                        trainer,
                        "build_train_kwargs",
                        return_value={"_device": "cpu"},
                    )
                )
                stack.enter_context(
                    patch.object(
                        trainer,
                        "parse_device_to_trainer_kwargs",
                        return_value={},
                    )
                )
                stack.enter_context(patch.object(trainer, "apply_validation_interval_to_trainer_kwargs"))
                stack.enter_context(patch.object(trainer, "apply_multigpu_ddp_strategy"))
                stack.enter_context(patch.object(trainer, "apply_multigpu_validation_safety"))
                stack.enter_context(
                    patch.object(
                        trainer,
                        "build_pitchobjectlab_architecture",
                        return_value={},
                    )
                )
                stack.enter_context(
                    patch.object(
                        trainer,
                        "_align_temporal_num_classes",
                        side_effect=align,
                    )
                )

                with self.assertRaises(StopAfterTemporalModuleBuild):
                    trainer._main_impl()

        self.assertEqual(
            events,
            ["p2_guard", "tracknet_attach", "datamodule", "align", "module"],
        )


if __name__ == "__main__":
    unittest.main()
