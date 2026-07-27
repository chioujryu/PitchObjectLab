"""Tests for the pluggable P2 (stride-4) feature level in rf_detr_p2.py.

These cover the config helpers and the in-process rfdetr patch:
* projector_scale validation/normalization,
* p2.overrides merging,
* build_model_kwargs wiring (enabled vs disabled),
* the relaxed projector_scale Literal,
* the patched Backbone actually building a P2 (scale-factor 4.0) branch and
  producing the right number/ratio of feature maps,
* the LWDETR.load_state_dict filter that drops size-mismatched pretrained tensors
  on non-strict loads (so a P2 feature-level-count change does not raise),
* idempotency.

The patch is process-global and only widens behavior (extra allowed scales; a
faithful Backbone.__init__ copy that reproduces upstream for P3/P4/P5), so it does
not affect other test modules in the same session.
"""

from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import rf_detr_p2  # noqa: E402
import train_rf_detr_model as trainer  # noqa: E402


class P2ConfigHelperTest(unittest.TestCase):
    def test_default_projector_scale(self):
        self.assertEqual(rf_detr_p2.resolve_p2_projector_scale({}), ["P2", "P3", "P4"])
        self.assertEqual(rf_detr_p2.resolve_p2_projector_scale(None), ["P2", "P3", "P4"])

    def test_projector_scale_normalizes_case_and_keeps_subsets(self):
        self.assertEqual(
            rf_detr_p2.resolve_p2_projector_scale({"projector_scale": ["p2", "p4"]}),
            ["P2", "P4"],
        )
        self.assertEqual(
            rf_detr_p2.resolve_p2_projector_scale({"projector_scale": ["P2", "P3", "P4", "P5"]}),
            ["P2", "P3", "P4", "P5"],
        )

    def test_projector_scale_rejects_invalid_level(self):
        with self.assertRaises(ValueError):
            rf_detr_p2.resolve_p2_projector_scale({"projector_scale": ["P2", "P9"]})

    def test_projector_scale_requires_ascending_order(self):
        with self.assertRaises(ValueError):
            rf_detr_p2.resolve_p2_projector_scale({"projector_scale": ["P3", "P2"]})

    def test_projector_scale_warns_when_p2_missing(self):
        with self.assertWarns(UserWarning):
            result = rf_detr_p2.resolve_p2_projector_scale({"projector_scale": ["P3", "P4"]})
        self.assertEqual(result, ["P3", "P4"])

    def test_apply_overrides_only_non_null(self):
        kwargs = {}
        rf_detr_p2.apply_p2_overrides(
            kwargs,
            {"overrides": {"num_queries": 500, "gradient_checkpointing": True, "resolution": None}},
        )
        self.assertEqual(kwargs["num_queries"], 500)
        self.assertIs(kwargs["gradient_checkpointing"], True)
        self.assertNotIn("resolution", kwargs)


class BuildModelKwargsP2Test(unittest.TestCase):
    def test_disabled_is_inert(self):
        kwargs = trainer.build_model_kwargs({"model": {"size": "large"}})
        self.assertNotIn("projector_scale", kwargs)

    def test_enabled_injects_scale_and_overrides(self):
        kwargs = trainer.build_model_kwargs(
            {
                "model": {
                    "size": "large",
                    "p2": {
                        "enabled": True,
                        "projector_scale": ["P2", "P3", "P4"],
                        "overrides": {"num_queries": 500, "num_select": 500},
                    },
                }
            }
        )
        self.assertEqual(kwargs["projector_scale"], ["P2", "P3", "P4"])
        self.assertEqual(kwargs["num_queries"], 500)
        self.assertEqual(kwargs["num_select"], 500)
        self.assertTrue(rf_detr_p2.is_patched())

    def test_overrides_resolution_wins_over_model_resolution(self):
        kwargs = trainer.build_model_kwargs(
            {
                "model": {
                    "size": "large",
                    "resolution": 704,
                    "p2": {"enabled": True, "overrides": {"resolution": 960}},
                }
            }
        )
        self.assertEqual(kwargs["resolution"], 960)


class P2PatchTest(unittest.TestCase):
    def test_literal_relaxed_allows_p2(self):
        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.config as rf_config

        # Large's projector_scale is the narrowest Literal upstream (only "P4").
        cfg = rf_config.RFDETRLargeConfig(projector_scale=["P2", "P3", "P4"], pretrain_weights=None)
        self.assertEqual(cfg.projector_scale, ["P2", "P3", "P4"])

    def test_literal_still_rejects_unknown_level(self):
        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.config as rf_config
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            rf_config.RFDETRLargeConfig(projector_scale=["P9"], pretrain_weights=None)

    def test_idempotent_patch(self):
        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.models.backbone.backbone as backbone_module

        first = backbone_module.Backbone.__init__
        rf_detr_p2.ensure_p2_support({"enabled": True})
        self.assertIs(backbone_module.Backbone.__init__, first)
        self.assertIs(first, rf_detr_p2._PATCHED_INIT)

    def test_forward_export_has_static_shape_stabilization_marker(self):
        rf_detr_p2.ensure_p2_support({"enabled": True})
        from rfdetr.models.backbone.backbone import Backbone

        self.assertTrue(getattr(Backbone.forward_export, "_p2_static_export_shapes", False))

    def test_export_stabilizer_preserves_values_and_fixed_chw(self):
        first = torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3)
        second = first + 100.0

        stabilized = rf_detr_p2._stabilize_raw_features_for_export(
            [first, second], channels=[4, 4], height=3, width=3
        )

        self.assertEqual([tuple(value.shape) for value in stabilized], [(2, 4, 3, 3)] * 2)
        self.assertTrue(torch.equal(stabilized[0], first))
        self.assertTrue(torch.equal(stabilized[1], second))

    def test_backbone_builds_and_forwards_p2(self):
        import torch
        from torch import nn

        rf_detr_p2.ensure_p2_support({"enabled": True})
        from rfdetr.models.backbone.backbone import Backbone
        from rfdetr.utilities.tensors import NestedTensor

        resolution = 256  # divisible by patch_size(16) * num_windows(2); fast on CPU
        backbone = Backbone(
            name="dinov2_windowed_small",
            out_feature_indexes=[3, 6, 9, 12],
            projector_scale=["P2", "P3", "P4"],
            out_channels=256,
            layer_norm=True,
            target_shape=(resolution, resolution),
            load_dinov2_weights=False,  # random init -> no network/download
            patch_size=16,
            num_windows=2,
            positional_encoding_size=resolution // 16,
        )

        # Structural: 3 projector stages; P2 stage uses two ConvTranspose2d per input branch.
        self.assertEqual(len(backbone.projector.stages), 3)
        p2_branch = backbone.projector.stages_sampling[0][0]
        n_deconv = sum(1 for m in p2_branch.modules() if isinstance(m, nn.ConvTranspose2d))
        self.assertEqual(n_deconv, 2)

        # Runtime: forward yields 3 feature maps in descending stride (P2>P3>P4), 2x apart.
        backbone.eval()
        tensors = torch.zeros(1, 3, resolution, resolution)
        mask = torch.zeros(1, resolution, resolution, dtype=torch.bool)
        with torch.no_grad():
            out, _cross = backbone(NestedTensor(tensors, mask))
        self.assertEqual(len(out), 3)
        h_p2 = out[0].tensors.shape[-1]
        h_p3 = out[1].tensors.shape[-1]
        h_p4 = out[2].tensors.shape[-1]
        self.assertEqual(h_p2, 2 * h_p3)
        self.assertEqual(h_p3, 2 * h_p4)
        self.assertEqual(h_p4, resolution // 16)  # P4 == native ViT stride-16 map

        with torch.no_grad():
            export_out, export_masks, _cross = backbone.forward_export(tensors)
        self.assertEqual(len(export_out), 3)
        self.assertEqual(len(export_masks), 3)
        self.assertEqual([value.shape[-1] for value in export_out], [h_p2, h_p3, h_p4])


class P2WeightLoaderFilterTest(unittest.TestCase):
    """The patched loader drops only known P2 mismatches on non-strict loads.

    Stock RF-DETR detection/seg checkpoints are single-scale (projector_scale ["P4"]);
    enabling P2 resizes the deformable-attention Linear layers + projector first stage,
    which torch's strict=False would otherwise reject. Non-P2 architecture mismatches
    continue to raise instead of being silently discarded.
    """

    @staticmethod
    def _module():
        from torch import nn

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.projector_scale = ["P2", "P3", "P4"]
                self.projector = nn.Linear(4, 3)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.ModuleList([Backbone()])
                self.class_embed = nn.Linear(4, 3)

        return Model()

    def test_loader_is_patched_on_lwdetr(self):
        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.models.lwdetr as lwdetr_module

        patched = lwdetr_module.LWDETR.__dict__.get("load_state_dict")
        self.assertIsNotNone(patched)
        self.assertTrue(getattr(patched, "_p2_drops_mismatch", False))

    def test_non_strict_drops_mismatch_keeps_matching(self):
        import torch

        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.models.lwdetr as lwdetr_module

        patched = lwdetr_module.LWDETR.__dict__["load_state_dict"]
        module = self._module()
        state_dict = {key: value.clone() for key, value in module.state_dict().items()}
        state_dict["backbone.0.projector.weight"] = torch.zeros(9, 4)
        state_dict["backbone.0.projector.bias"] = torch.ones(3)
        with self.assertWarns(UserWarning):
            result = patched(module, state_dict, strict=False)
        self.assertIn("backbone.0.projector.weight", result.missing_keys)
        self.assertTrue(torch.allclose(module.backbone[0].projector.bias, torch.ones(3)))

    def test_non_strict_does_not_drop_non_p2_mismatch(self):
        import torch

        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.models.lwdetr as lwdetr_module

        patched = lwdetr_module.LWDETR.__dict__["load_state_dict"]
        module = self._module()
        state_dict = {key: value.clone() for key, value in module.state_dict().items()}
        state_dict["class_embed.weight"] = torch.zeros(9, 4)
        with self.assertRaises(RuntimeError):
            patched(module, state_dict, strict=False)

    def test_strict_load_is_not_filtered(self):
        import torch

        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.models.lwdetr as lwdetr_module

        patched = lwdetr_module.LWDETR.__dict__["load_state_dict"]
        module = self._module()
        state_dict = {key: value.clone() for key, value in module.state_dict().items()}
        state_dict["backbone.0.projector.weight"] = torch.zeros(9, 4)
        with self.assertRaises(RuntimeError):  # strict=True keeps base behavior (no filtering)
            patched(module, state_dict, strict=True)


class P2CheckpointCompatibilityTest(unittest.TestCase):
    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.projector_scale = ["P2", "P3", "P4"]
            self.projector = nn.Linear(4, 3)

    class _Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.level_embed = nn.Parameter(torch.zeros(3, 4))
            self.cross_attn = nn.Module()
            self.cross_attn.sampling_offsets = nn.Linear(4, 12, bias=False)

    class _LWDETR(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.ModuleList([P2CheckpointCompatibilityTest._Backbone()])
            self.transformer = P2CheckpointCompatibilityTest._Transformer()
            self.class_embed = nn.Linear(4, 2)

    @staticmethod
    def _metadata(projector_scale=None):
        return {
            "schema_version": 1,
            "model_size": "medium",
            "p2": {
                "enabled": True,
                "projector_scale": projector_scale or ["P2", "P3", "P4"],
            },
        }

    @staticmethod
    def _record_successful_stock_load(model, *, dropped=None):
        model._pitchobjectlab_p2_load_report = {
            "schema_version": 1,
            "projector_scale": ["P2", "P3", "P4"],
            "dropped_p2_mismatches": dropped
            or ["transformer.cross_attn.sampling_offsets.weight"],
            "loaded": True,
            "missing_keys": [],
            "unexpected_keys": [],
        }

    def test_legacy_checkpoint_loads_when_p2_shapes_are_exact(self):
        model = self._LWDETR()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy_p2.pth"
            torch.save({"model": model.state_dict()}, checkpoint)

            rf_detr_p2.assert_p2_checkpoint_compatible(model, checkpoint)

    def test_compiled_wrapper_is_unwrapped_for_exact_checkpoint_validation(self):
        model = self._LWDETR()
        compiled = SimpleNamespace(_orig_mod=model)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "compiled_p2.pth"
            torch.save({"model": model.state_dict()}, checkpoint)

            rf_detr_p2.assert_p2_checkpoint_compatible(compiled, checkpoint)

    def test_legacy_checkpoint_rejects_p2_shape_mismatch(self):
        model = self._LWDETR()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state["backbone.0.projector.weight"] = torch.zeros(9, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mismatched_p2.pth"
            torch.save({"model": state}, checkpoint)

            with self.assertRaisesRegex(RuntimeError, "shape_mismatch"):
                rf_detr_p2.assert_p2_checkpoint_compatible(model, checkpoint)

    def test_checkpoint_rejects_p2_metadata_mismatch(self):
        model = self._LWDETR()
        expected = self._metadata()
        saved = self._metadata(["P2", "P4"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "metadata_mismatch.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "pitchobjectlab_architecture": saved,
                },
                checkpoint,
            )

            with self.assertRaisesRegex(RuntimeError, "P2 architecture metadata"):
                rf_detr_p2.assert_p2_checkpoint_compatible(
                    model, checkpoint, expected_architecture=expected
                )

    def test_training_guard_allows_single_level_stock_initialization(self):
        model = self._LWDETR()
        self._record_successful_stock_load(model)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state["transformer.cross_attn.sampling_offsets.weight"] = torch.zeros(4, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stock_single_level.pth"
            torch.save({"model": state}, checkpoint)

            rf_detr_p2.assert_p2_training_checkpoint_compatible(
                model,
                checkpoint,
                allow_stock_initialization=True,
            )

    def test_training_guard_allows_missing_empty_keypoint_mask_in_stock_initialization(self):
        model = self._LWDETR()
        self._record_successful_stock_load(model)
        model.register_buffer("_kp_active_mask", torch.zeros((0, 0), dtype=torch.bool))
        state = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if key != "_kp_active_mask"
        }
        state["transformer.cross_attn.sampling_offsets.weight"] = torch.zeros(4, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stock_without_empty_keypoint_mask.pth"
            torch.save({"model": state}, checkpoint)

            rf_detr_p2.assert_p2_training_checkpoint_compatible(
                model,
                checkpoint,
                allow_stock_initialization=True,
            )

    def test_training_guard_rejects_unsuccessful_normalized_load_report(self):
        model = self._LWDETR()
        model._pitchobjectlab_p2_load_report = {
            "schema_version": 1,
            "projector_scale": ["P2", "P3", "P4"],
            "dropped_p2_mismatches": [
                "transformer.cross_attn.sampling_offsets.weight"
            ],
            "loaded": False,
            "error": "RuntimeError: simulated non-P2 shape mismatch",
        }
        model.register_buffer("_kp_active_mask", torch.ones((1, 1), dtype=torch.bool))
        state = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if key != "_kp_active_mask"
        }
        state["transformer.cross_attn.sampling_offsets.weight"] = torch.zeros(4, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stock_without_nonempty_keypoint_mask.pth"
            torch.save({"model": state}, checkpoint)

            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                rf_detr_p2.assert_p2_training_checkpoint_compatible(
                    model,
                    checkpoint,
                    allow_stock_initialization=True,
                )

    def test_training_guard_rejects_stock_checkpoint_for_resume(self):
        model = self._LWDETR()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state["transformer.cross_attn.sampling_offsets.weight"] = torch.zeros(4, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stock_resume.pth"
            torch.save({"model": state}, checkpoint)

            with self.assertRaisesRegex(RuntimeError, "shape_mismatch"):
                rf_detr_p2.assert_p2_training_checkpoint_compatible(
                    model,
                    checkpoint,
                    allow_stock_initialization=False,
                )

    def test_training_guard_rejects_incompatible_p2_initialization(self):
        model = self._LWDETR()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state["transformer.cross_attn.sampling_offsets.weight"] = torch.zeros(8, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "incompatible_p2.pth"
            torch.save({"model": state}, checkpoint)

            with self.assertRaisesRegex(RuntimeError, "shape_mismatch"):
                rf_detr_p2.assert_p2_training_checkpoint_compatible(
                    model,
                    checkpoint,
                    allow_stock_initialization=True,
                )

    def test_training_guard_rejects_non_p2_filtered_tensor_report(self):
        model = self._LWDETR()
        self._record_successful_stock_load(
            model,
            dropped=["class_embed.weight"],
        )
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state["transformer.cross_attn.sampling_offsets.weight"] = torch.zeros(4, 4)
        state["class_embed.weight"] = torch.zeros(9, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "wrong_class_stock.pth"
            torch.save({"model": state}, checkpoint)

            with self.assertRaisesRegex(RuntimeError, "non-P2 filtered tensors"):
                rf_detr_p2.assert_p2_training_checkpoint_compatible(
                    model,
                    checkpoint,
                    allow_stock_initialization=True,
                )


class TrainingMainFlowOrderTest(unittest.TestCase):
    def test_p2_stock_guard_runs_before_motion_attach(self):
        class StopAfterMotionAttach(Exception):
            pass

        events = []
        motion_config = {"enabled": True, "type": "tracknet_v5"}
        config = {
            "runtime": {"verbose": False},
            "model": {
                "size": "small",
                "p2": {"enabled": True},
                "motion": motion_config,
            },
            "train": {},
            "trainer": {},
        }
        core_model = SimpleNamespace()

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            checkpoint = directory_path / "stock.pth"
            checkpoint.touch()
            rf_model = SimpleNamespace(
                model=core_model,
                model_config=SimpleNamespace(pretrain_weights=checkpoint),
            )
            model_cls = MagicMock(return_value=rf_model)
            model_cls.__name__ = "FakeRFDETR"

            def guard_side_effect(
                model,
                checkpoint_path,
                expected_architecture,
                *,
                allow_stock_initialization,
            ):
                self.assertIs(model, core_model)
                self.assertFalse(hasattr(model, "motion_module"))
                self.assertEqual(checkpoint_path, checkpoint)
                self.assertEqual(expected_architecture, {})
                self.assertTrue(allow_stock_initialization)
                events.append("guard")

            def attach_side_effect(model, received_config):
                self.assertIs(model, core_model)
                self.assertIs(received_config, motion_config)
                model.motion_module = object()
                events.append("attach")
                raise StopAfterMotionAttach

            p2_module = ModuleType("rf_detr_p2")
            p2_module.assert_p2_training_checkpoint_compatible = MagicMock(
                side_effect=guard_side_effect
            )
            motion_module = ModuleType("rf_detr_motion")
            motion_module.attach_motion_module = MagicMock(side_effect=attach_side_effect)
            training_module = ModuleType("rfdetr.training")
            training_module.RFDETRDataModule = object
            training_module.RFDETRModelModule = object
            training_module.build_trainer = MagicMock()
            auto_batch_module = ModuleType("rfdetr.training.auto_batch")
            auto_batch_module.resolve_auto_batch_config = MagicMock()

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
                            "rfdetr.training": training_module,
                            "rfdetr.training.auto_batch": auto_batch_module,
                        },
                    )
                )
                stack.enter_context(
                    patch.object(trainer, "is_nonzero_distributed_process", return_value=True)
                )
                stack.enter_context(patch.object(trainer, "load_yaml", return_value=config))
                stack.enter_context(
                    patch.object(
                        trainer,
                        "apply_distributed_child_runtime_overrides",
                        return_value={},
                    )
                )
                stack.enter_context(
                    patch.object(trainer, "build_output_dir", return_value=directory_path / "out")
                )
                stack.enter_context(
                    patch.object(trainer, "ensure_rfdetr_detection_hflip_support")
                )
                stack.enter_context(
                    patch.object(trainer, "get_model_class", return_value=model_cls)
                )
                stack.enter_context(
                    patch.object(trainer, "build_model_kwargs", return_value={})
                )
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
                stack.enter_context(
                    patch.object(trainer, "apply_validation_interval_to_trainer_kwargs")
                )
                stack.enter_context(patch.object(trainer, "apply_multigpu_ddp_strategy"))
                stack.enter_context(
                    patch.object(trainer, "apply_multigpu_validation_safety")
                )
                stack.enter_context(
                    patch.object(
                        trainer,
                        "build_pitchobjectlab_architecture",
                        return_value={},
                    )
                )

                with self.assertRaises(StopAfterMotionAttach):
                    trainer._main_impl()

        self.assertEqual(events, ["guard", "attach"])
        p2_module.assert_p2_training_checkpoint_compatible.assert_called_once()
        motion_module.attach_motion_module.assert_called_once_with(core_model, motion_config)


class CheckpointArchitectureMetadataTest(unittest.TestCase):
    @staticmethod
    def _metadata():
        return {
            "schema_version": 1,
            "model_size": "medium",
            "p2": {"enabled": True, "projector_scale": ["P2", "P3", "P4"]},
            "motion": {"enabled": True, "type": "tracknet_v5"},
            "tensorrt_export_abi": 2,
        }

    def test_best_checkpoint_callback_embeds_frozen_metadata(self):
        class BestModelCallback:
            def _build_checkpoint_payload(self):
                return {"model": {"weight": torch.ones(1)}, "args": {"epochs": 1}}

        callback = BestModelCallback()
        metadata = self._metadata()
        expected = self._metadata()
        trainer.install_best_checkpoint_metadata(
            SimpleNamespace(callbacks=[callback]),
            metadata,
        )
        metadata["p2"]["projector_scale"].append("P5")

        payload = callback._build_checkpoint_payload()
        key = trainer.PITCHOBJECTLAB_ARCHITECTURE_KEY
        self.assertEqual(payload[key], expected)
        self.assertEqual(payload["args"][key], expected)
        self.assertIsNot(payload[key], payload["args"][key])

    def test_enrichment_restores_top_level_metadata_from_nested_args(self):
        key = trainer.PITCHOBJECTLAB_ARCHITECTURE_KEY
        metadata = self._metadata()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = [
                output_dir / "checkpoint_best_regular.pth",
                output_dir / "checkpoint_best_ema.pth",
                output_dir / "checkpoint_best_total.pth",
            ]
            for path in paths:
                torch.save({"model": {}, "args": {key: metadata}}, path)

            trainer.enrich_best_checkpoint_metadata(output_dir)

            for path in paths:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(payload[key], metadata)
                self.assertEqual(payload["args"][key], metadata)


if __name__ == "__main__":
    unittest.main()
