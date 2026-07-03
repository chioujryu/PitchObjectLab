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

from pathlib import Path
import sys
import unittest

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


class P2WeightLoaderFilterTest(unittest.TestCase):
    """The patched LWDETR.load_state_dict drops size-mismatched tensors on non-strict loads.

    Stock RF-DETR detection/seg checkpoints are single-scale (projector_scale ["P4"]);
    enabling P2 resizes the deformable-attention Linear layers + projector first stage,
    which torch's strict=False would otherwise reject. The patched method is generic over
    ``self`` (uses ``self.state_dict()`` + the base loader), so it is exercised here on a
    plain ``nn.Linear`` rather than a full (heavy) LWDETR build.
    """

    def test_loader_is_patched_on_lwdetr(self):
        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.models.lwdetr as lwdetr_module

        patched = lwdetr_module.LWDETR.__dict__.get("load_state_dict")
        self.assertIsNotNone(patched)
        self.assertTrue(getattr(patched, "_p2_drops_mismatch", False))

    def test_non_strict_drops_mismatch_keeps_matching(self):
        import torch
        from torch import nn

        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.models.lwdetr as lwdetr_module

        patched = lwdetr_module.LWDETR.__dict__["load_state_dict"]
        module = nn.Linear(4, 3)  # weight [3, 4], bias [3]
        state_dict = {
            "weight": torch.zeros(9, 4),  # size-mismatched -> must be dropped
            "bias": torch.ones(3),        # matches -> must be loaded
        }
        with self.assertWarns(UserWarning):
            result = patched(module, state_dict, strict=False)
        self.assertIn("weight", result.missing_keys)  # dropped -> reported missing
        self.assertTrue(torch.allclose(module.bias, torch.ones(3)))

    def test_strict_load_is_not_filtered(self):
        import torch
        from torch import nn

        rf_detr_p2.ensure_p2_support({"enabled": True})
        import rfdetr.models.lwdetr as lwdetr_module

        patched = lwdetr_module.LWDETR.__dict__["load_state_dict"]
        module = nn.Linear(4, 3)
        state_dict = {"weight": torch.zeros(9, 4), "bias": torch.zeros(3)}
        with self.assertRaises(RuntimeError):  # strict=True keeps base behavior (no filtering)
            patched(module, state_dict, strict=True)


if __name__ == "__main__":
    unittest.main()
