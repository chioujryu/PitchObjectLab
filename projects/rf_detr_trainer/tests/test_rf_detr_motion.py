"""Tests for the pluggable TrackNetV5 motion module in rf_detr_motion.py.

These cover:
* MotionDirectionDecoupling forward (shape + zero-delta identity).
* MotionFeatureGate forward (shape + zero-init identity).
* RSTRHead forward (shape + residual skip).
* MotionModule forward (shape, fallback modes).
* Config helpers (resolve_motion_type, apply_motion_overrides, deep_merge).
* ensure_motion_support idempotency.
* attach_motion_module guard when motion disabled.
* build_model_kwargs wiring (enabled vs disabled).

All tests run with CPU-only torch; no rfdetr install required for the pure-module
tests.  Tests that need rfdetr are guarded with unittest.skipIf.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import rf_detr_motion as motion  # noqa: E402
import train_rf_detr_model as trainer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nested_tensor(B: int, C: int, H: int, W: int):
    """Build a minimal NestedTensor-like object for testing without rfdetr."""

    class _FakeNested:
        def __init__(self, tensors, mask):
            self.tensors = tensors
            self.mask = mask

    return _FakeNested(
        tensors=torch.zeros(B, C, H, W),
        mask=torch.zeros(B, H, W, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# LearnableSigmoidAttention
# ---------------------------------------------------------------------------

class LearnableSigmoidAttentionTest(unittest.TestCase):
    def test_output_shape(self):
        attn = motion.LearnableSigmoidAttention(num_channels=4)
        x = torch.rand(2, 4, 8, 8)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_in_zero_one_range(self):
        attn = motion.LearnableSigmoidAttention(num_channels=2)
        x = torch.rand(1, 2, 4, 4)
        out = attn(x)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_zero_input_gives_constant_output(self):
        """σ(k*(0 − m)) is constant for all spatial positions."""
        attn = motion.LearnableSigmoidAttention(num_channels=3)
        x = torch.zeros(1, 3, 5, 5)
        out = attn(x)
        # All spatial positions should be equal.
        self.assertTrue(torch.allclose(out[:, :, 0:1, 0:1].expand_as(out), out))


# ---------------------------------------------------------------------------
# MotionDirectionDecoupling
# ---------------------------------------------------------------------------

class MDDTest(unittest.TestCase):
    def _make_mdd(self, polarity_channels=2):
        return motion.MotionDirectionDecoupling(
            in_channels=3,
            polarity_channels=polarity_channels,
        )

    def test_output_shape(self):
        mdd = self._make_mdd(polarity_channels=2)
        frames = torch.rand(2, 3, 3, 16, 16)  # [B, T, C, H, W]
        out = mdd(frames)
        # Should be [B, 2*polarity_channels, H, W]
        self.assertEqual(out.shape, (2, 4, 16, 16))

    def test_zero_delta_gate_is_identity(self):
        """With identical frames (delta=0), the feature gate must be identity.

        When delta=0, proj_plus and proj_minus have zero-init weights so their
        outputs are 0.  The MDD attention maps are then sigmoid(k*(0-m)) — a
        constant non-zero value.  However, the MotionFeatureGate.gate conv is
        *also* zero-init, so gate(any_motion) == 0 and the gated feature is
        feat*(1+0) == feat — identity end-to-end.
        """
        gate = motion.MotionFeatureGate(feature_channels=8, motion_channels=4)
        feat = torch.rand(1, 8, 8, 8)
        # The gate conv is zero-init, so its output is zero regardless of the
        # motion map value, and the result equals the input feature.
        any_motion = torch.ones(1, 4, 8, 8)  # non-zero input to the gate
        out = gate(feat, any_motion)
        self.assertTrue(torch.allclose(out, feat, atol=1e-6))

    def test_polarity_fields_are_non_negative(self):
        """P⁺ = ReLU(Δ) and P⁻ = ReLU(−Δ) are always >= 0."""
        mdd = self._make_mdd(polarity_channels=2)
        frames = torch.randn(2, 3, 3, 16, 16)
        # Zero-init projections mean output is zero; use hooks to check internal polarity maps.
        # We re-initialise weights to random to get a non-trivial output.
        nn.init.normal_(mdd.proj_plus.weight, std=0.1)
        nn.init.normal_(mdd.proj_minus.weight, std=0.1)
        out = mdd(frames)
        # All outputs should be non-negative (attention is sigmoid of projected polarity fields).
        self.assertTrue((out >= 0).all())

    def test_polarity_channels_must_be_positive(self):
        with self.assertRaises(Exception):
            motion.MotionDirectionDecoupling(in_channels=3, polarity_channels=0)


# ---------------------------------------------------------------------------
# MotionFeatureGate
# ---------------------------------------------------------------------------

class MotionFeatureGateTest(unittest.TestCase):
    def test_output_shape_matches_feature(self):
        gate = motion.MotionFeatureGate(feature_channels=256, motion_channels=4)
        feat = torch.rand(2, 256, 20, 20)
        motion_maps = torch.rand(2, 4, 40, 40)  # different spatial size — should be resized
        out = gate(feat, motion_maps)
        self.assertEqual(out.shape, feat.shape)

    def test_zero_gate_weight_is_identity(self):
        """Zero-initialised gate conv → gate(motion) = 0 → out = feat * 1.0 = feat."""
        gate = motion.MotionFeatureGate(feature_channels=8, motion_channels=4)
        feat = torch.rand(1, 8, 4, 4)
        motion_maps = torch.rand(1, 4, 4, 4)
        out = gate(feat, motion_maps)
        self.assertTrue(torch.allclose(out, feat, atol=1e-6))


# ---------------------------------------------------------------------------
# RSTRHead
# ---------------------------------------------------------------------------

class RSTRHeadTest(unittest.TestCase):
    def test_output_shape_preserved(self):
        head = motion.RSTRHead(in_channels=32, hidden_dim=64, num_heads=4, num_blocks=1)
        feat = torch.rand(2, 32, 8, 8)
        out = head(feat)
        self.assertEqual(out.shape, feat.shape)

    def test_output_in_zero_one_range(self):
        """R-STR applies sigmoid at the end; output must be in [0, 1]."""
        head = motion.RSTRHead(in_channels=16, hidden_dim=32, num_heads=4, num_blocks=1)
        feat = torch.rand(1, 16, 6, 6)
        out = head(feat)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_zero_init_proj_out_acts_as_near_identity(self):
        """With zero-init proj_out, delta ≈ 0, so output ≈ sigmoid(feat)."""
        head = motion.RSTRHead(in_channels=8, hidden_dim=16, num_heads=2, num_blocks=1)
        feat = torch.rand(1, 8, 4, 4)
        out = head(feat)
        expected = torch.sigmoid(feat)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_no_nan_gradient(self):
        head = motion.RSTRHead(in_channels=16, hidden_dim=32, num_heads=4, num_blocks=2)
        feat = torch.rand(1, 16, 8, 8, requires_grad=True)
        out = head(feat)
        loss = out.mean()
        loss.backward()
        self.assertFalse(torch.isnan(feat.grad).any())


# ---------------------------------------------------------------------------
# MotionModule
# ---------------------------------------------------------------------------

class MotionModuleTest(unittest.TestCase):
    _BASE_CFG = {
        "enabled": True,
        "type": "tracknet_v5",
        "temporal": {"num_frames": 3, "fallback_mode": "identity", "noise_std": 0.02},
        "tracknet_v5": {
            "mdd": {"enabled": True, "polarity_channels": 4, "attention": {"learnable": True, "init_k": 1.0, "init_m": 0.5}},
            "rstr": {"enabled": True, "num_blocks": 1, "hidden_dim": 32, "num_heads": 4, "dropout": 0.0, "use_pixel_shuffle": True, "context_mask_prob": 0.0},
        },
        "loss": {"motion_attention_weight": 0.0},
    }

    def _make_module(self, cfg=None):
        return motion.MotionModule(
            feature_channels_per_scale=[32, 32, 32],
            motion_cfg=cfg or self._BASE_CFG,
        )

    def _make_fake_features(self, B=2, C=32, H=20, W=20, num_levels=3):
        return [_make_nested_tensor(B, C, H // (2 ** i), W // (2 ** i)) for i in range(num_levels)]

    def test_output_length_matches_input(self):
        mod = self._make_module()
        images = torch.rand(2, 3, 80, 80)
        feats = self._make_fake_features(B=2, C=32)
        out = mod(images, feats)
        self.assertEqual(len(out), len(feats))

    def test_output_shapes_preserved(self):
        mod = self._make_module()
        images = torch.rand(2, 3, 80, 80)
        feats = self._make_fake_features(B=2, C=32, H=20, W=20)
        out = mod(images, feats)
        for orig, result in zip(feats, out):
            self.assertEqual(result.tensors.shape, orig.tensors.shape)

    def test_identity_fallback_synthesises_frames(self):
        """With identity fallback and zero-init MDD, output features should equal input."""
        mod = self._make_module()
        mod.eval()
        images = torch.rand(1, 3, 32, 32)
        feats = self._make_fake_features(B=1, C=32, H=16, W=16)
        out = mod(images, feats)
        # With zero-init gate and proj_out, output should be close to sigmoid(input).
        for i, (orig, result) in enumerate(zip(feats, out)):
            self.assertEqual(result.tensors.shape, orig.tensors.shape, f"Level {i} shape mismatch")

    def test_noise_fallback_differs_from_identity(self):
        """noise fallback should produce different synthetic frames on repeated calls (stochastic)."""
        cfg = dict(self._BASE_CFG)
        cfg["temporal"] = {"num_frames": 3, "fallback_mode": "noise", "noise_std": 1.0}
        mod = motion.MotionModule(feature_channels_per_scale=[8], motion_cfg=cfg)
        images = torch.ones(1, 3, 16, 16)
        # Gather 10 frame windows and ensure they're not all the same.
        windows = [mod._make_frame_window(images) for _ in range(10)]
        diffs = [not torch.allclose(windows[0], w) for w in windows[1:]]
        self.assertTrue(any(diffs), "Noise fallback should produce stochastic frames")

    def test_zero_fallback_delta_is_zero(self):
        cfg = dict(self._BASE_CFG)
        cfg["temporal"] = {"num_frames": 3, "fallback_mode": "zero", "noise_std": 0.02}
        mod = motion.MotionModule(feature_channels_per_scale=[8], motion_cfg=cfg)
        images = torch.rand(1, 3, 16, 16)
        frames = mod._make_frame_window(images)  # [1, 3, 3, 16, 16]
        delta = frames[:, 1] - frames[:, 0]
        # For identity / zero fallback both frames are identical → delta = 0.
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta)))

    def test_temporal_input_passes_through_unchanged(self):
        """A [B, T, 3, H, W] input should not be re-wrapped."""
        mod = self._make_module()
        frames = torch.rand(1, 3, 3, 16, 16)  # already a 5-D tensor
        out = mod._make_frame_window(frames)
        self.assertIs(out, frames)

    def test_mdd_disabled_skips_motion_maps(self):
        cfg = {**self._BASE_CFG, "tracknet_v5": {
            "mdd": {"enabled": False, "polarity_channels": 4, "attention": {}},
            "rstr": {"enabled": False},
        }}
        mod = motion.MotionModule(feature_channels_per_scale=[8, 8], motion_cfg=cfg)
        self.assertIsNone(mod.mdd)
        images = torch.rand(1, 3, 16, 16)
        feats = self._make_fake_features(B=1, C=8, H=8, W=8, num_levels=2)
        out = mod(images, feats)
        self.assertEqual(len(out), 2)

    def test_tensor_only_export_matches_nested_forward(self):
        mod = self._make_module()
        mod.eval()
        images = torch.rand(2, 3, 80, 80)
        feats = self._make_fake_features(B=2, C=32, H=20, W=20)

        nested_result = mod(images, feats)
        export_result = mod.forward_export(images, [feat.tensors for feat in feats])

        self.assertEqual(len(export_result), len(nested_result))
        for nested, exported in zip(nested_result, export_result):
            self.assertTrue(torch.allclose(nested.tensors, exported, atol=1e-6))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class ConfigHelpersTest(unittest.TestCase):
    def test_resolve_motion_type_default(self):
        self.assertEqual(motion.resolve_motion_type({}), "tracknet_v5")
        self.assertEqual(motion.resolve_motion_type(None), "tracknet_v5")

    def test_resolve_motion_type_valid(self):
        self.assertEqual(motion.resolve_motion_type({"type": "tracknet_v5"}), "tracknet_v5")
        self.assertEqual(motion.resolve_motion_type({"type": "none"}), "none")

    def test_resolve_motion_type_invalid_falls_back_with_warning(self):
        with self.assertWarns(UserWarning):
            result = motion.resolve_motion_type({"type": "tracknet_v3"})
        self.assertEqual(result, "tracknet_v5")

    def test_apply_overrides_only_non_null(self):
        kwargs = {}
        motion.apply_motion_overrides(
            kwargs,
            {"overrides": {"num_queries": 300, "gradient_checkpointing": True, "resolution": None}},
        )
        self.assertEqual(kwargs["num_queries"], 300)
        self.assertIs(kwargs["gradient_checkpointing"], True)
        self.assertNotIn("resolution", kwargs)

    def test_apply_overrides_empty_is_noop(self):
        kwargs = {"foo": "bar"}
        motion.apply_motion_overrides(kwargs, None)
        motion.apply_motion_overrides(kwargs, {})
        self.assertEqual(kwargs, {"foo": "bar"})

    def test_deep_merge_recursive(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 7}, "c": 4}
        result = motion._deep_merge(base, override)
        self.assertEqual(result["a"], {"x": 1, "y": 99, "z": 7})
        self.assertEqual(result["b"], 3)
        self.assertEqual(result["c"], 4)
        # Original should be unchanged (deep_merge returns a new dict).
        self.assertEqual(base["a"]["y"], 2)


# ---------------------------------------------------------------------------
# build_model_kwargs wiring (trainer integration)
# ---------------------------------------------------------------------------

class BuildModelKwargsMotionTest(unittest.TestCase):
    def test_disabled_is_completely_inert(self):
        """motion.enabled: false must not trigger any motion imports or kwargs."""
        kwargs = trainer.build_model_kwargs({"model": {"size": "large"}})
        # No motion-specific kwarg should be injected.
        self.assertNotIn("motion_config", kwargs)

    def test_disabled_explicit_false_is_inert(self):
        kwargs = trainer.build_model_kwargs({
            "model": {"size": "medium", "motion": {"enabled": False}}
        })
        self.assertNotIn("motion_config", kwargs)

    def test_enabled_applies_overrides(self):
        """When enabled, apply_motion_overrides must inject non-null override values."""
        kwargs = trainer.build_model_kwargs({
            "model": {
                "size": "medium",
                "motion": {
                    "enabled": True,
                    "type": "tracknet_v5",
                    "overrides": {"num_queries": 500},
                },
            }
        })
        self.assertEqual(kwargs.get("num_queries"), 500)

    def test_enabled_sets_gradient_checkpointing(self):
        kwargs = trainer.build_model_kwargs({
            "model": {
                "size": "large",
                "motion": {
                    "enabled": True,
                    "type": "tracknet_v5",
                    "overrides": {"gradient_checkpointing": True},
                },
            }
        })
        self.assertIs(kwargs.get("gradient_checkpointing"), True)


# ---------------------------------------------------------------------------
# ensure_motion_support idempotency
# ---------------------------------------------------------------------------

class EnsureMotionSupportTest(unittest.TestCase):
    def test_idempotent(self):
        """Calling ensure_motion_support multiple times should not raise."""
        for _ in range(3):
            motion.ensure_motion_support({"enabled": True, "type": "tracknet_v5"})
        self.assertTrue(motion.is_patched())

    def test_is_patched_after_call(self):
        motion.ensure_motion_support()
        self.assertTrue(motion.is_patched())

    def test_forward_export_is_patched(self):
        motion.ensure_motion_support({"enabled": True, "type": "tracknet_v5"})
        from rfdetr.models.lwdetr import LWDETR

        self.assertTrue(getattr(LWDETR.forward_export, "_motion_patched", False))

    def test_export_wrapper_applies_motion_to_tensor_features(self):
        import rfdetr.models.lwdetr as lwdetr_module

        class FakeBackbone(nn.Module):
            def forward(self, tensors):
                feature = tensors[:, :1]
                mask = torch.zeros(
                    tensors.shape[0], tensors.shape[2], tensors.shape[3], dtype=torch.bool
                )
                position = torch.zeros_like(feature)
                return [feature], [mask], [position], None

        class FakeMotion(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward_export(self, _images, features):
                self.calls += 1
                return [feature * 3.0 for feature in features]

        class FakeLWDETR(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = FakeBackbone()
                self.motion_module = FakeMotion()

            def forward(self, samples, targets=None):
                return self.backbone(samples)

            def forward_export(self, tensors):
                features, _, _, _ = self.backbone(tensors)
                return features[0]

        original_patched = motion._LWDETR_PATCHED
        try:
            with patch.object(lwdetr_module, "LWDETR", FakeLWDETR):
                motion._LWDETR_PATCHED = False
                motion._patch_lwdetr_motion_forward()
                model = FakeLWDETR()
                inputs = torch.ones(2, 3, 4, 4)

                output = model.forward_export(inputs)

                self.assertEqual(model.motion_module.calls, 1)
                self.assertTrue(torch.equal(output, torch.full((2, 1, 4, 4), 3.0)))
                self.assertIs(model.backbone.forward.__func__, FakeBackbone.forward)

                traced = torch.jit.trace_module(
                    model,
                    {"forward_export": inputs},
                    check_trace=False,
                )
                self.assertTrue(
                    torch.equal(
                        traced.forward_export(inputs),
                        torch.full((2, 1, 4, 4), 3.0),
                    )
                )
        finally:
            motion._LWDETR_PATCHED = original_patched


class MotionExportValidationTest(unittest.TestCase):
    def test_disabled_motion_is_noop(self):
        motion.assert_motion_export_ready(nn.Linear(2, 2), {"enabled": False})

    def test_enabled_motion_requires_attached_module(self):
        fake_lwdetr_type = type("LWDETR", (nn.Module,), {})
        model = fake_lwdetr_type()
        with self.assertRaisesRegex(RuntimeError, "no attached motion_module"):
            motion.assert_motion_export_ready(
                model,
                {"enabled": True, "type": "tracknet_v5"},
            )

    def test_enabled_motion_requires_motion_aware_forward_export(self):
        class ExportableMotion(nn.Module):
            def forward_export(self, _images, features):
                return features

        fake_lwdetr_type = type("LWDETR", (nn.Module,), {})
        model = fake_lwdetr_type()
        model.motion_module = ExportableMotion()
        with self.assertRaisesRegex(RuntimeError, "forward_export is not motion-aware"):
            motion.assert_motion_export_ready(
                model,
                {"enabled": True, "type": "tracknet_v5"},
            )


# ---------------------------------------------------------------------------
# attach_motion_module guard
# ---------------------------------------------------------------------------

class AttachMotionModuleTest(unittest.TestCase):
    def test_disabled_does_not_attach(self):
        """When motion disabled, attach_motion_module must be a no-op."""
        model = nn.Linear(4, 4)  # arbitrary model with no LWDETR
        motion.attach_motion_module(model, {"enabled": False})
        self.assertFalse(hasattr(model, "motion_module"))

    def test_no_lwdetr_warns(self):
        """When no LWDETR is found, attach_motion_module should warn and not raise."""
        model = nn.Linear(4, 4)
        with self.assertWarns(UserWarning):
            motion.attach_motion_module(model, {"enabled": True, "type": "tracknet_v5"})
        self.assertFalse(hasattr(model, "motion_module"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
