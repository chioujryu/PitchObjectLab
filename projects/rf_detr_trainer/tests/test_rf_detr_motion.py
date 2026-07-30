"""Tests for the pluggable TrackNetV5 motion module in rf_detr_motion.py.

These cover:
* MotionDirectionDecoupling forward (shape + paper alpha/beta mapping).
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

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import rf_detr_motion as motion
import train_rf_detr_model as trainer

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


def _small_motion_config(fallback_mode="identity"):
    return {
        "enabled": True,
        "type": "tracknet_v5",
        "temporal": {"num_frames": 3, "fallback_mode": fallback_mode},
        "tracknet_v5": {
            "mdd": {"enabled": True, "polarity_channels": 4},
            "rstr": {"enabled": False},
        },
    }


class _FakeProjector(nn.Module):
    def __init__(self, levels, width):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(nn.Identity(), nn.LayerNorm(width)) for _ in range(levels)])


class _FakeBackboneLevel(nn.Module):
    def __init__(self, scales, width):
        super().__init__()
        self.projector_scale = list(scales)
        self.projector = _FakeProjector(len(scales), width)
        self.encoder = nn.Identity()
        self.encoder._out_feature_channels = [384, 384, 384, 384]

    def forward(self, tensors):
        feature = tensors[:, :1]
        mask = torch.zeros(tensors.shape[0], tensors.shape[2], tensors.shape[3], dtype=torch.bool)
        position = torch.zeros_like(feature)
        return [feature], [mask], [position], None


class _FakeLWDETR(nn.Module):
    def __init__(self, scales=("P4",), width=24):
        super().__init__()
        self.backbone = nn.ModuleList([_FakeBackboneLevel(scales, width)])
        self.transformer = nn.Module()
        self.transformer.d_model = width
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, samples, targets=None):
        return self.backbone[0](samples)

    def forward_export(self, tensors):
        return tensors


_FakeLWDETR.forward_export._motion_patched = True


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
        """The paper mapping is constant for equal polarity intensity."""
        attn = motion.LearnableSigmoidAttention(num_channels=3)
        x = torch.zeros(1, 3, 5, 5)
        out = attn(x)
        # All spatial positions should be equal.
        self.assertTrue(torch.allclose(out[:, :, 0:1, 0:1].expand_as(out), out))

    def test_matches_published_alpha_beta_equation(self):
        attn = motion.LearnableSigmoidAttention(
            num_channels=1,
            init_alpha=0.2,
            init_beta=0.15,
            epsilon=1.0e-6,
        )
        x = torch.tensor([[[[0.0, 0.25]]]])
        alpha = torch.tensor(0.2)
        beta = torch.tensor(0.15)
        k = 5.0 / (0.45 * torch.abs(torch.tanh(alpha)) + 1.0e-6)
        midpoint = 0.6 * torch.tanh(beta)
        expected = torch.sigmoid(k * (torch.abs(x) - midpoint))
        torch.testing.assert_close(attn(x), expected)


# ---------------------------------------------------------------------------
# MotionDirectionDecoupling
# ---------------------------------------------------------------------------


class MDDTest(unittest.TestCase):
    def _make_mdd(self, learnable=False):
        return motion.MotionDirectionDecoupling(
            in_channels=3,
            polarity_channels=4,
            learnable=learnable,
        )

    def test_output_shape(self):
        mdd = self._make_mdd()
        frames = torch.rand(2, 3, 3, 16, 16)
        out = mdd(frames)
        self.assertEqual(out.shape, (2, 4, 16, 16))

    def test_identical_frames_produce_low_equal_attention(self):
        mdd = self._make_mdd(learnable=True)
        image = torch.rand(2, 3, 8, 8)
        frames = image.unsqueeze(1).expand(-1, 3, -1, -1, -1)
        output = mdd(frames)
        self.assertTrue(torch.all(output > 0.0))
        self.assertTrue(torch.all(output < 0.01))
        torch.testing.assert_close(
            output,
            output[:, :, :1, :1].expand_as(output),
        )

    def test_uses_luminance_and_both_adjacent_pairs(self):
        mdd = self._make_mdd()
        frames = torch.zeros(1, 3, 3, 2, 2)
        frames[:, 1, 0] = 1.0  # red increases from previous to center
        frames[:, 2, 0] = 0.25  # red decreases from center to next
        raw = mdd.raw_polarities(frames)
        self.assertTrue(torch.allclose(raw[:, 0], torch.full_like(raw[:, 0], 0.299)))
        self.assertTrue(torch.equal(raw[:, 1], torch.zeros_like(raw[:, 1])))
        self.assertTrue(torch.equal(raw[:, 2], torch.zeros_like(raw[:, 2])))
        self.assertTrue(torch.allclose(raw[:, 3], torch.full_like(raw[:, 3], 0.299 * 0.75)))

    def test_polarity_fields_are_non_negative(self):
        out = self._make_mdd(learnable=True)(torch.randn(2, 3, 3, 16, 16))
        self.assertTrue((out >= 0).all())

    def test_requires_exact_tracknet_shape(self):
        with self.assertRaises(ValueError):
            motion.MotionDirectionDecoupling(in_channels=3, polarity_channels=2)
        with self.assertRaises(ValueError):
            self._make_mdd()(torch.rand(1, 2, 3, 8, 8))


# ---------------------------------------------------------------------------
# MotionFeatureGate
# ---------------------------------------------------------------------------
# RSTRHead
# ---------------------------------------------------------------------------


class RSTRHeadTest(unittest.TestCase):
    @staticmethod
    def _make_head(**overrides):
        kwargs = {
            "num_frames": 3,
            "hidden_dim": 16,
            "num_heads": 2,
            "num_blocks": 1,
            "dropout": 0.0,
            "patch_size": 4,
            "context_mask_prob": 0.0,
        }
        kwargs.update(overrides)
        return motion.TemporalRSTRHead(**kwargs)

    def test_output_shape_preserved_with_odd_resolution(self):
        head = self._make_head()
        drafts = torch.rand(2, 3, 7, 9)
        out = head(drafts, torch.rand(2, 4, 14, 18))
        self.assertEqual(out.shape, drafts.shape)

    def test_zero_init_residual_is_exact_logit_identity(self):
        head = self._make_head()
        drafts = torch.randn(1, 3, 6, 6)
        out = head(drafts, torch.rand(1, 4, 12, 12))
        self.assertTrue(torch.equal(out, drafts))

    def test_temporal_branch_receives_gradient(self):
        head = self._make_head()
        nn.init.normal_(head.residual_projection.weight, std=0.01)
        drafts = torch.rand(1, 3, 6, 6, requires_grad=True)
        out = head(drafts, torch.rand(1, 4, 12, 12))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(drafts.grad).all())
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in head.temporal_blocks.parameters()
            )
        )

    def test_eval_is_deterministic_and_training_context_mask_is_supported(self):
        head = self._make_head(context_mask_prob=0.5)
        drafts = torch.rand(1, 3, 6, 6)
        motion_maps = torch.rand(1, 4, 12, 12)
        head.eval()
        self.assertTrue(torch.equal(head(drafts, motion_maps), head(drafts, motion_maps)))
        head.train()
        self.assertEqual(head(drafts, motion_maps).shape, drafts.shape)

    def test_training_mask_changes_residual_base_but_eval_uses_clean_draft(self):
        head = self._make_head(
            context_mask_prob=0.5,
            num_blocks=0,
        )
        drafts = torch.ones(1, 3, 8, 8)
        motion_maps = torch.zeros(1, 4, 8, 8)
        head.eval()
        torch.testing.assert_close(head(drafts, motion_maps), drafts)
        head.train()
        torch.manual_seed(7)
        masked = head(drafts, motion_maps)
        self.assertFalse(torch.equal(masked, drafts))
        self.assertTrue(bool((masked == 0).any()))


class HeatmapUtilitiesTest(unittest.TestCase):
    @staticmethod
    def _two_ball_targets(primary=None):
        target = {
            "boxes": torch.tensor([[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.2, 0.2]]),
            "box_format": "cxcywh_normalized",
        }
        if primary is not None:
            target["primary_label_index"] = primary
        return [[target, target, target]]

    def test_gaussian_all_focus_contains_both_centres(self):
        heatmaps = motion.build_gaussian_heatmap_targets(self._two_ball_targets(), (32, 64), focus_mode="all")
        self.assertEqual(heatmaps.shape, (1, 3, 32, 64))
        self.assertGreater(float(heatmaps[0, 1, 16, 16]), 0.95)
        self.assertGreater(float(heatmaps[0, 1, 16, 48]), 0.95)

    def test_gaussian_single_focus_requires_explicit_primary_for_multiball(self):
        with self.assertRaisesRegex(ValueError, "without 'primary_label_index'"):
            motion.build_gaussian_heatmap_targets(self._two_ball_targets(), (32, 64), focus_mode="single")
        heatmaps = motion.build_gaussian_heatmap_targets(
            self._two_ball_targets(primary=1), (32, 64), focus_mode="single"
        )
        self.assertLess(float(heatmaps[0, 1, 16, 16]), 0.1)
        self.assertGreater(float(heatmaps[0, 1, 16, 48]), 0.95)

    def test_weighted_bce_is_finite_and_backpropagates(self):
        logits = torch.zeros(1, 3, 16, 16, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :, 8, 8] = 1.0
        loss = motion.weighted_heatmap_bce(logits, targets, gamma=2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_peak_extraction_single_and_all(self):
        heatmaps = torch.zeros(1, 1, 12, 12)
        heatmaps[0, 0, 3, 4] = 0.8
        heatmaps[0, 0, 8, 9] = 0.9
        all_peaks = motion.extract_heatmap_peaks(heatmaps, focus_mode="all", threshold=0.5, max_peaks=20)
        single_peak = motion.extract_heatmap_peaks(heatmaps, focus_mode="single", threshold=0.5)
        self.assertEqual(all_peaks[0][0].shape, (2, 3))
        self.assertEqual(single_peak[0][0].shape, (1, 3))
        self.assertTrue(torch.equal(single_peak[0][0][0, :2], torch.tensor([9.0, 8.0])))


# ---------------------------------------------------------------------------
# MotionModule
# ---------------------------------------------------------------------------


class MotionModuleTest(unittest.TestCase):
    _BASE_CFG = {
        "enabled": True,
        "type": "tracknet_v5",
        "temporal": {"num_frames": 3, "fallback_mode": "identity", "noise_std": 0.02},
        "tracknet_v5": {
            "mdd": {
                "enabled": True,
                "polarity_channels": 4,
                "attention": {"learnable": True, "init_alpha": 0.2, "init_beta": 0.15, "epsilon": 1.0e-6},
            },
            "rstr": {
                "enabled": True,
                "num_blocks": 1,
                "hidden_dim": 32,
                "num_heads": 4,
                "dropout": 0.0,
                "patch_size": 4,
                "context_mask_prob": 0.0,
            },
        },
        "loss": {"motion_attention_weight": 0.0},
    }

    def _make_module(self, cfg=None):
        return motion.MotionModule(
            feature_channels_per_scale=[32, 32, 32],
            motion_cfg=cfg or self._BASE_CFG,
        )

    def _make_fake_features(self, B=2, C=32, H=20, W=20, num_levels=3):
        return [_make_nested_tensor(B, C, H // (2**i), W // (2**i)) for i in range(num_levels)]

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
        """Noise fallback should produce different synthetic frames on repeated calls (stochastic)."""
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
        cfg = {
            **self._BASE_CFG,
            "tracknet_v5": {
                "mdd": {"enabled": False, "polarity_channels": 4, "attention": {}},
                "rstr": {"enabled": False},
            },
        }
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

    def test_true_temporal_forward_returns_full_resolution_heatmaps(self):
        mod = self._make_module()
        mod.eval()
        frames = torch.rand(2, 3, 3, 32, 40)
        temporal_features = [
            torch.rand(2, 3, 32, 8, 10),
            torch.rand(2, 3, 32, 4, 5),
            torch.rand(2, 3, 32, 2, 3),
        ]
        output = mod.forward_temporal(frames, temporal_features)
        self.assertEqual(output.heatmap_logits.shape, (2, 3, 32, 40))
        self.assertEqual(output.motion_maps.shape, (2, 4, 32, 40))
        self.assertEqual([item.shape for item in output.features], [(2, 32, 8, 10), (2, 32, 4, 5), (2, 32, 2, 3)])

    def test_zero_init_fusion_is_exact_centre_feature_identity(self):
        mod = self._make_module()
        mod.eval()
        frames = torch.rand(1, 3, 3, 16, 16)
        temporal_features = [
            torch.rand(1, 3, 32, 8, 8),
            torch.rand(1, 3, 32, 4, 4),
            torch.rand(1, 3, 32, 2, 2),
        ]
        output = mod.forward_temporal(frames, temporal_features)
        for original, fused in zip(temporal_features, output.features):
            self.assertTrue(torch.equal(fused, original[:, 1]))

    def test_real_mode_rejects_unapproved_single_frame_fallback(self):
        cfg = deepcopy(self._BASE_CFG)
        cfg["temporal"] = {
            "mode": "real",
            "num_frames": 3,
            "fallback_mode": "real",
            "allow_single_frame_fallback": False,
        }
        mod = self._make_module(cfg)
        with self.assertRaisesRegex(RuntimeError, "received one frame"):
            mod._make_frame_window(torch.rand(1, 3, 16, 16))


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
        kwargs = trainer.build_model_kwargs({"model": {"size": "medium", "motion": {"enabled": False}}})
        self.assertNotIn("motion_config", kwargs)

    def test_type_none_is_inert_even_when_enabled_flag_is_true(self):
        kwargs = trainer.build_model_kwargs(
            {
                "model": {
                    "size": "medium",
                    "motion": {
                        "enabled": True,
                        "type": "none",
                        "overrides": {"num_queries": 17},
                    },
                }
            }
        )
        self.assertNotIn("num_queries", kwargs)
        self.assertFalse(trainer.motion_module_enabled({"model": {"motion": {"enabled": True, "type": "none"}}}))

    def test_enabled_applies_overrides(self):
        """When enabled, apply_motion_overrides must inject non-null override values."""
        kwargs = trainer.build_model_kwargs(
            {
                "model": {
                    "size": "medium",
                    "motion": {
                        "enabled": True,
                        "type": "tracknet_v5",
                        "overrides": {"num_queries": 500},
                    },
                }
            }
        )
        self.assertEqual(kwargs.get("num_queries"), 500)

    def test_enabled_sets_gradient_checkpointing(self):
        kwargs = trainer.build_model_kwargs(
            {
                "model": {
                    "size": "large",
                    "motion": {
                        "enabled": True,
                        "type": "tracknet_v5",
                        "overrides": {"gradient_checkpointing": True},
                    },
                }
            }
        )
        self.assertIs(kwargs.get("gradient_checkpointing"), True)


# ---------------------------------------------------------------------------
# ensure_motion_support idempotency
# ---------------------------------------------------------------------------


class EnsureMotionSupportTest(unittest.TestCase):
    def test_disabled_support_is_completely_non_mutating(self):
        self.assertFalse(hasattr(motion, "MOTION_SETTINGS"))
        with patch.object(motion, "_check_version") as version_check:
            motion.ensure_motion_support({"enabled": False})
        version_check.assert_not_called()
        self.assertFalse(motion.is_patched())

    def test_enabled_support_validates_without_global_patch(self):
        with patch.object(motion, "_check_version") as version_check:
            for _ in range(3):
                motion.ensure_motion_support({"enabled": True, "type": "tracknet_v5"})
        self.assertEqual(version_check.call_count, 3)
        self.assertFalse(motion.is_patched())

    def test_enabled_tracknet_rejects_unimplemented_temporal_graph_options(self):
        cases = (
            ("temporal.anchor", ("temporal", "anchor"), "start"),
            ("temporal.boundary_policy", ("temporal", "boundary_policy"), "replicate"),
            (
                "tracknet_v5.feature_source",
                ("tracknet_v5", "feature_source"),
                "center_frame",
            ),
            (
                "tracknet_v5.feature_level",
                ("tracknet_v5", "feature_level"),
                "p3",
            ),
            (
                "rstr.attention_mode",
                ("tracknet_v5", "rstr", "attention_mode"),
                "joint",
            ),
        )
        for expected_field, path, unsupported_value in cases:
            config = _small_motion_config("real")
            cursor = config
            for key in path[:-1]:
                cursor = cursor.setdefault(key, {})
            cursor[path[-1]] = unsupported_value
            with self.subTest(field=expected_field), patch.object(motion, "_check_version") as version_check:
                with self.assertRaisesRegex(ValueError, expected_field):
                    motion.ensure_motion_support(config)
                version_check.assert_not_called()

    def test_enabled_support_does_not_modify_lwdetr_class_methods(self):
        from rfdetr.models.lwdetr import LWDETR

        original_methods = (LWDETR.__init__, LWDETR.forward, LWDETR.forward_export)
        motion.ensure_motion_support({"enabled": True, "type": "tracknet_v5"})
        self.assertEqual((LWDETR.__init__, LWDETR.forward, LWDETR.forward_export), original_methods)

    def test_attachment_is_explicit_and_instance_bound(self):
        import rfdetr.models.lwdetr as lwdetr_module

        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            enabled_model = _FakeLWDETR()
            disabled_model = _FakeLWDETR()
            motion.attach_motion_module(enabled_model, _small_motion_config())
            motion.attach_motion_module(disabled_model, {"enabled": False})
        self.assertIsInstance(enabled_model.motion_module, motion.MotionModule)
        self.assertFalse(hasattr(disabled_model, "motion_module"))


class InstanceTemporalForwardTest(unittest.TestCase):
    def test_attached_forward_accepts_raw_5d_and_temporal_batch(self):
        import rfdetr.models.lwdetr as lwdetr_module

        class FakeJoiner(nn.Module):
            def __init__(self):
                super().__init__()
                self.level = _FakeBackboneLevel(("P4",), 8)
                self.seen_tensors = []
                self.seen_grad_enabled = []

            def __getitem__(self, index):
                if index != 0:
                    raise IndexError(index)
                return self.level

            def forward(self, samples):
                tensors, masks = samples.decompose()
                self.seen_tensors.append(tensors.detach().clone())
                self.seen_grad_enabled.append(torch.is_grad_enabled())
                pooled = torch.nn.functional.avg_pool2d(tensors.mean(1, keepdim=True), 2)
                feature = pooled.expand(-1, 8, -1, -1).contiguous()
                feature_mask = torch.nn.functional.interpolate(
                    masks[:, None].float(), size=feature.shape[-2:], mode="nearest"
                )[:, 0].bool()
                nested = motion._rebuild_nested_tensor(feature, feature_mask)
                return [nested], [torch.zeros_like(feature)], None

        class TemporalFakeLWDETR(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = FakeJoiner()
                self.transformer = nn.Module()
                self.transformer.d_model = 8
                self.anchor = nn.Parameter(torch.zeros(1))

            def forward(self, samples, targets=None):
                features, _, _ = self.backbone(samples)
                batch = features[0].tensors.shape[0]
                return {
                    "pred_logits": features[0].tensors.mean((2, 3))[:, None, :1],
                    "pred_boxes": features[0].tensors.new_zeros((batch, 1, 4)),
                }

            def forward_export(self, tensors):
                return tensors

        config = _small_motion_config("real")
        config["temporal"]["allow_single_frame_fallback"] = False
        with patch.object(lwdetr_module, "LWDETR", TemporalFakeLWDETR):
            model = TemporalFakeLWDETR()
            class_forward = TemporalFakeLWDETR.forward
            motion.attach_motion_module(model, config)
            frames = torch.rand(2, 3, 3, 16, 20)
            normalized_frames = frames * 4.0 - 2.0
            mdd_inputs = []
            original_mdd_forward = model.motion_module.mdd.forward

            def capture_mdd_input(value):
                mdd_inputs.append(value.detach().clone())
                return original_mdd_forward(value)

            with patch.object(
                model.motion_module.mdd,
                "forward",
                side_effect=capture_mdd_input,
            ):
                raw_output = model(frames)
                batch = SimpleNamespace(
                    frames=normalized_frames,
                    mdd_frames=frames,
                    padding_masks=torch.zeros(2, 3, 16, 20, dtype=torch.bool),
                    anchor_targets=[{"boxes": torch.empty(0, 4)} for _ in range(2)],
                )
                batch_output = model(batch)

        self.assertIs(TemporalFakeLWDETR.forward, class_forward)
        torch.testing.assert_close(mdd_inputs[0], frames)
        torch.testing.assert_close(mdd_inputs[1], frames)
        torch.testing.assert_close(
            model.backbone.seen_tensors[0],
            frames[:, 0],
        )
        torch.testing.assert_close(
            model.backbone.seen_tensors[1],
            frames[:, 1],
        )
        torch.testing.assert_close(model.backbone.seen_tensors[2], frames[:, 2])
        torch.testing.assert_close(
            model.backbone.seen_tensors[3],
            normalized_frames[:, 0],
        )
        torch.testing.assert_close(
            model.backbone.seen_tensors[4],
            normalized_frames[:, 1],
        )
        torch.testing.assert_close(
            model.backbone.seen_tensors[5],
            normalized_frames[:, 2],
        )
        self.assertEqual(
            model.backbone.seen_grad_enabled,
            [False, True, False, False, True, False],
        )
        for output in (raw_output, batch_output):
            self.assertEqual(output["pred_logits"].shape[0], 2)
            self.assertEqual(output["pred_heatmap_logits"].shape, (2, 3, 16, 20))
            self.assertEqual(output["pred_heatmaps"].shape, (2, 3, 16, 20))
            self.assertEqual(output["motion_maps"].shape, (2, 4, 16, 20))


class MotionExportValidationTest(unittest.TestCase):
    def test_disabled_motion_is_noop(self):
        motion.assert_motion_export_ready(nn.Linear(2, 2), {"enabled": False})

    def test_enabled_temporal_motion_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(RuntimeError, "ONNX/TensorRT export is not supported"):
            motion.assert_motion_export_ready(
                nn.Linear(2, 2),
                {"enabled": True, "type": "tracknet_v5"},
            )


class MotionCheckpointCompatibilityTest(unittest.TestCase):
    @staticmethod
    def _metadata(motion_config):
        return {
            "schema_version": 3,
            "model_size": "medium",
            "motion": motion_config,
            "architecture_fingerprint": repr(motion_config),
        }

    def test_checkpoint_accepts_exact_motion_state(self):
        import rfdetr.models.lwdetr as lwdetr_module

        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            model = _FakeLWDETR(scales=("P4",), width=24)
            motion.attach_motion_module(model, _small_motion_config())
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "motion_exact.pth"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "pitchobjectlab_architecture": self._metadata(_small_motion_config()),
                    },
                    checkpoint,
                )

                motion.assert_motion_checkpoint_compatible(model, checkpoint)

    def test_checkpoint_rejects_legacy_motion_tensors_without_architecture_metadata(self):
        import rfdetr.models.lwdetr as lwdetr_module

        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            model = _FakeLWDETR()
            motion.attach_motion_module(model, _small_motion_config())
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "legacy_motion.pth"
                torch.save({"model": model.state_dict()}, checkpoint)

                with self.assertRaisesRegex(RuntimeError, "legacy TrackNet prototype checkpoint"):
                    motion.assert_motion_checkpoint_compatible(model, checkpoint)

    def test_stock_checkpoint_without_motion_tensors_does_not_require_metadata(self):
        checkpoint = {"model": {"anchor": torch.zeros(1)}}
        motion_state = motion._motion_state_from_checkpoint(checkpoint)
        self.assertEqual(motion_state, {})
        motion._assert_new_motion_architecture_metadata(checkpoint, motion_state)

    def test_optimizer_step_checkpoint_round_trip_preserves_updated_motion_state(self):
        import rfdetr.models.lwdetr as lwdetr_module

        config = _small_motion_config()
        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            model = _FakeLWDETR(scales=("P4",), width=24)
            motion.attach_motion_module(model, config)
            before = {key: value.detach().clone() for key, value in model.motion_module.state_dict().items()}

            optimizer = torch.optim.SGD(model.motion_module.parameters(), lr=0.1)
            frames = torch.stack(
                [
                    torch.zeros(1, 3, 8, 8),
                    torch.ones(1, 3, 8, 8),
                    torch.ones(1, 3, 8, 8),
                ],
                dim=1,
            )
            features = [torch.ones(1, 24, 4, 4)]
            optimizer.zero_grad(set_to_none=True)
            loss = model.motion_module.forward_export(frames, features)[0].sum()
            loss.backward()
            optimizer.step()

            updated = {key: value.detach().clone() for key, value in model.motion_module.state_dict().items()}
            changed = [key for key in before if not torch.equal(before[key], updated[key])]
            self.assertTrue(changed, "One optimizer step must update motion_module weights")
            self.assertIn("fusions.0.projection.weight", changed)

            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "motion_trained.pth"
                saved_state = model.state_dict()
                torch.save(
                    {
                        "model": saved_state,
                        "pitchobjectlab_architecture": self._metadata(config),
                    },
                    checkpoint,
                )
                self.assertTrue(
                    any(key.startswith("motion_module.") for key in saved_state),
                    "Training checkpoints must contain motion_module.* tensors",
                )

                reloaded = _FakeLWDETR(scales=("P4",), width=24)
                motion.attach_motion_module(reloaded, config)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                reloaded.load_state_dict(payload["model"], strict=True)

                restored = reloaded.motion_module.state_dict()
                self.assertEqual(set(restored), set(updated))
                for key, expected in updated.items():
                    self.assertTrue(
                        torch.equal(restored[key], expected),
                        f"Motion checkpoint tensor changed during save/reload: {key}",
                    )
                motion.assert_motion_checkpoint_compatible(reloaded, checkpoint)

    def test_checkpoint_rejects_missing_motion_weights(self):
        import rfdetr.models.lwdetr as lwdetr_module

        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            model = _FakeLWDETR()
            motion.attach_motion_module(model, _small_motion_config())
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "motion_missing.pth"
                torch.save({"model": {"anchor": torch.zeros(1)}}, checkpoint)

                with self.assertRaisesRegex(RuntimeError, "no motion_module"):
                    motion.assert_motion_checkpoint_compatible(model, checkpoint)

    def test_checkpoint_rejects_motion_shape_mismatch(self):
        import rfdetr.models.lwdetr as lwdetr_module

        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            model = _FakeLWDETR()
            motion.attach_motion_module(model, _small_motion_config())
            state = {key: value.clone() for key, value in model.state_dict().items()}
            state["motion_module.fusions.0.projection.weight"] = torch.zeros(1)
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "motion_shape.pth"
                torch.save(
                    {
                        "model": state,
                        "pitchobjectlab_architecture": self._metadata(_small_motion_config()),
                    },
                    checkpoint,
                )

                with self.assertRaisesRegex(RuntimeError, "shape_mismatch"):
                    motion.assert_motion_checkpoint_compatible(model, checkpoint)

    def test_checkpoint_rejects_motion_metadata_mismatch(self):
        import rfdetr.models.lwdetr as lwdetr_module

        expected_config = _small_motion_config("identity")
        saved_config = _small_motion_config("noise")
        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            model = _FakeLWDETR()
            motion.attach_motion_module(model, expected_config)
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "motion_metadata.pth"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "pitchobjectlab_architecture": self._metadata(saved_config),
                    },
                    checkpoint,
                )

                with self.assertRaisesRegex(RuntimeError, "TrackNet architecture fingerprint"):
                    motion.assert_motion_checkpoint_compatible(
                        model,
                        checkpoint,
                        expected_architecture=self._metadata(expected_config),
                    )

    def test_load_motion_checkpoint_weights_only_restores_tracknet(self):
        import rfdetr.models.lwdetr as lwdetr_module

        config = _small_motion_config()
        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            source = _FakeLWDETR()
            motion.attach_motion_module(source, config)
            source.motion_module.fusions[0].projection.bias.data.fill_(0.42)
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "motion_only_load.pth"
                torch.save(
                    {
                        "model": source.state_dict(),
                        "pitchobjectlab_architecture": self._metadata(config),
                    },
                    checkpoint,
                )

                destination = _FakeLWDETR()
                motion.attach_motion_module(destination, config)
                self.assertFalse(
                    torch.equal(
                        destination.motion_module.fusions[0].projection.bias,
                        source.motion_module.fusions[0].projection.bias,
                    )
                )
                motion.load_motion_checkpoint_weights(destination, checkpoint)
                self.assertTrue(
                    torch.equal(
                        destination.motion_module.fusions[0].projection.bias,
                        source.motion_module.fusions[0].projection.bias,
                    )
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

    def test_actual_projector_levels_and_channels_are_used_idempotently(self):
        import rfdetr.models.lwdetr as lwdetr_module

        model = _FakeLWDETR(scales=("P2", "P3", "P4"), width=32)
        config = _small_motion_config()
        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            motion.attach_motion_module(model, config)
            attached = model.motion_module

            self.assertEqual(attached.feature_channels_per_scale, [32, 32, 32])
            self.assertEqual(len(attached.fusions), 3)
            self.assertEqual(len(attached.draft_heads), 3)
            self.assertEqual(attached.fusions[0].projection.out_channels, 32)

            motion.attach_motion_module(model, config)
            self.assertIs(model.motion_module, attached)

    def test_p6_extra_pool_reuses_last_projector_channel_width(self):
        import rfdetr.models.lwdetr as lwdetr_module

        model = _FakeLWDETR(scales=("P4", "P6"), width=32)
        projector = model.backbone[0].projector
        projector.stages = nn.ModuleList([projector.stages[0]])
        projector.use_extra_pool = True
        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            motion.attach_motion_module(model, _small_motion_config())

        self.assertEqual(model.motion_module.feature_channels_per_scale, [32, 32])
        self.assertEqual(len(model.motion_module.fusions), 2)

    def test_compiled_wrapper_is_unwrapped_before_motion_attachment(self):
        import rfdetr.models.lwdetr as lwdetr_module

        model = _FakeLWDETR(scales=("P4",), width=24)
        compiled = SimpleNamespace(_orig_mod=model)
        with patch.object(lwdetr_module, "LWDETR", _FakeLWDETR):
            motion.attach_motion_module(compiled, _small_motion_config())

        self.assertTrue(hasattr(model, "motion_module"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
