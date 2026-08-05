from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import rf_detr_acceleration as acceleration


class _RawModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, tensor):
        batch = tensor.shape[0]
        return {
            "pred_boxes": torch.zeros(batch, 2, 4, device=tensor.device),
            "pred_logits": torch.zeros(batch, 2, 3, device=tensor.device),
        }


class _FakeRFDETR:
    def __init__(self, device="cpu"):
        self.model_config = SimpleNamespace(
            resolution=32,
            num_channels=3,
            segmentation_head=False,
            pretrain_weights=None,
        )
        self.model = SimpleNamespace(
            device=torch.device(device),
            resolution=32,
            model=_RawModule(),
            inference_model=None,
            postprocess=lambda outputs, target_sizes=None: (outputs, target_sizes),
        )
        self.optimize_calls = []
        self._is_optimized_for_inference = False

    @property
    def class_names(self):
        return ["ball"]

    def optimize_for_inference(self, **kwargs):
        self.optimize_calls.append(kwargs)
        self.model.inference_model = self.model.model

    def predict(self, *args, **kwargs):
        return (args, kwargs)


class _FakeONNXDimension:
    def __init__(self, value=None, symbolic=''):
        self.dim_value = 0 if value is None else value
        self.dim_param = symbolic
        self._has_value = value is not None

    def HasField(self, field):
        if field == 'dim_value':
            return self._has_value
        if field == 'dim_param':
            return bool(self.dim_param)
        raise ValueError(field)


def _fake_onnx_value_info(name, dimensions):
    resolved = []
    for dimension in dimensions:
        if isinstance(dimension, int):
            resolved.append(_FakeONNXDimension(value=dimension))
        elif isinstance(dimension, str):
            resolved.append(_FakeONNXDimension(symbolic=dimension))
        else:
            resolved.append(_FakeONNXDimension())
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=resolved))
        ),
    )


def _fake_p2_onnx_graph(*, p2_channel=384, label_slots=2):
    features = {
        'p2_feature': ['batch', p2_channel, 8, 8],
        'p3_feature': ['batch', 384, 4, 4],
        'p4_feature': ['batch', 384, 2, 2],
    }
    return SimpleNamespace(
        input=[_fake_onnx_value_info('input', ['batch', 3, 32, 32])],
        output=[
            _fake_onnx_value_info('dets', ['batch', 10, 4]),
            _fake_onnx_value_info('labels', ['batch', 10, label_slots]),
        ],
        value_info=[
            _fake_onnx_value_info(name, dimensions)
            for name, dimensions in features.items()
        ],
        initializer=[SimpleNamespace(name=f'{level}_weight') for level in ('p2', 'p3', 'p4')],
        node=[
            SimpleNamespace(
                op_type='ConvTranspose',
                name=f'{level}_upsample',
                input=[f'{level}_feature', f'{level}_weight'],
            )
            for level in ('p2', 'p3', 'p4')
        ],
    )


def _fake_onnx_module(model_proto):
    def save_model(_model, destination):
        Path(destination).write_bytes(b'shape-inferred-onnx')

    return SimpleNamespace(
        load=MagicMock(return_value=model_proto),
        checker=SimpleNamespace(check_model=MagicMock()),
        shape_inference=SimpleNamespace(infer_shapes=MagicMock(return_value=model_proto)),
        save_model=MagicMock(side_effect=save_model),
    )


class ResolveConfigTest(unittest.TestCase):
    def test_defaults_preserve_pytorch_fp32(self):
        resolved = acceleration.resolve_acceleration_config({"model": {"resolution": 576}})

        self.assertEqual(resolved.backend, "pytorch")
        self.assertEqual(resolved.precision, "fp32")
        self.assertEqual(resolved.resolution, 576)
        self.assertEqual(resolved.tensorrt.profile, acceleration.TensorRTProfile(1, 1, 1))

    def test_resolves_nested_tensorrt_and_auto_profile(self):
        resolved = acceleration.resolve_acceleration_config(
            {
                "model": {
                    "resolution": 320,
                    "inference_optimization": {
                        "backend": "tensorrt",
                        "tensorrt": {
                            "precision": "bf16",
                            "profile": {
                                "min_batch_size": 1,
                                "opt_batch_size": "auto",
                                "max_batch_size": "auto",
                            },
                        },
                    },
                }
            },
            batch_sizes=(8, 64, 7),
        )

        self.assertEqual(resolved.backend, "tensorrt")
        self.assertEqual(resolved.precision, "bf16")
        self.assertEqual(resolved.tensorrt.profile, acceleration.TensorRTProfile(1, 8, 64))

    def test_resolves_reusable_output_buffer_opt_in_strictly(self):
        resolved = acceleration.resolve_acceleration_config(
            {"backend": "tensorrt", "tensorrt": {"reuse_output_buffers": True}}
        )
        self.assertTrue(resolved.tensorrt.reuse_output_buffers)

        with self.assertRaisesRegex(ValueError, "reuse_output_buffers"):
            acceleration.resolve_acceleration_config(
                {"backend": "tensorrt", "tensorrt": {"reuse_output_buffers": "true"}}
            )

    def test_engine_path_derives_adjacent_manifest(self):
        resolved = acceleration.resolve_acceleration_config(
            {"backend": "tensorrt", "tensorrt": {"engine_path": "models/rfdetr.engine"}}
        )

        self.assertEqual(resolved.tensorrt.manifest_path, Path("models/rfdetr.engine.manifest.json"))

    def test_explicit_manifest_wins(self):
        resolved = acceleration.resolve_acceleration_config(
            {
                "backend": "tensorrt",
                "tensorrt": {"engine_path": "a.engine", "manifest_path": "custom.json"},
            }
        )

        self.assertEqual(resolved.tensorrt.manifest_path, Path("custom.json"))

    def test_rejects_invalid_backend_precision_combinations(self):
        invalid = [
            {"backend": "pytorch", "pytorch": {"precision": "fp16"}},
            {"backend": "tensorrt", "tensorrt": {"precision": "fp32"}},
            {"backend": "tensorrt", "tensorrt": {"manifest_path": "orphan.json"}},
            {
                "backend": "tensorrt",
                "tensorrt": {"engine_path": "model.engine", "force_rebuild": True},
            },
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                acceleration.resolve_acceleration_config(config)

    def test_rejects_reversed_profile(self):
        with self.assertRaisesRegex(ValueError, "min_batch_size"):
            acceleration.resolve_acceleration_config(
                {
                    "backend": "tensorrt",
                    "tensorrt": {"profile": {"min_batch_size": 2, "opt_batch_size": 8, "max_batch_size": 4}},
                }
            )

    def test_rejects_profile_minimum_above_one_for_real_tail_batches(self):
        with self.assertRaisesRegex(ValueError, "must be 1"):
            acceleration.resolve_acceleration_config(
                {
                    "backend": "tensorrt",
                    "tensorrt": {"profile": {"min_batch_size": 2, "opt_batch_size": 8, "max_batch_size": 8}},
                }
            )

    def test_common_tensorrt_profiles_cover_batch_1_4_8_16_and_explicit_optimum(self):
        profiles = acceleration.tensorrt_optimization_profiles(
            acceleration.TensorRTProfile(1, 7, 16)
        )

        self.assertEqual([profile.opt_batch_size for profile in profiles], [1, 4, 7, 8, 16])
        self.assertTrue(all(profile.min_batch_size == 1 for profile in profiles))
        self.assertTrue(all(profile.max_batch_size == 16 for profile in profiles))


class PyTorchOptimizationTest(unittest.TestCase):
    def test_fp32_is_noop_and_does_not_import_optional_packages(self):
        model = _FakeRFDETR()
        settings = acceleration.resolve_acceleration_config({})
        with patch.object(acceleration.importlib, "import_module") as importer:
            handle = acceleration.apply_pytorch_optimization(model, settings)

        importer.assert_not_called()
        self.assertEqual(model.optimize_calls, [])
        self.assertEqual(handle.backend, "pytorch")
        self.assertEqual(handle.metadata["effective_precision"], "fp32")
        self.assertEqual(handle.infer_raw(torch.zeros(2, 3, 32, 32))["pred_boxes"].shape, (2, 2, 4))

    def test_bf16_rejects_cpu_before_optimization(self):
        model = _FakeRFDETR(device="cpu")
        settings = acceleration.resolve_acceleration_config({"backend": "pytorch", "pytorch": {"precision": "bf16"}})

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            acceleration.apply_pytorch_optimization(model, settings)
        self.assertEqual(model.optimize_calls, [])

    def test_bf16_calls_exact_optimization_once(self):
        model = _FakeRFDETR(device="cuda:0")
        settings = acceleration.resolve_acceleration_config({"backend": "pytorch", "pytorch": {"precision": "bf16"}})
        with patch.object(acceleration, "_require_cuda"):
            first = acceleration.apply_pytorch_optimization(model, settings)
            second = acceleration.apply_pytorch_optimization(model, settings)

        self.assertEqual(model.optimize_calls, [{"compile": False, "dtype": torch.bfloat16}])
        self.assertFalse(first.metadata["already_applied"])
        self.assertTrue(second.metadata["already_applied"])

    def test_dynamic_mock_attribute_is_not_treated_as_real_marker(self):
        model = MagicMock()
        model.model.device = torch.device("cpu")
        model.model.model = _RawModule()
        model.model.inference_model = None
        settings = acceleration.resolve_acceleration_config({})

        handle = acceleration.apply_pytorch_optimization(model, settings)

        self.assertEqual(handle.backend, "pytorch")
        self.assertFalse(handle.metadata["already_applied"])

    def test_default_fp32_allows_non_executable_integration_fake(self):
        model = SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu"), model=object()))

        handle = acceleration.apply_pytorch_optimization(model, acceleration.resolve_acceleration_config({}))

        self.assertEqual(handle.backend, "pytorch")
        self.assertEqual(handle.consume_forward_seconds(), 0.0)
        self.assertEqual(handle.consume_postprocess_seconds(), 0.0)

    def test_handle_exposes_stable_segmentation_hooks(self):
        model = _FakeRFDETR()
        handle = acceleration.apply_pytorch_optimization(model, acceleration.resolve_acceleration_config({}))
        target_sizes = torch.tensor([[10, 20]])
        outputs = handle.infer_raw(torch.zeros(1, 3, 32, 32))

        postprocessed, seen_sizes = handle.postprocess(outputs, target_sizes)

        self.assertIs(postprocessed, outputs)
        self.assertIs(seen_sizes, target_sizes)
        self.assertEqual(handle.device, torch.device("cpu"))
        self.assertGreaterEqual(handle.consume_forward_seconds(), 0.0)
        self.assertEqual(handle.consume_forward_seconds(), 0.0)
        self.assertGreaterEqual(handle.consume_postprocess_seconds(), 0.0)
        self.assertEqual(handle.consume_postprocess_seconds(), 0.0)

    def test_handle_reusable_output_pool_allocates_once_per_thread_and_batch(self):
        model = _FakeRFDETR()
        handle = acceleration.apply_pytorch_optimization(model, acceleration.resolve_acceleration_config({}))
        allocations = []

        def allocate(batch_size):
            allocations.append(batch_size)
            return {
                "pred_boxes": torch.empty(batch_size, 2, 4),
                "pred_logits": torch.empty(batch_size, 2, 3),
            }

        handle._allocate_output_buffers = allocate
        handle._infer_into = lambda _tensor, buffers: buffers
        tensor = torch.zeros(4, 3, 32, 32)

        first = handle.infer_raw_reusing_buffers(tensor)
        second = handle.infer_raw_reusing_buffers(tensor)

        self.assertEqual(allocations, [4])
        self.assertIs(first["pred_boxes"], second["pred_boxes"])
        handle.clear_reusable_output_buffers()
        after_clear = handle.infer_raw_reusing_buffers(tensor)
        self.assertEqual(allocations, [4, 4])

        handle._reuse_output_buffers_by_default = True
        routed = handle.infer_raw(tensor)
        self.assertEqual(allocations, [4, 4])
        self.assertIs(routed["pred_boxes"], after_clear["pred_boxes"])


class FastBatchPreparationTest(unittest.TestCase):
    def test_uint8_hwc_images_are_batched_scaled_and_normalized_once(self):
        first = np.zeros((2, 3, 3), dtype=np.uint8)
        second = np.full((2, 3, 3), 255, dtype=np.uint8)

        prepared = acceleration.prepare_inference_batch(
            [first, second],
            shape=(2, 3),
            device="cpu",
        )

        self.assertEqual(prepared.tensor.shape, (2, 3, 2, 3))
        self.assertTrue(prepared.tensor.is_contiguous())
        self.assertEqual(prepared.target_sizes.tolist(), [[2, 3], [2, 3]])
        self.assertAlmostEqual(prepared.tensor[0, 0, 0, 0].item(), -0.485 / 0.229, places=5)
        self.assertAlmostEqual(prepared.tensor[1, 0, 0, 0].item(), (1.0 - 0.485) / 0.229, places=5)
        timing = prepared.consume_timing()
        self.assertGreaterEqual(timing["host_preprocess_seconds"], 0.0)
        self.assertEqual(timing["h2d_seconds"], 0.0)
        self.assertEqual(timing["resize_normalize_seconds"], 0.0)
        self.assertEqual(timing["device_preprocess_seconds"], 0.0)
        self.assertEqual(timing["total_seconds"], timing["host_preprocess_seconds"])

    def test_uniform_pil_crops_use_the_same_batch_contract(self):
        from PIL import Image

        crops = [
            Image.fromarray(np.full((3, 4, 3), value, dtype=np.uint8))
            for value in (0, 127, 255)
        ]

        prepared = acceleration.prepare_inference_batch(
            crops,
            shape=(3, 4),
            device="cpu",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )

        self.assertEqual(prepared.tensor.shape, (3, 3, 3, 4))
        self.assertEqual(prepared.target_sizes.tolist(), [[3, 4], [3, 4], [3, 4]])
        self.assertAlmostEqual(prepared.tensor[1].mean().item(), 127.0 / 255.0, places=6)

    def test_mixed_source_sizes_are_grouped_resized_and_returned_in_order(self):
        images = [
            torch.zeros(3, 2, 2),
            torch.ones(3, 3, 4),
            torch.full((3, 2, 2), 0.5),
        ]

        prepared = acceleration.prepare_inference_batch(
            images,
            shape=4,
            device=torch.device("cpu"),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )

        self.assertEqual(prepared.tensor.shape, (3, 3, 4, 4))
        self.assertEqual(prepared.target_sizes.tolist(), [[2, 2], [3, 4], [2, 2]])
        self.assertAlmostEqual(prepared.tensor[0].mean().item(), 0.0, places=6)
        self.assertAlmostEqual(prepared.tensor[1].mean().item(), 1.0, places=6)
        self.assertAlmostEqual(prepared.tensor[2].mean().item(), 0.5, places=6)

    def test_handle_prepares_using_model_normalization_and_resolution(self):
        model = _FakeRFDETR()
        model.means = (0.0, 0.0, 0.0)
        model.stds = (1.0, 1.0, 1.0)
        handle = acceleration.apply_pytorch_optimization(model, acceleration.resolve_acceleration_config({}))

        prepared = handle.prepare_batch(torch.ones(3, 4, 5))

        self.assertEqual(prepared.tensor.shape, (1, 3, 32, 32))
        self.assertEqual(prepared.target_sizes.tolist(), [[4, 5]])
        self.assertAlmostEqual(prepared.tensor.mean().item(), 1.0, places=6)
        first = handle.consume_preprocess_timing()
        second = handle.consume_preprocess_timing()
        self.assertEqual(first["batches"], 1)
        self.assertEqual(second["batches"], 0)

    def test_deferred_preprocess_timing_is_idempotent_and_uses_exclusive_stages(self):
        h2d = MagicMock()
        h2d.consume_seconds.return_value = 0.02
        resize = MagicMock()
        resize.consume_seconds.return_value = 0.03
        timing = acceleration.PreprocessTiming(
            host_seconds=0.01,
            prepare_wall_seconds=0.015,
            h2d=h2d,
            resize_normalize=resize,
        )

        first = timing.consume()
        second = timing.consume()

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["device_preprocess_seconds"], 0.05)
        self.assertAlmostEqual(first["total_seconds"], 0.06)
        self.assertAlmostEqual(first["prepare_wall_seconds"], 0.015)
        h2d.consume_seconds.assert_called_once_with()
        resize.consume_seconds.assert_called_once_with()

    def test_rejects_ambiguous_tensor_layout_without_full_image_reductions(self):
        with self.assertRaisesRegex(ValueError, "CHW"):
            acceleration.prepare_inference_batch(
                torch.zeros(4, 5, 3),
                shape=4,
                device="cpu",
            )


class TensorRTUtilityTest(unittest.TestCase):
    def test_forward_timing_recorder_returns_reusable_events_after_consumption(self):
        recorder = acceleration.ForwardTimingRecorder()
        start = MagicMock()
        start.elapsed_time.return_value = 12.5
        end = MagicMock()
        release = MagicMock()

        recorder.add_cuda_events(start, end, release=release)

        self.assertAlmostEqual(recorder.consume_seconds(), 0.0125)
        end.synchronize.assert_called_once_with()
        release.assert_called_once_with()

    def test_preflight_fp32_does_not_import_optional_dependencies(self):
        with patch.object(acceleration.importlib, "import_module") as importer:
            result = acceleration.preflight_inference_acceleration({}, device="cpu")

        importer.assert_not_called()
        self.assertEqual(result["backend"], "pytorch")
        self.assertEqual(result["precision"], "fp32")

    def test_preflight_tensorrt_checks_lazy_dependencies_and_precision_flag(self):
        settings = acceleration.resolve_acceleration_config({"backend": "tensorrt", "tensorrt": {"precision": "bf16"}})
        fake_onnx = SimpleNamespace(__version__="1.17.0")
        with patch.object(acceleration, "_require_cuda") as require_cuda, patch.object(
            acceleration, "_import_tensorrt", return_value=_FakeTRT
        ) as import_trt, patch.object(acceleration, "_import_onnx", return_value=fake_onnx) as import_onnx:
            result = acceleration.preflight_inference_acceleration(settings, device="cuda:0")

        require_cuda.assert_called_once_with(torch.device("cuda:0"), bf16=True)
        import_trt.assert_called_once_with()
        import_onnx.assert_called_once_with()
        self.assertEqual(result["tensorrt_version"], "10.16.0")

    def test_normalizes_export_output_names_and_masks(self):
        tensors = [torch.zeros(1, 2, 4), torch.zeros(1, 2, 3), torch.zeros(1, 2, 8, 8)]
        result = acceleration.normalize_raw_outputs({"dets": tensors[0], "labels": tensors[1], "masks": tensors[2]})

        self.assertEqual(set(result), {"pred_boxes", "pred_logits", "pred_masks"})

    @patch.object(
        acceleration,
        '_validate_onnx_export_contract',
        side_effect=lambda path, **_kwargs: Path(path),
    )
    def test_dynamic_onnx_trace_always_uses_single_sample(self, validate):
        calls = []

        class ExportModel:
            model = SimpleNamespace(resolution=32)
            model_config = SimpleNamespace(
                num_channels=3,
                num_queries=10,
                num_classes=1,
                mask_downsample_ratio=4,
            )

            def export(self, **kwargs):
                calls.append(kwargs)
                destination = Path(kwargs["output_dir"]) / "model.onnx"
                destination.write_bytes(b"onnx")
                return destination

        settings = acceleration.resolve_acceleration_config(
            {
                "backend": "tensorrt",
                "tensorrt": {"profile": {"min_batch_size": 1, "opt_batch_size": 64, "max_batch_size": 320}},
            },
            resolution=32,
        )
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            acceleration, "_import_onnx", return_value=SimpleNamespace(__version__="1.17.0")
        ):
            acceleration._export_dynamic_onnx(ExportModel(), Path(temporary), settings)

        self.assertEqual(calls[0]["batch_size"], 1)
        self.assertTrue(calls[0]["dynamic_batch"])

        validate.assert_called_once()
        self.assertEqual(validate.call_args.kwargs['resolution'], 32)
        self.assertEqual(validate.call_args.kwargs['num_channels'], 3)
        self.assertFalse(validate.call_args.kwargs['segmentation'])
        self.assertEqual(validate.call_args.kwargs['num_queries'], 10)
        self.assertEqual(validate.call_args.kwargs['num_logit_slots'], 2)
        self.assertEqual(validate.call_args.kwargs['onnx_module'].__version__, '1.17.0')

    def test_onnx_contract_accepts_three_p2_levels_with_only_dynamic_batch(self):
        model_proto = SimpleNamespace(graph=_fake_p2_onnx_graph())
        fake_onnx = _fake_onnx_module(model_proto)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'model.onnx'
            source.write_bytes(b'original-onnx')

            validated = acceleration._validate_onnx_export_contract(
                source,
                resolution=32,
                num_channels=3,
                segmentation=False,
                onnx_module=fake_onnx,
            )

            self.assertEqual(validated, source)
            self.assertEqual(source.read_bytes(), b'shape-inferred-onnx')
        self.assertEqual(fake_onnx.checker.check_model.call_count, 2)
        fake_onnx.shape_inference.infer_shapes.assert_called_once_with(
            model_proto,
            check_type=True,
            strict_mode=True,
            data_prop=True,
        )

    def test_onnx_contract_rejects_symbolic_non_batch_io_axis(self):
        graph = _fake_p2_onnx_graph(label_slots='classes')
        model_proto = SimpleNamespace(graph=graph)
        fake_onnx = _fake_onnx_module(model_proto)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'model.onnx'
            source.write_bytes(b'original-onnx')

            with self.assertRaises(RuntimeError) as raised:
                acceleration._validate_onnx_export_contract(
                    source,
                    resolution=32,
                    num_channels=3,
                    segmentation=False,
                    onnx_module=fake_onnx,
                )

        self.assertIn('labels', str(raised.exception))
        self.assertIn('axis 2 must be static, got classes', str(raised.exception))
        fake_onnx.save_model.assert_not_called()

    def test_onnx_contract_reports_dynamic_convtranspose_channel_with_node_and_tensor(self):
        graph = _fake_p2_onnx_graph(p2_channel='p2_channels')
        model_proto = SimpleNamespace(graph=graph)
        fake_onnx = _fake_onnx_module(model_proto)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'model.onnx'
            source.write_bytes(b'original-onnx')

            with self.assertRaises(RuntimeError) as raised:
                acceleration._validate_onnx_export_contract(
                    source,
                    resolution=32,
                    num_channels=3,
                    segmentation=False,
                    onnx_module=fake_onnx,
                )

        message = str(raised.exception)
        self.assertIn('ConvTranspose', message)
        self.assertIn('p2_upsample', message)
        self.assertIn('p2_feature', message)
        self.assertIn('channel axis must be static', message)
        self.assertIn('p2_channels', message)

    def test_onnx_contract_checks_conv_channels_too(self):
        graph = _fake_p2_onnx_graph(p2_channel='conv_channels')
        graph.node[0].op_type = 'Conv'

        issues = acceleration._onnx_convolution_contract_issues(graph)

        self.assertTrue(any('Conv node' in issue for issue in issues))
        self.assertTrue(any('p2_feature' in issue for issue in issues))
        self.assertTrue(any('conv_channels' in issue for issue in issues))

    def test_onnx_contract_rejects_legacy_scatternd_shape_rewrite(self):
        graph = _fake_p2_onnx_graph()
        graph.node.append(
            SimpleNamespace(
                op_type='ScatterND',
                name='legacy_dynamic_shape',
                input=['shape', 'indices', 'updates'],
                output=['rewritten_shape'],
            )
        )
        model_proto = SimpleNamespace(graph=graph)
        fake_onnx = _fake_onnx_module(model_proto)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'model.onnx'
            source.write_bytes(b'original-onnx')
            with self.assertRaisesRegex(RuntimeError, 'ScatterND.*rfdetr==1.8.3'):
                acceleration._validate_onnx_export_contract(
                    source,
                    resolution=32,
                    num_channels=3,
                    segmentation=False,
                    onnx_module=fake_onnx,
                )

    def test_optional_dependency_versions_are_strict(self):
        with patch.object(
            acceleration.importlib, "import_module", return_value=SimpleNamespace(__version__="10.15.0")
        ), self.assertRaisesRegex(RuntimeError, ">=10.16,<11"):
            acceleration._import_tensorrt()
        with patch.object(
            acceleration.importlib, "import_module", return_value=SimpleNamespace(__version__="11.0.0")
        ), self.assertRaisesRegex(RuntimeError, ">=10.16,<11"):
            acceleration._import_tensorrt()
        with patch.object(
            acceleration.importlib, "import_module", return_value=SimpleNamespace(__version__="1.15.0")
        ), self.assertRaisesRegex(RuntimeError, "ONNX >=1.16,<2"):
            acceleration._import_onnx()

    def test_logit_slots_include_rfdetr_background_class(self):
        model = _FakeRFDETR()
        model.model_config.num_classes = 2
        self.assertEqual(acceleration._model_num_logit_slots(model), 3)
        model.model.model.class_embed = torch.nn.Linear(4, 7)
        self.assertEqual(acceleration._model_num_logit_slots(model), 7)

    @staticmethod
    def _contract_runner(*, segmentation=True):
        runner = object.__new__(acceleration.TensorRTRunner)
        runner.manifest = {
            "identity": {
                "resolution": [32, 32],
                "num_channels": 3,
                "num_classes": 2,
                "num_logit_slots": 3,
                "num_queries": 10,
                "segmentation": segmentation,
                "mask_downsample_ratio": 4 if segmentation else None,
                "profile": {"min_batch_size": 1, "opt_batch_size": 7, "max_batch_size": 24},
                "outputs": ["dets", "labels", *(["masks"] if segmentation else [])],
                "io_contract": {
                    "input_dtype": "float32",
                    "input_rank": 4,
                    "output_dtypes": {},
                    "output_ranks": {},
                },
            }
        }
        runner.resolution = 32
        runner.min_shape = (1, 3, 32, 32)
        runner.opt_shape = (7, 3, 32, 32)
        runner.max_shape = (24, 3, 32, 32)
        runner._output_names = ["dets", "labels", *(["masks"] if segmentation else [])]
        runner._semantic_output_names = {"pred_boxes": "dets", "pred_logits": "labels"}
        runner._output_shapes = {"dets": (-1, 10, 4), "labels": (-1, 10, 3)}
        if segmentation:
            runner._semantic_output_names["pred_masks"] = "masks"
            runner._output_shapes["masks"] = (-1, 10, 8, 8)
        return runner

    def test_engine_contract_accepts_background_logits_and_downsampled_masks(self):
        runner = self._contract_runner()
        runner._validate_output_shapes()
        runner._validate_engine_contract()

    def test_engine_contract_rejects_wrong_logits_or_mask_shape(self):
        runner = self._contract_runner()
        runner._output_shapes["labels"] = (-1, 10, 2)
        with self.assertRaisesRegex(RuntimeError, "logit-slot count"):
            runner._validate_engine_contract()

        runner = self._contract_runner()
        runner._output_shapes["masks"] = (-1, 10, 7, 7)
        with self.assertRaisesRegex(RuntimeError, "mask size"):
            runner._validate_engine_contract()

    def test_runner_retains_only_incomplete_async_bindings(self):
        runner = object.__new__(acceleration.TensorRTRunner)
        incomplete = SimpleNamespace(query=lambda: False)
        complete = SimpleNamespace(query=lambda: True)
        runner._completion_events = MagicMock()
        runner._pending_bindings = [(incomplete, {"input": object()}), (complete, {"input": object()})]

        runner._release_completed_bindings()

        self.assertEqual(len(runner._pending_bindings), 1)
        self.assertIs(runner._pending_bindings[0][0], incomplete)
        runner._completion_events.release.assert_called_once_with(complete)

    def test_runner_rejects_new_manifest_from_another_physical_gpu_before_deserialize(self):
        runner = object.__new__(acceleration.TensorRTRunner)
        runner.device = torch.device("cuda:0")
        runner.manifest = {"identity": {"gpu": {"uuid": "GPU-engine-device"}}}
        properties = SimpleNamespace(uuid="selected-device")

        with patch.object(acceleration.torch.cuda, "device", return_value=nullcontext()), patch.object(
            acceleration.torch.cuda, "get_device_properties", return_value=properties
        ), self.assertRaisesRegex(RuntimeError, "physical GPU UUID"):
            runner._validate_physical_device()

    def test_runner_allows_legacy_manifest_without_physical_uuid(self):
        runner = object.__new__(acceleration.TensorRTRunner)
        runner.device = torch.device("cuda:0")
        runner.manifest = {
            "identity": {"gpu": {"name": "NVIDIA GeForce RTX 5090", "compute_capability": [12, 0]}}
        }

        with patch.object(acceleration.torch.cuda, "get_device_properties") as properties:
            runner._validate_physical_device()

        properties.assert_not_called()

    def test_chunking_preserves_dynamic_tail(self):
        self.assertEqual(acceleration.chunk_batch_ranges(135, 64), [(0, 64), (64, 128), (128, 135)])

    def test_runner_chunks_above_profile_max_without_real_tensorrt(self):
        runner = object.__new__(acceleration.TensorRTRunner)
        runner.max_shape = (64, 3, 4, 4)
        runner.max_batch_size = 64
        seen = []

        def fake_chunk(tensor):
            seen.append(tensor.shape[0])
            n = tensor.shape[0]
            return {
                "pred_boxes": torch.zeros(n, 2, 4),
                "pred_logits": torch.zeros(n, 2, 3),
                "pred_masks": torch.zeros(n, 2, 4, 4),
            }

        runner._infer_chunk = fake_chunk
        result = runner.infer(torch.zeros(135, 3, 4, 4))

        self.assertEqual(seen, [64, 64, 7])
        self.assertEqual(result["pred_boxes"].shape[0], 135)
        self.assertEqual(result["pred_masks"].shape[0], 135)

    def test_runner_preserves_common_dynamic_batch_sizes(self):
        runner = object.__new__(acceleration.TensorRTRunner)
        runner.max_shape = (64, 3, 4, 4)
        runner.max_batch_size = 64
        runner._infer_chunk = lambda tensor: {
            "pred_boxes": torch.zeros(tensor.shape[0], 2, 4),
            "pred_logits": torch.zeros(tensor.shape[0], 2, 3),
        }

        for batch_size in (1, 7, 24, 64):
            with self.subTest(batch_size=batch_size):
                result = runner.infer(torch.zeros(batch_size, 3, 4, 4))
                self.assertEqual(result["pred_boxes"].shape[0], batch_size)

    @staticmethod
    def _multi_profile_runner_on_cpu():
        runner = object.__new__(acceleration.TensorRTRunner)
        runner.device = torch.device("cpu")
        runner.input_dtype = torch.float32
        runner.min_shape = (1, 3, 4, 4)
        runner.opt_shape = (1, 3, 4, 4)
        runner.max_shape = (16, 3, 4, 4)
        runner.max_batch_size = 16
        runner.profile_shapes = (
            ((1, 3, 4, 4), (1, 3, 4, 4), (16, 3, 4, 4)),
            ((1, 3, 4, 4), (4, 3, 4, 4), (16, 3, 4, 4)),
            ((1, 3, 4, 4), (8, 3, 4, 4), (16, 3, 4, 4)),
            ((1, 3, 4, 4), (16, 3, 4, 4), (16, 3, 4, 4)),
        )
        runner._selected_profile_index = 0
        runner._semantic_output_names = {"pred_boxes": "dets", "pred_logits": "labels"}
        runner._output_shapes = {"dets": (-1, 2, 4), "labels": (-1, 2, 3)}
        runner._output_dtypes = {"dets": torch.float32, "labels": torch.float32}
        return runner

    def test_profile_selection_prefers_autotuned_profile_and_falls_back_by_optimum(self):
        runner = self._multi_profile_runner_on_cpu()
        runner._selected_profile_index = 2

        self.assertEqual(runner._profile_index_for_batch(4), 2)
        runner.profile_shapes = (runner.profile_shapes[0], runner.profile_shapes[1])
        runner._selected_profile_index = 9
        self.assertEqual(runner._profile_index_for_batch(4), 1)

    def test_caller_owned_output_buffers_can_be_reused_without_reallocation(self):
        runner = self._multi_profile_runner_on_cpu()
        buffers = runner.allocate_output_buffers(4)

        def fake_chunk(_tensor, *, output_bindings=None, profile_index=None):
            self.assertIsNone(profile_index)
            return acceleration.normalize_raw_outputs(output_bindings)

        runner._infer_chunk = fake_chunk
        tensor = torch.zeros(4, 3, 4, 4)
        first = runner.infer_into(tensor, buffers)
        second = runner.infer_into(tensor, buffers)

        self.assertIs(first["pred_boxes"], buffers["pred_boxes"])
        self.assertIs(second["pred_logits"], buffers["pred_logits"])
        with self.assertRaisesRegex(ValueError, "must contain"):
            runner.infer_into(tensor, {"pred_boxes": buffers["pred_boxes"]})

    def test_profile_autotune_selects_lowest_median_without_global_synchronize(self):
        runner = self._multi_profile_runner_on_cpu()
        durations = {0: 0.004, 1: 0.002, 2: 0.003, 3: 0.005}
        runner.allocate_output_buffers = lambda _batch: {}
        runner.infer_into = lambda _tensor, _buffers: {}
        runner.synchronize = MagicMock()
        runner.consume_forward_seconds = lambda: durations[runner._selected_profile_index]

        with patch.object(acceleration.torch.cuda, "synchronize") as global_sync:
            report = runner.autotune_profiles(4, warmup_iterations=0, measure_iterations=3)

        self.assertEqual(report["selected_profile_index"], 1)
        self.assertEqual(runner._selected_profile_index, 1)
        self.assertEqual(runner.synchronize.call_count, 12)
        global_sync.assert_not_called()

    def test_activate_profile_uses_private_stream_async_api(self):
        runner = object.__new__(acceleration.TensorRTRunner)
        runner._active_profile_index = 0
        runner._stream = SimpleNamespace(cuda_stream=123)
        runner._context = SimpleNamespace(set_optimization_profile_async=MagicMock(return_value=True))

        runner._activate_profile(2)

        runner._context.set_optimization_profile_async.assert_called_once_with(2, 123)
        self.assertEqual(runner._active_profile_index, 2)

    def test_manifest_detects_engine_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "model.engine"
            manifest = root / "model.engine.manifest.json"
            engine.write_bytes(b"valid-engine")
            expected = {"schema_version": 1, "cache_key": "abc", "identity": {"model": "x"}}
            expected['schema_version'] = acceleration._MANIFEST_SCHEMA_VERSION
            manifest.write_text(
                json.dumps(
                    {
                        **expected,
                        "engine_sha256": hashlib.sha256(b"valid-engine").hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            loaded = acceleration.validate_engine_manifest(engine, manifest, expected)
            self.assertEqual(loaded["cache_key"], "abc")
            engine.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "hash"):
                acceleration.validate_engine_manifest(engine, manifest, expected)

    def test_missing_tensorrt_has_actionable_lazy_error(self):
        real_import = acceleration.importlib.import_module

        def import_module(name):
            if name == "tensorrt":
                raise ImportError("missing")
            return real_import(name)

        with patch.object(acceleration.importlib, "import_module", side_effect=import_module):
            with self.assertRaisesRegex(ImportError, "TensorRT optional"):
                acceleration._import_tensorrt()

    def test_loaded_model_state_hash_handles_hosted_weight_token(self):
        model = _FakeRFDETR()
        model.model_config.pretrain_weights = "default"

        first = acceleration._checkpoint_identity(model, "default")
        model.model.model.weight.data.fill_(2)
        second = acceleration._checkpoint_identity(model, "default")

        self.assertEqual(first["source"], "loaded_model_state")
        self.assertNotEqual(first["sha256"], second["sha256"])


class TensorRTArtifactCacheTest(unittest.TestCase):
    @staticmethod
    def _identity(tag="primary"):
        return {
            "checkpoint": {"source": "loaded_model_state", "sha256": f"checkpoint-{tag}"},
            "model": {"size": "medium", "num_classes": 1},
            "resolution": [32, 32],
            "num_channels": 3,
            "outputs": ["dets", "labels"],
            "precision": "fp16",
            "profile": {"min_batch_size": 1, "opt_batch_size": 7, "max_batch_size": 64},
            "runtime": {"rfdetr": "test", "tensorrt": "10.16.0", "torch": "test", "cuda": "test"},
            "gpu": {"name": "fake-gpu", "compute_capability": [8, 9]},
        }

    @staticmethod
    def _settings(cache_dir, *, force_rebuild=False, engine_path=None, manifest_path=None):
        tensorrt = {
            "precision": "fp16",
            "cache_dir": str(cache_dir),
            "force_rebuild": force_rebuild,
            "profile": {"min_batch_size": 1, "opt_batch_size": 7, "max_batch_size": 64},
        }
        if engine_path is not None:
            tensorrt["engine_path"] = str(engine_path)
        if manifest_path is not None:
            tensorrt["manifest_path"] = str(manifest_path)
        return acceleration.resolve_acceleration_config(
            {"backend": "tensorrt", "tensorrt": tensorrt},
            resolution=32,
        )

    def test_manifest_schema_and_export_abi_change_cache_key(self):
        identity = self._identity()
        identity['export_abi'] = {
            'version': acceleration._TENSORRT_EXPORT_ABI_VERSION,
            'shape_contract': acceleration._TENSORRT_EXPORT_SHAPE_CONTRACT,
            'dynamic_axes': ['batch'],
        }

        current = acceleration._manifest_template(identity)
        repeated = acceleration._manifest_template(identity)
        old_identity = dict(identity)
        old_identity['export_abi'] = dict(identity['export_abi'], version=1)
        old = acceleration._manifest_template(old_identity)

        self.assertEqual(current['schema_version'], 2)
        self.assertEqual(current['cache_key'], repeated['cache_key'])
        self.assertNotEqual(current['cache_key'], old['cache_key'])

    def test_physical_gpu_uuid_and_vbios_are_part_of_cache_key(self):
        first_identity = self._identity()
        first_identity["gpu"].update(
            {
                "uuid": "GPU-first",
                "pci_bus_id": "00000000:16:00.0",
                "pci_sub_device_id": "0xf3181569",
                "vbios_version": "98.02.2e.80.10",
            }
        )
        second_identity = json.loads(json.dumps(first_identity))
        second_identity["gpu"]["uuid"] = "GPU-second"
        third_identity = json.loads(json.dumps(first_identity))
        third_identity["gpu"]["vbios_version"] = "98.02.2e.80.0f"

        first = acceleration._manifest_template(first_identity)
        second = acceleration._manifest_template(second_identity)
        third = acceleration._manifest_template(third_identity)

        self.assertNotEqual(first["cache_key"], second["cache_key"])
        self.assertNotEqual(first["cache_key"], third["cache_key"])

    def test_gpu_identity_combines_torch_properties_with_matching_nvidia_smi_row(self):
        properties = SimpleNamespace(
            uuid="f95dc184-c651-6edb-cdb0-6b17d64e57de",
            pci_domain_id=0,
            pci_bus_id=22,
            pci_device_id=0,
            total_memory=32_000,
            multi_processor_count=170,
            L2_cache_size=96_000,
            memory_bus_width=512,
            is_multi_gpu_board=False,
        )
        smi = {
            "uuid": "GPU-f95dc184-c651-6edb-cdb0-6b17d64e57de",
            "pci_bus_id": "00000000:16:00.0",
            "pci_device_id": "0x2b8510de",
            "pci_sub_device_id": "0xf3181569",
            "vbios_version": "98.02.2E.80.10",
            "driver_version": "580.82.07",
        }
        with patch.object(acceleration.torch.cuda, "device", return_value=nullcontext()), patch.object(
            acceleration.torch.cuda, "get_device_properties", return_value=properties
        ), patch.object(acceleration.torch.cuda, "get_device_capability", return_value=(12, 0)), patch.object(
            acceleration.torch.cuda, "get_device_name", return_value="NVIDIA GeForce RTX 5090"
        ), patch.object(acceleration, "_nvidia_smi_gpu_details", return_value=smi):
            identity = acceleration._gpu_runtime_identity(torch.device("cuda:0"))

        self.assertEqual(identity["uuid"], smi["uuid"])
        self.assertEqual(identity["pci_bus_id"], smi["pci_bus_id"])
        self.assertEqual(identity["pci_sub_device_id"], smi["pci_sub_device_id"])
        self.assertEqual(identity["vbios_version"], smi["vbios_version"])
        self.assertEqual(identity["driver_version"], smi["driver_version"])
        self.assertEqual(identity["compute_capability"], [12, 0])
        self.assertEqual(identity["multiprocessor_count"], 170)

    def test_nvidia_smi_details_select_the_physical_uuid_not_logical_index(self):
        output = "\n".join(
            [
                "GPU-first, 00000000:16:00.0, 0x2B8510DE, 0xF3181569, bios-a, 580.82.07",
                "GPU-second, 00000000:27:00.0, 0x2B8510DE, 0xF3181569, bios-b, 580.82.07",
            ]
        )
        completed = SimpleNamespace(returncode=0, stdout=output)
        with patch.object(acceleration.shutil, "which", return_value="/usr/bin/nvidia-smi"), patch.object(
            acceleration.subprocess, "run", return_value=completed
        ) as run:
            details = acceleration._nvidia_smi_gpu_details("second")

        self.assertEqual(details["uuid"], "GPU-second")
        self.assertEqual(details["pci_bus_id"], "00000000:27:00.0")
        self.assertEqual(details["vbios_version"], "bios-b")
        run.assert_called_once()

    def test_public_gpu_identity_wrapper_uses_the_cache_identity_implementation(self):
        expected = {"uuid": "GPU-test", "pci_bus_id": "00000000:01:00.0"}
        with patch.object(
            acceleration, "_preflight_device", return_value=torch.device("cuda:3")
        ) as resolve, patch.object(acceleration, "_require_cuda") as require, patch.object(
            acceleration, "_gpu_runtime_identity", return_value=expected
        ) as identify:
            result = acceleration.gpu_runtime_identity(3)

        self.assertEqual(result, expected)
        self.assertIsNot(result, expected)
        resolve.assert_called_once_with(3)
        require.assert_called_once_with(torch.device("cuda:3"))
        identify.assert_called_once_with(torch.device("cuda:3"))

    def test_onnx_cache_key_ignores_physical_gpu_and_tensorrt_only_fields(self):
        first = self._identity()
        first["gpu"].update({"uuid": "GPU-first", "vbios_version": "bios-a"})
        first["runtime"]["nvidia_driver"] = "driver-a"
        first["optimization_profiles"] = [{"opt_batch_size": 4}]
        second = json.loads(json.dumps(first))
        second["gpu"].update({"uuid": "GPU-second", "vbios_version": "bios-b"})
        second["runtime"].update({"nvidia_driver": "driver-b", "tensorrt": "10.16.2"})
        second["precision"] = "bf16"
        second["profile"]["opt_batch_size"] = 16
        second["optimization_profiles"] = [{"opt_batch_size": 16}]

        first_onnx = acceleration._onnx_manifest_template(first)
        second_onnx = acceleration._onnx_manifest_template(second)

        self.assertEqual(first_onnx["cache_key"], second_onnx["cache_key"])
        self.assertNotIn("gpu", first_onnx["identity"])
        self.assertNotIn("tensorrt", first_onnx["identity"]["runtime"])

    def test_onnx_cache_manifest_rejects_corrupted_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            onnx = root / "model.onnx"
            manifest_path = root / "model.onnx.manifest.json"
            onnx.write_bytes(b"valid-onnx")
            expected = acceleration._onnx_manifest_template(self._identity())
            manifest_path.write_text(
                json.dumps(
                    {
                        **expected,
                        "onnx_sha256": hashlib.sha256(b"valid-onnx").hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            acceleration._validate_cached_onnx(onnx, manifest_path, expected)
            onnx.write_bytes(b"corrupt-onnx")

            with self.assertRaisesRegex(RuntimeError, "hash"):
                acceleration._validate_cached_onnx(onnx, manifest_path, expected)

    def test_cache_miss_hit_force_rebuild_and_manifest_last_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            settings = self._settings(cache_dir)
            force_settings = self._settings(cache_dir, force_rebuild=True)
            build_payloads = []
            replace_destinations = []
            real_replace = acceleration.os.replace

            def fake_export(_model, output_dir, _settings):
                exported = Path(output_dir) / "exported.onnx"
                exported.write_bytes(b"fake-onnx")
                return exported

            def fake_build(_onnx_path, engine_path, _settings, **kwargs):
                payload = f"fake-engine-{len(build_payloads) + 1}".encode("ascii")
                build_payloads.append(payload)
                Path(engine_path).write_bytes(payload)
                timing_path = kwargs.get("timing_cache_path")
                if timing_path is not None:
                    Path(timing_path).write_bytes(b"fake-timing-cache")
                return Path(engine_path)

            def track_replace(source, destination):
                replace_destinations.append(Path(destination))
                return real_replace(source, destination)

            with patch.object(
                acceleration, "_import_tensorrt", return_value=SimpleNamespace(__version__="10.16.0")
            ), patch.object(acceleration, "_build_identity", return_value=self._identity()), patch.object(
                acceleration, "_export_dynamic_onnx", side_effect=fake_export
            ), patch.object(acceleration, "build_tensorrt_engine", side_effect=fake_build), patch.object(
                acceleration, "_model_device", return_value=torch.device("cuda:0")
            ), patch.object(acceleration.torch.cuda, "device", return_value=nullcontext()), patch.object(
                acceleration.os, "replace", side_effect=track_replace
            ):
                first = acceleration.prepare_tensorrt_engine(object(), settings, segmentation=False)
                second = acceleration.prepare_tensorrt_engine(object(), settings, segmentation=False)
                forced = acceleration.prepare_tensorrt_engine(object(), force_settings, segmentation=False)

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertFalse(forced.cache_hit)
            self.assertEqual(build_payloads, [b"fake-engine-1", b"fake-engine-2"])
            self.assertEqual(first.engine_path, second.engine_path)
            self.assertEqual(first.engine_path, forced.engine_path)
            self.assertEqual(forced.engine_path.read_bytes(), b"fake-engine-2")
            self.assertEqual(replace_destinations[-1], forced.manifest_path)
            self.assertTrue(forced.manifest_path.is_file())
            self.assertEqual(
                forced.manifest["engine_sha256"],
                hashlib.sha256(b"fake-engine-2").hexdigest(),
            )
            self.assertFalse(any(path.name.endswith(".tmp") for path in cache_dir.iterdir()))

    def test_two_physical_gpu_engine_keys_share_one_validated_onnx_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            settings = self._settings(cache_dir)
            first_identity = self._identity("shared-onnx")
            first_identity["gpu"].update({"uuid": "GPU-first", "vbios_version": "bios-a"})
            first_identity["runtime"]["nvidia_driver"] = "driver-a"
            second_identity = json.loads(json.dumps(first_identity))
            second_identity["gpu"].update({"uuid": "GPU-second", "vbios_version": "bios-b"})
            second_identity["runtime"]["nvidia_driver"] = "driver-b"
            exports = []
            builds = []

            def fake_export(_model, output_dir, _settings):
                exports.append(Path(output_dir))
                exported = Path(output_dir) / "exported.onnx"
                exported.write_bytes(b"shared-onnx")
                return exported

            def fake_build(_onnx_path, engine_path, _settings, **_kwargs):
                payload = f"engine-{len(builds)}".encode()
                builds.append(payload)
                Path(engine_path).write_bytes(payload)
                return Path(engine_path)

            with patch.object(
                acceleration, "_import_tensorrt", return_value=SimpleNamespace(__version__="10.16.0")
            ), patch.object(
                acceleration, "_build_identity", side_effect=[first_identity, second_identity]
            ), patch.object(
                acceleration, "_export_dynamic_onnx", side_effect=fake_export
            ), patch.object(
                acceleration, "build_tensorrt_engine", side_effect=fake_build
            ), patch.object(
                acceleration, "_model_device", return_value=torch.device("cuda:0")
            ), patch.object(acceleration.torch.cuda, "device", return_value=nullcontext()):
                first = acceleration.prepare_tensorrt_engine(object(), settings, segmentation=False)
                second = acceleration.prepare_tensorrt_engine(object(), settings, segmentation=False)

            self.assertEqual(len(exports), 1)
            self.assertEqual(len(builds), 2)
            self.assertNotEqual(first.engine_path, second.engine_path)
            self.assertEqual(first.onnx_path, second.onnx_path)
            self.assertTrue(first.onnx_path.is_file())
            self.assertFalse(first.manifest["onnx_cache_hit"])
            self.assertTrue(second.manifest["onnx_cache_hit"])

    def test_export_abi_change_rebuilds_once_then_hits_new_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary))
            old_identity = self._identity()
            old_identity['export_abi'] = {'version': 1}
            current_identity = self._identity()
            current_identity['export_abi'] = {
                'version': acceleration._TENSORRT_EXPORT_ABI_VERSION,
                'shape_contract': acceleration._TENSORRT_EXPORT_SHAPE_CONTRACT,
                'dynamic_axes': ['batch'],
            }
            identities = [old_identity, current_identity, current_identity]
            builds = []

            def fake_export(_model, output_dir, _settings):
                exported = Path(output_dir) / 'exported.onnx'
                exported.write_bytes(b'fake-onnx')
                return exported

            def fake_build(_onnx_path, engine_path, _settings, **_kwargs):
                payload = f'engine-{len(builds) + 1}'.encode('ascii')
                builds.append(payload)
                Path(engine_path).write_bytes(payload)
                return Path(engine_path)

            with patch.object(
                acceleration,
                '_import_tensorrt',
                return_value=SimpleNamespace(__version__='10.16.0'),
            ), patch.object(
                acceleration, '_build_identity', side_effect=identities
            ), patch.object(
                acceleration, '_export_dynamic_onnx', side_effect=fake_export
            ), patch.object(
                acceleration, 'build_tensorrt_engine', side_effect=fake_build
            ), patch.object(
                acceleration, '_model_device', return_value=torch.device('cuda:0')
            ), patch.object(
                acceleration.torch.cuda, 'device', return_value=nullcontext()
            ):
                old = acceleration.prepare_tensorrt_engine(object(), settings, segmentation=False)
                rebuilt = acceleration.prepare_tensorrt_engine(object(), settings, segmentation=False)
                hit = acceleration.prepare_tensorrt_engine(object(), settings, segmentation=False)

        self.assertFalse(old.cache_hit)
        self.assertFalse(rebuilt.cache_hit)
        self.assertTrue(hit.cache_hit)
        self.assertEqual(builds, [b'engine-1', b'engine-2'])
        self.assertNotEqual(old.engine_path, rebuilt.engine_path)
        self.assertEqual(rebuilt.engine_path, hit.engine_path)

    def test_artifact_lock_serializes_competing_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "artifact.lock"
            quoted_path = repr(str(lock_path))
            waiter_script = f"""
from pathlib import Path
import time
from rf_detr_acceleration import _ArtifactLock
print("ready", flush=True)
input()
started = time.perf_counter()
with _ArtifactLock(Path({quoted_path}), timeout=10.0, stale_after=60.0):
    waited = time.perf_counter() - started
print(waited, flush=True)
"""
            holder_script = f"""
from pathlib import Path
import time
from rf_detr_acceleration import _ArtifactLock
with _ArtifactLock(Path({quoted_path}), timeout=10.0, stale_after=60.0):
    print("entered", flush=True)
    time.sleep(0.8)
"""
            waiter = subprocess.Popen(
                [sys.executable, "-c", waiter_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            holder = None
            try:
                self.assertEqual(waiter.stdout.readline().strip(), "ready")
                holder = subprocess.Popen(
                    [sys.executable, "-c", holder_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(holder.stdout.readline().strip(), "entered")
                waiter.stdin.write("go\n")
                waiter.stdin.flush()
                waiter_output, waiter_error = waiter.communicate(timeout=15.0)
                holder_output, holder_error = holder.communicate(timeout=15.0)
                self.assertEqual(waiter.returncode, 0, waiter_error)
                self.assertEqual(holder.returncode, 0, holder_error + holder_output)
                self.assertGreaterEqual(float(waiter_output.strip()), 0.5)
                self.assertFalse(lock_path.exists())
            finally:
                for process in (waiter, holder):
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait(timeout=5.0)

    def test_supplied_engine_manifest_identity_mismatch_fails_before_export_or_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "trusted.engine"
            manifest = root / "trusted.engine.manifest.json"
            engine.write_bytes(b"trusted-engine")
            expected = acceleration._manifest_template(self._identity("original"))
            manifest.write_text(
                json.dumps(
                    {
                        **expected,
                        "engine_sha256": hashlib.sha256(b"trusted-engine").hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            settings = self._settings(
                root / "unused-cache",
                engine_path=engine,
                manifest_path=manifest,
            )

            with patch.object(
                acceleration, "_import_tensorrt", return_value=SimpleNamespace(__version__="10.16.0")
            ), patch.object(
                acceleration, "_build_identity", return_value=self._identity("different-checkpoint")
            ), patch.object(acceleration, "_export_dynamic_onnx") as export_onnx, patch.object(
                acceleration, "build_tensorrt_engine"
            ) as build_engine:
                with self.assertRaisesRegex(RuntimeError, "manifest (cache_key|identity)"):
                    acceleration.prepare_tensorrt_engine(object(), settings, segmentation=False)

            export_onnx.assert_not_called()
            build_engine.assert_not_called()


class _FakeInput:
    name = "input"
    shape = (-1, 3, 32, 32)


class _FakeNetwork:
    num_inputs = 1

    def get_input(self, _index):
        return _FakeInput()


class _FakeProfile:
    def __init__(self):
        self.shapes = None

    def set_shape(self, *shapes):
        self.shapes = shapes
        return True


class _FakeBuildConfig:
    def __init__(self):
        self.workspace = None
        self.flag = None
        self.profiles = []

    def set_memory_pool_limit(self, pool, size):
        self.workspace = (pool, size)

    def set_flag(self, flag):
        self.flag = flag

    def add_optimization_profile(self, profile):
        self.profiles.append(profile)
        return len(self.profiles) - 1


class _FakeBuilder:
    last = None
    platform_has_fast_fp16 = True
    platform_has_fast_bf16 = True

    def __init__(self, _logger):
        type(self).last = self
        self.network = _FakeNetwork()
        self.config = _FakeBuildConfig()

    def create_network(self, _flag):
        return self.network

    def create_builder_config(self):
        return self.config

    def create_optimization_profile(self):
        return _FakeProfile()

    def build_serialized_network(self, _network, _config):
        return b"fake-engine"


class _FakeParser:
    num_errors = 0

    def __init__(self, _network, _logger):
        pass

    def parse(self, _data):
        return True


class _FakeLogger:
    WARNING = 1

    def __init__(self, _severity=None):
        pass


class _FakeTRT:
    __version__ = "10.16.0"
    Logger = _FakeLogger
    Builder = _FakeBuilder
    OnnxParser = _FakeParser
    NetworkDefinitionCreationFlag = SimpleNamespace(EXPLICIT_BATCH=0)
    MemoryPoolType = SimpleNamespace(WORKSPACE="workspace")
    BuilderFlag = SimpleNamespace(FP16="fp16", BF16="bf16")


class TensorRTBuilderTest(unittest.TestCase):
    def test_builder_uses_dynamic_batch_profile_and_selected_precision(self):
        settings = acceleration.resolve_acceleration_config(
            {
                "backend": "tensorrt",
                "tensorrt": {
                    "precision": "bf16",
                    "profile": {"min_batch_size": 1, "opt_batch_size": 7, "max_batch_size": 64},
                },
            },
            resolution=32,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            onnx = root / "model.onnx"
            engine = root / "model.engine"
            onnx.write_bytes(b"fake-onnx")

            acceleration.build_tensorrt_engine(onnx, engine, settings, trt_module=_FakeTRT)

            builder = _FakeBuilder.last
            self.assertEqual(builder.config.flag, "bf16")
            self.assertEqual(
                [profile.shapes[2][0] for profile in builder.config.profiles],
                [1, 4, 7, 8, 16],
            )
            self.assertTrue(
                all(
                    profile.shapes[1] == (1, 3, 32, 32)
                    and profile.shapes[3] == (64, 3, 32, 32)
                    for profile in builder.config.profiles
                )
            )
            self.assertEqual(engine.read_bytes(), b"fake-engine")


class TensorRTPredictAdapterTest(unittest.TestCase):
    def test_adapter_delegates_predict_and_can_restore_pytorch_weights(self):
        source = _FakeRFDETR()
        original = source.model.model
        runner = SimpleNamespace(
            device=torch.device("cpu"),
            engine_path=Path("fake.engine"),
            resolution=32,
            input_dtype=torch.float32,
            infer=lambda tensor: {
                "pred_boxes": torch.zeros(tensor.shape[0], 2, 4),
                "pred_logits": torch.zeros(tensor.shape[0], 2, 3),
            },
        )

        adapter = acceleration.install_tensorrt_backend(source, runner)

        self.assertEqual(adapter.predict("image"), (("image",), {}))
        self.assertEqual(adapter.model_config.resolution, 32)
        self.assertEqual(adapter.class_names, ["ball"])
        self.assertEqual(adapter.infer_raw(torch.zeros(1, 3, 32, 32))["pred_boxes"].shape, (1, 2, 4))
        restored = adapter.restore_pytorch_model()
        self.assertIs(restored.model.model, original)

    def test_accuracy_parity_hook_preserves_metric_details_and_callback_errors(self):
        handle = SimpleNamespace()

        accepted = acceleration.run_inference_accuracy_parity_check(
            handle,
            lambda _handle: {"accepted": True, "delta_map": -0.2, "football_recall_delta": -0.004},
        )
        failed = acceleration.run_inference_accuracy_parity_check(
            handle,
            lambda _handle: (_ for _ in ()).throw(RuntimeError("validation unavailable")),
        )

        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["reference_precision"], "bf16")
        self.assertEqual(accepted["details"]["delta_map"], -0.2)
        self.assertFalse(failed["accepted"])
        self.assertIn("validation unavailable", failed["error"])

    def test_configure_falls_back_to_pytorch_bf16_when_parity_gate_rejects_fp16(self):
        source = _FakeRFDETR(device="cuda:0")
        settings = acceleration.resolve_acceleration_config(
            {
                "backend": "tensorrt",
                "tensorrt": {
                    "precision": "fp16",
                    "profile": {"min_batch_size": 1, "opt_batch_size": 4, "max_batch_size": 4},
                },
            },
            resolution=32,
        )
        manifest = {
            "engine_sha256": "engine-hash",
            "identity": {"gpu": {"uuid": "GPU-test"}},
        }
        artifact = acceleration.TensorRTArtifact(
            engine_path=Path("fake.engine"),
            manifest_path=Path("fake.engine.manifest.json"),
            onnx_path=None,
            cache_hit=True,
            manifest=manifest,
        )
        runner = SimpleNamespace(
            device=torch.device("cuda:0"),
            engine_path=Path("fake.engine"),
            resolution=32,
            input_dtype=torch.float32,
            forward_timing=acceleration.ForwardTimingRecorder(),
            infer=lambda tensor: {
                "pred_boxes": torch.zeros(tensor.shape[0], 2, 4),
                "pred_logits": torch.zeros(tensor.shape[0], 2, 3),
            },
            autotune_profiles=MagicMock(
                return_value={
                    "batch_size": 4,
                    "selected_profile_index": 0,
                    "benchmarked": False,
                    "profiles": [],
                }
            ),
            warmup=MagicMock(return_value=0.01),
            consume_forward_seconds=MagicMock(return_value=0.0),
            infer_into=MagicMock(),
            allocate_output_buffers=MagicMock(),
        )

        with patch.object(acceleration, "prepare_tensorrt_engine", return_value=artifact), patch.object(
            acceleration, "TensorRTRunner", return_value=runner
        ), patch.object(acceleration, "_require_cuda"):
            handle = acceleration.configure_inference_acceleration(
                source,
                settings,
                parity_check=lambda candidate: {
                    "accepted": False,
                    "candidate_backend": candidate.backend,
                    "delta_map": -0.7,
                },
            )

        self.assertEqual(handle.backend, "pytorch")
        self.assertEqual(handle.precision, "bf16")
        self.assertEqual(handle.metadata["requested_backend"], "tensorrt")
        self.assertEqual(handle.metadata["effective_backend"], "pytorch")
        self.assertEqual(handle.metadata["accuracy_parity"]["details"]["delta_map"], -0.7)
        self.assertEqual(source.optimize_calls, [{"compile": False, "dtype": torch.bfloat16}])


if __name__ == "__main__":
    unittest.main()
