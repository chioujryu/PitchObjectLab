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


class TensorRTUtilityTest(unittest.TestCase):
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

    def test_dynamic_onnx_trace_always_uses_single_sample(self):
        calls = []

        class ExportModel:
            model = SimpleNamespace(resolution=32)

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
        runner._pending_bindings = [(incomplete, {"input": object()}), (complete, {"input": object()})]

        runner._release_completed_bindings()

        self.assertEqual(len(runner._pending_bindings), 1)
        self.assertIs(runner._pending_bindings[0][0], incomplete)

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

    def test_manifest_detects_engine_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "model.engine"
            manifest = root / "model.engine.manifest.json"
            engine.write_bytes(b"valid-engine")
            expected = {"schema_version": 1, "cache_key": "abc", "identity": {"model": "x"}}
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
        self.profile = None

    def set_memory_pool_limit(self, pool, size):
        self.workspace = (pool, size)

    def set_flag(self, flag):
        self.flag = flag

    def add_optimization_profile(self, profile):
        self.profile = profile


class _FakeBuilder:
    last = None
    platform_has_fast_fp16 = True
    platform_has_fast_bf16 = True

    def __init__(self, _logger):
        type(self).last = self
        self.network = _FakeNetwork()
        self.profile = _FakeProfile()
        self.config = _FakeBuildConfig()

    def create_network(self, _flag):
        return self.network

    def create_builder_config(self):
        return self.config

    def create_optimization_profile(self):
        return self.profile

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
                builder.profile.shapes,
                ("input", (1, 3, 32, 32), (7, 3, 32, 32), (64, 3, 32, 32)),
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


if __name__ == "__main__":
    unittest.main()
