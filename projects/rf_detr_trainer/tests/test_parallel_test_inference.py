from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.object_detection_dataset_evaluator import object_detection_dataset_evaluator as evaluator  # noqa: E402


def make_dataset(image_count: int) -> evaluator.DatasetBundle:
    images = [
        evaluator.ImageRecord(
            image_id=index,
            file_name=f"image_{index}.jpg",
            path=f"/unused/image_{index}.jpg",
            width=100,
            height=80,
        )
        for index in range(1, image_count + 1)
    ]
    return evaluator.DatasetBundle(
        images=images,
        categories=[{"id": 1, "name": "football"}],
        annotations=[],
        coco={"images": [], "annotations": [], "categories": []},
        source_kind="test",
    )


def inference_config(*, devices=None, chunks=None):
    model = {
        "type": "ultralytics",
        "device": "cpu",
        "confidence_threshold": 0.25,
        "extra_predict_args": {},
    }
    if devices is not None:
        model["devices"] = devices
    inference = {"mode": "full_image"}
    if chunks is not None:
        inference["chunks"] = chunks
    return {
        "model": model,
        "inference": inference,
        "progress": {"images": False},
    }


def successful_worker_result(chunk_id, records, device):
    predictions = [
        {
            "image_id": int(record["image_id"]),
            "category_id": 1,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "score": 0.5,
        }
        for record in records
    ]
    stats = [
        {
            "image_id": int(record["image_id"]),
            "elapsed_seconds": 0.01,
            "batch_size": 2,
        }
        for record in records
    ]
    return {
        "predictions": predictions,
        "stats": stats,
        "visuals": [],
        "chunk_id": chunk_id,
        "device": device,
        "image_count": len(records),
        "prediction_count": len(predictions),
        "model_load_seconds": 0.1 + chunk_id,
        "inference_seconds": 0.2 + chunk_id,
        "effective_batch_sizes": [2],
        "status": "success",
        "error": None,
    }


class RecordingExecutor:
    instances = []
    result_builder = staticmethod(successful_worker_result)

    def __init__(self, *, max_workers, mp_context):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.submissions = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def submit(self, function, chunk_id, records, config, output_dir, device, visual_image_ids):
        self.submissions.append(
            {
                "function": function,
                "chunk_id": chunk_id,
                "records": records,
                "config": config,
                "output_dir": output_dir,
                "device": device,
                "visual_image_ids": visual_image_ids,
            }
        )
        future = Future()
        result = type(self).result_builder(chunk_id, records, device)
        if isinstance(result, BaseException):
            future.set_exception(result)
        else:
            future.set_result(result)
        return future


class ParallelInferencePlanningTest(unittest.TestCase):
    def setUp(self):
        RecordingExecutor.instances.clear()
        RecordingExecutor.result_builder = staticmethod(successful_worker_result)

    def test_chunk_count_defaults_to_one_or_device_count_and_explicit_is_authoritative(self):
        self.assertEqual(evaluator.resolve_inference_chunk_count(inference_config(), image_count=8), 1)
        self.assertEqual(
            evaluator.resolve_inference_chunk_count(
                inference_config(devices=["cuda:0", "cuda:1"]),
                image_count=8,
            ),
            2,
        )
        self.assertEqual(
            evaluator.resolve_inference_chunk_count(
                inference_config(devices=["cuda:0", "cuda:1"], chunks=6),
                image_count=8,
            ),
            6,
        )

    def test_chunk_devices_expand_round_robin_for_one_and_two_devices(self):
        self.assertEqual(evaluator.expand_chunk_devices(["cuda:0"], 6), ["cuda:0"] * 6)
        self.assertEqual(
            evaluator.expand_chunk_devices(["cuda:0", "cuda:1"], 6),
            ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0", "cuda:1"],
        )

    def test_invalid_or_excessive_chunks_fail_before_worker_creation(self):
        for value in (0, -1, False, "not-an-integer"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "inference.chunks.*positive integer"):
                    evaluator.resolve_inference_chunk_count(inference_config(chunks=value), image_count=3)

        with mock.patch.object(evaluator, "ProcessPoolExecutor") as executor:
            with self.assertRaisesRegex(ValueError, "inference.chunks.*3.*2"):
                evaluator.run_inference(
                    make_dataset(2),
                    inference_config(chunks=3),
                    Path("/unused"),
                    quiet=True,
                )
            executor.assert_not_called()

    def test_multi_chunk_rejects_prebuilt_model_before_worker_creation(self):
        prebuilt = object()
        with mock.patch.object(evaluator, "ProcessPoolExecutor") as executor:
            with mock.patch.object(evaluator, "build_inference_model") as build_model:
                with self.assertRaisesRegex(ValueError, "prebuilt_model.*multi-chunk"):
                    evaluator.run_inference(
                        make_dataset(2),
                        inference_config(chunks=2),
                        Path("/unused"),
                        quiet=True,
                        prebuilt_model=prebuilt,
                    )
                executor.assert_not_called()
                build_model.assert_not_called()

    def test_spawn_executor_uses_exact_chunk_count_and_round_robin_assignments(self):
        config = inference_config(devices=["cuda:0", "cuda:1"], chunks=6)

        def completed_after_all_submissions(futures):
            self.assertEqual(len(RecordingExecutor.instances[0].submissions), 6)
            return reversed(list(futures))

        with mock.patch.object(evaluator, "ProcessPoolExecutor", RecordingExecutor):
            with mock.patch.object(evaluator, "as_completed", completed_after_all_submissions):
                predictions, stats, visuals, summary = evaluator.run_inference(
                    make_dataset(6), config, Path("/unused"), quiet=True
                )

        executor = RecordingExecutor.instances[0]
        self.assertEqual(executor.max_workers, 6)
        self.assertEqual(executor.mp_context.get_start_method(), "spawn")
        self.assertEqual(
            [submission["device"] for submission in executor.submissions],
            ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0", "cuda:1"],
        )
        self.assertTrue(all(submission["function"] is evaluator.inference_worker for submission in executor.submissions))
        self.assertEqual([prediction["image_id"] for prediction in predictions], [1, 2, 3, 4, 5, 6])
        self.assertEqual([row["image_id"] for row in stats], [1, 2, 3, 4, 5, 6])
        self.assertEqual(visuals, [])
        self.assertEqual(summary["chunks"], 6)
        self.assertEqual(summary["devices_requested"], ["cuda:0", "cuda:1"])
        self.assertEqual([row["chunk_id"] for row in summary["assignments"]], list(range(6)))
        self.assertEqual([row["status"] for row in summary["assignments"]], ["success"] * 6)

    def test_visuals_follow_chunk_order_when_futures_complete_in_reverse(self):
        def result_builder(chunk_id, records, device):
            result = successful_worker_result(chunk_id, records, device)
            result["visuals"] = [
                {"chunk_id": chunk_id, "visual_index": 0},
                {"chunk_id": chunk_id, "visual_index": 1},
            ]
            return result

        RecordingExecutor.result_builder = staticmethod(result_builder)

        with mock.patch.object(evaluator, "ProcessPoolExecutor", RecordingExecutor):
            with mock.patch.object(
                evaluator,
                "as_completed",
                side_effect=lambda futures: reversed(list(futures)),
            ):
                _, _, visuals, _ = evaluator.run_inference(
                    make_dataset(3),
                    inference_config(chunks=3),
                    Path("/unused"),
                    quiet=True,
                )

        self.assertEqual(
            visuals,
            [
                {"chunk_id": 0, "visual_index": 0},
                {"chunk_id": 0, "visual_index": 1},
                {"chunk_id": 1, "visual_index": 0},
                {"chunk_id": 1, "visual_index": 1},
                {"chunk_id": 2, "visual_index": 0},
                {"chunk_id": 2, "visual_index": 1},
            ],
        )


class ModelFactoryTest(unittest.TestCase):
    def setUp(self):
        self.module_name = "_evaluator_parallel_factory_test_module"
        self.module = types.ModuleType(self.module_name)
        sys.modules[self.module_name] = self.module
        self.addCleanup(sys.modules.pop, self.module_name, None)

    def test_dotted_factory_is_resolved_and_called_before_builtin_builders(self):
        sentinel = object()
        calls = []

        def make_model(model_cfg, device):
            calls.append((model_cfg, device))
            return sentinel

        self.module.make_model = make_model
        config = inference_config()
        config["model"]["factory"] = f"{self.module_name}.make_model"

        with mock.patch.object(evaluator, "build_direct_model") as builtin:
            result = evaluator.build_inference_model(config, "cpu")

        self.assertIs(result, sentinel)
        self.assertEqual(calls, [(config["model"], "cpu")])
        builtin.assert_not_called()

    def test_absent_or_none_factory_uses_builtin_builder(self):
        sentinel = object()
        configs = [inference_config(), inference_config()]
        configs[1]["model"]["factory"] = None

        with mock.patch.object(evaluator, "build_direct_model", return_value=sentinel) as builtin:
            for config in configs:
                with self.subTest(factory=config["model"].get("factory", "absent")):
                    self.assertIs(evaluator.build_inference_model(config, "cpu"), sentinel)

        self.assertEqual(builtin.call_count, 2)

    def test_malformed_factory_spec_and_import_failure_are_clear(self):
        with mock.patch.object(evaluator, "build_direct_model", return_value=object()) as builtin:
            for spec in ("", "factory_without_module", 123):
                with self.subTest(spec=spec):
                    config = inference_config()
                    config["model"]["factory"] = spec
                    with self.assertRaisesRegex(ValueError, "model.factory.*dotted"):
                        evaluator.build_inference_model(config, "cpu")

            config = inference_config()
            config["model"]["factory"] = "package_that_does_not_exist.make_model"
            with self.assertRaisesRegex(ImportError, "model.factory.*package_that_does_not_exist"):
                evaluator.build_inference_model(config, "cpu")

        builtin.assert_not_called()

    def test_relative_factory_spec_never_leaks_raw_importlib_errors(self):
        config = inference_config()
        config["model"]["factory"] = ".relative.factory"

        with mock.patch.object(evaluator, "build_direct_model", return_value=object()) as builtin:
            with self.assertRaises((ValueError, ImportError)) as raised:
                evaluator.build_inference_model(config, "cpu")

        self.assertIn("model.factory", str(raised.exception))
        builtin.assert_not_called()

    def test_resolved_factory_must_be_callable(self):
        self.module.not_callable = 42
        config = inference_config()
        config["model"]["factory"] = f"{self.module_name}.not_callable"

        with mock.patch.object(evaluator, "build_direct_model", return_value=object()) as builtin:
            with self.assertRaisesRegex(TypeError, "model.factory.*callable"):
                evaluator.build_inference_model(config, "cpu")

        builtin.assert_not_called()


class InferenceWorkerMetadataTest(unittest.TestCase):
    def test_worker_returns_timed_chunk_metadata(self):
        config = inference_config()
        config["model"]["type"] = "rfdetr"
        records = [record.__dict__ for record in make_dataset(3).images]
        predictions = [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "score": 0.75,
            },
            {
                "image_id": 2,
                "category_id": 1,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "score": 0.5,
            },
        ]
        stats = [
            {"image_id": 1, "elapsed_seconds": 0.01, "batch_size": 4},
            {"image_id": 2, "elapsed_seconds": 0.02, "batch_size": 2},
            {"image_id": 3, "elapsed_seconds": 0.03, "batch_size": 4},
        ]
        inference_model = object()

        with mock.patch.object(evaluator, "validate_runtime_devices") as validate_devices:
            with mock.patch.object(evaluator, "build_inference_model", return_value=inference_model) as build_model:
                with mock.patch.object(
                    evaluator,
                    "run_batched_rfdetr_records",
                    return_value=(predictions, stats, []),
                ) as run_records:
                    with mock.patch.object(
                        evaluator.time,
                        "perf_counter",
                        side_effect=[10.0, 10.25, 20.0, 21.5],
                    ):
                        try:
                            result = evaluator.inference_worker(
                                7,
                                records,
                                config,
                                "/unused",
                                "cuda:1",
                                [],
                            )
                        except TypeError as error:
                            self.fail(f"inference_worker must accept chunk_id first: {error}")

        validated_config = validate_devices.call_args.args[0]
        self.assertEqual(validated_config["model"]["device"], "cuda:1")
        self.assertEqual(validated_config["model"]["devices"], [])
        build_model.assert_called_once_with(config, "cuda:1")
        run_records.assert_called_once()
        self.assertEqual(result["predictions"], predictions)
        self.assertEqual(result["stats"], stats)
        self.assertEqual(result["visuals"], [])
        self.assertEqual(
            {key: result[key] for key in (
                "chunk_id",
                "device",
                "image_count",
                "prediction_count",
                "model_load_seconds",
                "inference_seconds",
                "effective_batch_sizes",
                "status",
                "error",
            )},
            {
                "chunk_id": 7,
                "device": "cuda:1",
                "image_count": 3,
                "prediction_count": 2,
                "model_load_seconds": 0.25,
                "inference_seconds": 1.5,
                "effective_batch_sizes": [2, 4],
                "status": "success",
                "error": None,
            },
        )

    def test_parent_device_validation_can_be_deferred_to_spawn_workers(self):
        class StopAfterValidation(Exception):
            pass

        config = {
            "runtime": {
                "verbose": False,
                "quiet": True,
                "validate_devices_in_parent": False,
            },
            "inference": {"mode": "full_image"},
            "test_mode": {"mode": "full_image"},
            "model": {"type": "rfdetr", "device": "cuda:0"},
            "output": {"resolved_dir": "/unused"},
        }
        with mock.patch.object(evaluator, "validate_runtime_devices") as validate_devices:
            with mock.patch.object(evaluator, "load_dataset", side_effect=StopAfterValidation):
                with self.assertRaises(StopAfterValidation):
                    evaluator.run_evaluation(
                        config,
                        Path("/unused/config.yaml"),
                        already_normalized=True,
                    )
        validate_devices.assert_not_called()


class ParallelInferenceFailureTest(unittest.TestCase):
    def setUp(self):
        RecordingExecutor.instances.clear()
        RecordingExecutor.result_builder = staticmethod(successful_worker_result)

    def test_parallel_failure_contains_deterministic_terminal_summary(self):
        def result_builder(chunk_id, records, device):
            if chunk_id == 1:
                return RuntimeError("worker exploded")
            return successful_worker_result(chunk_id, records, device)

        RecordingExecutor.result_builder = staticmethod(result_builder)
        config = inference_config(devices=["cpu"], chunks=2)

        with mock.patch.object(evaluator, "ProcessPoolExecutor", RecordingExecutor):
            with self.assertRaises(evaluator.ParallelInferenceError) as raised:
                evaluator.run_inference(make_dataset(2), config, Path("/unused"), quiet=True)

        error = raised.exception
        self.assertIn("chunk_id=1", str(error))
        self.assertIn("device=cpu", str(error))
        self.assertEqual(error.summary["status"], "failed")
        self.assertEqual(error.summary["chunks"], 2)
        self.assertEqual(error.summary["devices_requested"], ["cpu"])
        self.assertEqual([row["chunk_id"] for row in error.summary["assignments"]], [0, 1])
        self.assertEqual([row["status"] for row in error.summary["assignments"]], ["success", "failed"])
        self.assertEqual(
            error.summary["assignments"][1]["error"],
            {"type": "RuntimeError", "message": "worker exploded"},
        )

    def test_executor_construction_or_entry_failure_covers_every_planned_chunk(self):
        class EntryFailingExecutor:
            def __init__(self, *, max_workers, mp_context):
                self.max_workers = max_workers
                self.mp_context = mp_context

            def __enter__(self):
                raise RuntimeError("executor entry failed")

            def __exit__(self, exc_type, exc, traceback):
                return False

        cases = (
            ("construction", mock.Mock(side_effect=RuntimeError("executor construction failed"))),
            ("entry", EntryFailingExecutor),
        )
        for label, executor_factory in cases:
            with self.subTest(label=label):
                with mock.patch.object(evaluator, "ProcessPoolExecutor", executor_factory):
                    with self.assertRaises(evaluator.ParallelInferenceError) as raised:
                        evaluator.run_inference(
                            make_dataset(3),
                            inference_config(chunks=3),
                            Path("/unused"),
                            quiet=True,
                        )

                assignments = raised.exception.summary["assignments"]
                self.assertEqual([row["chunk_id"] for row in assignments], [0, 1, 2])
                self.assertEqual([row["device"] for row in assignments], ["cpu"] * 3)
                self.assertEqual([row["image_count"] for row in assignments], [1, 1, 1])
                self.assertEqual([row["status"] for row in assignments], ["failed"] * 3)
                self.assertTrue(all(row["error"]["type"] == "RuntimeError" for row in assignments))
                self.assertIn("chunk_id=0", str(raised.exception))
                self.assertIn("chunk_id=1", str(raised.exception))
                self.assertIn("chunk_id=2", str(raised.exception))

    def test_submit_failure_drains_submitted_future_and_fails_remaining_plan(self):
        class SubmitFailingExecutor(RecordingExecutor):
            instances = []
            submitted_future = None

            def submit(self, function, chunk_id, records, config, output_dir, device, visual_image_ids):
                if chunk_id == 1:
                    raise RuntimeError("submit exploded")
                future = super().submit(
                    function,
                    chunk_id,
                    records,
                    config,
                    output_dir,
                    device,
                    visual_image_ids,
                )
                future.result = mock.Mock(wraps=future.result)
                type(self).submitted_future = future
                return future

        with mock.patch.object(evaluator, "ProcessPoolExecutor", SubmitFailingExecutor):
            with self.assertRaises(evaluator.ParallelInferenceError) as raised:
                evaluator.run_inference(
                    make_dataset(3),
                    inference_config(chunks=3),
                    Path("/unused"),
                    quiet=True,
                )

        assignments = raised.exception.summary["assignments"]
        self.assertEqual([row["chunk_id"] for row in assignments], [0, 1, 2])
        self.assertEqual([row["status"] for row in assignments], ["success", "failed", "failed"])
        self.assertEqual([row["prediction_count"] for row in assignments], [1, 0, 0])
        self.assertTrue(all("submit exploded" in row["error"]["message"] for row in assignments[1:]))
        self.assertEqual(SubmitFailingExecutor.submitted_future.result.call_count, 1)

    def test_malformed_worker_metadata_is_rejected_against_submitted_plan(self):
        mutations = (
            ("status", "unknown", "status"),
            ("chunk_id", 99, "chunk_id"),
            ("device", "cuda:99", "device"),
            ("image_count", 99, "image_count"),
            ("prediction_count", 99, "prediction_count"),
        )
        for field, value, expected_message in mutations:
            with self.subTest(field=field):
                RecordingExecutor.instances.clear()

                def result_builder(chunk_id, records, device, field=field, value=value):
                    result = successful_worker_result(chunk_id, records, device)
                    if chunk_id == 1:
                        result[field] = value
                    return result

                RecordingExecutor.result_builder = staticmethod(result_builder)
                with mock.patch.object(evaluator, "ProcessPoolExecutor", RecordingExecutor):
                    with self.assertRaises(evaluator.ParallelInferenceError) as raised:
                        evaluator.run_inference(
                            make_dataset(2),
                            inference_config(chunks=2),
                            Path("/unused"),
                            quiet=True,
                        )

                assignments = raised.exception.summary["assignments"]
                self.assertEqual([row["chunk_id"] for row in assignments], [0, 1])
                self.assertEqual([row["status"] for row in assignments], ["success", "failed"])
                self.assertEqual(assignments[1]["device"], "cpu")
                self.assertEqual(assignments[1]["image_count"], 1)
                self.assertEqual(assignments[1]["prediction_count"], 0)
                self.assertEqual(assignments[1]["error"]["type"], "ValueError")
                self.assertIn(expected_message, assignments[1]["error"]["message"])

    def test_malformed_nested_worker_rows_become_failed_assignments(self):
        cases = (
            ("prediction missing image_id", "predictions", "image_id", None, True),
            ("prediction invalid image_id", "predictions", "image_id", "bad", False),
            ("prediction missing category_id", "predictions", "category_id", None, True),
            ("prediction invalid category_id", "predictions", "category_id", "bad", False),
            ("prediction missing score", "predictions", "score", None, True),
            ("prediction invalid score", "predictions", "score", "bad", False),
            ("stat missing image_id", "stats", "image_id", None, True),
            ("stat invalid image_id", "stats", "image_id", "bad", False),
            ("visual row is not a mapping", "visuals", None, 42, False),
        )
        for label, collection, field, value, missing in cases:
            with self.subTest(label=label):
                RecordingExecutor.instances.clear()

                def result_builder(
                    chunk_id,
                    records,
                    device,
                    collection=collection,
                    field=field,
                    value=value,
                    missing=missing,
                ):
                    result = successful_worker_result(chunk_id, records, device)
                    if chunk_id != 1:
                        return result
                    if collection == "visuals":
                        result[collection] = [value]
                    elif missing:
                        result[collection][0].pop(field)
                    else:
                        result[collection][0][field] = value
                    return result

                RecordingExecutor.result_builder = staticmethod(result_builder)
                with mock.patch.object(evaluator, "ProcessPoolExecutor", RecordingExecutor):
                    with self.assertRaises(evaluator.ParallelInferenceError) as raised:
                        evaluator.run_inference(
                            make_dataset(2),
                            inference_config(chunks=2),
                            Path("/unused"),
                            quiet=True,
                        )

                assignments = raised.exception.summary["assignments"]
                self.assertEqual([row["chunk_id"] for row in assignments], [0, 1])
                self.assertEqual([row["status"] for row in assignments], ["success", "failed"])
                self.assertEqual(assignments[1]["prediction_count"], 0)
                self.assertEqual(assignments[1]["error"]["type"], "ValueError")
                self.assertIn(collection, assignments[1]["error"]["message"])

    def test_nonfinite_timing_and_invalid_effective_batch_members_are_rejected(self):
        cases = (
            ("model_load_seconds nan", "model_load_seconds", float("nan")),
            ("model_load_seconds infinity", "model_load_seconds", float("inf")),
            ("inference_seconds nan", "inference_seconds", float("nan")),
            ("inference_seconds infinity", "inference_seconds", float("inf")),
            ("batch bool", "effective_batch_sizes", [True]),
            ("batch zero", "effective_batch_sizes", [0]),
            ("batch negative", "effective_batch_sizes", [-1]),
            ("batch non-integer", "effective_batch_sizes", [2.0]),
        )
        for label, field, value in cases:
            with self.subTest(label=label):
                RecordingExecutor.instances.clear()

                def result_builder(chunk_id, records, device, field=field, value=value):
                    result = successful_worker_result(chunk_id, records, device)
                    if chunk_id == 1:
                        result[field] = value
                    return result

                RecordingExecutor.result_builder = staticmethod(result_builder)
                with mock.patch.object(evaluator, "ProcessPoolExecutor", RecordingExecutor):
                    with self.assertRaises(evaluator.ParallelInferenceError) as raised:
                        evaluator.run_inference(
                            make_dataset(2),
                            inference_config(chunks=2),
                            Path("/unused"),
                            quiet=True,
                        )

                assignments = raised.exception.summary["assignments"]
                self.assertEqual([row["status"] for row in assignments], ["success", "failed"])
                self.assertEqual(assignments[1]["error"]["type"], "ValueError")
                self.assertIn(field, assignments[1]["error"]["message"])

    def test_effective_batch_sizes_are_sorted_and_deduplicated(self):
        def result_builder(chunk_id, records, device):
            result = successful_worker_result(chunk_id, records, device)
            result["effective_batch_sizes"] = [4, 2, 4]
            return result

        RecordingExecutor.result_builder = staticmethod(result_builder)
        with mock.patch.object(evaluator, "ProcessPoolExecutor", RecordingExecutor):
            _, _, _, summary = evaluator.run_inference(
                make_dataset(2),
                inference_config(chunks=2),
                Path("/unused"),
                quiet=True,
            )

        self.assertEqual(
            [row["effective_batch_sizes"] for row in summary["assignments"]],
            [[2, 4], [2, 4]],
        )

    def test_failed_worker_payload_is_not_aggregated(self):
        def result_builder(chunk_id, records, device):
            result = successful_worker_result(chunk_id, records, device)
            if chunk_id == 1:
                result["status"] = "failed"
                result["error"] = {"type": "RuntimeError", "message": "reported failure"}
            return result

        RecordingExecutor.result_builder = staticmethod(result_builder)
        with mock.patch.object(evaluator, "ProcessPoolExecutor", RecordingExecutor):
            with self.assertRaises(evaluator.ParallelInferenceError) as raised:
                evaluator.run_inference(
                    make_dataset(2),
                    inference_config(chunks=2),
                    Path("/unused"),
                    quiet=True,
                )

        assignments = raised.exception.summary["assignments"]
        self.assertEqual([row["chunk_id"] for row in assignments], [0, 1])
        self.assertEqual(assignments[1]["status"], "failed")
        self.assertEqual(assignments[1]["prediction_count"], 0)
        self.assertEqual(assignments[1]["error"], {"type": "RuntimeError", "message": "reported failure"})

    def test_run_evaluation_writes_failed_parallel_summary_before_reraising(self):
        class StubParallelInferenceError(RuntimeError):
            def __init__(self, summary):
                super().__init__("parallel inference failed")
                self.summary = summary

        failed_summary = {
            "status": "failed",
            "chunks": 2,
            "devices_requested": ["cpu"],
            "wall_seconds": 0.5,
            "assignments": [
                {
                    "chunk_id": 0,
                    "device": "cpu",
                    "image_count": 1,
                    "prediction_count": 0,
                    "model_load_seconds": 0.0,
                    "inference_seconds": 0.0,
                    "effective_batch_sizes": [],
                    "status": "failed",
                    "error": {"type": "RuntimeError", "message": "worker exploded"},
                },
                {
                    "chunk_id": 1,
                    "device": "cpu",
                    "image_count": 1,
                    "prediction_count": 0,
                    "model_load_seconds": 0.0,
                    "inference_seconds": 0.0,
                    "effective_batch_sizes": [],
                    "status": "failed",
                    "error": {"type": "RuntimeError", "message": "worker also exploded"},
                },
            ],
        }
        dataset = make_dataset(2)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = inference_config(chunks=2)
            config.update(
                {
                    "runtime": {"verbose": False, "quiet": True},
                    "output": {
                        "resolved_dir": str(output_dir),
                        "save_config": False,
                        "save_ground_truth_json": False,
                        "save_predictions_json": True,
                        "save_model_input_batches": False,
                        "save_metrics": True,
                        "save_output_manifest": True,
                    },
                    "evaluation": {},
                }
            )
            render_visuals = mock.Mock()
            render_dataset_cases = mock.Mock()
            render_error_cases = mock.Mock()
            write_metrics = mock.Mock()

            with mock.patch.object(evaluator, "ParallelInferenceError", StubParallelInferenceError, create=True):
                with mock.patch.object(evaluator, "validate_runtime_devices"):
                    with mock.patch.object(evaluator, "load_dataset", return_value=dataset):
                        with mock.patch.object(evaluator, "estimate_resources", return_value={}):
                            with mock.patch.object(evaluator, "confirm_or_exit"):
                                with mock.patch.object(
                                    evaluator,
                                    "run_inference",
                                    side_effect=StubParallelInferenceError(failed_summary),
                                ):
                                    with mock.patch.object(evaluator, "render_visual_outputs", render_visuals):
                                        with mock.patch.object(evaluator, "render_dataset_case_outputs", render_dataset_cases):
                                            with mock.patch.object(evaluator, "render_error_case_outputs", render_error_cases):
                                                with mock.patch.object(evaluator, "write_metrics_tables", write_metrics):
                                                    with self.assertRaises(StubParallelInferenceError):
                                                        evaluator.run_evaluation(
                                                            config,
                                                            Path("/unused/config.yaml"),
                                                            already_normalized=True,
                                                            print_summary=False,
                                                        )

            summary_path = output_dir / "parallel_summary.json"
            self.assertTrue(summary_path.is_file())
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), failed_summary)
            self.assertFalse((output_dir / "predictions_coco.json").exists())
            self.assertFalse((output_dir / "metrics_summary.json").exists())
            self.assertFalse((output_dir / "_tmp_ground_truth_coco.json").exists())
            output_manifest = json.loads((output_dir / "output_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(
                any(Path(row["path"]).name == "parallel_summary.json" for row in output_manifest["outputs"])
            )
            render_visuals.assert_not_called()
            render_dataset_cases.assert_not_called()
            render_error_cases.assert_not_called()
            write_metrics.assert_not_called()

    def test_run_evaluation_adds_successful_parallel_summary_to_output_manifest(self):
        parallel_summary = {
            "status": "success",
            "chunks": 2,
            "devices_requested": ["cpu"],
            "wall_seconds": 0.5,
            "assignments": [],
        }
        operating = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "per_class": [],
            "per_image": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = inference_config(chunks=2)
            config.update(
                {
                    "runtime": {"verbose": False, "quiet": True},
                    "output": {
                        "resolved_dir": str(output_dir),
                        "save_config": False,
                        "save_ground_truth_json": False,
                        "save_predictions_json": False,
                        "save_model_input_batches": False,
                        "save_metrics": False,
                        "save_plots": False,
                        "save_output_manifest": True,
                    },
                    "evaluation": {
                        "save_coco_summary_text": False,
                        "curves": False,
                        "classwise": False,
                        "per_image_metrics": False,
                        "confusion_matrix": False,
                    },
                }
            )

            with mock.patch.object(evaluator, "validate_runtime_devices"):
                with mock.patch.object(evaluator, "load_dataset", return_value=make_dataset(2)):
                    with mock.patch.object(evaluator, "estimate_resources", return_value={}):
                        with mock.patch.object(evaluator, "confirm_or_exit"):
                            with mock.patch.object(
                                evaluator,
                                "run_inference",
                                return_value=([], [], [], parallel_summary),
                            ):
                                with mock.patch.object(evaluator, "render_dataset_case_outputs", return_value=[]):
                                    with mock.patch.object(evaluator, "render_visual_outputs", return_value=[]):
                                        with mock.patch.object(evaluator, "render_error_case_outputs", return_value=[]):
                                            with mock.patch.object(evaluator, "capture_coco_eval", return_value=(object(), "")):
                                                with mock.patch.object(
                                                    evaluator,
                                                    "match_predictions_at_threshold",
                                                    return_value=operating,
                                                ):
                                                    with mock.patch.object(
                                                        evaluator,
                                                        "coco_metrics_dict",
                                                        return_value={"mAP50": 0.0, "mAP50-95": 0.0},
                                                    ):
                                                        evaluator.run_evaluation(
                                                            config,
                                                            Path("/unused/config.yaml"),
                                                            already_normalized=True,
                                                            print_summary=False,
                                                        )

            summary_path = output_dir / "parallel_summary.json"
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), parallel_summary)
            output_manifest = json.loads((output_dir / "output_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(
                any(Path(row["path"]).name == "parallel_summary.json" for row in output_manifest["outputs"])
            )


if __name__ == "__main__":
    unittest.main()
