"""CPU thread-budget tests for RF-DETR entrypoints and presets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import rf_detr_cpu_runtime as cpu_runtime
from rf_detr_config import load_yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "config"


class CpuPolicyTests(unittest.TestCase):
    def tearDown(self) -> None:
        cpu_runtime.reset_for_tests()

    def test_balanced_budget_on_32_logical_cpus(self):
        policy = cpu_runtime.resolve_cpu_policy(
            {"runtime": {"cpu": {"enabled": True, "budget_percent": 50}}},
            "inference",
            CONFIG_DIR / "rf_detr_inference.yaml",
            logical_cpus=32,
            env={},
        )

        self.assertEqual(policy.total_thread_budget, 16)
        self.assertEqual(policy.model_processes, 1)
        self.assertEqual(policy.threads_per_process, 16)

    def test_inference_auto_caps_large_hosts_at_sixteen_threads(self):
        policy = cpu_runtime.resolve_cpu_policy(
            {"runtime": {"cpu": {"enabled": True, "budget_percent": 50}}},
            "inference",
            CONFIG_DIR / "rf_detr_inference.yaml",
            logical_cpus=128,
            env={},
        )

        self.assertEqual(policy.total_thread_budget, 64)
        self.assertEqual(policy.threads_per_process, 16)
        self.assertTrue(policy.auto_tune_intraop)

    def test_train_auto_keeps_full_per_process_budget(self):
        policy = cpu_runtime.resolve_cpu_policy(
            {"runtime": {"cpu": {"enabled": True, "budget_percent": 50}}},
            "train",
            CONFIG_DIR / "rf_detr_train.yaml",
            logical_cpus=128,
            env={},
        )

        self.assertEqual(policy.threads_per_process, 64)

    def test_explicit_intraop_thread_override(self):
        policy = cpu_runtime.resolve_cpu_policy(
            {
                "runtime": {
                    "cpu": {
                        "enabled": True,
                        "budget_percent": 50,
                        "torch_intraop_threads": 8,
                    }
                }
            },
            "test",
            CONFIG_DIR / "rf_detr_test.yaml",
            logical_cpus=128,
            env={},
        )

        self.assertEqual(policy.threads_per_process, 8)
        self.assertFalse(policy.auto_tune_intraop)

    def test_auto_intraop_benchmark_selects_and_records_fastest_candidate(self):
        policy = cpu_runtime.CpuRuntimePolicy(
            task="inference",
            enabled=True,
            budget_percent=50,
            logical_cpus=128,
            total_thread_budget=64,
            model_processes=1,
            threads_per_process=16,
            source_config="test.yaml",
            auto_tune_intraop=True,
        )
        state = {"threads": 24, "interop": 24}
        fake_torch = types.SimpleNamespace(
            set_num_threads=lambda value: state.__setitem__("threads", value),
            set_num_interop_threads=lambda value: state.__setitem__("interop", value),
            get_num_threads=lambda: state["threads"],
            get_num_interop_threads=lambda: state["interop"],
        )
        fake_cv2 = types.SimpleNamespace(setNumThreads=lambda _value: None, getNumThreads=lambda: 1)
        rows = [
            {"threads": 8, "median_seconds": 0.01, "samples": [0.01]},
            {"threads": 16, "median_seconds": 0.02, "samples": [0.02]},
        ]

        with patch.dict(sys.modules, {"torch": fake_torch, "cv2": fake_cv2}), patch.object(
            cpu_runtime,
            "benchmark_torch_intraop_candidates",
            side_effect=lambda module, candidates: (module.set_num_threads(8) or 8, rows),
        ):
            summary = cpu_runtime.apply_loaded_runtime(policy)

        self.assertEqual(summary["torch_intraop_threads"], 8)
        self.assertEqual(summary["torch_intraop_auto_benchmark"]["selected_threads"], 8)
        self.assertEqual([row["threads"] for row in rows], [8, 16])

    def test_test_chunks_share_the_total_budget(self):
        policy = cpu_runtime.resolve_cpu_policy(
            {
                "runtime": {"cpu": {"enabled": True, "budget_percent": 50}},
                "test": {
                    "parallel": {
                        "chunks": 6,
                        "allow_same_gpu_oversubscription": True,
                    }
                },
            },
            "test",
            CONFIG_DIR / "rf_detr_test.yaml",
            logical_cpus=32,
            env={},
        )

        self.assertEqual(policy.total_thread_budget, 16)
        self.assertEqual(policy.model_processes, 6)
        self.assertEqual(policy.threads_per_process, 2)

    def test_ddp_world_size_wins_over_explicit_devices(self):
        policy = cpu_runtime.resolve_cpu_policy(
            {
                "runtime": {"cpu": {"enabled": True, "budget_percent": 50}},
                "train": {"device": "0,1"},
            },
            "train",
            CONFIG_DIR / "rf_detr_train.yaml",
            logical_cpus=32,
            env={"WORLD_SIZE": "4"},
        )

        self.assertEqual(policy.model_processes, 4)
        self.assertEqual(policy.threads_per_process, 4)

    def test_cpu_only_execution_is_limited(self):
        policy = cpu_runtime.resolve_cpu_policy(
            {
                "runtime": {"cpu": {"enabled": True, "budget_percent": 25}},
                "model": {"device": "cpu"},
            },
            "inference",
            CONFIG_DIR / "rf_detr_inference.yaml",
            logical_cpus=32,
            env={},
        )

        self.assertTrue(policy.enabled)
        self.assertEqual(policy.total_thread_budget, 8)
        self.assertEqual(policy.threads_per_process, 8)

    def test_disabled_limit_preserves_existing_environment(self):
        config_path = CONFIG_DIR / "rf_detr_inference.yaml"
        existing = {
            name: str(index + 3)
            for index, name in enumerate(cpu_runtime.THREAD_ENVIRONMENT_VARIABLES)
        }
        with patch.dict(os.environ, existing, clear=False):
            policy = cpu_runtime.bootstrap_from_argv(
                config_path,
                "inference",
                ["--config", str(config_path), "--no-cpu-limit"],
            )
            self.assertFalse(policy.enabled)
            self.assertEqual(
                {name: os.environ.get(name) for name in existing},
                existing,
            )

    def test_cli_overrides_extended_yaml(self):
        child = CONFIG_DIR / "rf_detr_inference_smoke_temporal_tracknet_v5.yaml"
        with patch.object(cpu_runtime.os, "cpu_count", return_value=32), patch.dict(
            os.environ, {}, clear=False
        ):
            policy = cpu_runtime.bootstrap_from_argv(
                child,
                "inference",
                [
                    "--config",
                    str(child),
                    "--cpu-limit",
                    "--cpu-budget-percent",
                    "75",
                ],
            )

        self.assertTrue(policy.enabled)
        self.assertEqual(policy.budget_percent, 75)
        self.assertEqual(policy.total_thread_budget, 24)
        self.assertEqual(policy.threads_per_process, 16)

    def test_topology_cli_is_applied_before_numerical_imports(self):
        config_path = CONFIG_DIR / "rf_detr_test.yaml"
        with patch.object(cpu_runtime.os, "cpu_count", return_value=32), patch.dict(
            os.environ, {}, clear=False
        ):
            policy = cpu_runtime.bootstrap_from_argv(
                config_path,
                "test",
                ["--config", str(config_path), "--chunks", "6"],
            )
            environment = {
                name: os.environ.get(name)
                for name in cpu_runtime.THREAD_ENVIRONMENT_VARIABLES
            }

        self.assertEqual(policy.model_processes, 1)
        self.assertEqual(policy.threads_per_process, 16)
        self.assertEqual(set(environment.values()), {"16"})

    def test_train_device_cli_is_applied_before_numerical_imports(self):
        config_path = CONFIG_DIR / "rf_detr_train.yaml"
        with patch.object(cpu_runtime.os, "cpu_count", return_value=32), patch.dict(
            os.environ, {}, clear=False
        ):
            policy = cpu_runtime.bootstrap_from_argv(
                config_path,
                "train",
                ["--config", str(config_path), "--device", "0,1"],
            )

        self.assertEqual(policy.model_processes, 2)
        self.assertEqual(policy.threads_per_process, 8)

    def test_invalid_percentages_fail(self):
        for value in (0, 101, float("nan"), float("inf"), True, "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cpu_runtime.validate_budget_percent(value)

    def test_model_process_count_cannot_exceed_budget(self):
        with self.assertRaisesRegex(ValueError, "process count exceeds the CPU thread budget"):
            cpu_runtime.resolve_cpu_policy(
                {
                    "runtime": {"cpu": {"enabled": True, "budget_percent": 50}},
                    "test": {
                        "parallel": {
                            "chunks": 17,
                            "allow_same_gpu_oversubscription": True,
                        }
                    },
                },
                "test",
                CONFIG_DIR / "rf_detr_test.yaml",
                logical_cpus=32,
                env={},
            )

    def test_runtime_apis_are_applied_once(self):
        calls = {"torch": 0, "interop": 0, "opencv": 0}
        state = {"torch": 24, "interop": 24, "opencv": 32}
        fake_torch = types.SimpleNamespace(
            set_num_threads=lambda value: (
                calls.__setitem__("torch", calls["torch"] + 1),
                state.__setitem__("torch", value),
            ),
            set_num_interop_threads=lambda value: (
                calls.__setitem__("interop", calls["interop"] + 1),
                state.__setitem__("interop", value),
            ),
            get_num_threads=lambda: state["torch"],
            get_num_interop_threads=lambda: state["interop"],
        )
        fake_cv2 = types.SimpleNamespace(
            setNumThreads=lambda value: (
                calls.__setitem__("opencv", calls["opencv"] + 1),
                state.__setitem__("opencv", value),
            ),
            getNumThreads=lambda: state["opencv"],
        )
        policy = cpu_runtime.CpuRuntimePolicy(
            task="inference",
            enabled=True,
            budget_percent=50,
            logical_cpus=32,
            total_thread_budget=16,
            model_processes=1,
            threads_per_process=16,
            source_config="test.yaml",
        )

        with patch.dict(sys.modules, {"torch": fake_torch, "cv2": fake_cv2}):
            first = cpu_runtime.apply_loaded_runtime(policy)
            second = cpu_runtime.apply_loaded_runtime(policy)

        self.assertEqual(calls, {"torch": 1, "interop": 1, "opencv": 1})
        self.assertEqual(first, second)
        self.assertEqual(first["torch_intraop_threads"], 16)
        self.assertEqual(first["torch_interop_threads"], 1)
        self.assertEqual(first["opencv_threads"], 1)

    def test_cpu_cli_flags_validate_and_override(self):
        parser = argparse.ArgumentParser()
        cpu_runtime.add_cpu_cli_arguments(parser)
        args = parser.parse_args(
            ["--no-cpu-limit", "--cpu-budget-percent", "80", "--torch-intraop-threads", "8"]
        )
        config = {"runtime": {"cpu": {"enabled": True, "budget_percent": 50}}}

        cpu_runtime.apply_cpu_cli_overrides(config, args)

        self.assertEqual(
            config["runtime"]["cpu"],
            {"enabled": False, "budget_percent": 80.0, "torch_intraop_threads": 8},
        )


class CpuPresetTests(unittest.TestCase):
    def test_all_execution_presets_resolve_the_cpu_block(self):
        paths = sorted(CONFIG_DIR.glob("*.yaml"))
        standalone = []
        extended = []
        for path in paths:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            (extended if "extends" in raw else standalone).append(path)
            merged = load_yaml(path)
            cpu = merged.get("runtime", {}).get("cpu", {})
            self.assertTrue(cpu.get("enabled"), path.name)
            self.assertEqual(cpu.get("budget_percent"), 50, path.name)
            cpu_runtime.parse_torch_intraop_threads(cpu.get("torch_intraop_threads", "auto"))

        self.assertGreaterEqual(len(paths), 57)
        self.assertTrue(standalone)
        self.assertTrue(extended)
        for path in standalone:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            cpu = raw.get("runtime", {}).get("cpu", {})
            self.assertTrue(cpu.get("enabled"), path.name)
            self.assertEqual(cpu.get("budget_percent"), 50, path.name)

    def test_explicit_zero_num_workers_survives_test_config_translation(self):
        import test_rf_detr_model as test_runner

        internal = test_runner.build_internal_test_config(
            {"model": {}, "dataset": {}, "test": {"num_workers": 0}}
        )

        self.assertEqual(internal["train"]["num_workers"], 0)


class CpuSpawnTests(unittest.TestCase):
    def test_spawned_child_reapplies_actual_thread_limits(self):
        probe = PROJECT_DIR / "tests" / "cpu_spawn_probe.py"
        config_path = CONFIG_DIR / "rf_detr_train.yaml"
        result = subprocess.run(
            [sys.executable, str(probe), str(config_path)],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )

        summary = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(summary["model_processes"], 6)
        self.assertEqual(summary["threads_per_process"], 2)
        self.assertEqual(summary["torch_intraop_threads"], 2)
        self.assertEqual(summary["torch_interop_threads"], 1)
        self.assertEqual(summary["opencv_threads"], 1)
        self.assertTrue(summary["native_threadpools"])
        for pool in summary["native_threadpools"]:
            threads = pool.get("num_threads")
            if isinstance(threads, int) and threads > 0:
                self.assertLessEqual(threads, 2, pool)
        self.assertEqual(
            set(summary["thread_environment"].values()),
            {"2"},
        )
if __name__ == "__main__":
    unittest.main()
