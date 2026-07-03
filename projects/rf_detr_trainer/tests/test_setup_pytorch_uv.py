"""Tests for region-aware uv/PyTorch setup helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "setup_pytorch_uv.py"


def load_setup_module():
    spec = importlib.util.spec_from_file_location("setup_pytorch_uv_for_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SetupPyTorchUvTest(unittest.TestCase):
    def test_update_dependency_line_does_not_match_torchvision_for_torch(self):
        module = load_setup_module()
        text = """[project]
dependencies = [
    "torchvision==0.26.0",
    "torch==2.10.0",
]
"""
        updated = module.update_dependency_line(text, "torch", "2.11.0")
        self.assertIn('"torchvision==0.26.0"', updated)
        self.assertEqual(updated.count('"torch==2.11.0"'), 1)
        self.assertNotIn('"torch==2.10.0"', updated)


if __name__ == "__main__":
    unittest.main()
