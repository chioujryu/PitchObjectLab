from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_rf_detr_model as trainer  # noqa: E402


class TrainingResumeTest(unittest.TestCase):
    def test_resume_cli_override_updates_train_config(self):
        args = trainer.build_parser().parse_args(["--resume", "checkpoint_2.ckpt"])
        config = {"train": {}}

        trainer.apply_cli_overrides(config, args)

        self.assertEqual(config["train"]["resume"], "checkpoint_2.ckpt")

    def test_resume_checkpoint_resolves_relative_to_source_config(self):
        with tempfile.TemporaryDirectory() as temp:
            config_dir = Path(temp)
            source_config = config_dir / "train.yaml"
            checkpoint = config_dir / "checkpoint_2.ckpt"
            source_config.touch()
            checkpoint.touch()
            config = {"train": {"resume": checkpoint.name}}

            resolved = trainer.resolve_train_resume_checkpoint(config, source_config)

            self.assertEqual(resolved, checkpoint.resolve())
            self.assertEqual(config["train"]["resume"], str(checkpoint.resolve()))

    def test_resume_checkpoint_rejects_inference_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            config_dir = Path(temp)
            source_config = config_dir / "train.yaml"
            checkpoint = config_dir / "checkpoint_best_ema.pth"
            source_config.touch()
            checkpoint.touch()
            config = {"train": {"resume": checkpoint.name}}

            with self.assertRaisesRegex(ValueError, r"Lightning \.ckpt"):
                trainer.resolve_train_resume_checkpoint(config, source_config)

    def test_resume_checkpoint_must_exist(self):
        with tempfile.TemporaryDirectory() as temp:
            source_config = Path(temp) / "train.yaml"
            source_config.touch()
            config = {"train": {"resume": "missing.ckpt"}}

            with self.assertRaisesRegex(FileNotFoundError, "train.resume"):
                trainer.resolve_train_resume_checkpoint(config, source_config)


if __name__ == "__main__":
    unittest.main()
