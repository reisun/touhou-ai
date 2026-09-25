import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import subprocess
import sys
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "learner dependencies run in Docker")
class ConfigTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "Docker SIGTERM lifecycle test")
    def test_sigterm_saves_model(self):
        from bridge import make_server
        server = make_server("127.0.0.1", 0, "signal-test" * 4)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        process = None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = {"backend": "mock", "reward_profile": "mock_zero_test_only", "seed": 1,
                          "total_timesteps": 8192, "n_steps": 32, "batch_size": 16, "n_epochs": 1,
                          "frames_per_step": 8, "checkpoint_frequency": 4, "evaluation_episodes": 1}
                config_path = root / "input.json"
                config_path.write_text(json.dumps(config))
                environment = os.environ | {"BRIDGE_URL": f"http://127.0.0.1:{server.server_port}",
                                            "BRIDGE_TOKEN": "signal-test" * 4}
                process = subprocess.Popen([sys.executable, "-m", "touhou_ai.runner", "train",
                                            "--config", str(config_path), "--output", str(root / "run")],
                                           env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                deadline = time.monotonic() + 25
                while not (root / "run/progress.json").exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail("training did not reach a checkpoint")
                    time.sleep(0.05)
                process.terminate()
                _, error = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, error.decode())
                status = json.loads((root / "run/status.json").read_text())
                self.assertEqual(status["status"], "stopped")
                self.assertTrue((root / "run/model.zip").exists())
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
            server.shutdown()
            thread.join()
            server.server_close()

    def test_invalid_real_game_and_rollout_settings(self):
        from touhou_ai.runner import read_config
        good = {"backend": "mock", "reward_profile": "mock_zero_test_only", "seed": 1,
                "total_timesteps": 64, "n_steps": 32, "batch_size": 16, "n_epochs": 1,
                "frames_per_step": 8, "checkpoint_frequency": 32, "evaluation_episodes": 2}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(good))
            self.assertEqual(read_config(path), good)
            for changes in ({"backend": "th10"}, {"total_timesteps": 33},
                            {"n_steps": 1}, {"batch_size": 64}, {"seed": True}):
                path.write_text(json.dumps(good | changes))
                with self.assertRaises(ValueError):
                    read_config(path)

    def test_stop_checkpoint_and_resume(self):
        from bridge import make_server
        from touhou_ai.client import BridgeClient
        from touhou_ai.env import MockBridgeEnv
        from touhou_ai.runner import RunCallback, train
        server = make_server("127.0.0.1", 0, "test" * 16)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        client = BridgeClient(f"http://127.0.0.1:{server.server_port}", "test" * 16)
        config = {"backend": "mock", "reward_profile": "mock_zero_test_only", "seed": 1,
                  "total_timesteps": 8, "n_steps": 8, "batch_size": 4, "n_epochs": 1,
                  "frames_per_step": 8, "checkpoint_frequency": 4, "evaluation_episodes": 1}
        original_step = RunCallback._on_step

        def stop_after_four(callback):
            if callback.n_calls == 4:
                (callback.output / "STOP").touch()
            return original_step(callback)

        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config_path = root / "input.json"
                config_path.write_text(json.dumps(config))
                factory = lambda **kwargs: MockBridgeEnv(client=client, **kwargs)
                with patch("touhou_ai.runner.MockBridgeEnv", side_effect=factory):
                    with patch.object(RunCallback, "_on_step", stop_after_four):
                        train(config_path, root / "stopped")
                    status = json.loads((root / "stopped/status.json").read_text())
                    self.assertEqual(status["status"], "stopped")
                    self.assertEqual(status["timesteps"], 4)
                    self.assertTrue((root / "stopped/checkpoint-4.zip").exists())
                    self.assertFalse((root / "stopped/evaluation.json").exists())
                    train(config_path, root / "resumed", root / "stopped/model.zip")
                    status = json.loads((root / "resumed/status.json").read_text())
                    self.assertEqual(status["status"], "completed")
                    self.assertEqual(status["timesteps"], 12)
                    self.assertTrue((root / "resumed/evaluation.json").exists())
                    with self.assertRaises(FileExistsError):
                        train(config_path, root / "resumed")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
