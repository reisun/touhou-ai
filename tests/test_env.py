import importlib.util
import threading
import unittest

from bridge import make_server

HAS_GYM = importlib.util.find_spec("gymnasium") is not None


@unittest.skipUnless(HAS_GYM, "learner dependencies run in Docker")
class EnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = "testing" * 8
        cls.server = make_server("127.0.0.1", 0, cls.token)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def setUp(self):
        from touhou_ai.client import BridgeClient
        from touhou_ai.env import MockBridgeEnv
        client = BridgeClient(f"http://127.0.0.1:{self.server.server_port}", self.token)
        self.env = MockBridgeEnv(client=client)

    def test_episode_and_reset(self):
        obs, _ = self.env.reset(seed=1)
        self.assertTrue(self.env.observation_space.contains(obs))
        for _ in range(75):
            obs, reward, done, truncated, info = self.env.step(1)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(reward, 0)
        self.assertEqual(info["frame"], 600)
        with self.assertRaises(RuntimeError):
            self.env.step(0)

    def test_outside_reset_is_rejected(self):
        self.env.reset()
        self.env.step(0)
        self.env.client.request("/reset", {})
        from urllib.error import HTTPError
        with self.assertRaises(HTTPError):
            self.env.step(0)

    def test_invalid_observation(self):
        with self.assertRaises(ValueError):
            self.env.decode({"backend": "mock", "frame": 1, "x": float("nan"),
                             "y": 0, "terminated": False})
