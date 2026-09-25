import json
import threading
import unittest
import urllib.error
import urllib.request

from bridge import MockGame, make_server


class GameTests(unittest.TestCase):
    def test_stale_frame_does_not_advance(self):
        game = MockGame()
        game.step(1, 8, 0)
        with self.assertRaises(ValueError):
            game.step(1, 8, 0)
        self.assertEqual(game.frame, 8)

    def test_bounds_and_terminal(self):
        game = MockGame()
        while game.frame < 600:
            game.step(1, 8, game.frame)
        self.assertEqual(game.x, 0)
        with self.assertRaises(ValueError):
            game.step(0, 1, 600)
        self.assertEqual(game.reset()["frame"], 0)

    def test_invalid_input_does_not_mutate(self):
        game = MockGame()
        for action, frames in [(True, 1), (5, 1), (0, 9), (0, -1)]:
            with self.assertRaises(ValueError):
                game.step(action, frames, 0)
        self.assertEqual(game.frame, 0)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = "test" * 16
        cls.server = make_server("127.0.0.1", 0, cls.token)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def test_auth_required(self):
        with self.assertRaises(urllib.error.HTTPError) as result:
            urllib.request.urlopen(self.url + "/health")
        self.assertEqual(result.exception.code, 401)

    def test_malformed_request(self):
        req = urllib.request.Request(self.url + "/step", data=b"[]", headers={
            "Authorization": "Bearer " + self.token})
        with self.assertRaises(urllib.error.HTTPError) as result:
            urllib.request.urlopen(req)
        self.assertEqual(result.exception.code, 400)

    def test_health(self):
        req = urllib.request.Request(self.url + "/health", headers={
            "Authorization": "Bearer " + self.token})
        with urllib.request.urlopen(req) as response:
            self.assertEqual(json.load(response)["backend"], "mock")


if __name__ == "__main__":
    unittest.main()
