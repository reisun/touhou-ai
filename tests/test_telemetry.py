import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from touhou_ai.dashboard import make_server
from touhou_ai.telemetry import packet


def sample():
    return {"stage_frame": 60, "gate_tick": 100, "stage": 1, "character": 0,
            "shot": 1, "difficulty": 1, "lives_raw": 2, "power_raw": 20,
            "player": {"position": [0, 400], "status": 1},
            "bullets": [{"position": [3, 404]}], "enemies": [], "items": [], "lasers": None,
            "frame_locked": True}


class TelemetryTests(unittest.TestCase):
    def test_coordinates_provenance_and_missing_not_zero(self):
        result = packet(sample(), "episode", timestamp=100)
        self.assertEqual(result["player"]["position"], [192, 400])
        self.assertEqual(result["metrics"]["nearest_bullet_center_pixels"], 5)
        self.assertEqual(result["game"]["power"], 1)
        self.assertIsNone(result["reward"])
        self.assertIsNone(result["policy"])
        self.assertIsNone(result["entities"]["lasers"])
        self.assertIsNone(result["applied_input"])
        self.assertFalse(result["capabilities"]["live_training"])

    def test_nonfinite_position_rejected(self):
        raw = sample()
        raw["player"]["position"][0] = float("nan")
        with self.assertRaises(ValueError):
            packet(raw, "episode")

    def test_api_is_read_only_and_confined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / "artifacts/live"
            live.mkdir(parents=True)
            (live / "example.jsonl").write_text(json.dumps(sample())+'\n'+ '{"partial":')
            server = make_server(root, 0)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                def request(path, method="GET", headers=None):
                    client = http.client.HTTPConnection("127.0.0.1", server.server_port)
                    try:
                        client.request(method, path, headers=headers or {})
                        response = client.getresponse()
                        return response.status, response.read()
                    finally:
                        client.close()
                code, body = request("/api/frame?id=example&index=0")
                self.assertEqual(code, 200)
                self.assertIsNone(json.loads(body)["reward"])
                self.assertEqual(request("/api/frame?id=../.env&index=0")[0], 404)
                self.assertEqual(request("/.env")[0], 404)
                self.assertEqual(request("/api/frame?id=example&index=-1")[0], 400)
                self.assertEqual(request("/api/health", headers={"Host": "attacker.invalid"})[0], 403)
                self.assertEqual(request("/api/reset", "POST")[0], 501)
                self.assertEqual(request("/api/live")[0], 200)
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
