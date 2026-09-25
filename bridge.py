"""Local, authenticated mock game bridge. No real game input is sent."""

import argparse
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockGame:
    def __init__(self):
        self.reset()

    def observation(self):
        return {"frame": self.frame, "x": self.x, "y": self.y,
                "terminated": self.frame >= 600, "backend": "mock"}

    def reset(self):
        self.frame, self.x, self.y = 0, 0.5, 0.8
        return self.observation()

    def step(self, action, frames, expected_frame):
        if type(action) is not int or not 0 <= action <= 4:
            raise ValueError("action must be an integer in 0..4")
        if type(frames) is not int or not 1 <= frames <= 8:
            raise ValueError("frames must be an integer in 1..8")
        if type(expected_frame) is not int or expected_frame != self.frame:
            raise ValueError("expected_frame does not match current frame")
        if self.frame >= 600:
            raise ValueError("episode ended; reset required")
        elapsed = min(frames, 600 - self.frame)
        dx, dy = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]
        self.x = max(0.0, min(1.0, self.x + dx * elapsed * 0.01))
        self.y = max(0.0, min(1.0, self.y + dy * elapsed * 0.01))
        self.frame += elapsed
        return self.observation()


def make_server(host, port, token):
    if len(token) < 32:
        raise ValueError("BRIDGE_TOKEN must contain at least 32 characters")
    game, lock = MockGame(), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)

        def respond(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            received = self.headers.get("Authorization", "").encode()
            if not hmac.compare_digest(received, ("Bearer " + token).encode()):
                self.respond(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/health":
                self.respond(200, {"status": "ok", "backend": "mock", "protocol": 1})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path == "/shutdown":
                self.respond(200, {"status": "stopping"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path not in ("/reset", "/step"):
                self.respond(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("body must be 1..4096 bytes")
                self.connection.settimeout(5)
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("body must be an object")
                with lock:
                    if self.path == "/reset":
                        result = game.reset()
                    else:
                        result = game.step(data.get("action"), data.get("frames", 1),
                                           data.get("expected_frame"))
                self.respond(200, result)
            except (ValueError, TimeoutError) as error:
                self.respond(400, {"error": str(error)})

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    server = make_server(args.host, args.port, os.environ.get("BRIDGE_TOKEN", ""))
    print(f"Mock bridge listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
