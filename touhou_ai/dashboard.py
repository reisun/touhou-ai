"""Read-only localhost dashboard. No game control or arbitrary filesystem API."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

from touhou_ai.report import summarize
from touhou_ai.telemetry import packet
from touhou_ai.obs_stats import ObsStats
from touhou_ai.telemetry_memory import read as read_telemetry

ROOT = Path(__file__).resolve().parents[1]


class Recordings:
    def __init__(self, root):
        self.root = root.resolve()
        self.lock = threading.Lock()
        self.cache = {}
        self.obs = ObsStats(self.root)

    def paths(self):
        return {p.stem: p for p in self.root.glob("live/*.jsonl")
                if p.is_file() and p.resolve().is_relative_to(self.root)}

    def load(self, name):
        path = self.paths().get(name)
        if path is None:
            raise KeyError("recording not found")
        with self.lock:
            signature = (path.stat().st_mtime_ns, path.stat().st_size)
            if path.stat().st_size > 128*1024*1024:
                raise ValueError("recording exceeds dashboard limit")
            cached = self.cache.get(name)
            if cached is None or cached[0] != signature:
                rows = []
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            continue  # A running recorder may have an incomplete final line.
                        if len(rows) > 20000:
                            raise ValueError("recording exceeds frame limit")
                if len(self.cache) >= 4:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[name] = (signature, rows)
            return self.cache[name][1]

    def catalog(self):
        records = []
        for name, path in sorted(self.paths().items(), reverse=True):
            try:
                status_path = path.with_suffix(".status.json")
                status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
                rows = self.load(name)
                if not rows:
                    continue
                counts = {key: max(len(r.get(key) or []) for r in rows)
                          for key in ("bullets", "enemies", "items", "lasers")}
                records.append({"id": name, "frames": len(rows), "stage_frame": rows[-1].get("stage_frame"),
                                "status": status.get("status", "unverified"),
                                "terminated": status.get("terminated"), "counts_max": counts,
                                "training_steps": 0})
            except (ValueError, OSError, TypeError, KeyError):
                continue
        return records


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, store, root, **kwargs):
        self.store, self.root = store, root
        super().__init__(*args, **kwargs)

    def respond(self, value, code=200, content_type="application/json; charset=utf-8"):
        body = json.dumps(value, allow_nan=False).encode() if isinstance(value, (dict, list)) else value
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
            return self.respond({"error": "invalid host"}, 403)
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/api/live-stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                previous = None
                heartbeat = time.monotonic()
                try:
                    while True:
                        try:
                            update = read_telemetry(self.store.root, previous)
                            if update is not None:
                                signature, payload = update
                                self.wfile.write(b"data: " + payload + b"\n\n")
                                self.wfile.flush()
                                previous = signature
                        except (OSError, ValueError):
                            pass
                        if time.monotonic() - heartbeat >= 1:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                            heartbeat = time.monotonic()
                        time.sleep(1/60)
                except (ConnectionError, OSError):
                    return
            if url.path == "/api/health":
                return self.respond({"service": "touhou-observer", "schema_version": 1, "read_only": True})
            if url.path == '/api/obs':
                return self.respond(self.store.obs.snapshot())
            if url.path == "/api/catalog":
                return self.respond({"recordings": self.store.catalog(), "training_runs": summarize(self.store.root)})
            if url.path == "/api/profile":
                return self.respond(json.loads((self.root / "configs/sharu-inspired-v1.json").read_text()))
            if url.path == "/api/live-learning":
                results = []
                for path in sorted(self.store.root.glob("live-learning-*/status.json"), reverse=True):
                    if path.resolve().is_relative_to(self.store.root):
                        results.append(json.loads(path.read_text(encoding="utf-8")) | {"run": path.parent.name})
                return self.respond({"runs": results})
            if url.path == "/api/frame":
                name = query.get("id", [""])[0]
                rows = self.store.load(name)
                index = int(query.get("index", ["0"])[0])
                if not 0 <= index < len(rows):
                    raise ValueError("frame index out of range")
                return self.respond(packet(rows[index], name) | {"index": index, "total": len(rows)})
            if url.path == "/api/series":
                rows = self.store.load(query.get("id", [""])[0])
                return self.respond({"points": [{"index": i, "frame": row.get("stage_frame"),
                    "bullets": None if row.get("bullets") is None else len(row["bullets"]),
                    "lives": row.get("lives_raw")} for i, row in enumerate(rows)]})
            if url.path == "/api/live":
                update = read_telemetry(self.store.root)
                return self.respond(update[1] if update else {"available": False})
            if url.path == "/api/policy-diagnostic":
                paths = sorted(self.store.root.glob("policy-*/policy-telemetry.json"), reverse=True)
                return self.respond(json.loads(paths[0].read_text()) if paths else {"available": False})
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/obs.js": ("obs.js", "text/javascript; charset=utf-8"),
                      "/obs-status.js": ("obs-status.js", "text/javascript; charset=utf-8"),
                      "/obs.css": ("obs.css", "text/css; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/app.css": ("app.css", "text/css; charset=utf-8"),
                      "/icons.js": ("icons.js", "text/javascript; charset=utf-8")}
            if url.path not in static:
                return self.respond({"error": "not found"}, 404)
            filename, mime = static[url.path]
            if url.path == '/' and query.get('mode', [''])[0] in ('obs1', 'obs2'):
                filename = 'obs.html'
            return self.respond((self.root / "dashboard" / filename).read_bytes(), content_type=mime)
        except KeyError:
            self.respond({"error": "recording not found"}, 404)
        except (ValueError, IndexError, TypeError):
            self.respond({"error": "invalid data or request"}, 400)
        except OSError:
            self.respond({"error": "data unavailable"}, 503)


def make_server(root=ROOT, port=18767):
    store = Recordings(root / "artifacts")
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, store=store, root=root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18767)
    args = parser.parse_args()
    server = make_server(port=args.port)
    print(f"Touhou observer: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
