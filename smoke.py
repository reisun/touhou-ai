"""Exercise the bridge protocol from Windows, WSL, or a container."""

import json
import os
import urllib.request


def request(path, data=None):
    base = os.environ.get("BRIDGE_URL", "http://127.0.0.1:18765")
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base + path, data=payload, headers={
        "Authorization": "Bearer " + os.environ["BRIDGE_TOKEN"],
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


if __name__ == "__main__":
    health = request("/health")
    assert health["backend"] == "mock" and health["protocol"] == 1, health
    state = request("/reset", {})
    assert state["frame"] == 0, state
    for _ in range(75):
        state = request("/step", {"action": 0, "frames": 8,
                                  "expected_frame": state["frame"]})
    assert state["frame"] == 600 and state["terminated"], state
    print("PASS: authenticated health, reset, 75 synchronized steps, episode end")
