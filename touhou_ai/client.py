"""Bounded, authenticated bridge requests without implicit action retries."""
import json
import os
import urllib.request


class BridgeClient:
    def __init__(self, url=None, token=None, timeout=5):
        self.url = (url or os.environ.get("BRIDGE_URL", "http://127.0.0.1:18765")).rstrip("/")
        self.token = token if token is not None else os.environ["BRIDGE_TOKEN"]
        self.timeout = timeout

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data=data, headers={
            "Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError("bridge response must be an object")
        return result

    def require_mock(self):
        health = self.request("/health")
        if health.get("backend") != "mock" or health.get("protocol") != 1:
            raise ValueError("this diagnostic environment requires mock protocol 1")
        return health
