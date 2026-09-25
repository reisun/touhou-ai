"""Gymnasium adapter for infrastructure tests, not a Touhou reward design."""
import gymnasium as gym
import numpy as np

from touhou_ai.client import BridgeClient


class MockBridgeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, client=None, frames_per_step=8):
        super().__init__()
        if type(frames_per_step) is not int or not 1 <= frames_per_step <= 8:
            raise ValueError("frames_per_step must be in 1..8")
        self.client = client or BridgeClient()
        self.client.require_mock()
        self.frames_per_step = frames_per_step
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(0, 1, shape=(3,), dtype=np.float32)
        self.state = None

    def decode(self, state):
        if state.get("backend") != "mock":
            raise ValueError("unexpected bridge backend")
        frame = state.get("frame")
        if type(frame) is not int or not 0 <= frame <= 600:
            raise ValueError("invalid frame")
        if type(state.get("terminated")) is not bool or state["terminated"] != (frame == 600):
            raise ValueError("invalid termination state")
        observation = np.asarray([state["x"], state["y"], frame / 600], dtype=np.float32)
        if not self.observation_space.contains(observation):
            raise ValueError("invalid observation")
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.client.request("/reset", {})
        observation = self.decode(state)
        if state["frame"] != 0:
            raise ValueError("reset did not return frame zero")
        self.state = state
        return observation, {"frame": 0, "backend": "mock"}

    def step(self, action):
        if self.state is None or self.state["terminated"]:
            raise RuntimeError("reset required")
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        expected = min(600, self.state["frame"] + self.frames_per_step)
        state = self.client.request("/step", {
            "action": int(action), "frames": self.frames_per_step,
            "expected_frame": self.state["frame"]})
        observation = self.decode(state)
        if state["frame"] != expected:
            raise ValueError("bridge frame advance mismatch")
        self.state = state
        return observation, 0.0, state["terminated"], False, {
            "frame": state["frame"], "backend": "mock", "reward_profile": "mock_zero_test_only"}
