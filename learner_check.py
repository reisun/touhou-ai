"""Verify PPO dependencies and inference without performing training."""
import gymnasium as gym
import stable_baselines3
import torch
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
try:
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu", seed=0)
    observation, _ = env.reset(seed=0)
    action, _ = model.predict(observation, deterministic=True)
    env.step(action)
    print(f"PASS: PPO construction and inference; torch={torch.__version__}, "
          f"gymnasium={gym.__version__}, sb3={stable_baselines3.__version__}; no training")
finally:
    env.close()
