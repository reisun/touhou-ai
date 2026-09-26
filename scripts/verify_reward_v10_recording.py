"""Final-contract replay of the bounded live recording; no game attachment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from touhou_ai.dual_grid import DualGridContract, DualGridFeatures, GridRolloutBuffer
from touhou_ai.live_rewards import LiveRewards, observed_events, WEIGHTS, VERSION
from touhou_ai.focused_policy import REWARD_KEYS
from touhou_ai.timed_bomb import TimedBombPolicy, BombClock, SPEC
from touhou_ai.live_learning import finish_buffer
from touhou_ai.directml_update import update
from touhou_ai.learning_discount import GAMMA, SPEC as DISCOUNT_CONTRACT


def main(trace, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rewards = LiveRewards(); rewards.reset('replay')
    total = {k: 0. for k in REWARD_KEYS}
    before = rows[0]['after']
    for row in rows:
        after = row['after']
        _, components = rewards.calculate('replay', observed_events(before, after))
        assert set(components) == set(REWARD_KEYS)
        for k, value in components.items():
            total[k] += value
        before = after
    assert abs(total['damage'] - 1.539) < 1e-9 and total['power_down'] == 0
    env = DualGridContract()
    model = PPO(TimedBombPolicy, env, n_steps=12, batch_size=6, n_epochs=1, seed=7, gamma=GAMMA, gae_lambda=.95,
                rollout_buffer_class=GridRolloutBuffer,
                policy_kwargs={'features_extractor_class': DualGridFeatures,
                               'net_arch': {'pi': [256, 128], 'vf': [256, 128]}})
    b = GridRolloutBuffer(12, env.observation_space, env.action_space, device='cpu', gamma=model.gamma, gae_lambda=model.gae_lambda)
    clock = BombClock()
    for i, row in enumerate(rows[:12]):
        obs = env.encode(row['after']); obs['bomb_clock'] = clock.encode()
        with torch.no_grad():
            a, v, lp = model.policy(model.policy.obs_to_tensor(obs)[0])
        b.add(obs, a.numpy(), np.array([i % 3], np.float32), np.array([i == 0]), v, lp)
        clock.advance(2)
    finish_buffer(b, 12, v, True)
    # Recomputed mixed decision/nondecision probabilities must match before update.
    tensor = {k: torch.from_numpy(v[:, 0].astype(np.float32)) for k, v in b.observations.items()}
    actions = torch.from_numpy(b.actions[:, 0]).long()
    torch.testing.assert_close(model.policy.evaluate_actions(tensor, actions)[1], torch.from_numpy(b.log_probs[:, 0]))
    model.rollout_buffer = b; model.set_logger(configure(None, []))
    candidate, detail = update(model, b, output/'isolated-synthetic-update')
    assert detail['backend'] == 'directml', detail
    loaded = PPO.load(output/'isolated-synthetic-update/after.zip', device='cpu')
    assert loaded.gamma == model.gamma == GAMMA and loaded.gae_lambda == model.gae_lambda == .95
    for actual, expected in zip(loaded.policy.evaluate_actions(tensor, actions),
                                candidate.policy.evaluate_actions(tensor, actions)):
        torch.testing.assert_close(actual, expected)
    summary = {'status': 'passed', 'live_training_updates': 0,
               'scope': 'recording replay plus isolated synthetic PPO update',
               'reward_version': VERSION, 'reward_weights': WEIGHTS,
               'reward_components': list(REWARD_KEYS), 'totals': total,
               'bomb_schedule': SPEC, 'gamma': model.gamma, 'synthetic_update': detail,
               'discount_contract': DISCOUNT_CONTRACT, 'gae_lambda': loaded.gae_lambda,
               'checkpoint_reload': True, 'source': str(trace)}
    (output/'status.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.trace, args.output)
