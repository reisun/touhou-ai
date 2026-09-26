"""Bounded scripted live checks and isolated scheduled-policy GPU update.

Scripted live actions never enter PPO. The GPU check uses a separate synthetic
rollout and checkpoint, not the user's learned model.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from touhou_ai.live_runtime import LiveRuntime
from touhou_ai.live_reset import resume_paused_episode
from touhou_ai.live_acceptance import pause
from touhou_ai.live_rewards import LiveRewards, observed_events, VERSION
from touhou_ai.dual_grid import DualGridContract, DualGridFeatures, GridRolloutBuffer
from touhou_ai.timed_bomb import TimedBombPolicy, BombClock
from touhou_ai.live_learning import finish_buffer
from touhou_ai.learning_discount import GAMMA


def run(output, pid):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = DualGridContract()
    model = PPO(TimedBombPolicy, env, n_steps=12, batch_size=6, n_epochs=1,
                seed=7, gamma=GAMMA, gae_lambda=.95, rollout_buffer_class=GridRolloutBuffer,
                policy_kwargs={'features_extractor_class': DualGridFeatures,
                               'net_arch': {'pi': [256, 128], 'vf': [256, 128]}})
    report = {'status': 'running', 'reward_version': VERSION, 'live_training_updates': 0,
              'scripted_actions': True, 'damage': {'bomb': 0, 'nonbomb': 0},
              'damage_reward': 0., 'activations': [], 'pid': pid}
    r = LiveRewards(); r.reset('acceptance')
    clock = BombClock(); observations = []
    runtime = LiveRuntime(pid)
    try:
        before = runtime.snapshot()
        if before['pause_words'][1] == 2:
            resume_paused_episode(runtime)
        before = runtime.snapshot()
        with (output/'trace.jsonl').open('x', encoding='utf-8') as stream:
            for i in range(180):
                obs = env.encode(before); obs['bomb_clock'] = clock.encode()
                if i < 12:
                    observations.append(obs)
                # First wait for any pre-existing bomb; then request one 2F pulse.
                mask = 1 | (2 if i == 48 else 0)
                after = runtime.step_gameplay(mask, 2)
                events = observed_events(before, after)
                reward, components = r.calculate('acceptance', events)
                for e in events:
                    if e['kind'] == 'damage':
                        report['damage']['bomb' if e['bomb_state'] == 1 else 'nonbomb'] += e['amount']
                report['damage_reward'] += components['damage']
                if before['bomb']['state'] == 0 and after['bomb']['state'] == 1:
                    report['activations'].append({'frame': after['stage_frame'],
                        'power_before': before['power_raw'], 'power_after': after['power_raw']})
                stream.write(json.dumps({'before_frame': before['stage_frame'], 'mask': mask,
                    'bomb_clock': clock.encode().tolist(), 'after': after,
                    'reward': reward, 'components': components, 'events': events})+'\n')
                stream.flush()
                if after.get('transition') != 'gameplay':
                    raise RuntimeError('bounded check reached terminal or transition')
                clock.advance(2); before = after
        if not report['activations'] or not all(report['damage'].values()) or report['damage_reward'] <= 0:
            raise AssertionError('need real activation plus bomb and nonbomb damage')
        report['live_status'] = 'passed'
    finally:
        try:
            report['final'] = pause(runtime)
        finally:
            runtime.close()
            (output/'status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Exercise real DirectML with this policy/observation contract, using synthetic
    # actions and rewards on saved scenes. No live collection or campaign update.
    b = GridRolloutBuffer(12, env.observation_space, env.action_space, device='cpu', gamma=model.gamma, gae_lambda=model.gae_lambda)
    for i, obs in enumerate(observations):
        with torch.no_grad():
            a, v, lp = model.policy(model.policy.obs_to_tensor(obs)[0])
        b.add(obs, a.numpy(), np.array([i % 3], np.float32), np.array([i == 0]), v, lp)
    finish_buffer(b, 12, v, True)
    model.rollout_buffer = b; model.set_logger(configure(None, []))
    from touhou_ai.directml_update import update
    _, detail = update(model, b, output/'isolated-synthetic-update')
    report['synthetic_update'] = detail
    report['status'] = 'passed' if detail['backend'] == 'directml' else 'gpu_fallback'
    (output/'status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'final'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    run(args.output, args.pid)
