"""Isolated DirectML PPO updates; the live CPU policy is the rollback state."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from touhou_ai.checkpoint_rng import load_preserving_rng, capture_rng, restore_rng

ROOT = Path(__file__).resolve().parents[1]


def move_policy(model, device):
    optimizer = model.policy.optimizer
    state = optimizer.state_dict()
    model.policy.to(device)
    model.policy.optimizer = type(optimizer)(model.policy.parameters(), **optimizer.defaults)
    model.policy.optimizer.load_state_dict(state)
    model.device = device
    for values in model.policy.optimizer.state.values():
        for key, value in values.items():
            if torch.is_tensor(value):
                values[key] = value.to('cpu' if key == 'step' else device)
    assert {id(p) for g in model.policy.optimizer.param_groups for p in g['params']} == {id(p) for p in model.policy.parameters()}


def compatible_log_prob(distribution, value):
    mask = value.long().unsqueeze(-1) == torch.arange(distribution.logits.shape[-1], device=value.device)
    return (distribution.logits * mask.to(distribution.logits.dtype)).sum(-1)


def worker(directory, device=None):
    from touhou_ai.dual_grid import GridRolloutBuffer
    torch.set_num_threads(8)
    if device is None:
        import torch_directml
        device = torch_directml.device()
    torch.distributions.Categorical.log_prob = compatible_log_prob
    from touhou_ai.live_model import algorithm_class
    request = directory / 'algorithm.json'
    name = json.loads(request.read_text(encoding='utf-8'))['algorithm'] if request.exists() else 'PPO'
    model = load_preserving_rng(algorithm_class(name), directory / 'before.zip', device='cpu')
    move_policy(model, device)
    with np.load(directory / 'rollout.npz', allow_pickle=False) as data:
        n = len(data['actions'])
        buffer = GridRolloutBuffer(n, model.observation_space, model.action_space, device=device,
                                  gamma=model.gamma, gae_lambda=model.gae_lambda)
        for key in buffer.observations:
            buffer.observations[key] = data['obs_' + key].copy()
        for key in ('actions', 'values', 'log_probs', 'advantages', 'returns'):
            setattr(buffer, key, data[key].copy())
    buffer.full = True
    buffer.pos = n
    model.rollout_buffer = buffer
    model.set_logger(configure(str(directory), ['json']))
    rng_request = directory / 'rng-before.json'
    if rng_request.exists():
        restore_rng(json.loads(rng_request.read_text(encoding='utf-8')))
    else:
        # Compatibility with an already-running collector from before this fix.
        np.random.seed(model.seed)
    started = time.monotonic()
    model.train()
    rng_after = capture_rng()
    move_policy(model, torch.device('cpu'))
    seconds = time.monotonic() - started
    if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
        raise ValueError('nonfinite GPU parameters')
    metrics = {k: float(v) for k, v in model.logger.name_to_value.items() if isinstance(v, (int, float, np.number))}
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError('nonfinite GPU metrics')
    model.save(directory / 'after.zip')
    (directory / 'result.json').write_text(json.dumps({'update_seconds': seconds, 'metrics': metrics,
        'gamma': model.gamma, 'gae_lambda': model.gae_lambda,
        'buffer_gamma': buffer.gamma, 'buffer_gae_lambda': buffer.gae_lambda,
        'rng_after': rng_after}), encoding='utf-8')


def validate_update_metadata(model, candidate):
    if type(candidate) is not type(model):
        raise ValueError('GPU update changed learning algorithm')
    if candidate.num_timesteps != model.num_timesteps:
        raise ValueError('GPU update timestep mismatch')
    if candidate.n_epochs != model.n_epochs or candidate.target_kl != model.target_kl:
        raise ValueError('GPU update changed epoch/KL settings')
    completed = candidate._n_updates - model._n_updates
    if not 1 <= completed <= model.n_epochs:
        raise ValueError('GPU update epoch count outside budget')
    if model.target_kl is None and completed != model.n_epochs:
        raise ValueError('GPU update stopped without KL limit')
    return completed


def update(model, buffer, directory, timeout=180):
    """Return verified GPU candidate, or update untouched parent on CPU on failure."""
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        from touhou_ai.live_model import algorithm_class
        name = type(model).__name__
        algorithm = algorithm_class(name)
        (directory / 'algorithm.json').write_text(json.dumps({'algorithm': name}), encoding='utf-8')
        (directory / 'rng-before.json').write_text(json.dumps(capture_rng()), encoding='utf-8')
        # Close even if serialization fails (e.g. an unpicklable diagnostic mock).
        with (directory / 'before.zip').open('wb') as stream:
            model.save(stream)
        if buffer.generator_ready:
            raise ValueError('expected fresh, unflattened episode buffer')
        payload = {'obs_' + key: value for key, value in buffer.observations.items()}
        payload.update({key: getattr(buffer, key) for key in ('actions', 'values', 'log_probs', 'advantages', 'returns')})
        np.savez(directory / 'rollout.npz', **payload)
        python = ROOT / '.venv-directml/Scripts/python.exe'
        with (directory / 'worker.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(python), '-m', 'touhou_ai.directml_update', str(directory.resolve())],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        candidate = load_preserving_rng(algorithm, directory / 'after.zip', device='cpu')
        if candidate.gamma != model.gamma or candidate.gae_lambda != model.gae_lambda:
            raise ValueError('GPU update changed discount/GAE contract')
        completed_epochs = validate_update_metadata(model, candidate)
        old, new = model.policy.state_dict(), candidate.policy.state_dict()
        if not all(torch.isfinite(v).all() for v in new.values()) or all(torch.equal(v, new[k]) for k, v in old.items()):
            raise ValueError('invalid or unchanged GPU parameters')
        info = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        if 'rng_after' not in info:
            raise ValueError('GPU worker did not preserve training RNG continuity')
        candidate.set_logger(model.logger)
        for key, value in info['metrics'].items():
            candidate.logger.record(key, value)
        restore_rng(info['rng_after'])
        return candidate, {'backend': 'directml', 'completed_epochs': completed_epochs,
                           'kl_early_stopped': completed_epochs < model.n_epochs,
                           'worker_update_seconds': info['update_seconds'],
                           'total_seconds': time.monotonic() - started}
    except Exception as error:
        # Worker cannot mutate parent policy, optimizer, or rollout arrays.
        model.train()
        return model, {'backend': 'cpu_fallback', 'error': str(error),
                       'total_seconds': time.monotonic() - started}
    finally:
        # Large temporary rollout is disposable; preserve before/after checkpoints and logs.
        (directory / 'rollout.npz').unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    worker(parser.parse_args().directory)
