"""Offline PPO workload benchmark. Never controls the game or writes live models.

Real recorded scenes, synthetic actions/advantages: tests computation, not learning quality.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from touhou_ai.dual_grid import DualGridContract, GridRolloutBuffer


def move_policy(model, device):
    # PrivateUse1 transfer can replace Parameter objects; rebind the optimizer.
    old = model.policy.optimizer
    state = old.state_dict()
    model.policy.to(device)
    model.policy.optimizer = type(old)(model.policy.parameters(), **old.defaults)
    model.policy.optimizer.load_state_dict(state)
    model.device = device
    assert {id(p) for g in model.policy.optimizer.param_groups for p in g['params']} == {id(p) for p in model.policy.parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['prepare', 'cpu', 'dml', 'dml-compatible'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(71)
    np.random.seed(71)
    contract = DualGridContract()
    data_path = args.output / 'workload.npz'
    if args.device == 'prepare':
        observations = []
        with args.trace.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                observations.append(contract.encode(row['raw'], row['telemetry']['reward']['components']))
        arrays = {k: np.stack([o[k] for o in observations]).astype(
            np.float16 if k.endswith('_grid') else np.float32) for k in observations[0]}
        n = len(observations)
        arrays['actions'] = np.stack([np.random.randint(int(s), size=n) for s in contract.action_space.nvec], axis=1)
        arrays['advantages'] = np.random.default_rng(71).normal(0, .25, n).astype(np.float32)
        model = PPO.load(args.checkpoint, device='cpu')
        values, log_probs = [], []
        with torch.no_grad():
            for i in range(0, n, 64):
                obs = {k: torch.from_numpy(arrays[k][i:i+64]).float() for k in contract.observation_space.spaces}
                v, lp, _ = model.policy.evaluate_actions(obs, torch.from_numpy(arrays['actions'][i:i+64]))
                values.extend(v.flatten().numpy()); log_probs.extend(lp.numpy())
        arrays['values'] = np.asarray(values)
        arrays['log_probs'] = np.asarray(log_probs)
        arrays['returns'] = arrays['values'] + arrays['advantages']
        np.savez(data_path, **arrays)
        print(json.dumps({'prepared': n, 'parameters': sum(p.numel() for p in model.policy.parameters())}), flush=True)
        return

    report = {'backend': args.device, 'torch': torch.__version__, 'threads': 8,
              'checkpoint': str(args.checkpoint), 'rounds': [], 'status': 'running',
              'scope': 'recorded scenes; synthetic PPO targets; live learner remains running'}
    result = args.output / f'{args.device}-{torch.__version__.split("+")[0]}{args.label}.json'
    def save():
        result.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        device = torch.device('cpu')
        if args.device.startswith('dml'):
            import torch_directml
            device = torch_directml.device()
            report['gpu'] = torch_directml.device_name(0)
        if args.device == 'dml-compatible':
            # Categorical gather backward uses a scatter variant unsupported by DML.
            # The action axis is small (at most 9); a one-hot reduction is equivalent.
            def log_prob_without_gather(distribution, value):
                mask = value.long().unsqueeze(-1) == torch.arange(
                    distribution.logits.shape[-1], device=value.device)
                return (distribution.logits * mask.to(distribution.logits.dtype)).sum(-1)
            logits = torch.randn(5, 9, requires_grad=True)
            actions = torch.tensor([0, 3, 8, 1, 4])
            dist = torch.distributions.Categorical(logits=logits)
            expected = dist.log_prob(actions)
            actual = log_prob_without_gather(dist, actions)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
                                       torch.autograd.grad(expected.sum(), logits)[0])
            torch.distributions.Categorical.log_prob = log_prob_without_gather
            report['compatibility_patch'] = 'categorical log_prob gather replaced by equivalent masked sum; CPU output/gradient checked'
        save()
        model = PPO.load(args.checkpoint, device='cpu')
        move_policy(model, device)
        for state in model.policy.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and k != 'step':
                    state[k] = v.to(device)
        model.set_logger(configure(str(args.output / ('logs-' + args.device)), ['json']))
        with np.load(data_path) as data:
            n = len(data['actions'])
            buffer = GridRolloutBuffer(n, contract.observation_space, contract.action_space, device=device)
            for k in buffer.observations:
                buffer.observations[k][:, 0] = data[k]
            for k in ['actions', 'values', 'log_probs', 'advantages', 'returns']:
                getattr(buffer, k)[:, 0] = data[k]
        buffer.full = True
        buffer.pos = n
        model.rollout_buffer = buffer
        report.update(samples=n, batch_size=model.batch_size, epochs=model.n_epochs,
                      parameters=sum(p.numel() for p in model.policy.parameters()))
        probe_indices = np.linspace(0, n - 1, 8, dtype=int)
        probe = {k: torch.tensor(v[probe_indices, 0], device=device).float() for k, v in buffer.observations.items()}
        with torch.no_grad():
            model.policy.predict_values(probe).cpu()  # synchronize and initialize backend
        if args.device.startswith('dml'):
            reference = PPO.load(args.checkpoint, device='cpu')
            cpu_probe = {k: v.cpu() for k, v in probe.items()}
            actions = torch.tensor(buffer.actions[probe_indices, 0])
            cpu_outputs = reference.policy.evaluate_actions(cpu_probe, actions)
            gpu_outputs = model.policy.evaluate_actions(probe, actions.to(device))
            for a, b in zip(cpu_outputs, gpu_outputs):
                torch.testing.assert_close(a, b.cpu(), rtol=1e-3, atol=1e-4)
            sum(x.mean() for x in cpu_outputs).backward()
            sum(x.mean() for x in gpu_outputs).backward()
            gradient_error = 0.
            for a, b in zip(reference.policy.parameters(), model.policy.parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad.cpu(), rtol=2e-2, atol=2e-4)
                    gradient_error = max(gradient_error, float((a.grad - b.grad.cpu()).abs().max()))
            report['initial_cpu_gpu_max_gradient_error'] = gradient_error
            model.policy.optimizer.zero_grad()
            del reference, cpu_outputs, gpu_outputs
        print(json.dumps({'backend': args.device, 'ready': True, 'samples': n}), flush=True)
        original = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        for iteration in range(args.rounds):
            np.random.seed(100 + iteration)
            started = time.perf_counter()
            model.train()
            state = {k: v.detach().cpu() for k, v in model.policy.state_dict().items()}
            seconds = time.perf_counter() - started
            assert all(torch.isfinite(v).all() for v in state.values()), 'nonfinite parameters'
            assert any(not torch.equal(state[k], original[k]) for k in state), 'parameters unchanged'
            assert all(p.grad is None or torch.isfinite(p.grad.detach().cpu()).all() for p in model.policy.parameters()), 'nonfinite gradients'
            original = {k: v.clone() for k, v in state.items()}
            metrics = {k: float(v) for k, v in model.logger.name_to_value.items() if isinstance(v, (int, float, np.number))}
            assert all(np.isfinite(v) for v in metrics.values()), 'nonfinite metrics'
            entry = {'iteration': iteration + 1, 'seconds': seconds, 'metrics': metrics}
            report['rounds'].append(entry)
            save(); print(json.dumps(entry), flush=True)
        with torch.no_grad():
            expected = model.policy.predict_values(probe).cpu()
        # Portable save and CPU reload: production inference remains on the CPU.
        move_policy(model, torch.device('cpu'))
        for state in model.policy.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()
        model.device = torch.device('cpu')
        path = args.output / ('checkpoint-' + args.device + args.label + '.zip')
        model.save(path)
        reloaded = PPO.load(path, device='cpu')
        assert all(torch.equal(v, reloaded.policy.state_dict()[k]) for k, v in model.policy.state_dict().items())
        with torch.no_grad():
            actual = reloaded.policy.predict_values({k: v.cpu() for k, v in probe.items()})
        report['cpu_reload_max_value_error'] = float((expected - actual).abs().max())
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
        move_policy(reloaded, device)
        for state in reloaded.policy.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and k != 'step':
                    state[k] = v.to(device)
        reloaded.rollout_buffer = buffer
        reloaded.set_logger(model.logger)
        reloaded.n_epochs = 1
        reloaded.train()
        assert all(torch.isfinite(p.detach().cpu()).all() for p in reloaded.policy.parameters())
        report.update(status='passed', save_reload_resume=True)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        print(report['error'], flush=True)
    finally:
        save()
        print(str(result), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
