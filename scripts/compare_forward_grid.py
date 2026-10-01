"""Fresh forward-grid trials with paired evaluation and observation-free controls."""
import json
import pathlib
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from touhou_ai.forward_grid import build
from touhou_ai.simulation_speed import tune_cpu
from scripts.compare_critic_tanh import Record
from scripts.assess_autumn_holdout import assess_holdout


def run(variant, seed):
    destination = pathlib.Path('artifacts/forward-grid-20261001') / variant / str(seed)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'result.json').exists():
        return
    torch.set_num_threads(1)
    model, environment, config = build(variant, seed)
    tune_cpu(model, update_threads=4)
    started = time.perf_counter()
    print('START', variant, seed, flush=True)
    model.learn(16384, callback=Record(destination / 'learning.json'))
    training_seconds = time.perf_counter() - started
    model.save(destination / 'model.zip')
    restored = type(model).load(destination / 'model.zip', device='cpu')
    assert all(torch.equal(value, restored.policy.state_dict()[key]) for key, value in model.policy.state_dict().items())
    assert all(torch.isfinite(value).all() for value in model.policy.parameters())
    evaluations = {}
    for mode in ('sample', 'greedy', 'frozen_sample', 'frozen_greedy'):
        evaluations[mode] = assess_holdout(model, environment, n=96, mode=mode)
        (destination / 'evaluation-progress.json').write_text(json.dumps(evaluations, indent=2))
        print('EVAL', variant, seed, mode, evaluations[mode]['survival'], flush=True)
    result = dict(variant=variant, seed=seed, steps=16384, config=config,
                  training_seconds=training_seconds, seconds=time.perf_counter()-started,
                  parameters=sum(value.numel() for value in model.policy.parameters()),
                  observation_elements=sum(space.shape[0] if len(space.shape)==1 else __import__('math').prod(space.shape)
                                           for space in model.observation_space.spaces.values()),
                  evaluations=evaluations, save_reload_equal=True)
    (destination / 'result.json').write_text(json.dumps(result, indent=2))
    print('DONE', variant, seed, flush=True)


if __name__ == '__main__':
    seed = int(sys.argv[1])
    for variant in ('control', 'detail', 'wide', 'both'):
        run(variant, seed)
