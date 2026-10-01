"""Aggregate paired trials and benchmark all encoders without training contention."""
import json
import pathlib
import statistics
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from touhou_ai.forward_grid import ForwardEncoder, ForwardEnv, VARIANTS, build
from touhou_ai.simulation_speed import distributions

ROOT = pathlib.Path('artifacts/forward-grid-20261001')


def benchmark():
    torch.set_num_threads(1)
    environment = ForwardEnv()
    scenes = []
    for seed in (5100, 5110, 5120, 5130):
        environment.reset(seed=seed)
        for frame in range(200):
            if frame % 10 == 0:
                scenes.append(environment.raw_observation())
            _, _, done, _, _ = environment.step([0, 0, 0, 0])
            if done:
                break
    timings = {variant: dict(encode_ms=[], actor_ms=[]) for variant in VARIANTS}
    objects = {}
    for variant in VARIANTS:
        model, _, _ = build(variant, 7)
        model.policy.set_training_mode(False)
        encoder = ForwardEncoder(variant)
        observations = [encoder.encode(scene) for scene in scenes]
        objects[variant] = model, encoder, observations
        for observation in observations[:5]:
            with torch.inference_mode():
                distributions(model.policy, observation)
    for repetition in range(5):
        for variant in VARIANTS[repetition % 4:] + VARIANTS[:repetition % 4]:
            model, encoder, observations = objects[variant]
            started = time.perf_counter()
            for scene in scenes:
                encoder.encode(scene)
            timings[variant]['encode_ms'].append((time.perf_counter()-started)*1000/len(scenes))
            started = time.perf_counter()
            with torch.inference_mode():
                for observation in observations:
                    distributions(model.policy, observation)
            timings[variant]['actor_ms'].append((time.perf_counter()-started)*1000/len(scenes))
    return dict(scenes=len(scenes), torch_threads=1, repeats=5,
                note='Same recorded scenes; batch 1; training trials finished, real game may still compete for CPU',
                variants={variant: {key: dict(median=statistics.median(values), repetitions=values)
                                    for key, values in row.items()} for variant, row in timings.items()})


def summarize():
    output = {}
    reference = json.loads((ROOT/'control'/'7'/'result.json').read_text())['config']
    for variant in VARIANTS:
        rows = [json.loads((ROOT/variant/str(seed)/'result.json').read_text()) for seed in (7, 17, 27)]
        assert all(row['config'] == reference and row['steps'] == 16384 and row['save_reload_equal'] for row in rows)
        summary = dict(parameters=rows[0]['parameters'], observation_elements=rows[0]['observation_elements'],
                       training_seconds=[row['training_seconds'] for row in rows])
        for mode in ('sample', 'greedy', 'frozen_sample', 'frozen_greedy'):
            values = [row['evaluations'][mode]['survival'] for row in rows]
            summary[mode] = dict(mean=float(np.mean(values)), seeds=values,
                                 mean_seconds=float(np.mean([row['evaluations'][mode]['mean_frames']/60 for row in rows])))
        matched = []
        for row in rows:
            actual = {episode['seed']: episode for episode in row['evaluations']['greedy']['episodes']}
            frozen = row['evaluations']['frozen_greedy']['episodes']
            matched.append(sum((episode['success'], episode['frames'], episode['end_xy']) ==
                               (actual[episode['seed']]['success'], actual[episode['seed']]['frames'], actual[episode['seed']]['end_xy'])
                               for episode in frozen))
        summary['greedy_frozen_matching_outcomes_per_seed'] = matched
        output[variant] = summary
    output['benchmark'] = benchmark()
    (ROOT/'summary.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    summarize()
