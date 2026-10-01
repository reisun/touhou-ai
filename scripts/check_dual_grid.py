"""Offline, synthetic dual-grid preview and timing. Never opens the game."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from stable_baselines3 import PPO
from touhou_ai.dual_grid import (DualGridContract, DualGridFeatures, GridRolloutBuffer,
                                 CONTRACT, LOCAL_CHANNELS, GLOBAL_CHANNELS)


def scene():
    bullets = []
    for radius in (60, 110, 165):
        for angle in np.linspace(-np.pi, 0, 29):
            if -.65 < angle < -.4:
                continue
            bullets.append({'position': [float(radius*np.cos(angle)), float(330+radius*np.sin(angle))],
                'velocity_raw': [float(2*np.cos(angle)), float(-2*np.sin(angle))],
                'hitbox_raw': [6, 6], 'flags_raw': 2, 'status': 1})
    return {'stage': 1, 'lives_raw': 2, 'power_raw': 60,
        'player': {'position': [0, 330], 'velocity_raw': [0, 0], 'hitbox_raw': [1, 1],
                   'status': 1, 'invincibility_raw': 0, 'focus_raw': 1},
        'bullets': bullets,
        'enemies': [{'position': [x, 80], 'velocity_raw': [0, 1], 'hp': 50, 'hp_max': 100,
                     'is_boss': x == 0} for x in (-96, 0, 96)],
        'items': [{'position': [x, 200], 'velocity_raw': [0, 1], 'type': 1 if x < 0 else 4} for x in (-60, -20, 20, 60)],
        'lasers': [], 'player_shots': [], 'bomb': {'state': 0}, 'spell': None}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = DualGridContract(); raw = scene(); observation = env.encode(raw)
    model = PPO('MultiInputPolicy', env, n_steps=4, batch_size=2, seed=7,
        rollout_buffer_class=GridRolloutBuffer,
        policy_kwargs={'features_extractor_class': DualGridFeatures,
                       'net_arch': {'pi': [256, 128], 'vf': [256, 128]}})
    for _ in range(5):
        model.predict(observation, deterministic=True)
    timings = []
    for _ in range(30):
        start = time.perf_counter(); model.predict(observation, deterministic=True)
        timings.append(1000*(time.perf_counter()-start))
    rng = np.random.default_rng(7)
    stress = scene()
    stress['bullets'] = [{'position': [float(x), float(y)], 'velocity_raw': [2, -1],
                         'hitbox_raw': [4, 4], 'flags_raw': 2, 'status': 1}
                        for x, y in zip(rng.uniform(-192, 192, 2000), rng.uniform(0, 448, 2000))]
    encode_ms = []
    for _ in range(15):
        start = time.perf_counter(); env.encode(stress); encode_ms.append(1000*(time.perf_counter()-start))
    report = {'contract': CONTRACT, 'source': 'synthetic_only', 'game_control': False,
        'observation_shapes': {k: list(v.shape) for k, v in observation.items()},
        'local_channels': LOCAL_CHANNELS, 'global_channels': GLOBAL_CHANNELS,
        'input_values': sum(v.size for v in observation.values()),
        'parameters': sum(p.numel() for p in model.policy.parameters()),
        'model_predict_ms_median': float(np.median(timings)),
        'model_predict_ms_p95': float(np.percentile(timings, 95)),
        'encode_2000_bullets_ms_median': float(np.median(encode_ms)),
        'encode_2000_bullets_ms_p95': float(np.percentile(encode_ms, 95)),
        'grid_storage_bytes_per_step': sum(v.size*2 for k, v in observation.items() if k.endswith('_grid')),
        'note': 'Does not measure frame-gate/logging or learning improvement; no live hit/miss validation.'}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    whole, local = observation['global_grid'], observation['local_grid']
    rgb = np.zeros((56, 48, 3)); rgb[:, :, 0] = np.clip(whole[3]*3, 0, 1)
    rgb[:, :, 1] = np.clip(whole[6]*3+whole[7]+whole[11]*5, 0, 1)
    rgb[:, :, 2] = np.clip(whole[0]*3, 0, 1)
    axes[0].imshow(rgb, extent=(-192, 192, 448, 0), interpolation='nearest')
    axes[0].add_patch(Rectangle((-96, 234), 192, 192, fill=False, edgecolor='white'))
    axes[0].set_title('Global: 8 px cells\nBlue bullets / red enemies / green items & player')
    axes[0].set_xlabel('Game x (px)'); axes[0].set_ylabel('Game y (px)')
    rgb = np.zeros((96, 96, 3));rgb[:, :, 2] = local[1];rgb[:, :, 1] = local[0]
    axes[1].imshow(rgb, extent=(-96, 96, 96, -96), interpolation='nearest')
    axes[1].set_title('Local: 192 x 192 px, 2 px cells\nBlue bullet coverage / green player coverage')
    axes[1].set_xlabel('Relative x (px)'); axes[1].set_ylabel('Relative y (px)')
    plot=axes[2].imshow(local[3], extent=(-96, 96, 96, -96), interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title('Local: bullet coverage offset by 4 frames\nSingle instant, no history')
    axes[2].set_xlabel('Relative x (px)');fig.colorbar(plot, ax=axes[2], shrink=.6, label='coverage')
    fig.suptitle('Synthetic observation preview (not gameplay or a trained policy)', fontsize=13)
    fig.savefig(output/'preview.png', dpi=150);plt.close(fig)
    print(json.dumps(report))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
