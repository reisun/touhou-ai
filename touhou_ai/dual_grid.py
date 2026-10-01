"""Two spatial scales, one instant: no nearest-entity truncation or history.

Pinned TH10 binary geometry is documented in docs/dual-grid-design.md.
Coverage is a spatial feature, not a promise of exact pixel collision testing.
"""
import math
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from touhou_ai.focused_policy import reward_input
from touhou_ai.power_items import power_item_raw, POWER_GRID_SCALE, POWER_GRID_MAX, MAX_ITEMS

CONTRACT = 'th10-dual-grid-v6'
LOCAL_CHANNELS = ('player_coverage', 'bullet_coverage', 'bullet_coverage_2f', 'bullet_coverage_4f',
                  'laser_coverage', 'laser_field_validated')
GLOBAL_CHANNELS = ('bullet_density', 'bullet_density_2f', 'bullet_density_4f', 'enemy_density',
                   'enemy_hp', 'enemy_hp_max', 'item_density', 'player',
                   'laser_coverage', 'laser_field_validated', 'player_shot_coverage', 'power_amount')
SPEC = {'version': 6, 'shape': 'dual_grid', 'local_pixels': 192, 'local_cell_pixels': 2,
        'global_pixels': [384, 448], 'global_cell_pixels': 8, 'history': 0,
        'individual_bullets': 0, 'individual_items': 0, 'near_bullets': 0,
        'local_channels': list(LOCAL_CHANNELS), 'global_channels': list(GLOBAL_CHANNELS),
        'geometry': 'pinned_binary_aabb_v1', 'bullet_offsets_frames': [0, 2, 4],
        'motion_encoding': 'per_bullet_observed_velocity_offsets',
        'grid_storage': 'float16_roundtrip_before_policy', 'all_entities': True,
        'player_shots': 'th10-player-shot-aabb-v1',
        'power_items': 'pinned_types_1_4_10_11', 'power_amount_raw_scale': POWER_GRID_SCALE}


def pair(value, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError(f'invalid {name}')
    return a


def rectangle(position, half_size, origin, cell, shape):
    """Exact rectangle/cell intersection fractions, clipped to this viewport."""
    lo = (position-half_size-origin)/cell
    hi = (position+half_size-origin)/cell
    x0, y0 = np.maximum(np.floor(lo), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(hi), [shape[1], shape[0]]).astype(int)
    if x0 >= x1 or y0 >= y1:
        return None
    xs, ys = np.arange(x0, x1), np.arange(y0, y1)
    dx = np.maximum(0, np.minimum(xs+1, hi[0])-np.maximum(xs, lo[0]))
    dy = np.maximum(0, np.minimum(ys+1, hi[1])-np.maximum(ys, lo[1]))
    return (slice(y0, y1), slice(x0, x1)), (dy[:, None]*dx).astype(np.float32)


def paint_laser(grid, channel, verified_channel, collision, origin, cell):
    """2x2 subcell sampling of a rotated binary-derived laser rectangle."""
    if not collision['active']:
        return
    pos = pair(collision['origin'], 'laser origin')
    angle, length, width = [float(collision[k]) for k in ('angle', 'length', 'width')]
    if not all(math.isfinite(v) for v in (angle, length, width)) or min(length, width) < 0:
        raise ValueError('invalid laser geometry')
    c, s = math.cos(angle), math.sin(angle)
    direction, normal = np.array([c, s]), np.array([-s, c])
    corners = np.array([pos+direction*t+normal*w for t in (0, length) for w in (-width/2, width/2)])
    lo = np.maximum(np.floor((corners.min(0)-origin)/cell), 0).astype(int)
    hi = np.minimum(np.ceil((corners.max(0)-origin)/cell), [grid.shape[2], grid.shape[1]]).astype(int)
    x0, y0 = lo; x1, y1 = hi
    if x0 >= x1 or y0 >= y1:
        return
    # Evaluate all four samples together, retaining the original arithmetic
    # order at rectangle edges. Axes are (sample offset, y, x).
    dx = origin[0]+(np.arange(x0, x1)[None, None, :]+np.array([.25,.25,.75,.75])[:,None,None])*cell-pos[0]
    dy = origin[1]+(np.arange(y0, y1)[None, :, None]+np.array([.25,.75,.25,.75])[:,None,None])*cell-pos[1]
    along, across = dx*c+dy*s, -dx*s+dy*c
    coverage = (((along >= 0) & (along <= length) & (np.abs(across) <= width/2))
                .sum(axis=0).astype(np.float32)*.25)
    sl = (slice(y0, y1), slice(x0, x1))
    grid[channel][sl] = np.maximum(grid[channel][sl], coverage)
    if collision['field_validated']:
        grid[verified_channel][sl] = np.maximum(grid[verified_channel][sl], coverage)


def bin_entities(entities, channels, grid, frames=0):
    """Point density, optionally offset each bullet before binning. No top-N cap."""
    counts = np.zeros(grid.shape[1:], dtype=np.float32)
    records = []
    if entities:
        positions = entity_pairs(entities, 'position') + [192, 0]
        if frames:
            positions += frames * entity_pairs(entities, 'velocity_raw')
        selected = np.flatnonzero((positions[:, 0] >= 0) & (positions[:, 0] < 384)
                                 & (positions[:, 1] >= 0) & (positions[:, 1] < 448))
        ix, iy = (positions[selected]//8).astype(int).T
        np.add.at(counts, (iy, ix), 1)
        records = [(entities[i], y, x) for i, y, x in zip(selected, iy, ix)]
    grid[channels[0]] = np.log1p(counts)/np.log(17)
    return counts, records


def entity_pairs(entities, key):
    values = np.asarray([e.get(key) for e in entities], dtype=np.float64)
    if values.shape != (len(entities), 2) or not np.isfinite(values).all():
        raise ValueError(f'invalid entity {key}')
    return values


def paint_bullets(local, active, origin):
    if not active:
        return
    positions = entity_pairs(active, 'position')
    sizes = entity_pairs(active, 'hitbox_raw')
    velocities = entity_pairs(active, 'velocity_raw')
    if np.any(sizes <= 0) or np.any(sizes > 512):
        raise ValueError('invalid bullet hitbox extent')
    for channel, frames in ((1, 0), (2, 2), (3, 4)):
        shifted = positions + frames * velocities
        paint_bullet_coverage(local[channel], shifted, sizes, origin)


def paint_bullet_coverage(layer, positions, sizes, origin):
    """Rasterize each full-size hitbox independently, then take maximum coverage."""
    lo, hi = (positions-sizes*.5-origin)/2, (positions+sizes*.5-origin)/2
    starts = np.clip(np.floor(lo), 0, 96).astype(int)
    ends = np.clip(np.ceil(hi), 0, 96).astype(int)
    widths = ends-starts
    areas = widths[:, 0]*widths[:, 1]
    selected = np.flatnonzero(areas > 0)
    # At most 128 * 96 * 96 cells per temporary batch even for huge hitboxes.
    for offset in range(0, len(selected), 128):
        group = selected[offset:offset+128]
        counts = areas[group]
        index = np.repeat(group, counts)
        ordinal = np.arange(counts.sum())-np.repeat(np.cumsum(counts)-counts, counts)
        x = starts[index, 0]+ordinal % widths[index, 0]
        y = starts[index, 1]+ordinal // widths[index, 0]
        dx = np.maximum(0, np.minimum(x+1, hi[index, 0])-np.maximum(x, lo[index, 0]))
        dy = np.maximum(0, np.minimum(y+1, hi[index, 1])-np.maximum(y, lo[index, 1]))
        coverage = (dx*dy).astype(np.float32)
        np.maximum.at(layer, (y, x), coverage)


class DualGridContract(gym.Env):
    def __init__(self):
        global_high = np.ones((len(GLOBAL_CHANNELS), 56, 48), dtype=np.float32)
        global_high[GLOBAL_CHANNELS.index('power_amount')] = POWER_GRID_MAX
        self.observation_space = gym.spaces.Dict({
            'local_grid': gym.spaces.Box(-1, 1, (len(LOCAL_CHANNELS), 96, 96), dtype=np.float32),
            'global_grid': gym.spaces.Box(-np.ones_like(global_high), global_high, dtype=np.float32),
            'player': gym.spaces.Box(-1, 1, (21,), dtype=np.float32),
            'previous_rewards': gym.spaces.Box(-1, 1, (4,), dtype=np.float32),
            'bomb_clock': gym.spaces.Box(0, 1, (1,), dtype=np.float32)})
        self.action_space = gym.spaces.MultiDiscrete([9, 2, 2, 2])

    def reset(self, **kwargs):
        raise RuntimeError('real collector owns reset')

    def step(self, action):
        raise RuntimeError('real collector owns stepping')

    def encode(self, raw, previous_rewards=None):
        player = raw.get('player')
        if player is None:
            raise ValueError('player missing')
        # A missing manager must not be mistaken for an empty safe scene.
        for key in ('bullets', 'enemies', 'items', 'lasers', 'player_shots'):
            if raw.get(key) is None:
                raise ValueError(f'{key} manager unavailable')
        out = {k: np.zeros(s.shape, dtype=np.float32) for k, s in self.observation_space.spaces.items()}
        local, whole = out['local_grid'], out['global_grid']
        pos = pair(player['position'], 'player position')
        origin = pos-96
        half = pair(player.get('hitbox_raw'), 'player hitbox')
        if np.any(half <= 0) or np.any(half > 64):
            raise ValueError('invalid player hitbox extent')
        patch = rectangle(pos, half, origin, 2, (96, 96))
        if patch is not None:
            sl, coverage = patch; local[0][sl] = coverage
        active = []
        for bullet in raw['bullets']:
            flags = bullet.get('flags_raw')
            if not isinstance(flags, int):
                raise ValueError('bullet collision flags unavailable; use current reader')
            if not flags & 2:
                continue
            active.append(bullet)
        # Maximum coverage is not exact union area for multiple objects in a cell.
        paint_bullets(local, active, origin)
        for channel, frames in enumerate((0, 2, 4)):
            bin_entities(active, (channel,), whole, frames)
        counts, enemies = bin_entities(raw['enemies'], (3,), whole)
        known_counts = np.zeros((56, 48), dtype=np.float32)
        for enemy, iy, ix in enemies:
            hp, maximum = enemy.get('hp'), enemy.get('hp_max')
            known = isinstance(hp, int) and isinstance(maximum, int) and 0 <= hp <= maximum <= 10000000 and maximum > 0
            if known:
                known_counts[iy, ix] += 1
                whole[4, iy, ix] += hp/100000
                whole[5, iy, ix] += maximum/100000
        whole[4] /= np.maximum(known_counts, 1)
        whole[5] /= np.maximum(known_counts, 1)
        if len(raw['items']) > MAX_ITEMS:
            raise ValueError('item count exceeds pinned pool')
        ordinary_items = []
        power_counts = np.zeros((56, 48), dtype=np.float64)
        for item in raw['items']:
            amount = power_item_raw(item)
            if not amount:
                ordinary_items.append(item)
                continue
            x, y = pair(item.get('position'), 'power item position') + [192, 0]
            if 0 <= x < 384 and 0 <= y < 448:
                power_counts[int(y//8), int(x//8)] += amount
        whole[11] = power_counts / POWER_GRID_SCALE
        bin_entities(ordinary_items, (6,), whole)
        px, py = pos+[192, 0]
        if 0 <= px < 384 and 0 <= py < 448:
            whole[7, int(py//8), int(px//8)] = 1
        for laser in raw['lasers']:
            collision = laser.get('collision')
            if collision is None:
                raise ValueError('unsupported laser geometry')
            paint_laser(local, 4, 5, collision, origin, 2)
            paint_laser(whole, 8, 9, collision, np.array([-192, 0]), 8)
        for shot in raw['player_shots']:
            if shot.get('geometry') != 'th10-player-shot-aabb-v1':
                raise ValueError('unverified player shot geometry')
            size = pair(shot.get('hitbox_raw'), 'player shot hitbox')
            if np.any(size <= 0) or np.any(size > 4096):
                raise ValueError('invalid player shot hitbox')
            patch = rectangle(pair(shot['position'], 'player shot position'), size/2,
                              np.array([-192, 0]), 8, (56, 48))
            if patch is not None:
                sl, coverage = patch
                np.maximum(whole[10][sl], coverage, out=whole[10][sl])
        bomb, spell = raw.get('bomb'), raw.get('spell')
        active_spell = bool(spell and spell['flags_raw'] & 1)
        pv = pair(player['velocity_raw'], 'player velocity')
        out['player'][:] = [pos[0]/192, pos[1]/448, * (pv/1000), raw['lives_raw']/8,
            raw['power_raw']/100, player['status']/4, player['invincibility_raw']/300,
            player['focus_raw'], 1, 1, 1, 1, raw['stage']/6, 1,
            bomb is not None, bool(bomb and bomb['state'] == 1),
            bool(bomb and bomb['state'] == 0 and raw['power_raw'] >= 20),
            any(e.get('is_boss') for e in raw['enemies']), active_spell,
            spell['id_raw']/128 if active_spell else 0]
        out['previous_rewards'] = reward_input(previous_rewards)
        for key, array in out.items():
            if not np.isfinite(array).all():
                raise ValueError(f'nonfinite {key}')
            space = self.observation_space.spaces[key]
            np.clip(array, space.low, space.high, out=array)
            if key.endswith('_grid'):
                # Same values at action sampling and PPO update, with half-size storage.
                out[key] = array.astype(np.float16).astype(np.float32)
        return out


class DualGridFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        # First local layer keeps resolution. Preserve a 12x12 spatial layout;
        # do not collapse the scene to a 2x2 average like the old far-grid path.
        self.local = nn.Sequential(nn.Conv2d(len(LOCAL_CHANNELS), 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16*12*12, 128), nn.ReLU())
        self.global_scene = nn.Sequential(nn.Conv2d(len(GLOBAL_CHANNELS), 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16*14*12, 128), nn.ReLU())
        self.player = nn.Sequential(nn.Linear(21, 32), nn.ReLU())
        self.reward = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(321, 256), nn.ReLU())

    def forward(self, observations):
        return self.merge(torch.cat([self.local(observations['local_grid']),
            self.global_scene(observations['global_grid']), self.player(observations['player']),
            self.reward(observations['previous_rewards']), observations['bomb_clock']], dim=1))


class GridRolloutBuffer(DictRolloutBuffer):
    """SB3-compatible storage; float16 grids, float32 rewards/GAE/actions."""
    def reset(self):
        self.observations = {k: np.zeros((self.buffer_size, self.n_envs, *shape),
            dtype=np.float16 if k.endswith('_grid') else np.float32)
            for k, shape in self.obs_shape.items()}
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        for key in ('rewards', 'returns', 'episode_starts', 'values', 'log_probs', 'advantages'):
            setattr(self, key, np.zeros((self.buffer_size, self.n_envs), dtype=np.float32))
        self.generator_ready = False
        self.pos, self.full = 0, False
