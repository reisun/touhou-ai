"""Extended measured interface. Missing values have explicit availability channels."""
import math

EXTENDED_CONTRACT = 'th10-live-observed-v2'


def laser_collision(kind, state, position, angle, length, width):
    """Binary-derived rectangle with narrowly scoped live-call verification evidence."""
    if kind not in ('line', 'infinite'):
        return None
    if not all(math.isfinite(v) for v in (*position, angle, length, width)) or min(length, width) < 0:
        raise ValueError('invalid laser geometry')
    # Calls at 41d6be / 41e935 pass a 10%-offset origin and an 80%-length rectangle.
    reduced = width/2 if width <= 32 else width-(width+16)/(2 if kind == 'line' else 3)
    active = state != 1 and length > 16 and (width > 3 if kind == 'line' else state in (2, 4))
    # 1,600 actual calls, one hit, zero mismatches; do not generalize to other widths/states.
    sampled = kind == 'line' and state == 2 and width == 14 and 144 <= length <= 180
    return {'origin': [position[0]+math.cos(angle)*length*.1,
                       position[1]+math.sin(angle)*length*.1],
            'angle': angle, 'length': length*.8, 'width': max(0, reduced),
            'active': active, 'provenance': 'binary_derived', 'field_validated': sampled,
            'validation_scope': 'line_state2_width14_length144_180' if sampled else None,
            'activation_transition_validated': False}


def bomb_events(before, after):
    old, new = before.get('bomb'), after.get('bomb')
    if old is None or new is None:
        raise ValueError('bomb state unavailable')
    if old['state'] not in (0, 1) or new['state'] not in (0, 1):
        raise ValueError('unknown bomb state')
    if old['state'] == 0 and new['state'] == 1:
        return [{'id': f"bomb-start:{after['stage_frame']}", 'kind': 'bomb', 'confirmed': True}]
    return []


def extended_space(base):
    import gymnasium as gym
    import numpy as np
    shapes = {'player': (21,), 'bullets': (128, 8), 'enemies': (24, 9), 'lasers': (64, 12)}
    return gym.spaces.Dict({key: gym.spaces.Box(-1, 1, shapes[key], dtype=np.float32)
                           if key in shapes else space for key, space in base.spaces.items()})


def encode_extended(raw, base_observation):
    import numpy as np
    out = {key: np.zeros(shape, dtype=np.float32) for key, shape in
           {'player': (21,), 'bullets': (128, 8), 'enemies': (24, 9), 'lasers': (64, 12)}.items()}
    for key in base_observation:
        if key not in out:
            out[key] = base_observation[key]
        elif key == 'player':
            out[key][:15] = base_observation[key]
        else:
            out[key][:, :5] = base_observation[key]
    bomb, spell = raw.get('bomb'), raw.get('spell')
    bosses = [e for e in raw.get('enemies') or [] if e.get('is_boss')]
    active_spell = bool(spell and spell['flags_raw'] & 1)
    out['player'][15:] = [bomb is not None, bool(bomb and bomb['state'] == 1),
        bool(bomb and bomb['state'] == 0 and raw['power_raw'] >= 20),
        bool(bosses), active_spell, (spell['id_raw']/128 if active_spell else 0)]
    x, y = raw['player']['position']
    for kind in ('bullets', 'enemies', 'lasers'):
        ordered = sorted(raw.get(kind) or [], key=lambda e: ((e['position'][0]-x)**2+(e['position'][1]-y)**2,
                                                          *e['position'], *e['velocity_raw']))
        for index, entity in enumerate(ordered[:len(out[kind])]):
            if kind == 'bullets':
                acceleration = entity.get('acceleration')
                out[kind][index, 5:] = [*(acceleration or [0, 0]), acceleration is not None]
            elif kind == 'enemies':
                hp, maximum = entity.get('hp'), entity.get('hp_max')
                known = isinstance(hp, int) and isinstance(maximum, int) and 0 <= hp <= maximum <= 10000000 and maximum > 0
                out[kind][index, 5:] = [hp/maximum if known else 0, maximum/100000 if known else 0,
                                      known, bool(entity.get('is_boss'))]
            else:
                collision = entity.get('collision')
                if collision is None:
                    raise ValueError('unsupported laser type; no fabricated collision geometry')
                out[kind][index, 1:3] = [(collision['origin'][0]-x)/384,
                                         (collision['origin'][1]-y)/448]
                # The extra validity channel explicitly distinguishes binary-derived from field-verified data.
                out[kind][index, 5:] = [math.cos(collision['angle']), math.sin(collision['angle']),
                    collision['length']/448, collision['width']/64, collision['active'],
                    1, collision['field_validated']]
    for array in out.values():
        if not np.isfinite(array).all():
            raise ValueError('nonfinite extended observation')
        np.clip(array, -1, 1, out=array)
    return out
