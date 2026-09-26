"""Full-scene entity selection for OBS, matching the live policy encoder."""
import math
SPEC = {'shape': 'nearest', 'individual_bullets': 40, 'individual_items': 40,
        'near_bullets': 10, 'grid_cell_pixels': None, 'version': 5}

def scoped(raw):
    return raw

def display_observation(raw, entities):
    result = {'bullet_scope': SPEC, 'bullets': None, 'items': None, 'bullet_grid': []}
    if raw.get('player') is None:
        return result
    px, py = raw['player']['position']
    for kind, limit in (('bullets', SPEC['individual_bullets']), ('items', SPEC['individual_items'])):
        values = raw.get(kind)
        if values is None:
            continue
        if any(e.get('velocity_raw') is None for e in values):
            continue  # Old diagnostic recordings may not contain policy-ready data.
        indices = sorted(range(len(values)), key=lambda i: (
            (values[i]['position'][0]-px)**2+(values[i]['position'][1]-py)**2,
            *values[i]['position'], *values[i]['velocity_raw']))
        result[kind] = [entities[kind][i] for i in indices[:limit]]
        if kind != 'bullets' or SPEC['grid_cell_pixels'] is None:
            continue
        cells = {}
        for i in indices[limit:]:
            e = values[i]
            x, y = e['position'][0]+192, e['position'][1]
            if 0 <= x < 384 and 0 <= y < 448:
                cell = cells.setdefault((int(x//4), int(y//4)), [0, 0., 0.])
                cell[0] += 1
                cell[1] += e['velocity_raw'][0]/10
                cell[2] += e['velocity_raw'][1]/10
        for (x, y), (count, vx, vy) in cells.items():
            result['bullet_grid'].append({'position': [x*4, y*4], 'count': count,
                'density': min(1., math.log1p(count)/math.log(17)),
                'velocity': [max(-1., min(1., v/count)) for v in (vx, vy)]})
    return result
