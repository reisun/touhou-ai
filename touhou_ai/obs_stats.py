"""Lossless reward windows from the collector log, independent of SSE sampling."""
from collections import deque
import json
import threading
import time
from touhou_ai.live_rewards import WEIGHTS, ENABLED, VERSION, MILESTONES, VERIFIED_PROGRESS_SOURCES
from touhou_ai.bullet_scope import SPEC
from touhou_ai.dual_grid import SPEC as GRID_SPEC


class ObsStats:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.path = None
        self.offset = 0
        self.rows = deque(maxlen=10000)
        self.weights = WEIGHTS
        self.run = None
        self.progress_cache = {}

    def episode_progress(self, run, episode):
        """Join by run + explicit episode number, never list position or reward.

        Completed logs are cached; appended/replaced logs are re-read. No verified
        milestone (including old logs) is unknown, not a zero achievement.
        """
        if type(episode) is not int or episode < 1:
            return None
        path = run / f'episode-{episode}.jsonl'
        try:
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            cached = self.progress_cache.get(path)
            if cached and cached[0] == signature:
                return cached[1]
            best = None
            with path.open('rb') as stream:
                for line in stream:
                    if not line.endswith(b'\n'):
                        break
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if not isinstance(row, dict):
                        continue
                    if row.get('telemetry', {}).get('episode_id') != f'{run.name}-{episode}':
                        continue
                    for event in row.get('events', []):
                        if not isinstance(event, dict):
                            continue
                        stage, milestone = event.get('stage'), event.get('milestone')
                        if (event.get('kind') != 'progress' or event.get('confirmed') is not True
                                or event.get('source') not in VERIFIED_PROGRESS_SOURCES
                                or type(stage) is not int or not 1 <= stage <= 6
                                or milestone not in MILESTONES):
                            continue
                        rank = (stage - 1) * len(MILESTONES) + MILESTONES.index(milestone) + 1
                        if best is None or rank > best['rank']:
                            best = dict(stage=stage, milestone=milestone, rank=rank)
            self.progress_cache[path] = (signature, best)
            return best
        except OSError:
            return None

    def snapshot(self):
        with self.lock:
            manifests = []
            for path in sorted(self.root.glob('live-learning-*/status.json')):
                try:
                    status = json.loads(path.read_text(encoding='utf-8'))
                    if (status.get('reward_version') == VERSION
                            and status.get('bullet_scope') in (SPEC, GRID_SPEC)):
                        manifests.append((path, status))
                except (OSError, ValueError):
                    continue
            if manifests:
                latest_path, latest = manifests[-1]
                manifests = [(p, s) for p, s in manifests
                             if s.get('bullet_scope') == latest.get('bullet_scope')
                             and s.get('contract') == latest.get('contract')]
                if self.run != latest_path.parent:
                    self.rows.clear()
                    self.path, self.offset = None, 0
                    self.weights = latest.get('reward_weights', WEIGHTS)
                    self.run = latest_path.parent
            growth = [dict(e, run=p.parent.name, max_progress=self.episode_progress(p.parent, e.get('episode'))) for p, s in manifests
                      if s.get('backend') == 'real_th10' for e in s.get('episodes', [])
                      if e.get('reload_verified')]
            paths = [] if not manifests else sorted(manifests[-1][0].parent.glob('episode-*.jsonl'),
                                                    key=lambda p: int(p.stem.split('-')[-1]))
            # Include adjacent episodes in the same wall-clock window.
            for path in paths:
                if self.path is not None and int(path.stem.split('-')[-1]) < int(self.path.stem.split('-')[-1]) and path.parent == self.path.parent:
                    continue
                if self.path != path:
                    if self.path is not None and path.parent != self.path.parent:
                        self.rows.clear()
                    self.path, self.offset = path, 0
                with path.open('rb') as stream:
                    stream.seek(self.offset)
                    while True:
                        line = stream.readline()
                        if not line or not line.endswith(b'\n'):
                            break
                        self.offset = stream.tell()
                        try:
                            row = json.loads(line)
                            t = row['telemetry']
                            if t.get('policy') and t.get('reward'):
                                self.weights = t.get('model', {}).get('reward_weights', {})
                                self.rows.append({'time': t['timestamp'], 'episode': t['episode_id'],
                                    'value': t['policy']['value'], 'reward': t['reward']['total'],
                                    'components': t['reward']['components'], 'enabled': t['reward']['enabled'],
                                    'terminal': row['raw']['lives_raw'] < 0,
                                    'gamma': t.get('model', {}).get('ppo', {}).get('gamma', .99)})
                        except (ValueError, KeyError, TypeError):
                            continue
            end = time.time()
            points = []
            rows = list(self.rows)
            for i, row in enumerate(rows):
                if not end-30 < row['time'] <= end:
                    continue
                next_row = rows[i+1] if i+1 < len(rows) else None
                td = row['reward']-row['value'] if row['terminal'] else (
                    row['reward']+row['gamma']*next_row['value']-row['value']
                    if next_row and next_row['episode'] == row['episode'] else None)
                points.append(dict(row, td=td))
            enabled = rows[-1]['enabled'] if rows else ENABLED
            totals = {k: sum(p['components'].get(k, 0) for p in points) for k in enabled}
            return {'start': end-30, 'end': end, 'points': points, 'totals': totals, 'weights': self.weights, 'enabled': enabled,
                    'growth': growth, 'samples': len(points),
                    'td_mean': (sum(p['td'] for p in points if p['td'] is not None)/
                                sum(p['td'] is not None for p in points))
                               if any(p['td'] is not None for p in points) else None}
