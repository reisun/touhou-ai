"""Lossless reward windows from the collector log, independent of SSE sampling."""
from collections import deque
from touhou_ai.telemetry_memory import read as read_memory
import json
import threading
import time
from pathlib import Path
from touhou_ai.live_rewards import WEIGHTS, ENABLED, VERSION, MILESTONES, VERIFIED_PROGRESS_SOURCES
from touhou_ai.bullet_scope import SPEC
from touhou_ai.dual_grid import SPEC as GRID_SPEC
from touhou_ai.progress_schema import progress_point, PROGRESS_AXIS


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
        self.manifest_cache = {}
        self.manifests = []
        self.next_manifest_check = 0
        self.growth_key = None
        self.growth = []

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
                                or type(stage) is not int or not 1 <= stage <= 6):
                            continue
                        try:
                            point = progress_point(event)
                        except ValueError:
                            continue
                        rank = point['rank']
                        if best is None or rank > best['rank']:
                            best = dict(point)
            self.progress_cache[path] = (signature, best)
            return best
        except OSError:
            return None

    def _read_cached(self, path):
        signature = (path.stat().st_mtime_ns, path.stat().st_size)
        cached = self.manifest_cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        self.manifest_cache[path] = (signature, value)
        return value

    def _refresh_manifests(self):
        now = time.monotonic()
        if now < self.next_manifest_check:
            return self.manifests
        self.next_manifest_check = now + 1
        record_path = self.root.parent / '.runtime' / 'live-learning.json'
        try:
            if record_path.exists():
                record = self._read_cached(record_path)
                name = record['RunId']
                if not isinstance(name, str) or not name.startswith('live-learning-') or Path(name).name != name:
                    return []
                path = self.root / name / 'status.json'
            elif self.manifests:
                path = self.manifests[-1][0]
            else:
                # Offline/legacy artifacts only. Managed LIVE never scans other runs.
                paths = sorted(self.root.glob('live-learning-*/status.json'))
                if not paths:
                    return []
                path = paths[-1]
            latest = self._read_cached(path)
            chain = [(path, latest)]
            seen = {path.resolve()}
            current = latest
            while current.get('resumed_from'):
                parent = Path(current['resumed_from']).parent / 'status.json'
                resolved = parent.resolve()
                if resolved in seen or resolved.parent.parent != self.root.resolve():
                    break
                seen.add(resolved)
                # Completed ancestors cannot change while this learner runs.
                previous = self.manifest_cache.get(parent)
                previous = previous[1] if previous else self._read_cached(parent)
                transition = current.get('reward_transition') or {}
                reward_match = all(previous.get(k) == current.get(k) for k in ('reward_version', 'reward_weights'))
                explicit_transition = (transition.get('from') == previous.get('reward_version')
                    and transition.get('to') == current.get('reward_version')
                    and transition.get('old_weights') == previous.get('reward_weights')
                    and transition.get('new_weights') == current.get('reward_weights')
                    and transition.get('policy_optimizer_rng_preserved') is True)
                if (any(previous.get(k) != current.get(k) for k in
                        ('contract', 'action_contract', 'bullet_scope'))
                        or not (reward_match or explicit_transition)):
                    break
                chain.append((parent, previous))
                current = previous
            if not record_path.exists():
                # Preserve standalone archive viewing when no managed run exists.
                chain = []
                for archive in sorted(self.root.glob('live-learning-*/status.json')):
                    item = self._read_cached(archive)
                    if all(item.get(k) == latest.get(k) for k in ('reward_version', 'contract', 'action_contract', 'bullet_scope')):
                        chain.append((archive, item))
                self.manifests = chain
            else:
                self.manifests = list(reversed(chain))
        except (OSError, ValueError, KeyError, TypeError):
            # A new run may be registered before its manifest is ready. Never show the old run.
            self.manifests = []
        return self.manifests

    def snapshot(self):
        with self.lock:
            manifests = self._refresh_manifests()
            latest_path, latest = manifests[-1] if manifests else (None, {})
            active = latest_path.parent if latest_path else None
            if self.run != active:
                self.rows.clear()
                self.path, self.offset = None, 0
                self.run = active
            self.weights = latest.get('reward_weights', {})
            key = tuple((str(p), id(s)) for p, s in manifests)
            if key != self.growth_key:
                self.growth = [dict(e, run=p.parent.name, reward_version=status.get('reward_version'),
                    max_progress=e['max_progress'] if 'max_progress' in e else self.episode_progress(p.parent, e.get('episode')))
                    for p, status in manifests if status.get('backend') == 'real_th10'
                    for e in status.get('episodes', []) if e.get('reload_verified')]
                self.growth_key = key
            growth = self.growth
            memory_mode = bool(manifests and latest.get('ui_stats_transport') == 'shared_memory_v1')
            if memory_mode:
                update = read_memory(self.root / 'ui-stats')
                if update is not None:
                    summary = json.loads(update[1])
                    if summary.get('run') == latest_path.parent.name:
                        self.rows = deque(summary['rows'], maxlen=10000)
                        self.weights = summary['weights']
            paths = [] if not manifests or memory_mode else sorted(manifests[-1][0].parent.glob('episode-*.jsonl'),
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
            enabled = rows[-1]['enabled'] if rows else (latest.get('reward_enabled', ENABLED) if manifests else ENABLED)
            totals = {k: sum(p['components'].get(k, 0) for p in points) for k in enabled}
            return {'start': end-30, 'end': end, 'points': points, 'totals': totals, 'weights': self.weights, 'enabled': enabled,
                    'active_run': self.run.name if self.run else None, 'reward_version': latest.get('reward_version'),
                    'growth': growth, 'progress_axis': PROGRESS_AXIS, 'samples': len(points),
                    'td_mean': (sum(p['td'] for p in points if p['td'] is not None)/
                                sum(p['td'] is not None for p in points))
                               if any(p['td'] is not None for p in points) else None}
