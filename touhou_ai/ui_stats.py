"""Compact UI reward window in memory; never contains raw observations."""
from collections import deque
import time
from touhou_ai.telemetry_memory import publish
from touhou_ai.progress_schema import progress_point
from touhou_ai.live_rewards import VERIFIED_PROGRESS_SOURCES


class UiStats:
    def __init__(self, output):
        self.output = output
        self.rows = deque()
        self.best = None
        self.weights = {}
        self.last_publish = 0

    def begin_episode(self):
        self.best = None

    def add(self, telemetry, events, terminal, episode, gamma):
        now = telemetry.get('timestamp', time.time())
        self.weights = telemetry.get('model', {}).get('reward_weights', {})
        reward = telemetry['reward']
        self.rows.append({'time': now, 'episode': episode,
            'value': telemetry['policy']['value'], 'reward': reward['total'],
            'components': reward['components'], 'enabled': reward['enabled'],
            'terminal': terminal, 'gamma': gamma})
        while self.rows and self.rows[0]['time'] <= now-31:
            self.rows.popleft()
        for event in events:
            if (event.get('kind') != 'progress' or event.get('confirmed') is not True
                    or event.get('source') not in VERIFIED_PROGRESS_SOURCES):
                continue
            try:
                point = progress_point(event)
            except ValueError:
                continue
            if self.best is None or point['rank'] > self.best['rank']:
                self.best = dict(point)
        if terminal or time.monotonic()-self.last_publish >= 1:
            self.flush()

    def flush(self):
        if publish({'run': self.output.name, 'rows': list(self.rows), 'weights': self.weights},
                   self.output.parent / 'ui-stats'):
            self.last_publish = time.monotonic()
