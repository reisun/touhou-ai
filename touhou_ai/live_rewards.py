"""Versioned real-game rewards; event-time evidence only, never inferred kills."""
import math
from touhou_ai.progress_schema import MILESTONES, progress_point
VERSION = 'th10-rewards-v21'
# v17 scale retained, except the explicitly increased progress Power coefficient.
WEIGHTS = {'damage': 15./60, 'damage_power': .5, 'progress': 20./60, 'progress_life': 30./60, 'progress_power': .1, 'hit': -1., 'power_down': 0., 'jitter': -.1, 'power_gain': .1}

def validate_power_upgrade(manifest):
    if (manifest.get('reward_version') != 'th10-rewards-v17'
            or manifest.get('reward_weights') != {k: v for k, v in WEIGHTS.items() if k not in ('jitter', 'power_gain')} | {'progress_power': 2./60}
            or manifest.get('evasion_only')):
        raise ValueError('Power upgrade requires exactly the full v17 reward contract')
def validate_power_gain_upgrade(manifest):
    if (manifest.get('reward_version') != 'th10-rewards-v19'
            or manifest.get('reward_weights') != {k: v for k, v in WEIGHTS.items() if k != 'power_gain'}
            or manifest.get('evasion_only')):
        raise ValueError('Power gain addition requires exactly the full v19 reward contract')


ENABLED = ['damage', 'progress', 'hit', 'jitter', 'power_gain']
# Engine paths: live stage-1 acceptance; stage 1..6 mappings: pinned ECL review.
VERIFIED_PROGRESS_SOURCES = frozenset({'verified_ecl_progress_v1', 'verified_ecl_progress_v2'})


def power_value(raw):
    if type(raw) is not int or not 0 <= raw <= 100:
        raise ValueError('invalid event-time power')
    return raw / 20


class LiveRewards:
    def __init__(self, weights=None):
        self.weights = dict(WEIGHTS if weights is None else weights)
        self.seen = set()
        self.milestones = set()

    def reset(self, episode):
        self.episode = episode
        self.seen.clear()
        self.milestones.clear()

    def calculate(self, episode, events):
        if episode != self.episode:
            raise ValueError('reward episode mismatch')
        result = dict.fromkeys(ENABLED, 0.)
        seen, milestones = set(self.seen), set(self.milestones)
        for e in events:
            if e.get('confirmed') is not True or not isinstance(e.get('id'), str):
                raise ValueError('unconfirmed reward event')
            if e['id'] in seen:
                continue
            kind = e['kind']
            reward = 0.
            if kind == 'damage':
                amount = e['amount']
                if not math.isfinite(amount) or amount < 0:
                    raise ValueError('invalid damage amount')
                if e.get('source') != 'verified_game_event':
                    raise ValueError('unverified damage source')
                power = power_value(e.get('power_raw'))
                if type(e.get('bomb_state')) is not int or e['bomb_state'] not in (0, 1):
                    raise ValueError('unknown event-time bomb state')
                if e['bomb_state'] == 0:
                    reward = self.weights['damage'] * amount / 1000 * (1 + self.weights['damage_power'] * power)
            elif kind == 'progress':
                if e.get('source') not in VERIFIED_PROGRESS_SOURCES:
                    raise ValueError('progress source has not passed real-game acceptance')
                point = progress_point(e)
                stage, milestone = point['stage'], point['milestone']
                key = (stage, milestone, point.get('spell_id_raw'))
                lives = e.get('lives_raw')
                if type(lives) is not int or not 0 <= lives <= 8:
                    raise ValueError('invalid milestone reserve lives')
                power = power_value(e.get('power_raw'))
                if key not in milestones:
                    if type(e.get('bomb_state')) is not int or e['bomb_state'] not in (0, 1):
                        raise ValueError('unknown event-time bomb state')
                    if e['bomb_state'] == 0:
                        reward = self.weights['progress'] + self.weights['progress_life'] * lives + self.weights['progress_power'] * power
                    # Suppressed milestones are consumed too; no delayed bonus.
                    milestones.add(key)
            elif kind == 'power_gain':
                from touhou_ai.power_items import POWER_RAW_BY_TYPE
                before, after, amount = (e.get(k) for k in ('before_raw', 'after_raw', 'amount_raw'))
                nominal = POWER_RAW_BY_TYPE.get(e.get('item_type'))
                if (e.get('source') != 'verified_power_pickup_v1' or nominal is None
                        or not all(type(v) is int for v in (before, after, amount))
                        or not 0 <= before < after <= 100
                        or amount != after-before or amount != min(nominal, 100-before)):
                    raise ValueError('invalid verified Power gain')
                if type(e.get('bomb_state')) is not int or e['bomb_state'] not in (0, 1):
                    raise ValueError('unknown pickup-time bomb state')
                if e['bomb_state'] == 0:
                    reward = self.weights['power_gain'] * amount / 20
            elif kind == 'jitter':
                if e.get('source') != 'actual_displacement_12f_v1':
                    raise ValueError('unverified jitter source')
                reward = self.weights['jitter']
            elif kind == 'hit':
                reward = self.weights['hit']
            else:
                raise ValueError('unsupported reward kind')
            if kind in result:
                result[kind] += reward
            seen.add(e['id'])
        self.seen, self.milestones = seen, milestones
        return sum(result.values()), result


def observed_events(before, after, action=None):
    from touhou_ai.live_learning import hit_events
    events = hit_events(before, after, allow_stage_transition=True)
    for e in after.get('combat_reward_events', []):
        if e.get('kind') == 'power_gain' and e.get('source') == 'verified_power_pickup_v1':
            events.append(e)
            continue
        if e.get('source') != 'verified_game_event' or e.get('kind') not in ('damage', 'kill'):
            raise ValueError('unverified combat reward source')
        # Retain kills in raw combat logs, not in the abolished reward component.
        if e['kind'] == 'damage':
            events.append(e)
    for e in after.get('progress_reward_events', []):
        if e.get('kind') != 'progress' or e.get('source') not in VERIFIED_PROGRESS_SOURCES:
            raise ValueError('unverified progress reward source')
        events.append(e)
    return events
