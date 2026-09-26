"""Versioned real-game rewards; event-time evidence only, never inferred kills."""
import math
VERSION = 'th10-rewards-v11'
WEIGHTS = {'damage': 5., 'damage_power': .5, 'progress': 20., 'progress_life': 30., 'progress_power': 5., 'hit': -10., 'power_down': 0.}
ENABLED = ['damage', 'progress', 'hit', 'power_down']
MILESTONES = ('midboss_arrival', 'midboss_defeat', 'boss_arrival', 'boss_defeat')
# Engine paths: live stage-1 acceptance; stage 1..6 mappings: pinned ECL review.
VERIFIED_PROGRESS_SOURCES = frozenset({'verified_ecl_progress_v1'})


def power_value(raw):
    if type(raw) is not int or not 0 <= raw <= 100:
        raise ValueError('invalid event-time power')
    return raw / 20


class LiveRewards:
    def __init__(self):
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
                    reward = WEIGHTS['damage'] * amount / 1000 * (1 + WEIGHTS['damage_power'] * power)
            elif kind == 'progress':
                if e.get('source') not in VERIFIED_PROGRESS_SOURCES:
                    raise ValueError('progress source has not passed real-game acceptance')
                stage, milestone = e.get('stage'), e.get('milestone')
                if type(stage) is not int or stage not in range(1, 7) or milestone not in MILESTONES:
                    raise ValueError('invalid milestone')
                lives = e.get('lives_raw')
                if type(lives) is not int or not 0 <= lives <= 8:
                    raise ValueError('invalid milestone reserve lives')
                power = power_value(e.get('power_raw'))
                if (stage, milestone) not in milestones:
                    reward = WEIGHTS['progress'] + WEIGHTS['progress_life'] * lives + WEIGHTS['progress_power'] * power
                    milestones.add((stage, milestone))
            elif kind == 'hit':
                reward = WEIGHTS['hit']
            elif kind == 'power_down':
                amount = e.get('amount')
                if type(amount) not in (int, float) or not math.isfinite(amount) or not 0 < amount <= 5:
                    raise ValueError('invalid observed power decrease')
                reward = WEIGHTS['power_down'] * amount
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
    power_before, power_after = power_value(before.get('power_raw')), power_value(after.get('power_raw'))
    if power_after < power_before:
        events.append(dict(id=f"power-down:{after['stage']}:{after['stage_frame']}",
                           kind='power_down', amount=power_before-power_after, confirmed=True))
    for e in after.get('combat_reward_events', []):
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
