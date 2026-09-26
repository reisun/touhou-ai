"""Continue a live episode through dialogue/loading without resetting or optimizing."""


class CollectionInterrupted(Exception):
    pass


def bridge_transition(runtime, state, previous_stage, *, should_stop=lambda: False,
                      publish=None, budget=1800):
    info = {'from_stage': previous_stage, 'to_stage': state['stage'],
            'automatic_frames': 0, 'dialogue_frames': 0, 'loading_frames': 0}
    progress = {e['id']: e for e in state.get('progress_reward_events', [])}

    def validate(current):
        if ((current.get('character'), current.get('shot'), current.get('difficulty')) != (0, 1, 1)
                or current.get('mode_flags') != 0 or current.get('replay_mode') not in (None, 0)
                or current.get('stage') not in (previous_stage, previous_stage+1)
                or current.get('stage') not in range(1, 7)):
            raise ValueError('unsupported transition, replay, or game mode')
        words = current.get('pause_words')
        if current['lives_raw'] >= 0 and words is not None and words[1] != 0:
            raise ValueError('user pause/menu must not be automatically dismissed')

    def ready(current):
        return (current.get('player') is not None and current.get('replay_mode') == 0
                and current.get('pause_words') is not None
                and current['pause_words'][1] == 0 and not current.get('dialogue_raw')
                and current['stage_frame'] > 0)

    for index in range(budget+1):
        if should_stop():
            raise CollectionInterrupted('stop requested during transition')
        validate(state)
        info['to_stage'] = state['stage']
        if state['lives_raw'] < 0 or ready(state):
            # A full, final snapshot is needed for the policy, not a loading placeholder.
            final = runtime.snapshot()
            validate(final)
            if final['lives_raw'] < 0 or ready(final):
                info['to_stage'] = final['stage']
                progress.update((e['id'], e) for e in final.get('progress_reward_events', []))
                final['progress_reward_events'] = list(progress.values())
                return final, info
            state = final
        if index == budget:
            break
        dialogue = bool(state.get('dialogue_raw'))
        mask = 1 if dialogue and index % 8 == 0 else 0
        state = runtime.step(mask, 1, full=False)
        progress.update((e['id'], e) for e in state.get('progress_reward_events', []))
        info['automatic_frames'] += 1
        info['dialogue_frames' if dialogue else 'loading_frames'] += 1
        if publish is not None and info['automatic_frames'] % 15 == 0:
            publish(state, dict(info))
    raise TimeoutError('dialogue/loading did not finish within frame budget')
