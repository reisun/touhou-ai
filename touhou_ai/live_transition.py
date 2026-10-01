"""Continue a live episode through dialogue/loading without resetting or optimizing."""


class CollectionInterrupted(Exception):
    pass


def bridge_transition(runtime, state, previous_stage, *, should_stop=lambda: False,
                      publish=None, budget=1800):
    info = {'from_stage': previous_stage, 'to_stage': state['stage'],
            'automatic_frames': 0, 'dialogue_frames': 0, 'loading_frames': 0}
    progress = {e['id']: e for e in state.get('progress_reward_events', [])}

    def loading(current):
        # Pinned executable: 0x417d26..0x417d47 sets bit 2 for screen 11;
        # the stage initializer at 0x417c34 clears it before gameplay resumes.
        # Live trace: the next stage's screen has already become 7 at frame 0,
        # while bit 2 remains set until that stage's initialization completes.
        # Score-record bit 4 can coexist with loading bit 2. Keep the same
        # screen/stage/frame restrictions; 6 is never ordinary gameplay.
        return (previous_stage < 6 and current.get('mode_flags') in (2, 6)
                and (current.get('screen_state_raw') == 11
                     or (current.get('screen_state_raw') == 7
                         and current.get('stage') == previous_stage + 1
                         and current.get('stage_frame') == 0)))

    def validate(current):
        if ((current.get('character'), current.get('shot'), current.get('difficulty')) != (0, 1, 1)
                or (current.get('mode_flags') not in (0, 4) and not loading(current))
                or current.get('replay_mode') not in (None, 0)
                or current.get('stage') not in (previous_stage, previous_stage+1)
                or current.get('stage') not in range(1, 7)):
            fields = {key: current.get(key) for key in
                      ('character', 'shot', 'difficulty', 'stage', 'stage_frame',
                       'mode_flags', 'replay_mode', 'screen_state_raw', 'pause_words', 'dialogue_raw',
                       'stage_init_pending', 'stage_manager_raw')}
            raise ValueError(f'unsupported transition, replay, or game mode: {fields}')
        words = current.get('pause_words')
        if current['lives_raw'] >= 0 and words is not None and words[1] != 0:
            raise ValueError('user pause/menu must not be automatically dismissed')

    def ready(current):
        return (current.get('mode_flags') in (0, 4)
                and current.get('stage_init_pending', False) is False
                and current.get('player') is not None and current.get('replay_mode') == 0
                and current.get('pause_words') is not None
                and current['pause_words'][1] == 0 and not current.get('dialogue_raw')
                and current['stage_frame'] > 0)

    for index in range(budget+1):
        if should_stop():
            raise CollectionInterrupted('stop requested during transition')
        validate(state)
        info['to_stage'] = state['stage']
        # Reuse only the complete, frame-locked observation from this exact gate.
        # Dialogue/loading, stage changes and terminal states keep the full refresh.
        if (index == 0 and state['stage'] == previous_stage and state['lives_raw'] >= 0
                and ready(state) and state.get('snapshot_full') is True
                and runtime.owns_full_snapshot(state)):
            return state, info
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
        dialogue = (bool(state.get('dialogue_raw')) and not loading(state)
                    and state.get('stage_init_pending', False) is False)
        if state.get('stage_init_pending') is True:
            info['stage_initialization_frames'] = info.get('stage_initialization_frames', 0) + 1
        mask = 1 if dialogue and index % 8 == 0 else 0
        state = runtime.step(mask, 1, full=False)
        progress.update((e['id'], e) for e in state.get('progress_reward_events', []))
        info['automatic_frames'] += 1
        info['dialogue_frames' if dialogue else 'loading_frames'] += 1
        if publish is not None and info['automatic_frames'] % 15 == 0:
            publish(state, dict(info))
    raise TimeoutError('dialogue/loading did not finish within frame budget')
