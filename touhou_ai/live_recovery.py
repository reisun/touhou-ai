"""Conservative recovery eligibility; explicit user stops always win."""
from touhou_ai.live_transition import CollectionInterrupted


def recovery_allowed(runtime, error, stopped, attempts, max_attempts=2):
    if stopped or (max_attempts is not None and attempts >= max_attempts) or isinstance(error, CollectionInterrupted):
        return False
    if runtime is None:
        return False  # Do not retry ownership, configuration, or attachment failures.
    try:
        if runtime.api.status().get('fault') == 'emergency F8':
            return False
    except Exception:
        pass  # A crashed process cannot answer RPCs.
    try:
        state = runtime.snapshot(full=False)
        words = state.get('pause_words')
        if state.get('lives_raw', -1) >= 0 and words and words[1] == 2:
            return False
    except Exception:
        pass
    return isinstance(error, (RuntimeError, ValueError, OSError, TimeoutError)) or type(error).__module__.startswith(('frida', '_frida'))
