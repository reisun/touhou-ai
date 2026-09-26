"""Versioned discount time unit, independent of the bomb decision clock."""
GAMMA = 0.9995
SPEC = {'version': 'th10-discount-2f-v1', 'gamma': GAMMA,
        'gameplay_frames_per_step': 2, 'unit': 'collected_gameplay_transition'}


def verify_settings(settings, prior=None):
    if settings.get('gamma') != GAMMA:
        raise ValueError('configured gamma differs from the 2F discount contract')
    if prior is not None and (prior.get('discount_contract') != SPEC or
                              prior.get('configured_ppo', {}).get('gamma') != GAMMA):
        raise ValueError('checkpoint discount contract differs; explicit new campaign required')
