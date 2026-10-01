"""Explicit weight transfer from the death-only live experiment to full rewards."""
import torch
from touhou_ai.checkpoint_rng import load_preserving_rng, restore_rng
from touhou_ai.evasion_only import EvasionPolicy, VERSION, WEIGHTS
from touhou_ai.timed_bomb import TimedBombPolicy
from touhou_ai.separate_clip_ppo import SeparateClipPPO

def transfer_evasion(checkpoint, manifest, target):
    if (manifest.get('reward_version') != VERSION or manifest.get('reward_weights') != WEIGHTS
            or not manifest.get('evasion_only')
            or manifest.get('cnn_architecture') != 'narrow_action_grid'
            or manifest.get('algorithm') != 'SeparateClipPPO'):
        raise ValueError('Transfer requires the verified action-grid death-only experiment')
    source = load_preserving_rng(SeparateClipPPO, checkpoint, device='cpu')
    if type(source.policy) is not EvasionPolicy or type(target.policy) is not TimedBombPolicy:
        raise ValueError('Unexpected transfer policy types')
    target.policy.load_state_dict(source.policy.state_dict(), strict=True)
    with torch.no_grad():
        target.policy.value_net.weight.div_(60)
        target.policy.value_net.bias.div_(60)
    # New objective: discard old Adam moments; retain learned policy/features.
    if getattr(source, 'collector_rng_state', None) is not None:
        restore_rng(source.collector_rng_state)
    return source
