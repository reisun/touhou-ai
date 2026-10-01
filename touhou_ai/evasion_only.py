"""Death-only experiment with shot and bomb excluded from policy optimization."""
import torch
from torch.distributions import Categorical
from touhou_ai.timed_bomb import TimedBombPolicy, TimedDistribution
from touhou_ai.live_rewards import LiveRewards
VERSION = 'th10-evasion-death-only-v1'
WEIGHTS = dict(damage=0., damage_power=0., progress=0., progress_life=0., progress_power=0., hit=-60., power_down=0.)
ENABLED = ['hit']
ACTION_CONTRACT = 'movement-focus-only-v1'

class EvasionPolicy(TimedBombPolicy):
    def _scheduled(self, latent, obs):
        base = self._get_action_dist_from_latent(latent)
        # Constant categorical factors: no sampling, entropy, or gradient from disabled actions.
        for index in (1, 3):
            logits = torch.zeros((latent.shape[0], 2), device=latent.device, dtype=latent.dtype)
            logits[:, 1] = -1e9  # Exact zero probability in float32; finite for DirectML masked log-prob.
            base.distribution[index] = Categorical(logits=logits)
        return TimedDistribution(base, torch.zeros(latent.shape[0], device=latent.device, dtype=latent.dtype))

class DeathOnlyRewards(LiveRewards):
    def __init__(self):
        # Preserve the independent live experiment's reward contract.
        super().__init__(weights=WEIGHTS)

    def calculate(self, episode, events):
        _, components = super().calculate(episode, [e for e in events if e.get('kind') == 'hit'])
        return components['hit'], dict(damage=0., progress=0., hit=components['hit'], power_down=0.)
