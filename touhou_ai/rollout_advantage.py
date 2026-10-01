"""Keep each sample's normalized advantage fixed throughout one PPO update."""
import numpy as np
from touhou_ai.separate_clip_ppo import SeparateClipPPO

class RolloutAdvantagePPO(SeparateClipPPO):
    def train(self):
        b=self.rollout_buffer
        original=b.advantages
        flag=self.normalize_advantage
        if flag and original.size>1:
            b.advantages=(original-original.mean())/(original.std(ddof=1)+1e-8)
            self.normalize_advantage=False
        try:
            super().train()
        finally:
            b.advantages=original
            self.normalize_advantage=flag

def build(seed):
    from touhou_ai.autumn_training import build_model
    model,cls,config=build_model('cnn',seed)
    model.__class__=RolloutAdvantagePPO
    return model,cls,config
