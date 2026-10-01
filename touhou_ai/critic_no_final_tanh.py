"""Experimental policy: only the critic's final hidden Tanh is removed."""
from torch import nn
from touhou_ai.autumn_ablation import SpeedPolicy

class CriticNoFinalTanhPolicy(SpeedPolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        layers=self.mlp_extractor.value_net
        if not isinstance(layers[-1],nn.Tanh):
            raise ValueError('Expected final critic hidden Tanh')
        # Parameter-free substitution preserves initialization and optimizer links.
        layers[-1]=nn.Identity()
