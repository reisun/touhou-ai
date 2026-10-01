"""Offline single-factor critic normalization; no actor or default changes."""
from torch import nn
from touhou_ai.autumn_ablation import SpeedPolicy

VARIANTS = ('control', 'global', 'pre_tanh')

def install(policy, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if policy.share_features_extractor:
        raise ValueError('Critic-only experiment requires separate extractors')
    if variant == 'global':
        layers = policy.vf_features_extractor.global_scene
        assert isinstance(layers[-1], nn.ReLU)
        layers[-1] = nn.Sequential(layers[-1], nn.LayerNorm(128, elementwise_affine=False))
    elif variant == 'pre_tanh':
        layers = policy.mlp_extractor.value_net
        assert isinstance(layers[-1], nn.Tanh)
        layers[-1] = nn.Sequential(nn.LayerNorm(128, elementwise_affine=False), layers[-1])

class NormalizedCriticPolicy(SpeedPolicy):
    def __init__(self, *args, norm_variant='control', **kwargs):
        self.norm_variant = norm_variant
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        install(self, self.norm_variant)

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | {'norm_variant': self.norm_variant}

def build(variant, seed):
    from touhou_ai.autumn_training import build_model
    model, cls, config = build_model('cnn', seed)
    # No new parameters/random draws: preserve all initial tensors and RNG.
    model.policy.__class__ = NormalizedCriticPolicy
    model.policy.norm_variant = variant
    install(model.policy, variant)
    model.policy_class = NormalizedCriticPolicy
    model.policy_kwargs = dict(model.policy_kwargs, norm_variant=variant)
    return model, cls, config
