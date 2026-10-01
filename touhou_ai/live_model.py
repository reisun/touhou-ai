"""Explicit, resumable model choices for real-game evasion training."""
from stable_baselines3 import PPO
from touhou_ai.separate_clip_ppo import SeparateClipPPO

ALGORITHMS={'PPO':PPO,'SeparateClipPPO':SeparateClipPPO}
def algorithm_class(name):
    if name not in ALGORITHMS:raise ValueError('Unsupported learning algorithm: '+str(name))
    return ALGORITHMS[name]

def model_config(profile,evasion_only,dual_grid,prior=None):
    overrides=profile.get('evasion_policy_overrides' if evasion_only else 'full_policy_overrides',{})
    if not evasion_only and not dual_grid:
        overrides = {}
    result={'algorithm':overrides.get('algorithm','PPO'),'cnn_architecture':overrides.get('cnn_architecture','dual_grid' if dual_grid else 'legacy')}
    algorithm_class(result['algorithm'])
    if result['cnn_architecture'] not in ['dual_grid','narrow_grid','narrow_action_grid','legacy']:raise ValueError('Unknown CNN architecture')
    if result['cnn_architecture'] in ['narrow_grid','narrow_action_grid'] and not dual_grid:raise ValueError('Narrow CNN requires dual-grid mode')
    if result['cnn_architecture']=='narrow_action_grid' and result['algorithm']!='SeparateClipPPO':raise ValueError('Action grid requires SeparateClipPPO')
    if result['algorithm']=='SeparateClipPPO' and overrides.get('share_features_extractor',True):
        raise ValueError('Separate clipping requires separate actor/critic evasion features')
    if prior:
        old={'algorithm':prior.get('algorithm','PPO'),'cnn_architecture':prior.get('cnn_architecture','dual_grid' if prior.get('contract','').startswith('th10-dual-grid') else 'legacy')}
        if old!=result:raise ValueError('Model architecture/algorithm changed; start a fresh campaign')
    return result
