"""Current adopted offline Autumn Sky settings; historical runners remain fixed."""
import functools,json
from pathlib import Path
from stable_baselines3 import PPO
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.autumn_ablation import SpeedPolicy,NumericalAblation,GridAblation
from touhou_ai.narrow_grid import NarrowGridFeatures
from touhou_ai.dual_grid import GridRolloutBuffer

CONFIG=Path(__file__).resolve().parents[1]/'configs/autumn-learning-baseline.json'
REFERENCE_CONFIG=CONFIG.with_name('autumn-learning-reference-v2.json')

def _build_standard(config,representation,seed):
    setting=config['representations'][representation]
    cls=functools.partial(NumericalAblation,relative=True) if representation=='relative' else GridAblation
    kw={'share_features_extractor':False,'net_arch':{'pi':[256,128],'vf':[256,128]}}
    if representation=='cnn':kw['features_extractor_class']=NarrowGridFeatures
    algorithm={'PPO':PPO,'SeparateClipPPO':SeparateClipPPO}[setting['algorithm']]
    model=algorithm(SpeedPolicy,cls(),seed=seed,device='cpu',rollout_buffer_class=GridRolloutBuffer,policy_kwargs=kw,verbose=0,**config['ppo'])
    return model,cls,config

def build_reference_model(representation,seed=7):
    """Frozen v2 control for historical comparisons."""
    return _build_standard(json.loads(REFERENCE_CONFIG.read_text(encoding='utf-8')),representation,seed)

def _build_configured(config,representation,seed):
    setting=config['representations'][representation]
    variant=setting.get('observation','dual_grid')
    if variant not in ('dual_grid','action_grid'):raise ValueError('Unsupported adopted observation: '+variant)
    model,cls,_=_build_standard(config,representation,seed)
    if representation=='cnn' and variant=='action_grid':
        from touhou_ai.spatial_input_candidates import build_candidate_model
        model,cls=build_candidate_model('action_grid',seed,model,config)
        death=setting.get('reward_death',config.get('reward_death',-60))
        if death!=-60:
            from touhou_ai.scaled_candidate import ScaledCandidateEnv
            cls=functools.partial(ScaledCandidateEnv,variant=variant,death_reward=death)
            model.set_env(cls())
            model.env.seed(config.get('training_environment_initial_seed',seed))
    return model,cls,config

def build_spatial_reference_model(representation,seed=7):
    """Frozen v3 for the completed reward/input comparison."""
    return _build_configured(json.loads(CONFIG.with_name('autumn-learning-reference-v3.json').read_text(encoding='utf-8')),representation,seed)

def build_model(representation,seed=7):
    """Fresh model using the current adopted offline contract and reward."""
    return _build_configured(json.loads(CONFIG.read_text(encoding='utf-8')),representation,seed)
