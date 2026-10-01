"""Offline affine decomposition on identical recorded boss observations."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from touhou_ai.checkpoint_rng import load_preserving_rng
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.live_action_grid import LiveActionGridContract

ROOT = pathlib.Path(__file__).resolve().parents[1]
study = json.loads((ROOT/'artifacts/shot-reward-study-20260930/study.json').read_text())
capture = json.loads((ROOT/'artifacts/boss-reward-tanh-20260930/capture.json').read_text())
torch.set_num_threads(1)
env = LiveActionGridContract()
observations = []
for row in capture['observations']:
    obs = env.encode(row['raw'], row['rewards'])
    if row['bomb_clock'] is not None:
        obs['bomb_clock'] = np.asarray([((round(row['bomb_clock'][0]*12)+2)%12)/12], np.float32)
    observations.append(obs)

def rms(a): return float(np.sqrt(np.mean(np.square(a))))
def sat(a): return float(np.mean(np.abs(np.tanh(a)) > .99))
def stats(a):
    return dict(rms=rms(a), abs_p50=float(np.quantile(abs(a), .5)),
                abs_p95=float(np.quantile(abs(a), .95)), abs_max=float(abs(a).max()))

paths = {'common': study['source']}
for name, arm in study['arms'].items():
    paths[name] = str(pathlib.Path(arm['record']['Output'])/arm['status']['episodes'][-1]['checkpoint'])
report = dict(samples=len(observations), tanh_threshold=float(np.arctanh(.99)), models={})
for label, path in paths.items():
    model = load_preserving_rng(SeparateClipPPO, pathlib.Path(path), device='cpu')
    policy = model.policy
    policy.set_training_mode(False)
    captured, handles = {}, []
    def hook(name):
        def collect(module, args, value):
            captured.setdefault(name, []).append((args[0].detach().numpy().copy(), value.detach().numpy().copy()))
        return collect
    for name, module in policy.named_modules():
        if (name.startswith('vf_features_extractor') or name.startswith('mlp_extractor.value_net')) and isinstance(module, (torch.nn.Linear, torch.nn.ReLU, torch.nn.Tanh)):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.no_grad():
        for obs in observations:
            policy.forward_with_distribution(policy.obs_to_tensor(obs)[0])
    for handle in handles: handle.remove()
    result = dict(checkpoint=path, linear={}, tanh={}, activations={})
    modules = dict(policy.named_modules())
    for name, parts in captured.items():
        x = np.concatenate([p[0] for p in parts]).astype(np.float64)
        y = np.concatenate([p[1] for p in parts]).astype(np.float64)
        module = modules[name]
        result['activations'][name] = stats(y)
        if isinstance(module, torch.nn.Linear):
            w = module.weight.detach().numpy().astype(np.float64)
            b = module.bias.detach().numpy().astype(np.float64)
            wx = x @ w.T
            error = float(abs(wx+b-y).max())
            assert error < 1e-3, (name, error)
            mean = x.mean(axis=0)
            common = mean @ w.T + b
            variation = (x-mean) @ w.T
            result['linear'][name] = dict(input=stats(x), weight=stats(w), bias=stats(b),
                weight_row_norm_mean=float(np.linalg.norm(w, axis=1).mean()),
                weighted_input=stats(wx), output=stats(y), reconstruction_max_error=error,
                common_component=stats(common), scene_variation=stats(variation),
                saturation=sat(y), saturation_without_bias=sat(wx),
                saturation_centered_input=sat(variation+b),
                saturation_half_input=sat(.5*wx+b),
                saturation_unit_row_norm=sat(wx/np.linalg.norm(w,axis=1).clip(1e-12)+b))
            if name == 'vf_features_extractor.merge.0':
                groups = {'local':(0,128), 'global':(128,256), 'player':(256,288),
                          'previous_rewards':(288,320), 'bomb_clock':(320,321), 'action':(321,353)}
                result['merge_branch_contributions'] = {
                    key: stats(x[:,start:end] @ w[:,start:end].T)
                    for key,(start,end) in groups.items()}
        if isinstance(module, torch.nn.Tanh):
            saturated = abs(y) > .99
            fixed_positive = np.all(y > .99, axis=0)
            fixed_negative = np.all(y < -.99, axis=0)
            flip = (y.min(axis=0)<0)&(y.max(axis=0)>0)
            both_ends = (y.min(axis=0)<-.99)&(y.max(axis=0)>.99)
            result['tanh'][name] = dict(units=y.shape[1], saturation=float(saturated.mean()),
                fixed_positive=int(fixed_positive.sum()), fixed_negative=int(fixed_negative.sum()),
                sign_flip_units=int(flip.sum()), both_saturated_ends_units=int(both_ends.sum()),
                fixed_unit_share_of_saturated_samples=float((fixed_positive|fixed_negative).sum()*len(y)/saturated.sum()),
                std_below_001=int((y.std(axis=0)<.01).sum()),
                per_unit_std=y.std(axis=0).tolist(), per_unit_min=y.min(axis=0).tolist(), per_unit_max=y.max(axis=0).tolist())
    a=np.concatenate([p[1] for p in captured['mlp_extractor.value_net.3']]).astype(np.float64)
    fixed=np.all(a>.99,axis=0)|np.all(a<-.99,axis=0)
    head_w=policy.value_net.weight.detach().numpy().astype(np.float64).ravel()
    fixed_value=a[:,fixed]@head_w[fixed]
    other_value=a[:,~fixed]@head_w[~fixed]
    result['value_head_decomposition']=dict(
        fixed_units=int(fixed.sum()),fixed_contribution_mean=float(fixed_value.mean()),
        fixed_contribution_std=float(fixed_value.std()),
        other_contribution_mean=float(other_value.mean()),other_contribution_std=float(other_value.std()),
        head_bias=float(policy.value_net.bias.detach().item()),
        total_value_std=float((fixed_value+other_value).std()),
        caveat='Components can cancel; variance shares are not additive. Fixed means fixed on these 20 observations only.')
    report['models'][label] = result
out = ROOT/'artifacts/tanh-sources-20260930'
out.mkdir(exist_ok=True)
(out/'analysis.json').write_text(json.dumps(report, indent=2))
for label, result in report['models'].items():
    print(label)
    for name, data in result['linear'].items():
        print(name, json.dumps(data))
    for name, data in result['tanh'].items():
        print(name, json.dumps({k:v for k,v in data.items() if not k.startswith('per_unit')}))
