import sys,pathlib,json,numpy as np,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from scripts.audit_observation_credit import collect
torch.set_num_threads(1)
m=PPO.load('artifacts/autumn-reward-scale-20260928/relative-switch-7.zip',device='cpu')
_,rows=collect(m)
v=np.array([r['value']*60 for r in rows]);ret=np.array([r['mc_return'] for r in rows])
report={'model':'relative-switch-7, death reward -1','comparison_units':'value multiplied by60; realized returns from equivalent physical environment with death -60','episodes':16,'death_episodes':len({r['episode'] for r in rows if r['death']}),'mean_value_in_original_units':float(v.mean()),'mean_realized_return_in_original_units':float(ret.mean()),'mae':float(abs(v-ret).mean()),'realized_return_std':float(ret.std()),'value_std':float(v.std()),'constant_mean_predictor_mae':float(abs(ret-ret.mean()).mean()),'note':'on-policy diagnostic; different trajectories from baseline; not paired state value comparison; constant predictor fitted on evaluation returns is descriptive, not a deployed predictor'}
pathlib.Path('artifacts/autumn-reward-scale-20260928/value-calibration.json').write_text(json.dumps(report,indent=2))
print(report)
