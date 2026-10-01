import pathlib,json,functools,torch
from stable_baselines3 import PPO
from touhou_ai.autumn_ablation import NumericalAblation
from touhou_ai.direct_risk import DirectRiskInput
from scripts.assess_autumn_holdout import assess_holdout
out=pathlib.Path('artifacts/autumn-direct-risk-20260928/relative');rows=[]
torch.set_num_threads(1)
for seed in [7,17,27]:
 m=PPO.load(out/f'direct-{seed}',device='cpu')
 e=assess_holdout(m,functools.partial(DirectRiskInput,functools.partial(NumericalAblation,relative=True),enabled=False),mode='greedy')
 rows.append({'seed':seed,'zero_risk_greedy':e});print(seed,e['survival'],e['mean_frames']/60,flush=True)
 (out/'zero-risk-greedy.json').write_text(json.dumps(rows,indent=2))
