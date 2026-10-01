"""Check whether greedy results require changing observations."""
import functools,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from touhou_ai.separate_clip_ppo import SeparateClipPPO
from touhou_ai.scaled_candidate import ScaledCandidateEnv
from scripts.assess_autumn_holdout import assess_holdout
torch.set_num_threads(1)
root=pathlib.Path('artifacts/critic-normalization-20260930')
path=root/'frozen-greedy.json'
report=json.loads(path.read_text()) if path.exists() else {}
cls=functools.partial(ScaledCandidateEnv,variant='action_grid',death_reward=-1)
for variant in ('control','global','pre_tanh'):
    for seed in (7,17,27):
        key=f'{variant}/{seed}';folder=root/key
        if key in report or not (folder/'model.zip').exists():continue
        model=SeparateClipPPO.load(folder/'model.zip',device='cpu')
        assert model.num_timesteps==16384
        report[key]=assess_holdout(model,cls,n=96,mode='frozen_greedy')
        path.write_text(json.dumps(report,indent=2))
        print(key,report[key]['survival'],flush=True)
