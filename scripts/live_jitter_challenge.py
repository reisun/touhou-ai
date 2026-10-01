import sys,time,json,torch
from live_jitter_trial import OUT,build,evaluate,SeparateClipPPO
torch.set_num_threads(1)
if sys.argv[1]=='system':
    m=build(7,0)
    for h in (None,0,.05,.15):
        r=evaluate(m,32,hysteresis=h,start_y=160);(OUT/f'challenge-system-{h}.json').write_text(json.dumps(r))
else:
    seed=int(sys.argv[1])
    for p in ('0.0','0.001','0.003'):
        folder=OUT/f'train-{seed}-{p}'
        while not (folder/'evaluation.json').exists():time.sleep(2)
        m=SeparateClipPPO.load(folder/'model.zip',device='cpu')
        r=evaluate(m,32,start_y=160);(folder/'challenge.json').write_text(json.dumps(r))
