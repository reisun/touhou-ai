"""Train from scratch using the adopted offline baseline configuration."""
import sys,pathlib,argparse,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from touhou_ai.autumn_training import build_model
from touhou_ai.simulation_speed import tune_cpu
from scripts.continue_dodge_comparison import Progress
from scripts.assess_autumn_holdout import assess_holdout

def main():
    ap=argparse.ArgumentParser();ap.add_argument('representation',choices=['relative','cnn']);ap.add_argument('output',type=pathlib.Path);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--steps',type=int,default=16384);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model,cls,config=build_model(args.representation,args.seed)
    (args.output/'manifest.json').write_text(json.dumps({'configuration':config,'representation':args.representation,'seed':args.seed,'steps':args.steps},ensure_ascii=False,indent=2),encoding='utf-8')
    tune_cpu(model);model.learn(args.steps,callback=Progress());model.save(args.output/'model')
    results={mode:assess_holdout(model,cls,mode=mode) for mode in ['sample','greedy','frozen_sample','frozen_greedy']}
    (args.output/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
