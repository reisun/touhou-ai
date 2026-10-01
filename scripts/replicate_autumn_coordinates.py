import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.compare_autumn_ablation import main
if __name__=='__main__':
 main([(rep,speed,seed) for speed in ['switch','fast'] for seed in [17,27] for rep in ['absolute','relative']])
