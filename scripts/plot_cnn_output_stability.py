import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.replicate_cnn_output_stability import OUT

def make_plot(s):
 fig,axes=plt.subplots(1,2,figsize=(9,4.8),sharey=True)
 limit=max(r[v]['survival']*100 for key in ['primary_fresh','exploratory_original'] for r in s[key]['pairs'] for v in ['independent','joint'])+7
 for ax,title,key in zip(axes,['Original 3 seeds (discovery)','Additional 7 seeds (replication)'],['exploratory_original','primary_fresh']):
  labels={}
  for r in s[key]['pairs']:
   y=[r['independent']['survival']*100,r['joint']['survival']*100]
   ax.plot([0,1],y,color='#9ca3af',lw=1,alpha=.7);ax.scatter([0,1],y,c=['#2563eb','#ea580c'],s=35,zorder=3)
   labels.setdefault(y[0],[]).append(str(r['seed']))
  last=-100
  for y,names in sorted(labels.items()):
   ly=max(y,last+1.5);last=ly
   ax.annotate(', '.join(names),xy=(0,y),xytext=(-.07,ly),ha='right',va='center',fontsize=8,arrowprops={'arrowstyle':'-','color':'#bbb','lw':.6})
  ax.set_xticks([0,1],['Independent 9+2','Joint 18']);ax.set_xlim(-.3,1.2);ax.set_ylim(0,limit);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('10-second survival (%)');fig.suptitle('Small CNN: each dot is one trained model, evaluated on 192 episodes');fig.tight_layout();fig.savefig(OUT/'stability.png',dpi=160);plt.close(fig)
if __name__=='__main__':make_plot(json.loads((OUT/'summary.json').read_text()))
