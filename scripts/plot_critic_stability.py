import json,pathlib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path('artifacts/critic-stability-20260929')
report=json.loads((root/'summary.json').read_text(encoding='utf-8'))
fresh=report['primary_additional'];all_rows=report['secondary_all']['pairs'];x=np.arange(len(all_rows))
fig,axes=plt.subplots(1,2,figsize=(11,4.3),gridspec_kw={'width_ratios':[1.7,1]})
ax=axes[0]
for i,r in enumerate(all_rows):ax.plot([i,i],[100*r['control']['survival'],100*r['no_tanh']['survival']],color='#aab2be',lw=1.3,zorder=1)
for name,color,shift in [('control','#747c8b',0),('no_tanh','#087eac',0)]:
    ax.scatter(x+shift,[100*r[name]['survival'] for r in all_rows],color=color,label='Original Tanh' if name=='control' else 'Final critic Tanh removed',zorder=2)
ax.axvspan(-.5,2.5,color='#eaeef3',zorder=0)
ax.set_xticks(x,[str(r['seed']) for r in all_rows]);ax.set_xlabel('Training initialization seed (shaded: original 3)')
ax.set_ylabel('Survival to 600 frames (%)');ax.set_title('Paired models: 192 evaluation cases each');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
ax=axes[1]
for i,metric in enumerate(['mean','sd','minimum']):
    ax.bar(i-.18,100*fresh['control'][metric],width=.35,color='#747c8b')
    ax.bar(i+.18,100*fresh['no_tanh'][metric],width=.35,color='#087eac')
ax.set_xticks([0,1,2],['Mean','SD','Minimum']);ax.set_ylabel('Percent / percentage points');ax.set_title('Primary: 7 additional initializations')
ci=fresh['bootstrap']['sd_ratio_95'];ax.text(.02,.98,f"SD ratio: {fresh['sd_ratio']:.2f}\n95% bootstrap interval: [{ci[0]:.2f}, {ci[1]:.2f}]",transform=ax.transAxes,va='top',fontsize=9)
ax.set_ylim(0,max(fresh['control']['mean'],fresh['no_tanh']['mean'])*145+4);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(root/'stability.png',dpi=170);plt.close(fig)
