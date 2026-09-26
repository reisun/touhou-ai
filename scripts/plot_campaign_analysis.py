"""Plot already aggregated learning records; never reads or changes a model."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(path):
    data=json.loads(path.read_text());rows=data['episodes'];x=np.array([r['episode'] for r in rows])
    series=[('Controlled play time (seconds)',[r['seconds'] for r in rows]),
        ('Time to first hit (seconds)',[r['first_hit'] for r in rows]),
        ('Damage per controlled second',[r['counts'].get('damage',0)/r['seconds'] for r in rows]),
        ('Mean power within each play',[r['power_sum']/r['player_n'] for r in rows]),
        ('Episode return',[r['return'] for r in rows]),
        ('PPO approximate KL',[r['optimizer']['train/approx_kl'] for r in rows])]
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for ax,(title,values) in zip(axes.flat,series):
        y=np.array(values,dtype=float);rolling=np.convolve(y,np.ones(20)/20,mode='valid')
        ax.plot(x,y,color='#b5bec9',linewidth=.7,alpha=.65,label='Each play')
        ax.plot(x[19:],rolling,color='#126a8a',linewidth=2,label='20-play mean')
        ax.set_title(title);ax.set_xlabel('Completed play / update');ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Focused model: 381 training plays\nChanging policies, not fixed-checkpoint evaluations',fontsize=15)
    fig.savefig(path.parent/'learning-trends.png',dpi=160);plt.close(fig)
    early=data['windows']['1-50']['reward'];late=data['windows']['332-381']['reward']
    keys=['damage','kill','stage_clear','hit','power','low_power','invalid_bomb']
    y=np.arange(len(keys));fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    ax.barh(y-.18,[early[k] for k in keys],height=.35,label='First 50 plays',color='#8695a8')
    ax.barh(y+.18,[late[k] for k in keys],height=.35,label='Last 50 plays',color='#126a8a')
    ax.set_yticks(y,keys);ax.axvline(0,color='#444',linewidth=.8);ax.set_xlabel('Mean reward per play')
    ax.set_title('What changed in reward?');ax.legend(frameon=False);ax.grid(axis='x',alpha=.2)
    fig.savefig(path.parent/'reward-components.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('analysis',type=Path);args=parser.parse_args();plot(args.analysis)
