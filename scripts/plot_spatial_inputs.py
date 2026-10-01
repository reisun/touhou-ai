"""Show actual candidate tensors from a common diagnostic scene."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.validate_spatial_inputs import scenes
from scripts.compare_spatial_inputs import OUT
from touhou_ai.spatial_input_candidates import CandidateEncoder,risk_grid
from touhou_ai.dual_grid import DualGridContract

data=scenes()
raw,state=max((item for item in data if item[1]['frame']<=584),
              key=lambda item: min(risk_grid(item[0])[:2].sum(),18-risk_grid(item[0])[:2].sum()))
base=DualGridContract().encode(raw)
fine=CandidateEncoder('pixel1').encode(raw)
geom=CandidateEncoder('geometry').encode(raw)
risk=risk_grid(raw)
fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
for ax,array,title in zip(axes[0],
        (base['local_grid'][1],fine['local_grid'][1],geom['local_grid'][6]),
        ('Baseline: 2px cells','Candidate 1: 1px cells','Candidate 3: expanded danger region')):
    ax.imshow(array,extent=(-96,96,96,-96),cmap='magma',vmin=0,vmax=1,interpolation='nearest')
    ax.set(xlim=(-24,24),ylim=(24,-24),title=title,xlabel='x relative to player (px)',ylabel='y relative to player (px)')
    ax.plot(0,0,'+',color='cyan',ms=10)
for ax,array,title in zip(axes[1,:2],risk[:2],('Candidate 2: fast, next 2F','Candidate 2: slow, next 2F')):
    ax.imshow(array,cmap='Reds',vmin=0,vmax=1)
    for y in range(3):
        for x in range(3):ax.text(x,y,str(int(array[y,x])),ha='center',va='center',color='black' if array[y,x]==0 else 'white',fontsize=18)
    ax.set(xticks=[0,1,2],xticklabels=['Left','Center','Right'],yticks=[0,1,2],yticklabels=['Up','Center','Down'],title=title)
ax=axes[1,2]
rgb=np.zeros((96,96,3));rgb[:,:,0]=geom['local_grid'][6]
rgb[:,:,1]=np.maximum(geom['local_grid'][11],geom['local_grid'][9])
rgb[:,:,2]=np.maximum(geom['local_grid'][12],geom['local_grid'][10])
ax.imshow(rgb,extent=(-96,96,96,-96),interpolation='nearest')
ax.set(xlim=(-16,16),ylim=(16,-16),title='Candidate 3: fast (green), slow (blue) paths',xlabel='x relative to player (px)',ylabel='y relative to player (px)')
fig.suptitle(f'Actual input tensors, frame {state["frame"]}; current bullets shown\nPrediction inputs are estimates from current observations; red tiles mean predicted collision')
fig.savefig(OUT/'input-example.png',dpi=150)
print(OUT/'input-example.png')
