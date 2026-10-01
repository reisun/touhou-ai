import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from touhou_ai.autumn_two_sets import TwoSetsNumerical
from touhou_ai.observed_avoidance import choose
out=pathlib.Path('artifacts/observed-reference-20260928-200540');e=TwoSetsNumerical();e.reset(seed=10000);images=[];positions=[];actions=[]
for step in range(900):
 a=choose(e.pos,e.xy[e.alive],e.vel[e.alive],4.);o,r,done,trunc,info=e.step(a);positions.append(e.pos.copy());actions.append(a.tolist())
 if step%3==0 or done or trunc:
  im=Image.new('RGB',(408,500),'#101722');d=ImageDraw.Draw(im);d.rectangle((12,36,396,484),outline='#64748b')
  d.text((12,8),f'Observed-only controller / {e.frame/60:.2f}s',fill='white')
  for i in np.flatnonzero(e.alive):
   x,y=e.xy[i]+[204,36]
   if 12<x<396 and 36<y<484:d.ellipse((x-2,y-2,x+2,y+2),fill='#ff7777' if e.group[i]%2==0 else '#72a7ff')
  x,y=e.pos+[204,36];d.ellipse((x-4,y-4,x+4,y+4),fill='#6bff9c');d.ellipse((x-1,y-1,x+1,y+1),fill='white')
  d.text((12,488),f"Sets cleared: {info['sets_cleared']}/2 / collisions ON",fill='#cbd5e1');images.append(im)
 if done or trunc:break
images[0].save(out/'reference.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
(out/'example.json').write_text(json.dumps({'seed':10000,**info,'x_range':[float(np.min(np.array(positions)[:,0])),float(np.max(np.array(positions)[:,0]))],'y_range':[float(np.min(np.array(positions)[:,1])),float(np.max(np.array(positions)[:,1]))],'actions':actions},indent=2))
print(info)
