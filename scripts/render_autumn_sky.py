import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from touhou_ai.autumn_sky import AutumnNumerical

def main():
 e=AutumnNumerical();e.reset(seed=1000);original=e._frame_velocity
 def ghost():
  ids,active=original();return ids,np.zeros_like(active)
 e._frame_velocity=ghost
 images=[];panels=[]
 for t in range(301):
  if e.frame%10==0:
   im=Image.new('RGB',(408,500),'#101722');d=ImageDraw.Draw(im)
   d.rectangle((12,36,396,484),outline='#64748b')
   d.text((12,8),f'Autumn Sky / Normal / {e.frame/60:.2f}s',fill='white')
   for i in np.flatnonzero(e.alive):
    x,y=e.xy[i]+[204,36]
    if 12<x<396 and 36<y<484:
     c='#ff7777' if e.group[i]%2==0 else '#72a7ff';d.ellipse((x-3,y-3,x+3,y+3),fill=c)
   x,y=e.boss+[204,36];d.ellipse((x-7,y-7,x+7,y+7),outline='#fbd38d',width=2)
   d.text((12,488),'Pattern preview; collisions disabled',fill='#cbd5e1')
   images.append(im)
   if e.frame in [100,200,300,400,500,600]:panels.append(im)
  if t<300:e.step([0,0,0,0])
 out=pathlib.Path('artifacts/autumn-sky-source')
 images[0].save(out/'pattern.gif',save_all=True,append_images=images[1:],duration=167,loop=0)
 sheet=Image.new('RGB',(408*3,500*2))
 for i,im in enumerate(panels):sheet.paste(im,((i%3)*408,(i//3)*500))
 sheet.save(out/'pattern.png')
 print(out/'pattern.png')
if __name__=='__main__':main()
