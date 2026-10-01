"""All observed bullets, player-relative coordinates ordered by distance; no truncation."""
import numpy as np
from touhou_ai.autumn_two_sets import TwoSetsNumerical
from touhou_ai.autumn_sky import MAX_BULLETS
class RelativeTwoSetsNumerical(TwoSetsNumerical):
 def observe(self):
  state=np.zeros(5+MAX_BULLETS*5,dtype=np.float32);state[:5]=[self.pos[0]/192,self.pos[1]/448,0,0,self.focus]
  ids=np.flatnonzero(self.alive);relative=self.xy[ids]-self.pos
  order=np.argsort(np.sum(relative**2,axis=1),kind='stable');ids=ids[order]
  slots=state[5:].reshape(-1,5);slots[:len(ids),:2]=(self.xy[ids]-self.pos)/[192,448]
  slots[:len(ids),2:4]=self.vel[ids]/4.5;slots[:len(ids),4]=1
  return {'state':state,'bomb_clock':np.zeros(1,dtype=np.float32)}
