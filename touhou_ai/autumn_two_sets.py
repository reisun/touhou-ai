"""Two A/B emission cycles; success only after all four volleys leave."""
import numpy as np
from touhou_ai.autumn_sky import AutumnSky,AutumnNumerical
class TwoSets:
 horizon=1800
 def reset(self,**kwargs):
  self.completed=False;self.clear_frames=[None,None]
  return super().reset(**kwargs)
 def _emit(self):
  if len(self.emissions)<4:super()._emit()
 def step(self,action):
  if self.completed:raise RuntimeError('reset required')
  obs,reward,done,trunc,info=super().step(action)
  for i in range(2):
   if len(self.emissions)>=(i+1)*2 and not np.any(self.alive[i*304:(i+1)*304]) and self.clear_frames[i] is None:
    if not self.dead:self.clear_frames[i]=self.frame
  self.completed=not self.dead and all(f is not None for f in self.clear_frames)
  timeout=self.frame>=self.horizon and not self.completed and not self.dead
  info.update(success=self.completed,sets_cleared=sum(f is not None for f in self.clear_frames),clear_frames=self.clear_frames.copy(),timeout=timeout)
  return obs,reward,self.dead or self.completed,timeout,info
class TwoSetsGrid(TwoSets,AutumnSky):pass
class TwoSetsNumerical(TwoSets,AutumnNumerical):pass
