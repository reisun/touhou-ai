"""Narrower grid CNN; identical spatial resolution and 256-feature interface."""
from torch import nn
from touhou_ai.dual_grid import DualGridFeatures,LOCAL_CHANNELS,GLOBAL_CHANNELS

class NarrowGridFeatures(DualGridFeatures):
    def __init__(self,observation_space):
        super().__init__(observation_space)
        self.local=nn.Sequential(nn.Conv2d(len(LOCAL_CHANNELS),4,3,padding=1),nn.ReLU(),
            nn.Conv2d(4,4,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(4,4,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(4,4,3,stride=2,padding=1),nn.ReLU(),
            nn.Flatten(),nn.Linear(4*12*12,128),nn.ReLU())
        self.global_scene=nn.Sequential(nn.Conv2d(len(GLOBAL_CHANNELS),4,3,padding=1),nn.ReLU(),
            nn.Conv2d(4,4,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(4,4,3,stride=2,padding=1),nn.ReLU(),
            nn.Flatten(),nn.Linear(4*14*12,128),nn.ReLU())
