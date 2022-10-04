from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 200
        time_embed_dim = 1
        self.layer1 = nn.Linear(dim + time_embed_dim,dim)
        self.layer2 = nn.Linear(dim,dim)
        self.layer3 = nn.Linear(dim,dim)
        self.layer4 = nn.Linear(dim,dim)
        self.layer5 = nn.Linear(dim,dim)

    def forward(self, x, timesteps):
        timesteps = timesteps.unsqueeze(1)

        x = th.cat((x,timesteps), dim=1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer5(x)

        return x

