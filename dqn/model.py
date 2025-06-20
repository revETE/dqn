# -*- coding: utf-8 -*
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, d_action_space):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),  # 32x20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=1),  # 64x6x6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, d_action_space),
        )

    def forward(self, x):
        return self.network(x / 255.0)
