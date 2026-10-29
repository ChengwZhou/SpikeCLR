import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class BartonTwins(nn.Module):
    def __init__(self, backbone, in_dim=512, out_dim=512, hidden_dim=2048):
        super(BartonTwins, self).__init__()

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward_one(self, x):
        feature = self.backbone(x)
        z = self.projector(feature)
        return feature, z

    def forward(self, x1, x2):
        f1, z1 = self.forward_one(x1)
        f2, z2 = self.forward_one(x2)
        return f1, f2, z1, z2


class BartonTwins_imagenet(nn.Module):
    def __init__(self, backbone, in_dim=512, out_dim=8192, hidden_dim=8192):
        super(BartonTwins_imagenet, self).__init__()

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward_one(self, x):
        feature = self.backbone(x)
        z = self.projector(feature)
        return feature, z

    def forward(self, x1, x2):
        f1, z1 = self.forward_one(x1)
        f2, z2 = self.forward_one(x2)
        return f1, f2, z1, z2