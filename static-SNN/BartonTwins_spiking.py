import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from modules.spike_layer import *


class BartonTwinsSpiking(nn.Module):
    def __init__(self, backbone, in_dim=512, out_dim=512, hidden_dim=2048, act_func=LIFt, timestep=4):
        super(BartonTwinsSpiking, self).__init__()

        self.timestep = timestep
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy'),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward_one(self, x):
        _, feature = self.backbone(x)  # [T, 2*B, C]

        T, B, D = feature.shape
        z = self.projector(feature.flatten(0, 1))
        z = z.reshape(T, B, -1).contiguous()
        return feature, z

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)
        f, z = self.forward_one(x)  # [T, 2*B, C], [T, 2*B, C]

        b_size = int(z.shape[1] // 2)
        f1 = f[:, :b_size, ...]
        f2 = f[:, b_size:, ...]
        z1 = z[:, :b_size, ...]
        z2 = z[:, b_size:, ...]

        return f1, f2, z1, z2


class BartonTwinsSpiking_imagenet(nn.Module):
    def __init__(self, backbone, in_dim=512, out_dim=8192, hidden_dim=8192, act_func=LIFt, timestep=4):
        super(BartonTwinsSpiking_imagenet, self).__init__()

        self.timestep = timestep
        self.backbone = backbone
        self.projector_1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.lif_1 = act_func(tau=4.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.projector_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.lif_2 = act_func(tau=4.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward_one(self, x):
        _, feature = self.backbone(x)  # [T, B, C]

        T, B, D = feature.shape
        z = self.lif_1(self.projector_1(feature.flatten(0, 1)))
        z = self.lif_2(self.projector_2(z))
        z = self.linear(z)
        z = z.reshape(T, B, -1).contiguous()
        return feature, z

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)
        f, z = self.forward_one(x)  # [T, 2*B, C], [T, 2*B, C]

        b_size = int(z.shape[1] // 2)
        f1 = f[:, :b_size, ...]
        f2 = f[:, b_size:, ...]
        z1 = z[:, :b_size, ...]
        z2 = z[:, b_size:, ...]

        return f1, f2, z1, z2
