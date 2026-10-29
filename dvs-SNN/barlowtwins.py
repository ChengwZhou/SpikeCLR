import torch
import torch.nn as nn
from timm.models.registry import register_model
from spiking_layer import MixedLIF, LIFt, LIF
from model import get_resnet_spiking

# Output feature dimension after AdaptiveAvgPool for each backbone variant.
# resnet14/20/32 have 3 stages (no layer4) → 256 channels.
# resnet18/34     have 4 stages (BasicBlock)  → 512 channels.
# resnet50/101    have 4 stages (BottleNeck, expansion=4) → 2048 channels.
BACKBONE_OUT_DIM = {
    'resnet14':  256,
    'resnet18':  512,
    'resnet20':  256,
    'resnet32':  256,
    'resnet34':  512,
    'resnet50':  2048,
    'resnet101': 2048,
}


class BarlowTwins(nn.Module):
    def __init__(self, backbone, act_func_lif=LIFt, in_dim=256, out_dim=1024, hidden_dim=2048, T=4):
        super(BarlowTwins, self).__init__()

        self.T = T
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act_func_lif(tau=4.0, v_threshold=0.5, detach_reset=True, backend='cupy'),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward_one(self, x):
        _, feature = self.backbone(x)
        T, B, C = feature.shape
        z = self.projector(feature.flatten(0, 1)).reshape(T, B, -1).contiguous()
        return feature, z

    def forward(self, x1, x2):
        # print(x1.size())  # torch.Size([128, 10, 2, 128, 128])
        x = torch.cat((x1, x2), dim=0)
        f, z = self.forward_one(x)

        T, B, C = f.shape
        b = int(B//2)
        # fetch each path
        f1 = f[:, :b, ...]
        f2 = f[:, b:, ...]
        z1 = z[:, :b, ...]
        z2 = z[:, b:, ...]
        return f1, f2, z1, z2

@register_model
def barlow_twins(act_func, name='resnet18', **kwargs):
    act_func_map = {'MixedLIF': MixedLIF, 'LIFt': LIFt, 'LIF': LIF}
    act_func_cls = act_func_map.get(act_func, LIFt)

    if name not in BACKBONE_OUT_DIM:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(BACKBONE_OUT_DIM)}")

    backbone = get_resnet_spiking(
        name=name,
        timestep=kwargs['timestep'],
        sync_norm=kwargs.get('sync_norm', True),
        act_func_lif=act_func_cls,
        num_classes=kwargs.get('num_classes', 10),
    )
    backbone.fc = nn.Identity()
    model = BarlowTwins(
        backbone=backbone,
        act_func_lif=act_func_cls,
        in_dim=BACKBONE_OUT_DIM[name],
        T=kwargs['timestep'],
    )
    return model
