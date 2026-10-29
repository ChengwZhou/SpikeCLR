# SPIKING VERSION
from modules.spike_layer import *
from math import sqrt
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, timestep, sync_norm, act_func=LIF, modified=True):
        super(BasicBlock, self).__init__()

        self.timestep = timestep
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spike_func_1 = act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy')
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spike_func_2 = act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy')
        self.shortcut = nn.Sequential()
        self.modified = modified

        if stride != 1 or in_planes != planes:
            if self.modified:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes),
                    act_func(tau=4.0, v_threshold=0.5, detach_reset=True, backend='cupy')
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

    def forward(self, x):
        out = self.spike_func_1(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        if self.modified:
            out = self.spike_func_2(out)
            out += self.shortcut(x)         # Equivalent to union of all spikes
        else:
            out += self.shortcut(x)
            out = self.spike_func_2(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, timestep, sync_norm, act_func=LIF, modified=True):
        super(BottleNeck, self).__init__()

        self.timestep = timestep
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spike_func_1 = act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy')
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spike_func_2 = act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy')
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.spike_func_3 = act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy')
        self.shortcut = nn.Sequential()
        self.modified = modified

        if stride != 1 or in_planes != planes * self.expansion:
            if self.modified:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(planes * self.expansion),
                    act_func(tau=4.0, v_threshold=0.5, detach_reset=True, backend='cupy')
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(planes * self.expansion),
                )

    def forward(self, x):
        out = self.spike_func_1(self.bn1(self.conv1(x)))
        out = self.spike_func_2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.modified:
            out = self.spike_func_3(out)
            out += self.shortcut(x)         # Equivalent to union of all spikes
        else:
            out += self.shortcut(x)
            out = self.spike_func_3(out)
        return out


class BottleNeckRelu(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, timestep, sync_norm, act_func=LIF, modified=True):
        super(BottleNeckRelu, self).__init__()

        self.timestep = timestep
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.spike_func_1 = act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy')
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.spike_func_2 = act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy')
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # self.spike_func_3 = act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy')
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        self.modified = modified

        if stride != 1 or in_planes != planes * self.expansion:
            if self.modified:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(planes * self.expansion),
                    # act_func(tau=4.0, v_threshold=0.5, detach_reset=True, backend='cupy')
                    nn.ReLU(inplace=True)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(planes * self.expansion),
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class BLock_Layer(nn.Module):
    def __init__(self, block, in_planes, planes, num_block, timestep, sync_norm, downsample, act_func, modified):
        super(BLock_Layer, self).__init__()

        layers = []
        if downsample:
            layers.append(block(in_planes, planes, 2, timestep, sync_norm, act_func, modified))
        else:
            layers.append(block(in_planes, planes, 1, timestep, sync_norm, act_func, modified))
        for _ in range(1, num_block):
            layers.append(block(planes * block.expansion, planes, 1, timestep, sync_norm, act_func, modified))
        self.execute = nn.Sequential(*layers)

    def forward(self, x):
        return self.execute(x)


class ResNet(nn.Module):
    """ Establish ResNet.
     Spiking DS-ResNet with “modified=True.”
     Spiking ResNet with “modified=False.”
     """
    def __init__(self, block, num_block_layers, timestep, sync_norm, act_func=LIF, num_classes=10, exp=1):
        super(ResNet, self).__init__()

        self.timestep = timestep
        self.exp = exp
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # self.conv1 = conv3x3(3, 64, stride=1) # for cifar
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = BLock_Layer(block, 64, 64, num_block_layers[0], self.timestep, sync_norm, False, act_func=act_func, modified=True)
        self.layer2 = BLock_Layer(block, 64*self.exp, 128, num_block_layers[1], self.timestep, sync_norm, True, act_func=act_func, modified=True)
        self.layer3 = BLock_Layer(block, 128*self.exp, 256, num_block_layers[2], self.timestep, sync_norm, True, act_func=act_func, modified=True)
        self.fc = nn.Linear(256*self.exp, num_classes)
        if len(num_block_layers) == 4:
            self.layer4 = BLock_Layer(block, 256*self.exp, 512, num_block_layers[3], self.timestep, sync_norm, True, act_func=act_func, modified=True)
            self.fc = nn.Linear(512*self.exp, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spike_func = act_func(tau=4.0, v_threshold=0.5, timestep=timestep, detach_reset=True, backend='cupy')
        # self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # zero_init_residual = True
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, BottleNeck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x):
        # expand x to 4 time-steps
        B, C, H, W = x.shape
        x = (x.unsqueeze(0)).repeat(self.timestep, 1, 1, 1, 1)
        x = x.flatten(0, 1)

        out = self.spike_func(self.maxpool(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if hasattr(self, 'layer4'):
            out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)  # time augmented output

        out = out.reshape(self.timestep, B, -1)
        o = out.mean(0)
        return o, out  # [bs*2, D], [T, bs*2, D]


def resnet14(timestep, sync_norm, act_func, num_classes):
    return ResNet(BasicBlock, [2, 2, 2], timestep, sync_norm, act_func, num_classes, 1)


def resnet18(timestep, sync_norm, act_func, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], timestep, sync_norm, act_func, num_classes, 1)


def resnet20(timestep, sync_norm, act_func, num_classes):
    return ResNet(BasicBlock, [3, 3, 3], timestep, sync_norm, act_func, num_classes, 1)


def resnet32(timestep, sync_norm, act_func, num_classes):
    return ResNet(BasicBlock, [5, 5, 5], timestep, sync_norm, act_func, num_classes, 1)


def resnet34(timestep, sync_norm, act_func, num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], timestep, sync_norm, act_func, num_classes, 1)

def resnet50(timestep, sync_norm, act_func, num_classes):
    return ResNet(BottleNeck, [3, 4, 6, 3], timestep, sync_norm, act_func, num_classes, 4)

def resnet101(timestep, sync_norm, act_func, num_classes):
    return ResNet(BottleNeck, [3, 4, 23, 3], timestep, sync_norm, act_func, num_classes, 4)


def get_resnet_spiking(name, timestep, sync_norm, act_func, num_classes):
    if name == "resnet14":
        return resnet14(timestep, sync_norm, act_func, num_classes)
    elif name == "resnet18":
        return resnet18(timestep, sync_norm, act_func, num_classes)
    elif name == "resnet20":
        return resnet20(timestep, sync_norm, act_func, num_classes)
    elif name == "resnet32":
        return resnet32(timestep, sync_norm, act_func, num_classes)
    elif name == "resnet34":
        return resnet34(timestep, sync_norm, act_func, num_classes)
    elif name == "resnet50":
        return resnet50(timestep, sync_norm, act_func, num_classes)
    elif name == "resnet101":
        return resnet101(timestep, sync_norm, act_func, num_classes)