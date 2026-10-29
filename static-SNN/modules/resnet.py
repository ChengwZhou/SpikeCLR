# NON-SPIKING VERSION

import torch
import torch.nn as nn
import torchvision
# from torchsummary import summary

def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ImageNet-based structure.
    """
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False, exp=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.exp = exp
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # self.conv1 = conv3x3(3, 64, stride=1) # for cifar
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.fc = nn.Linear(256, num_classes)
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.fc = nn.Linear(512*self.exp, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        # # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion

        for i in range(1, len(strides)):
            layers.append(block(self.in_planes, planes, strides[i], downsample=None))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.maxpool(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if hasattr(self, 'layer4'):
            out = self.layer4(out)
        out = self.avg_pool(out)
        # out = out.view(out.size(0), -1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet14(num_classes):
    return ResNet(BasicBlock, [2, 2, 2], num_classes, 1)


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, exp=1)


def resnet20(num_classes):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, exp=1)


def resnet32(num_classes):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, exp=1)


def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, exp=1)

def resnet50(num_classes):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, exp=4)

def resnet101(num_classes):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, exp=4)


def get_resnet(name, num_classes, pretrained=False):
    resnets = {
        # "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        # "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        # "resnet50": torchvision.models.resnet50(pretrained=pretrained),

        "resnet14": resnet14(num_classes),
        "resnet18": resnet18(num_classes),
        "resnet20": resnet20(num_classes),
        "resnet32": resnet32(num_classes),
        "resnet34": resnet34(num_classes),
        "resnet50": resnet50(num_classes),
        "resnet101": resnet101(num_classes),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model1 = get_resnet('resnet18', num_classes=10)
    model2 = torchvision.models.resnet18()
    # model2.conv1 =nn.Conv2d(3,64,3,1,1,bias=False)
    # model2.maxpool = nn.Identity()
    model1.fc = nn.Identity()
    model2.fc = nn.Identity()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model1 = model1.to(device)
    # model2 = model2.to(device)
    # print(f"Custom ResNet18: {count_parameters(model1)} parameters")
    # print(f"torchvision ResNet18: {count_parameters(model2)} parameters")
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    print("ResNet18_custom state_dict keys:\n")
    for k in state_dict1.keys():
        print(k)

    print("ResNet18_torcvision state_dict keys:\n")
    for k in state_dict2.keys():
        print(k)

    # summary(model1, (3, 32, 32))
    # summary(model2, (3, 32, 32))