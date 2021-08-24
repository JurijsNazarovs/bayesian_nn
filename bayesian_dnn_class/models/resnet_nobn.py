'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")
import bayes_layers as bl
import importlib
importlib.reload(bl)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **bayes_args):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.Identity(in_planes)
        self.conv1 = bl.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False,
                               **bayes_args)
        self.bn2 = nn.Identity(planes)
        self.conv2 = bl.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,
                               **bayes_args)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = bl.Conv2d(in_planes,
                                      self.expansion * planes,
                                      kernel_size=1,
                                      stride=stride,
                                      bias=False,
                                      **bayes_args)

    def forward(self, x):
        kl = 0

        out = F.relu(self.bn1(x))
        if hasattr(self, 'shortcut'):
            shortcut, kl_ = self.shortcut(out)
            kl += kl_
        else:
            shortcut = x

        out, kl_ = self.conv1(out)
        kl += kl_
        out, kl_ = self.conv2(F.relu(self.bn2(out)))
        kl += kl_

        out += shortcut
        return out, kl


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, **bayes_args):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.Identity(in_planes)
        self.conv1 = bl.Conv2d(in_planes,
                               planes,
                               kernel_size=1,
                               bias=False,
                               **bayes_args)
        self.bn2 = nn.Identity(planes)
        self.conv2 = bl.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False,
                               **bayes_args)
        self.bn3 = nn.Identity(planes)
        self.conv3 = bl.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False,
                               **bayes_args)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = bl.Conv2d(in_planes,
                                      self.expansion * planes,
                                      kernel_size=1,
                                      stride=stride,
                                      bias=False,
                                      **bayes_args)

    def forward(self, x):
        kl = 0

        out = F.relu(self.bn1(x))
        if hasattr(self, 'shortcut'):
            shortcut, kl_ = self.shortcut(out)
            kl += kl_
        else:
            shortcut = x

        out, kl_ = self.conv1(out)
        kl += kl_
        out, kl_ = self.conv2(F.relu(self.bn2(out)))
        kl += kl_
        out, kl_ = self.conv3(F.relu(self.bn3(out)))
        kl += kl_

        out += shortcut
        return out, kl


class PreActResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 num_classes=10,
                 in_channels=3,
                 **bayes_args):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = bl.Conv2d(in_channels,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,
                               **bayes_args)
        self.layer1 = self._make_layer(block,
                                       64,
                                       num_blocks[0],
                                       stride=1,
                                       **bayes_args)
        self.layer2 = self._make_layer(block,
                                       128,
                                       num_blocks[1],
                                       stride=2,
                                       **bayes_args)
        self.layer3 = self._make_layer(block,
                                       256,
                                       num_blocks[2],
                                       stride=2,
                                       **bayes_args)
        self.layer4 = self._make_layer(block,
                                       512,
                                       num_blocks[3],
                                       stride=2,
                                       **bayes_args)
        self.linear = bl.Linear(512 * block.expansion, num_classes,
                                **bayes_args)

    def _make_layer(self, block, planes, num_blocks, stride, **bayes_args):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **bayes_args))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        kl = 0
        out, kl_ = self.conv1(x)
        kl += kl_

        for layer_iter in range(1, 5):
            layer = eval("self.layer%d" % layer_iter)
            if isinstance(layer, nn.Sequential):
                for i in range(len(layer)):
                    out, kl_ = layer[i](out)
                    kl += kl_
            else:
                out, kl_ = layer(out)
                kl += kl_

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out, kl_ = self.linear(out)
        kl += kl_
        return out, kl


def PreActResNet18(**kwargs):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], **kwargs)


def PreActResNet34(**kwargs):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], **kwargs)


def PreActResNet50(**kwargs):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], **kwargs)


def PreActResNet101(**kwargs):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], **kwargs)


def PreActResNet152(**kwargs):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], **kwargs)
