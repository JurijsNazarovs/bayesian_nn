import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
#from torch.jit.annotations import List

__all__ = [
    'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161'
]

import sys
sys.path.append("../")
import bayes_layers as bl


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 **bayes_args):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            bl.Conv2d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False,
                      **bayes_args)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv2',
            bl.Conv2d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False,
                      **bayes_args)),
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        # import pdb
        # pdb.set_trace()

        concated_features = torch.cat(inputs, 1)
        bottleneck_output, kl = self.conv1(
            self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output, kl

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # @torch.jit.unused  # noqa: T484
    # def call_checkpoint_bottleneck(self, input):
    #     # type: (List[Tensor]) -> Tensor
    #     def closure(*inputs):
    #         return self.bn_function(*inputs)

    #     return cp.checkpoint(closure, input)

    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input):
    #     # type: (List[Tensor]) -> (Tensor)
    #     pass

    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input):
    #     # type: (Tensor) -> (Tensor)
    #     pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        kl = 0
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output, kl_ = self.bn_function(prev_features)

        kl += kl_
        new_features, kl_ = self.conv2(
            self.relu2(self.norm2(bottleneck_output)))
        kl += kl_
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return new_features, kl


class _DenseBlock(nn.Module):
    _version = 2
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, **bayes_args):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                **bayes_args)
            self.layers['denselayer%d' % (i + 1)] = layer

    def forward(self, init_features):
        # import pdb
        # pdb.set_trace()
        kl = 0

        features = [init_features]
        for name, layer in self.layers.items():
            # import pdb
            # pdb.set_trace()

            new_features, kl_ = layer(features)
            kl += kl_
            features.append(new_features)
        return torch.cat(features, 1), kl


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, **bayes_args):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            bl.Conv2d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False,
                      **bayes_args))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        kl = 0
        for layer in self:

            tmp = layer(x)
            if isinstance(tmp, tuple):
                x, kl_ = tmp
                kl += kl_
            else:
                x = tmp

        return x, kl


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first
        convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    __constants__ = ['features']

    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 in_channels=3,
                 num_classes=1000,
                 **bayes_args):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 bl.Conv2d(in_channels,
                           num_init_features,
                           kernel_size=7,
                           stride=2,
                           padding=3,
                           bias=False,
                           **bayes_args)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                **bayes_args)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    **bayes_args)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = bl.Linear(num_features, num_classes, **bayes_args)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        kl = 0
        for layer in self.features:
            tmp = layer(x)
            if isinstance(tmp, tuple):
                x, kl_ = tmp
                kl += kl_
            else:
                x = tmp

        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        x, kl_ = self.classifier(x)
        kl += kl_
        return x, kl


def _densenet(arch, growth_rate, block_config, num_init_features, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        progress (bool): If True, displays a progress bar of the download to 
        stderr
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, progress,
                     **kwargs)


def densenet161(progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks"
 <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        progress (bool): If True, displays a progress bar of the download to
        stder
     """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, progress,
                     **kwargs)


def densenet169(progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        progress (bool): If True, displays a progress bar of the download to
        stderr
     """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, progress,
                     **kwargs)


def densenet201(progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        progress (bool): If True, displays a progress bar of the download to 
        stderr
     """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, progress,
                     **kwargs)
