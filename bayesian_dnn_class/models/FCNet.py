import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append("../")
import bayes_layers as bl


class FCNet(nn.Module):
    def __init__(self, bias=True, **bayes_args):
        super(FCNet, self).__init__()
        self.InitBayes(**bayes_args)

    def forward(self, x):
        x = x.reshape(-1, 28**2)
        kl = torch.tensor(0.).to(x.device)

        for layer_id, layer in enumerate(self.layers):
            tmp = layer(x)

            if isinstance(tmp, tuple):
                x, kl_ = tmp
                kl += kl_
            else:
                x = tmp

        out = self.act(x)
        return out, kl

    def Init(self):
        layers = []
        layers.append(nn.Linear(784, 10, bias=False))
        # layers.append(nn.functional.ReLu())
        # layers.append(nn.BatchNorm1d(650))

        #layers.append(nn.Linear(64, 10, bias=True))
        # layers.append(nn.BatchNorm1d(10))

        self.layers = nn.ModuleList(layers)
        self.act = nn.Sigmoid()

    def InitBayes(self, **bayes_args):
        layers = []
        layers.append(
            bl.Linear(784, 64, bias=False, **bayes_args,
                      activation=F.relu))

        layers.append(bl.Linear(64, 10, bias=False, **bayes_args))
        self.layers = nn.ModuleList(layers)
        self.act = nn.Sigmoid()


class MyMul(nn.Module):
    def __init__(self, size):
        super(MyMul, self).__init__()
        self.weight = nn.Parameter(torch.rand(1))

    def forward(self, x):
        out = x * torch.abs(self.weight)
        return out


class MyAdd(nn.Module):
    def __init__(self, size):
        super(MyAdd, self).__init__()
        self.weight = nn.Parameter(torch.rand(size))

    def forward(self, x):
        out = x + self.weight
        return out
