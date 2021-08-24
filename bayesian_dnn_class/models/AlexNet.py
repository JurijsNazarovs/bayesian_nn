import torch.nn as nn
import sys

sys.path.append("../")
import bayes_layers as bl


class AlexNet(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, num_classes, in_channels, **bayes_args):
        super(AlexNet, self).__init__()

        self.classifier = bl.Linear(1 * 1 * 128, num_classes, **bayes_args)

        self.conv1 = bl.Conv2d(in_channels,
                               64,
                               kernel_size=11,
                               stride=4,
                               padding=5,
                               **bayes_args)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = bl.Conv2d(64, 192, kernel_size=5, padding=2, **bayes_args)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = bl.Conv2d(192,
                               384,
                               kernel_size=3,
                               padding=1,
                               **bayes_args)
        self.soft3 = nn.Softplus()

        self.conv4 = bl.Conv2d(384,
                               256,
                               kernel_size=3,
                               padding=1,
                               **bayes_args)
        self.soft4 = nn.Softplus()

        self.conv5 = bl.Conv2d(256,
                               128,
                               kernel_size=3,
                               padding=1,
                               **bayes_args)
        self.soft5 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.fc1 = bl.Linear(128, num_classes, **bayes_args)

        layers = [
            self.conv1, self.soft1, self.pool1, self.conv2, self.soft2,
            self.pool2, self.conv3, self.soft3, self.conv4, self.soft4,
            self.conv5, self.soft5, self.pool3
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        kl = 0
        for layer in self.layers:
            tmp = layer(x)
            if isinstance(tmp, tuple):
                x, kl_ = tmp
                kl += kl_
            else:
                x = tmp

        x = x.view(x.size(0), -1)
        logits, _kl = self.classifier.forward(x)
        kl += _kl

        return logits, kl
