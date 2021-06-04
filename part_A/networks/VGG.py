import sys
import torch
import torch.nn as nn
import torch.optim as optim

config = {
    "vgg": [256, 128, 64, 64, 32, 16, 1]
}


class MyVGGNet(nn.Module):
    def __init__(self, version, in_channel):
        super(MyVGGNet, self).__init__()
        layers_param = config[version]
        self.features = self._build(in_channel, layers_param)

    def _build(self, in_channel, layers_param):
        layers = []
        in_dim = in_channel
        for out_dim in layers_param[:-1]: 
            layers += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_dim), 
                nn.ReLU(inplace=True)
            ]
            in_dim = out_dim
        layers += nn.Conv2d(in_dim, layers_param[-1], kernel_size=3, padding=1),

        return nn.Sequential(*layers)


    def forward(self, x):
        scores = self.features(x)
        return scores
