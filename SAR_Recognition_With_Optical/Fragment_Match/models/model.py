# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 0006 20:54
# @Author  : Ronin
# @File    : model.py
# @Software: PyCharm
import torch
from torch import nn


def core_net(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(output_channels)
    )


def conv_flow():
    return nn.Sequential(
        core_net(input_channels=1, output_channels=32),
        core_net(input_channels=32, output_channels=32),
        nn.MaxPool2d(kernel_size=2, stride=2),
        core_net(input_channels=32, output_channels=64),
        core_net(input_channels=64, output_channels=64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        core_net(input_channels=64, output_channels=128),
        core_net(input_channels=128, output_channels=128),
        nn.MaxPool2d(kernel_size=2, stride=2),
        core_net(input_channels=128, output_channels=128),
        core_net(input_channels=128, output_channels=128),
    )


class SRWO(nn.Module):
    def __init__(self):
        super(SRWO, self).__init__()
        self.convflow1 = conv_flow()
        self.convflow2 = conv_flow()
        self.fusion = nn.Sequential(
            core_net(input_channels=256, output_channels=256),
            core_net(input_channels=256, output_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=4608, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=2)
        )

    def forward(self, x, y):
        x = self.convflow1(x)
        y = self.convflow2(y)
        z = torch.cat((x, y), dim=1)
        z = self.fusion(z)
        z = nn.functional.softmax(z, dim=1)
        return z
