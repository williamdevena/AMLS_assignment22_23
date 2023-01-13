"""
This module contains the GenderNN class (see the documentation of the class GenderNN for more details)
"""

import os
import torch
from torch import nn
#from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
#from torchvision import transforms

#from models.linear_layer import LinearLayer


class GenderCNN(nn.Module):
    """
    This class represents a convolutional neural network
    used for the task of gender classification (binary classification)
    """

    def __init__(self):
        super(GenderCNN, self).__init__()
        self.pool = nn.MaxPool2d(3, 3)
        self.activation = nn.ReLU()
        self.last_activation = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3)

        self.conv4 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3)
        self.conv5 = nn.Conv2d(
            in_channels=1024, out_channels=2056, kernel_size=3)

        self.fc1 = nn.Linear(in_features=256*4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        # print("------")
        # print(x.shape)
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        # print(x.shape)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.conv4(x))
        x = self.pool(x)
        x = self.activation(self.conv5(x))
        x = self.pool(x)
        # print(x.shape)

        x = x.view(-1, 256*4)

        # print(x.shape)
        x = self.activation(self.fc1(x))
        # print(x.shape)
        x = self.activation(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        x = torch.squeeze(x)
        x = self.last_activation(x)

        return x
