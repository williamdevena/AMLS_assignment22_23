"""
This module contains the SimpleNN class (see the documentation of the class SimpleNN for more details)
"""

import os
import torch
from torch import nn
#from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
#from torchvision import transforms

#from models.linear_layer import LinearLayer


class CNN(nn.Module):
    """
    This class represents a convolutional neural network 
    """

    def __init__(self, input_size, ouput_size, hidden_layers, activation):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool2d(3, 3)
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3)
        self.fc1 = nn.Linear(in_features=384*3*3, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 384*3*3)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x
