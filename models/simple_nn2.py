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


class SimpleNN2(nn.Module):
    """
    This class represents a fully connected neural network 
    """

    def __init__(self, input_size, ouput_size, activation):
        """
        Args:
            - input_size (int): size of the input samples
            - ouput_size (int): size of the output samples
            - hidden_layers (List): contains the input sizes of each hidden layer (in order)
            - activation (nn.Module): type of activation (for example: nn.ReLU)
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = ouput_size
        self.activation = activation
        self.output_activation = nn.Sigmoid() if self.output_size == 1 else nn.Softmax(dim=1)

        self.fc1 = nn.Linear(self.input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, self.output_size)

    def forward(self, x):
        """
        Forward pass
        """
        y = self.activation(self.fc1(x))
        y = self.activation(self.fc2(y))
        y = self.output_activation(self.fc3(y))

        return y
