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


class SimpleNN(nn.Module):
    """
    This class represents a fully connected neural network 
    """

    def __init__(self, input_size, ouput_size, hidden_layers, activation):
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
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = nn.Sigmoid() if self.output_size == 1 else nn.Softmax(dim=1)

        self.layers = nn.Sequential(
            self.print_shape,
            nn.Linear(self.input_size, self.hidden_layers[0]),
            self.activation
        )

        for index, _ in enumerate(self.hidden_layers[:-1]):
            hidden_layer_input_size = self.hidden_layers[index]
            hidden_layer_output_size = self.hidden_layers[index+1]
            self.layers.append(
                nn.Linear(hidden_layer_input_size, hidden_layer_output_size),
            )
            self.layers.append(self.activation)

        self.layers.append(
            nn.Linear(self.hidden_layers[-1], self.output_size),
        )
        self.layers.append(self.output_activation)

        print(self.layers)

    def forward(self, x):
        """
        Forward pass
        """
        y = self.layers(x)

        return y
