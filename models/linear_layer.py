"""
This module contains the LinearLayer class
"""

from torch import nn


class LinearLayer(nn.Module):
    """
    Represents a layer of NN and it's composed by a Linear layer and a activation layer.
    """

    def __init__(self, input_size, output_size, activation):
        """
        Args:
            - input_size (int): size of the input of the layer
            (number of features that every neuron gets as input)
            - output_size (int): size of the output of the layer
            (number of neurons in the layer)
            - activation (nn.Module): type of activation (for example: nn.ReLU)
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            self.activation
        )
        self.weight = self.layers[0].weight

    def forward(self, x):
        # print(self.layers)
        y = self.layers(x)
        return y
