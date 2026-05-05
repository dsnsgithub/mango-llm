import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, *, input_dimension: int, output_dimension: int):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(input_dimension, output_dimension))
        self.biases = nn.Parameter(torch.zeros(output_dimension))

        # initially weights is randomly distributed with a standard deviation of 1
        # using kaiming_uniform brings these weights closer
        nn.init.kaiming_uniform_(self.weights)

    def forward(self, x):
        return torch.matmul(x, self.weights) + self.biases
