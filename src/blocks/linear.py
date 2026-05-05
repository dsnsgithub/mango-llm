import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, *, input_dimension: int, output_dimension: int):
        super().__init__()

        # only reason to use nn.Parameter is because without it, pytorch can't use autograd
        self.weights = nn.Parameter(torch.empty(input_dimension, output_dimension))
        self.biases = nn.Parameter(torch.zeros(output_dimension))

        # initially weights is randomly distributed with a standard deviation of 1
        # using kaiming_uniform brings these weights closer
        # use "fan-in" for forward pass, a=0 for ReLU
        nn.init.kaiming_uniform_(self.weights, mode="fan_in", a=0)

    def forward(self, x):
        return torch.matmul(x, self.weights) + self.biases
