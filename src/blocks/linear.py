import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, *, input_dimension: int, output_dimension: int):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(input_dimension, output_dimension))
        self.biases = nn.Parameter(torch.randn(output_dimension))

    def forward(self, x):
       return torch.matmul(x, self.weights) + self.biases 
    