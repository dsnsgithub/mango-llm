import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_length: int, embedding_dimensions: int):
        super().__init__()

        # only reason to use nn.Parameter is because without it, pytorch can't use autograd
        self.weights = nn.Parameter(torch.randn(vocab_length, embedding_dimensions))

    def forward(self, x: torch.Tensor):
        # returns the rows in the matrix that are actually needed
        return self.weights[x]
