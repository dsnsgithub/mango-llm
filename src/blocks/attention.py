import math

import torch
import torch.nn as nn

from constants import EMBEDDING_DIMENSIONS


class Attention(nn.Module):
    def __init__(self, *, embedding_dimensions: int):
        super().__init__()

        self.W_q = nn.Parameter(torch.randn(embedding_dimensions, embedding_dimensions))
        self.W_k = nn.Parameter(torch.randn(embedding_dimensions, embedding_dimensions))
        self.W_v = nn.Parameter(torch.randn(embedding_dimensions, embedding_dimensions))

    def forward(self, x: torch.Tensor):
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        scores_before_mask = torch.matmul(Q, K.T) / math.sqrt(EMBEDDING_DIMENSIONS)

        sequence_length = x.shape[0]
        ones_matrix = torch.ones(sequence_length, sequence_length)
        mask = torch.triu(ones_matrix, diagonal=1).bool().to(x.device)

        scores = scores_before_mask.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        # weights:
        #          "this"   "was"
        # "this" [  1.0,    0.0  ]   ← how much "this" attends to each token
        # "was"  [  0.27,    0.73  ]   ← how much "was" attends to each token

        return torch.matmul(weights, V)
