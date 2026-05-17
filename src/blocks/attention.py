import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, *, embedding_dimensions: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads
        self.head_dimensions = embedding_dimensions // num_heads

        # only reason to use nn.Parameter is because without it, pytorch can't use autograd
        self.W_q = nn.Parameter(torch.randn(embedding_dimensions, embedding_dimensions))
        self.W_k = nn.Parameter(torch.randn(embedding_dimensions, embedding_dimensions))
        self.W_v = nn.Parameter(torch.randn(embedding_dimensions, embedding_dimensions))

    def forward(self, x: torch.Tensor):
        sequence_length, embedding_dimensions = x.size()
    
        # create the matrix Q, split it into multiple heads of (sequence_length, self.embedding_dimensions / self.head_dimensions)
        # then transpose the first and second dimension, so that it becomes (num_heads, sequence_length, self.embedding_dimensions / self.head_dimensions)
        Q = torch.matmul(x, self.W_q).view(sequence_length, self.num_heads, self.head_dimensions).transpose(0, 1)
        K = torch.matmul(x, self.W_k).view(sequence_length, self.num_heads, self.head_dimensions).transpose(0, 1)
        V = torch.matmul(x, self.W_v).view(sequence_length, self.num_heads, self.head_dimensions).transpose(0, 1)
        
        # .tranpose(-2, -1) allows Q to be multiplied by K (swap sequence_length, self.head_dimensions in K)
        scores_before_mask = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dimensions)

        ones_matrix = torch.ones(sequence_length, sequence_length)
        mask = torch.triu(ones_matrix, diagonal=1).bool().to(x.device)

        scores = scores_before_mask.masked_fill(mask, float("-inf"))

        # dim=-1 means each row in the tensor will add to 1
        # the operation is happening across the last dimensions (the columns)
        weights = torch.softmax(scores, dim=-1)

        # weights:
        #          "this"   "was"
        # "this" [  1.0,    0.0  ]   ← how much "this" attends to each token
        # "was"  [  0.27,    0.73  ]   ← how much "was" attends to each token

        return torch.matmul(weights, V).transpose(0, 1).view(sequence_length, embedding_dimensions)
