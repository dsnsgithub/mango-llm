import torch.nn as nn
from blocks.attention import Attention
from blocks.feedforward import FeedForward

from constants import EMBEDDING_DIMENSIONS, HIDDEN_FEED_FORWARD_DIMENSIONS


class Transformer(nn.Module):
    def __init__(
        self, *, embedding_dimensions: int, hidden_feed_forward_dimensions: int
    ):
        super().__init__()
        self.embedding_dimensions = embedding_dimensions

        self.attention = Attention(embedding_dimensions=EMBEDDING_DIMENSIONS)
        self.feed_forward = FeedForward(
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            hidden_dimensions=HIDDEN_FEED_FORWARD_DIMENSIONS,
        )

        self.norm1 = nn.LayerNorm(EMBEDDING_DIMENSIONS)
        self.norm2 = nn.LayerNorm(EMBEDDING_DIMENSIONS)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x
