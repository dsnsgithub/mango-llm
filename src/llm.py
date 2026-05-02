import torch
import torch.nn as nn

from blocks.embedding import Embedding
from blocks.linear import Linear
from blocks.transformer import Transformer
from constants import (
    EMBEDDING_DIMENSIONS,
    HIDDEN_FEED_FORWARD_DIMENSIONS,
    MAX_LENGTH,
    TRANSFORMER_BLOCKS,
)
from parse_dataset import vocab
from pytorch_check import device


class LLM(nn.Module):
    def __init__(
        self,
        *,
        embedding_dimensions: int,
        hidden_feed_forward_dimensions: int,
        max_length: int,
        vocab_length: int,
    ):
        super().__init__()

        self.token_embedding_table = Embedding(vocab_length, embedding_dimensions)
        self.positional_embedding_table = Embedding(max_length, embedding_dimensions)

        self.transformer_blocks = [
            Transformer(
                embedding_dimensions=embedding_dimensions,
                hidden_feed_forward_dimensions=hidden_feed_forward_dimensions,
            )
            for _ in range(TRANSFORMER_BLOCKS)
        ]
        self.layers = nn.ModuleList(
            self.transformer_blocks
            + [
                Linear(
                    input_dimension=embedding_dimensions, output_dimension=vocab_length
                )
            ]
        )

    def forward(self, token_ids):
        positions = torch.arange(len(token_ids)).to(token_ids.device)
        x = self.token_embedding_table(token_ids) + self.positional_embedding_table(
            positions
        )

        for layer in self.layers:
            x = layer(x)

        # torch.softmax(x, dim=-1) is not needed, as the cross entropy loss will do it automatically
        return x


model = LLM(
    embedding_dimensions=EMBEDDING_DIMENSIONS,
    hidden_feed_forward_dimensions=HIDDEN_FEED_FORWARD_DIMENSIONS,
    max_length=MAX_LENGTH,
    vocab_length=len(vocab),
).to(device)
