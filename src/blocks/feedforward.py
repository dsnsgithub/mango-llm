import torch.nn as nn

from blocks.linear import Linear

class FeedForward(nn.Module):
    def __init__(self, *, embedding_dimensions: int, hidden_dimensions: int):
        super().__init__()

        self.network = nn.Sequential(
            Linear(input_dimension=embedding_dimensions, output_dimension=hidden_dimensions),
            nn.ReLU(),
            Linear(input_dimension=hidden_dimensions, output_dimension=embedding_dimensions),
        )

    def forward(self, x):
        return self.network(x)
