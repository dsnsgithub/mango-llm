import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, *, embedding_dimensions: int, hidden_dimensions: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dimensions, hidden_dimensions),
            nn.ReLU(),
            nn.Linear(hidden_dimensions, embedding_dimensions),
        )

    def forward(self, x):
        return self.network(x)
