import pandas as pd

import torch
import torch.nn as nn

import math
import re


## PYTORCH  -----------------------------------------------
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.random.manual_seed(seed=17)
print(device)

## LLM CONSTANTS -----------------------------------------------
EMBEDDING_DIMENSIONS = 5
MAX_LENGTH = 12

## DATASET PARSING -----------------------------------------------
dataset = pd.read_csv("./dataset/TinyStories/train.csv")["text"][:10]

def list_tokens(string): # turns "This is a test." into {"this", "is", "a", "test", "."}
    return re.findall(r"\w+|[^\w\s]", string.lower())

vocab = set() # {'heard', 'celebrated', 'that', 'again', 'stopped'... }
for story in dataset:
    words = list_tokens(story)
    vocab.update(words)

token_to_index_map = {word: i for i, word in enumerate(sorted(vocab))} # {'after': 7, 'again': 8, 'all': 9, 'always': 10, 'am': 11, 'and': 12}

index_to_token_map = {i: word for i, word in enumerate(sorted(vocab))}

## LLM -----------------------------------------------

class Embedding(nn.Module):
    def __init__(self, vocab_length: int, embedding_dimensions: int):
        super().__init__()

        # only reason to use nn.Parameter is because without it, pytorch can't use autograd
        self.weights = nn.Parameter(torch.randn(vocab_length, embedding_dimensions))

    def forward(self, x: torch.Tensor):
        # returns the rows in the matrix that are actually needed
        return self.weights[x]


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
        #          "this"   "was"
        # "this" [  1.0,    0.0  ]   ← how much "this" attends to each token
        # "was"  [  0.27,    0.73  ]   ← how much "was" attends to each token

        return torch.matmul(weights, V)


class LLM(nn.Module):
    def __init__(self, EMBEDDING_DIMENSIONS, MAX_LENGTH):
        super().__init__()

        self.token_embedding_table = Embedding(len(vocab), EMBEDDING_DIMENSIONS)
        self.positional_embedding_table = Embedding(MAX_LENGTH, EMBEDDING_DIMENSIONS)

        self.layers = nn.ModuleList([
            Attention(embedding_dimensions=EMBEDDING_DIMENSIONS),
            nn.LayerNorm(EMBEDDING_DIMENSIONS),
            nn.Linear(EMBEDDING_DIMENSIONS, len(vocab))
        ])

    def forward(self, token_ids):
        positions = torch.arange(len(token_ids)).to(token_ids.device)
        x = self.token_embedding_table(token_ids) + self.positional_embedding_table(positions)

        for layer in self.layers:
            x = layer(x)
        
        return torch.softmax(x, dim=-1)
    
token_input = ["this", "was"]
token_ids = torch.tensor([token_to_index_map[word] for word in token_input])

model = LLM(EMBEDDING_DIMENSIONS, MAX_LENGTH).to(device)
output: torch.Tensor = model(token_ids)

next_token_index = output.argmax().item()
print("Input text:", token_input)
print("Next token:", index_to_token_map[next_token_index])