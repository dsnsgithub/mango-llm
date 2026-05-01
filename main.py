import datasets

import torch
import torch.nn as nn
import re


## PYTORCH  -----------------------------------------------
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.random.manual_seed(seed=17)
print(device)

## LLM CONSTANTS -----------------------------------------------
EMBEDDING_DIMENSIONS = 5
MAX_LENGTH = 12

## DATASET PARSING -----------------------------------------------
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train[:10]")

def list_tokens(string): # turns "This is a test." into {"this", "is", "a", "test", "."}
    return re.findall(r"\w+|[^\w\s]", string.lower())

vocab = set() # {'heard', 'celebrated', 'that', 'again', 'stopped'... }
for story in dataset:
    words = list_tokens(story["text"])
    vocab.update(words)

token_mapping = {word: i for i, word in enumerate(sorted(vocab))} # {'after': 7, 'again': 8, 'all': 9, 'always': 10, 'am': 11, 'and': 12}
# print(token_mapping)

## LLM -----------------------------------------------

class Embedding(nn.Module):
    def __init__(self, vocab_length: int, embedding_dimensions: int):
        super().__init__()

        # only reason to use nn.Parameter is because without it, pytorch can't use autograd
        self.weights = nn.Parameter(torch.randn(vocab_length, embedding_dimensions))

    def forward(self, x):
        # returns the rows in the matrix that are actually needed
        return self.weights[x]


class Attention(nn.Module):
    def __init__(self, head_count: int, embedding_dimensions: int):
        super().__init__()

        self.W_q = nn.Parameter(torch.randn(embedding_dimensions, embedding_dimensions))
        self.W_k = nn.Parameter(torch.randn(embedding_dimensions, embedding_dimensions))
        self.W_v = nn.Parameter(torch.randn(embedding_dimensions, embedding_dimensions))

    def forward(self, x):
        return x


class LLM(nn.Module):
    def __init__(self, EMBEDDING_DIMENSIONS, MAX_LENGTH):
        super().__init__()

        self.token_embedding_table = Embedding(len(vocab), EMBEDDING_DIMENSIONS)
        self.positional_embedding_table = Embedding(MAX_LENGTH, EMBEDDING_DIMENSIONS) # each row index is what is applied depending on where it is in the input
        self.positions = torch.arange(len(token_input))

        self.layers = nn.ModuleList([
        ])

    def forward(self, token_ids):
        x = self.token_embedding_table(token_ids) + self.positional_embedding_table(self.positions)

        for layer in self.layers:
            x = layer(x)
        
        return x
    
token_input = ["this", "was"]
token_ids = torch.tensor([token_mapping[word] for word in token_input])

model = LLM(EMBEDDING_DIMENSIONS, MAX_LENGTH)
output = model(token_ids)

print(output)