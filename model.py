import torch
import torch.nn as nn

import datasets
import re


## PYTORCH  -----------------------------------------------
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.random.manual_seed(seed=17)


## LLM CONSTANTS -----------------------------------------------
EMBEDDING_DIMENSIONS = 5


## DATASET PARSING -----------------------------------------------
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train[:10]")

def list_tokens(string):
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
        return self.weights[x]

embedding_table = Embedding(len(vocab), 5)
tokens = [token_mapping["this"], token_mapping["was"]]

print(tokens)
print(embedding_table(torch.tensor(tokens)))