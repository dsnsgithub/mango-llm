import re

import pandas as pd
import torch

import pytorch_check

raw_dataset = pd.read_csv("./dataset/TinyStories/train.csv")["text"]
dataset = raw_dataset[:2000]


def list_tokens(string: str):
    return re.findall(r"\w+|[^\w\s]", string.lower())


vocab = set()
for story in dataset:
    words = list_tokens(story)
    vocab.update(words)
print("Vocab Length: ", len(vocab))

token_to_index_map = {word: i for i, word in enumerate(sorted(vocab))}
index_to_token_map = {i: word for i, word in enumerate(sorted(vocab))}


def string_to_token_ids(string: str):
    token_list = list_tokens(string)
    return torch.tensor([token_to_index_map[word] for word in token_list]).to(
        device=pytorch_check.device
    )
