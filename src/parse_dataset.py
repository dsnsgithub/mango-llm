import re

import pandas as pd
import torch

import pytorch_check
from constants import TOTAL_DATASET_ELEMENTS

raw_dataset = pd.read_csv("./dataset/TinyStories/train.csv")["text"]
dataset = raw_dataset[:TOTAL_DATASET_ELEMENTS]


def list_tokens(string: str):
    return re.findall(r"\w+|[^\w\s]", string)


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
