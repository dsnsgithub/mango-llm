import re

import pandas as pd
import torch

import pytorch_check
from constants import TOTAL_DATASET_ELEMENTS

from torch.nn.utils.rnn import pad_sequence


raw_dataset = pd.read_csv("./dataset/TinyStories/train.csv")["text"].astype(str)
raw_dataset = raw_dataset[raw_dataset.str.len() >= 200]
dataset = raw_dataset[:TOTAL_DATASET_ELEMENTS] if TOTAL_DATASET_ELEMENTS else raw_dataset

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

def collate_fn(batch):
    inputs, targets = zip(*batch)

    # for inputs, the padding_value doesn't matter, since it will evaluate to -1 in the loss and be ignored.
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)

    return inputs_padded, targets_padded


class StoryDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenized_stories = [string_to_token_ids(story) for story in dataset]

    def __len__(self):
        return len(self.tokenized_stories)

    def __getitem__(self, index):
        # find the token vector for the given story index
        story_tokens = self.tokenized_stories[index]

        # the model should output the token after given the input
        # add [:, ] because it is for each batch
        input_tokens = story_tokens[:, :-1]  # everything except last token
        expected_tokens = story_tokens[:, 1:]  # everything after first token

        return input_tokens, expected_tokens
