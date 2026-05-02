import pandas as pd

import torch
import torch.nn as nn

import math
import re


## PYTORCH  -----------------------------------------------
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.random.manual_seed(seed=17)
print("PyTorch Accelerator: ", device)

## LLM CONSTANTS -----------------------------------------------
EMBEDDING_DIMENSIONS = 64
HIDDEN_FEED_FORWARD_DIMENSIONS = EMBEDDING_DIMENSIONS * 4
MAX_LENGTH = 4096
LEARNING_RATE = 0.01
EPOCH_COUNT = 200

TRAINING = False

## DATASET PARSING -----------------------------------------------
dataset = pd.read_csv("./dataset/TinyStories/train.csv")["text"][:1000]

def list_tokens(string): # turns "This is a test." into {"this", "is", "a", "test", "."}
    return re.findall(r"\w+|[^\w\s]", string.lower())

vocab = set() # {'heard', 'celebrated', 'that', 'again', 'stopped'... }
for story in dataset:
    words = list_tokens(story)
    vocab.update(words)
print("Vocab Length: ", len(vocab))

token_to_index_map = {word: i for i, word in enumerate(sorted(vocab))} # {'after': 7, 'again': 8, 'all': 9, 'always': 10, 'am': 11, 'and': 12}
index_to_token_map = {i: word for i, word in enumerate(sorted(vocab))}

def string_to_token_ids(string: str):
    token_list = list_tokens(string)
    return torch.tensor([token_to_index_map[word] for word in token_list]).to(device=device)

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

        # weights:
        #          "this"   "was"
        # "this" [  1.0,    0.0  ]   ← how much "this" attends to each token
        # "was"  [  0.27,    0.73  ]   ← how much "was" attends to each token

        return torch.matmul(weights, V)


class FeedForward(nn.Module):
    def __init__(self, *, embedding_dimensions: int, hidden_dimensions: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dimensions, hidden_dimensions),
            nn.ReLU(),
            nn.Linear(hidden_dimensions, embedding_dimensions)
        )

    def forward(self, x):
        return self.network(x)


class Transformer(nn.Module):
    def __init__(self, *, embedding_dimensions: int, hidden_feed_forward_dimensions: int):
        super().__init__()
        self.embedding_dimensions = embedding_dimensions

        self.attention = Attention(embedding_dimensions=EMBEDDING_DIMENSIONS)
        self.feed_forward = FeedForward(embedding_dimensions=EMBEDDING_DIMENSIONS, hidden_dimensions=HIDDEN_FEED_FORWARD_DIMENSIONS)

        self.norm1 = nn.LayerNorm(EMBEDDING_DIMENSIONS)
        self.norm2 = nn.LayerNorm(EMBEDDING_DIMENSIONS)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.norm1(x)
        x = x + self.feed_forward(x)
        x = x + self.norm1(x)

        return x

class LLM(nn.Module):
    def __init__(self, *, embedding_dimensions: int, hidden_feed_forward_dimensions: int, max_length: int):
        super().__init__()

        self.token_embedding_table = Embedding(len(vocab), embedding_dimensions)
        self.positional_embedding_table = Embedding(max_length, embedding_dimensions)

        self.layers = nn.ModuleList([
            Transformer(embedding_dimensions=embedding_dimensions, hidden_feed_forward_dimensions=hidden_feed_forward_dimensions),
            nn.Linear(embedding_dimensions, len(vocab))
        ])

    def forward(self, token_ids):
        positions = torch.arange(len(token_ids)).to(token_ids.device)
        x = self.token_embedding_table(token_ids) + self.positional_embedding_table(positions)

        for layer in self.layers:
            x = layer(x)

        # torch.softmax(x, dim=-1) is not needed, as the cross entropy loss will do it automatically
        return x
    
def run_forward_pass(input):
    token_ids = string_to_token_ids(input)

    if len(token_ids) > MAX_LENGTH:
        raise ValueError("Input exceeds MAX_LENGTH of ", MAX_LENGTH, " tokens.")

    output: torch.Tensor = model(token_ids)

    next_token_index = int(output[-1].argmax().item())
    return index_to_token_map[next_token_index]

def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    tokenized_stories = []
    for story in dataset:
        token_ids = string_to_token_ids(story)
        tokenized_stories.append(token_ids)
    
    for epoch in range(EPOCH_COUNT):
        total_loss = 0
        for story_tokens in tokenized_stories:
            inputs = story_tokens[:-1] # everything except last token
            targets = story_tokens[1:] # everything after first token
            
            logits: torch.Tensor = model(inputs)    
            
            loss = loss_fn(logits, targets) # compare the output from the model given the first token (input[0]) to the given story next token (targets[0])
            total_loss += loss.item()
            
            optimizer.zero_grad() # reset parameter gradients across model
            loss.backward() # calculate gradients using loss
            optimizer.step() # updates parameters through the whole model

        print(f"Epoch {epoch} Loss: ", total_loss / len(tokenized_stories))

    torch.save(model.state_dict(), "model.pth")

def generate(prompt: str, new_tokens = 30):
    current_string = prompt
    
    for _ in range(new_tokens):
        new_token = run_forward_pass(current_string)

        separator = " " 
        if new_token in ("!", ".", "?", "'", ",") or current_string[-1] == "'":
           separator = "" 
        
        current_string = current_string + separator + new_token

    print(current_string)

model = LLM(embedding_dimensions=EMBEDDING_DIMENSIONS, hidden_feed_forward_dimensions=HIDDEN_FEED_FORWARD_DIMENSIONS, max_length=MAX_LENGTH).to(device)

if TRAINING:
    train()
else:
    model.load_state_dict(torch.load("model.pth"))

total_parameters = sum(p.numel() for p in model.parameters())
print("This model has: ", total_parameters, " parameters.")

generate("This was", 30)