import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

# knobs
VOCAB_SIZE  = 3000
EMBED_DIM   = 128
CONTEXT_LEN = 8
NUM_HEADS   = 4
FFN_HIDDEN  = 512
NUM_LAYERS  = 2
STEPS       = 30000
BATCH_SIZE  = 64
LR          = 3e-4
device      = "cuda" if torch.cuda.is_available() else "cpu"

# data
with open("input.txt") as f:
    text = f.read()
words     = text.lower().split()
top_words = [w for w, _ in Counter(words).most_common(VOCAB_SIZE)]
tokens    = {w: i for i, w in enumerate(top_words)}
words     = [w for w in words if w in tokens]
all_ids   = torch.tensor([tokens[w] for w in words], dtype=torch.long, device=device)

def get_batch():
    ix = torch.randint(len(all_ids) - CONTEXT_LEN, (BATCH_SIZE,))
    return (torch.stack([all_ids[i:i+CONTEXT_LEN] for i in ix]),
            torch.stack([all_ids[i+CONTEXT_LEN]   for i in ix]))

# model — entire thing in one block
model = nn.Sequential(
    nn.Embedding(VOCAB_SIZE, EMBED_DIM),                         # tok embeddings (pos added in forward)
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(EMBED_DIM, NUM_HEADS, FFN_HIDDEN, batch_first=True),
        num_layers=NUM_LAYERS,
    ),
    nn.Linear(EMBED_DIM, VOCAB_SIZE),
).to(device)

pos_emb = nn.Embedding(CONTEXT_LEN, EMBED_DIM).to(device)
mask    = nn.Transformer.generate_square_subsequent_mask(CONTEXT_LEN, device=device)
opt     = torch.optim.Adam(list(model.parameters()) + list(pos_emb.parameters()), lr=LR)

def forward(x):
    pos = torch.arange(x.shape[1], device=device)
    h   = model[0](x) + pos_emb(pos)        # embed + position
    h   = model[1](h, mask=mask)             # transformer blocks
    return model[2](h[:, -1, :])             # project last token → vocab

# train
for step in range(STEPS):
    x, y   = get_batch()
    loss   = F.cross_entropy(forward(x), y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 1000 == 0:
        print(f"step {step:5d}  loss: {loss.item():.4f}")

# generate
def generate(seed_words, n_words=20, temperature=0.8):
    model.eval()
    context = [tokens[w] for w in seed_words if w in tokens]
    result  = list(seed_words)
    with torch.no_grad():
        for _ in range(n_words):
            x      = torch.tensor([context[-CONTEXT_LEN:]], device=device)
            probs  = F.softmax(forward(x)[0] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            result.append(top_words[next_id])
            context.append(next_id)
    model.train()
    return " ".join(result)

print("\ngenerated:", generate(["to", "be", "or"], n_words=20))