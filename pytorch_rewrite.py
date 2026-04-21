import torch
import torch.nn.functional as F
from collections import Counter

# ─────────────────────────────────────────────
# 1. the knobs
# ─────────────────────────────────────────────

EMBED_DIM     = 128      # 256 is overkill for ~65k chars; 128 trains faster, similar quality
STEPS         = 30000    # Shakespeare is small — needs more steps to generalize
BATCH_SIZE    = 64       # Bigger batches = more stable gradients, Shakespeare fits in RAM
CONTEXT_LEN   = 8        # Shakespeare has longer dependencies; 5 misses a lot of structure
MAX_VOCAB     = 3000     # ~3k covers most unique words; 2k cuts off too many rare words
learning_rate = 3e-4     # Classic Adam LR — 0.01 is too high, causes instability
FFN_HIDDEN    = 4 * EMBED_DIM  # Keep this — the 4x ratio is well-established

temperature = 1

device     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using: {device}")

# ─────────────────────────────────────────────
# 2. load text + build vocab
# ─────────────────────────────────────────────

with open("input.txt", "r") as f:
    text = f.read()

words       = text.lower().split()
top_words   = [w for w, _ in Counter(words).most_common(MAX_VOCAB)]
tokens      = {word: i for i, word in enumerate(top_words)}
VOCAB_SIZE  = len(tokens)
words       = [w for w in words if w in tokens]

print(f"vocab size: {VOCAB_SIZE}, training words: {len(words)}")

# ─────────────────────────────────────────────
# 3. build training data
# ─────────────────────────────────────────────

all_ids = torch.tensor([tokens[w] for w in words], dtype=torch.long)

def get_batch():
    # pick BATCH_SIZE random positions in the text
    ix      = torch.randint(len(all_ids) - CONTEXT_LEN, (BATCH_SIZE,))
    x       = torch.stack([all_ids[i : i + CONTEXT_LEN] for i in ix])
    y       = torch.stack([all_ids[i + CONTEXT_LEN]     for i in ix])
    return x.to(device), y.to(device)

# ─────────────────────────────────────────────
# 4. weights
# ─────────────────────────────────────────────

def param(shape):
    return (torch.randn(shape, device=device) * 0.1).requires_grad_(True)

embedding_table = param((VOCAB_SIZE, EMBED_DIM))

Wq = param((EMBED_DIM, EMBED_DIM))
Wk = param((EMBED_DIM, EMBED_DIM))
Wv = param((EMBED_DIM, EMBED_DIM))

weights     = param((EMBED_DIM, FFN_HIDDEN))
biases      = torch.zeros(FFN_HIDDEN, device=device, requires_grad=True)
weights_out = param((FFN_HIDDEN, VOCAB_SIZE))
biases_out  = torch.zeros(VOCAB_SIZE, device=device, requires_grad=True)

ln1_w = torch.ones(EMBED_DIM, device=device, requires_grad=True)
ln1_b = torch.zeros(EMBED_DIM, device=device, requires_grad=True)

parameters = [embedding_table, Wq, Wk, Wv, weights, biases, weights_out, biases_out, ln1_w, ln1_b]
optimizer  = torch.optim.Adam(parameters, lr=learning_rate)




# ─────────────────────────────────────────────
# 5. causal mask
# ─────────────────────────────────────────────

mask = torch.triu(torch.full((CONTEXT_LEN, CONTEXT_LEN), float("-inf"), device=device), diagonal=1)

# ─────────────────────────────────────────────
# 6. training loop
# ─────────────────────────────────────────────

for step in range(STEPS):

    x, y = get_batch()   # x: [BATCH, CONTEXT]  y: [BATCH]

    # ── forward pass ──────────────────────────

    token_vectors = embedding_table[x]             # [BATCH, CONTEXT, EMBED]

    Q = token_vectors @ Wq                         # [BATCH, CONTEXT, EMBED]
    K = token_vectors @ Wk
    V = token_vectors @ Wv

    scores  = (Q @ K.transpose(-2, -1)) / EMBED_DIM ** 0.5 + mask
    attn_w  = F.softmax(scores, dim=-1)            # [BATCH, CONTEXT, CONTEXT]

    context = attn_w @ V + token_vectors                                          # ← residual
    last    = F.layer_norm(context[:, -1, :], (EMBED_DIM,), ln1_w, ln1_b)     
       # ← norm, replaces old `last = context[:, -1, :]`
    pre_relu    = last @ weights + biases          # [BATCH, FFN]
    hidden      = F.relu(pre_relu)                 # [BATCH, FFN]
    logits      = hidden @ weights_out + biases_out  # [BATCH, VOCAB]

    loss = F.cross_entropy(logits, y)

    # ── backward pass ──────────────────────────

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)    # ← add this
    optimizer.step()

    if step % 1000 == 0:
        predicted = logits.argmax(dim=-1)
        accuracy  = (predicted == y).float().mean().item()
        print(f"step {step:5d}  loss: {loss.item():.4f}  accuracy: {accuracy:.1%}")

# ─────────────────────────────────────────────
# 7. generate text
# ─────────────────────────────────────────────

def generate(seed_words, n_words=20):
    result  = list(seed_words)
    context = [tokens[w] for w in seed_words if w in tokens]

    with torch.no_grad():
        for _ in range(n_words):
            x = torch.tensor([context[-CONTEXT_LEN:]], device=device)  # [1, CONTEXT]

            token_vectors = embedding_table[x]
            Q = token_vectors @ Wq
            K = token_vectors @ Wk
            V = token_vectors @ Wv

            seq_len  = x.shape[1]
            m        = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
            scores   = (Q @ K.transpose(-2, -1)) / EMBED_DIM ** 0.5 + m
            attn_w   = F.softmax(scores, dim=-1)
            ctx      = attn_w @ V

            last    = ctx[:, -1, :]
            hidden  = F.relu(last @ weights + biases)
            logits  = hidden @ weights_out + biases_out

            # next_id = logits.argmax(dim=-1).item()
            probs   = F.softmax(logits[0] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            result.append(top_words[next_id])
            context.append(next_id)

    return " ".join(result)

print()
print("generated:", generate(["to", "be", "or"], n_words=20))