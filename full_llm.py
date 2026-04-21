import numpy as np
from collections import Counter

# ─────────────────────────────────────────────
# 1. the one knob
# ─────────────────────────────────────────────

EMBED_DIM = 256
STEPS = 20000

FFN_HIDDEN = 4 * EMBED_DIM
NUM_HEADS  = max(1, EMBED_DIM // 64)
NUM_LAYERS = max(1, EMBED_DIM // 16)

CONTEXT_LEN = 5
MAX_VOCAB   = 2000
learning_rate = 0.01


# ─────────────────────────────────────────────
# 2. load text + build vocab
# ─────────────────────────────────────────────

with open("input.txt", "r") as f:
    text = f.read()

words = text.lower().split()

word_counts = Counter(words)
top_words   = [w for w, _ in word_counts.most_common(MAX_VOCAB)]

tokens     = {word: i for i, word in enumerate(top_words)}
VOCAB_SIZE = len(tokens)

words = [w for w in words if w in tokens]

print(f"vocab size: {VOCAB_SIZE}, training words: {len(words)}")

# ─────────────────────────────────────────────
# 3. build training data
# ─────────────────────────────────────────────


training_data = []
for i in range(len(words) - CONTEXT_LEN):
    context = words[i : i + CONTEXT_LEN]
    target  = words[i + CONTEXT_LEN]
    training_data.append((context, target))

print(f"training examples: {len(training_data)}")

# ─────────────────────────────────────────────
# 4. weights
# ─────────────────────────────────────────────

embedding_table = np.random.randn(VOCAB_SIZE, EMBED_DIM) * 0.1

Wq = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.1
Wk = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.1
Wv = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.1

weights     = np.random.randn(EMBED_DIM, FFN_HIDDEN) * 0.1
biases      = np.zeros(FFN_HIDDEN)
weights_out = np.random.randn(FFN_HIDDEN, VOCAB_SIZE) * 0.1
biases_out  = np.zeros(VOCAB_SIZE)


# ─────────────────────────────────────────────
# 5. helpers
# ─────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ─────────────────────────────────────────────
# 6. training loop
# ─────────────────────────────────────────────

for step in range(STEPS):  # fix 1: was 200

    context_words, correct_word = training_data[step % len(training_data)]
    token_id   = [tokens[w] for w in context_words]
    correct_id = tokens[correct_word]

    # ── forward pass ──────────────────────────

    token_vectors = embedding_table[token_id]

    Q = np.dot(token_vectors, Wq)
    K = np.dot(token_vectors, Wk)
    V = np.dot(token_vectors, Wv)

    scores = np.dot(Q, K.T) / np.sqrt(EMBED_DIM)
    mask   = np.triu(np.full_like(scores, -1e9), k=1)
    scores = scores + mask

    attention_weights = np.array([softmax(row) for row in scores])
    context_vectors   = np.dot(attention_weights, V)

    token_vector    = context_vectors[-1]
    pre_relu        = np.dot(token_vector, weights) + biases
    relu_activation = np.maximum(0, pre_relu)
    output          = softmax(np.dot(relu_activation, weights_out) + biases_out)

    loss           = -np.log(output[correct_id] + 1e-9)  # fix 2: epsilon guard
    predicted_word = top_words[np.argmax(output)]

    if step % 1000 == 0:
        print(f"step {step:5d}  loss: {loss:.4f}  "
              f"input: {' '.join(context_words)} → "
              f"predicted: {predicted_word}  target: {correct_word}")

    # ── backward pass ──────────────────────────

    d_output              = output.copy()
    d_output[correct_id] -= 1

    d_W_out    = np.outer(relu_activation, d_output)
    d_B_out    = d_output
    d_hidden   = np.dot(d_output, weights_out.T)
    d_pre_relu = d_hidden * (pre_relu > 0)
    d_W        = np.outer(token_vector, d_pre_relu)
    d_biases   = d_pre_relu

    weights_out -= learning_rate * d_W_out
    biases_out  -= learning_rate * d_B_out
    weights     -= learning_rate * d_W
    biases      -= learning_rate * d_biases

    embedding_table[token_id[-1]] -= learning_rate * np.dot(d_pre_relu, weights.T)

# ─────────────────────────────────────────────
# 7. generate text
# ─────────────────────────────────────────────

def generate(seed_words, n_words=20):
    result  = list(seed_words)
    context = list(seed_words)

    for _ in range(n_words):
        token_id      = [tokens[w] for w in context if w in tokens]
        token_vectors = embedding_table[token_id]

        Q = np.dot(token_vectors, Wq)
        K = np.dot(token_vectors, Wk)
        V = np.dot(token_vectors, Wv)

        scores = np.dot(Q, K.T) / np.sqrt(EMBED_DIM)
        mask   = np.triu(np.full_like(scores, -1e9), k=1)
        scores = scores + mask

        attention_weights = np.array([softmax(row) for row in scores])
        context_vectors   = np.dot(attention_weights, V)

        token_vector    = context_vectors[-1]
        pre_relu        = np.dot(token_vector, weights) + biases
        relu_activation = np.maximum(0, pre_relu)
        output          = softmax(np.dot(relu_activation, weights_out) + biases_out)

        next_word = top_words[np.argmax(output)]
        result.append(next_word)
        context = context[1:] + [next_word]

    return " ".join(result)

# fix 3: just print top 5 instead of all 500
print()
top5 = np.argsort(output)[-5:][::-1]
print("top predictions for last training example:")
for i in top5:
    print(f"  {top_words[i]:>15}: {output[i]:.1%}")

print()
print("generated:", generate(["to", "be", "or"], n_words=20))