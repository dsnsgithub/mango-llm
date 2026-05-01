import numpy as np

# ─────────────────────────────────────────────
# 1. tokens
# ─────────────────────────────────────────────

tokens = {
    "cat":   0,
    "dog":   1,
    "king":  2,
    "queen": 3,
}

# input: "dog king" → predict "cat" as next word
token_id       = [tokens["dog"], tokens["king"]]
correct_id     = tokens["cat"]


# ─────────────────────────────────────────────
# 2. weights (initialized once, updated each step)
# ─────────────────────────────────────────────

embedding_table_dimensions = 3
neuron_count               = 6
vocab_size                 = len(tokens)

embedding_table = np.random.randn(vocab_size, embedding_table_dimensions) * 0.1

# attention weights — learned just like everything else
# this will make more sense later
Wq = np.random.randn(embedding_table_dimensions, embedding_table_dimensions) * 0.1
Wk = np.random.randn(embedding_table_dimensions, embedding_table_dimensions) * 0.1
Wv = np.random.randn(embedding_table_dimensions, embedding_table_dimensions) * 0.1


weights         = np.random.randn(embedding_table_dimensions, neuron_count) * 0.1
biases          = np.zeros(neuron_count)
weights_out     = np.random.randn(neuron_count, vocab_size) * 0.1
biases_out      = np.zeros(vocab_size)

learning_rate = 0.1


# ─────────────────────────────────────────────
# 3. helpers
# ─────────────────────────────────────────────

def softmax(scores):
    e = np.exp(scores - np.max(scores))
    return e / e.sum()


# ─────────────────────────────────────────────
# 4. training loop
# ─────────────────────────────────────────────

for step in range(200):

    # ── forward pass ──────────────────────────

    token_vectors    = embedding_table[token_id]
    print(token_vectors)

    # all of these are 2 rows (two tokens), 3 columns
    # Q = query, K = key, V = value
    Q = np.dot(token_vectors, Wq)   
    K = np.dot(token_vectors, Wk)
    V = np.dot(token_vectors, Wv)

    scores = np.dot(Q, K.T) / np.sqrt(embedding_table_dimensions)

    # mask the upper triangle — token i cannot see token j > i
    mask = np.triu(np.full_like(scores, -1e9), k=1)
    scores = scores + mask

    # turn scores into probabilities — each row sums to 1
    # each row is one token's attention distribution over all previous tokens
    attention_weights = np.array([softmax(row) for row in scores])

    # weighted sum of values
    # each token's new vector is a mix of all the value vectors
    # weighted by how much it attended to each token
    context_vectors = np.dot(attention_weights, V)   # [2, 3]

    token_vector = context_vectors[-1]   # [3]
    pre_relu     = np.dot(token_vector, weights) + biases

    relu_activation = np.maximum(0, pre_relu)
    output          = softmax(np.dot(relu_activation, weights_out) + biases_out)

    loss            = -np.log(output[correct_id])
    predicted_word  = list(tokens.keys())[np.argmax(output)]

    if step % 20 == 0:
        print(f"step {step:3d}  loss: {loss:.4f}  prediction: {predicted_word}")

    # ── backward pass ─────────────────────────

    scores_copy              = output.copy()
    scores_copy[correct_id] -= 1                          # gradient of loss w.r.t scores

    d_W_out    = np.outer(relu_activation, scores_copy)   # [6, 4]
    d_B_out    = scores_copy                              # [4]
    d_hidden   = np.dot(scores_copy, weights_out.T)       # [6]
    d_pre_relu = d_hidden * (pre_relu > 0)                # [6] — zero out dead neurons
    d_W = np.outer(token_vector, d_pre_relu)   # [3, 6]

    d_biases   = d_pre_relu                               # [6]

    # ── weight update ─────────────────────────

    weights_out -= learning_rate * d_W_out
    biases_out  -= learning_rate * d_B_out
    weights     -= learning_rate * d_W
    biases      -= learning_rate * d_biases

    # embedding row for the input token also gets updated
    embedding_table[token_id[-1]] -= learning_rate * np.dot(d_pre_relu, weights.T)


print()
print("final probabilities:")
for word, prob in zip(tokens.keys(), output):
    print(f"  {word:>8}: {prob:.1%}")

print("final embedding table:")
print(embedding_table)
print("final weights:")
print(weights)
print("final biases:")
print(biases)
print("final weights_out:")
print(weights_out)
print("final biases_out:")
print(biases_out)