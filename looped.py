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

# input: "dog" → predict "cat"
token_id       = tokens["dog"]
correct_id     = tokens["cat"]


# ─────────────────────────────────────────────
# 2. weights (initialized once, updated each step)
# ─────────────────────────────────────────────

embedding_table_dimensions = 3
neuron_count               = 6
vocab_size                 = len(tokens)

embedding_table = np.random.randn(vocab_size, embedding_table_dimensions) * 0.1
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

    token_vector    = embedding_table[token_id]
    pre_relu        = np.dot(token_vector, weights) + biases
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
    d_W        = np.outer(token_vector, d_pre_relu)       # [3, 6]
    d_biases   = d_pre_relu                               # [6]

    # ── weight update ─────────────────────────

    weights_out -= learning_rate * d_W_out
    biases_out  -= learning_rate * d_B_out
    weights     -= learning_rate * d_W
    biases      -= learning_rate * d_biases

    # embedding row for the input token also gets updated
    embedding_table[token_id] -= learning_rate * np.dot(d_pre_relu, weights.T)


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