import numpy as np


# 1. convert words to tokens

# basically assign every word a unique number
tokens = {
    "cat": 0,
    "dog": 1,
    "king": 2,
    "queen": 3,
}

token_id = tokens["dog"] # token_id = 1

# 2. create an embedding table
# we need to turn a token into a vector of numbers
# this embedding table serves as a dictionary
# attention will come later to provide context

# embedding is really general relationships to save compute before doing attention
# [
#   [0.2, 0.9, 0.1],   ← cat
#   [0.3, 0.8, 0.2],   ← dog
#   [0.1, 0.05, 0.9],  ← king
#   [0.0, 0.1, 0.85],  ← queen
# ]

# so every token is a row in the embedding table, represented by a vector of numbers
# during training, the model can adjust these vectors to be closer or farther


# the more dimensions (columns) you give the embedding table, the more information you can give to the model
# about the granualar relationships between words
embedding_table_dimensions = 3

# the numpy below is the same as below, it just generates random numbers for each of the vectors
# embedding_table = 
# [[-0.88242251  0.75637682 -1.0488347 ] ← cat
#  [-1.7702211  -1.48023958  0.10160537] ← dog
#  [-0.58799501  0.99543889  1.59246246] ← king
#  [ 0.17856731 -0.5651276  -1.19920559]] ← queen

embedding_table = np.random.randn(len(tokens), embedding_table_dimensions)
# print("embedding_table: ", embedding_table)

# calculate token vector 
token_vector = embedding_table[token_id] # dog's vector


# 3. feed the token's vector through the first layer in the neural network
# this layer has 6 neurons in it
neuron_count_per_layer = 6

# for each neuron, there is a weight for every column (dimension) in the embedding table
# but there is only one bias number for each neuron

weights = np.random.randn(embedding_table_dimensions, neuron_count_per_layer)
# print("weights: ", weights)
#     neuron0     neuron1      neuron2     neuron3     neuron4     neuron5
# [[ 2.23063535 -1.16859472 -0.56835858 -0.99788731 -0.3496884   1.2225576 ]
#  [ 1.04338873 -1.11165662 -0.03213841  0.35260761  0.73204122  0.04414724]
#  [-1.62763605  1.3088863  -1.18127857 -0.96459291  0.98635657 -1.35887833]]

biases = np.zeros(neuron_count_per_layer)  

# print("biases: ", biases)
# one for each neuron
# [0. 0. 0. 0. 0. 0.]


# dot product of the token's vector and the weights, plus the biases
#  remember: [0.3, 0.8, 0.2],   ← dog's vector


# dog's vector * weight of one neuron (neuron 0)
# [0.3, 0.8, 0.2] * [2.23063535, 1.04338873, -1.62763605]

# dot product of the token's vector and the weights, plus the biases
# [0.3, 0.8, 0.2] * [2.23063535, 1.04338873, -1.62763605] + [0. 0. 0.]
# = [0.3 * 2.23063535 + 0.8 * 1.04338873 + 0.2 * -1.62763605] + [0. 0. 0.]
# = [0.66919065 + 0.83471104 - 0.32552321] + [0. 0. 0.]
# = [1.17837848] + [0. 0. 0.]
# = [1.17837848]

# do the same for the other 5 neurons

# we can add the bias at the end
# [1.17837848, ..., ...] + [0. 0. 0. ...]
# each of these vectors have 6 numbers in them

# @ is the matrix multiplication operator
pre_relu = np.dot(token_vector, weights) + biases

print(pre_relu)
# to get
# [ 1.17837848  ... ]


# pass through ReLU activation function, all negative values are set to 0
relu_activation = np.maximum(0, pre_relu)
print(relu_activation)


# 4. feed the output through the second outputlayer in the neural network

# four output neurons, one for each token

# each neuron has 6 weights this time, since there were 6 neurons in the previous layer
# one weight for each input a neuron receives


# weights for the output layer
weights_out = np.random.randn(6, 4)
# biases for the output layer
biases_out = np.zeros(4)

print("weights_out: ", weights_out)
print("biases_out: ", biases_out)


# the output neurons don't have a ReLU activation function, they are just the raw scores
# this will allow softmax to work properly
output = np.dot(relu_activation, weights_out) + biases_out

def softmax(scores):
    # subtract the max for numerical stability (prevents huge numbers)
    e = np.exp(scores - np.max(scores))
    return e / e.sum()

print("output: ", output)

output = softmax(output)

print("softmax output: ", output)

# find the token id (column) with the highest score
predicted_token_id = np.argmax(output)
print("predicted_token", predicted_token_id)
print("predicted word", list(tokens.keys())[predicted_token_id])


# lets say the correct word is "cat"
loss = -np.log(output[tokens["cat"]])

# this is the "cross entropy loss", using log to punish low probability predictions harder
print(loss)


# make a copy, we are about to do backpropagation, and we don't want to modify the original output
scores_copy = output.copy()

# we need to subtract 1 the score/probability for the correct word 
# this is because we are saying the probability of this word should be higher (since the result will be negative)
# for the other words, we are saying the probability should be lower (since the result will be positive)
scores_copy[tokens["cat"]] -= 1



# so we have a list of scores (which is just probabilities of the words with the correct word subtracted by 1)
# now we can figure out how much the weight of the previous layer contributed to the score by multiplying the score by the weight

# hidden   = [h0, h1, h2, h3, h4, h5]      # 6 numbers
# scores = [s0, s1, s2, s3]              # 4 numbers

#         s0       s1       s2       s3
# h0  [h0*s0,  h0*s1,  h0*s2,  h0*s3]
# h1  [h1*s0,  h1*s1,  h1*s2,  h1*s3]
# h2  [h2*s0,  h2*s1,  h2*s2,  h2*s3]
# h3  [h3*s0,  h3*s1,  h3*s2,  h3*s3]
# h4  [h4*s0,  h4*s1,  h4*s2,  h4*s3]
# h5  [h5*s0,  h5*s1,  h5*s2,  h5*s3]

d_W_out = np.outer(relu_activation, scores_copy)

print("d_W_out: ", d_W_out)


# the bias always contributes 100% to the score (there is no multiplier, it is a constant add/subtract)
# so the gradient would just be the scores_copy

d_B_out = scores_copy

# we transpose the weights_out matrix so we can multiply it by the d_W_out matrix
# this will give us the gradient of the first layer weights and biases


# we can't just use d_hidden
# how much did each neuron contribute to the final error
d_hidden = np.dot(scores_copy, weights_out.T)
print("d_hidden: ", d_hidden)

# we need to use the pre_relu matrix from before to figure out how much the weights of the previous layer contributed to the score
d_pre_relu = d_hidden * (pre_relu > 0) # set the gradient to 0 for negative values, as they didn't contribute anything to the score as the neuron didn't fire
print("d_pre_relu: ", d_pre_relu)

# reconstruct the weights matrix (with derivative error) from the token's vector and the neurons' errors
d_W = np.outer(token_vector, d_pre_relu)

print("d_W: ", d_W)

learning_rate = 0.01

# we can update the weights and biases based on the gradient
weights_out -= learning_rate * d_W_out
biases_out  -= learning_rate * d_B_out
weights     -= learning_rate * d_W
biases      -= learning_rate * d_pre_relu

# how much did the token's vector contribute to each neuron in the first layer's error
d_embedding_table = np.dot(d_pre_relu, weights.T)
embedding_table[token_id] -= learning_rate * d_embedding_table

