import numpy as np


# 1. convert words to tokens

# basically assign every word a unique number
tokens = {
    "cat": 0,
    "dog": 1,
    "king": 2,
    "queen": 3,
}

token_id = tokens["dog"] # token_id = 2


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

print("biases: ", biases)
# one for each neuron
# [0. 0. 0. 0. 0. 0.]


