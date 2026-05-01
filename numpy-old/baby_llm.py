import torch
import torch.nn as nn
import torch.nn.functional as F

shakespeare_filepath = "input.txt"
with open(shakespeare_filepath, "r") as shakespeare_file:
    training_text = shakespeare_file.read()

print(f"Dataset size: {len(training_text):,} characters")

# ── Device ────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

# ── Hyperparameters ───────────────────────────────────────────────
CONTEXT_LENGTH       = 64
EMBEDDING_DIMENSION  = 128
NUMBER_OF_HEADS      = 4
NUMBER_OF_LAYERS     = 3
LEARNING_RATE        = 1e-3
NUMBER_OF_EPOCHS     = 5000
BATCH_SIZE           = 32

# ── Tokenize ──────────────────────────────────────────────────────
unique_characters  = sorted(set(training_text))
vocabulary_size    = len(unique_characters)
character_to_index = {character: index for index, character in enumerate(unique_characters)}
index_to_character = {index: character for character, index in character_to_index.items()}
tokenized_data     = torch.tensor([character_to_index[character] for character in training_text], dtype=torch.long)

split_index        = int(len(tokenized_data) * 0.9)
training_tokens    = tokenized_data[:split_index]
validation_tokens  = tokenized_data[split_index:]

print(f"Vocabulary size: {vocabulary_size} unique characters")

def get_random_batch(token_split):
    random_offsets   = torch.randint(len(token_split) - CONTEXT_LENGTH, (BATCH_SIZE,))
    input_sequences  = torch.stack([token_split[offset : offset + CONTEXT_LENGTH    ] for offset in random_offsets])
    target_sequences = torch.stack([token_split[offset + 1 : offset + CONTEXT_LENGTH + 1] for offset in random_offsets])
    return input_sequences.to(device), target_sequences.to(device)

# ── Layers ────────────────────────────────────────────────────────
token_embedding    = nn.Embedding(vocabulary_size, EMBEDDING_DIMENSION).to(device)
position_embedding = nn.Embedding(CONTEXT_LENGTH,  EMBEDDING_DIMENSION).to(device)
transformer_encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=EMBEDDING_DIMENSION, nhead=NUMBER_OF_HEADS,
        dim_feedforward=EMBEDDING_DIMENSION * 4, dropout=0.1, batch_first=True
    ),
    num_layers=NUMBER_OF_LAYERS
).to(device)
output_projection  = nn.Linear(EMBEDDING_DIMENSION, vocabulary_size).to(device)

all_parameters = (
    list(token_embedding.parameters())     +
    list(position_embedding.parameters())  +
    list(transformer_encoder.parameters()) +
    list(output_projection.parameters())
)
optimizer = torch.optim.AdamW(all_parameters, lr=LEARNING_RATE)

# ── Training ──────────────────────────────────────────────────────
causal_mask = nn.Transformer.generate_square_subsequent_mask(CONTEXT_LENGTH).to(device)

for epoch_number in range(NUMBER_OF_EPOCHS):
    input_sequences, target_sequences = get_random_batch(training_tokens)

    token_embeddings    = token_embedding(input_sequences)
    position_indices    = torch.arange(CONTEXT_LENGTH, device=device)
    position_embeddings = position_embedding(position_indices)
    combined_embeddings = token_embeddings + position_embeddings

    hidden_states     = transformer_encoder(combined_embeddings, mask=causal_mask, is_causal=True)
    vocabulary_logits = output_projection(hidden_states)

    loss = F.cross_entropy(vocabulary_logits.view(-1, vocabulary_size), target_sequences.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch_number + 1) % 500 == 0:
        # Validation loss
        with torch.no_grad():
            validation_input_sequences, validation_target_sequences = get_random_batch(validation_tokens)
            validation_embeddings = token_embedding(validation_input_sequences) + position_embedding(torch.arange(CONTEXT_LENGTH, device=device))
            validation_hidden     = transformer_encoder(validation_embeddings, mask=causal_mask, is_causal=True)
            validation_logits     = output_projection(validation_hidden)
            validation_loss       = F.cross_entropy(validation_logits.view(-1, vocabulary_size), validation_target_sequences.view(-1))

        print(f"epoch {epoch_number + 1:>5}  train loss {loss.item():.4f}  val loss {validation_loss.item():.4f}")

# ── Generate ──────────────────────────────────────────────────────
def generate_text(prompt_string, max_new_tokens=200):
    generated_indices = torch.tensor(
        [[character_to_index[character] for character in prompt_string]],
        dtype=torch.long, device=device
    )
    for _ in range(max_new_tokens):
        cropped_indices      = generated_indices[:, -CONTEXT_LENGTH:]
        cropped_embeddings   = token_embedding(cropped_indices) + position_embedding(torch.arange(cropped_indices.shape[1], device=device))
        cropped_hidden       = transformer_encoder(cropped_embeddings, mask=nn.Transformer.generate_square_subsequent_mask(cropped_indices.shape[1], device=device), is_causal=True)
        cropped_logits       = output_projection(cropped_hidden)
        next_token_probs     = F.softmax(cropped_logits[:, -1, :] / 0.8, dim=-1)
        next_token_index     = torch.multinomial(next_token_probs, num_samples=1)
        generated_indices    = torch.cat([generated_indices, next_token_index], dim=1)
    return "".join(index_to_character[index.item()] for index in generated_indices[0])

print("\n── Generated Shakespeare ──")
print(generate_text("ROMEO:", max_new_tokens=300))