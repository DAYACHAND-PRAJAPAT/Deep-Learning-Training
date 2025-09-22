import torch
import torch.nn as nn

# Step 1: Input words
sentence = ["I", "love", "my", "dog"]
print("Input:", sentence)

# --- Step 2: Tensor Conversion ---
# Simple word-to-index mapping
word_to_index = {"I": 0, "love": 1, "my": 2, "dog": 3}
indices = [word_to_index[word] for word in sentence]

# Convert indices to tensor
input_tensor = torch.tensor(indices)
print("Tensor Conversion:", input_tensor)

# Embedding layer to convert word indices → dense vectors
embedding = nn.Embedding(num_embeddings=len(word_to_index), embedding_dim=4)
embedded = embedding(input_tensor)
print("Embedded Tensor Representation:\n", embedded)

# --- Step 3: Translate (simulate translation using a linear layer) ---
translator = nn.Linear(4, 4)  # pretend this is the translation model
translated = translator(embedded)
print("Translated Tensor:\n", translated)

# --- Step 4: Decode (map tensor outputs back to tokens) ---
output_vocab = ["Yo", "Amo", "Mi", "Perro"]  # Spanish for demo
decoded_indices = torch.argmax(translated, dim=1) % len(output_vocab)
decoded_words = [output_vocab[idx] for idx in decoded_indices]
print("Decoded Words:", decoded_words)

# --- Step 5: Combine ---
final_translation = " ".join(decoded_words)
print("Final Translation:", final_translation)