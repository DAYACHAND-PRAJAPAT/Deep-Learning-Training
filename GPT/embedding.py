from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["The sky is blue.", "Dogs are friendly.", "Transformers power GPT."]
embeddings = model.encode(sentences, normalize_embeddings=True)

print("Embedding shape:", embeddings.shape)
