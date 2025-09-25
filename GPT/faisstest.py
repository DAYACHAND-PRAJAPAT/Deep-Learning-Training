from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["The sky is blue.", "Dogs are friendly.", "Transformers power GPT."]
embeddings = model.encode(sentences, normalize_embeddings=True)

print("Embedding shape:", embeddings.shape)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

query = "what powers GPT models?"
q_emb = model.encode([query], normalize_embeddings=True)

D, I = index.search(np.array(q_emb).astype("float32"), 2)
print("Best matches:", [sentences[i] for i in I[0]])