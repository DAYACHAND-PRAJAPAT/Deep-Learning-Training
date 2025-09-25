from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def sentences(filename):
    with open(filename, 'r') as f:
        data = f.read()
        f.close()
    sentences  = re.split(r'(?<=[.!?])\s+', data.strip())
    sentences = [s for s in sentences if s]
    return sentences

sentences = sentences('knowledge.txt')
embeddings = model.encode(sentences, normalize_embeddings=True)

print("Embedding shape:", embeddings.shape)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

query = input("Enter your query: ")
while query != "quit":
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), 2)
    match = [sentences[i] for i in I[0]]
    print("Best match:", match[0])
    query = input("Enter your query: ")
