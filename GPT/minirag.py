import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

docs = [
    "GPT models are powered by transformers.",
    "FAISS is a library for efficient similarity search.",
    "OpenAI created the GPT family of models.",
    "Neural networks learn from large datasets."
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs, normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype("float32"))

generator = pipeline("text-generation", model="gpt2")

def ask(query):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), 1)
    context = docs[I[0][0]]
    prompt = f"{context}\nQ: {query}\nA:"
    return generator(prompt, max_length=60, num_return_sequences=1, do_sample=True)[0]["generated_text"]

print(ask("Explain GPT"))
