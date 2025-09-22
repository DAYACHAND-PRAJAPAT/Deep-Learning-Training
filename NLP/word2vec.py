import torch
import torch.nn as nn
import torch.optim as optim

corpus = "king queen man woman"

words = corpus.split()
vocab = list(set(words))
word_to_ix = {w: i for i,w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

pairs = [("king", "queen"), ("queen", "king"),
         ("man", "woman"), ("woman", "man")]

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.out(self.emb(x))
    

model = Word2Vec(len(vocab), 5)
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.05)

for epochs in range(50):
    total_loss = 0
    for center, context in pairs:
        context_ix = torch.tensor([word_to_ix[context]])
        center_ix = torch.tensor([word_to_ix[center]])

        opt.zero_grad()
        pred = model(center_ix)
        loss = loss_fn(pred, context_ix)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epochs} trainned (Loss = {total_loss:.2f})")

emmbedings = model.emb.weight.data
print("Vector for king", emmbedings[word_to_ix['king']])
print("Vector for queen", emmbedings[word_to_ix['queen']])
