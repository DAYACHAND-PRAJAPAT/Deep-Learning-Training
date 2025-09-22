import torch
import torch.nn as nn
import torch.optim as optim

sentences = ["i love this movie", "this film is terrible", "an awesome film", "i hated it"]
labels = [1,0,1,0]

vocab = {w:i+1 for i,w in enumerate(set(" ".join(sentences).split()))}
maxlen = 6

def encode(s):
    ids = [vocab[w] for w in s.split()]
    return ids[:maxlen] + [0]*(maxlen-len(ids))

X = torch.tensor([encode(s) for s in sentences])
y = torch.tensor(labels)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.linear = nn.Linear(hidden, 2)
    def forward(self, x):
        e = self.embed(x)
        _, (h, _) = self.lstm(e)
        return self.linear(h[-1])

model = LSTMClassifier(len(vocab)+1)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch+1)%10==0:
        pred = logits.argmax(1)
        print(f"Epoch {epoch+1}, acc={(pred==y).float().mean().item():.2f}")
