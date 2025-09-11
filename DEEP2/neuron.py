import torch
import torch.nn as nn

x = torch.tensor([1.0,2.0,3.0])
w = torch.tensor([0.4,0.3,0.2])
b = torch.tensor(0.5)

z = torch.dot(x,w) + b
print("Linear : ",z.item())

print("Sigmoid : ",torch.sigmoid(z).item())
print("Tanh : ",torch.tanh(z).item())
print("ReLU : ",torch.relu(z).item())

layer = nn.Linear(3, 1)
output = layer(x)
print("Forward Propogation:", output.item())