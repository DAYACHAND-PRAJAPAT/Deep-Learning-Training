# model.py
import torch.nn as nn
from torchvision import models

class PlantDiseaseResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pre-trained ResNet-18 model for transfer learning
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer for the new number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)