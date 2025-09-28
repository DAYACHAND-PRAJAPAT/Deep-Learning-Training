# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import os
from model import PlantDiseaseResNet18
from utils import train_transform, val_transform, PLANT_DISEASE_CLASSES

def train(
    data_dir="./data",
    epochs=5,
    batch_size=16,
    lr=0.001,
    weight_decay=5e-4,
    device=None,
    save_path="saved_models/plant_disease_model_small.pth"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_ds = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(PLANT_DISEASE_CLASSES)
    model = PlantDiseaseResNet18(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch} validation accuracy: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"Saved new best model to {save_path} (val_acc={val_acc:.2f}%)")

    print("Training finished. Best val acc:", best_acc)

if __name__ == "__main__":
    train(epochs=5, batch_size=16, lr=0.001, save_path="saved_models/plant_disease_model_small.pth")