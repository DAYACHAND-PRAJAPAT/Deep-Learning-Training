# utils.py
import torch
from torchvision import transforms
from PIL import Image
import io

# Limited classes for reduced computation
PLANT_DISEASE_CLASSES = [
    "Apple_scab",
    "Apple_healthy",
]

# Standard transforms for images using ImageNet mean/std
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def preprocess_image_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = val_transform(img).unsqueeze(0)
    return tensor