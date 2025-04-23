import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from wildlife_datasets import datasets, loader
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
import os

import sys

# ---- DATASET ----
class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if row['dataset'].lower() == 'dogfacenet':
            img_path = os.path.join('data', 'DogFaceNet', row['path'])
        elif row['dataset'].lower() == 'liondata':
            img_path = os.path.join('data', 'LionData', row['path'])
        elif row['dataset'].lower() == 'macaquefaces':
            img_path = os.path.join('data', 'MacaqueFaces', row['path'])
        else:
            raise ValueError(f"Unknown dataset: {row['dataset']}")
        image = Image.open(img_path).convert('RGB')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# ---- DATA PREP ----
def load_and_merge(classes):
    dfs = []
    for cls in classes:
        try:
            d = loader.load_dataset(cls, 'data', 'dataframes')
            df = d.df.copy()
            df['dataset'] = cls.__name__
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {cls.__name__}: {e}")
    if not dfs:
        raise RuntimeError("No datasets could be loaded. Please check your data folders.")
    return pd.concat(dfs, ignore_index=True)

DATASET_CLASSES = [datasets.DogFaceNet, datasets.LionData, datasets.MacaqueFaces]
merged_df = load_and_merge(DATASET_CLASSES)
merged_df['identity'] = merged_df['identity'].astype(str)
label2idx = {label: idx for idx, label in enumerate(sorted(merged_df['identity'].unique()))}
merged_df['label'] = merged_df['identity'].map(label2idx)
train_df, test_df = train_test_split(merged_df, test_size=0.2, stratify=merged_df['label'], random_state=42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

train_dataset = AnimalDataset(train_df, transform=transform)
test_dataset = AnimalDataset(test_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# ---- CUSTOM CNN ----
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(label2idx)
model = ImprovedCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

from tqdm import tqdm

if __name__ == "__main__":
    epochs = int(sys.argv[1])  # You can change this or make it a command-line argument

    print('Model and data ready for training!')
    print(f'Starting training for {epochs} epochs...')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_iter.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    print('Training complete!')
    torch.save(model.state_dict(), 'custom_cnn.pth')
    print('Model saved to custom_cnn.pth')
