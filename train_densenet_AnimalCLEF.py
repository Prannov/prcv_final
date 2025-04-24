"""
Animal CLEF 2025 - DenseNet Training

This script trains a DenseNet model on the Animal CLEF 2025 wildlife dataset,
consisting of dogs, lions, and macaques. It loads pretrained weights and fine-tunes
only the classifier layer to adapt to the animal classification task.

Author: Jae Hun Cho
Date: April 2025
"""


# Standard library imports
import os
import sys

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local imports
from wildlife_datasets import datasets, loader

# Configuration
DATASET_CLASSES = [
    datasets.MacaqueFaces,
    datasets.LionData,
    datasets.DogFaceNet
]

# Default parameters
EPOCHS = 10


class AnimalDataset(Dataset):
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


def get_densenet(num_classes):
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model


if __name__ == "__main__":
    # Get epochs from command line
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCHS
    
    # Load and prepare data
    merged_df = load_and_merge(DATASET_CLASSES)

    # Label encoding
    merged_df['identity'] = merged_df['identity'].astype(str)
    label2idx = {label: idx for idx, label in enumerate(sorted(merged_df['identity'].unique()))}
    merged_df['label'] = merged_df['identity'].map(label2idx)

    # Split data
    train_df, test_df = train_test_split(
        merged_df, 
        test_size=0.2, 
        stratify=merged_df['label'], 
        random_state=42
    )

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create datasets and loaders
    train_dataset = AnimalDataset(train_df, transform=transform)
    test_dataset = AnimalDataset(test_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(label2idx)
    model = get_densenet(num_classes)
    model = model.to(device)
    
    # Freeze all layers except classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
    
    print('Model and data ready for training!')
    print(f'Starting training for {epochs} epochs...')
    
    # Training loop
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

    # Save trained model
    torch.save(model.state_dict(), 'densenet_model.pth')
    print('Model saved to densenet_model.pth')