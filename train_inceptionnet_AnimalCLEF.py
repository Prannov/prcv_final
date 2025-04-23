from wildlife_datasets import datasets, loader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ---- CONFIG ----
# Only use CowDataset and DogFaceNet for now
DATASET_CLASSES = [
    datasets.DogFaceNet,
    datasets.LionData,
    datasets.MacaqueFaces
]

# ---- LOAD AND MERGE DATA ----
import pandas as pd

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

merged_df = load_and_merge(DATASET_CLASSES)

# ---- LABEL ENCODING ----
merged_df['identity'] = merged_df['identity'].astype(str)
label2idx = {label: idx for idx, label in enumerate(sorted(merged_df['identity'].unique()))}
merged_df['label'] = merged_df['identity'].map(label2idx)

# ---- DATASET ----
class AnimalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Build correct image path for each dataset

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

# ---- SPLIT ----
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    return train_df, test_df

# ---- MODEL ----
from torchvision.models import Inception_V3_Weights

def get_inception(num_classes):
    # Use the recommended weights argument and aux_logits=True for pretrained
    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the classifier head (trainable)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # Optionally, disable the auxiliary classifier head after loading weights
    model.aux_logits = False
    return model

if __name__ == "__main__":
    merged_df = load_and_merge(DATASET_CLASSES)

    # ---- LABEL ENCODING ----
    merged_df['identity'] = merged_df['identity'].astype(str)
    label2idx = {label: idx for idx, label in enumerate(sorted(merged_df['identity'].unique()))}
    merged_df['label'] = merged_df['identity'].map(label2idx)

    # ---- SPLIT ----
    train_df, test_df = train_test_split(merged_df, test_size=0.2, stratify=merged_df['label'], random_state=42)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    train_dataset = AnimalDataset(train_df, transform=transform)
    test_dataset = AnimalDataset(test_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    num_classes = len(label2idx)
    model = get_inception(num_classes)

    # ---- TRAINING AND EVAL SCAFFOLD ----
    # Add your training and evaluation loop here
    print('Model and data ready for training!')

    # ---- TRAINING LOOP ----
    import matplotlib.pyplot as plt
    import random
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
    import sys
    # Allow user to specify epochs as a third argument
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except Exception:
            print('Invalid epoch argument, using default.')
            epochs = 10
    else:
        epochs = 10
    print(f'Starting training for {epochs} epochs...')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):  # Inception returns (main, aux)
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_iter.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    print('Training complete!')

    # ---- SAVE TRAINED MODEL ----
    torch.save(model.state_dict(), 'inception_model.pth')
    print('Model saved to inception_model.pth')