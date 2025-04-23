import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# File paths
CSV_PATH = "/kaggle/input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv"
TRAIN_DIR = "/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/train/720p"
VAL_DIR   = "/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/val/720p"

# Load label mapping
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['filename', 'scientificName'])
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['scientificName'])
filename_to_label = dict(zip(df['filename'], df['label']))

# Dataset class
class FungiImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        label = filename_to_label.get(file_name, -1)  # -1 for test if not labeled
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets & DataLoaders
train_ds = FungiImageDataset(TRAIN_DIR, transform)
val_ds = FungiImageDataset(VAL_DIR, transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Load DenseNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, df['label'].nunique())
model = model.to(device)

# Optimizer and Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(30):  # increase for full training
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "densenet_fungi_model.pth")

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading broken images

# New Safe Validation Dataset Class
class SafeValFungiDataset(Dataset):
    def __init__(self, image_dir, filename_to_label, transform=None):
        self.image_dir = image_dir
        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.filename_to_label = filename_to_label
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, file_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping corrupted validation image: {file_name} | Error: {e}")
            image = Image.new("RGB", (224, 224), (255, 255, 255))  # dummy white image

        label = self.filename_to_label.get(file_name, -1)
        if self.transform:
            image = self.transform(image)
        return image, label

# Use this only for validation
safe_val_dataset = SafeValFungiDataset(VAL_DIR, filename_to_label, transform)
safe_val_loader = DataLoader(safe_val_dataset, batch_size=32, shuffle=False)
