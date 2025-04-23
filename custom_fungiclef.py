#Prannov Jamadagni
#PRCV

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Debug
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Paths
CSV_PATH = "/kaggle/input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv"
TRAIN_DIR = "/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/train/720p"
VAL_DIR   = "/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/val/720p"

# Read and encode labels
df = pd.read_csv(CSV_PATH).dropna(subset=["filename", "scientificName"])
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["scientificName"])
filename_to_label = dict(zip(df["filename"], df["label"]))
num_classes = df["label"].max() + 1
print(f"Number of classes: {num_classes}")

# Custom dataset
class FungiDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and f in filename_to_label
        ])
        self.transform = transform

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        image = Image.open(os.path.join(self.image_dir, file_name)).convert("RGB")
        label = filename_to_label[file_name]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms for 150x150 image input
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Loaders
train_ds = FungiDataset(TRAIN_DIR, transform)
val_ds   = FungiDataset(VAL_DIR, transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# Custom CNN model
class CustomFungiCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomFungiCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Input: 150x150 → After 3 pools: 18x18
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomFungiCNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1, 11): 
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Custom CNN] Epoch {epoch}, Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "custom_fungi_cnn.pth")

model.eval()
