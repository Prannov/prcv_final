import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Always set this for debugging (shows exact crash line)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Paths
CSV_PATH = "/kaggle/input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv"
TRAIN_DIR = "/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/train/720p"
VAL_DIR   = "/kaggle/input/fungi-clef-2025/images/FungiTastic-FewShot/val/720p"

# Load and encode labels
df = pd.read_csv(CSV_PATH).dropna(subset=["filename", "scientificName"])
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["scientificName"])
filename_to_label = dict(zip(df["filename"], df["label"]))

# Use .max() + 1 to prevent out-of-bounds
num_classes = df["label"].max() + 1
print(f"num_classes = {num_classes}")

# Dataset class with strict filtering
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

# Transforms for Inception (299x299 required)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# DataLoaders
train_ds = FungiDataset(TRAIN_DIR, transform)
val_ds   = FungiDataset(VAL_DIR, transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# Inception v3 model with aux_logits
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)  # (also update pretrained->weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)  # <-- this line added
model = model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with main + aux loss
for epoch in range(1, 31):  # Change to 30 for full training
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Inception v3 returns 2 outputs during training
        outputs, aux_outputs = model(imgs)
        loss_main = criterion(outputs, labels)
        loss_aux  = criterion(aux_outputs, labels)
        loss = loss_main + 0.4 * loss_aux

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[Inception] Epoch {epoch} Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "inception_fungi_model.pth")
