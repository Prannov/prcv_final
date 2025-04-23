import torch
from train_densenet_AnimalCLEF import get_densenet
from wildlife_datasets import datasets, loader
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import os

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

def get_test_data():
    DATASET_CLASSES = [
        datasets.MacaqueFaces,
        datasets.LionData,
        datasets.DogFaceNet
    ]
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
    merged_df['identity'] = merged_df['identity'].astype(str)
    label2idx = {label: idx for idx, label in enumerate(sorted(merged_df['identity'].unique()))}
    merged_df['label'] = merged_df['identity'].map(label2idx)
    train_df, test_df = train_test_split(merged_df, test_size=0.2, stratify=merged_df['label'], random_state=42)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    test_dataset = AnimalDataset(test_df, transform=transform)
    return test_df, test_dataset, label2idx

if __name__ == "__main__":
    test_df, test_dataset, label2idx = get_test_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(label2idx)
    model = get_densenet(num_classes)
    model.load_state_dict(torch.load('densenet_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    total = len(test_dataset)
    correct = 0
    misclassified = []
    for idx in range(total):
        img, label = test_dataset[idx]
        input_img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_img)
            pred_label = output.argmax(dim=1).item()
        true_dataset_name = test_df.iloc[idx]['dataset']
        if 'dog' in true_dataset_name.lower():
            true_species = 'dog'
        elif 'lion' in true_dataset_name.lower():
            true_species = 'lion'
        elif 'macaque' in true_dataset_name.lower():
            true_species = 'macaque'
        else:
            true_species = true_dataset_name.lower()
        pred_identity = list(label2idx.keys())[list(label2idx.values()).index(pred_label)]
        pred_rows = test_df[test_df['identity'] == pred_identity]
        if not pred_rows.empty:
            pred_dataset_name = pred_rows['dataset'].iloc[0]
        else:
            pred_dataset_name = 'unknown'
        if 'dog' in pred_dataset_name.lower():
            pred_species = 'dog'
        elif 'lion' in pred_dataset_name.lower():
            pred_species = 'lion'
        elif 'macaque' in pred_dataset_name.lower():
            pred_species = 'macaque'
        else:
            pred_species = pred_dataset_name.lower()

        if pred_species == true_species:
            correct += 1
        else:
            misclassified.append((idx, true_species, pred_species))
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Number of misclassified samples: {len(misclassified)}")
    if misclassified:
        print("Some misclassified samples (index, true, pred):")
        for idx, true_s, pred_s in misclassified[:30]:
            print(f"Index: {idx}, True: {true_s}, Pred: {pred_s}")