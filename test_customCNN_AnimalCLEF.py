"""
Animal CLEF 2025 - Custom CNN Testing

This script evaluates the trained custom CNN model on the Animal CLEF 2025 test set.
It loads the saved model weights, performs inference on the test data, and reports
accuracy metrics along with details about misclassified samples.

Author: Jae Hun Cho
Date: April 2025
"""


# Standard library imports
import os

# Third-party imports
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Local imports
from train_customCNN_AnimalCLEF import ImprovedCNN, label2idx, test_dataset, test_df
from wildlife_datasets import datasets, loader


if __name__ == "__main__":
    # Initialize model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(label2idx)
    model = ImprovedCNN(num_classes)
    model.load_state_dict(torch.load('custom_cnn.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # Testing
    total = len(test_dataset)
    correct = 0
    misclassified = []
    
    for idx in range(total):
        # Get image and perform inference
        img, label = test_dataset[idx]
        input_img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_img)
            pred_label = output.argmax(dim=1).item()
            
        # Determine true species
        true_dataset_name = test_df.iloc[idx]['dataset']
        if 'dog' in true_dataset_name.lower():
            true_species = 'dog'
        elif 'lion' in true_dataset_name.lower():
            true_species = 'lion'
        elif 'macaque' in true_dataset_name.lower():
            true_species = 'macaque'
        else:
            true_species = true_dataset_name.lower()
            
        # Determine predicted species
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

        # Track accuracy
        if pred_species == true_species:
            correct += 1
        else:
            misclassified.append((idx, true_species, pred_species))
            
    # Print results
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Number of misclassified samples: {len(misclassified)}")
    
    if misclassified:
        print("Some misclassified samples (index, true, pred):")
        for idx, true_s, pred_s in misclassified[:30]:
            print(f"Index: {idx}, True: {true_s}, Pred: {pred_s}")