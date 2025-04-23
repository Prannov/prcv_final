import torch
from train_customCNN_AnimalCLEF import ImprovedCNN, label2idx, test_dataset, test_df

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(label2idx)
model = ImprovedCNN(num_classes)
model.load_state_dict(torch.load('custom_cnn.pth', map_location=device))
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
