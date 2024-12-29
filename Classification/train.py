import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from PIL import Image
from data import CellDataset
import helpers


MODEL = 'vit'  # vgg, vit, resnet
PRETRAINED = True
NUM_CLASSES = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = helpers.get_model(MODEL, PRETRAINED, NUM_CLASSES).to(device)

helpers.set_seed(123)

transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]), 
])

train_dataset = CellDataset(main_dir='final_dataset/train', transform=transforms)
val_dataset = CellDataset(main_dir='final_dataset/val', transform=transforms)
test_dataset = CellDataset(main_dir='final_dataset/test', transform=transforms)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def train():
    try:
        os.mkdir('checkpoints')
        print(f"Directory '{'checkpoints'}' created successfully.")
    except FileExistsError:
        print(f"Directory '{'checkpoints'}' already exists.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)

    num_epochs = 200
    max_accuracy = 0  
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_dataset)
        val_accuracy = 100 * np.mean(np.array(all_val_predictions) == np.array(all_val_labels))
        val_precision = precision_score(all_val_labels, all_val_predictions, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_predictions, average='weighted')
        val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')
        val_mcc = matthews_corrcoef(all_val_labels, all_val_predictions)

        print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy}%, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, MCC: {val_mcc:.4f}")

        if val_accuracy > max_accuracy:
            torch.save(model, f'checkpoints/{MODEL}')
            max_accuracy = val_accuracy
            print("Saved checkpoint")
            early_stop_counter = 0

        early_stop_counter += 1
        if early_stop_counter == 30:
            print("Early stopping triggered.")
            break

        scheduler.step(val_loss)


def test():
    model = torch.load(f'checkpoints/{MODEL}')
    model.eval()

    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_predictions)

    print(f"TEST -> Accuracy: {accuracy}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}\n")
    return accuracy, precision, recall, f1, mcc

def get_labels(main_folder, output_file):
    model = torch.load(f'checkpoints/best_vit')
    model.eval() 
    
    with open(output_file, 'w') as file:
        for subdir, dirs, files in os.walk(main_folder):
            for file_name in files:
                if file_name.endswith('.jpg'):  
                    image_path = os.path.join(subdir, file_name)
                    image = Image.open(image_path).convert('RGB')
                    image = transforms(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(image)
                        _, predicted = torch.max(outputs, 1)
                        predicted_class = predicted.item()

                    relative_path = os.path.relpath(image_path, main_folder)
                    file.write(f"{relative_path} class: {predicted_class}\n")

# get_labels("tests/saved_frames_I06_s2_segmented", "I06_s2_classes.txt")
train()
accuracy, precision, recall, f1, mcc = test()


# checkpoint_filename = f"{MODEL}_{PRETRAINED}_ckpt"
# torch.save(model.state_dict(), f'checkpoints/{checkpoint_filename}')

# Write results
# with open('results.txt', 'a') as file:
#     file.write(f"Model: {MODEL}, Pretrained: {PRETRAINED}, ")
#     file.write(f"Accuracy: {accuracy}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}\n")