from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch
from typing import List, Dict, Any
import dataset
import model_sam
import losses


def custom_collate_fn(batch):
    images, image_paths, masks = zip(*batch)  
    images = torch.stack(images)
    return images, image_paths, masks  

def load_datasets(img_size):
    """ load the training and validation datasets in PyTorch DataLoader objects
    ARGS:
        img_size (Tuple(int, int)): image size
    RETURNS:
        train_dataloader (DataLoader): training dataset
        val_dataloader (DataLoader): validation dataset

    """
    transform = dataset.ResizeAndPad(1024)
    traindata = dataset.COCODataset(root_dir=train,
                        annotation_file="data/train/annotations.json", transform=transform)
    valdata = dataset.COCODataset(root_dir=test,
                      annotation_file="data/val/annotations.json", transform=transform)
    train_dataloader = DataLoader(traindata,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=1, 
                                  collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(valdata,
                                batch_size=1,
                                shuffle=True,
                                num_workers=1,
                                collate_fn=custom_collate_fn)
    return train_dataloader, val_dataloader


def get_total_mask(images, batched_masks):
    batch_size = images.shape[0]
    _, H, W = images.shape[1:]  
    total_masks = torch.zeros((batch_size, H, W), device=DEVICE) 
    for idx, masks in enumerate(batched_masks): 
        for mask in masks:
            total_masks[idx] += mask.to(DEVICE)
    return total_masks


def train_one_epoch(model, trainloader, optimizer, epoch_idx):
    running_loss = 0.
    for images, _, batched_masks in tqdm(trainloader, desc="Training"):
        images = images.to(DEVICE)
        optimizer.zero_grad()

        pred, _ = model(images)
        total_masks = get_total_mask(images, batched_masks)
        total_masks = total_masks.unsqueeze(1)
        # print(f'pred shape: {pred.shape}, total_masks shape: {total_masks.shape}')

        loss = losses.criterion(pred, total_masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f'Average batch loss for epoch {epoch_idx}: {avg_loss}')
    return avg_loss


def train_model():
    """Trains the model for the given number of epochs."""
    bestmodel_path = "best_model.pth" 
    model = model_sam.ModelSimple()
    model.setup()
    model.to(DEVICE)
    img_size = model.model.image_encoder.img_size
    trainloader, validloader = load_datasets(img_size=img_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_loss = float('inf')
    for epch in range(epochs):
        model.train(True)
        avg_batchloss = train_one_epoch(model, trainloader, optimizer, epch)

        running_vloss = 0.
        with torch.no_grad():
            for images, _, batched_masks in tqdm(validloader, desc="Validation"):
                images, total_mask = images.to(DEVICE), get_total_mask(images, batched_masks).unsqueeze(1).to(DEVICE)
                model.eval()
                preds, iou = model(images)
                vloss = losses.criterion(preds, total_mask)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / len(validloader)
        print(f'Epoch {epch}: Training Loss: {avg_batchloss}, Validation Loss: {avg_vloss}')

        if avg_vloss < best_valid_loss:
            print(f'New best model with loss {avg_vloss}')
            best_valid_loss = avg_vloss
            torch.save(model.state_dict(), bestmodel_path)  

    return model


def main():
    global train
    global test
    global annot

    train = "data/train/images"
    test = "data/val/images"
    annot = "annotations.json"

    global batch_size
    global epochs
    global lr
    global weight_decay
    global DEVICE

    batch_size = 4
    epochs = 100
    lr = 0.001
    weight_decay = 0.0005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = train_model()

if __name__ == '__main__':
    main()
