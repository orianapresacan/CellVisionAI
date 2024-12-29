import torch
import argparse
import wandb
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import train
import metrics
import os

from dataset import CellDataset
import helpers


ENCODER = 'se_resnext50_32x4d'
LOCAL_ENCODER_WEIGHTS = 'pretrained_encoder/se_resnext50_32x4d-a260b3a4.pth'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['cell']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0
MAX_EPOCHS = 200
MODEL = "UNET++"  #"DEEPLABV3+", "UNET++"
USE_WANDB = False

helpers.fix_seed()

try:
    os.mkdir('train_outputs')
    print(f"Directory '{'train_outputs'}' created successfully.")
except FileExistsError:
    print(f"Directory '{'train_outputs'}' already exists.")


if USE_WANDB:
    wandb.login(key='9b63fbfe37acd6800197e818ec55b0d6bd8724f1') 
    wandb.init(project='Segmentation Models', name=f'{MODEL}')

model_types = { 
    "UNET++": smp.UnetPlusPlus,
    "FPN": smp.FPN, 
    "DEEPLABV3+": smp.DeepLabV3Plus,
}

model = model_types[MODEL](
    encoder_name=ENCODER,
    encoder_weights=None,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

model.encoder.load_state_dict(torch.load(LOCAL_ENCODER_WEIGHTS))
# model.load_state_dict(torch.load(f"checkpoints/ckpt_{MODEL}.pt"))
# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Validation dataset is always the same
val_dataset = CellDataset(["data/val/cropped_images"], ["data/val/cropped_masks"]) 
valid_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

# Train dataset - combining multiple image folders
# images_dirs, masks_dirs = helpers.combine_folders(options.folders)
train_dataset = CellDataset(["data/train/cropped_images"], ["data/train/cropped_masks"]) #, augmentation=get_training_augmentation()) #, preprocessing=get_preprocessing(preprocessing_fn)) #, augmentation=get_training_augmentation(), 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)

early_stopping = helpers.EarlyStopper(patience=20, verbose=True, path=f'checkpoints/ckpt_{MODEL}.pt')
loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
metrics = [metrics.IoU(threshold=0.5), metrics.Fscore(), metrics.Precision(), metrics.Accuracy()]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001)

train_epoch = train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

for i in range(0, MAX_EPOCHS):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    early_stopping(valid_logs['iou_score'], model)

    if early_stopping.early_stop:
        print("Early stopping")
        break
    scheduler.step(valid_logs['iou_score'])
    
    n = np.random.choice(len(val_dataset))
    image, gt_mask = val_dataset[n]
    x_tensor = image.to(DEVICE).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = pr_mask.squeeze(0).cpu().numpy().round()
    denormalized_image = helpers.denormalize(image, [0.4437, 0.4503, 0.2327], [0.2244, 0.2488, 0.0564])

    grid = helpers.visualize(
        image=denormalized_image,
        ground_truth_mask=np.transpose(gt_mask*255, (1, 2, 0)), 
        predicted_mask=np.transpose(pr_mask*255, (1, 2, 0)), 
    )
    grid.save(f"train_outputs/sample_validation_dataset_epoch{i}.png")

    if USE_WANDB:
        wandb.log({"IOU": valid_logs['iou_score'], "Accuracy": valid_logs['accuracy'], "Precision": valid_logs['precision'], 
                "F1": valid_logs['fscore'], 'examples': wandb.Image(np.array(grid))})
