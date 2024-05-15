import numpy as np
import os
import albumentations as albu
import torch
from PIL import Image
import random


def fix_seed():
    random_seed = 123
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def denormalize(tensor, mean, std):
    # Reshape mean and std to match the tensor dimensions: [C, 1, 1]
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)

    # Perform the denormalization
    denormalized = tensor.numpy() * std + mean

    # Clip the values to be in the range [0, 1] and scale to [0, 255]
    denormalized = np.clip(denormalized, 0, 1) * 255

    # Convert to uint8 and transpose from [C, H, W] to [H, W, C]
    return denormalized.astype(np.uint8).transpose(1, 2, 0)

def normalize(image, mean, std):
    # Convert image to float and scale to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Reshape mean and std to match the image dimensions: [1, 1, C]
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)

    # Perform the normalization
    normalized = (image - mean) / std

    # Transpose from [H, W, C] to [C, H, W]
    return normalized.transpose(2, 0, 1)

def visualize(**images):
    """Concatenate and return images as a single PIL Image."""
    n = len(images)
    max_height = max(image.shape[0] for image in images.values())
    total_width = sum(image.shape[1] for image in images.values())
    concatenated_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    x_offset = 0
    for name, image in images.items():
        new_height = image.shape[0]
        concatenated_image[:new_height, x_offset:x_offset + image.shape[1], :] = image
        x_offset += image.shape[1]

    return Image.fromarray(concatenated_image)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=256, width=256, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(256, 256)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class EarlyStopper:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoints/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation IOU increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def combine_folders(folder_list):
    combined_images_dirs = []
    combined_masks_dirs = []

    for source in folder_list:
        images_dir = os.path.join(source, "images/train")
        masks_dir = os.path.join(source, "masks/train")

        combined_images_dirs.append(images_dir)
        combined_masks_dirs.append(masks_dir)

    return combined_images_dirs, combined_masks_dirs


