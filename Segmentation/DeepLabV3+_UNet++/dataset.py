import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision.transforms as transforms


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])

class KvasirDataset(Dataset):
    def __init__(self, images_dir, masks_dir, num_images_per_folder=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.num_images_per_folder = num_images_per_folder

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]),
        ])

        self.image_files = []
        self.mask_files = []
        for images_dir, masks_dir in zip(images_dir, masks_dir):
            image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if is_image_file(f)]
            mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if is_image_file(f)]

            if num_images_per_folder is not None:
                image_files = image_files[:num_images_per_folder]
                mask_files = mask_files[:num_images_per_folder]

            self.image_files.extend(image_files)
            self.mask_files.extend(mask_files)

        self.image_files.sort()
        self.mask_files.sort()

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply transformations
        image = self.transform(image)

        # For mask, you might want to apply a different set of transformations
        # For example, you might just want to resize and convert it to a tensor
        mask_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), 
            lambda x: (x > 0).float()
        ])
        mask = mask_transform(mask)

        return image, mask
    
    def __len__(self):
        return len(self.image_files)
    

