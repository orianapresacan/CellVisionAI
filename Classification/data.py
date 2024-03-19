import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CellDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.classes = sorted(os.listdir(main_dir)) 
        self.image_paths = []
        self.image_labels = []

        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(main_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.image_labels.append(class_index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  
        label = self.image_labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label



