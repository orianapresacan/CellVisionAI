from torchvision import transforms
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class RotateSequence:
    """Rotate the whole sequence of images by the same random angle."""
    def __init__(self, angles=[0, 90, 180, 270], p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, images):
        if random.random() > self.p:
            angle = random.choice(self.angles)
            images = [TF.rotate(img, angle) for img in images]
        return images

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_transform = sequence_transform
        self.samples = []

        video_folders = sorted(os.listdir(root_dir))
        for video_folder in video_folders:
            video_path = os.path.join(root_dir, video_folder)
            frame_images = sorted(os.listdir(video_path))
            if len(frame_images) < 5:
                continue

            input_frames = [os.path.join(video_path, frame_images[i]) for i in range(4)]
            target_frame = os.path.join(video_path, frame_images[4])
            self.samples.append((input_frames, target_frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_frames_paths, target_frame_path = self.samples[idx]

        # Load images from paths
        input_frames = [self.load_image(path) for path in input_frames_paths]
        target_frame = self.load_image(target_frame_path)

        # Apply individual image transformations
        if self.transform:
            input_frames = [self.transform(frame) for frame in input_frames]
            target_frame = self.transform(target_frame)
        
        # Apply sequence transformation
        if self.sequence_transform:
            input_frames.append(target_frame)  # Combine for sequence transformation
            transformed_frames = self.sequence_transform(input_frames)
            target_frame = transformed_frames.pop()  # Separate them again
            input_frames = transformed_frames

        return torch.stack(input_frames), target_frame

    def load_image(self, path):
        return Image.open(path).convert('RGB')

