import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def compute_mean_std(directory):
    transform = transforms.Compose([transforms.ToTensor()])
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))

    mean_sum = torch.zeros(3)
    sq_sum = torch.zeros(3)
    total_pixels = 0

    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        mean_sum += img_tensor.mean([1, 2]) * img_tensor.shape[1] * img_tensor.shape[2]
        sq_sum += img_tensor.pow(2).mean([1, 2]) * img_tensor.shape[1] * img_tensor.shape[2]
        total_pixels += img_tensor.shape[1] * img_tensor.shape[2]

    mean = mean_sum / total_pixels
    std = (sq_sum / total_pixels - mean.pow(2)).sqrt()

    return mean, std

directory = 'data'  
mean, std = compute_mean_std(directory)
print(f"Mean: {mean}")
print(f"Std: {std}")
