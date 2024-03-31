import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from unet import UNet  # Make sure to have this import path correct
from dataset import VideoDataset  # Adjust import paths as needed
import os
from helpers import denormalize

# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]),
])

test_dataset = VideoDataset(root_dir="data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch size set to 1 for individual processing

model = UNet(in_channels=4*3, out_channels=3).to(device)
model.load_state_dict(torch.load('plot_output/best_model.pth'))
model.eval()

# Directory for saving outputs
output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)

# Processing
with torch.no_grad():
    for idx, (frames, target) in enumerate(test_loader):
        frames, target = frames.to(device), target.to(device)
        frames = frames.view(frames.size(0), -1, frames.size(3), frames.size(4))

        output = model(frames)
        output = denormalize(output)
        target = denormalize(target)
        # Save images
        save_image(output.cpu(), os.path.join(output_dir, f"{idx+1}_pred.png"))
        save_image(target.cpu(), os.path.join(output_dir, f"{idx+1}_true.png"))

print("Test images saved.")
