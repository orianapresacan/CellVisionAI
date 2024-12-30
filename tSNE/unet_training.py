import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            transform (callable, optional): Transform to apply to images.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  
        if self.transform:
            image = self.transform(image)
        return image

def train_autoencoder(data_dir, epochs=200, batch_size=256, learning_rate=0.0001):
    mean = [0.4437, 0.4503, 0.2327]
    std = [0.2244, 0.2488, 0.0564]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),         
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = CustomImageDataset(image_dir=data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=3).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    best_loss=100
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        reconstructed_images = None
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)
                if reconstructed_images is None:
                    reconstructed_images = (images.cpu(), outputs.cpu())

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if epoch%5==0:
            if reconstructed_images:
                original, reconstructed = reconstructed_images
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                for i in range(5):
                    orig_img = unnormalize(original[i])
                    axes[0, i].imshow(orig_img.permute(1, 2, 0).clamp(0, 1))  # Unnormalize and clip
                    axes[0, i].axis('off')
                    recon_img = unnormalize(reconstructed[i])
                    axes[1, i].imshow(recon_img.permute(1, 2, 0).clamp(0, 1))  # Unnormalize and clip
                    axes[1, i].axis('off')
                plt.suptitle(f"Epoch {epoch+1} - Original (Top) and Reconstructed (Bottom)")
                plt.show()

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(model.state_dict(), f"unet.pth")
            print(f"Model saved ")

train_autoencoder(data_dir="mixed", epochs=200)
