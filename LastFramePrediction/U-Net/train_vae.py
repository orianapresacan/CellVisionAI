import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from vae import VAE 
from dataset import VideoDataset  
import os
from tqdm import tqdm
import random
from dataset import RotateSequence
from helpers import plot_images  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]),
])

sequence_transform = RotateSequence(angles=[0, 90, 180, 270], p=0.5)

train_dataset = VideoDataset(root_dir="data/train", transform=transform, sequence_transform=sequence_transform)
val_dataset = VideoDataset(root_dir="data/val", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True)

model = VAE().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

plot_dir = "plot_output"
os.makedirs(plot_dir, exist_ok=True)

best_val_loss = float('inf')
num_epochs = 100

# Adapted loss function for VAE
def loss_function(recon_x, x, mu, logvar):
    # MSE loss for reconstruction
    MSE = nn.functional.mse_loss(recon_x, x)
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

    for frames, target in train_loader_tqdm:
        frames, target = frames.to(device), target.to(device)
        # print(frames.shape)
        frames = frames.view(frames.size(0), -1, frames.size(3), frames.size(4))
        # print(frames.shape)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(frames)
        # print(recon_batch.shape)
        # exit()
        loss = loss_function(recon_batch, target, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_loader_tqdm.set_postfix(loss=loss.item() / frames.size(0))

    train_loss = total_loss / len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for frames, target in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
            frames, target = frames.to(device), target.to(device)
            frames = frames.view(frames.size(0), -1, frames.size(3), frames.size(4))

            recon_batch, mu, logvar = model(frames)
            loss = loss_function(recon_batch, target, mu, logvar)
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(plot_dir, 'best_model_vae.pth'))
        print(f"Epoch [{epoch + 1}/{num_epochs}]: New best model saved with val_loss: {val_loss:.4f}")

    scheduler.step(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if epoch % 1 == 0:  # Adjust as needed
        model.eval()
        with torch.no_grad():
            random_batch_index = random.choice(range(len(val_loader)))
            for batch_idx, (frames, target) in enumerate(val_loader):
                if batch_idx == random_batch_index:
                    frames, target = frames.to(device), target.to(device)
                    reshaped_frames = frames.view(frames.size(0), -1, frames.size(3), frames.size(4))
                    reconstructed,_,_ = model(reshaped_frames)
                    frames = frames.view(frames.size(0), frames.size(1), frames.size(2), frames.size(3), frames.size(4))
                    plot_images(frames[0].cpu(), target[0].cpu(), reconstructed[0].cpu(), epoch, batch_idx, plot_dir)
