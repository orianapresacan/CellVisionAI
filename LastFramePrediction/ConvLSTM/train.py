import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VideoDataset  
from convlstm import ConvLSTM_Michael
from convlstm_attention import ConvLSTM_Attention
import os 
from tqdm import tqdm
import random
from dataset import RotateSequence
from helpers import plot_images


MODEL = 'attention'
input_dim = 3  
hidden_dims = [128, 128, 128, 128]  # 128, 128, 128, 128 -> val loss: 0.010 after epoch 50
kernel_sizes = [5, 5, 5, 5]   
n_layers = 4  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]),
])

sequence_transform = RotateSequence(angles=[0, 90, 180, 270], p=0.5)

train_dataset = VideoDataset(root_dir="data/train", transform=transform, sequence_transform=sequence_transform)
val_dataset = VideoDataset(root_dir="data/val", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

if MODEL == 'attention':
     model = ConvLSTM_Attention(input_dim=input_dim,
                     hidden_dims=hidden_dims,
                     kernel_sizes=kernel_sizes,
                     n_layers=n_layers,
                     device=device,
                     batch_first=True,
                     return_sequences=False).to(device)
else:
    model = ConvLSTM_Michael(input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        kernel_sizes=kernel_sizes,
                        n_layers=n_layers,
                        device=device,
                        batch_first=True,
                        return_sequences=False).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

plot_dir = "plot_output"
os.makedirs(plot_dir, exist_ok=True)

best_val_loss = float('inf') 
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

    for frames, target in train_loader_tqdm:
        frames, target = frames.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * frames.size(0)
        train_loader_tqdm.set_postfix(loss=loss.item())

    train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")

    with torch.no_grad():
        for frames, target in val_loader_tqdm:
            frames, target = frames.to(device), target.to(device)

            outputs = model(frames)
            loss = criterion(outputs, target)
            val_loss += loss.item() * frames.size(0)
            val_loader_tqdm.set_postfix(loss=loss.item())

    val_loss /= len(val_loader.dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(plot_dir, 'best_model.pth'))
        print(f"Epoch [{epoch + 1}/{num_epochs}]: New best model saved with val_loss: {val_loss:.4f}")

    scheduler.step(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if epoch % 1 == 0: 
        model.eval()
        with torch.no_grad():
            random_batch_index = random.choice(range(len(val_loader)))
            for batch_idx, (frames, target) in enumerate(val_loader):
                if batch_idx == random_batch_index:
                    frames, target = frames.to(device), target.to(device)                
                    reconstructed = model(frames)
                    plot_images(frames[0].cpu(), target[0].cpu(), reconstructed[0].cpu(), epoch, batch_idx, plot_dir)
                    
                