import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset import VideoDataset  
from helpers import denormalize
from torchvision.utils import save_image
from convlstm import ConvLSTM_Michael
from convlstm_attention import ConvLSTM_Attention
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]),
])

test_dataset = VideoDataset(root_dir="data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

MODEL = 'attention'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]),
])


input_dim = 3  
hidden_dims = [128, 128, 128, 128]  
kernel_sizes = [5, 5, 5, 5]   
n_layers = 4  

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

model_path = os.path.join('best_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval() 

total_loss = 0.0
criterion = torch.nn.MSELoss() 

with torch.no_grad():
    for frames, target in tqdm(test_loader, desc="Testing"):
        frames, target = frames.to(device), target.to(device)

        if frames.size(1) == 4 * 3:  
            frames = frames.view(frames.size(0), -1, frames.size(3), frames.size(4))

        outputs = model(frames)
        loss = criterion(outputs, target)
        total_loss += loss.item() * frames.size(0)

test_loss = total_loss / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")


output_dir = "test_output"  
os.makedirs(output_dir, exist_ok=True)  

for batch_idx, (frames, target) in enumerate(test_loader):
    frames, target = frames.to(device), target.to(device)
    outputs = model(frames)
    outputs = denormalize(outputs)
    target = denormalize(target)
    for i in range(outputs.size(0)):
        save_image(outputs[i], os.path.join(output_dir, f"{batch_idx * test_loader.batch_size + i}_pred.png"))
        save_image(target[i], os.path.join(output_dir, f"{batch_idx * test_loader.batch_size + i}_true.png"))

