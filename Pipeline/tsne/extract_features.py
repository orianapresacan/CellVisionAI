import os
import torch
from torchvision import transforms
from PIL import Image
from unet import UNet
import shutil


def merge_images():
    data_dir = 'data'
    """
    Moves all images from 'C03_s3' and 'I06_s2' and their subfolders into a single 'merged_images' folder.

    :param DATA_DIR: Root directory containing the two parent folders.
    """
    source_folders = ["C03_s3", "I06_s2"]

    merged_folder = os.path.join(data_dir, "merged_images")
    os.makedirs(merged_folder, exist_ok=True)

    for folder in source_folders:
        folder_path = os.path.join(data_dir, folder)

        if not os.path.exists(folder_path):
            print(f"Skipping {folder_path} (does not exist).")
            continue

        for subdir, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith('.png'):  
                    source_path = os.path.join(subdir, file_name)

                    base_name, ext = os.path.splitext(file_name)
                    new_file_name = f"{base_name}{ext}"
                    destination_path = os.path.join(merged_folder, new_file_name)

                    shutil.move(source_path, destination_path)
                    print(f"Moved: {source_path} -> {destination_path}")

    print("Merging complete! All images are now in:", merged_folder)

def compute_and_save_features(image_folder_path, model_type="unet"):
    model = UNet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load("checkpoints/unet.pth"))
    model.eval()
    model = model.cuda()

    def extract_features(input_tensor):
        with torch.no_grad():
            features = model(input_tensor)
            return features.view(features.size(0), -1).cpu().numpy()  # Flatten feature maps

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4437, 0.4503, 0.2327], std=[0.2244, 0.2488, 0.0564]),
    ])


    output_file_path = f'tsne/{model_type}_features.csv'
    with open(output_file_path, 'w') as f:
        for filename in os.listdir(image_folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(image_folder_path, filename)
                image = Image.open(image_path).convert("RGB")

                input_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

                feature_vector = extract_features(input_tensor)

                f.write(f"{filename} {' '.join(map(str, feature_vector.flatten()))}\n")

    print(f"Features saved to {output_file_path} using {model_type} model.")

# merge_images()
compute_and_save_features('data/merged_images', model_type="unet")
