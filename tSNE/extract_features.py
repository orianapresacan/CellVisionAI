import os
import torch
from torchvision import models, transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from unet import UNet

def compute_and_save_features(image_folder_path, model_type="resnet"):
    """
    Computes features for images in a given folder using the specified model.
    Saves the features along with image names to a file.

    Args:
    - image_folder_path (str): Path to the folder containing images.
    - model_type (str): Type of model to use for feature extraction ("resnet", "unet", or "dinov2").
    """

    if model_type == "unet":
        model = UNet(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load("unet.pth"))
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

    elif model_type == "resnet":
        resnet = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the FC layer
        model.eval()
        model = model.cuda()

        def extract_features(input_tensor):
            with torch.no_grad():
                features = model(input_tensor)
                return features.view(features.size(0), -1).cpu().numpy()  # Flatten feature maps

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif model_type == "dinov2":
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()
        model = model.cuda()

        def extract_features(input_tensor):
            with torch.no_grad():
                inputs = processor(images=input_tensor, return_tensors="pt").to("cuda")
                outputs = model(**inputs)
                return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use CLS token as feature

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # DinoV2 expects 224x224
            transforms.ToTensor(),
        ])

    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'resnet', 'unet', or 'dinov2'.")

    output_file_path = f'{model_type}_features.csv'
    with open(output_file_path, 'w') as f:
        for filename in os.listdir(image_folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(image_folder_path, filename)
                image = Image.open(image_path).convert("RGB")

                input_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

                feature_vector = extract_features(input_tensor)

                f.write(f"{filename} {' '.join(map(str, feature_vector.flatten()))}\n")

    print(f"Features saved to {output_file_path} using {model_type} model.")


compute_and_save_features('data_test/mixed_real_images', model_type="unet")
