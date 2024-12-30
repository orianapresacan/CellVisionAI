
import torch
from PIL import Image
import os
from transformers import AutoImageProcessor, AutoModel
from torchvision import models, transforms


def compute_and_save_features(image_folder_path, output_file_path):
    """
    Computes features for images in a given folder using a specified model and processor.
    Saves the features along with image names and time points to a file.
    
    Args:
    - image_folder_path (str): Path to the folder containing images.
    - output_file_path (str): Path to the output file where features are saved.
    - model (Model): Pre-loaded model for feature extraction.
    - processor (Processor): Pre-loaded processor for image preprocessing.
    """

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    model.eval()  

    with open(output_file_path, 'a') as f:
        for filename in os.listdir(image_folder_path):
            if filename.endswith('.png'):  
                time_point = filename.split('_')[1]  
                image_path = os.path.join(image_folder_path, filename)
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                    feature_vector = outputs[0].cpu().numpy().flatten()
                    
                    f.write(f"{filename} {' '.join(map(str, feature_vector))}\n")


compute_and_save_features(image_folder_path='data/mixed_real_images', output_file_path='features_test.csv')