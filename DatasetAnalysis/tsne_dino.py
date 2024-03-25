import torch
from PIL import Image
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel


def create_color_gradient(base_color, n_shades=5):
    """
    Generates a list of color shades from a lighter version of the base color to the base color.
    Args:
    - base_color: A tuple of RGB values ranging from 0 to 1.
    - n_shades: Number of shades to generate.
    Returns: A list of color codes.
    """
    # Calculate a lighter start color by increasing the RGB values towards 1
    start_color = tuple(min(bc + 0.5, 1) for bc in base_color)  # Ensuring not to exceed 1
    # Generate shades from lighter to the base color
    return [tuple(start_color[i] + (base_color[i] - start_color[i]) * (j / (n_shades - 1)) for i in range(3)) for j in range(n_shades)]

    
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
            if filename.endswith('.jpg'):  
                time_point = filename.split('_')[1]  
                image_path = os.path.join(image_folder_path, filename)
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                    feature_vector = outputs[0].cpu().numpy().flatten()
                    
                    f.write(f"{filename} {' '.join(map(str, feature_vector))}\n")


def read_features_and_visualize(input_file_path):
    features = []
    time_points = []
    classes = []
    with open(input_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0]
            time_point_str = filename.split('_')[1]
            class_str = filename.split('_')[-1]
            try:
                time_point = int(time_point_str)
                class_id = int(class_str.split('.')[0][-1])
            except ValueError:
                print(f"Error parsing time point or class from filename: {filename}")
                continue

            feature_vector = np.array(parts[1:], dtype=np.float32)
            features.append(feature_vector)
            time_points.append(time_point)
            classes.append(class_id)

    features_array = np.array(features)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_array)

    class_names = {1: 'fed', 2: 'starved', 3: 'unidentified'}
    base_colors = {'fed': (1, 1, 0), 'starved': (1, 0, 0), 'unidentified': (0, 1, 0)}  

    plt.figure(figsize=(8, 5))
    legend_elements = []

    for class_id in sorted(set(classes)):
        class_name = class_names[class_id]
        colors = create_color_gradient(base_colors[class_name], n_shades=5)

        for time_point in sorted(set(time_points)):
            indices = [j for j, (tp, cls) in enumerate(zip(time_points, classes)) if tp == time_point and cls == class_id]
            color = colors[time_point - 1]
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=color, alpha=0.6, label=f'{class_name} TP{time_point}' if indices else "")

            if time_point == 1:  
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=class_name, markersize=3, markerfacecolor=color))

    plt.legend(handles=legend_elements)
    plt.title('t-SNE of Cell Features Across Time Points and Classes')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.show()


def tsne_only_classes(input_file_path):
    features = []
    classes = []
    with open(input_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0]
            class_str = filename.split('_')[-1]
            try:
                class_id = int(class_str.split('.')[0][-1])
            except ValueError:
                print(f"Error parsing time point or class from filename: {filename}")
                continue

            feature_vector = np.array(parts[1:], dtype=np.float32)
            features.append(feature_vector)
            classes.append(class_id)

    features_array = np.array(features)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_array)

    class_names = {1: 'fed', 2: 'starved', 3: 'unidentified'}
    class_colors = {
        'fed': 'gold', 
        'starved': 'red', 
        'unidentified': 'green'  
    }
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 5))

    for class_id in sorted(set(classes)):
        class_name = class_names[class_id]
        color = class_colors[class_name]
        indices = [i for i, cls in enumerate(classes) if cls == class_id]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=color, alpha=0.6, label=class_name, s=10)

    plt.legend(title="Class")
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.show()

compute_and_save_features(image_folder_path='cropped_image_mixed', output_file_path='features_mini.txt')
# tsne_only_classes('features_mini.txt')