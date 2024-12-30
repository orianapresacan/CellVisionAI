import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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



def tsne_only_classes(input_csv_path):
    features = []
    classes = []
    filenames = []

    # Read the CSV file line by line
    with open(input_csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0]  # First item is the filename
            filenames.append(filename)
            
            # Extract the class ID from the filename
            class_str = filename.split('_')[-1]
            try:
                class_id = int(class_str.split('.')[0][-1])  # Extract numeric class ID
            except ValueError:
                print(f"Error parsing class ID from filename: {filename}")
                continue

            # Extract feature vector
            feature_vector = np.array(parts[1:], dtype=np.float32)
            features.append(feature_vector)
            classes.append(class_id)

    # Convert features to numpy array
    features_array = np.array(features)

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_array)

    # Map class IDs to names and colors
    class_names = {1: 'fed', 2: 'starved', 3: 'unidentified'}
    class_colors = {
        'fed': '#FFD700',       # Gold
        'starved': '#FF4500',   # OrangeRed
        'unidentified': '#228B22'  # ForestGreen
    }

    # Plot t-SNE results
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))

    for class_id in sorted(set(classes)):
        class_name = class_names[class_id]
        color = class_colors[class_name]
        indices = [i for i, cls in enumerate(classes) if cls == class_id]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    color=color, alpha=0.7, label=class_name, s=50)

    plt.legend(title="Class", loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title('t-SNE Visualization of Classes', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Specify the path to your CSV file
tsne_only_classes('features_test.csv')
