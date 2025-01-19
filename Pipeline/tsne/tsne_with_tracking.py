import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def hex_to_rgb(hex_color):
    """Convert hex color to normalized RGB tuple (0-1)."""
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))


def read_features_and_visualize(input_file_path):
    # Define color mappings for each class and time point
    class_colors = {
        "unidentified": [hex_to_rgb(c) for c in ["8cf37e", "68e756", "50cb3e", "35a725", "137006"]],
        "basal": [hex_to_rgb(c) for c in ["dfea72", "ceda57", "beca44", "adba28", "9ca912"]],
        "activated": [hex_to_rgb(c) for c in ["ee7979", "db5757", "d34444", "c43030", "9f1111"]]
    }

    features = []
    time_points = []
    classes = []

    with open(input_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0]
            time_point_str = filename.split('_')[1]  # Extract TP number
            class_name = filename.split('_')[-1].split('.')[0]  # Extract class

            try:
                time_point = int(time_point_str)
                if class_name not in class_colors:
                    print(f"Skipping unknown class: {class_name}")
                    continue
            except ValueError:
                print(f"Error parsing time point or class from filename: {filename}")
                continue

            feature_vector = np.array(parts[1:], dtype=np.float32)
            features.append(feature_vector)
            time_points.append(time_point)
            classes.append(class_name)

    features_array = np.array(features)

    tsne = TSNE(n_components=2, random_state=42, perplexity=10, metric='euclidean')
    tsne_results = tsne.fit_transform(features_array)

     # Apply gray background style
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("lightgray")  # Set gray background
    
    # Store legend patches
    legend_patches = {}

    for class_name, color_shades in class_colors.items():
        for tp in range(1, 6):  # Time points 1 to 5
            indices = [i for i, (t, c) in enumerate(zip(time_points, classes)) if t == tp and c == class_name]
            if indices:
                color = color_shades[tp - 1]
                plt.scatter(
                    tsne_results[indices, 0], 
                    tsne_results[indices, 1], 
                    color=color, 
                    s=40,               # Larger points for better visibility
                    edgecolors="none",  # No dark borders
                    alpha=1.0           # Stronger colors (no transparency)
                )

                # Add only one legend entry per class
                if class_name not in legend_patches:
                    legend_patches[class_name] = mpatches.Patch(color=color_shades[2], label=class_name)  # Middle shade

    plt.legend(handles=legend_patches.values(), title="Classes", fontsize=12)
    plt.title('t-SNE of Cell Features Across Time Points and Classes', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True)
    plt.show()


read_features_and_visualize('tsne/unet_features.csv')