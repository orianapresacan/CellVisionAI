import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def tsne_only_classes_with_preprocessing(input_csv_path, exclude_class=None):
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

            if class_id == exclude_class:
                continue

            # Extract feature vector
            feature_vector = np.array(parts[1:], dtype=np.float32)
            features.append(feature_vector)
            classes.append(class_id)

    # Convert features to numpy array
    features_array = np.array(features)

    # Preprocessing: Standardize the features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features_array)

    # Dimensionality reduction: PCA to reduce to 50 dimensions (or less if fewer features)
    # pca = PCA(n_components=min(50, standardized_features.shape[1]), random_state=42)
    # reduced_features = pca.fit_transform(standardized_features)

    # Compute t-SNE on the reduced features
    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=5)
    tsne_results = tsne.fit_transform(standardized_features)

    # Map class IDs to names and colors
    class_names = {1: 'Cells in Basal Autophagy', 2: 'Cells in Activated Autophagy', 3: 'Unidentified Cells'}
    class_colors = {
        'Cells in Basal Autophagy': '#FFD700',       # Gold
        'Cells in Activated Autophagy': '#FF4500',   # OrangeRed
        'Unidentified Cells': '#228B22'  # ForestGreen
    }

    # Plot t-SNE results
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))

    for class_id in sorted(set(classes)):
        class_name = class_names[class_id]
        color = class_colors[class_name]
        indices = [i for i, cls in enumerate(classes) if cls == class_id]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    color=color, alpha=0.7, label=class_name, s=30)

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title('t-SNE Visualization of Classes with Preprocessing', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Specify the path to your CSV file
tsne_only_classes_with_preprocessing('resnet_features.csv', exclude_class=None)
