import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os


def tsne_with_multiple_params(input_csv_path, output_dir):
    features = []
    classes = []
    filenames = []

    with open(input_csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0]  # First item is the filename
            filenames.append(filename)

            class_str = filename.split('_')[-1]
            try:
                class_id = int(class_str.split('.')[0][-1])  # Extract numeric class ID
            except ValueError:
                print(f"Error parsing class ID from filename: {filename}")
                continue

            feature_vector = np.array(parts[1:], dtype=np.float32)
            features.append(feature_vector)
            classes.append(class_id)

    features_array = np.array(features)

    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features_array)

    os.makedirs(output_dir, exist_ok=True)

    perplexities = [5, 15, 25,  35, 50]
    learning_rates = [10, 100, 500]
    metrics = ['euclidean', 'cosine']

    for perplexity in perplexities:
        for learning_rate in learning_rates:
            for metric in metrics:
                tsne = TSNE(
                    n_components=2, random_state=42,
                    perplexity=perplexity, learning_rate=learning_rate,
                    n_iter=3000, metric=metric
                )
                tsne_results = tsne.fit_transform(standardized_features)

                class_names = {1: 'fed', 2: 'starved', 3: 'unidentified'}
                class_colors = {
                    'fed': '#FFD700',       # Gold
                    'starved': '#FF4500',   # OrangeRed
                    'unidentified': '#228B22'  # ForestGreen
                }

                plt.style.use("ggplot")
                plt.figure(figsize=(10, 8))

                for class_id in sorted(set(classes)):
                    class_name = class_names[class_id]
                    color = class_colors[class_name]
                    indices = [i for i, cls in enumerate(classes) if cls == class_id]
                    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                                color=color, alpha=0.7, label=class_name, s=50)

                plt.legend(title="Class", loc='upper right', bbox_to_anchor=(1.15, 1))
                plt.title(f't-SNE (Perplexity={perplexity}, LR={learning_rate}, Metric={metric})', fontsize=14)
                plt.xlabel('t-SNE Dimension 1', fontsize=12)
                plt.ylabel('t-SNE Dimension 2', fontsize=12)
                plt.grid(color='gray', linestyle='--', linewidth=0.5)
                plt.tight_layout()

                plot_filename = f"tsne_p{perplexity}_lr{learning_rate}_m{metric}.png"
                plt.savefig(os.path.join(output_dir, plot_filename))
                plt.close()

tsne_with_multiple_params('resnet_features.csv', 'tsne_plots')
