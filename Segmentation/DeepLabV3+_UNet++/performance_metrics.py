import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_mask(path):
    """Load and binarize a mask from the given path."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (mask > 128).astype(np.uint8)

def calculate_metrics(mask_gt, mask_pred):
    """Calculate evaluation metrics between two binary masks."""
    precision = precision_score(mask_gt.flatten(), mask_pred.flatten(), zero_division=0)
    recall = recall_score(mask_gt.flatten(), mask_pred.flatten(), zero_division=0)
    f1 = f1_score(mask_gt.flatten(), mask_pred.flatten(), zero_division=0)
    accuracy = accuracy_score(mask_gt.flatten(), mask_pred.flatten())
    dice_coeff = 2 * np.sum(mask_pred * mask_gt) / (np.sum(mask_pred) + np.sum(mask_gt)) if np.sum(mask_pred) + np.sum(mask_gt) > 0 else 0
    iou = np.sum(mask_pred * mask_gt) / np.sum((mask_pred + mask_gt) > 0) if np.sum((mask_pred + mask_gt) > 0) else 0
    return dice_coeff, iou, precision, recall, f1, accuracy

def evaluate_masks(folder1, folder2):
    """Evaluate masks between two folders."""
    metrics_results = []
    files1 = {os.path.splitext(file)[0]: file for file in os.listdir(folder1)}
    files2 = {os.path.splitext(file)[0]: file for file in os.listdir(folder2)}
    common_bases = set(files1.keys()).intersection(set(files2.keys()))

    if not common_bases:
        print("No common files in both folders. Please check the directories and filenames.")
        return

    for base in common_bases:
        mask_gt_path = os.path.join(folder1, files1[base])
        mask_pred_path = os.path.join(folder2, files2[base])
        mask_gt = load_mask(mask_gt_path)
        mask_pred = load_mask(mask_pred_path)
        
        metrics = calculate_metrics(mask_gt, mask_pred)
        metrics_results.append(metrics)
        # print(f"Metrics for {base}: Dice={metrics[0]:.3f}, IoU={metrics[1]:.3f}, Precision={metrics[2]:.3f}, Recall={metrics[3]:.3f}, F1={metrics[4]:.3f}, Accuracy={metrics[5]:.3f}")

    if metrics_results:
        averages = np.mean(metrics_results, axis=0)
        print("\nAverage Metrics:")
        print(f"Average Dice Coefficient: {averages[0]:.3f}")
        print(f"Average IOU: {averages[1]:.3f}")
        print(f"Average Precision: {averages[2]:.3f}")
        print(f"Average Recall: {averages[3]:.3f}")
        print(f"Average F1 Score: {averages[4]:.3f}")
        print(f"Average Accuracy: {averages[5]:.3f}")
    else:
        print("No metrics to average. Please check the mask processing.")

folder1 = 'data/test/cropped_masks'  # Path to the folder containing the ground truth masks
folder2 = 'data/test/cropped_images/segmented_cropped_images'  # Path to the folder containing the predicted masks
evaluate_masks(folder1, folder2)
