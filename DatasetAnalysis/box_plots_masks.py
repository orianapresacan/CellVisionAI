import cv2
import numpy as np
import os
from scipy.stats import mode
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

root_dir = 'masks_consistent'  

fed_areas = []
unfed_areas = []
fed_circularities = []
unfed_circularities = []

def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)

def process_masks(directory):
    areas = []
    circularities = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  
            path = os.path.join(directory, filename)
            image = cv2.imread(path)

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = calculate_circularity(area, perimeter)

                areas.append(area)
                circularities.append(circularity)

    return areas, circularities


def process_directory(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if 'fed' in dirs:
            fed_dir = os.path.join(root, 'fed')
            areas, circularities = process_masks(fed_dir)
            fed_areas.extend(areas)
            fed_circularities.extend(circularities)
        if 'unfed' in dirs:
            unfed_dir = os.path.join(root, 'unfed')
            areas, circularities = process_masks(unfed_dir)
            unfed_areas.extend(areas)
            unfed_circularities.extend(circularities)

process_directory(root_dir)

def compute_statistics(data):
    statistics = {
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': mode(data)[0][0] if data else 0,
        'range': np.ptp(data),
        'variance': np.var(data, ddof=1) if data else 0,  # ddof=1 for sample variance
        'standard_deviation': np.std(data, ddof=1) if data else 0  # ddof=1 for sample standard deviation
    }
    return statistics

fed_stats_area = compute_statistics(fed_areas)
unfed_stats_area = compute_statistics(unfed_areas)
fed_stats_circularity = compute_statistics(fed_circularities)
unfed_stats_circularity = compute_statistics(unfed_circularities)

print("Fed Cells - Area Statistics:", fed_stats_area)
print("Unfed Cells - Area Statistics:", unfed_stats_area)

print("Fed Cells - Circularity Statistics:", fed_stats_circularity)
print("Unfed Cells - Circularity Statistics:", unfed_stats_circularity)

plt.style.use("ggplot")
plt.figure(figsize=(8, 6))
areas_data_combined = fed_areas + unfed_areas
labels_combined = ['Fed'] * len(fed_areas) + ['Starved'] * len(unfed_areas)  # Change 'Unfed' to 'Starved'
sns.boxplot(x=labels_combined, y=areas_data_combined, palette={'Fed': '#E7E439', 'Starved': '#9E4B4B'}, width=0.1)
plt.ylabel('Area', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=1)
plt.show()

plt.style.use("ggplot")
plt.figure(figsize=(8, 6))
circularity_data_combined = fed_circularities + unfed_circularities
labels_combined = ['Fed'] * len(fed_circularities) + ['Starved'] * len(unfed_circularities)  # Change 'Unfed' to 'Starved'
sns.boxplot(x=labels_combined, y=circularity_data_combined, palette={'Fed': '#E7E439', 'Starved': '#9E4B4B'}, width=0.1)
plt.ylabel('Circularity', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=1)
plt.show()


def ttest(fed, unfed):
    t_stat, p_value = stats.ttest_ind(fed, unfed, equal_var=False)  # equal_var=False for Welch's t-test

    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    # Interpretation
    if p_value < 0.05:
        print("There is a significant difference between the cell areas of fed and unfed groups.")
    else:
        print("There is no significant difference between the cell areas of fed and unfed groups.")


ttest(fed_areas, unfed_areas)