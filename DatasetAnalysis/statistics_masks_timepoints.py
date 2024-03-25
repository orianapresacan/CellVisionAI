import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = 'C_H_fed_masks'

areas_data = {'fed': {}, 'unfed': {}}
cell_count_data = {'fed': {}, 'unfed': {}}  # Dictionary to track cell counts

def process_masks(directory):
    areas = []
    count = 0  # Initialize count
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

                areas.append(area)
                count += 1  # Increment count for each mask processed
    return areas, count

def extract_timepoint(folder_name):
    parts = folder_name.split('_')
    if len(parts) > 1:
        return parts[1]
    return None

for root, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        time_point = extract_timepoint(dir_name)
        if time_point:
            for class_type in ['fed', 'unfed']:
                class_path = os.path.join(root, dir_name, class_type)
                if os.path.exists(class_path):
                    areas, count = process_masks(class_path)  # Modified to also return count
                    if time_point not in areas_data[class_type]:
                        areas_data[class_type][time_point] = []
                        cell_count_data[class_type][time_point] = 0  # Initialize count for this time point
                    areas_data[class_type][time_point].extend(areas)
                    cell_count_data[class_type][time_point] += count  # Sum counts

# Printing cell counts per time point
for class_type, time_points in cell_count_data.items():
    for time_point, count in sorted(time_points.items()):
        print(f"Time Point: {time_point}, Class: {'Fed' if class_type == 'fed' else 'Starved'}, Count: {count}")

plot_data = {'Time Point': [], 'Mean Area': [], 'Class': []}
for class_type, time_points in areas_data.items():
    for time_point, areas in sorted(time_points.items()):
        mean_area = np.mean(areas)
        plot_data['Time Point'].append(time_point)
        plot_data['Mean Area'].append(mean_area)
        plot_data['Class'].append('Fed' if class_type == 'fed' else 'Starved')

overall_mean_area = {'Fed': [], 'Starved': []}

for class_type, time_points in areas_data.items():
    all_areas = [area for areas in time_points.values() for area in areas]
    class_label = 'Fed' if class_type == 'fed' else 'Starved'
    overall_mean_area[class_label] = np.mean(all_areas)

print(f"Overall Mean Area for Fed Cells: {overall_mean_area['Fed']:.2f}")
print(f"Overall Mean Area for Starved Cells: {overall_mean_area['Starved']:.2f}")

plot_df = pd.DataFrame(plot_data)

custom_colors = {"Fed": "#E7E439", "Starved": "#9E4B4B"}

plt.style.use("ggplot")
plt.figure(figsize=(8, 5))
sns.lineplot(data=plot_df, x='Time Point', y='Mean Area', hue='Class', marker='o', palette=custom_colors)
plt.xlabel('Time Point', fontsize=18)
plt.ylabel('Mean Area', fontsize=18)
plt.xticks(rotation=45, fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
