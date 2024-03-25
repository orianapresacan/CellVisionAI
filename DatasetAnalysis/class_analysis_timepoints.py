import os
import matplotlib.pyplot as plt
from collections import defaultdict


def count_cells_by_timepoint_and_class(folder_path):
    cell_counts = defaultdict(lambda: defaultdict(int))

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            timepoint = filename.split('_')[1]
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        class_id = int(parts[0])
                        # Assuming 1: fed, 2: starved, 3: unidentified
                        if class_id == 1:
                            cell_counts[timepoint]['fed'] += 1
                        elif class_id == 2:
                            cell_counts[timepoint]['starved'] += 1
                        elif class_id == 3:
                            cell_counts[timepoint]['unidentified'] += 1

    return cell_counts

def plot_time_series(cell_counts):
    sorted_counts = dict(sorted(cell_counts.items()))
    
    time_points = list(sorted_counts.keys())
    fed_counts = [counts['fed'] for counts in sorted_counts.values()]
    starved_counts = [counts['starved'] for counts in sorted_counts.values()]
    unidentified_counts = [counts['unidentified'] for counts in sorted_counts.values()]

    plt.style.use("ggplot")  

    plt.figure(figsize=(8, 5))
    plt.plot(time_points, fed_counts, 'o-', color='#E7E439', linewidth=2, markersize=5, label='Fed')
    plt.plot(time_points, starved_counts, 'o-', color='#9E4B4B', linewidth=2, markersize=5, label='Starved')
    plt.plot(time_points, unidentified_counts, 'o-', color='#59B970', linewidth=2, markersize=5, label='Unidentified')
    
    plt.xlabel('Time Point', fontsize=18)
    plt.ylabel('Number of Cells', fontsize=18)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    plt.tight_layout()
    
    ax = plt.gca()  
    ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    
    plt.show()


folder_path = 'I_N_starved_images_bbox' 
cell_counts = count_cells_by_timepoint_and_class(folder_path)
plot_time_series(cell_counts)
