import os


def count_cells(folder_path='bounding-boxes'):
    class_counts = {1: 0, 2: 0, 3: 0}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    class_info = line.strip().split()[0]  
                    if class_info.isdigit(): 
                        class_id = int(class_info)
                        if class_id in class_counts:
                            class_counts[class_id] += 1

    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} objects")

    return class_counts


def find_max_min_objects(folder_path='bounding-boxes'):
    max_objects = 0  
    min_objects = float('inf')  

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                num_objects = len(lines)  
                
                if num_objects > max_objects:
                    max_objects = num_objects
                if num_objects < min_objects:
                    min_objects = num_objects

    if min_objects == float('inf'):
        min_objects = 0

    return max_objects, min_objects

max_objects, min_objects = find_max_min_objects()

print(max_objects)
print(min_objects)

count_cells()