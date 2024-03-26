import os
import glob
import re

root_directory = 'bounding_boxes_consistent'  

def convert_yolo_to_mot(folder_path):
    output_file_path = os.path.join(folder_path, 'det.txt')
    with open(output_file_path, 'w') as output_file:
        for frame_file in sorted(glob.glob(os.path.join(folder_path, 'Timepoint_*.txt'))):
            # Extract frame number using regular expression
            match = re.search(r"Timepoint_(\d+)_", frame_file)
            if not match:
                continue
            frame_number = int(match.group(1))
            with open(frame_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # Skip invalid lines
                    # Convert YOLO format to MOT Challenge format
                    # Skip the class ID, and assign a confidence score of 1 (max)
                    center_x, center_y, width, height = map(float, parts[1:5])
                    top_left_x = center_x - (width / 2)
                    top_left_y = center_y - (height / 2)
                    output_line = f"{frame_number},-1,{top_left_x},{top_left_y},{width},{height},1,-1,-1,-1\n"
                    output_file.write(output_line)

for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    if os.path.isdir(folder_path):
        convert_yolo_to_mot(folder_path)
