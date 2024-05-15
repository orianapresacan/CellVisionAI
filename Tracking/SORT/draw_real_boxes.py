import cv2
import numpy as np
import os

# Load the detection output from file
detection_file = 'data/train/C03_s3/det/det.txt'
image_dir = 'data/train/C03_s3/images/'
output_dir = 'data/train/C03_s3/processed_images/'  # Directory to save processed images

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read detection data and organize by frame
detection_data_by_frame = {}
with open(detection_file, 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        frame_num = int(parts[0])
        bbox = [float(coord) for coord in parts[2:6]]  # x, y, w, h
        detection_data_by_frame.setdefault(frame_num, []).append(bbox)

# Function to draw bounding boxes on an image
def draw_bboxes(image_path, bboxes, output_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Draw each bounding box in red
    for bbox in bboxes:
        x, y, w, h = bbox
        top_left = (int(x * width), int(y * height))
        bottom_right = (int((x + w) * width), int((y + h) * height))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)  # Red color
    
    cv2.imwrite(output_path, image)

# Process and save images with drawn bounding boxes
for frame_num, bboxes in detection_data_by_frame.items():
    image_path = f"{image_dir}{frame_num:06d}.jpg"
    output_path = f"{output_dir}{frame_num:06d}_bboxes.jpg"  # Naming processed images
    draw_bboxes(image_path, bboxes, output_path)
