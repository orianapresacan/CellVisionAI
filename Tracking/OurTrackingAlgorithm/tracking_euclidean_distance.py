import os
import cv2
import numpy as np

def read_bboxes(file_path):
    with open(file_path, 'r') as f:
        bboxes = [list(map(float, line.split()[1:])) for line in f.readlines()]
    return bboxes

def calculate_euclidean_distance(bbox1, bbox2):
    center1 = np.array([bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2])
    center2 = np.array([bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2])
    return np.linalg.norm(center1 - center2)

def track_bboxes(prev_bboxes, current_bboxes, distance_threshold=0.15):
    tracked = [None] * len(prev_bboxes)
    used_indices = set()

    for i, prev_bbox in enumerate(prev_bboxes):
        min_distance = float('inf')
        min_index = -1

        for j, current_bbox in enumerate(current_bboxes):
            if j in used_indices:
                continue

            distance = calculate_euclidean_distance(prev_bbox['bbox'], current_bbox)
            if distance < min_distance:
                min_distance = distance
                min_index = j

        if min_distance < distance_threshold and min_index != -1:
            tracked[i] = {'id': prev_bbox['id'], 'bbox': current_bboxes[min_index]}
            used_indices.add(min_index)

    return [t for t in tracked if t is not None]

def crop_and_save(image, tracked_bboxes, folder_path, frame_number):
    h, w = image.shape[:2]
    for bbox_info in tracked_bboxes:
        bbox = bbox_info['bbox']
        x1, y1 = int((bbox[0] - bbox[2] / 2) * w), int((bbox[1] - bbox[3] / 2) * h)
        x2, y2 = int((bbox[0] + bbox[2] / 2) * w), int((bbox[1] + bbox[3] / 2) * h)

        # Check if coordinates are within image boundaries
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)

        # Check if the bbox is valid
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid bbox for cell_{bbox_info['id']} in frame {frame_number}: [{x1}, {y1}, {x2}, {y2}]")
            continue

        cropped = image[y1:y2, x1:x2]

        # Check if the cropped image is empty
        if cropped.size == 0:
            print(f"Empty crop for cell_{bbox_info['id']} in frame {frame_number}")
            continue

        # Resize the cropped image to 64x64
        resized_cropped = cv2.resize(cropped, (64, 64))

        cv2.imwrite(os.path.join(folder_path, f'cell_{bbox_info["id"]}.png'), resized_cropped)


main_folder = 'data'
subfolders = sorted([os.path.join(main_folder, f) for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))])

for subfolder in subfolders:
    image_files = sorted([f for f in os.listdir(subfolder) if f.endswith('.jpg') or f.endswith('.png')])
    text_files = sorted([f for f in os.listdir(subfolder) if f.endswith('.txt')])
    prev_bboxes = None

    for frame_number, (image_file, text_file) in enumerate(zip(image_files, text_files)):
        image_path = os.path.join(subfolder, image_file)
        text_path = os.path.join(subfolder, text_file)
        
        image = cv2.imread(image_path)
        current_bboxes = read_bboxes(text_path)

        if frame_number == 0:
            # Initialize tracking in the first frame
            prev_bboxes = [{'id': idx+1, 'bbox': bbox} for idx, bbox in enumerate(current_bboxes)]
        else:
            prev_bboxes = track_bboxes(prev_bboxes, current_bboxes)
        
        output_folder = os.path.join(subfolder, f'frame_{frame_number+1}')
        os.makedirs(output_folder, exist_ok=True)
        crop_and_save(image, prev_bboxes, output_folder, frame_number)
    break

