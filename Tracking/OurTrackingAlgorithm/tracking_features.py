import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm
from helpers import identify_merge_groups, match_with_history, update_history, get_random_color, read_bboxes, calculate_cosine_similarity, calculate_euclidean_distance, calculate_iou_cost, calculate_overlap_area


def detect_merge(prev_bboxes, current_bbox, size_increase_threshold=1.3, overlap_threshold=0.7):
    """Detect if a merge has occurred based on size increase and overlap."""
    current_area = current_bbox[2] * current_bbox[3]
    overlaps = []

    for prev_bbox in prev_bboxes:
        prev_area = prev_bbox['bbox'][2] * prev_bbox['bbox'][3]
        
        # Check for significant size increase
        if current_area >= prev_area * size_increase_threshold:
            overlap_area = calculate_overlap_area(prev_bbox['bbox'], current_bbox)
            overlap_ratio = overlap_area / prev_area
            overlaps.append(overlap_ratio)

    # If multiple previous bboxes significantly overlap with the current bbox, a merge is likely
    if len([overlap for overlap in overlaps if overlap > overlap_threshold]) >= 2:
        return True
    return False

def calculate_cost_matrix(prev_bboxes, current_bboxes, current_features, num_neighbors=4, image_size=None):
    # Placeholder for the final cost matrix with a high default cost
    HIGH_COST = 1e9
    cost_matrix = np.full((len(prev_bboxes), len(current_bboxes)), HIGH_COST, dtype=np.float32)
    
    if image_size is not None:
        h, w = image_size

    for i, prev_bbox in enumerate(prev_bboxes):
        distances = [calculate_euclidean_distance(prev_bbox['bbox'], curr['bbox']) for curr in current_bboxes]
        sorted_indices = np.argsort(distances)
        
        neighbors_indices = sorted_indices[:1+num_neighbors]
        
        for j in neighbors_indices:
            current_bbox = current_bboxes[j]
            distance_cost = calculate_euclidean_distance(prev_bbox['bbox'], current_bbox['bbox'])
            iou_cost = calculate_iou_cost(prev_bbox['bbox'], current_bbox['bbox'])
            feature_similarity = calculate_cosine_similarity(prev_bbox['features'], current_features[j])
            feature_cost = 1 - feature_similarity
            
            # Calculate the total cost for these bboxes, including distance cost
            total_cost = distance_cost #feature_cost + distance_cost*3 + iou_cost
            cost_matrix[i, j] = total_cost
    
    return cost_matrix

def track_bboxes(prev_bboxes, current_bboxes, current_features, max_age=4, distance_threshold=0.04):
    cost_matrix = calculate_cost_matrix(prev_bboxes, current_bboxes, current_features)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    updated_bboxes = []
    active_indices = set()  # To keep track of matched (active) prev_bboxes indices

    # Mark all as initially inactive before checking for matches
    for bbox_info in prev_bboxes:
        bbox_info['active'] = False
    print("FRAME")
   
    for i, j in zip(row_ind, col_ind):
        print(cost_matrix[i, j])

        if cost_matrix[i, j] < distance_threshold:
            prev_bboxes[i]['history'].append(current_bboxes[j])

            if len(prev_bboxes[i]['history']) > max_age:
                prev_bboxes[i]['history'].pop(0)
            # Update current bbox with the matched one
            prev_bboxes[i]['bbox'] = current_bboxes[j]['bbox']
            prev_bboxes[i]['features'] = current_features[j]
            prev_bboxes[i]['age'] = 0
            prev_bboxes[i]['merged'] = False
            prev_bboxes[i]['active'] = True
            prev_bboxes[i]['history'].append(current_bboxes[j])  # Ensure this matches expected structure
            updated_bboxes.append(prev_bboxes[i])
            active_indices.add(i)

    # Mark unmatched prev_bboxes as inactive
    for i, bbox_info in enumerate(prev_bboxes):
        if i not in active_indices:
            bbox_info['age'] += 1  # Increment age for unmatched bboxes
            if bbox_info['age'] > max_age:
                bbox_info['merged'] = True  # Consider permanently inactive if beyond max_age

    # Append unmatched prev_bboxes if still within age limit
    updated_bboxes.extend([bbox_info for i, bbox_info in enumerate(prev_bboxes) if i not in active_indices and bbox_info['age'] <= max_age])

    # Handle new bboxes
    unmatched_current_indices = set(range(len(current_bboxes))) - set(col_ind)
    next_id = max([bbox['id'] for bbox in prev_bboxes], default=0) + 1
    for idx in unmatched_current_indices:
        new_bbox_info = {
            'id': next_id,
            'bbox': current_bboxes[idx]['bbox'] if isinstance(current_bboxes[idx], dict) else current_bboxes[idx],
            'features': current_features[idx],
            'age': 0,  # Ensure 'age' is initialized here
            'merged': False,
            'active': True,  # New detections are considered active by default
            'history': [current_bboxes[idx]['bbox'] if isinstance(current_bboxes[idx], dict) else current_bboxes[idx]]  # Initialize history with the current bbox
        }
        updated_bboxes.append(new_bbox_info)
        next_id += 1

    return updated_bboxes

def process_bboxes(image, tracked_bboxes, folder_path, frame_number, image_path=None):
    h, w = image.shape[:2]
    for bbox_info in tracked_bboxes:
        if not bbox_info.get('active', False):
            continue  # Skip drawing if bbox is not active

        bbox = bbox_info['bbox']

        if bbox_info['id'] != 115 and bbox_info['id'] != 172:
            continue
        
        # Directly unpack the bbox coordinates
        try:
            x_center, y_center, bbox_w, bbox_h = bbox
        except ValueError:
            print(f"Error with bbox format: {bbox}, skipping...")
            continue
        
        x1, y1 = int((x_center - bbox_w / 2) * w), int((y_center - bbox_h / 2) * h)
        x2, y2 = int((x_center + bbox_w / 2) * w), int((y_center + bbox_h / 2) * h)

        if DRAW:
            if bbox_info['id'] not in id_colors:
                id_colors[bbox_info['id']] = get_random_color()
            color = id_colors[bbox_info['id']]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, str(bbox_info['id']), (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if DRAW and image_path:
        output_path = os.path.join(folder_path, os.path.basename(image_path))
        cv2.imwrite(output_path, image)

    if CROP_SAVE:
        cropped = image[y1:y2, x1:x2]
        if cropped.size != 0:
            cv2.imwrite(os.path.join(folder_path, f'cell_{bbox_info["id"]}.png'), cropped)


CROP_SAVE = False
DRAW = True
id_colors = {}
main_folder = 'data'
subfolders = sorted([os.path.join(main_folder, f) for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))])

for subfolder in subfolders:
    image_files = sorted([f for f in os.listdir(subfolder) if f.endswith('.jpg') or f.endswith('.png')])
    text_files = sorted([f for f in os.listdir(subfolder) if f.endswith('.txt')])
    prev_bboxes = None

    active_ids_per_frame = {}

    for frame_number, (image_file, text_file) in enumerate(zip(image_files, text_files)):
        image_path = os.path.join(subfolder, image_file)
        text_path = os.path.join(subfolder, text_file)
        
        image = cv2.imread(image_path)
        current_bboxes, current_features = read_bboxes(text_path)

        if frame_number == 0:
            prev_bboxes = [
                {'id': idx + 1, 'bbox': bbox, 'features': feature, 'merged': False, 'active': True, 'history': [bbox], 'age': 0}
                for idx, (bbox, feature) in enumerate(zip(current_bboxes, current_features))
            ]
        else:
            prev_bboxes = track_bboxes(prev_bboxes, [{'bbox': bbox, 'features': feature} for bbox, feature in zip(current_bboxes, current_features)], current_features)

            # Process merge groups
            merge_groups = identify_merge_groups(prev_bboxes, overlap_threshold=0.5)
            for group in merge_groups:
                # Determine the largest cell in the group based on the area
                largest_cell_index = max(group, key=lambda idx: prev_bboxes[idx]['bbox'][2] * prev_bboxes[idx]['bbox'][3])
                
                # Mark other cells in the group as inactive
                for idx in group:
                    if idx != largest_cell_index:
                        prev_bboxes[idx]['active'] = False
                        prev_bboxes[idx]['merged'] = True

        active_ids_per_frame[frame_number] = {bbox['id'] for bbox in prev_bboxes if bbox.get('active', False)}
        
        output_folder = os.path.join(subfolder, f'frame_{frame_number+1}')
        os.makedirs(output_folder, exist_ok=True)
        process_bboxes(image, prev_bboxes, output_folder, frame_number, image_path if DRAW else None)
    # After all frames have been processed
    consistently_active_ids = set.intersection(*active_ids_per_frame.values())
    count_consistently_active_cells = len(consistently_active_ids)
    print(f"Count of consistently active cells across all frames: {count_consistently_active_cells}")

    break
