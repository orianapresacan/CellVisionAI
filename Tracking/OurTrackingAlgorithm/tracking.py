import os
import cv2
import numpy as np
from helpers import *
from scipy.spatial import cKDTree


def calculate_cost_matrix_optimized(prev_bboxes, current_bboxes, current_features=None, distance_threshold=100.0, use_features=True):
    if not prev_bboxes or not current_bboxes:
        return np.array([])

    HIGH_COST = 1e9
    cost_matrix = np.full((len(prev_bboxes), len(current_bboxes)), HIGH_COST, dtype=np.float32)
    
    prev_centers = np.array([[bbox['bbox'][0], bbox['bbox'][1]] for bbox in prev_bboxes])
    current_centers = np.array([[bbox[0], bbox[1]] for bbox in current_bboxes])
    
    # Use KDTree to find nearby matches
    tree = cKDTree(current_centers)
    for i, prev_center in enumerate(prev_centers):
        indices = tree.query_ball_point(prev_center, r=distance_threshold)
        for j in indices:
            distance_cost = calculate_euclidean_distance(prev_bboxes[i]['bbox'], current_bboxes[j])
            iou_cost = calculate_iou_cost(prev_bboxes[i]['bbox'], current_bboxes[j])
            
            normalized_distance = distance_cost / distance_threshold  #  [0,1]
            
            total_cost = 1.0 * normalized_distance + 2.0 * iou_cost
            if use_features and current_features is not None:
                feature_cost = 1 - calculate_cosine_similarity(prev_bboxes[i]['features'], current_features[j])
                total_cost += 0.5 * feature_cost  # Add feature cost with weight
            cost_matrix[i, j] = total_cost
    
    print(f"Cost matrix stats - min: {cost_matrix.min()}, max: {cost_matrix.max()}, mean: {cost_matrix.mean()}, median: {np.median(cost_matrix)}")
    
    high_cost_assignments = np.sum(cost_matrix > 2.0)
    medium_cost_assignments = np.sum((cost_matrix > 1.0) & (cost_matrix <= 2.0))
    low_cost_assignments = np.sum(cost_matrix <= 1.0)
    print(f"Assignments - Low: {low_cost_assignments}, Medium: {medium_cost_assignments}, High: {high_cost_assignments}")
    
    return cost_matrix

def track_bboxes_with_masks(prev_bboxes, current_bboxes, current_masks, current_features=None, distance_threshold=100.0, max_age=4, use_features=True):
    if not prev_bboxes and not current_bboxes:
        return []
    
    if not prev_bboxes:
        new_bboxes = []
        next_id = 1
        for idx, (bbox, mask) in enumerate(zip(current_bboxes, current_masks)):
            new_bbox = {
                'id': next_id,
                'bbox': bbox,
                'mask': mask,
                'features': current_features[idx] if use_features and current_features else None,
                'age': 0,
                'merged': False,
                'active': True,
                'history': [bbox]
            }
            new_bboxes.append(new_bbox)
            next_id += 1
        return new_bboxes
    
    if not current_bboxes:
        for bbox_info in prev_bboxes:
            bbox_info['age'] += 1
            if bbox_info['age'] > max_age:
                bbox_info['active'] = False
        return prev_bboxes
    
    cost_matrix = calculate_cost_matrix_optimized(
        prev_bboxes,
        current_bboxes,
        current_features=current_features,
        distance_threshold=distance_threshold,
        use_features=use_features
    )
    
    if cost_matrix.size == 0:
        for bbox_info in prev_bboxes:
            bbox_info['age'] += 1
            if bbox_info['age'] > max_age:
                bbox_info['active'] = False
        next_id = max([bbox['id'] for bbox in prev_bboxes], default=0) + 1
        for idx, (bbox, mask) in enumerate(zip(current_bboxes, current_masks)):
            new_bbox = {
                'id': next_id,
                'bbox': bbox,
                'mask': mask,
                'features': current_features[idx] if use_features and current_features else None,
                'age': 0,
                'merged': False,
                'active': True,
                'history': [bbox]
            }
            prev_bboxes.append(new_bbox)
            next_id += 1
        return prev_bboxes
    
    assignments = hungarian_assignment(cost_matrix, max_cost=1.0)  
    updated_bboxes = []
    active_prev_indices = set()
    active_current_indices = set()
    
    for i, j in assignments:
        prev_bboxes[i]['bbox'] = current_bboxes[j]
        prev_bboxes[i]['mask'] = current_masks[j]
        if use_features and current_features is not None:
            prev_bboxes[i]['features'] = current_features[j]
        prev_bboxes[i]['age'] = 0  
        prev_bboxes[i]['active'] = True
        prev_bboxes[i]['history'].append(current_bboxes[j])
        updated_bboxes.append(prev_bboxes[i])
        active_prev_indices.add(i)
        active_current_indices.add(j)
    
    for i, bbox_info in enumerate(prev_bboxes):
        if i not in active_prev_indices:
            bbox_info['age'] += 1  
            if bbox_info['age'] > max_age:
                bbox_info['active'] = False  
            updated_bboxes.append(bbox_info)
    
    unmatched_current_indices = set(range(len(current_bboxes))) - active_current_indices
    next_id = max([bbox['id'] for bbox in prev_bboxes], default=0) + 1
    for idx in unmatched_current_indices:
        new_bbox = {
            'id': next_id,
            'bbox': current_bboxes[idx],
            'mask': current_masks[idx],
            'features': current_features[idx] if use_features and current_features else None,
            'age': 0,
            'merged': False,
            'active': True,
            'history': [current_bboxes[idx]]
        }
        updated_bboxes.append(new_bbox)
        next_id += 1
    
    return updated_bboxes

def process_bboxes(image, tracked_bboxes, folder_path, frame_number, image_path=None, consistent_ids=None, use_features=True):
    h, w = image.shape[:2]  
    image_with_drawings = image.copy()
    
    for bbox_info in tracked_bboxes:
        if not bbox_info.get('active', False) or (consistent_ids and bbox_info['id'] not in consistent_ids):
            continue  
        
        bbox = bbox_info['bbox']
        
        x1, y1 = int((bbox[0] - bbox[2] / 2) * w), int((bbox[1] - bbox[3] / 2) * h)
        x2, y2 = int((bbox[0] + bbox[2] / 2) * w), int((bbox[1] + bbox[3] / 2) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if CROP_SAVE:
            cropped = image[y1:y2, x1:x2] 
            if cropped.size != 0:
                output_path = os.path.join(folder_path, f'cell_{bbox_info["id"]}_frame_{frame_number}.png')
                cv2.imwrite(output_path, cropped)  
        
        if DRAW:
            if bbox_info['id'] not in id_colors:
                id_colors[bbox_info['id']] = get_random_color()
            color = id_colors[bbox_info['id']]
            cv2.rectangle(image_with_drawings, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_with_drawings, str(bbox_info['id']), (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if use_features and bbox_info.get('features') is not None:
                feature = bbox_info['features']
                if len(feature) >= 2:
                    start_point = (x1, y1)
                    end_point = (x1 + int(feature[0] * 10), y1 + int(feature[1] * 10))  # Scale for visibility
                    cv2.arrowedLine(image_with_drawings, start_point, end_point, color, 1, tipLength=0.3)
    
    if DRAW and image_path:
        output_path = os.path.join(folder_path, os.path.basename(image_path))
        cv2.imwrite(output_path, image_with_drawings)

CROP_SAVE = True
DRAW = True
use_features = False
id_colors = {}
main_folder = 'data/C03_s3'

active_ids_per_frame = {}

prev_bboxes = None  # Memory for tracked cells
consistently_active_ids = None

all_images = sorted([f for f in os.listdir(main_folder) if f.endswith('.jpg')])
all_texts = sorted([f for f in os.listdir(main_folder) if f.endswith('.txt')])

for frame_number, (image_file, text_file) in enumerate(zip(all_images, all_texts)):
    image_path = os.path.join(main_folder, image_file)
    text_path = os.path.join(main_folder, text_file)
    base_name = os.path.splitext(image_file)[0]
    mask_folder = os.path.join(main_folder, base_name)
    
    print(f"--- Processing frame {frame_number} ---")
    print(f"Image: {image_file}, Text: {text_file}, Mask folder: {mask_folder}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_file}")
        continue  #
    
    current_bboxes, current_features = read_bboxes(text_path, use_features=use_features)
    current_masks, current_mask_bboxes = load_masks_and_bboxes(mask_folder)
    
    print(f"Frame {frame_number}: Detected {len(current_bboxes)} cells.")
    
    if frame_number == 0:
        if use_features and current_features:
            feature_iter = current_features
        else:
            feature_iter = [None] * len(current_bboxes)  
        
        prev_bboxes = [
            {
                'id': idx + 1,
                'bbox': bbox,
                'mask': msk,
                'features': feat,
                'merged': False,
                'active': True,
                'history': [bbox],
                'age': 0
            }
            for idx, (bbox, msk, feat) in enumerate(zip(current_bboxes, current_masks, feature_iter))
        ]
        
        consistently_active_ids = {b['id'] for b in prev_bboxes}
        active_ids = consistently_active_ids  
    else:
        prev_bboxes = track_bboxes_with_masks(
            prev_bboxes,
            current_bboxes,
            current_masks,
            current_features,
            distance_threshold=30.0,  # Ensure consistent threshold
            max_age=0, 
            use_features=use_features
        )
        
        active_ids = {bbox['id'] for bbox in prev_bboxes if bbox.get('active', False)}
        
        if consistently_active_ids is not None:
            consistently_active_ids &= active_ids
        else:
            consistently_active_ids = active_ids.copy()
    
    output_folder = os.path.join(main_folder, f'frame_{frame_number+1}')
    os.makedirs(output_folder, exist_ok=True)
    
    process_bboxes(image, prev_bboxes, output_folder, frame_number, image_path, consistent_ids=consistently_active_ids, use_features=use_features)
    
    active_ids_per_frame[frame_number] = active_ids  

if active_ids_per_frame:
    final_consistent_ids = set.intersection(*active_ids_per_frame.values())
    print(f"\nTotal consistently active cells across all frames: {len(final_consistent_ids)}")
else:
    print("No frames processed.")
