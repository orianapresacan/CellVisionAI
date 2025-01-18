import numpy as np
from numpy.linalg import norm
import cv2
import os
from scipy.optimize import linear_sum_assignment


def bbox_to_xyxy(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def calculate_overlap_area(box1, box2):
    box1_xyxy = bbox_to_xyxy(box1)
    box2_xyxy = bbox_to_xyxy(box2)
    
    x1_max = min(box1_xyxy[2], box2_xyxy[2])
    x1_min = max(box1_xyxy[0], box2_xyxy[0])
    y1_max = min(box1_xyxy[3], box2_xyxy[3])
    y1_min = max(box1_xyxy[1], box2_xyxy[1])
    
    overlap_width = x1_max - x1_min
    overlap_height = y1_max - y1_min

    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0
    else:
        return overlap_width * overlap_height

def calculate_iou_cost(bbox1, bbox2):
    inter_area = calculate_overlap_area(bbox1, bbox2)
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    union_area = bbox1_area + bbox2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    iou_cost = 1 - iou
    return iou_cost

def calculate_euclidean_distance(bbox1, bbox2):
    center1 = np.array([bbox1[0], bbox1[1]])
    center2 = np.array([bbox2[0], bbox2[1]])
    return np.linalg.norm(center1 - center2)

def calculate_cosine_similarity(feature1, feature2):
    if feature1 is None or feature2 is None:
        return 0.0
    norm1 = norm(feature1)
    norm2 = norm(feature2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(feature1, feature2) / (norm1 * norm2)

def read_bboxes(file_path, use_features=True):
    with open(file_path, 'r') as f:
        bboxes = []
        features = [] if use_features else None
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Warning: Line {line_num} in {file_path} is malformed. Skipping.")
                continue
            try:
                bbox = list(map(float, parts[1:5]))
                bboxes.append(bbox)
                if use_features:
                    if len(parts) < 6:
                        print(f"Warning: Line {line_num} in {file_path} lacks feature data. Appending zero vector.")
                        feature = [0.0]
                    else:
                        feature = list(map(float, parts[5:]))
                        norm_val = norm(feature)
                        if norm_val != 0:
                            feature = [f / norm_val for f in feature]
                    features.append(feature)
            except ValueError as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}. Skipping.")
                continue
    return (bboxes, features) if use_features else (bboxes, None)

def load_masks_and_bboxes(mask_folder):
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png') or f.endswith('.jpg')])
    masks = []
    bboxes = []
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Could not read mask {mask_file}")
            continue  
        
        _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.size == 0:
                continue
            single_mask = np.zeros_like(binary_mask)
            cv2.drawContours(single_mask, [contour], -1, 1, thickness=cv2.FILLED)
            masks.append(single_mask)
            
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x + w/2, y + h/2, w, h]) 
    
    print(f"Loaded {len(bboxes)} bounding boxes from masks.")
    return masks, bboxes

def hungarian_assignment(cost_matrix, max_cost=3.5):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_cost:
            assignments.append((i, j))
    print(f"Number of assignments: {len(assignments)}")
    return assignments

def get_random_color(exclude_green=True):
    while True:
        color = np.random.randint(0, 256, 3).tolist()
        if not exclude_green or color[1] < max(color[0], color[2]):
            break
    return tuple(color)

def update_history(tracked_object, new_bbox):
    history_limit = 5  
    tracked_object['history'].append(new_bbox)
    if len(tracked_object['history']) > history_limit:
        tracked_object['history'].pop(0)

def calculate_average_bbox(history):
    avg_bbox = np.mean(np.array(history), axis=0)
    return avg_bbox.tolist()

def match_with_history(current_bbox, tracked_objects):
    best_match_id = None
    best_match_distance = np.inf
    for obj in tracked_objects:
        avg_bbox = calculate_average_bbox(obj['history'])
        distance = calculate_euclidean_distance(current_bbox, avg_bbox)
        if distance < best_match_distance:
            best_match_id = obj['id']
            best_match_distance = distance
    return best_match_id
