import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm
import cv2


def identify_merge_groups(prev_bboxes, overlap_threshold):
    """
    Identify groups of overlapping bounding boxes, focusing on merges indicated by a larger box.
    
    :param prev_bboxes: List of previous bounding boxes with their properties.
    :param overlap_threshold: The minimum overlap ratio to consider two boxes as overlapping.
    :return: A list of sets, each containing the indices of bboxes considered to be merging.
    """
    merge_groups = []
    for i, bbox_i in enumerate(prev_bboxes):
        if not bbox_i['active']:  # Skip inactive boxes
            continue
        for j, bbox_j in enumerate(prev_bboxes):
            if i != j and bbox_j['active'] and calculate_overlap_area(bbox_i['bbox'], bbox_j['bbox']) >= overlap_threshold:
                # Find if either box is already part of a merge group
                found_group = False
                for group in merge_groups:
                    if i in group or j in group:
                        group.update([i, j])
                        found_group = True
                        break
                if not found_group:
                    merge_groups.append({i, j})
    return [list(group) for group in merge_groups]



def read_bboxes(file_path):
    with open(file_path, 'r') as f:
        bboxes = []
        features = []
        for line in f.readlines():
            parts = line.split()
            bbox = list(map(float, parts[1:5]))
            feature = list(map(float, parts[5:]))
            bboxes.append(bbox)
            features.append(feature)
    return bboxes, features

def calculate_euclidean_distance(bbox1, bbox2):
    center1 = np.array([bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2])
    center2 = np.array([bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2])
    return np.linalg.norm(center1 - center2)

def calculate_overlap_area(box1, box2):
    """Calculate the area of overlap between two bounding boxes."""
    x1_max = box1[0] + box1[2] / 2
    x1_min = box1[0] - box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    y1_min = box1[1] - box1[3] / 2

    x2_max = box2[0] + box2[2] / 2
    x2_min = box2[0] - box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    y2_min = box2[1] - box2[3] / 2

    overlap_width = min(x1_max, x2_max) - max(x1_min, x2_min)
    overlap_height = min(y1_max, y2_max) - max(y1_min, y2_min)

    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0  # No overlap
    else:
        return overlap_width * overlap_height

def calculate_iou_cost(bbox1, bbox2):
    """
    Calculate the IoU cost for two bounding boxes.
    """
    # Calculate the intersection area
    inter_area = calculate_overlap_area(bbox1, bbox2)
    # Calculate the union area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    # Convert IoU to a cost (1-IoU ensures higher IoU results in lower cost)
    iou_cost = 1 - iou
    return iou_cost

def calculate_cosine_similarity(feature1, feature2):
    return np.dot(feature1, feature2) / (norm(feature1) * norm(feature2))

def get_random_color(exclude_green=True):
    while True:
        color = np.random.randint(0, 256, 3).tolist()
        if not exclude_green or color[1] < max(color[0], color[2]):
            break
    return tuple(color)

def update_history(tracked_object, new_bbox):
    history_limit = 5  # Keep the history of the last 5 frames
    tracked_object['history'].append(new_bbox)
    if len(tracked_object['history']) > history_limit:
        tracked_object['history'].pop(0)

def calculate_average_bbox(history):
    # Calculate the average bbox from the history
    avg_bbox = np.mean(np.array(history), axis=0)
    return avg_bbox.tolist()

def match_with_history(current_bbox, tracked_objects):
    # Example of a simple matching strategy using historical average
    best_match_id = None
    best_match_distance = np.inf
    for obj in tracked_objects:
        avg_bbox = calculate_average_bbox(obj['history'])
        distance = calculate_euclidean_distance(current_bbox, avg_bbox)
        if distance < best_match_distance:
            best_match_id = obj['id']
            best_match_distance = distance
    return best_match_id
