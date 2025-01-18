import json
import os
import numpy as np
import cv2
from PIL import Image

image_folder = 'images'
# annotation_folder = 'bounding-boxes'
mask_folder = 'masks'  
output_json_path = 'data/annotations.json'

def mask_to_bbox_area_segmentation(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    bbox = [np.inf, np.inf, -np.inf, -np.inf] 
    area = 0

    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()
        if approx.ndim == 1 or len(approx) < 3: 
            continue
        segmentation.append(approx.flatten().tolist())

        x, y, w, h = cv2.boundingRect(contour)
        bbox[0] = min(bbox[0], x)
        bbox[1] = min(bbox[1], y)
        bbox[2] = max(bbox[2], x + w)
        bbox[3] = max(bbox[3], y + h)

        area += cv2.contourArea(contour)

    if bbox[0] == np.inf or bbox[2] == -np.inf:  #
        return None, None, None  

    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

    if bbox[2] <= 0 or bbox[3] <= 0: 
        return None, None, None  

    return bbox, area, segmentation

coco_dataset = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "cell"}]
}

annotation_id = 1
for img_id, image_name in enumerate(os.listdir(image_folder), start=1):
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path)
    width, height = image.size

    coco_dataset['images'].append({
        "id": img_id,
        "file_name": image_name,
        "width": width,
        "height": height
    })

    mask_prefix = os.path.splitext(image_name)[0] + '_mask'
    for mask_file in os.listdir(mask_folder):
        if mask_file.startswith(mask_prefix):
            mask_path = os.path.join(mask_folder, mask_file)
            bbox, area, segmentation = mask_to_bbox_area_segmentation(mask_path)

            coco_dataset['annotations'].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0
            })
            annotation_id += 1

with open(output_json_path, 'w') as json_file:
    json.dump(coco_dataset, json_file)
