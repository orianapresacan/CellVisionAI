import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os


def overlay_masks(image_path, annotations, color, alpha=0.5):
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape
    full_mask = np.zeros((height, width), dtype=np.uint8)

    for annotation in annotations:
        mask = np.zeros((height, width), dtype=np.uint8)
        for segmentation in annotation['segmentation']:
            np_poly = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [np_poly], 255)

        full_mask = cv2.bitwise_or(full_mask, mask)

    overlay = np.zeros_like(image)
    overlay[:, :] = color

    where_mask = full_mask.astype(bool)
    image[where_mask] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[where_mask]

    return image

annotations_path = 'data/sample_image/annotations.json'
with open(annotations_path, 'r') as file:
    data = json.load(file)

image_folder_path = 'data/sample_image/image'

mask_color = (128, 0, 128)  # Purple
mask_transparency = 0.5  

for img_info in data['images']:
    image_path = os.path.join(image_folder_path, img_info['file_name'])
    annotations_for_image = [anno for anno in data['annotations'] if anno['image_id'] == img_info['id']]

    overlaid_image = overlay_masks(image_path, annotations_for_image, mask_color, mask_transparency)

    if overlaid_image is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(overlaid_image)
        plt.axis('off')
        plt.show()
