import os
import cv2
import numpy as np
import shutil
import json

# Paths
MASK_FOLDER = "data/masks"
IMAGE_FOLDER = "data/images"
DIVIDED_FOLDER = "divided_data"
ANNOTATIONS_OUTPUT_FOLDER = "annotations"
DATA_DIVISION_FILE = "data_division.txt"

# Create necessary directories
os.makedirs(DIVIDED_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_OUTPUT_FOLDER, exist_ok=True)

# Step 1: Mix Images from Subfolders and Binarize Each Mask
def mix_and_binarize_masks(mask_folder):
    processed_masks = {}
    for image_name in os.listdir(mask_folder):
        image_path = os.path.join(mask_folder, image_name)
        if os.path.isdir(image_path):
            for subfolder in os.listdir(image_path):
                subfolder_path = os.path.join(image_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for mask_file in os.listdir(subfolder_path):
                        mask_path = os.path.join(subfolder_path, mask_file)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            print(f"Warning: Could not read mask {mask_path}")
                            continue
                        # Binarize the mask
                        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
                        combined_key = f"{image_name}_{os.path.splitext(mask_file)[0]}"
                        processed_masks[combined_key] = binary_mask
    return processed_masks

print("Mixing masks and binarizing...")
processed_masks = mix_and_binarize_masks(MASK_FOLDER)

# Save the binarized masks
BINARIZED_MASK_FOLDER = "binarized_masks"
os.makedirs(BINARIZED_MASK_FOLDER, exist_ok=True)
for mask_name, binary_mask in processed_masks.items():
    output_path = os.path.join(BINARIZED_MASK_FOLDER, f"{mask_name}.png")
    cv2.imwrite(output_path, binary_mask)

# Step 2: Divide Images and Masks into Train, Test, and Validation
def parse_data_division(file_path):
    split_data = {"train": [], "val": [], "test": []}
    current_split = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Train:"):
                current_split = "train"
            elif line.startswith("Validation:"):
                current_split = "val"
            elif line.startswith("Test:"):
                current_split = "test"
            elif line and current_split:
                split_data[current_split].append(line)
    return split_data

print("Dividing dataset into splits...")
split_data = parse_data_division(DATA_DIVISION_FILE)
for split, image_names in split_data.items():
    split_dir = os.path.join(DIVIDED_FOLDER, split)
    os.makedirs(split_dir, exist_ok=True)
    image_split_folder = os.path.join(split_dir, "images")
    mask_split_folder = os.path.join(split_dir, "masks")
    os.makedirs(image_split_folder, exist_ok=True)
    os.makedirs(mask_split_folder, exist_ok=True)

    for image_name in image_names:
        # Copy image
        image_path = os.path.join(IMAGE_FOLDER, f"{image_name}.jpg")
        if os.path.exists(image_path):
            shutil.copy(image_path, image_split_folder)
        else:
            print(f"Warning: Image {image_name}.jpg not found in {IMAGE_FOLDER}")

        # Copy corresponding masks
        for mask_file in os.listdir(BINARIZED_MASK_FOLDER):
            if mask_file.startswith(image_name):
                mask_path = os.path.join(BINARIZED_MASK_FOLDER, mask_file)
                shutil.copy(mask_path, mask_split_folder)

# Step 3: Generate COCO JSON Annotations for Each Split
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

    if bbox[0] == np.inf or bbox[2] == -np.inf:
        return None, None, None

    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

    if bbox[2] <= 0 or bbox[3] <= 0:
        return None, None, None

    return bbox, area, segmentation

print("Generating COCO JSON annotations...")
for split in ["train", "val", "test"]:
    split_folder = os.path.join(DIVIDED_FOLDER, split)
    image_folder = os.path.join(split_folder, "images")
    mask_folder = os.path.join(split_folder, "masks")
    output_json_path = os.path.join(ANNOTATIONS_OUTPUT_FOLDER, f"{split}_annotations.json")

    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "cell"}]
    }

    annotation_id = 1
    for img_id, image_name in enumerate(os.listdir(image_folder), start=1):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        height, width = image.shape[:2]

        coco_dataset['images'].append({
            "id": img_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        mask_prefix = os.path.splitext(image_name)[0]
        for mask_file in os.listdir(mask_folder):
            if mask_file.startswith(mask_prefix):
                mask_path = os.path.join(mask_folder, mask_file)
                bbox, area, segmentation = mask_to_bbox_area_segmentation(mask_path)

                if bbox is not None:
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

    with open(output_json_path, "w") as json_file:
        json.dump(coco_dataset, json_file)
    print(f"Annotations saved to {output_json_path}")
