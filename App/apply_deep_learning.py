import object_detection, classification, segmentation
import cv2
import numpy as np
import os 


class_counters = {}

def model_process(filePath):
    object_detection.get_detections(filePath)
    image = cv2.imread(filePath)
    image_height, image_width, _ = image.shape

    base_directory = os.path.dirname(filePath)
    base_filename = os.path.splitext(os.path.basename(filePath))[0]
    
    txt_filename = f"{base_filename}.txt"
    txt_filePath = os.path.join(base_directory, txt_filename)

    bounding_boxes = object_detection.read_bounding_boxes(txt_filePath, image_width, image_height)

    base_dir = "segmentation_masks"
    ensure_folder_exists(base_dir)

    updated_class_labels = []
    all_masks = []
    for box in bounding_boxes:
        mask_image, class_label = process_image_and_classify(image, box)
        class_folder = os.path.join(base_dir, str(class_label))
        ensure_folder_exists(class_folder)

        if class_label not in class_counters:
            class_counters[class_label] = 0
        else:
            class_counters[class_label] += 1

        updated_class_labels.append(class_label)

        mask_filename = f"{base_filename}_{class_counters[class_label]}.png"
        mask_path = os.path.join(class_folder, mask_filename)
        all_masks.append(mask_image)
        cv2.imwrite(mask_path, mask_image)

    update_bounding_box_file(txt_filename, updated_class_labels, bounding_boxes)


def process_image_and_classify(image, box):
    x, y, w, h = box
    cropped_image = image[y:y+h, x:x+w]
    
    mask = segmentation.get_segmentation(cropped_image)
    resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    class_label = classification.get_classification(cropped_image, resized_mask)  

    full_mask = np.zeros_like(image, dtype=np.uint8)

    expanded_mask = np.stack([resized_mask]*3, axis=-1)
    full_mask[y:y+h, x:x+w] = expanded_mask  
    return full_mask, class_label
    

def update_bounding_box_file(file_path, new_labels, boxes):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    updated_lines = []
    for line, new_label, box in zip(lines, new_labels, boxes):
        parts = line.split()
        _, xc, yc, w, h = parts
        updated_line = f"{new_label} {xc} {yc} {w} {h}\n"
        updated_lines.append(updated_line)
    
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)


def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)