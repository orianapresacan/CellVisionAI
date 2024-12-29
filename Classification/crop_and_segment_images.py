import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(file_path):
    """Ensure the directory of the file path exists."""
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

def process_images(images_dir, bboxes_dir, masks_dir, output_dir, max_attempts=10):
    for image_name in sorted(os.listdir(images_dir)):
        if not image_name.endswith('.jpg'):
            continue

        base_name = image_name.split('.')[0]
        image_path = os.path.join(images_dir, image_name)
        bbox_path = os.path.join(bboxes_dir, base_name + '.txt')

        image = cv2.imread(image_path)
        if image is None:
            continue  # Skip if the image cannot be loaded

        with open(bbox_path, 'r') as bbox_file:
            bboxes = bbox_file.readlines()

        for idx, bbox_line in enumerate(bboxes):
            parts = bbox_line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            class_folders = ['fed', 'unfed', 'unidentified']
            class_folder = class_folders[int(class_id) - 1]

            abs_width, abs_height = int(width * image.shape[1]), int(height * image.shape[0])
            if abs_width <= 0 or abs_height <= 0:
                print(f"Invalid bounding box dimensions for {base_name}_{idx}. Skipping.")
                continue
            abs_x_center, abs_y_center = int(x_center * image.shape[1]), int(y_center * image.shape[0])
            x_start, y_start = int(abs_x_center - abs_width / 2), int(abs_y_center - abs_height / 2)

            cropped_image = image[y_start:y_start+abs_height, x_start:x_start+abs_width]

            # Try to find a mask that exists, incrementing the mask index if necessary
            mask_found = False
            attempt = 0
            while not mask_found and attempt < max_attempts:
                mask_path = os.path.join(masks_dir, base_name, class_folder, f"{base_name}_{idx + attempt}.png")
                if os.path.exists(mask_path):
                    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_image is None or np.all(mask_image == 0):
                        print(f"Mask is empty or completely black for {mask_path}.")
                    else:
                        cropped_mask = mask_image[y_start:y_start+abs_height, x_start:x_start+abs_width]
                        segmented_image = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)
                        if segmented_image.size > 0:
                            mask_found = True
                            output_class_dir = os.path.join(output_dir, base_name, class_folder)
                            ensure_dir(output_class_dir)
                            output_image_path = os.path.abspath(os.path.join(output_class_dir, f"{base_name}_{idx}.png"))
                            cv2.imwrite(output_image_path, segmented_image)
                            print(f"Processed and saved: {output_image_path}")
                else:
                    print(f"Mask not found for {mask_path}.")
                attempt += 1  # Ensure attempt is incremented regardless of mask validity

            if not mask_found:
                print(f"No suitable mask found for {base_name} after {max_attempts} attempts. Skipping this cell.")


# Example usage
images_dir = 'data/images_53'
bboxes_dir = 'data/bounding-boxes'
masks_dir = 'data/masks'
output_dir = 'data/output'
os.makedirs(output_dir, exist_ok=True)

process_images(images_dir, bboxes_dir, masks_dir, output_dir)
