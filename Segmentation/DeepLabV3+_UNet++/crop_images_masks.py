import os
import cv2
import numpy as np

def convert_to_black_white(mask_img):
    """
    Converts a mask image to strictly black-and-white (binary format).
    """
    if len(mask_img.shape) > 2:  # If the mask is colored
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)
    return binary_mask

def get_bounding_box_from_mask(mask_img):
    """
    Calculates the bounding box from a binary mask.
    """
    coords = np.where(mask_img > 0)  
    if len(coords[0]) == 0 or len(coords[1]) == 0:  
        return None
    ymin, ymax = coords[0].min(), coords[0].max()
    xmin, xmax = coords[1].min(), coords[1].max()
    return xmin, ymin, xmax, ymax

def crop_and_save(image, mask, bbox, masks_save_path, images_save_path, base_name, idx):
    """
    Crops and saves the original image and its corresponding mask based on the bounding box.
    """
    xmin, ymin, xmax, ymax = bbox

    if xmin >= xmax or ymin >= ymax:
        print(f"Invalid bounding box: {bbox} for {base_name}_{idx}")
        return

    img_height, img_width = image.shape[:2]
    if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
        print(f"Bounding box {bbox} out of image bounds for {base_name}_{idx}")
        return

    cropped_mask = mask[ymin:ymax, xmin:xmax]
    if cropped_mask.size == 0:  # Ensure the cropped mask is not empty
        print(f"Empty cropped mask for bbox: {bbox} in {base_name}_{idx}")
        return
    mask_save_path = os.path.join(masks_save_path, f"{base_name}_mask_{idx}.png")
    cv2.imwrite(mask_save_path, cropped_mask)

    cropped_image = image[ymin:ymax, xmin:xmax]
    if cropped_image.size == 0:  # Ensure the cropped image is not empty
        print(f"Empty cropped image for bbox: {bbox} in {base_name}_{idx}")
        return
    image_save_path = os.path.join(images_save_path, f"{base_name}_image_{idx}.jpg")
    cv2.imwrite(image_save_path, cropped_image)


def process_parent_dir(parent_dir):
    """
    Processes a single directory (train, test, or val).
    """
    images_path = os.path.join(parent_dir, "images")
    masks_path = os.path.join(parent_dir, "masks")
    cropped_masks_path = os.path.join(parent_dir, "cropped_masks")
    cropped_images_path = os.path.join(parent_dir, "cropped_images")

    os.makedirs(cropped_masks_path, exist_ok=True)
    os.makedirs(cropped_images_path, exist_ok=True)

    for image_file in os.listdir(images_path):
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(images_path, image_file)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        mask_folder = os.path.join(masks_path, base_name)
        if not os.path.isdir(mask_folder):
            print(f"Mask folder not found for image: {image_path}")
            continue

        mask_files = sorted(os.listdir(mask_folder))
        for idx, mask_file in enumerate(mask_files):
            mask_path = os.path.join(mask_folder, mask_file)

            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"Failed to read mask: {mask_path}")
                continue

            binary_mask = convert_to_black_white(mask)

            bbox = get_bounding_box_from_mask(binary_mask)
            if bbox is None:
                print(f"No non-zero pixels in mask: {mask_path}")
                continue

            crop_and_save(image, binary_mask, bbox, cropped_masks_path, cropped_images_path, base_name, idx)

data_dir = "data"  
for parent_dir in ["train", "test", "val"]:
    parent_path = os.path.join(data_dir, parent_dir)
    if os.path.exists(parent_path):
        process_parent_dir(parent_path)
    else:
        print(f"Directory not found: {parent_path}")
