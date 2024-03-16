import os
import cv2
import argparse
import logging
import numpy as np

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_combine_masks(folder_path):
    """Load and combine grayscale images from the specified folder."""
    combined_mask = None
    object_id = 1  # Start with ID 1 for the first object
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if combined_mask is None:
                combined_mask = np.zeros_like(img, dtype=np.uint16)
            # Assign a unique ID to each object in the mask
            mask_indices = img > 0
            combined_mask[mask_indices] = object_id
            object_id += 1  # Increment the object ID for the next mask
        else:
            logging.warning(f"Failed to read image: {filename}")
    return combined_mask

def create_directory(directory_path):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

def process_main_folder(main_folder_path, output_dir, format):
    """Process and combine mask images from all subfolders in a main folder."""
    combined_mask = None
    object_id = 1  # Start with ID 1 for the first object
    for subfolder in ['fed', 'unfed', 'unidentified']:
        subfolder_path = os.path.join(main_folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if combined_mask is None:
                        combined_mask = np.zeros_like(img, dtype=np.uint16)
                    # Assign a unique ID to each object in the mask
                    mask_indices = img > 0
                    combined_mask[mask_indices] = object_id
                    object_id += 1  # Increment the object ID for the next mask
                else:
                    logging.warning(f"Failed to read image: {filename}")

    if combined_mask is not None:
        main_folder_name = os.path.basename(main_folder_path)
        save_filepath = os.path.join(output_dir, f"{main_folder_name}_combined_masks.{format}")
        cv2.imwrite(save_filepath, combined_mask)
        logging.info(f"Combined mask image saved as {save_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine mask images from subfolders.')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input directory containing folders of mask images.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to the output directory to save combined masks.')
    parser.add_argument('-f', '--format', type=str, default='png', help='File format to save the masks.')

    args = parser.parse_args()

    setup_logging()
    create_directory(args.output_dir)

    for main_folder_name in os.listdir(args.input_path):
        main_folder_path = os.path.join(args.input_path, main_folder_name)
        if os.path.isdir(main_folder_path):
            process_main_folder(main_folder_path, args.output_dir, args.format)
