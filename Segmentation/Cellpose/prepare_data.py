import os
import cv2

import zipfile
import requests
import argparse
import logging

import numpy as np
from shutil import copy2
from collections import defaultdict


def parse_split_file(file_path):
    """Parse the split text file to extract train, validation, and test filenames."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    split_data = {"train": [], "valid": [], "test": []}
    current_split = None

    for line in lines:
        line = line.strip()

        if line.startswith("Train:"):
            current_split = "train"
        elif line.startswith("Validation:"):
            current_split = "valid"
        elif line.startswith("Test:"):
            current_split = "test"
        elif line and current_split:  # Non-empty line belonging to the current split
            split_data[current_split].append(line)

    return split_data


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_mask_images(folder_path):
    """Load grayscale images from the specified folder based on the base file name."""
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            logging.warning(f"Failed to read image: {filename}")
    return images

def copy_corresponding_files(corresponding_files_dir, output_dir, filename_base):
    """Copy files that correspond to the mask based on the filename."""
    for file in os.listdir(corresponding_files_dir):
        if file.startswith(filename_base):
            source_file_path = os.path.join(corresponding_files_dir, file)
            destination_file_path = os.path.join(output_dir, file)
            copy2(source_file_path, destination_file_path)
            logging.info(f"Copied corresponding file: {file}")

def combine_masks(mask_images):
    """Combine multiple mask images into a single image with distinct object IDs."""
    height, width = mask_images[0].shape
    combined_mask = np.zeros((height, width), dtype=np.uint16)
    object_id = 1
    
    for mask in mask_images:
        indices = np.where(mask > 0)
        combined_mask[indices] = object_id
        object_id += 1

    return combined_mask

def create_directory(directory_path):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

def split_data_from_parsed_splits(directory_path, split_data):
    """Split data into train, validation, and test sets using predefined splits."""
    data = {"train": [], "valid": [], "test": []}

    for split_name, filenames in split_data.items():
        for base_name in filenames:
            # Look for matching directories in the dataset
            dir_path = os.path.join(directory_path, base_name)
            if os.path.exists(dir_path):
                data[split_name].append(dir_path)
            else:
                logging.warning(f"Directory {dir_path} not found, skipping.")

    return data


def process_masks(input_path, image_dirname, mask_dirname, output_dir, split_file, format):
    """Process and save combined mask images by using predefined splits."""
    if not os.path.exists(input_path):
        logging.error(f"The input path {input_path} does not exist.")
        return
    
    masks_dir = os.path.join(input_path, mask_dirname)
    images_dir = os.path.join(input_path, image_dirname)

    # Parse the split file and create splits
    split_data = parse_split_file(split_file)
    data = split_data_from_parsed_splits(masks_dir, split_data)
    
    # Process and save combined masks for each split
    for split_name, split_files in data.items():
        combined_output_dir = os.path.join(output_dir, split_name)
        create_directory(combined_output_dir)

        logging.info(f"Processing {split_name} set...")

        for dirpath in split_files:
            filename_base = os.path.basename(dirpath)

            # Gather all masks across class subdirectories
            all_masks = []
            for root, dirs, files in os.walk(dirpath):  # Traverse all subdirectories
                for file in files:
                    mask_path = os.path.join(root, file)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        all_masks.append(mask)
                    else:
                        logging.warning(f"Failed to read mask: {mask_path}")

            if not all_masks:
                logging.warning(f"No masks found for {dirpath}. Skipping...")
                continue

            # Combine all masks into a single mask with distinct object IDs
            combined_mask = combine_masks(all_masks)

            # Save the combined mask
            save_filepath = os.path.join(combined_output_dir, f"{filename_base}_masks.{format}")
            cv2.imwrite(save_filepath, combined_mask)
            logging.info(f"Combined mask image saved as {save_filepath}")

            # Copy the corresponding files based on the filename of the mask
            copy_corresponding_files(images_dir, combined_output_dir, filename_base)


def download_and_unzip(url, directory_path):
    # Check if directory exists
    if not os.path.exists(directory_path):

        logging.info("Input dataset does not exist.")
        logging.info("Downloading")

        os.makedirs(directory_path)

        # Download the zip file
        zip_file_path = os.path.join(directory_path, "temp.zip")
        response = requests.get(url)
        
        if response.status_code == 200:

            logging.info("Download complete.")

            with open(zip_file_path, 'wb') as zip_file:
                zip_file.write(response.content)

            logging.info("Unzipping dataset.")

            # Unzip the file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(directory_path)

            logging.info("Unzip complete.")

            # Remove the temporary zip file
            os.remove(zip_file_path)
        else:
            logging.info(f"Failed to download zip from {url}. HTTP Status Code: {response.status_code}")
    else:
        logging.info(f"The directory {directory_path} already exists.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and combine mask images.')
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input directory containing mask images.')
    parser.add_argument('-d', '--image_dirname', default="", type=str, help='Name of image directory.')
    parser.add_argument('-m', '--mask_dirname', default="masks", type=str, help='Name of mask directory.')
    parser.add_argument('-o', '--output_dir', type=str, help='Path to the output directory to save combined masks.')
    parser.add_argument('-f', '--format', type=str, default='png', help='File format to save the masks.')
    parser.add_argument('-s', '--split_file', type=str, help='Path to the split text file (train, validation, test).')

    args = parser.parse_args()

    setup_logging()

    if os.path.exists(args.output_dir):
        logging.info(f"The output path {args.output_dir} already exists.")
        exit()

    if not os.path.exists(args.split_file):
        logging.error(f"The split file {args.split_file} does not exist.")
        exit()

    process_masks(args.input_path, args.image_dirname, args.mask_dirname, args.output_dir, args.split_file, args.format)
