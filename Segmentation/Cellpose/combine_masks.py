import cv2
import os
import numpy as np

# Set the path to the root directory containing 'X', 'Y', etc.
root_dir = 'C:/Users/oriana/Desktop/Projects/Autophagy_cells/Segmentation/Cellpose/cellular-experiments/masks'

# List all the main folders like 'X', 'Y' etc.
main_folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

# Subfolders names
subfolders = ['fed', 'unfed', 'unidentified']

# Dictionary to hold the combined images for each main folder
combined_images = {}

for main_folder in main_folders:
    main_folder_path = os.path.join(root_dir, main_folder)
    combined_image = None

    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder_path, subfolder)

        # Check if subfolder exists
        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping.")
            continue

        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)

            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # Check if the image is read properly
            if image is None:
                print(f"Failed to read image {image_path}")
                continue

            # Initialize combined_image or combine with existing
            if combined_image is None:
                combined_image = image
            else:
                # Modify this line if a different combining strategy is needed
                combined_image = np.maximum(combined_image, image)

    # Store the combined image in the dictionary
    if combined_image is not None:
        combined_images[main_folder] = combined_image

# Create a new folder to save all combined images
all_combined_folder = os.path.join(root_dir, 'all_combined')
if not os.path.exists(all_combined_folder):
    os.makedirs(all_combined_folder)

# Save each combined image in the new folder with the name of the main folder
for main_folder, image in combined_images.items():
    combined_output_path = os.path.join(all_combined_folder, f"{main_folder}.png")
    cv2.imwrite(combined_output_path, image)