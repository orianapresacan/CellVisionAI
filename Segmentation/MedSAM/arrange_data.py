import os
import shutil

# Define paths to your image and mask directories
image_dir = 'data/train/images'
mask_dir = 'data/train/masks'

# Create new folders to store the results (if not existing)
new_image_dir = os.path.join(image_dir, 'new_images')
new_mask_dir = os.path.join(mask_dir, 'new_masks')
os.makedirs(new_image_dir, exist_ok=True)
os.makedirs(new_mask_dir, exist_ok=True)

# Debug: Print the directories being used
print(f"Image directory: {image_dir}")
print(f"Mask directory: {mask_dir}")
print(f"New image directory: {new_image_dir}")
print(f"New mask directory: {new_mask_dir}")

# Iterate over each subfolder in the mask directory
for image_folder in os.listdir(mask_dir):
    image_folder_path = os.path.join(mask_dir, image_folder)
    if os.path.isdir(image_folder_path):
        for class_folder in os.listdir(image_folder_path):
            class_folder_path = os.path.join(image_folder_path, class_folder)
            if os.path.isdir(class_folder_path):
                for mask_file in os.listdir(class_folder_path):
                    # Split the mask file into name and extension
                    mask_base_name, extension = os.path.splitext(mask_file)
                    new_mask_name = f"{mask_base_name}_{class_folder}{extension}"
                    mask_file_path = os.path.join(class_folder_path, mask_file)
                    new_mask_path = os.path.join(new_mask_dir, new_mask_name)
                    shutil.copy(mask_file_path, new_mask_path)

                    # Locate the corresponding original image
                    for file in os.listdir(image_dir):
                        if file.startswith(image_folder):
                            original_image_path = os.path.join(image_dir, file)
                            original_ext = os.path.splitext(file)[1]
                            new_image_name = f"{mask_base_name}_{class_folder}{original_ext}"  # Preserve the original extension
                            new_image_path = os.path.join(new_image_dir, new_image_name)
                            
                            if os.path.isfile(original_image_path):
                                shutil.copy(original_image_path, new_image_path)
                            else:
                                print(f"Original image not found: {original_image_path}")

print("Processing complete. Images and masks have been renamed and copied.")
