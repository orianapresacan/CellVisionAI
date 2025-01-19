import os
import pandas as pd

# Define directories
image_dir = "data/train/images/new_images"  # Path where processed images are stored
mask_dir = "data/train/masks/new_masks"  # Path where processed masks are stored
output_csv = "data/train/image_mask_paths.csv"  # Output CSV file

# Get lists of images and masks
image_files = sorted(os.listdir(image_dir))  # Ensure sorted order for consistency
mask_files = sorted(os.listdir(mask_dir))

# Create dataframe
data = {"image": [], "mask": []}

# Match images to masks based on filename prefixes
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    
    # Find corresponding mask (assuming mask has the same base name but different extension)
    mask_filename = os.path.splitext(image_file)[0] + ".png"  # Assuming masks are PNG
    mask_path = os.path.join(mask_dir, mask_filename)
    
    if os.path.exists(mask_path):
        data["image"].append(image_path)
        data["mask"].append(mask_path)
    else:
        print(f"Warning: No matching mask found for {image_file}")

# Convert to DataFrame
df = pd.DataFrame(data)

# Save CSV
df.to_csv(output_csv, index_label="index")

print(f"CSV file created: {output_csv}")
