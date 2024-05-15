from PIL import Image
import numpy as np
import os

def convert_to_binary(input_directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):  # Assuming the mask files are in PNG format
            file_path = os.path.join(input_directory, filename)
            with Image.open(file_path) as img:
                # Convert image to RGB to ensure consistent color space processing
                img = img.convert('RGB')
                data = np.array(img)

                # Create a binary mask where all non-black pixels are set to white
                # This assumes that the mask is a single color on a black background
                binary_mask = np.any(data != [0, 0, 0], axis=-1) * 255

                # Convert the numpy array back to an image
                binary_image = Image.fromarray(binary_mask.astype(np.uint8), mode='L')
                
                # Save the converted image
                output_path = os.path.join(output_directory, filename)
                binary_image.save(output_path)
                print(f"Converted {filename} and saved to {output_path}")

# Usage
input_directory = 'data/train/masks'
output_directory = 'data/train/masks_binary'
convert_to_binary(input_directory, output_directory)
