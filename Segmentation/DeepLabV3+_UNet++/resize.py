import os
from PIL import Image

def resize_and_replace_images_in_folder(folder, size=(64, 64)):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img = img.resize(size, Image.ANTIALIAS)

            # Overwrite the original image
            img.save(img_path)

# Example usage
folder = 'data/train/cropped_masks'
resize_and_replace_images_in_folder(folder, size=(64, 64))
