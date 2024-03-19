#%%
from PIL import Image
import os
import shutil
import random

#%%
def crop_images(image_folder, bbox_folder, output_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            bbox_path = os.path.join(bbox_folder, os.path.splitext(filename)[0] + '.txt')

            if os.path.exists(bbox_path):
                image = Image.open(image_path)
                width, height = image.size

                with open(bbox_path, 'r') as file:
                    for i, line in enumerate(file):
                        class_index, x_center, y_center, bbox_width, bbox_height = [
                            float(x) if i else int(x) 
                            for i, x in enumerate(line.strip().split())
                        ]
                        
                        # Convert from YOLO format to pixel format and ensure coordinates are within image boundaries
                        x_center, bbox_width = x_center * width, bbox_width * width
                        y_center, bbox_height = y_center * height, bbox_height * height
                        x_min = max(0, x_center - bbox_width / 2)
                        y_min = max(0, y_center - bbox_height / 2)
                        x_max = min(width, x_center + bbox_width / 2)
                        y_max = min(height, y_center + bbox_height / 2)

                        # Crop and save the image if the area is valid
                        if x_max - x_min > 0 and y_max - y_min > 0:
                            cropped_image = image.crop((x_min, y_min, x_max, y_max))
                            cropped_image.save(os.path.join(output_folder, f'{class_index}_{filename[:-4]}_{i}.jpg'))

image_folder = 'images-color'
bbox_folder = 'bounding-boxes'
output_folder = 'cropped_images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

crop_images(image_folder, bbox_folder, output_folder)


def sort_images_by_class(source_folder, target_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            class_index = filename.split('_')[0]
            class_folder = os.path.join(target_folder, class_index)

            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            src_file = os.path.join(source_folder, filename)
            dst_file = os.path.join(class_folder, filename)
            shutil.move(src_file, dst_file)

# Usage
source_folder = 'cropped_images'
target_folder = 'cropped_images_class'

sort_images_by_class(source_folder, target_folder)

def resize_images(root_folder, size=(128, 128)):
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                with Image.open(file_path) as img:
                    img = img.resize(size, Image.ANTIALIAS)
                    img.save(file_path)

# root_folder = 'data_resized'
# resize_images(root_folder)
