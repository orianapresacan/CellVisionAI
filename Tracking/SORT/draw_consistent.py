import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
VIDEO = "I06_s2"
tracked_output_file = f'output/{VIDEO}.txt'
image_dir = f'data/train/{VIDEO}/images/'
output_dir = f'data/train/{VIDEO}/processed_images/'
base_cropped_dir = f'data/train/{VIDEO}/cropped_images/'  # Base directory for cropped images
plot_dir = f'data/train/{VIDEO}/plots/'
CONSISTENT = True  
CROP_AND_SAVE = True  # Feature flag for cropping and saving images
DRAW_BBOXES = False  # Feature flag for drawing bounding boxes and IDs
PLOT_SEQUENCE = False

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def save_image_sequences(max_images=5):
    id_frame_paths = {}

    # Organize image paths by ID
    for frame_num, data in tracked_data_by_frame.items():
        for obj_id, _ in data:
            if CONSISTENT and obj_id not in consistent_ids:
                continue
            frame_dir = os.path.join(base_cropped_dir, f'frame_{frame_num}')
            image_path = os.path.join(frame_dir, f'cell_{obj_id}.jpg')
            if os.path.exists(image_path):
                if obj_id not in id_frame_paths:
                    id_frame_paths[obj_id] = []
                id_frame_paths[obj_id].append(image_path)

    # Save sequences for each ID
    for obj_id, paths in id_frame_paths.items():
        image_paths = paths[:max_images]  # Limit the number of images
        fig, axs = plt.subplots(1, len(image_paths), figsize=(15, 3))
        fig.suptitle(f'ID: {obj_id}', fontsize=16)
        for i, img_path in enumerate(image_paths):
            img = plt.imread(img_path)
            if len(image_paths) > 1:
                axs[i].imshow(img)
                axs[i].axis('off')
            else:
                axs.imshow(img)
                axs.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for title
        plot_filename = os.path.join(plot_dir, f'sequence_id_{obj_id}.png')
        plt.savefig(plot_filename)
        plt.close() 

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Unique color generator, excluding green shades
def get_random_color(exclude_green=True):
    while True:
        color = np.random.randint(0, 256, 3).tolist()
        if not exclude_green or color[1] < max(color[0], color[2]):
            break
    return tuple(color)

# Color mapping, data organization
id_colors = {}
tracked_data_by_frame = {}
with open(tracked_output_file, 'r') as file:
    for line in file:
        frame_num, obj_id, x, y, w, h, _, _, _, _ = map(float, line.strip().split(','))
        frame_num, obj_id = int(frame_num), int(obj_id)
        bbox = (x, y, w, h)
        tracked_data_by_frame.setdefault(frame_num, []).append((obj_id, bbox))
        if obj_id not in id_colors:
            id_colors[obj_id] = get_random_color()

# Identify consistent IDs
ids_in_frames = {frame_num: set(obj_id for obj_id, _ in objs) for frame_num, objs in tracked_data_by_frame.items()}
consistent_ids = set.intersection(*ids_in_frames.values()) if CONSISTENT else set(id_colors.keys())

# Function to process images, crop, and optionally draw bounding boxes & IDs
def process_images(image_path, data, output_path, frame_num):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return  # Skip this image

    height, width = image.shape[:2]

    frame_cropped_dir = os.path.join(base_cropped_dir, f'frame_{frame_num}')
    if CROP_AND_SAVE and not os.path.exists(frame_cropped_dir):
        os.makedirs(frame_cropped_dir)

    for obj_id, (x, y, w, h) in data:
        is_consistent = obj_id in consistent_ids
        if CROP_AND_SAVE and (not CONSISTENT or is_consistent):
            # Ensure crop coordinates are within image dimensions
            x, y, w, h = max(x, 0), max(y, 0), min(w, 1), min(h, 1)
            crop_x1, crop_y1 = int(x * width), int(y * height)
            crop_x2, crop_y2 = int((x + w) * width), int((y + h) * height)
            # Validate crop dimensions
            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                print(f"Invalid crop dimensions for ID {obj_id} in frame {frame_num}.")
                continue  # Skip this crop
            cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if cropped_image.size == 0:
                print(f"Empty crop for frame {frame_num}, object {obj_id}.")
                continue  # Skip saving this crop
            cropped_image_filename = os.path.join(frame_cropped_dir, f"cell_{obj_id}.jpg")
            cv2.imwrite(cropped_image_filename, cropped_image)

        if DRAW_BBOXES and (not CONSISTENT or is_consistent):
            font_scale = 0.5
            font_thickness = 2
            color = id_colors[obj_id]
            top_left = (int(x * width), int(y * height))
            bottom_right = (int((x + w) * width), int((y + h) * height))
            cv2.rectangle(image, top_left, bottom_right, color, 2)
            label_background_top_left = (top_left[0], top_left[1] - 20)
            label_background_bottom_right = (top_left[0] + 50, top_left[1])
            cv2.rectangle(image, label_background_top_left, label_background_bottom_right, color, -1)
            cv2.putText(image, str(obj_id), (top_left[0], top_left[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    cv2.imwrite(output_path, image)

# Processing images
for frame_num, data in tracked_data_by_frame.items():
    image_path = os.path.join(image_dir, f"{frame_num:06d}.jpg")
    output_path = os.path.join(output_dir, f"{frame_num:06d}.jpg")
    process_images(image_path, data, output_path, frame_num)

# Print the count of consistent cells
if CONSISTENT:
    print(f"Total consistent cells: {len(consistent_ids)}")

if PLOT_SEQUENCE and CROP_AND_SAVE:
    save_image_sequences()