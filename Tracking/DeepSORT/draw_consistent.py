import cv2
import numpy as np
import os

# Configuration
tracked_output_file = 'data/C03_s3/tmp/ori.txt'
image_dir = 'data/C03_s3/img1/'
output_dir = 'data/C03_s3/processed_images/'
CONSISTENT = False  # Modify as needed to draw all IDs or only consistent ones

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

# Color mapping and data organization
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

# Draw bounding boxes and IDs
def draw_bboxes_and_ids(image_path, data, output_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    font_scale = 0.5
    font_thickness = 2

    for obj_id, (x, y, w, h) in data:
        if obj_id not in consistent_ids:
            continue
        if obj_id not in (171, 227, 113):
            continue
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
    draw_bboxes_and_ids(image_path, data, output_path)
