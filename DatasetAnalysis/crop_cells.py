import os
import cv2

def read_bbox_file(file_path):
    """
    Reads a YOLOv5 formatted bounding box file and returns the bounding boxes as a list of tuples.
    Each tuple contains (class_id, x_center, y_center, width, height).
    """
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.split())
            bboxes.append((class_id, x_center, y_center, width, height))
    return bboxes

def convert_yolo_to_bbox(img_shape, bbox):
    """
    Converts YOLO format (center x, center y, width, height) to bounding box format (x1, y1, x2, y2).
    """
    dw = 1. / img_shape[1]
    dh = 1. / img_shape[0]
    x, y, w, h = bbox[1], bbox[2], bbox[3], bbox[4]
    x1 = int((x - w / 2) / dw)
    y1 = int((y - h / 2) / dh)
    x2 = int((x + w / 2) / dw)
    y2 = int((y + h / 2) / dh)
    return (x1, y1, x2, y2)

def crop_and_save_images(image_dir, bbox_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(bbox_dir):
        if file_name.endswith('.txt'):
            image_path = os.path.join(image_dir, file_name.replace('.txt', '.jpg'))
            bbox_path = os.path.join(bbox_dir, file_name)
            
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is None or img.size == 0:
                    print(f"Image {image_path} could not be loaded.")
                    continue
                
                bboxes = read_bbox_file(bbox_path)
                
                for i, bbox in enumerate(bboxes):
                    class_id = int(bbox[0])
                    bbox_coords = convert_yolo_to_bbox(img.shape, bbox)

                    # Ensure bounding box is within image dimensions
                    bbox_coords = (
                        max(0, bbox_coords[0]),
                        max(0, bbox_coords[1]),
                        min(img.shape[1], bbox_coords[2]),
                        min(img.shape[0], bbox_coords[3])
                    )
                    
                    # Check if the cropped area is valid
                    if bbox_coords[2] <= bbox_coords[0] or bbox_coords[3] <= bbox_coords[1]:
                        print(f"Invalid bounding box for {file_name} with coordinates {bbox_coords}.")
                        continue
                    
                    cropped_img = img[bbox_coords[1]:bbox_coords[3], bbox_coords[0]:bbox_coords[2]]
                    
                    output_file_name = f"{os.path.splitext(file_name)[0]}_cell_{i+1}_class_{class_id}.jpg"
                    cv2.imwrite(os.path.join(output_dir, output_file_name), cropped_img)


image_dir = 'I_N_starved' 
bbox_dir = 'I_N_starved'  
output_dir = 'cropped_images_unfed'  

crop_and_save_images(image_dir, bbox_dir, output_dir)
