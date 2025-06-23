import cv2
import os

class_colors = {
    0: (255, 0, 0),    
    1: (0, 0, 255),   
    2: (0, 110, 140)    
}

def draw_boxes(image_path, bbox_path, output_image_path):
    def read_bboxes(file_path, img_width, img_height):
        bboxes = []
        with open(file_path, 'r') as file:
            for line in file:
                class_id, x_center, y_center, width, height = map(float, line.split())
                x = int((x_center - width / 2) * img_width)
                y = int((y_center - height / 2) * img_height)
                w = int(width * img_width)
                h = int(height * img_height)
                bboxes.append((x, y, w, h, class_id))
        return bboxes

    def put_text_with_background(img, text, org, font, font_scale, font_color, bg_color):
        text_size = cv2.getTextSize(text, font, font_scale, thickness=2)[0]
        x, y = org
        cv2.rectangle(img, (x, y - text_size[1]), (x + text_size[0], y), bg_color, -1)
        cv2.putText(img, text, org, font, font_scale, font_color, 2)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    img_height, img_width = image.shape[:2]

    bboxes = read_bboxes(bbox_path, img_width, img_height)
    for bbox in bboxes:
        x, y, w, h, class_id = bbox
        color = class_colors.get(class_id, (255, 255, 255))  
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"{int(class_id)}"
        put_text_with_background(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), color)

    cv2.imwrite(output_image_path, image)
    print(f"Visualized image saved to: {output_image_path}")


