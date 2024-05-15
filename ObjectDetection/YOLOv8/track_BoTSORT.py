from collections import defaultdict
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best_detect_no_classes.pt')

# Open the video file
video_path = "I06_s2.mp4"
cap = cv2.VideoCapture(video_path)

# Base directory to save frames and cropped cells
base_save_dir = 'saved_frames_I06_s2'
os.makedirs(base_save_dir, exist_ok=True)

# Store the track history
track_history = defaultdict(lambda: [])

# Frame counter
frame_counter = 0

def draw_boxes(image, boxes, track_ids):
    img_height, img_width = image.shape[:2]

    for box, track_id in zip(boxes, track_ids):
        # Convert from normalized to pixel coordinates
        x_center, y_center, w, h = box
        x = int((x_center - w / 2) * img_width)
        y = int((y_center - h / 2) * img_height)
        w = int(w * img_width)
        h = int(h * img_height)

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add the track ID (or other label) without confidence score
        label = f"ID: {track_id}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.5)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywhn.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = draw_boxes(frame.copy(), boxes, track_ids)

        # Create a directory for the current frame
        frame_dir = os.path.join(base_save_dir, f"frame_{frame_counter}")
        os.makedirs(frame_dir, exist_ok=True)

        # Prepare to save bounding box coordinates in a text file
        bbox_file_path = os.path.join(frame_dir, f"frame_{frame_counter}_bboxes.txt")
        with open(bbox_file_path, 'w') as bbox_file:

            # Crop and save each cell image in the frame directory
            for box, track_id in zip(boxes, track_ids):
                # Convert from normalized to pixel coordinates
                x_center, y_center, w, h = box

                # Write bounding box coordinates to the text file
                bbox_file.write(f"{track_id}, {x_center}, {y_center}, {w}, {h}\n")

                x = int((x_center - w / 2) * frame.shape[1])
                y = int((y_center - h / 2) * frame.shape[0])
                w = int(w * frame.shape[1])
                h = int(h * frame.shape[0])

                # Ensure cropping does not go out of image bounds
                x, y, w, h = max(x, 0), max(y, 0), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

                # Crop the image
                cropped_cell = frame[y:y+h, x:x+w]

                # Save the cropped image in the frame directory
                cell_filename = os.path.join(frame_dir, f"cell_{track_id}.jpg")
                cv2.imwrite(cell_filename, cropped_cell)

        # Save the annotated frame
        frame_filename = os.path.join(frame_dir, "annotated_frame.jpg")
        cv2.imwrite(frame_filename, annotated_frame)

        frame_counter += 1

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
# cv2.destroyAllWindows()
