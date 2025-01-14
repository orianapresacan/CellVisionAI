
# SORT

This guide explains how to run the SORT (Simple Online and Realtime Tracking) algorithm using the [SORT codebase](https://github.com/abewley/sort) on the CELLULAR data set.

## Data Preparation
1. **Convert Bounding Box Annotations:** Convert bounding box annotations from YOLO format to MOTChallenge format.
   - Organize videos into separate folders.
   - Each folder should contain text files with bounding box information for each video.
   - Use the script `convert_to_MOT.py` for conversion.

## Directory Structure for Each Video
- In the `data` folder, create a directory for each video.
- Each directory must contain:
  - A `det/det.txt` file with bounding boxes in the MOTChallenge format.
    - Example entry: `1,-1,0.048828125,0.06689453125,0.04345703125,0.064453125,1,-1,-1,-1`
    - Format details: frame number, tracking ID (placeholder as -1), bounding box coordinates, confidence score, placeholders for unused values.
  - An `images` directory containing sequence images (e.g., `000001.jpg`, `000002.jpg`, etc.).

## Running the Tracking
- Execute the provided code to process the videos and generate output.
- The results are saved in the `output` directory as text files.

## Additional Utilities
- **Drawing Bounding Boxes:** Use `draw_real_boxes.py` to overlay actual bounding boxes on the images.
- **Drawing Tracking IDs:** Use `draw_consistent.py` to display bounding boxes with tracking IDs on the images.


# DeepSORT

This guide provides instructions on how to run the DeepSORT tracking algorithm using the [DeepSORT codebase](https://github.com/nwojke/deep_sort) on the CELLULAR data set.

## Data Preparation
1. **Convert Bounding Box Annotations:** Convert annotations from YOLO format to MOTChallenge format.
   - Organize videos into separate folders.
   - Each folder should include text files with bounding box information for each video.
   - Use the script `convert_to_MOT.py` for the conversion process.

2. **Generate Feature Vectors:**
   - Follow the instructions in the original repository's README to generate the necessary feature vectors for tracking.

## Directory Structure for Each Video
In the `data` folder, create a separate directory for each video:
- **`gt` Directory:** Should be empty, as ground truth data is not used in our implementation.
- **`img1` Directory:** Contains the sequence images (e.g., `000001.jpg`, `000002.jpg`, etc.).
- **`data.npy` File:** Stores the data needed for tracking.
