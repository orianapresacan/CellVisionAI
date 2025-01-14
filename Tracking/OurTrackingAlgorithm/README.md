# Our Tracking Algorithm 

This guide details the setup and use of our custom tracking model.

## Data Preparation
- **Annotation Format:** We use the YOLO format for bounding box annotations.

## Create the Directory Structure
- Create a directory, e.g., `C03_s2`, containing the following:
  1. **5 Images**: Include the required 5 images in the directory.
  2. **5 Corresponding Text Files**: Add text files containing bounding box annotations for each image.
  3. **5 Mask Folders**: Each image should have its own folder containing its masks.

### Mask Processing
- All masks must be combined into a single folder (not divided into categories such as `fed`, `unfed`, or `unidentified` as in the original dataset).
- To mix and convert the masks to binary format, run the script `mix_binary.py`

## Tracking Instructions

### Run the Tracking Script
- Execute `tracking.py` to track the cells. This script will:
  1. **Create Five Folders**: `frame_1`, `frame_2`, `frame_3`, `frame_4`, `frame_5`. Each folder will contain cropped images of cells.
  2. **Name Convention**: The cropped image names will follow this pattern:  
     `cell_1_frame_0`, `cell_1_frame_2`, etc., ensuring consistency across frames.
  3. **Draw Tracking Information**: Annotate the original images with tracked bounding boxes and their tracking IDs for each frame.

### Configurations
- Set `CROP_SAVE = True` in the script.
- Set `DRAW = True` to enable drawing of tracking annotations.
- Update the `main_folder` variable in the script to your dataset path:  

## Feature Vectors (optional)
- We observed that feature vectors were not particularly effective, likely due to the high similarity between the cells. However, if you want yo use them:
  
- **Generating Feature Vectors:** Use the `feature_vectors.py` script to generate feature vectors.
  - This script utilizes the resnet50 model to extract features.
  - Ensure you have complete images and their corresponding bounding box text files before running this script.

