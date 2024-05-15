


## SORT
- For running SORT, we followed this [codebase](https://github.com/abewley/sort).
- First, the bounding box annotations must be converted from the YOLO format to the MOTChallenge format. Begin by organizing the videos into separate folders; each folder should contain text files with bounding box information corresponding to each video. Then, use the `convert_to_MOT.py`.
- For each video, create a separate directory in the `data` folder. Each video directory must contain:
      - `det/det.txt` file with the bounding boxes in MOTChallenge format. Example: "1,-1,0.048828125,0.06689453125,0.04345703125,0.064453125,1,-1,-1,-1" where 1 is the frame number, -1 is the tracking ID, the 4 next numbers are the bounding box coordinates, the next 1 is the confidence score
      - an `images` directory with the five images: `000001.jpg`, `000002.jpg`, etc.
- The code will generate a text file with the results in an `output` directory.
- `draw_real_boxes.py` can be used to draw the real bounding boxes on top of the real images.
- `draw_consistent.py` can be used to draw the bounding boxes with the tracking IDs on top of the real images.

## DeepSORT
- For running DeepSORT, we followed this [codebase](https://github.com/nwojke/deep_sort).
- First, the bounding box annotations must be converted from the YOLO format to the MOTChallenge format. Begin by organizing the videos into separate folders; each folder should contain text files with bounding box information corresponding to each video. Then, use the `convert_to_MOT.py`.
- This algorithm requires feature vectors. Follow the instructions from their repository's README to generate them.
- For each video, create a separate directory in the `data` folder. Each video directory must contain:
        - `gt` directory which is empty in our case because we do not have ground truth data.
        - `img1` directory containing the five images.
        - `data.npy` file with the data. 

## Our Tracking Model

- YOLO format will be used for annotations.
- Each bounding box file must include feature vectors. Use the `feature_vectors.py` script to generate these vectors using resnet50. You must have the complete images and the corresponding text files with bounding boxes.
- `tracking_euclidean_distance.py` is a simple tracking method based only on Euclidean distance. This served as the baseline for performance comparisons with subsequent algorithms.
- Our custom tracking algorithm is implemented in `tracking_features.py`.
