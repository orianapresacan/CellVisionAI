


## SORT
- For implementing DeepSORT, we followed this [tutorial](https://github.com/abewley/sort).
- First, the bounding box annotations must be converted from the YOLO format to the MOTChallenge format. Begin by organizing the videos into separate folders; each folder should contain text files with bounding box information corresponding to each video. Then, use the `convert_to_MOT.py`.

## DeepSORT
- For implementing DeepSORT, we followed this [tutorial](https://github.com/nwojke/deep_sort).
- First, the bounding box annotations must be converted from the YOLO format to the MOTChallenge format. Begin by organizing the videos into separate folders; each folder should contain text files with bounding box information corresponding to each video. Then, use the `convert_to_MOT.py`.

## Our Tracking Model

- YOLO format will be used for annotations.
- Each bounding box file must include feature vectors. Use the `feature_vectors.py` script to generate these vectors using resnet50. You must have the complete images and the corresponding text files with bounding boxes.
- `tracking_euclidean_distance.py` is a simple tracking method based only on Euclidean distance. This served as the baseline for performance comparisons with subsequent algorithms.
- Our custom tracking algorithm is implemented in `tracking_features.py`.
