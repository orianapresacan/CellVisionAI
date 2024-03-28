


## Dataset Preparation

- To utilize SORT and DeepSORT algorithms, we need to convert the bounding boxes from the YOLO format to the MOTChallenge format. Begin by organizing the videos into separate folders; each folder should contain text files with bounding box information corresponding to each video. Then, use the `convert_to_MOT.py`.

- For our proposed model, YOLO format will be used. However, the feature vectors must be added to the bounding box files. To extract these via resnet50, use `feature_vectors.py` script.
