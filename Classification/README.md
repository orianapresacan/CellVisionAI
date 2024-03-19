# Cell Classification

This project focuses on the classification of cell images using deep learning models, specifically VGG, ResNet, and Vision Transformer (ViT). The aim is to classify cell images into three categories: fed, unfed, and unidentified, based on their visual characteristics.

## Dataset Preparation

To crop each cell into its own image based on bounding box coordinates, utilize the `crop_images.py` script.

### Segmented Data

To prepare the segmented data, utilize the `crop_and_segment_images.py` script. This requires having all bounding boxes, masks, and original images into separate directories. After executing the cropping and segmentation process, distribute the images into training, validation, and test sets according to the guidelines in `data_division.txt`. Following this, employ the `merge_folders.py` script to mix images from each category across all sets.

## Model Training

For model training, execute the `train.py` script. Within this script, adjust the `MODEL` variable to select your desired model (`vgg`, `vit`, `resnet`) and set the `PRETRAINED` flag to indicate whether to initiate training from scratch or use a pretrained model as a starting point.

## Model Evaluation

To evaluate the model's performance on the test set, run the `test()` function within the `train.py` script. 

### Get Labels

For obtaining labels of each cell image in the test dataset, use the `get_labels()` function found in the `train.py` script. This step is useful for generating t-SNE plots.
