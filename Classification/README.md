# Cell Classification

This directory focuses on the classification of cell images using VGG, ResNet, and Vision Transformer (ViT) into three different categories: basal, activated, and unidentified.

<img src="https://github.com/orianapresacan/CellVisionAI/blob/7e7bf82374f490d0e212d38c201019b31eca1a9d/Images/cell_class_examples.jpg" width="800" height="150"/>

## Dataset Preparation

To crop each cell into its own image based on bounding box coordinates, utilize the `crop_images.py` script.

### Segmented Data

To prepare the segmented data, utilize the `crop_and_segment_images.py` script. This requires having all bounding boxes, masks, and original images in separate directories. After executing the cropping and segmentation process, distribute the images into training, validation, and test sets according to the guidelines in `data_division.txt`. Following this, employ the `merge_folders.py` script to mix images from each category across all sets.

## Model Training

For model training, execute the `train.py` script. Within this script, adjust the `MODEL` variable to select your desired model (`vgg`, `vit`, `resnet`) and set the `PRETRAINED` flag to indicate whether to initiate training from scratch or use a pre-trained model as a starting point.

## Model Evaluation

To evaluate the model's performance on the test set, run the `test()` function within the `train.py` script. 

### Get Labels

For obtaining labels of each cell image in the test dataset, use the `get_labels()` function found in the `train.py` script. This step is useful for generating t-SNE plots.

### Model Checkpoints

You can download the checkpoints for the three models trained on the CELLULAR data set from [here](https://drive.google.com/drive/folders/1SQpfsqEfRrEO1e5esKhRJNG29iOavM2C?usp=sharing).
