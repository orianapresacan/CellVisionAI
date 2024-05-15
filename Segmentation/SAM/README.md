# SAM Model

## Overview

This repository is based on the [Segment Anything Model](https://segment-anything.com/) developed by Meta, complemented by fine-tuning techniques exemplified in this [notebook](https://colab.research.google.com/drive/1Jb422MehJ6TYUCfcy6yxuSOJkSGAfJXj#scrollTo=5GhzOeOFbCQa) from Medium. We used this tutorial for fine-tuning SAM without bounding boxes.

## Installation

To integrate SAM into your projects, run the following command:

```bash
pip install segment_anything
```

## Data

- **Data Preparation:** Convert your bounding boxes and masks into COCO format using the script found at `data/convert_to_COCO_format.py`.
  
- Add the train, test, and val folders in the data directory. Each one of these must contain an annotations.json along with the images.
- Add a bounding_boxes directory with all 53 text files containing the bounding box coordinates in YOLOv5 format.
- Add a masks directory with all masks in binary format, one mask per image.


## Usage Guide

- **Model Download:** Download the weights of the pre-trained model provided by Meta from [here](https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints). You can choose the default model.
  
- **Model Fine-Tuning:** To fine-tune the Segment Anything Model (SAM), utilize the `fine_tune.py` script.

- **Model Evaluation:** Assess the performance of either the pre-trained or fine-tuned SAM on your dataset with the `test.py` script. This provides metrics such as IoU, Accuracy, Precision, Recall, and F1 score.

- **Visualization:** For visual inspection, use `visualize_predicted_masks.py` to overlay SAM's predicted masks on images, or `visualize_real_masks.py` to view the actual masks. 


