# SAM Model

## Overview

This repository is based on the [Segment Anything Model](https://segment-anything.com/) developed by Meta, complemented by fine-tuning techniques exemplified in this [notebook](https://colab.research.google.com/drive/1Jb422MehJ6TYUCfcy6yxuSOJkSGAfJXj#scrollTo=5GhzOeOFbCQa) from Medium. We used this tutorial for fine-tuning SAM without bounding boxes.


## Virtual Environment
Go to the project folder. Create a virtual environment:
```bash
python -m venv .venv
```

Activate the virtual environment:
```bash
source ./.venv/Scripts/activate
```

Install the requirements:
```bash
pip install -r requirements.txt
```


## Data Preparation

To prepare the data for training, you need to convert the bounding boxes from YOLO to COCO format. Follow these steps:
- Add the original `masks` and `images` folders to your project directory.
- Run the `prepare_data.py` script.

The script will:
- Convert the masks to binary format and remove the subfolders (`fed`, `unfed`, `unidentified`) by merging all masks for each image into a single directory.
- Create a `data` directory containing `train`, `val`, and `test` folders, each with its own `images` and `masks` subfolders.
- Generate an `annotations` directory containing three JSON files (`train_annotations.json`, `val_annotations.json`, and `test_annotations.json`), each storing the COCO-format annotations for the corresponding dataset split.

The final dataset strucutre should be like:

```bash
data/ 
├── train/ 
│ ├── images/ 
│ └── annotations.json 
├── val/ 
│ ├── images/ 
│ └── annotations.json 
└── test/ 
├── images/ 
└── annotations.json
```

If you want to skip these steps, you can download the preprocessed dataset from [here](https://drive.google.com/file/d/1uAHCs4SWPBXvQh65u0RjgwFS_q3TvYIY/view?usp=sharing).


## Model Fine-Tunining

- Download the weights of the pre-trained model provided by Meta from [here](https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints). Choose the default model: `sam_vit_h`. Add the checkpoints to your project folder (`SAM`).
  
- To fine-tune the model, utilize the `fine_tune.py` script.

## Model Evaluation

- Assess the performance of either the model with the `test.py` script. This provides metrics such as IoU, Accuracy, Precision, Recall, and F1 score.
- To evaluate the pre-trained model, set `FINETUNED = 0`, to evaluate the fine-tuned SAM, set `FINETUNED = 1`.

## Visualization

- For visual inspection, use `visualize_predicted_masks.py` to overlay SAM's predicted masks on images, or `visualize_real_masks.py` to view the actual masks. 


