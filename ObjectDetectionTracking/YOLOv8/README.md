# YOLO

This guide for fine-tuning object detection models follows instructions from [Ultralytics tutorials](https://docs.ultralytics.com/tasks/detect/).

## Installation

```bash
pip install ultralytics
```

## Data
  
Structure your dataset as follows, ensuring that the labels folder contains text files with bounding box coordinates in YOLO format. For tasks without class differentiation, label all objects with a single class identifier (0).

    data
    ├── train
    │   ├── images
    │   └── labels
    └── val
        ├── images
        └── labels

- Update file paths in `cell.yaml`.
- For tasks including classification, specify all 3 classes in `cell.yaml` and prefix text files in labels with appropriate class identifiers.

## Usage Guide

- **Model Fine-Tuning:**

```python
yolo train model=yolov8x.pt data=cell.yaml epochs=500 imgsz=2048 batch=4 device=0,1
```

- **Model Evaluation:**




